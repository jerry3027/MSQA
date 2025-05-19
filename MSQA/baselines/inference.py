import os
import json
from typing import Union, List
from uuid import uuid4

from huggingface_hub import snapshot_download
from peft import PeftConfig
from utils import query_llm, MODEL_CONFIG
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
import argparse
from tqdm import tqdm



def inference(model_checkpoint: str, prompts: list[str], sampling_params=None) -> list[RequestOutput]:
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=3600) if not sampling_params else sampling_params
    
    if model_checkpoint == "honeybee":
        return inference_honeybee_original(prompts, sampling_params)
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    llm = LLM(model=model_checkpoint)
    
    # Format prompts to be in proper format for model
    batched_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    formatted_prompts = tokenizer.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
    
    batched_outputs = llm.generate(formatted_prompts, sampling_params)
        
    return batched_outputs

def inference_lora(model_checkpoint: str, prompts: list[str], sampling_params=None) -> list[RequestOutput]:
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=3600) if not sampling_params else sampling_params
    
    base_model_path = PeftConfig.from_pretrained(model_checkpoint).base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # Adding chat template for specific models
    if base_model_path == "mistralai/Mistral-7B-v0.1":
        tokenizer.chat_template = "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"
    
    lora_path = snapshot_download(repo_id=model_checkpoint)
    llm = LLM(model=base_model_path, enable_lora=True)
    
    # Format prompts to be in proper format for model
    batched_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    formatted_prompts = tokenizer.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
    
    with open("baselines/adapter_mapping.json", "r") as stream:
        adapter_mapping = json.load(stream)
    
    if model_checkpoint in adapter_mapping:
        adapter_unique_id = adapter_mapping[model_checkpoint]
    else:
        max_id = 0
        for k in adapter_mapping:
            max_id = adapter_mapping[k]
        adapter_unique_id = max_id + 1
        adapter_mapping[model_checkpoint] = adapter_unique_id
        with open("baselines/adapter_mapping.json", "w") as stream:
            json.dump(adapter_mapping, stream)
 
    lora_request = LoRARequest(model_checkpoint, adapter_unique_id, lora_path)
    batched_outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
    return batched_outputs

def inference_honeybee_original(prompts, sampling_params):
    llm = LLM(model="yahma/llama-7b-hf", enable_lora=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    lora_request = LoRARequest("honeybee_adapter", 1, 'baselines/honeybee_weights')

    # Format prompts to be in proper format for model
    batched_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    # formatted_prompts = tokenizer.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
    formatted_prompts = prompts # v2
    batched_outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
    return batched_outputs

def inference_honeybee(prompts, sampling_params):
    llm = LLM(model="yahma/llama-7b-hf", enable_lora=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    lora_request = LoRARequest("honeybee_adapter", 1, 'baselines/honeybee_weights')

    # max_length = 2048  # Max context length for llama-7b-hf

    batched_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

    formatted_prompts = []
    for msg in batched_messages:
        formatted = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)

    batched_outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
    return batched_outputs

# Run ID: 1540179
def inference_on_dataset_vllm(dataset_path: str, model_checkpoint: str, rag_db: Union[List[str], None], lora=False):
    result = []

    with open(dataset_path, 'r') as stream:
        dataset = json.load(stream)
    
    for item in dataset:
        result.append({"dataset_item": item})
    
    if rag_db:
        tokenized_rag_db = [doc.split(" ") for doc in rag_db]
        bm25 = BM25Okapi(tokenized_rag_db)
    
    questions = []
    
    if model_checkpoint == "honeybee":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    else:
        tokenizer = None
    
    # Batch prompts
    for item in tqdm(result):
        question = item["dataset_item"]["question"]
        if rag_db:
            top_contexts = bm25.get_top_n(question.split(" "), rag_db, n=5)
            question = add_contexts_to_question(question=question, contexts=top_contexts, tokenizer=tokenizer)
            # Add retrieved context to "result" for saving
            item["inference_contexts"] = top_contexts
        
        questions.append(question)
    
    # Inference
    if lora:
        batched_outputs = inference_lora(model_checkpoint, prompts=questions)
    else:
        batched_outputs = inference(model_checkpoint, prompts=questions)
    
    # Store outputs
    for item, output in zip(result, batched_outputs):
        item["inference_result"] = output.outputs[0].text
        item["inference_model"] = model_checkpoint

    return result

def inference_yes_or_no_question(dataset_path: str, model_checkpoint: str, rag_db=None, lora=False, reasoning=False):
    result = []

    with open(dataset_path, 'r') as stream:
        dataset = json.load(stream)
    
    for item in dataset:
        result.append({"dataset_item": item})
        
    if rag_db:
        tokenized_rag_db = [doc.split(" ") for doc in rag_db]
        bm25 = BM25Okapi(tokenized_rag_db)
    
    questions = []

    if model_checkpoint == "honeybee":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    else:
        tokenizer = None

    # Batch prompts
    for item in result:
        if reasoning:
            question = item["dataset_item"]["true_false_question"] + "\n Provide your reasoning first, then answer the question with YES or NO on a new line."
        else:
            question = item["dataset_item"]["true_false_question"] + "\n You must answer the question with YES or NO. Do not include other words."
        if rag_db:
            top_contexts = bm25.get_top_n(question.split(" "), rag_db, n=5)
            question = add_contexts_to_question(question=question, contexts=top_contexts, tokenizer=tokenizer)
            # Add retrieved context to "result" for saving
            item["inference_contexts"] = top_contexts
        questions.append(question)
    
    # Inference
    if lora:
        batched_outputs = inference_lora(model_checkpoint, prompts=questions)
    else:
        batched_outputs = inference(model_checkpoint, prompts=questions)
    
    # Store outputs
    for item, output in zip(result, batched_outputs):
        item["inference_result"] = output.outputs[0].text
        item["inference_model"] = model_checkpoint

    return result





def inference_on_dataset_private_llm(dataset_path: str, rag_db = None, true_false=None):
    result = []

    with open(dataset_path, 'r') as stream:
        dataset = json.load(stream)
    
    for item in dataset:
        result.append({"dataset_item": item})

    if rag_db:
        tokenized_rag_db = [doc.split(" ") for doc in rag_db]
        bm25 = BM25Okapi(tokenized_rag_db)

    questions = []
    # Batch prompts
    for item in result:
        if true_false:
            # question = item["dataset_item"]["true_false_question"] + "\n You must answer the question with YES or NO. Do not include other words."
            question = item["dataset_item"]["true_false_question"] + "\n Provide your reasoning first, then answer the question with YES or NO on a new line."
        else:
            question = item["dataset_item"]["question"]
        if rag_db:
            top_contexts = bm25.get_top_n(question.split(" "), rag_db, n=5)
            question = add_contexts_to_question(question=question, contexts=top_contexts)
            # Add retrieved context to "result" for saving
            item["inference_contexts"] = top_contexts
        
        questions.append(question)


    # Prepare prompts
    prompts = questions
    batched_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    token_usage = {}
    # Inference
    batched_outputs = []
    for message in tqdm(batched_messages, desc=f"Inference with model {MODEL_CONFIG}: "):
        try:
            response_str = query_llm(message, token_usage=token_usage)
        except Exception as e:
            response_str = "Error: " + str(e)
        batched_outputs.append(response_str)
    
    # Store outputs
    for item, output in zip(result, batched_outputs):
        item["inference_result"] = output
        item["inference_model"] = MODEL_CONFIG
    print("Token usage: ", token_usage)
    return result



def inference_on_dataset_private_llm_2(dataset_path: str, rag_db = None, true_false=None):
    """"
    If previous run private llm partial failed due to billing reason, 
    this function can continue the inference from the last checkpoint.
    """

    with open(dataset_path, 'r') as stream:
        result = json.load(stream)
    token_usage = {}
    for item in tqdm(result, desc=f"Inference with model {MODEL_CONFIG}: "):
        if item.get("inference_result") is not None:
            continue
        if true_false:
            question = item["dataset_item"]["true_false_question"] + "\n Provide your reasoning first, then answer the question with YES or NO on a new line."
        else:
            question = item["dataset_item"]["question"]
        message = [{"role": "user", "content": question}]
        try:
            response_str = query_llm(message, token_usage=token_usage)
        except Exception as e:
            response_str = "Error: " + str(e)
        item["inference_result"] = response_str
    
    print("Token usage: ", token_usage)
    return result



def add_contexts_to_question(question: str, contexts: list[str], tokenizer=None):
    template = """{question}

Consider the following context when answering the context above:
{contexts}"""
    if tokenizer:
        # check if context is too long for honeybee
        tmp_prompt = ""
        for idx, context in enumerate(contexts):
            if idx == 0:
                tmp_prompt = template.format(question=question, contexts=context)
            else:
                tmp_prompt += "\n" + context
            
            tmp_question = [{"role": "user", "content": tmp_prompt}]
            formatted = tokenizer.apply_chat_template(tmp_question, tokenize=True, add_generation_prompt=True)
            if len(formatted) >= 1500:
                tmp_prompt = tmp_prompt[:len(tmp_prompt)-len(context)-1]
                # if len(tokenizer.apply_chat_template([{"role": "user", "content": tmp_prompt}], tokenize=True, add_generation_prompt=True)) >= 2048:
                #     print(tmp_prompt)
                return tmp_prompt

    return template.format(question=question, contexts="\n".join(contexts))

if __name__ == "__main__":
    dataset_path="MSQA_dataset.json"
    true_false_dataset_path="MSQA_dataset.json"
    """
    Inference on private LLMs
    """
    # result = inference_on_dataset_private_llm(dataset_path=true_false_dataset_path, rag_db=None, true_false=True)
    # output_prefix = "./baselines/results/inference"
    # output_file_name = MODEL_CONFIG + "-True-False-Reasoning.json"
    # with open(os.path.join(output_prefix, output_file_name), 'w') as stream:
    #     json.dump(result, stream, indent=4)

    # result = inference_on_dataset_private_llm_2(dataset_path="baselines/results/inference/GROK.json", rag_db=None)
    # output_prefix = "./baselines/results/inference"
    # output_file_name = MODEL_CONFIG + "_2.json"
    # with open(os.path.join(output_prefix, output_file_name), 'w') as stream:
    #     json.dump(result, stream, indent=4)


    """
    Inference on open LLMs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_checkpoint")
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--true_false", action="store_true")
    parser.add_argument("--reasoning", action="store_true")
    args = parser.parse_args()
    
    if args.rag:
        rag_dir = "baselines/rag_db/rag_database.json"
        with open(rag_dir, "r") as stream:
            rag_db = json.load(stream)
    else:
        rag_db = None
        
    if args.true_false:
        result = inference_yes_or_no_question(dataset_path=true_false_dataset_path, model_checkpoint=args.model_checkpoint, rag_db=rag_db, lora=args.lora, reasoning=args.reasoning)
    else:
        result = inference_on_dataset_vllm(dataset_path=dataset_path, model_checkpoint=args.model_checkpoint, rag_db=rag_db, lora=args.lora)
        
    output_prefix = "./baselines/results/inference"
    output_file_name = args.model_checkpoint.split("/")[-1] + ("-RAG" if args.rag else "") + ("-True-False" if args.true_false else "") + ("-Reasoning" if args.reasoning else "") + ".json"

    with open(os.path.join(output_prefix, output_file_name), 'w') as stream:
        json.dump(result, stream, indent=4)