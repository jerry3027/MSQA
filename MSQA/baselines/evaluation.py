import asyncio
import os

import aiohttp
from utils import async_gpt_judger, gpt_judger
from baselines.inference import *
from tqdm import tqdm
import time




# Pid for llama3_evaluation: 4108165
token_tracking_dict = {}

def evaluate_with_gpt_judge(dataset_path, model_checkpoint, with_context=False):
    if with_context:
        # Build RAG DB if not exist
        rag_dir = "baselines/rag_db/rag_database.json"
        if not os.path.exists(rag_dir):
            build_rag_database("ms_pipeline2_qa_5000_4o_mini/", rag_dir)
        with open(rag_dir, "r") as stream:
            rag_db = json.load(stream)
    else:
        rag_db = None
    
    inference_result = inference_on_dataset_vllm(dataset_path, model_checkpoint, rag_db=rag_db)
            
    for item in tqdm(inference_result):
        question = item["dataset_item"]["question"]
        gold_answer = item["dataset_item"]["answer"]
        raw_answer = item["inference_result"]
        
        judgement = gpt_judger(question, gold_answer, raw_answer, token_tracking_dict=token_tracking_dict)
        print("Cumulative tokens used:", token_tracking_dict)
        
        if judgement != None:
            item["judgment_reasoning"] = judgement["reasoning"]
            item["judgment"] = judgement["judgment"]
        else:
            item["judgment_reasoning"] = ""
            item["judgment"] = ""
   
    return inference_result

def evaluate_true_false_by_file(inference_dataset_path):
    with open(inference_dataset_path, "r") as file:
        inference_result = json.load(file)
    inference_result= filter_bad_question(inference_result)
    for item in inference_result:
        gold_answer = item["dataset_item"]["true_false_question_answer"]
        raw_answer = item["inference_result"]
        if "Reasoning" in inference_dataset_path:
            if gold_answer == "YES" and "YES" in raw_answer[-5:] or gold_answer == "NO" and "NO" in raw_answer[-5:]:
                item["judgment"] = "correct"
            else:
                item["judgment"] = "incorrect"
        else:    
            if gold_answer == "YES" and "YES" in raw_answer or gold_answer == "NO" and "NO" in raw_answer:
                item["judgment"] = "correct"
            else:
                item["judgment"] = "incorrect"
    print("dataset_path: ", inference_dataset_path)
    print(compute_metrics(inference_result))
    
    return inference_result


def evaluate_with_gpt_judge_by_file(inference_dataset_path):
    
    with open(inference_dataset_path, "r") as file:
        inference_result = json.load(file)

    for item in tqdm(inference_result):
        question = item["dataset_item"]["question"]
        gold_answer = item["dataset_item"]["answer"]
        raw_answer = item["inference_result"]
        
        judgement = gpt_judger(question, gold_answer, raw_answer, token_tracking_dict=token_tracking_dict)
        
        try:
            if isinstance(judgement, str):
                judgement = json.loads(judgement)
            item["judgment_reasoning"] = judgement["reasoning"]
            item["judgment"] = judgement["judgment"]
        except Exception as e:
            print("gpt_judger Error: ", e)
            print("get judge output: ", judgement)
            item["judgment_reasoning"] = ""
            item["judgment"] = ""

    print("Cumulative tokens used:", token_tracking_dict)
    return inference_result


async def async_evaluate_file(inference_dataset_path):
    
    sem = asyncio.Semaphore(10)
    
    with open(inference_dataset_path, "r") as file:
        inference_result = json.load(file)
    
    tasks = []
    
    with open("data/final_dataset_filtered.json", "r") as file:
        filtered_good_question = json.load(file)
    
    good_questions = set()
    for item in filtered_good_question:
        good_questions.add(item["question"])
    
    for item in inference_result:
        question = item["dataset_item"]["question"]
        gold_answer = item["dataset_item"]["answer"]
        raw_answer = item["inference_result"]
        
        if question not in good_questions:
            continue
        
        tasks.append(async_gpt_judger(question, gold_answer, raw_answer, sem, token_tracking_dict=token_tracking_dict))

    judgments = await asyncio.gather(*tasks, return_exceptions=True)
    
    print("Cumulative tokens used:", token_tracking_dict)

    for idx, judgment in enumerate(judgments):
        if judgment: # Skip instances where GPT throws an error!
            try:
                inference_result[idx]["judgment_reasoning"] = judgment["reasoning"]
                inference_result[idx]["judgment"] = judgment["judgment"]
            except:
                print("Key Error for judgment", judgment)
                continue

    return inference_result

def build_rag_database(input_dir, output_path):
    rag_db = []
    
    f_names = os.listdir(input_dir)
    for f_name in f_names:
        # Opening datset
        path = os.path.join(input_dir, f_name)
        with open(path, "r") as stream:
            dataset_item = json.load(stream)
        source_path = dataset_item["source"]
        source_section = dataset_item["source_section"]
        # Getting context
        with open(source_path, "r") as stream:
            context = json.load(stream)[source_section]
        rag_db.extend(context)
        
    with open(output_path, "w+") as stream:
        json.dump(rag_db, stream, indent=4)

    return rag_db

def compute_metrics(inference_result_with_judgements):
    correct = 0
    mostly_correct = 0
    total = 0
    for item in inference_result_with_judgements:
        if "judgment" not in item:
            continue
        judgement = item["judgment"].lower()
        if judgement == "correct":
            correct += 1 
        elif judgement == "mostly correct":
            mostly_correct += 1
        total += 1
    return f"{correct}/{mostly_correct}/{total-correct-mostly_correct} -> {round((correct+mostly_correct)/(total+0.00001),4)*100}%"   

def compute_metrics_by_question_type(inference_result):
    results = {
        "structure-property relationships": [0, 0, 0], 
        "synthesis and processing": [0, 0, 0], 
        "computational": [0, 0, 0], 
        "material analysis techniques": [0, 0, 0], 
        "material modeling": [0, 0, 0], 
        "failure analysis and degradation": [0, 0, 0]
    }

    question_type_mapping = {}
    with open("data/final_dataset_agg_with_question_types.json", "r") as stream:
        raw_file = json.load(stream)
    for i in raw_file:
        question_type_mapping[i["question"]] = i["question_type"]
 
    for item in inference_result:
        if "judgment" not in item:
            continue
        curr_question_type = question_type_mapping[item["dataset_item"]["question"]]
        if curr_question_type in results:
            judgement = item["judgment"].lower()
            if judgement == "correct":
                results[curr_question_type][0] += 1 
            elif judgement == "mostly correct":
                results[curr_question_type][1] += 1
            else:
                results[curr_question_type][2] += 1
    for k in results:
        print(k, results[k], round((results[k][0] + results[k][1]) / (results[k][0] + results[k][1] + results[k][2]), 4)*100)
    return results

def filter_bad_question(dataset, add_question_type=False):
    with open("data/final_dataset_filtered.json", "r") as file:
            filtered_good_question = json.load(file)
        
    # Normalize questions from the JSON
    good_questions = set()
    for item in filtered_good_question:
        good_questions.add(item["question"])
    
    filtered_dataset = [item for item in dataset if item["dataset_item"]["question"] in good_questions]
    # add filtered_good_question["question_type"] to the filtered dataset
    if add_question_type:
        for item in filtered_dataset:
            for good_item in filtered_good_question:
                if item["dataset_item"]["question"] == good_item["question"]:
                    item["dataset_item"]["question_type"] = good_item["question_type"]
                    break
    
    return filtered_dataset

def update_table_5():
    folder_path = "baselines/results/eval/4o-mini"
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path) and "RAG" not in filename:
            with open(full_path, "r") as file:
                inference_result = json.load(file)
                inference_result= filter_bad_question(inference_result, add_question_type=True)
                print("dataset_path: ", filename)
                print(compute_metrics(inference_result))
                print("==========================")


def check_Qwen_question_type_accuracy():
    """
    Update Table 6 in the paper
    """
    with open("baselines/results/eval/4o/Phi-4-mini-instruct-Eval.json", "r") as file:
        inference_result = json.load(file)
    inference_result= filter_bad_question(inference_result, add_question_type=True)
    all_count={'structure-property relationships': 818, 'synthesis and processing': 257, 'computational': 216, 'material analysis techniques': 187, 'material modeling': 125, 'failure analysis and degradation': 93, 'material properties': 61}
    correct_count={'structure-property relationships': 0, 'synthesis and processing': 0, 'computational': 0, 'material analysis techniques': 0, 'material modeling': 0, 'failure analysis and degradation': 0, 'material properties': 0}    
    for item in inference_result:
        curr_question_type = item["dataset_item"]["question_type"]
        judgement = item.get("judgment", "incorrect")
        if judgement != "incorrect":
            correct_count[curr_question_type] += 1
    #print correct_count ratio
    for k in correct_count:
        print(k, correct_count[k], all_count[k], round(correct_count[k] / all_count[k], 4)*100)
    
def eval_all_with_new_gpt_prompt():
    exist_eval_list = ["CLAUDE","GEMINI","GROK","Meta-Llama","gemini-pro","4O"]
    folder_path = "baselines/results/inference"
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path) and "True-False" not in filename:
            need_re_eval =True
            for exist_eval in exist_eval_list:
                if exist_eval in filename:
                    need_re_eval = False
                    break
            if need_re_eval:
                file_path = full_path
                print(f"started at {time.strftime('%X')}")
                result = asyncio.run(async_evaluate_file(file_path))
                # Accuracy with GPT as Judge
                print("Accuracy:", compute_metrics(result))
                output_prefix = "./baselines/results/eval/4o"
                output_file_name = file_path.split("/")[-1][:-5] + "-Eval.json" 
                print("output_file_name: ", output_file_name)
                with open(os.path.join(output_prefix, output_file_name), "w") as stream:
                    json.dump(result, stream, indent=4)
                print(f"finished at {time.strftime('%X')}")


def get_all_matrix():    
    folder_path = "baselines/results/inference"
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            if "True-False" in filename:
                evaluate_true_false_by_file(full_path)
            print("==========================")

    folder_path = "baselines/results/eval/4o"
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path) and "True-False" not in filename:
            with open(full_path, "r") as file:
                inference_result = json.load(file)
            print("dataset_path: ", filename)
            print(compute_metrics(inference_result))
            print("==========================")


 

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--inference_file_path", nargs='+')
    args = parser.parse_args()

    # # Run with async
    inference_file_paths = args.inference_file_path
    
    for file_path in inference_file_paths:
        if "True-False" in file_path:
            evaluate_true_false_by_file(file_path)
        else:
            print(f"started at {time.strftime('%X')}")
            result = asyncio.run(async_evaluate_file(file_path))
            # Accuracy with GPT as Judge
            print("Accuracy:", compute_metrics(result))
            output_prefix = "./baselines/results/eval/4o"
            output_file_name = file_path.split("/")[-1][:-5] + "-Eval.json" 
            with open(os.path.join(output_prefix, output_file_name), "w") as stream:
                json.dump(result, stream, indent=4)
            print(f"finished at {time.strftime('%X')}")

    
















    
    

            

    