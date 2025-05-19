from utils import query_llm, gpt_judger
import json
from tqdm import tqdm
short_answer_system_prompt="""
You are an AI assistant specialized in processing material science QA datasets. Your goal is to analyze a long answer and attempt to extract a short answer while following these strict rules:

1. Analyze the given QA pair carefully: For each given question and its long answer, determine if a short answer can be extracted verbatim that fully and directly answers the question.
2. Short Answer Criteria: A valid short answer must be:
   - A single term or phrase such as:
     - A material name (e.g., `"Silicon"`)
     - A numerical value (e.g., `"3.5 GPa"`)
     - A specific scientific term (e.g., `"Amorphous structure"`)
   - `"YES"` or `"NO"` if the question is a boolean type (e.g., starts with `"Can"`, `"Does"`, `"Is"`, etc.).
   - It must fully answer the question in the short form. If it does not directly answer the question, return `"NULL"`.
3. Verbatim Extraction Only: Extract the short answer exactly as it appears in the long answer without rewording or summarizing.
4. NULL for Non-Extractable Answers: If a short answer cannot be directly extracted while fully answering the question, return `"NULL"`.
5. No Inference: Do not generate or infer new information beyond what is explicitly stated in the long answer.
6. Return Format: The output must always be either:
   - A verbatim short answer
   - `"YES"` or `"NO"` for boolean questions
   - `"NULL"` if no short answer can be extracted that fully answers the question.
"""

short_answer_user_prompt = """
Task:
Process the following QA pair and attempt to extract a short answer verbatim. If a short answer cannot be directly extracted while fully answering the question, return `"NULL"`.

Input Data:
-- Material Science Question: {question}
-- Long Answer: {long_answer}

Output Format (JSON):
{{
    "short answer": "string"
}}"""

# 2978373
def convert_short_answer(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        question_json = json.load(f)
    final_list = []

    for item in tqdm(question_json, desc="Covert Short Answer ing: "):
        question = item["question"]
        answer = item["answer"]
        
        # Skip NULL answers (merge inconsistency)
        if answer == "NULL":
            continue

        messages = [
            {"role": "system", "content": short_answer_system_prompt},
            {"role": "user", "content": short_answer_user_prompt.format(question=question, long_answer=answer)}
        ]
        short_answer = query_llm(messages, is_output_json=True)
        item["short_answer"] = short_answer.get("short answer")
        final_list.append(item)

    if final_list and len(final_list)>0:
        output_path = input_file.split(".json")[0]+"_shortAnswer.json"
        with open(f"{output_path}", "w",encoding='utf-8') as stream:
            json.dump(final_list, stream, indent=4)

def judge_short_answer(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        question_json = json.load(f)
    final_list = []
    for item in tqdm(question_json, desc="Judge Short Answer ing: "):
        question = item["question"]
        answer = item["answer"]
        short_answer = item["short_answer"]
        if short_answer == "NULL":
            continue
        judge = gpt_judger(question, answer, short_answer)
        item["short_answer_judgment"] = judge.get("judgment")
        item["short_answer_jude_reasoning"] = judge.get("reasoning")
        final_list.append(item)
       
    if final_list and len(final_list)>0:
        output_path = input_file.split(".json")[0]+"_judge.json"
        with open(f"{output_path}", "w",encoding='utf-8') as stream:
            json.dump(final_list, stream)

# judge_short_answer("temp_files/pipline2_output_merge_shortAnswer.json")
# convert_short_answer("final_merged_dataset.json")

def extract_short_answer_only(input_file):
    with open(input_file, "r", encoding="utf-8") as stream:
        dataset_file = json.load(stream)
    
    res = []
    
    for item in dataset_file:
        if item["short_answer"] != "NULL":
            res.append(item)
    
    return res

def convert_dataset_to_true_false_question(input_file):
    user_prompt = """Given a scientific or technical question and its detailed answer, rewrite the original question into a short, technically accurate YES/NO question in which the correct answer is {yes_or_no} that:
    - Focuses on the single most critical mechanism, cause, or concept explained in the answer.
    - Preserves all necessary scientific nuance, avoiding oversimplifications or distortions.
    - Uses natural, domain-appropriate phrasing.
    - Is challenging. The question should not just require surface-level knowledge.
    - Avoids unnecessary length.
    - Prefer approximations ("around", "more or less") where rigid details are non-essential to understanding, but do not use approximations if precise concepts (e.g., named electronic states, molecular structures) are central.
    - Is answered by NO.

Important: Do not rephrase the entire explanation, and do not add background or context in the question.

Input Data:
-- Material Science Question: {question}
-- Long Answer: {long_answer}

Output Format (JSON):
{{
    "true_false_question": "string"
}}"""


    with open(input_file, "r", encoding="utf-8") as f:
            question_json = json.load(f)
    
    final_list = []

    for idx, item in enumerate(tqdm(question_json)):
        question = item["question"]
        answer = item["answer"]
        
        # Skip NULL answers (merge inconsistency)
        if answer == "NULL":
            continue
        
        true_or_false = "YES" if idx % 2 == 0 else "NO"
        
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_prompt.format(yes_or_no=true_or_false, question=question, long_answer=answer)}
        ]
        
        true_false_question = query_llm(messages, is_output_json=True)
        item["true_false_question"] = true_false_question.get("true_false_question")
        item["true_false_question_answer"] = true_or_false
        final_list.append(item)

    if final_list and len(final_list)>0:
        output_path = input_file.split(".json")[0]+"_with_true_false.json"
        with open(f"{output_path}", "w",encoding='utf-8') as stream:
            json.dump(final_list, stream, indent=4)




if __name__ == "__main__":
    # convert_short_answer("data/final_dataset_agg_with_question_types.json")
    # with open("data/final_dataset_agg_with_question_types_shortAnswer.json", "r") as stream:
    #     f = json.load(stream)
    
    # for i in f:
    #     if i["short_answer"] != "NULL":
    #         print(i["question"])
        
    # res = extract_short_answer_only("data/final_dataset_agg_with_question_types_shortAnswer.json")
    # with open("data/short_answer_only.json", "w") as stream:
    #     json.dump(res, stream, indent=4)
    # print(len(res))
    
    convert_dataset_to_true_false_question("data/final_dataset_agg_with_question_types.json")