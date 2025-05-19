from utils import query_llm
import json
from tqdm import tqdm
from pathlib import Path
import random
import os
import re

# run ID: 3714326
token_usage = {}
system_prompt = ""
user_prompt = """
You are given a materials science domain question along with three different answers. Your task is to:

1. **Extract key bullet points** from any of the three answers. A key bullet point should represent a significant fact, concept, numerical value, or conclusion related to the question.
2. **Evaluate consistency** across the three answers for each extracted bullet point:
   - **1** → The answer explicitly mentions or supports the bullet point.
   - **0** → The answer does not mention the bullet point.
   - **-1** → The answer contradicts or disagrees with the bullet point.

## **Input Format** ##
**Question:** {question}  
**Answer1:** {answer1}  
**Answer2:** {answer2}  
**Answer3:** {answer3}  

## **Output Format (JSON)** ##
{{
    "result": [
        {{
            "point": "Extracted bullet point from the answers",
            "consistency": {{
                "Answer1": -1/0/1, 
                "Answer2": -1/0/1, 
                "Answer3": -1/0/1
            }}                    
        }},
        ...
    ]
}}
"""

user_prompt_v2 = """### Here is the problem:
"question": {question},

### Reference Solutions:
Solution 1: {answer1}

Solution 2: {answer2}

Solution 3: {answer3}


### Instructions:
1. Review the above solutions.
2. Generate an improved and refined solution by aggregating the strengths from the provided solutions. Enclose the solution within <SOLUTION> and </SOLUTION> tag.
3. Provide a brief explanation of your reasoning.
4. Ensure your answer is clear, concise, and structured logically.
"""


def load_answers_from_folder(folder_path):
    """Load all JSON files from a folder and return a dictionary {filename: content}."""
    answers = {}
    for file_path in Path(folder_path).glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:  # Check if the file is empty
                print(f"Warning: Empty file detected -> {file_path}")
                continue
            try:
                answers[file_path.name] = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file: {file_path} -> {e}")
                continue
    return answers

# 3038088
def process_folders_v2(folder1, folder2, folder3, output_file, num=20):
    answers1 = load_answers_from_folder(folder1) #gpt
    answers2 = load_answers_from_folder(folder2) #deepseek
    answers3 = load_answers_from_folder(folder3) #gemini
    
    result = []
    count = 0
    
    for filename in tqdm(answers1.keys()):
        count += 1
        if count > num and num != -1:
            break

        q1, q2, q3 = answers1[filename], answers2[filename], answers3[filename]
        
        shuffle_list = [("GPT",q1["answer"]), ("Deepseek",q2["answer"]),("Gemini", q3["answer"])]
        random.shuffle(shuffle_list)
        
        dic = {"answer2source":{"Answer1":shuffle_list[0][0], "Answer2":shuffle_list[1][0], "Answer3": shuffle_list[2][0] },
            "source2answer": {shuffle_list[0][0]:"Answer1", shuffle_list[1][0]:"Answer2", shuffle_list[2][0]:"Answer3"}
        }        
    
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_v2.format(question = q1["question"], answer1=shuffle_list[0][1], answer2=shuffle_list[1][1], answer3=shuffle_list[2][1])},
        ]


        raw_response = query_llm(messages, token_usage=token_usage)
        print(f"Cumulative token usage: {token_usage}")
        
        response = re.search("<SOLUTION>(.*)</SOLUTION>", raw_response, re.DOTALL)
        if not response: # Pattern not found
            continue
        
        response = response.group(1).strip()
        
        rtn = {}
        rtn["mapping"] = dic
        rtn["question"] = q1["question"]
        rtn["all_answers"] = shuffle_list
        rtn["file_name"] = filename
        rtn["final_answer"] = response
        rtn["raw_gpt_answer"] = raw_response
        
        result.append(rtn)
        
    # Save results to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Consistency evaluation saved to {output_file}")
        
def process_folders(folder1, folder2, folder3, output_file="long_answer_consistency_results_diverse3000.json", num = 20):
    """Load answers from three folders, evaluate consistency, and determine the most correct answer."""
    answers1 = load_answers_from_folder(folder1) #gpt
    answers2 = load_answers_from_folder(folder2) #deepseek
    answers3 = load_answers_from_folder(folder3) #gemini
    count = 0
    result = []
    if num == -1:
        num = len(answers1)
    
    # Iterate through all matching files (since filenames are the same)
    with tqdm(total=num, desc=f"Measure long answer consistency:") as pbar:
        for filename in answers2.keys():
        
            if count >= num:
                break
            count+=1
            pbar.update(1)
            try:
                q1, q2, q3 = answers1[filename], answers2[filename], answers3[filename]
            except:
                continue

            # Ensure the questions are identical
            if not (q1["question"] == q2["question"] == q3["question"]):
                print(f"Warning: Mismatched questions in {filename}")
                continue
            
            shuffle_list = [("GPT",q1["answer"]), ("Deepseek",q2["answer"]),("Gemini", q3["answer"])]
            random.shuffle(shuffle_list)
            dic = {"answer2source":{"Answer1":shuffle_list[0][0], "Answer2":shuffle_list[1][0], "Answer3": shuffle_list[2][0] },
                "source2answer": {shuffle_list[0][0]:"Answer1", shuffle_list[1][0]:"Answer2", shuffle_list[2][0]:"Answer3"}
            }


            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(question = q1["question"], answer1=shuffle_list[0][1], answer2=shuffle_list[1][1], answer3=shuffle_list[2][1])},
            ]
            # Extract consistency report
            response = query_llm(messages, is_output_json=True, token_usage=token_usage)
            print(f"Cumulative token usage: {token_usage}")
            
            rtn = {}
            rtn["mapping"] = dic
            rtn["question"] = q1["question"]
            rtn["all_answers"] = shuffle_list
            rtn["file_name"] = filename
            
            rtn["bullet_points"] = response["result"]
            bullet_points = response["result"]
            consistency = [(sorted(list(point["consistency"].values()))) for point in bullet_points]
            final_consistency = ["Answer1", "Answer2", "Answer3"]
            """
            Consistency including:
            if -1, return disagree
            else:
                001 -> discard the bullet point. Select the one in 0 0.
                011 -> Select the one in 1 1.
                111 -> complete consistence
            """

            for i in range(len(consistency)):
                if -1 in consistency[i]:
                    final_consistency = []
                    break
                elif [0,0,1] == consistency[i]:
                    for key,answer in bullet_points[i]["consistency"].items():
                        if answer == 1:
                            try:
                                final_consistency.remove(key)
                            except:
                                continue
                elif [0,1,1] == consistency[i]:
                    for key,answer in bullet_points[i]["consistency"].items():
                        if answer == 0:
                            try:
                                final_consistency.remove(key)
                            except:
                                continue
            
            rtn["final_consistency"] = final_consistency 
            if len(rtn["final_consistency"])>0:
                index = int(final_consistency[0][-1])-1
                rtn["final_answer"] = rtn["all_answers"][index][1]
            else:
                rtn["final_answer"] = "NULL"



            result.append(rtn)

    # Save results to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Consistency evaluation saved to {output_file}")

def convert_to_final_dataset_format(input_path):
    with open(input_path, "r") as stream:
        merged_answer_json = json.load(stream)

    gpt_4o_path = "data/diverse_sampled_paper_dataset/ms_qa_3000_4o"

    final_dataset = []

    for item in merged_answer_json:
        final_answer = item["final_answer"]

        with open(os.path.join(gpt_4o_path, item["file_name"]), "r") as stream:
            source_item = json.load(stream)

        source_item["answer"] = final_answer
        final_dataset.append(source_item)

    print(len(final_dataset))
    with open("final_dataset.json", "w") as stream:
        json.dump(final_dataset, stream, indent=4)

if __name__ == "__main__":            
    # Example usage
    # process_folders("data/diverse_sampled_paper_dataset/ms_qa_3000_4o", "data/diverse_sampled_paper_dataset/ms_qa_3000_deepseek_v2", "data/diverse_sampled_paper_dataset/ms_qa_3000_gemini_v2", num = -1,
    #                 output_file="final_merged_dataset.json")
    # print("Total token_usage: ", token_usage)

    # process_folders_v2("data/diverse_sampled_paper_dataset/ms_qa_3000_4o", "data/diverse_sampled_paper_dataset/ms_qa_3000_deepseek_v2", "data/diverse_sampled_paper_dataset/ms_qa_3000_gemini_v2", output_file="final_merged.json", num=-1)
    convert_to_final_dataset_format("data/final_merged_agg_dataset.json")