from utils import query_llm
import json
from tqdm import tqdm
from pathlib import Path
import random
import os
import re

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

def process_folders(folder1, folder2, folder3, output_file, num=20):
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
        


def convert_to_final_dataset_format(input_path):
    with open(input_path, "r") as stream:
        merged_answer_json = json.load(stream)

    gpt_4o_path = "../data/msqa_3000_4o"

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
    # process_folders("../data/msqa_3000_4o", "../data/msqa_3000_deepseek", "../data/msqa_3000_gemini", output_file="final_merged.json", num=-1)
    convert_to_final_dataset_format("../data/final_merged_agg_dataset.json")