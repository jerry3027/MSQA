import os
import json
from utils import *
from tqdm import tqdm


token_tracking_dict = {}

def generate_qa_paper(file):
    with open(file, "r",encoding='utf-8') as f:
        paper = json.load(f)
    key_abstract = None
    key_method = None
    key_result = None
    key_detail = None

    # Identify keys for abstract and method sections

    for key in paper.keys():
        if "abstract" in key.lower():
            key_abstract = key
        if key_method is None and "method" in key.lower() or "experimen" in key.lower():
            if paper[key]:  # Ensure method section is not empty
                key_method = key
        if key_result is None and "result" in key.lower() or "conclusion" in key.lower() or "summary" in key.lower():
            if paper[key]:
                key_result = key

    # Return None if necessary sections are missing
    if key_abstract is None:
        return None
    else:
        if key_method is None and key_result is None:
            return None

    # Define system and user prompts
    system_prompt = """
    Your role is to act as a materials science researcher with methodology knowledge about materials science researching.
    """

    user_prompt_1 = f"""
    Here is the "abstract" of a materials science paper. Please complete the following tasks:

    1. Summarize the purpose of the paper in clear and concise terms. 
    2. Classify the purpose as emphasizing "<method>" or "<result>".
    3. Identify research questions relevant to the abstract's themes and materials science interests.

    "Abstract": {paper[key_abstract]}
    """


    messages_1 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_1},
    ]

    # Query LLM for curated abstract and questions
    llm_curate_abstract = query_llm(messages_1, token_usage=token_tracking_dict)

    if "<method>" in llm_curate_abstract and key_method:
        key_detail = key_method
    elif "<result>" in llm_curate_abstract and key_result:
        key_detail = key_result
    else:
        key_detail = key_result if key_result else key_method

    user_prompt_2 = f"""I will provide the purpose of a materials science paper, related research questions, and a detailed section of the paper.

Your tasks:
    1. Select the Most Relevant Question: Choose the research question that is most specific, clearly phrased, and directly related to the provided section.
    2. Refine the Question: Modify the selected question to ensure it is:
        - Grounded on information from the provided section, but answerable even without using the provided section.
        - Standalone and unambiguous. Do not use definite articles when referring to compounds.
        - Clearly phrased for precision.
    3. Generate a Direct Answer: Provide a concise and well-structured response that:
        - Directly answers the question.
        - Is based on the provided section but remains meaningful out of context.
        - Avoids vague references such as “this study” or “this paragraph.”
        - Clearly conveys the information without requiring the reader to see the original section.

Present the output as a JSON shown below:
{{
    "question": "A clear and specific question.",
    "answer": "A concise and relevant answer that remains meaningful without additional context."
}}

    Input data:
    - "Purpose and related questions": {llm_curate_abstract}
    - "Detailed section": {paper[key_detail]}
    """

    messages_2 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_2},
    ]

    # Generate questions and answers
    generated_question = query_llm(messages_2, is_output_json=True, token_usage=token_tracking_dict)
    print("Cumulative tokens used:", token_tracking_dict)

    
    if generated_question:
        if generated_question.get("question") and generated_question.get("answer"):
            if any(word in generated_question['question'].lower() for word in bad_keywords["bad_words_question"]):
                return None
            if any(word in generated_question['answer'].lower() for word in bad_keywords["bad_words_answer"]):
                return None

    qa_pair = {
        "question": generated_question["question"],
        "answer": generated_question["answer"],
        "topic": llm_curate_abstract,
        "source": file,
        "source_section": key_detail
    }

    # generated_qa_pairs = None
    # if json_response:
    #     generated_qa_pairs = []
    #     for qa_pair in json_response:
    #         if qa_pair.get("question") and qa_pair.get("answer"):
    #             # Key Words Filter
    #             if any(word in qa_pair['question'].lower() for word in bad_keywords["bad_words_question"]):
    #                 continue
    #             elif any(word in qa_pair['answer'].lower() for word in bad_keywords["bad_words_answer"]):
    #                 continue
    #             qa_pair["topic"] = llm_curate_abstract
    #             qa_pair["source"] = file
    #             qa_pair["source_section"] = key_detail

    #             generated_qa_pairs.append(qa_pair)
    return qa_pair


def process_papers(input_folder, output_folder, num_files):
    json_files = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".json")
    ]
    valid_count = 0

    if num_files == -1:
        num_files = len(json_files)

    with tqdm(total=num_files, desc="Processing files", unit="file") as pbar:
        for file in json_files:
            file_name = file.split("/")[-1]
            if valid_count < num_files:
                try:
                    # Generate questions
                    generated_qa_pairs = generate_qa_paper(file)
                    if generated_qa_pairs: # and len(generated_qa_pairs)>0:
                        valid_count += 1
                        with open(f"{output_folder}/{file_name}", "w",encoding='utf-8') as stream:
                            json.dump(generated_qa_pairs, stream, indent=4)
                            generated_qa_pairs = []
                        pbar.update(1)  # Update the progress bar
                    else:
                        print(f"File does not contain experiment or result section OR is QA pair is filtered by keywords. {file}")

                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue
            else:
                break

def generate_answer_with_other_models(output_folder):
    # Make output dir if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Only complete difference
    input_folder = "../data/msqa_3000_4o"
    to_be_completed = os.listdir(input_folder)
    already_completed = os.listdir(output_folder)
    
    to_be_completed = [item for item in to_be_completed if item not in already_completed]
    
    json_files = [
        os.path.join(input_folder, f) for f in to_be_completed if f.endswith(".json")
    ]
        
    print(f"Number of files to generate: {len(json_files)}")
    

    system_prompt = """Your role is to act as a materials science researcher with methodology knowledge about materials science researching."""
    
    user_prompt_template = """I will provide a materials science research question, the purpose of a materials science paper, and a detailed section of the paper related to the question.

Your tasks is to provide a concise and well-structured short paragraph that:
    - Directly answers the question.
    - Is based on the provided section but remains meaningful out of context.
    - Avoids vague references such as “this study” or “this paragraph.”
    - Clearly conveys the information without requiring the reader to see the original section.

Input data:
- "Materials science question": {question} 
- "Purpose and related questions": {llm_curate_abstract}
- "Detailed section": {source_section}"""

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as stream:
            json_file_content = json.load(stream)
    
        question = json_file_content["question"]
        topic = json_file_content["topic"]
        # Remove relevant research questions from topic
        if "### Relevant Research Questions" in topic:
            topic = topic[:topic.rindex("### Relevant Research Questions")]
        elif "### Task 3:" in topic:
            topic = topic[:topic.rindex("### Task 3:")]
        elif "### 3." in topic:
            topic = topic[:topic.rindex("### 3.")]
        else:
            topic = topic[:topic.rindex("3.")]

        source_section_name = json_file_content["source_section"]      
        with open(json_file_content["source"], "r", encoding="utf-8") as stream:
            source_section = json.load(stream)[source_section_name]
        
        user_prompt = user_prompt_template.format(question=question, llm_curate_abstract=topic, source_section=source_section)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generate questions and answers
        answer = query_llm(messages, token_usage=token_tracking_dict)
        print("Cumulative tokens used:", token_tracking_dict)

        qa_pair = {
            "question": json_file_content["question"],
            "answer": answer,
            "topic": json_file_content["topic"],
            "source": json_file_content["source"],
            "source_section": json_file_content["source_section"]
        }

        file_name = json_file.split("/")[-1]
        with open(f"{output_folder}/{file_name}", "w",encoding='utf-8') as stream:
            json.dump(qa_pair, stream, indent=4)




if __name__ == "__main__":
    #Generate question and answer
    input_folder = "PAPER_FOLDER"  
    output_folder = "../data/msqa_3000_4o"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    process_papers(input_folder, output_folder, num_files=3000)

    # Generate Deepseek
    # deepseek_output_folder = "../data/msqa_3000_deepseek"
    # generate_answer_with_other_models(deepseek_output_folder)
    
    # Generate Gemini
    # gemini_output_folder = "../data/msqa_3000_gemini"
    # generate_answer_with_other_models(gemini_output_folder)
