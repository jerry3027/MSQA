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

def generate_multi_step_qa(question_json, output_file, method):
    """ This method generates mutli-step QA pairs by decomposing the question
    to sub-questions, answering those decomposed questions with additional context
    and comparing the question answered with multi-step reasoning with the 
    original answer. Only matching answers are kept.

    Args:
        question_json (dict): The question_json must contain "question" and "answer" key.
        output_file (str): Path to save the generated QA pairs.
        num_files (int): Number of files to process from the folder.
        method (str): Method for generating mutli-step questions. Can only take two values: decompose OR cot.
    """

    multi_step_qa_pairs = []

    if method == "decompose":
        system_prompt = DECOMPOSITION_SYSTEM_PROMPT_V2
    elif method == "cot":
        system_prompt = COT_SYSTEM_PROMPT

    user_prompt = """Consider the following question and its answer:
Question: {question}
Answer: {answer}"""

    for item in question_json:
        question = item["question"]
        answer = item["answer"]

        # Decomposing the question into subquestions
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=question, answer=answer)}
        ]

        try:
            multi_step_answer = query_llm(messages=messages, is_output_json=True, token_usage=token_tracking_dict)["response"]
        except Exception as e:
            multi_step_answer = None
            print(f"Error in breaking down question: {question}. Error message: {e}")
        
        # Compare the generated questions with the original answer
        if multi_step_answer: # and compare_multi_step_answer_with_original_answer(sub_questions[-1]["answer"], answer):
            multi_step_qa_pairs.append({
                "question": question,
                "answer": answer,
                "multi_step_answer": multi_step_answer
            })

    with open(output_file, "w") as stream:
        json.dump(multi_step_qa_pairs, stream, indent=4)


def generate_coherent_multi_step_qa(question_json, output_file):
    multi_step_qa_pairs = []
    
    system_prompt = COHERENT_COT_SYSTEM_PROMPT.format(MAX_SEARCH_LIMIT=3)
    user_prompt = "Consider the following question: {question}"
    continued_user_prompt = user_prompt + "\n\nContinue the reasponing process shown below, make sure you include the previous reasoning process in your response:\n{previous_reasoning_process}"

    for item in question_json:
        question = item["question"]
        answer = item["answer"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=question)}
        ]

        multi_step_answer = query_llm(messages=messages, is_output_json=True, token_usage=token_tracking_dict)["response"]
        
        # Continue searching for context + generating reasoning chain unitil finish 
        web_search_idx = 0
        for _ in range(5):
            # Break if API errored
            if not multi_step_answer:
                break
            
            # Remove steps after "Web Search"
            for i, step in enumerate(multi_step_answer):
                # Check if Web Search is in action
                if step["action"] == "Web Search" and i >= web_search_idx:
                    web_search_idx = i
                    break
            multi_step_answer = multi_step_answer[:web_search_idx+1]

            if multi_step_answer and multi_step_answer[-1]["action"] == "Web Search":
                multi_step_answer.append({
                    "action": "Search Tool Response",
                    "content": refined_website_results(query=multi_step_answer[-1]["query"], previous_reasoning=json.dumps(multi_step_answer[:-1]))
                })
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": continued_user_prompt.format(question=question, previous_reasoning_process=json.dumps(multi_step_answer))}
                ]
                # Generate continued reasoning chain with LLM
                multi_step_answer = query_llm(messages=messages, is_output_json=True, token_usage=token_tracking_dict)["response"]
                web_search_idx = len(multi_step_answer)-1         
            else:
                break

        # Compare the generated questions with the original answer
        if multi_step_answer: # and compare_multi_step_answer_with_original_answer(sub_questions[-1]["answer"], answer):
            multi_step_qa_pairs.append({
                "question": question,
                "answer": answer,
                "multi_step_answer": multi_step_answer
            })

    if output_file:
        with open(output_file, "w") as stream:
            json.dump(multi_step_qa_pairs, stream, indent=4)
        
    return multi_step_qa_pairs


def refined_website_results(query, previous_reasoning):
    website_search_results = get_external_resource(query, top_k=3)

    user_prompt = """- Previous Reasoning Steps:
{prev_reasoning}
- Current Search Query:
{search_query}
- Searched Web Pages:
{document}
Now you should analyze each web page and find helpful information based on the current search query
“{search_query}” and previous reasoning steps."""

    messages = [
        {"role": "system", "content": REASON_IN_DOCUMENT_PROMPT},
        {"role": "user", "content": user_prompt.format(prev_reasoning=previous_reasoning, search_query=query, document=json.dumps(website_search_results))}
    ]
    return query_llm(messages=messages, token_usage=token_tracking_dict)

def get_external_resource(query, top_k=3):
    website_texts = []
    # Search with microsoft bing
    website_urls = query_microsoft_bing(query=query, top_k=top_k)
    # Query Jina
    for website_url in website_urls:
        website_text = query_jina(website_url)
        website_texts.append({"website_link": website_url, "website_content": website_text})
    return website_texts


def generate_answer_with_other_models(output_folder):
    # Make output dir if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Only complete difference
    input_folder = "data/diverse_sampled_paper_dataset/ms_qa_3000_4o"
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


def check_is_standalone(dir):
    """Check if the provided answer is standalone with GPT."""
    file_paths = [os.path.join(dir, f_name) for f_name in os.listdir(dir)]
        
    user_prompt_template = """I will provide a materials science research question and its corresponding answer. Your task is to determine whether the answer is standalone, meaning it does not rely on external context.

Output "True and reason" if the answer is self-contained and does not reference external context.
Output "False and reason" if the answer assumes prior knowledge, relies on unstated information, or refers to external context.

Question: {question}
Answer: {answer}"""    

    res = []
    
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as stream:
            content = json.load(stream)
        
        user_prompt = user_prompt_template.format(question=content["question"], answer=content["answer"])
        
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_prompt},
        ]
        
        response = query_llm(messages)
        
        content["is_answer_standalone"] = response
        
        res.append(content)
    
    with open("test.json", "w+") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    # Generate question and answer
    # input_folder = "data/diverse_sampled_paper_dataset/diverse_papers_3000"  
    # output_folder = "data/diverse_sampled_paper_dataset/ms_qa_3000_4o"
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # process_papers(input_folder, output_folder, num_files=3000)

    # Generate Deepseek
    # deepseek_output_folder = "data/diverse_sampled_paper_dataset/ms_qa_3000_deepseek_v2"
    # generate_answer_with_other_models(deepseek_output_folder)
    
    # Generate Gemini
    gemini_output_folder = "data/diverse_sampled_paper_dataset/ms_qa_3000_gemini_v2"
    generate_answer_with_other_models(gemini_output_folder)

    # Generate answer only
    # output_folder = "ms_pipeline2_qa_v2_4o_test"
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # generate_answer_with_other_models(output_folder=output_folder)

    # print("Total tokens:", token_tracking_dict)



    # merge_files(output_folder, "pipline2_output_merge.json")

    # with open("./L2_questions.json", 'r') as stream:
    #     question_json = json.load(stream)
    # # generate_multi_step_qa(question_json=question_json, output_file="./cot_improved_multi_step_qa_v2.json", method='cot')
    # generate_coherent_multi_step_qa(question_json=question_json[:3], output_file="./coherent_multi_step_qa.json")
    
    # Check is standalone
    # dir = "ms_pipeline2_qa_30_test"
    # check_is_standalone(dir)
