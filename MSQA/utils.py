import os
from openai import AsyncAzureOpenAI, AzureOpenAI, OpenAI
from google import genai
from google.genai import types
import requests
import json
import statistics
import time
import anthropic


bad_keywords = {"bad_words_question": ["this", "article", "paper", "study", "abstract", "experimen",
                                        "studied", "investigat", "discuss", "describ", "example", "document", "section", "the finding","the sample"],
                "bad_words_answer": ["article", "paper", "abstract", "document", "section", "paragraph", 
                                        "investigat", "the finding", "the stud", "the sample", "sorry", "no valid", "not provide", "not valid"]
            }

MODEL_CONFIG = os.getenv("MODEL_CONFIG")
print("Current MODEL_CONFIG: ", MODEL_CONFIG )

if MODEL_CONFIG == "DEEPSEEK":
    client = OpenAI(
        api_key=os.getenv(""), 
        base_url="https://api.deepseek.com"
    )
elif MODEL_CONFIG in ["4O", "4O-MINI"]:
    client = AzureOpenAI(
        azure_endpoint=os.getenv(""),
        api_key=os.getenv(""),
        api_version=os.getenv(""),
        timeout=60
    )
elif MODEL_CONFIG == "GEMINI":
    client = genai.Client(api_key=os.getenv(""))
elif MODEL_CONFIG == "CLAUDE":
    client = anthropic.Anthropic()
elif MODEL_CONFIG == "GROK":
    client = OpenAI(
        api_key=os.getenv(""),
        base_url="https://api.x.ai/v1"
    )




def query_llm(messages, n=1, temperature=0.85, top_p=0.8, is_output_json: bool=False, token_usage={}):
    try:
        # Constructing parameters to query GPT
        if MODEL_CONFIG in ["4O", "4O-MINI"]:
            model = os.getenv("") if MODEL_CONFIG == "4O" else os.getenv("")
            chat_completion_paramters = {
                "model": model,
                "messages": messages,
                "n": n, # Number of answer choices
                "temperature": temperature,
                "top_p": top_p,
                }
            if is_output_json:
                chat_completion_paramters["response_format"] = {"type": "json_object"}
        elif MODEL_CONFIG in ["DEEPSEEK", "GROK"]:
            cur_model = "deepseek-chat" if MODEL_CONFIG == "DEEPSEEK" else "grok-3-beta"
            chat_completion_paramters = {
                "model": cur_model,
                "messages": messages,
                "n": n, # Number of answer choices
                "temperature": temperature,
                "top_p": top_p,
                }
          
        elif MODEL_CONFIG == "GEMINI":
            chat_completion_paramters = {
                "model": "gemini-2.0-flash",
                "config": types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    # system_instruction=messages[0]["content"] # System Prompt. If you want to use this parameter, please set the system prompt not empty str.
                ),
                "contents": [messages[0]["content"]] # User Prompt
            }
        elif MODEL_CONFIG == "CLAUDE":

            chat_completion_paramters = {
                "model": "claude-3-7-sonnet-20250219",
                "temperature": temperature,
                "top_p": top_p,
                # "system": messages[0]["content"],
                "messages": [{"role": "user","content": [{"type": "text","text": messages[0]["content"]}]
                            }],
                "max_tokens": 1024,
            }


        # Query GPT with constructed parameters
        if MODEL_CONFIG in ["4O", "4O-MINI", "DEEPSEEK","GROK"]:
            completion = client.chat.completions.create(**chat_completion_paramters)
        elif MODEL_CONFIG == "GEMINI":
            completion = client.models.generate_content(**chat_completion_paramters)
        elif MODEL_CONFIG == "CLAUDE":
            completion = client.messages.create(**chat_completion_paramters)
           
          
        # Tracking token usage
        if MODEL_CONFIG in ["4O", "4O-MINI", "DEEPSEEK"]:
            token_usage["input"] = token_usage.get("input", 0) + completion.usage.prompt_tokens
            token_usage["output"] = token_usage.get("output", 0) + completion.usage.completion_tokens
        elif MODEL_CONFIG == "GEMINI":
            token_usage["input"] = token_usage.get("input", 0) + completion.usage_metadata.prompt_token_count
            token_usage["output"] = token_usage.get("output", 0) + completion.usage_metadata.candidates_token_count
        elif MODEL_CONFIG == "CLAUDE":
            token_usage["input"] = token_usage.get("input", 0) + completion.usage.input_tokens
            token_usage["output"] = token_usage.get("output", 0) + completion.usage.output_tokens
        # Output Responses
        responses = []
        if MODEL_CONFIG in ["4O", "4O-MINI", "DEEPSEEK","GROK"]:
            for choice in completion.choices:
                if is_output_json and MODEL_CONFIG in ["4O", "4O-MINI"]:
                    responses.append(json.loads(choice.message.content))
                else:
                    responses.append(choice.message.content)
        elif MODEL_CONFIG == "GEMINI":
            responses.append(completion.text)
        elif MODEL_CONFIG == "CLAUDE":
            responses.append(completion.content[0].text)
        
        return responses if n > 1 else responses[0]
    
    except Exception as e:
        print(f"Error generating Response: {str(e)}")
        return None

async_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv(""),
    api_key=os.getenv(""),
    api_version=os.getenv(""),
    timeout=60
)

# semaphore is used for rate limit in async calls, see https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore
async def async_query_gpt(messages, semaphore, n=1, temperature=0.85, top_p=0.8, is_output_json: bool=False, token_usage={}):
    async with semaphore:
        try:
            # Constructing parameters to query GPT
            model = os.getenv("GPT4O") if MODEL_CONFIG == "4O" else os.getenv("GPT4O_MINI")
            chat_completion_paramters = {
                "model": model,
                "messages": messages,
                "n": n, # Number of answer choices
                "temperature": temperature,
                "top_p": top_p,
                }
            if is_output_json:
                chat_completion_paramters["response_format"] = {"type": "json_object"}
            # Query GPT with constructed parameters
            completion = await async_client.chat.completions.create(**chat_completion_paramters)
            # Tracking token usage
            token_usage["input"] = token_usage.get("input", 0) + completion.usage.prompt_tokens
            token_usage["output"] = token_usage.get("output", 0) + completion.usage.completion_tokens
                
            # Output Responses
            responses = []
            for choice in completion.choices:
                if is_output_json:
                    responses.append(json.loads(choice.message.content))
                else:
                    responses.append(choice.message.content)
            return responses if n > 1 else responses[0]
        
        except Exception as e:
            print(f"Error generating Response: {str(e)}")
            return None

    
def query_microsoft_bing(query, top_k=3):
    url = 'https://api.bing.microsoft.com/v7.0/search'
    params = { 'q': query, 'mkt': 'en-US' }
    headers = { 'Ocp-Apim-Subscription-Key': os.getenv("MICROSOFT_API_KEY") }
    response = json.loads(requests.get(url, headers=headers, params=params).text)
    
    website_urls = [search_result["url"] for search_result in response["webPages"]["value"]]
    website_urls = website_urls[:min(len(website_urls), top_k)]
    return website_urls

def query_jina(query_url):
    url = 'https://r.jina.ai/' + query_url
    headers = {'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}'}
    response = requests.get(url, headers=headers).text
    return response

def merge_files(input_folder, output_file):
    all_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)

    # Write merged data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(all_data, out, indent=4)

    print(f"{output_file} -- total len: {len(all_data)}")

def gpt_judger(question, gold_answer, raw_answer, mode="default", token_tracking_dict={}):
    if mode == "default":
#         system_prompt = """You are an expert evaluator in materials science. Your task is to assess student responses to materials science questions.

# For each question, you will be provided with:
#     - A gold answer that represents an accurate and complete response.
#     - A student's answer that needs evaluation.

# Evaluation Criteria:
#     - correct: The student's answer fully aligns with the correct answer, capturing key concepts and necessary details.
#     - mostly correct: The student's answer reaches the correct conclusion or aligns well with the overall direction of the correct answer. Minor errors, missing details, or extra context should not prevent a 'mostly correct' judgment if the core idea is sound.
#     - incorrect: The student's answer fundamentally misunderstands or contradicts key concepts in the correct answer.

# Your response should follow this structured JSON format:        
# {{
#     "reasoning": "A concise explanation supporting your judgment.",
#     "judgment": "correct|mostly correct|incorrect"
# }}"""

        system_prompt = """Your task is to evaluate the accuracy of LLM-generated answers to materials science questions by comparing them to expert-validated "gold" answers.

For each evaluation, you will receive:
	- A materials science question
	- A gold answer, based on authoritative domain knowledge
	- An LLM-generated inference answer, which you must assess

Your goal is to evaluate how well the inference answer aligns with the gold answer in terms of factual accuracy, conceptual completeness, and relevance.

Use the following evaluation rubric:
	- Correct: The inference answer fully captures all essential concepts from the gold answer, with no significant omissions or factual errors.
	- Mostly Correct: The inference answer conveys the main idea or correct conclusion, even if minor details are missing or slight inaccuracies are present. Additional non-conflicting information is acceptable.
	- Incorrect: The inference answer demonstrates substantial misunderstanding, includes major factual errors, or omits core concepts present in the gold answer.

Provide a short justification for your rating, highlighting key similarities or discrepancies between the inference and gold answers. Output your response in the following JSON format:
{{
    "reasoning": "A concise explanation supporting your judgment.",
    "judgment": "correct|mostly correct|incorrect"
}}"""
        
        user_prompt = f"""**Input Data**:
        - Material Science Question: {question}
        - Gold Answer: {gold_answer}
        - Student Answer: {raw_answer}"""

    if mode == "default0":

        system_prompt = """You are an expert evaluator in materials science. Your task is to critically assess student responses to materials science questions.

For each question, you will be provided with:
    - A correct answer that represents an accurate and complete response.
    - A student's answer that needs evaluation.
   
Your evaluation should focus strictly on technical correctness, avoiding unnecessary elaboration.

Evaluation Criteria:
    - correct: The student's answer fully aligns with the correct answer, including all key details.
    - mostly correct: The student's answer captures the main concepts. The answer can lack certain details or be some what general.
    - incorrect: The student's answer either contradicts the correct answer or lacks any concepts necessary for correctness.

Your response should follow this structured JSON format:        
{{
    "reasoning": "An explanation supporting your judgment. Keep your reasoning concise and to the point."
    "judgment": "correct|mostly correct|incorrect",
}}"""

        user_prompt = f"""**Input Data**:
        - Material Science Question: {question}
        - Correct Answer: {gold_answer}
        - Student Answer: {raw_answer}"""

    elif mode == "score":

        system_prompt = """"
        You are an expert material science evaluator and grader. Your role is to assess student answers against a correct reference answer, assigning a fractional score based on the number of key points correctly addressed by the student.
        
        When evaluating:
        - Identify key bullet points in the correct answer.
        - Match the student’s answer against these bullet points to determine which points are correctly addressed, partially addressed, or missed.
        - Calculate a fractional score as (number of points matched) / (total key points in the correct answer).
        - Provide a detailed reasoning section that:
        - - Lists the matched points and explains why they are correct.
        - - Highlights the missing or incorrect points and explains the errors or omissions.
        
        Maintain a constructive tone to aid in student learning.
        Do not summarize the question or the correct answer unless asked. Focus only on the score and detailed reasoning.
        """

        user_prompt = f""""
        **Input Data**:
        - Material Science Question: {question}
        - Correct Answer: {gold_answer}
        - Student Answer: {raw_answer}

        **Task**:
        Evaluate the student's answer:
        
        - Identify key bullet points from the correct answer.
        - Match the student's answer against these points and calculate a fractional score.
        - Include a reasoning section:
        -- Highlight points that match with explanations.
        -- Highlight missing or incorrect points with details on the errors.

        **Output Format**:
        {{
            "reasoning": "Highlight points that match with explanations. And Highlight missing or incorrect points with details on the errors"
            "judgment": "a fractional score as (number of points matched) / (total key points in the correct answer)",
        }}
        """


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Generate questions and answers
    json_response = query_llm(messages, is_output_json=True, token_usage=token_tracking_dict)
    return json_response

async def async_gpt_judger(question, gold_answer, raw_answer, semaphore, token_tracking_dict={}):
    system_prompt = """Your task is to evaluate the accuracy of LLM-generated answers to materials science questions by comparing them to expert-validated "gold" answers.

For each evaluation, you will receive:
	- A materials science question
	- A gold answer, based on authoritative domain knowledge
	- An LLM-generated inference answer, which you must assess

Your goal is to evaluate how well the inference answer aligns with the gold answer in terms of factual accuracy, conceptual completeness, and relevance.

Use the following evaluation rubric:
	- Correct: The inference answer fully captures all essential concepts from the gold answer, with no significant omissions or factual errors.
	- Mostly Correct: The inference answer conveys the main idea or correct conclusion, even if minor details are missing or slight inaccuracies are present. Additional non-conflicting information is acceptable.
	- Incorrect: The inference answer demonstrates substantial misunderstanding, includes major factual errors, or omits core concepts present in the gold answer.

Provide a short justification for your rating, highlighting key similarities or discrepancies between the inference and gold answers. Output your response in the following JSON format:
{{
    "reasoning": "A concise explanation supporting your judgment.",
    "judgment": "correct|mostly correct|incorrect"
}}"""

    user_prompt = f"""**Input Data**:
    - Material Science Question: {question}
    - Correct Answer: {gold_answer}
    - Student Answer: {raw_answer}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Generate questions and answers with retry if request failed
    for _ in range(3):
        json_response = await async_query_gpt(messages, semaphore, is_output_json=True, token_usage=token_tracking_dict)
        
        if json_response:
            print("Finished evaluation for:", question)
            return json_response

        print("Sleeping 30 seconds before continuing for question:", question)
        time.sleep(30)

    return json_response



def split_level(input_json_file, L_a_json_file, L_b_json_file, judge_score=-1, level="L0/L1"):
    """
    By default score=0, which represents spliting l0,l1 by "incorrect" Judgment
    If score > 0, (eg 0.8) then splits l0,l1 by score.
    """
    with open(input_json_file, "r", encoding="utf-8") as f:
        qa_jsons = json.load(f)
    La_list = []
    Lb_list = []
    if judge_score == -1: # split mode = default
        for qa in qa_jsons:
            if level=="L1/L2":
                actions = [step["action"] for step in qa["multi_step_answer"]]
                web_search_count = actions.count("Web Search")
                if web_search_count >= 2:
                    Lb_list.append(qa)
                    continue
            if qa["judgment"] == "incorrect":
                Lb_list.append(qa)
            else:
                La_list.append(qa)
    else: # split mode = score
        all_scores = []
        for qa in qa_jsons:
            if level=="L1/L2":
                actions = [step["action"] for step in qa["multi_step_answer"]]
                web_search_count = actions.count("Web Search")
                if web_search_count >= 2:
                    Lb_list.append(qa)
                    continue
            try:
                if '/' in qa["judgment"]:
                    num, denom = qa["judgment"].split('/', 1)
                cur_score =  float(num) / float(denom)
                all_scores.append(cur_score)
                if cur_score <= judge_score:
                    Lb_list.append(qa)
                else:
                    La_list.append(qa)
                
            except Exception as e:
                print(f"Error processing score: {e}")
                continue

        def analyze_floats(data):
            sorted_data = sorted(data)
            # Compute quartiles (n=4 means we split data into 4 equal parts; 
            quartiles = statistics.quantiles(sorted_data, n=4, method='inclusive')
            return {
                "min":   sorted_data[0],
                "Q1":    quartiles[0],
                "median": quartiles[1],
                "Q3":    quartiles[2],
                "max":   sorted_data[-1],
                "mean":  statistics.mean(data),
            }
        
        print(analyze_floats(all_scores))
    
    with open(L_a_json_file, "w",encoding='utf-8') as stream:
        json.dump(La_list, stream)
    with open(L_b_json_file, "w",encoding='utf-8') as stream:
        json.dump(Lb_list, stream)

    print(f"L_a list len : {len(La_list)}")
    print(f"L_b list len : {len(Lb_list)}")

if __name__ == "__main__":
    pass
    # query_microsoft_bing(query="test")
    # query_jina(query_url="https://www.speedtest.net/")
    # merge_files("ms_pipeline2_qa_5000_4o_mini", "merge_5000_qa.json")
    
    # print(gpt_judger(question="What is sun", gold_answer="Sun is round", raw_answer="Sun is Sun"))
    print(query_llm(messages=[{"role": "user", "content": "What is the capital of France?"}]))
