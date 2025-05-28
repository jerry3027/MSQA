import json

def mini_as_judge_analysis():
    with open("baselines/results/eval/4o-mini/Meta-Llama-3.1-8B-Instruct-Eval.json", "r") as stream:
        gpt_mini = json.load(stream)
        
    with open("baselines/results/eval/4o/Meta-Llama-3.1-8B-Instruct-Eval.json", "r") as stream:
        gpt = json.load(stream)
        
        
    gpt_incorrect_mini_correct = []
    gpt_correct_mini_incorrect = []
    for a, b in zip(gpt, gpt_mini):
        # Questions are aligned.
        if "judgment" not in a or "judgment" not in b:
            continue
        
        if a["judgment"] == "incorrect" and b["judgment"] in ["correct", "mostly correct"]:
            gpt_incorrect_mini_correct.append([a, b])
        if a["judgment"] in ["correct", "mostly correct"] and b["judgment"] == "incorrect":
            gpt_correct_mini_incorrect.append([a, b])


    with open("gpt_incorrect_mini_correct.json", "w") as stream:
        json.dump(gpt_incorrect_mini_correct, stream, indent=4)

    with open("gpt_correct_mini_incorrect.json", "w") as stream:
        json.dump(gpt_correct_mini_incorrect, stream, indent=4)

def gpt_4o_as_judge_analysis():
    with open("baselines/results/eval/4o/Meta-Llama-3.1-8B-Instruct-Eval.json", "r") as stream:
        gpt = json.load(stream)

    gpt_incorrect = []

    for item in gpt:
        # Questions are aligned.
        if "judgment" not in item:
            continue
        
        if item["judgment"] == "incorrect":
            gpt_incorrect.append(item)

    with open("gpt_incorrect.json", "w") as stream:
        json.dump(gpt_incorrect, stream, indent=4)


if __name__ == "__main__":
    gpt_4o_as_judge_analysis()