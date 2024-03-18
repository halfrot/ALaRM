import argparse
import glob
from collections import defaultdict
import json
import tiktoken
import utils
from pathlib import Path
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--generations_dir", required=True, type=str)
parser.add_argument("--max_compare", type=int, default=3000)
parser.add_argument("--use_gpt", action="store_true", default=False)
parser.add_argument("--use_cache", action="store_true", default=False)
parser.add_argument("--prompt_template", type=str, default="basic_function_prompt.txt")
parser.add_argument("--task_type", type=str, default="qa", choices=["qa", "mt"])
parser.add_argument("--api_key", required=True, type=str)
parser.add_argument("--api_call_program_path", type=str, default="api_request_parallel_processor.py")
args = parser.parse_args()

results = defaultdict(dict)
generations_paths = glob.glob(args.generations_dir + "/*.json")
if args.task_type == "qa":
    reward_types = ["holistic_reward", "factuality_rate", "verbosity_rate", "completeness_reward", "response_length"]
elif args.task_type == "mt":
    reward_types = ["holistic_reward", "readability", "language_confidence", "grammar_errors", "response_length",
                    "error_rate"]
save_dir = Path(args.generations_dir + "_results")
if os.path.exists(save_dir) and not args.use_cache:
    raise ValueError(f"Directory {save_dir} already exists")
os.makedirs(save_dir, exist_ok=True)

import random

random.seed(42)
with open(generations_paths[0]) as f:
    generation_list = json.load(f)
    if len(generation_list) > args.max_compare:
        unique_random_integers = random.sample(range(0, len(generation_list)), args.max_compare)


def extract_elements(generation_list):
    new_generation_list = []
    for i in unique_random_integers:
        new_generation_list.append(generation_list[i])
    return new_generation_list


for path in generations_paths:
    name_path = path.split(".json")[0].split("/")[-1]
    with open(path, "r") as f:
        generation_list = json.load(f)
        if len(generation_list) > args.max_compare:
            generation_list = extract_elements(generation_list)
        else:
            generation_list = generation_list[:args.max_compare]
    for reward_type in reward_types:
        print(f"---------------------------------{reward_type}----------------------------------")
        max_reward = -100
        min_reward = 100
        mean = 0
        for i, gen in enumerate(generation_list):
            max_reward = max(max_reward, generation_list[i][reward_type])
            min_reward = min(min_reward, generation_list[i][reward_type])
            mean += generation_list[i][reward_type]
        mean /= len(generation_list)
        results[name_path][f"{reward_type}-Mean"] = mean
        print(f"{name_path} - Mean = {round(mean, 2)}, Min = {round(min_reward, 2)}, Max = {round(max_reward, 2)}")
print("-------------------------------------------------------------------")
gpt_eval_results = defaultdict(dict)
if args.use_gpt:
    model_name = "gpt-3.5-turbo-1106"
    enc = tiktoken.encoding_for_model(model_name)
    gpt_tasks = []
    for a in generations_paths:
        name_a = a.split(".json")[0].split("/")[-1]
        for b in generations_paths:
            name_b = b.split(".json")[0].split("/")[-1]
            gpt_eval_results[name_a][name_b] = {
                "tie": 0,
                "lose": 0,
                "win": 0
            }
            if a == b:
                continue
            with open(a, "r") as f:
                generation_list_a = json.load(f)
                if len(generation_list_a) > args.max_compare:
                    generation_list_a = extract_elements(generation_list_a)
                else:
                    generation_list_a = generation_list_a[:args.max_compare]
            with open(b, "r") as f:
                generation_list_b = json.load(f)
                if len(generation_list_b) > args.max_compare:
                    generation_list_b = extract_elements(generation_list_b)
                else:
                    generation_list_b = generation_list_b[:args.max_compare]
            for i, gen in enumerate(generation_list_a):
                if generation_list_a[i]["generation"] == generation_list_b[i]["generation"]:
                    gpt_eval_results[name_a][name_b]["tie"] += 1
                else:
                    conversation = utils.make_conversation(generation_list_a[i]["query"],
                                                           generation_list_a[i]["generation"],
                                                           generation_list_b[i]["generation"],
                                                           args.prompt_template)
                    token_consume = len(enc.encode(conversation[0]["content"])) + len(
                        enc.encode(conversation[1]["content"])) + 25
                    single_task = {
                        "model": model_name,
                        "messages": conversation,
                        "max_tokens": token_consume,
                        "temperature": 0,
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "print_best_model",
                                    "description": "Print the best model given the preferred output.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "best_output": {
                                                "type": "string",
                                                "description": "Name of the best output, should be 'Output (a)' or "
                                                               "'Output (b)'"
                                            },
                                        },
                                        "required": ["best_output"]
                                    }
                                }
                            }
                        ],
                        "tool_choice": "auto",
                        "metadata": {
                            "name_a": name_a,
                            "name_b": name_b,
                            "id": i
                        }
                    }
                    gpt_tasks.append(single_task)
    tasks_path = save_dir / "api-tasks.jsonl"
    gpt_eval_results_save_path = save_dir / "api-tasks-results.jsonl"
    if not args.use_cache:
        with open(tasks_path, "w") as f:
            for task in gpt_tasks:
                json_string = json.dumps(task)
                f.write(json_string + "\n")
        parallel_api_request_cmd = f"python {args.api_call_program_path} \
              --requests_filepath {tasks_path} \
              --save_filepath {gpt_eval_results_save_path} \
              --api_key {args.api_key} \
              --request_url https://api.openai.com/v1/chat/completions \
              --max_requests_per_minute 2500 \
              --max_tokens_per_minute 2750000 \
              --token_encoding_name cl100k_base \
              --max_attempts 5 \
              --logging_level 20"
        os.system(parallel_api_request_cmd)
    with open(gpt_eval_results_save_path, "r", encoding="UTF-8") as f:
        for line in f:
            single_line = json.loads(line)
            metadata = single_line[2]
            single_response = single_line[1]
            if "tool_calls" not in single_response["choices"][0]["message"]:
                response_content = single_response["choices"][0]["message"]["content"]
            else:
                response_content = single_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            if bool(re.search(r"(?i)output \(a\)", response_content)):
                single_eval_result = 1
            elif bool(re.search(r"(?i)output \(b\)", response_content)):
                single_eval_result = 0
            gpt_eval_results[metadata["name_a"]][metadata["name_b"]][metadata["id"]] = single_eval_result

for a in generations_paths:
    name_a = a.split(".json")[0].split("/")[-1]
    Avg = 0
    for b in generations_paths:
        name_b = b.split(".json")[0].split("/")[-1]
        results[name_a][name_b] = {}
        with open(a, "r") as f:
            generation_list_a = json.load(f)
            if len(generation_list_a) > args.max_compare:
                generation_list_a = extract_elements(generation_list_a)
            else:
                generation_list_a = generation_list_a[:args.max_compare]
        with open(b, "r") as f:
            generation_list_b = json.load(f)
            if len(generation_list_b) > args.max_compare:
                generation_list_b = extract_elements(generation_list_b)
            else:
                generation_list_b = generation_list_b[:args.max_compare]
        if a == b:
            results[name_a][name_b] = "*"
            continue
        if args.use_gpt:
            win_a = 0
            tie = 0
            lose_a = 0
            for i, gen in enumerate(generation_list_a):
                if i in gpt_eval_results[name_a][name_b]:
                    if gpt_eval_results[name_a][name_b][i] != gpt_eval_results[name_b][name_a][i]:
                        win_a += gpt_eval_results[name_a][name_b][i]
                        lose_a += 1 - gpt_eval_results[name_a][name_b][i]
                    else:
                        tie += 1
                else:
                    tie += 1
            gpt_eval_results[name_a][name_b]["win"] = win_a
            gpt_eval_results[name_a][name_b]["lose"] = lose_a
            gpt_eval_results[name_a][name_b]["tie"] = tie
            if win_a + lose_a > 0:
                results[name_a][name_b]["gpt"] = 100 * win_a / (win_a + lose_a)
            else:
                results[name_a][name_b]["gpt"] = 50
        for reward_type in reward_types:
            results[name_a][name_b][reward_type] = {}
            win_a = 0
            tie = 0
            lose_a = 0
            holistic_reward_inconsistent = 0
            avg_gap = 0
            ai_tot = 0
            filtered_win = 0
            filtered_lose = 0
            for i, gen in enumerate(generation_list_a):
                # win, tie, or lose by *reward_type*
                if generation_list_a[i][reward_type] > generation_list_b[i][reward_type]:
                    win_a += 1
                    flag = 1
                elif generation_list_a[i][reward_type] == generation_list_b[i][reward_type]:
                    tie += 1
                    flag = -1
                else:
                    lose_a += 1
                    flag = 0
                if (reward_type == "error_rate" or reward_type == "grammar_errors") and flag != -1:
                    flag = 1 - flag

                # if is inconsistent with *holistic reward*
                if generation_list_a[i]["holistic_reward"] > generation_list_b[i]["holistic_reward"]:
                    ai_flag = 1
                    ai_tot += 1
                elif generation_list_a[i]["holistic_reward"] == generation_list_b[i]["holistic_reward"]:
                    ai_flag = -1
                else:
                    ai_flag = 0
                    ai_tot += 1
                if ai_flag != -1 and flag != -1 and ai_flag != flag:
                    holistic_reward_inconsistent += 1
            # calculate win rate
            if win_a + lose_a > 0:
                results[name_a][name_b][reward_type]["win_rate"] = 100 * win_a / (win_a + lose_a)
            else:
                results[name_a][name_b][reward_type]["win_rate"] = 50
            # calculate inconsistency
            if ai_tot > 0:
                results[name_a][name_b][reward_type][
                    "holistic_reward_inconsistent"] = 100 * holistic_reward_inconsistent / ai_tot
            else:
                results[name_a][name_b][reward_type]["holistic_reward_inconsistent"] = 0
    results[name_a]["Avg."] = {}
    for reward_type in reward_types:
        Avg = 0
        for b in generations_paths:
            name_b = b.split(".json")[0].split("/")[-1]
            if name_a == name_b:
                continue
            Avg += results[name_a][name_b][reward_type]["win_rate"]
        Avg /= len(generations_paths) - 1
        results[name_a]["Avg."][reward_type] = Avg

with open(save_dir / "eval-results.json", "w") as f:
    json.dump(results, f, indent=4)
