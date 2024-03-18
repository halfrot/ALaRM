import re


def extract_between_tags(text, start_tag="<|im_start|>", end_tag="\n<|im_end|>"):
    pattern = re.compile(rf'{re.escape(start_tag)}(.*?){re.escape(end_tag)}', re.DOTALL)
    return pattern.findall(text)


def make_conversation(query, output_1, output_2, prompt_template):
    with open(prompt_template, "r", encoding="UTF-8") as f:
        file_content = f.read()
        system_prompt = extract_between_tags(file_content, start_tag="<|im_start|>system\n")[0]
        user_prompt = extract_between_tags(file_content, start_tag="<|im_start|>user\n")[0].replace(
            "{instruction}", query).replace(
            "{output_1}", f"{output_1}").replace(
            "{output_2}", f"{output_2}")
    conversation = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        },
    ]
    return conversation
