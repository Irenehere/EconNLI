import openai
import backoff
import time
import torch

# we use the engine provided by our university
# you may need to modify the prompting function if you use the openai API
# openai.api_key = ''
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""


def prompt_chatgpt(prompt):
    try:
        completion = openai.ChatCompletion.create(
            engine = "gpt-35-turbo",
            messages=[
                {"role": "user", "content":prompt}
            ])
    except Exception as e:
        print(str(e))
        time.sleep(20)
        completion = openai.ChatCompletion.create(
            engine = "gpt-35-turbo",
            messages=[
                {"role": "user", "content":prompt}
            ])
    return completion["choices"][0]["message"]["content"]


def prompt_gpt4(prompt):
    try:
        completion = openai.ChatCompletion.create(
            engine = "gpt-4",
            messages=[
                {"role": "user", "content":prompt}
            ]
            )
    except Exception as e:
        print(str(e))
        time.sleep(20)
        completion = openai.ChatCompletion.create(
            engine = "gpt-4",
            messages=[
                {"role": "user", "content":prompt}
            ]
            )
    return completion["choices"][0]["message"]["content"]


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def prompt_chatgpt_with_backoff(prompt):
    return prompt_chatgpt(prompt)


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def prompt_gpt4_with_backoff(prompt):
    return prompt_gpt4(prompt)


def prompt_llama_like_model(prompt,model,tokenizer,max_new_tokens=100):
    inputs =  tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=max_new_tokens)
        outputs_string = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return outputs_string