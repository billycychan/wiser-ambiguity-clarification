from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

pipeline = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-128k-instruct",
    trust_remote_code=False,
    device_map="auto",
    # model_kwargs={"quantization_config": quantization_config},
    batch_size=32,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")


def format_prompt(query, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(query=query)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
