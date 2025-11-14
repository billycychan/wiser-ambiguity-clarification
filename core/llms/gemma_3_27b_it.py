from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

pipeline = pipeline(
    "text-generation",
    model="google/gemma-3-27b-it",
    trust_remote_code=True,
    device_map="auto",
    # model_kwargs={"quantization_config": quantization_config},
    batch_size=32,
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")


def format_prompt(query, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(query=query)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
