from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.3-70B-Instruct",
    trust_remote_code=True,
    device_map="auto",
    dtype=torch.bfloat16,
    model_kwargs={
        "quantization_config": quantization_config,
    },
    batch_size=32,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")


def format_prompt(query, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(query=query)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
