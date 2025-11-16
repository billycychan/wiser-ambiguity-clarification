from transformers import pipeline, AutoTokenizer
import os
from tqdm import tqdm
from datetime import datetime

from prompt import get_user_prompt, SYSTEM_PROMPT
from topics import topics


def build_prompt(tokenizer: AutoTokenizer, topic: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_user_prompt(topic)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_pipeline_and_tokenizer(model: str):
    pipe = pipeline("text-generation", model=model, trust_remote_code=True)
    tok = pipe.tokenizer
    return pipe, tok


def main():
    model = "meta-llama/Llama-3.1-8B-Instruct"
    pipe, tok = get_pipeline_and_tokenizer(model)

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/llama31_8b_instruct_{timestamp}.log"

    print(f"Logging responses to: {log_file}")

    with open(log_file, "w", encoding="utf-8") as log:
        for topic in tqdm(topics, desc="Generating queries"):
            # Build prompt
            prompt = build_prompt(tok, topic)

            # Generate response
            outputs = pipe(
                prompt,
                max_new_tokens=10000,
                do_sample=True,
                temperature=1.0,
            )

            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            response = generated_text[len(prompt) :]

            log.write(f"{response}\n")
            log.flush()  # Ensure it's written immediately

    print(f"Completed! Log saved to: {log_file}")


if __name__ == "__main__":
    main()
