from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
import torch
import importlib
import sys

import os
from tqdm import tqdm
from datetime import datetime

from prompt import get_user_prompt, SYSTEM_PROMPT
from topics import topics

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)


def build_prompt(tokenizer: AutoTokenizer, topic: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_user_prompt(topic)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _is_accelerate_process():
    # If running under accelerate/torch.distributed, LOCAL_RANK is set
    return os.environ.get("LOCAL_RANK") is not None


def get_pipeline_and_tokenizer(model: str, device_map="auto"):
    """
    Loads a text-generation pipeline and tokenizer with options enabling
    multi-GPU device mapping and quantization to reduce VRAM usage.

    device_map: str|dict - if 'auto', the HF loader will map across all visible GPUs.
    When running with `accelerate launch`, avoid 'auto' and let accelerate handle mapping.
    """

    # Attempt to reduce fragmentation and increase chance to fit
    # Modern PyTorch prefers PYTORCH_ALLOC_CONF; set both for compatibility
    if os.environ.get("PYTORCH_ALLOC_CONF") is None:
        os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Clear any leftover allocations before loading the model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Use small memory map when there are many GPUs; user can override via env
    pipe = None
    try:
        kwargs = dict(
            model=model,
            trust_remote_code=True,
            device_map=device_map,
            dtype=torch.bfloat16,
            # model_kwargs supports the low-level config like quantization
            model_kwargs={"quantization_config": quantization_config},
        )
        # If accelerate ran this process (LOCAL_RANK), don't try to auto map
        if _is_accelerate_process():
            # Each process is assigned a single local rank GPU. Set model to that device
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            kwargs["device_map"] = {"": local_rank}

        pipe = pipeline("text-generation", **kwargs)
        tok = pipe.tokenizer
        # Print device mapping
        try:
            device_map_print = getattr(pipe.model, "hf_device_map", None) or getattr(
                pipe.model, "device_map", None
            )
            print("Model device_map:", device_map_print)
        except Exception:
            pass

    except Exception as e:
        # If OOM or other failure occurs, offer helpful fallback suggestions
        print(f"Failed to create pipeline with device_map={device_map}: {e}")
        print("Trying a fallback: single-GPU with quantization and/or offload to CPU.")
        try:
            # Try smaller dtype and offload
            kwargs = dict(
                model=model,
                trust_remote_code=True,
                device_map="auto",
                dtype=torch.float16,
                model_kwargs={
                    "quantization_config": quantization_config,
                    "offload_folder": "./offload",
                    "offload_state_dict": True,
                },
            )
            pipe = pipeline("text-generation", **kwargs)
            tok = pipe.tokenizer
            print("Fallback pipeline created (auto, float16 + offload).")
            try:
                print("Model device_map:", pipe.model.hf_device_map)
            except Exception:
                pass
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print(
                "Please ensure sufficient free GPU memory or add more devices and use CUDA_VISIBLE_DEVICES=0,1,.. or use 'accelerate launch' with multiple processes."
            )
            raise

    return pipe, tok


def main():
    model = "meta-llama/Llama-3.3-70B-Instruct"
    # Print GPU/status diagnostics and recommended run command
    try:
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPUs")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(
                f"GPU {i}: {props.name} - total memory: {props.total_memory / (1024**3):.2f} GiB"
            )
    except Exception:
        print("CUDA not available or GPU introspection failed; proceeding.")

    if os.environ.get("LOCAL_RANK") is not None:
        print("WARNING: This process was launched by accelerate (LOCAL_RANK set).")
        print(
            "If you intentionally used accelerate, ensure you want each process to load model or use accelerate dispatch/zero to shard model across processes."
        )
        print(
            "Recommended single-process multi-GPU run: `CUDA_VISIBLE_DEVICES=0,1 python sythgen/llama33_70b_instruct.py` to let Transformers auto device_map across GPUs in one process."
        )
        print(
            "Recommended accelerate usage (advanced): `accelerate launch --config_file <config> sythgen/llama33_70b_instruct.py` with `offload_folder` and `cpu_offload` configured.)"
        )
    pipe, tok = get_pipeline_and_tokenizer(model)

    # Create logs directory (and any parent directories for the chosen model path)
    # The model string can include a namespace (eg. 'meta-llama/Llama-3.3-70B-Instruct'),
    # so ensure the entire directory path exists before creating the log file.
    os.makedirs("logs", exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{model}_{timestamp}.log"
    # Make sure parent path exists (handles 'logs/meta-llama' when model='meta-llama/...')
    log_parent = os.path.dirname(log_file)
    if log_parent:
        os.makedirs(log_parent, exist_ok=True)

    print(f"Logging responses to: {log_file}")

    with open(log_file, "w", encoding="utf-8") as log:
        for topic in tqdm(topics, desc="Generating queries"):
            # Build prompt
            prompt = build_prompt(tok, topic)

            # Generate response
            # Keep `model_kwargs` empty — generate args should be passed at top-level.
            try:
                outputs = pipe(
                    prompt,
                    use_cache=True,
                    max_new_tokens=2000,  # set lower if possible for speed
                    do_sample=False,
                    batch_size=4,  # use >1 if VRAM allows
                )
            except ValueError as e:
                # Print diagnostics for invalid generate kwargs
                print("Generation ValueError:", e)
                raise

            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            response = generated_text[len(prompt) :]

            log.write(f"{response}\n")
            log.flush()  # Ensure it's written immediately

    print(f"Completed! Log saved to: {log_file}")


if __name__ == "__main__":
    main()
