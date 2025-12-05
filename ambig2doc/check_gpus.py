#!/usr/bin/env python3
"""
Script to check GPU availability and provide recommendations for running the model.
"""

import torch
import subprocess
import sys


def check_gpu_status():
    """Check GPU availability and current usage."""
    print("=" * 70)
    print("GPU Status Check")
    print("=" * 70)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. PyTorch cannot detect any GPUs.")
        return False

    # Get number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"✓ CUDA available: {num_gpus} GPU(s) detected by PyTorch\n")

    # Check each GPU
    for i in range(num_gpus):
        try:
            device = torch.device(f"cuda:{i}")
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")

            # Try to allocate a small tensor to check if GPU is accessible
            try:
                test_tensor = torch.zeros(1).to(device)
                del test_tensor
                torch.cuda.empty_cache()
                print(f"  - Status: ✓ Accessible")
            except Exception as e:
                print(f"  - Status: ❌ Not accessible - {str(e)}")
        except Exception as e:
            print(f"GPU {i}: ❌ Error - {str(e)}")
        print()

    # Run nvidia-smi if available
    print("-" * 70)
    print("nvidia-smi output:")
    print("-" * 70)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                parts = line.split(", ")
                if len(parts) >= 5:
                    idx, name, util, mem_used, mem_total = parts[:5]
                    print(f"GPU {idx}: {name}")
                    print(f"  - Utilization: {util}%")
                    print(
                        f"  - Memory: {mem_used} MB / {mem_total} MB ({float(mem_used)/float(mem_total)*100:.1f}%)"
                    )
                    print()
        else:
            print("nvidia-smi command failed")
    except FileNotFoundError:
        print("nvidia-smi not found")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")

    print("=" * 70)
    print("Recommendations:")
    print("=" * 70)

    if num_gpus >= 2:
        print(f"✓ You have {num_gpus} GPUs available.")
        print(f"\nTo use specific GPUs, set CUDA_VISIBLE_DEVICES:")
        print(f"  export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1")
        print(f"  export CUDA_VISIBLE_DEVICES=2,3  # Use GPUs 2 and 3")
        print(f"\nTo change tensor parallel size:")
        print(f"  export TENSOR_PARALLEL_SIZE=2  # Use 2 GPUs")
        print(f"  export TENSOR_PARALLEL_SIZE=4  # Use 4 GPUs")
    elif num_gpus == 1:
        print(f"⚠ Only 1 GPU available. The 70B FP8 model may not fit on a single GPU.")
        print(f"  Consider using a smaller model or getting access to more GPUs.")

    print("\nIf you see 'CUDA-capable device(s) is/are busy or unavailable':")
    print("  1. Check if another process is using the GPUs (see nvidia-smi above)")
    print("  2. Use CUDA_VISIBLE_DEVICES to select free GPUs")
    print("  3. Kill or wait for other processes to finish")
    print("  4. Check if you have proper permissions to access the GPUs")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = check_gpu_status()
    sys.exit(0 if success else 1)
