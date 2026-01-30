#!/usr/bin/env python3
"""
Inspect what was actually quantized in the moondream3 model.
"""

import os
import shutil

from transformers import AutoModelForCausalLM


def copy_model_files_to_cache(model_path):
    """Copy Python files from model directory to transformers cache."""
    # Create a simple hash from the path for cache directory name
    import hashlib
    path_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]

    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    target_dir = os.path.join(cache_dir, path_hash)

    if os.path.exists(target_dir):
        return  # Already exists

    os.makedirs(target_dir, exist_ok=True)

    # Copy all Python files
    py_files = [f for f in os.listdir(model_path) if f.endswith('.py')]
    for py_file in py_files:
        src = os.path.join(model_path, py_file)
        dst = os.path.join(target_dir, py_file)
        if os.path.islink(src):
            shutil.copy2(os.path.realpath(src), dst)
        else:
            shutil.copy2(src, dst)


def analyze_model_parameters(model, model_name="Model"):
    """Analyze and categorize model parameters."""

    total_params = 0
    quantized_params = 0
    excluded_params = 0

    quantized_layers = []
    excluded_layers = []
    normal_layers = []

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count

        # Check if this is a quantized parameter
        parent_module_name = '.'.join(name.split('.')[:-1])
        try:
            parent_module = model.get_submodule(parent_module_name)
            is_quantized = hasattr(parent_module, 'sdnq_dequantizer')
        except Exception:
            is_quantized = False

        if is_quantized:
            quantized_params += param_count
            quantized_layers.append((name, param.shape, param.dtype, param_count))
        elif name.endswith('.weight') and param.ndim == 2:
            # This is a linear layer that wasn't quantized
            excluded_params += param_count
            excluded_layers.append((name, param.shape, param.dtype, param_count))
        else:
            normal_layers.append((name, param.shape, param.dtype, param_count))

    print(f"\n{'='*80}")
    print(f"{model_name} PARAMETER ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Quantized parameters: {quantized_params:,} ({quantized_params/total_params*100:.1f}%)")
    print(f"Excluded linear layers: {excluded_params:,} ({excluded_params/total_params*100:.1f}%)")
    print(f"Other parameters: {(total_params-quantized_params-excluded_params):,} ({(total_params-quantized_params-excluded_params)/total_params*100:.1f}%)")

    if quantized_layers:
        print(f"\n{'='*80}")
        print("QUANTIZED LAYERS (sample - first 20):")
        print(f"{'='*80}")
        for name, shape, dtype, count in quantized_layers[:20]:
            print(f"  ✓ {name:70s} {shape!s:30s} {dtype} ({count:,} params)")
        if len(quantized_layers) > 20:
            print(f"  ... and {len(quantized_layers) - 20} more quantized layers")
    else:
        print(f"\n{'='*80}")
        print("⚠ WARNING: NO LAYERS WERE QUANTIZED!")
        print(f"{'='*80}")

    if excluded_layers:
        print(f"\n{'='*80}")
        print("EXCLUDED LINEAR LAYERS (sample - first 20):")
        print(f"{'='*80}")
        for name, shape, dtype, count in excluded_layers[:20]:
            print(f"  ✗ {name:70s} {shape!s:30s} {dtype} ({count:,} params)")
        if len(excluded_layers) > 20:
            print(f"  ... and {len(excluded_layers) - 20} more excluded layers")

    return total_params, quantized_params, excluded_params


def get_directory_size(path):
    """Get total size of a directory in bytes."""
    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                # Resolve symlinks
                if os.path.islink(filepath):
                    filepath = os.path.realpath(filepath)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    return total_size


def format_size(size_bytes):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def main():
    original_path = os.path.expanduser("~/database/models/huggingface/models--moondream--moondream3-preview/snapshots/e86382f00368618bfbbef8026cb606e9c0e3cd0e")
    quantized_path = os.path.expanduser("~/database/models/moondream3-preview-int8-sdnq")

    # Check directory sizes
    print(f"\n{'='*80}")
    print("DIRECTORY SIZE COMPARISON")
    print(f"{'='*80}")

    if os.path.exists(original_path):
        orig_size = get_directory_size(original_path)
        print(f"\nOriginal model:  {format_size(orig_size):>12s} ({orig_size:,} bytes)")

    if os.path.exists(quantized_path):
        quant_size = get_directory_size(quantized_path)
        print(f"Quantized model: {format_size(quant_size):>12s} ({quant_size:,} bytes)")

        if os.path.exists(original_path):
            ratio = quant_size / orig_size
            reduction = (1 - ratio) * 100
            print(f"\nSize ratio: {ratio:.2f}x")
            print(f"Size reduction: {reduction:.1f}%")

    # Load and analyze quantized model
    if os.path.exists(quantized_path):
        print(f"\n{'='*80}")
        print("Loading quantized model for analysis...")
        print(f"{'='*80}")

        try:
            # Copy model files to cache
            copy_model_files_to_cache(quantized_path)

            model = AutoModelForCausalLM.from_pretrained(
                quantized_path,
                trust_remote_code=True,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )

            analyze_model_parameters(model, "Quantized Model")

        except Exception as e:
            print(f"Error loading quantized model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nQuantized model not found at: {quantized_path}")


if __name__ == "__main__":
    main()
