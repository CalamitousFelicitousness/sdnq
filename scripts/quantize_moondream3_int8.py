#!/usr/bin/env python3
"""
Quantize moondream3-preview model to INT8 using SDNQ.

This script quantizes the moondream3 model while excluding:
- Vision encoder
- Region encoder/decoder modules
- Layer normalization layers
- Language model head (lm_head)
- MoE router and expert layers
- Embeddings

Usage:
    python scripts/quantize_moondream3_int8.py

The quantized model will be saved to ~/database/models/moondream3-preview-int8-sdnq
"""

import argparse
import os
import shutil
import sys

import torch
from transformers import AutoModelForCausalLM

from sdnq import SDNQConfig, save_sdnq_model, sdnq_post_load_quant


def prepare_model_files(model_path):
    """
    Prepare model files for loading with trust_remote_code=True.
    Copies Python files from the model directory to transformers cache,
    resolving symlinks in the process.
    """
    # Get the snapshot hash from the path
    snapshot_hash = os.path.basename(model_path)

    # Transformers cache directory for custom modules
    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    target_dir = os.path.join(cache_dir, snapshot_hash)

    # If the cache already exists and has files, we're good
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        # Check if all Python files exist
        model_py_files = [f for f in os.listdir(model_path) if f.endswith('.py')]
        cache_py_files = [f for f in os.listdir(target_dir) if f.endswith('.py')]
        if set(model_py_files).issubset(set(cache_py_files)):
            return  # All files already in cache

    # Create cache directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Copy all Python files from model directory to cache, resolving symlinks
    py_files = [f for f in os.listdir(model_path) if f.endswith('.py')]

    print("Preparing model files in transformers cache...")
    for py_file in py_files:
        src_path = os.path.join(model_path, py_file)
        dst_path = os.path.join(target_dir, py_file)

        # Resolve symlink and copy the actual file
        if os.path.islink(src_path):
            real_path = os.path.realpath(src_path)
            shutil.copy2(real_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    print(f"Copied {len(py_files)} Python files to transformers cache")


def check_exclusions(model, exclusion_patterns):
    """Print information about excluded and included parameters."""
    excluded_params = []
    included_params = []

    for name, param in model.named_parameters():
        if not isinstance(param, torch.nn.Parameter):
            continue

        # Check if this parameter matches any exclusion pattern
        is_excluded = False
        for pattern in exclusion_patterns:
            if pattern.startswith("."):
                # Suffix or substring match
                if pattern[1:] in name:
                    is_excluded = True
                    break
            elif pattern in name:
                is_excluded = True
                break

        # Only consider Linear layer weights for quantization
        is_linear_weight = name.endswith('.weight') and param.ndim == 2

        if is_excluded:
            excluded_params.append((name, param.shape, param.dtype))
        elif is_linear_weight:
            included_params.append((name, param.shape, param.dtype))

    print(f"\n{'='*80}")
    print("QUANTIZATION PREVIEW")
    print(f"{'='*80}")
    print(f"\n✓ Parameters to be quantized: {len(included_params)}")
    print(f"✗ Parameters excluded: {len(excluded_params)}")

    if included_params:
        print("\nSample parameters to be quantized (showing first 10):")
        for name, shape, dtype in included_params[:10]:
            print(f"  ✓ {name:60s} {shape!s:30s} {dtype}")
        if len(included_params) > 10:
            print(f"  ... and {len(included_params) - 10} more")

    if excluded_params:
        print("\nSample excluded parameters (showing first 15):")
        for name, shape, dtype in excluded_params[:15]:
            print(f"  ✗ {name:60s} {shape!s:30s} {dtype}")
        if len(excluded_params) > 15:
            print(f"  ... and {len(excluded_params) - 15} more")

    print(f"\n{'='*80}\n")

    # Ask for confirmation
    response = input("Proceed with quantization? [y/N]: ").strip().lower()
    if response != 'y':
        print("Quantization cancelled.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize moondream3-preview model to INT8 using SDNQ"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="~/database/models/huggingface/models--moondream--moondream3-preview/snapshots/e86382f00368618bfbbef8026cb606e9c0e3cd0e/",
        help="Path to the moondream3 model",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="~/database/models/moondream3-preview-int8-sdnq",
        help="Path to save the quantized model",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip the preview and proceed directly to quantization",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for quantization (default: 128)",
    )

    args = parser.parse_args()

    # Model path - handle HuggingFace cache directory structure
    model_path = os.path.expanduser(args.model_path)

    # If path points to HF cache directory (models--org--name), find the latest snapshot
    if os.path.isdir(model_path):
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_dir) and not os.path.exists(os.path.join(model_path, "config.json")):
            # This is a HF cache directory, find the snapshot
            snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshot_dirs:
                # Use the first (and usually only) snapshot
                snapshot_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                print("Detected HuggingFace cache directory structure.")
                print(f"Using snapshot: {snapshot_dirs[0]}")
                model_path = snapshot_path
            else:
                print(f"Error: No snapshots found in {snapshots_dir}")
                sys.exit(1)

    # Verify model path has required files
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Error: config.json not found in {model_path}")
        print("Please provide a valid model directory.")
        sys.exit(1)

    # Output path for quantized model
    output_path = os.path.expanduser(args.output_path)

    # Define modules to exclude from quantization
    modules_to_not_convert = [
        # Vision encoder - entire vision module
        "model.vision",
        ".vision",

        # Region encoder/decoder modules
        "model.region",
        ".region",

        # Layer normalization layers
        ".ln",  # Per-block layer norms
        ".post_ln",  # Final layer norm before lm_head

        # Language model head
        ".lm_head",

        # MoE components (router only - expert weights are nn.Parameter, not nn.Linear, so won't be quantized anyway)
        ".mlp.router",  # MoE router

        # Embeddings
        ".wte",  # Word token embeddings

        # Additional components that should not be quantized
        ".freqs_cis",  # RoPE frequency embeddings
        ".tau",  # Attention tau parameters
        ".kv_cache",  # KV cache buffers

        # Region model parameters
        ".coord_features",
        ".size_features",
    ]

    # Configure SDNQ for INT8 quantization
    sdnq_config = SDNQConfig(
        weights_dtype="int8",
        group_size=args.group_size,  # Group size for quantization
        svd_rank=0,  # Disable SVD compression for now
        svd_steps=8,
        use_svd=False,
        quant_conv=False,  # Don't quantize conv layers
        use_quantized_matmul=True,  # Use optimized int8 matmul kernels
        use_quantized_matmul_conv=False,
        dequantize_fp32=False,
        non_blocking=False,
        quantization_device="cpu",  # Quantize on CPU
        return_device="cpu",
        modules_to_not_convert=modules_to_not_convert,
    )

    print("\nQuantization configuration:")
    print("  - Dtype: int8")
    print(f"  - Group size: {sdnq_config.group_size}")
    print(f"  - Use quantized matmul: {sdnq_config.use_quantized_matmul}")
    print(f"  - Excluded modules: {len(modules_to_not_convert)} patterns")

    # Prepare model files for loading (copy Python files to transformers cache)
    prepare_model_files(model_path)

    # Preview what will be quantized (unless skipped)
    if not args.skip_preview:
        print(f"\nLoading model for inspection from: {model_path}")
        print("Loading model (this may take a minute)...")

        model_preview = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        # Show what will be quantized vs excluded
        check_exclusions(model_preview, modules_to_not_convert)

        # Clean up preview model
        del model_preview
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print("\nSkipping preview, proceeding directly to quantization...")

    print(f"\nLoading model from: {model_path}")
    print("This will take a few minutes...")

    # Load the model normally first
    # Note: We load first then quantize because moondream3 is a custom model with trust_remote_code=True
    # and transformers doesn't recognize the model_type when quantization_config is passed directly
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,  # Use dtype instead of torch_dtype (deprecated)
        device_map="cpu",  # Load on CPU first
        low_cpu_mem_usage=True,
    )

    print("Model loaded successfully!")
    print("\nApplying INT8 quantization...")
    print("This will take several minutes depending on your system...")

    # Apply quantization to the loaded model
    model = sdnq_post_load_quant(
        model,
        weights_dtype=sdnq_config.weights_dtype,
        group_size=sdnq_config.group_size,
        svd_rank=sdnq_config.svd_rank,
        svd_steps=sdnq_config.svd_steps,
        use_svd=sdnq_config.use_svd,
        quant_conv=sdnq_config.quant_conv,
        use_quantized_matmul=sdnq_config.use_quantized_matmul,
        use_quantized_matmul_conv=sdnq_config.use_quantized_matmul_conv,
        dequantize_fp32=sdnq_config.dequantize_fp32,
        non_blocking=sdnq_config.non_blocking,
        add_skip_keys=False,  # Don't auto-add skip keys, we're specifying them manually
        quantization_device=sdnq_config.quantization_device,
        return_device=sdnq_config.return_device,
        modules_to_not_convert=sdnq_config.modules_to_not_convert,
        modules_dtype_dict={},  # Empty dict, we're not using per-module dtype specifications
    )

    print("Quantization applied successfully!")

    # Save the quantized model
    print(f"\nSaving quantized model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    save_sdnq_model(
        model,
        output_path,
        max_shard_size="5GB",
        is_pipeline=False,
        sdnq_config=sdnq_config,
    )

    print("\n" + "="*80)
    print("✓ QUANTIZATION COMPLETE!")
    print("="*80)
    print(f"\n✓ Quantized model saved to: {output_path}")
    print(f"✓ Quantization type: INT8 (group_size={args.group_size})")
    print(f"✓ Optimized matmul: {'Enabled' if sdnq_config.use_quantized_matmul else 'Disabled'}")

    print("\n" + "="*80)
    print("LOADING THE QUANTIZED MODEL")
    print("="*80)
    print("\nTo load and use the quantized model:")
    print("\n```python")
    print("from transformers import AutoModelForCausalLM")
    print("import sdnq  # Import to register SDNQ with transformers")
    print()
    print("# Load the quantized model")
    print("model = AutoModelForCausalLM.from_pretrained(")
    print(f'    "{output_path}",')
    print("    trust_remote_code=True,")
    print("    device_map='auto',  # or 'cuda' for GPU")
    print(")")
    print()
    print("# The model is ready to use!")
    print("# For moondream3 inference:")
    print("# image_embeds = model.encode_image(image)")
    print("# result = model.query(image_embeds, 'What is in this image?')")
    print("```")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
