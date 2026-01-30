#!/usr/bin/env python3
"""
Quantize moondream3-preview model to INT8 using SDNQ with MoE support.

This script properly quantizes MoE expert weights which are stored as nn.Parameter
instead of nn.Linear modules.
"""

import argparse
import os
import shutil
import sys

import torch
from transformers import AutoModelForCausalLM

from sdnq import SDNQConfig
from sdnq.common import dtype_dict
from sdnq.quantizer import check_param_name_in, quantize_weight


def prepare_model_files(model_path):
    """Copy Python files from model directory to transformers cache."""
    snapshot_hash = os.path.basename(model_path)
    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    target_dir = os.path.join(cache_dir, snapshot_hash)

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        model_py_files = [f for f in os.listdir(model_path) if f.endswith('.py')]
        cache_py_files = [f for f in os.listdir(target_dir) if f.endswith('.py')]
        if set(model_py_files).issubset(set(cache_py_files)):
            return

    os.makedirs(target_dir, exist_ok=True)
    py_files = [f for f in os.listdir(model_path) if f.endswith('.py')]

    print("Preparing model files in transformers cache...")
    for py_file in py_files:
        src_path = os.path.join(model_path, py_file)
        dst_path = os.path.join(target_dir, py_file)
        if os.path.islink(src_path):
            real_path = os.path.realpath(src_path)
            shutil.copy2(real_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    print(f"Copied {len(py_files)} Python files to transformers cache")


def quantize_moe_experts(model, weights_dtype="int8", group_size=128, modules_to_not_convert=None):
    """
    Quantize MoE expert weights which are stored as nn.Parameter.

    MoE experts in moondream3 have shape: [n_experts, out_dim, in_dim]
    We quantize each expert independently and store the int8 weights.
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    quantized_count = 0
    total_params_quantized = 0
    quantized_params_info = {}

    for name, param in model.named_parameters():
        # Skip if in exclusion list
        if check_param_name_in(name, modules_to_not_convert):
            continue

        # Look for MoE expert weights: 3D tensors named fc1.weight or fc2.weight
        if param.ndim == 3 and ('.mlp.fc1.weight' in name or '.mlp.fc2.weight' in name):
            print(f"  Quantizing MoE expert: {name} {list(param.shape)}")

            n_experts, out_dim, in_dim = param.shape

            # Quantize each expert independently
            quantized_experts = []
            scales_list = []
            zero_points_list = []

            for expert_idx in range(n_experts):
                expert_weight = param.data[expert_idx]  # [out_dim, in_dim]

                # Determine reduction axes based on group_size
                if group_size > 0 and group_size < in_dim:
                    # Group-wise quantization
                    # Reshape to [out_dim, n_groups, group_size]
                    n_groups = in_dim // group_size
                    expert_weight_reshaped = expert_weight[:, :n_groups * group_size].reshape(out_dim, n_groups, group_size)
                    reduction_axes = 2  # Quantize over group_size dimension
                else:
                    # Channel-wise quantization
                    expert_weight_reshaped = expert_weight
                    reduction_axes = 1  # Quantize over in_dim

                # Quantize
                q_weight, scale, zero_point = quantize_weight(
                    expert_weight_reshaped.float(),
                    reduction_axes,
                    weights_dtype
                )

                # Reshape back if needed
                if group_size > 0 and group_size < in_dim:
                    q_weight = q_weight.reshape(out_dim, -1)

                quantized_experts.append(q_weight)
                scales_list.append(scale)
                if zero_point is not None:
                    zero_points_list.append(zero_point)

            # Stack quantized experts and convert to int8
            q_weights_stacked = torch.stack(quantized_experts, dim=0)

            # Store the int8 weights directly (don't convert back to original dtype!)
            # Need to disable gradients for integer tensors
            param.requires_grad = False
            param.data = q_weights_stacked.to(dtype_dict[weights_dtype]["torch_dtype"])

            # Store quantization info for later reference
            quantized_params_info[name] = {
                'scales': torch.stack(scales_list, dim=0) if scales_list else None,
                'zero_points': torch.stack(zero_points_list, dim=0) if zero_points_list else None,
            }

            quantized_count += 1
            total_params_quantized += param.numel()

    print(f"\nQuantized {quantized_count} MoE expert parameters ({total_params_quantized:,} total params)")
    return model, quantized_params_info


def main():
    parser = argparse.ArgumentParser(
        description="Quantize moondream3 with MoE expert quantization"
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
        default="~/database/models/moondream3-preview-int8-sdnq-with-moe",
        help="Path to save the quantized model",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for quantization (default: 128, 0 for channel-wise)",
    )

    args = parser.parse_args()

    # Model path - handle HuggingFace cache directory structure
    model_path = os.path.expanduser(args.model_path)

    if os.path.isdir(model_path):
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_dir) and not os.path.exists(os.path.join(model_path, "config.json")):
            snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshot_dirs:
                snapshot_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                print("Detected HuggingFace cache directory structure.")
                print(f"Using snapshot: {snapshot_dirs[0]}")
                model_path = snapshot_path
            else:
                print(f"Error: No snapshots found in {snapshots_dir}")
                sys.exit(1)

    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Error: config.json not found in {model_path}")
        sys.exit(1)

    output_path = os.path.expanduser(args.output_path)

    # Define modules to exclude from quantization
    modules_to_not_convert = [
        ".vision",  # Vision encoder
        ".region",  # Region encoder/decoder
        ".ln",  # Layer norms
        ".post_ln",
        ".lm_head",  # LM head
        ".mlp.router",  # MoE router
        ".wte",  # Embeddings
        ".freqs_cis",
        ".tau",
        ".kv_cache",
        ".coord_features",
        ".size_features",
    ]

    print("\nQuantization configuration:")
    print("  - Dtype: int8")
    print(f"  - Group size: {args.group_size}")
    print("  - MoE expert quantization: ENABLED")
    print(f"  - Excluded modules: {len(modules_to_not_convert)} patterns")

    # Prepare model files
    prepare_model_files(model_path)

    print(f"\nLoading model from: {model_path}")
    print("This will take a few minutes...")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print("Model loaded successfully!")
    print("\nApplying INT8 quantization...")
    print("=" * 80)

    # Step 1: Quantize nn.Linear modules using SDNQ
    print("\n[1/2] Quantizing nn.Linear modules (attention layers, dense MLPs)...")
    from sdnq import sdnq_post_load_quant

    model = sdnq_post_load_quant(
        model,
        weights_dtype="int8",
        group_size=args.group_size,
        svd_rank=0,
        svd_steps=8,
        use_svd=False,
        quant_conv=False,
        use_quantized_matmul=True,
        use_quantized_matmul_conv=False,
        dequantize_fp32=False,
        non_blocking=False,
        add_skip_keys=False,
        quantization_device="cpu",
        return_device="cpu",
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict={},
    )

    print("✓ nn.Linear modules quantized")

    # Step 2: Quantize MoE expert parameters
    print("\n[2/2] Quantizing MoE expert parameters...")
    model, quantized_params_info = quantize_moe_experts(
        model,
        weights_dtype="int8",
        group_size=args.group_size,
        modules_to_not_convert=modules_to_not_convert
    )

    print("\n" + "=" * 80)
    print("✓ All quantization complete!")

    # Save the quantized model
    print(f"\nSaving quantized model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # Create config for saving
    _sdnq_config = SDNQConfig(
        weights_dtype="int8",
        group_size=args.group_size,
        svd_rank=0,
        use_svd=False,
        quant_conv=False,
        use_quantized_matmul=True,
        modules_to_not_convert=modules_to_not_convert,
    )

    # Use transformers' save_pretrained instead since we have int8 params now
    print("Saving model with int8 MoE expert weights...")
    model.save_pretrained(output_path, max_shard_size="5GB")

    # DON'T save quantization_config.json - it will cause issues when loading
    # The model has mixed quantization (SDNQ for Linear, raw int8 for MoE)
    # and the quantization_config would make transformers try to re-quantize everything
    print("Note: Not saving quantization_config.json to avoid reload issues with mixed quantization")

    # Save quantization info for MoE experts (scales/zero_points)
    if quantized_params_info:
        quant_info_path = os.path.join(output_path, "moe_quantization_info.pt")
        torch.save(quantized_params_info, quant_info_path)
        print(f"Saved MoE quantization info to {quant_info_path}")

    print(f"\n{'='*80}")
    print("✓ QUANTIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\n✓ Quantized model saved to: {output_path}")
    print("✓ Quantization includes MoE expert weights")
    print("\nNote: MoE expert weights are quantized as raw parameters.")
    print("They will use less disk space but won't use optimized int8 matmul during inference.")


if __name__ == "__main__":
    main()
