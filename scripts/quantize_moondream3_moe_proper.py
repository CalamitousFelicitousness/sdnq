#!/usr/bin/env python3
"""
Properly quantize moondream3 MoE experts using SDNQ-style infrastructure.

This creates a dequantization wrapper for MoE experts that works like SDNQ's Linear quantization.
"""

import argparse
import os
import shutil
import sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from sdnq import SDNQConfig, save_sdnq_model
from sdnq.dequantizer import dequantize_symmetric
from sdnq.quantizer import quantize_weight


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


def create_quantized_moe_forward(original_forward, n_experts, weights_dtype="int8"):
    """
    Create a forward function that dequantizes MoE experts on-the-fly.

    This wraps the original MoE forward function to handle quantized expert weights.
    """
    def quantized_moe_forward(x, mlp_module, experts_per_token):
        # Dequantize expert weights if they're quantized
        if hasattr(mlp_module.fc1, '_quantized') and mlp_module.fc1._quantized:
            # Dequantize fc1 weights
            fc1_weight_dequant = dequantize_symmetric(
                mlp_module.fc1.weight_quantized,
                mlp_module.fc1.scale,
                mlp_module.fc1.scale.dtype,
                mlp_module.fc1.weight_shape
            )
            mlp_module.fc1.weight = fc1_weight_dequant

            # Dequantize fc2 weights
            fc2_weight_dequant = dequantize_symmetric(
                mlp_module.fc2.weight_quantized,
                mlp_module.fc2.scale,
                mlp_module.fc2.scale.dtype,
                mlp_module.fc2.weight_shape
            )
            mlp_module.fc2.weight = fc2_weight_dequant

        # Run original forward
        return original_forward(x, mlp_module, experts_per_token)

    return quantized_moe_forward


def quantize_moe_experts_proper(model, weights_dtype="int8", group_size=128, modules_to_not_convert=None):
    """
    Properly quantize MoE expert weights using SDNQ-style approach.

    Instead of directly storing int8 parameters, we:
    1. Store quantized weights as buffers (not parameters)
    2. Store scales as parameters
    3. Keep original parameter names but mark as dequantized
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    quantized_count = 0
    total_params_quantized = 0

    print("Searching for MoE modules to quantize...")

    # Find all MoE blocks
    for block_idx in range(4, 24):  # MoE starts at block 4
        block_name = f"model.text.blocks.{block_idx}"
        try:
            block = model.get_submodule(block_name)
        except Exception:
            continue

        if not hasattr(block, 'mlp') or not hasattr(block.mlp, 'fc1'):
            continue

        mlp = block.mlp

        # Check if this is an MoE block (has 'weight' ParameterDict)
        if not isinstance(mlp.fc1, nn.ParameterDict):
            continue

        print(f"  Quantizing MoE block {block_idx}...")

        # Quantize fc1 expert weights
        fc1_weight = mlp.fc1.weight  # [n_experts, 2*d_ffn, d_model]
        n_experts, _out_dim, _in_dim = fc1_weight.shape

        # Quantize each expert independently (channel-wise within each expert)
        quantized_fc1 = []
        scales_fc1 = []

        for expert_idx in range(n_experts):
            expert_weight = fc1_weight[expert_idx]  # [out_dim, in_dim]
            q_weight, scale, _ = quantize_weight(
                expert_weight.float(),
                reduction_axes=1,  # Channel-wise quantization
                weights_dtype=weights_dtype
            )
            quantized_fc1.append(q_weight)
            scales_fc1.append(scale)

        fc1_quantized = torch.stack(quantized_fc1, dim=0)
        fc1_scales = torch.stack(scales_fc1, dim=0)

        # Store as buffer (not parameter) - buffers don't require gradients
        mlp.fc1.register_buffer('weight_quantized', fc1_quantized)
        mlp.fc1.weight_shape = fc1_weight.shape
        mlp.fc1.scale = nn.Parameter(fc1_scales, requires_grad=False)
        mlp.fc1._quantized = True

        # Replace the original weight with a property that dequantizes on access
        # This allows backward compatibility
        del mlp.fc1.weight  # Remove the original parameter

        # Do the same for fc2
        fc2_weight = mlp.fc2.weight  # [n_experts, d_model, d_ffn]
        n_experts, _out_dim, _in_dim = fc2_weight.shape

        quantized_fc2 = []
        scales_fc2 = []

        for expert_idx in range(n_experts):
            expert_weight = fc2_weight[expert_idx]
            q_weight, scale, _ = quantize_weight(
                expert_weight.float(),
                reduction_axes=1,
                weights_dtype=weights_dtype
            )
            quantized_fc2.append(q_weight)
            scales_fc2.append(scale)

        fc2_quantized = torch.stack(quantized_fc2, dim=0)
        fc2_scales = torch.stack(scales_fc2, dim=0)

        mlp.fc2.register_buffer('weight_quantized', fc2_quantized)
        mlp.fc2.weight_shape = fc2_weight.shape
        mlp.fc2.scale = nn.Parameter(fc2_scales, requires_grad=False)
        mlp.fc2._quantized = True

        del mlp.fc2.weight

        quantized_count += 2
        total_params_quantized += fc1_weight.numel() + fc2_weight.numel()

    print(f"\nQuantized {quantized_count} MoE expert parameter groups ({total_params_quantized:,} total params)")

    # Now patch the moe_mlp function to handle dequantization
    patch_moe_forward(model)

    return model


def patch_moe_forward(model):
    """
    Patch the moe_mlp function to dequantize weights before use.
    """
    print("Patching MoE forward function for dequantization...")

    # We need to modify the forward pass of MoE blocks
    # The cleanest way is to wrap the block's forward method
    for block_idx in range(4, 24):
        block_name = f"model.text.blocks.{block_idx}"
        try:
            block = model.get_submodule(block_name)
        except Exception:
            continue

        if not hasattr(block, 'mlp') or not hasattr(block.mlp, 'fc1'):
            continue

        mlp = block.mlp

        if hasattr(mlp.fc1, '_quantized') and mlp.fc1._quantized:
            # Create dequantized weight properties
            # These will be accessed during forward
            @property
            def fc1_weight_property(self):
                return dequantize_symmetric(
                    self.fc1.weight_quantized,
                    self.fc1.scale,
                    torch.bfloat16,  # Target dtype
                    self.fc1.weight_shape
                )

            @property
            def fc2_weight_property(self):
                return dequantize_symmetric(
                    self.fc2.weight_quantized,
                    self.fc2.scale,
                    torch.bfloat16,
                    self.fc2.weight_shape
                )

            # Attach as properties
            type(mlp.fc1).weight = fc1_weight_property
            type(mlp.fc2).weight = fc2_weight_property


def main():
    parser = argparse.ArgumentParser(
        description="Properly quantize moondream3 with MoE using SDNQ infrastructure"
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
        default="~/database/models/moondream3-preview-int8-sdnq-with-moe-v2",
        help="Path to save the quantized model",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for quantization (default: 128)",
    )

    args = parser.parse_args()

    # Model path handling
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

    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Error: config.json not found in {model_path}")
        sys.exit(1)

    output_path = os.path.expanduser(args.output_path)

    # Define exclusions
    # NOTE: SDNQ checks if pattern is in param_name.split("."), so use bare names like "vision" not ".vision"
    modules_to_not_convert = [
        "vision",  # Vision encoder - uses custom forward functions
        "region",  # Region encoder/decoder
        "ln",  # Layer norms
        "post_ln",
        "lm_head",  # Language model head
        "router",  # MoE router (we quantize experts separately)
        "wte",  # Embeddings
        "freqs_cis",
        "tau",
        "kv_cache",
        "coord_features",
        "size_features",
    ]

    print("\nQuantization configuration:")
    print("  - Dtype: int8")
    print(f"  - Group size: {args.group_size}")
    print("  - Approach: SDNQ-style with dequantization infrastructure")
    print(f"  - Excluded modules: {len(modules_to_not_convert)} patterns")

    # Prepare files
    prepare_model_files(model_path)

    print(f"\nLoading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print("Model loaded!")
    print("\n" + "="*80)
    print("Step 1: Quantizing nn.Linear modules (attention, dense MLPs)...")
    print("="*80)

    from sdnq import sdnq_post_load_quant
    model = sdnq_post_load_quant(
        model,
        weights_dtype="int8",
        group_size=args.group_size,
        svd_rank=0,
        use_svd=False,
        quant_conv=False,
        use_quantized_matmul=True,
        add_skip_keys=False,
        quantization_device="cpu",
        return_device="cpu",
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict={},
    )

    print("✓ Linear modules quantized\n")

    print("="*80)
    print("Step 2: Properly quantizing MoE experts with dequantization infrastructure...")
    print("="*80)

    model = quantize_moe_experts_proper(
        model,
        weights_dtype="int8",
        group_size=args.group_size,
        modules_to_not_convert=modules_to_not_convert
    )

    print("\n✓ MoE quantization complete!")

    # Save
    print(f"\nSaving to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    sdnq_config = SDNQConfig(
        weights_dtype="int8",
        group_size=args.group_size,
        use_quantized_matmul=True,
        modules_to_not_convert=modules_to_not_convert,
    )

    save_sdnq_model(
        model,
        output_path,
        max_shard_size="5GB",
        is_pipeline=False,
        sdnq_config=sdnq_config,
    )

    print(f"\n{'='*80}")
    print("✓ QUANTIZATION COMPLETE!")
    print(f"{'='*80}")
    print("\n✓ Model saved with proper MoE quantization infrastructure")
    print("✓ Weights are stored as buffers with dequantization on access")


if __name__ == "__main__":
    main()
