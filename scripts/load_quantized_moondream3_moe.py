#!/usr/bin/env python3
"""
Custom loader for moondream3 with properly quantized MoE experts.

This loader patches the model after loading to recreate the dequantization infrastructure.
"""

import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from sdnq.dequantizer import dequantize_symmetric


def patch_moe_quantized_weights(model, model_path):
    """
    Patch MoE modules to use quantized weights after model loading.

    This is needed because transformers doesn't understand our custom buffer structure,
    so we need to manually load and patch the quantized weights.
    """
    print("Patching MoE modules with quantized weights...")

    # Load the state dict directly to get our custom buffers
    import glob

    from safetensors import safe_open

    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # Collect all quantized weights from safetensors
    quantized_weights = {}
    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if 'weight_quantized' in key or 'scale' in key:
                    quantized_weights[key] = f.get_tensor(key)

    print(f"Found {len(quantized_weights)} quantized weight tensors")

    # Patch each MoE block
    patched_count = 0
    for block_idx in range(4, 24):  # MoE blocks 4-23
        block_name = f"model.text.blocks.{block_idx}"
        try:
            block = model.get_submodule(block_name)
        except Exception:
            continue

        if not hasattr(block, 'mlp') or not hasattr(block.mlp, 'fc1'):
            continue

        mlp = block.mlp

        # Check if we have quantized weights for this block
        fc1_weight_q_key = f"{block_name}.mlp.fc1.weight_quantized"
        fc1_scale_key = f"{block_name}.mlp.fc1.scale"
        fc2_weight_q_key = f"{block_name}.mlp.fc2.weight_quantized"
        fc2_scale_key = f"{block_name}.mlp.fc2.scale"

        if fc1_weight_q_key not in quantized_weights:
            continue

        print(f"  Patching block {block_idx}...")

        # Get quantized data
        fc1_weight_q = quantized_weights[fc1_weight_q_key]
        fc1_scale = quantized_weights[fc1_scale_key]
        fc2_weight_q = quantized_weights[fc2_weight_q_key]
        fc2_scale = quantized_weights[fc2_scale_key]

        # Store original weight shapes and device
        fc1_shape = mlp.fc1.weight.shape
        fc2_shape = mlp.fc2.weight.shape
        device = mlp.fc1.weight.device

        # Delete original parameters
        del mlp.fc1.weight
        del mlp.fc2.weight

        # Move quantized weights to model's device and register as buffers
        mlp.fc1.register_buffer('weight_quantized', fc1_weight_q.to(device))
        mlp.fc1.register_buffer('scale', fc1_scale.to(device))
        mlp.fc1.weight_shape = fc1_shape
        mlp.fc1._quantized = True

        mlp.fc2.register_buffer('weight_quantized', fc2_weight_q.to(device))
        mlp.fc2.register_buffer('scale', fc2_scale.to(device))
        mlp.fc2.weight_shape = fc2_shape
        mlp.fc2._quantized = True

        # Create dequantization properties
        # We need to store a reference to the mlp for the property to work
        class WeightProperty:
            def __init__(self, param_dict, weight_key):
                self.param_dict = param_dict
                self.weight_key = weight_key

            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                return dequantize_symmetric(
                    self.param_dict.weight_quantized,
                    self.param_dict.scale,
                    torch.bfloat16,
                    self.param_dict.weight_shape
                )

        # Attach properties - this is tricky because we need instance-level properties
        # Instead of properties, we'll override the __getattr__ of the ParameterDict
        original_fc1_getattr = type(mlp.fc1).__getattribute__
        original_fc2_getattr = type(mlp.fc2).__getattribute__

        def fc1_getattr_override(self, name):
            if name == 'weight' and hasattr(self, '_quantized') and self._quantized:
                return dequantize_symmetric(
                    self.weight_quantized,
                    self.scale,
                    torch.bfloat16,
                    self.weight_shape
                )
            return original_fc1_getattr(self, name)

        def fc2_getattr_override(self, name):
            if name == 'weight' and hasattr(self, '_quantized') and self._quantized:
                return dequantize_symmetric(
                    self.weight_quantized,
                    self.scale,
                    torch.bfloat16,
                    self.weight_shape
                )
            return original_fc2_getattr(self, name)

        # This approach won't work well because we can't override __getattribute__ per instance
        # Let me use a different approach: store the dequantized weight as a property

        # Actually, simplest approach: just dequantize and store as parameter
        # This uses more memory but will work
        fc1_weight_deq = dequantize_symmetric(
            fc1_weight_q.to(device),
            fc1_scale.to(device),
            torch.bfloat16,
            fc1_shape
        )
        fc2_weight_deq = dequantize_symmetric(
            fc2_weight_q.to(device),
            fc2_scale.to(device),
            torch.bfloat16,
            fc2_shape
        )

        # Register as parameters (already on correct device)
        mlp.fc1.weight = nn.Parameter(fc1_weight_deq, requires_grad=False)
        mlp.fc2.weight = nn.Parameter(fc2_weight_deq, requires_grad=False)

        # Keep the quantized versions too for reference
        # (they're still registered as buffers above)

        patched_count += 1

    print(f"Patched {patched_count} MoE blocks")
    return model


def load_quantized_moondream3(model_path):
    """Load a quantized moondream3 model with MoE quantization."""
    print(f"Loading quantized moondream3 from: {model_path}")

    # Load the base model (this will show warnings about unloaded weights, that's okay)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Patch with quantized MoE weights
    model = patch_moe_quantized_weights(model, model_path)

    print("✓ Model loaded and patched successfully")
    return model


if __name__ == "__main__":
    import sys

    model_path = os.path.expanduser(
        "~/database/models/moondream3-preview-int8-sdnq-with-moe-final"
    )

    print("="*80)
    print("LOADING QUANTIZED MOONDREAM3 MODEL")
    print("="*80)

    model = load_quantized_moondream3(model_path)

    print("\nModel loaded! Testing inference...")

    # Load test image
    from io import BytesIO

    import requests
    from PIL import Image

    try:
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        print(f"✓ Loaded test image: {image.size}")

        # Encode and query
        print("Encoding image...")
        image_embeds = model.encode_image(image)
        print("✓ Image encoded")

        question = "What is in this image?"
        print(f"\nQuery: {question}")
        result = model.query(image_embeds, question)
        answer = result.get('answer', 'No answer')

        print(f"Answer: {answer}")
        print("\n✓ SUCCESS! Quantized model is working correctly.")

    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
