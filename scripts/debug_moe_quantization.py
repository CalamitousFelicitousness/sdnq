#!/usr/bin/env python3
"""Debug: Check if MoE parameters were actually quantized."""


# Load one of the safetensors files directly
from safetensors import safe_open

model_path = "/home/ohiom/database/models/moondream3-preview-int8-sdnq-with-moe/model-00001-of-00002.safetensors"

print("Checking MoE expert parameter dtypes in saved model...\n")

with safe_open(model_path, framework="pt", device="cpu") as f:
    keys = f.keys()

    # Check MoE expert weights from blocks 4+ (where MoE starts)
    moe_keys = [k for k in keys if ('.mlp.fc1.weight' in k or '.mlp.fc2.weight' in k) and
                any(f'.blocks.{i}.' in k for i in range(4, 24))]

    print(f"Found {len(moe_keys)} MoE expert parameters\n")

    for key in moe_keys[:5]:  # Check first 5
        tensor = f.get_tensor(key)
        print(f"{key}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Size: {tensor.numel() * tensor.element_size() / 1024 / 1024:.2f} MB")
        print()
