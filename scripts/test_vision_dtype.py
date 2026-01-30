#!/usr/bin/env python3
"""Quick test to check vision encoder dtype issue."""

import os
from io import BytesIO

import requests
from PIL import Image

# Add model path to Python path for imports
model_path = os.path.expanduser("~/database/models/moondream3-preview-int8-sdnq-with-moe-v2")

from load_quantized_moondream3_moe import load_quantized_moondream3

print("Loading model...")
model = load_quantized_moondream3(model_path)

print("\nChecking vision encoder layers...")
for name, module in model.named_modules():
    if 'vision' in name and hasattr(module, 'weight'):
        print(f"{name}: weight dtype = {module.weight.dtype if hasattr(module, 'weight') else 'N/A'}")

print("\nLoading test image...")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
response = requests.get(url, timeout=10)
image = Image.open(BytesIO(response.content))

print("Running image preprocessing...")
try:
    # Check what model.vision expects

    # Try to encode with explicit dtype
    image_embeds = model.encode_image(image)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

    # Check model device and dtype
    print(f"\nModel device: {model.device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
