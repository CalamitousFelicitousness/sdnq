#!/usr/bin/env python3
"""Quantize a single-model HuggingFace repo with dynamic per-layer dtype selection.

For models that are NOT diffusers pipelines (no model_index.json), e.g. transformers
models loaded via AutoModelForCausalLM with trust_remote_code.

Uses dynamic quantization with configurable minimum dtype and loss threshold.
Copies all non-weight files (READMEs, configs, tokenizers, custom code, etc.)
to produce a complete repo for uploading to HuggingFace Hub.

Usage:
    source /home/ohiom/sdnq/venv/bin/activate
    python scripts/quantize_single_model_dynamic.py \
        --model-id tencent/HunyuanImage-3.0-Instruct-Distil \
        --output-path /path/to/output
"""

import argparse
import os
import shutil
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Quantize a single HuggingFace model with dynamic per-layer dtype selection"
    )
    p.add_argument("--model-id", required=True,
                    help="HuggingFace model ID or local path")
    p.add_argument("--output-path", required=True,
                    help="Output directory for the quantized model")
    p.add_argument("--cache-dir", default="/home/ohiom/database/models/huggingface",
                    help="HuggingFace cache directory")
    p.add_argument("--max-shard-size", default="5GB",
                    help="Max shard size for saved model files")
    p.add_argument("--weights-dtype", default="uint4",
                    help="Minimum quantization dtype (default: uint4)")
    p.add_argument("--dynamic-loss-threshold", type=float, default=1e-2,
                    help="Std-normalized MSE loss threshold (default: 1e-2)")
    p.add_argument("--group-size", type=int, default=0,
                    help="Group size for quantization (default: 0 = auto 32)")
    p.add_argument("--model-class", default=None,
                    help="Model class to use, e.g. AutoModelForCausalLM (default: auto-detect)")
    p.add_argument("--verbose", action="store_true",
                    help="Show per-layer quantization decisions")
    return p.parse_args()


def resolve_source_path(model_id, cache_dir):
    """Resolve the local cache path for a model ID."""
    local_path = Path(model_id)
    if local_path.is_dir():
        return str(local_path)
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(model_id, cache_dir=cache_dir, local_files_only=True)
    except Exception:
        print(f"Downloading {model_id}...")
        return snapshot_download(model_id, cache_dir=cache_dir)


def copy_non_weight_files(source_path, output_path):
    """Copy all non-weight files from the source repo to the output directory."""
    weight_extensions = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
    skip_names = {".git", ".gitattributes"}

    for item in Path(source_path).iterdir():
        if item.name in skip_names:
            continue
        dst = Path(output_path) / item.name
        if dst.exists():
            continue
        if item.is_file():
            if item.suffix in weight_extensions:
                continue
            shutil.copy2(item, dst)
            print(f"  Copied {item.name}")
        elif item.is_dir():
            # Copy non-weight subdirectories (assets, utils, etc.)
            # Skip if it looks like a weight cache directory
            shutil.copytree(item, dst)
            print(f"  Copied {item.name}/")


def detect_model_class(source_path):
    """Auto-detect the model class from config.json auto_map."""
    import json
    config_path = os.path.join(source_path, "config.json")
    if not os.path.exists(config_path):
        return "AutoModel"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    auto_map = config.get("auto_map", {})
    # Prefer CausalLM > Model
    for key in ["AutoModelForCausalLM", "AutoModelForSeq2SeqLM",
                 "AutoModelForSequenceClassification", "AutoModel"]:
        if key in auto_map:
            return key
    return "AutoModel"


def main():
    import logging
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # 1. Resolve source path
    source_path = resolve_source_path(args.model_id, args.cache_dir)
    print(f"Source: {source_path}")

    # 2. Copy non-weight files first
    print("\nCopying non-weight files...")
    copy_non_weight_files(source_path, output_path)

    # 3. Detect and load model
    model_class_name = args.model_class or detect_model_class(source_path)
    print(f"\nLoading model with {model_class_name} (trust_remote_code=True)...")

    import transformers
    model_cls = getattr(transformers, model_class_name)
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = model_cls.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        device_map=device_map,
    )

    # 4. Quantize with dynamic per-layer dtype selection
    from sdnq import SDNQConfig, save_sdnq_model
    from sdnq.quantizer import sdnq_post_load_quant

    print("\nQuantizing with dynamic dtype selection:")
    print(f"  min_dtype={args.weights_dtype}, threshold={args.dynamic_loss_threshold}, "
          f"group_size={args.group_size}")

    model = sdnq_post_load_quant(
        model,
        add_skip_keys=True,
        weights_dtype=args.weights_dtype,
        group_size=args.group_size,
        use_dynamic_quantization=True,
        dynamic_loss_threshold=args.dynamic_loss_threshold,
    )

    # 5. Build config and save
    sdnq_config = SDNQConfig(
        weights_dtype=args.weights_dtype,
        group_size=args.group_size,
        use_dynamic_quantization=True,
        dynamic_loss_threshold=args.dynamic_loss_threshold,
    )
    if hasattr(model, "quantization_config"):
        mc = model.quantization_config
        sdnq_config.modules_to_not_convert = mc.modules_to_not_convert
        sdnq_config.modules_dtype_dict = mc.modules_dtype_dict

    # Print dtype distribution
    if sdnq_config.modules_dtype_dict:
        print("\nDtype distribution:")
        for dtype_name, layers in sorted(sdnq_config.modules_dtype_dict.items()):
            print(f"  {dtype_name}: {len(layers)} layers")
    if sdnq_config.modules_to_not_convert:
        print(f"  skipped (above threshold): {len(sdnq_config.modules_to_not_convert)} layers")

    print(f"\nSaving quantized model to {output_path}...")
    save_sdnq_model(model, output_path, max_shard_size=args.max_shard_size,
                     is_pipeline=False, sdnq_config=sdnq_config)

    print(f"\nQuantization complete! Output: {output_path}")


if __name__ == "__main__":
    main()
