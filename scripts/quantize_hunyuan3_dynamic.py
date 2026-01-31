#!/usr/bin/env python3
"""Quantize HunyuanImage-3.0-Instruct with dynamic per-layer dtype selection.

Uses dynamic quantization with:
  - Minimum dtype: uint4 (int4 equivalent)
  - Std-normalized MSE threshold: 1e-2
  - SVD disabled
  - Group size: 32 (default)

Each layer's dtype is selected by trial-and-error starting from uint4, stepping
up through int5, float5, uint5, int6, ... until the quantization loss is below
the threshold. This produces a mixed-precision model.

The output directory is a complete pipeline repo suitable for uploading to
HuggingFace Hub, including all non-model files (README, configs, tokenizers,
schedulers, etc.).

Usage:
    source /home/ohiom/sdnq/venv/bin/activate
    python scripts/quantize_hunyuan3_dynamic.py \
        --model-id tencent/HunyuanImage-3.0-Instruct \
        --output-path /path/to/output
"""

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Quantize HunyuanImage-3.0-Instruct with dynamic per-layer dtype selection"
    )
    p.add_argument("--model-id", default="tencent/HunyuanImage-3.0-Instruct",
                    help="HuggingFace model ID or local path")
    p.add_argument("--output-path", required=True,
                    help="Output directory for the quantized pipeline")
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
    p.add_argument("--components", nargs="+", default=None,
                    help="Subset of components to quantize (subfolder names)")
    p.add_argument("--verbose", action="store_true",
                    help="Show per-layer quantization decisions")
    return p.parse_args()


def resolve_model_path(model_id, cache_dir):
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


def resolve_class(class_name):
    """Resolve a model class name from diffusers or transformers."""
    for pkg in ["diffusers", "transformers"]:
        try:
            mod = importlib.import_module(pkg)
            cls = getattr(mod, class_name, None)
            if cls is not None:
                return cls
        except ImportError:
            continue
    raise ValueError(f"Could not resolve class '{class_name}' from diffusers or transformers")


def discover_components(model_index):
    """Discover quantizable and non-quantizable components from model_index.json."""
    skip_types = {
        "FlowMatchEulerDiscreteScheduler", "EulerDiscreteScheduler",
        "DDIMScheduler", "PNDMScheduler", "DPMSolverMultistepScheduler",
        "LMSDiscreteScheduler", "UniPCMultistepScheduler",
        "Qwen2Tokenizer", "CLIPTokenizer", "T5Tokenizer",
        "T5TokenizerFast", "CLIPTokenizerFast", "LlamaTokenizer",
        "LlamaTokenizerFast", "PreTrainedTokenizerFast",
        "CLIPImageProcessor", "CLIPFeatureExtractor",
    }
    quantizable = []
    non_quantizable = []
    for key, value in model_index.items():
        if not isinstance(value, list) or len(value) < 2:
            continue
        class_name = value[1]
        if class_name in skip_types:
            non_quantizable.append((key, class_name))
        else:
            quantizable.append((key, class_name))
    return quantizable, non_quantizable


def copy_all_extra_files(source_path, output_path):
    """Copy all non-model files (READMEs, configs, images, etc.) from the source repo."""
    # Files at the root level that aren't model weights or component subdirs
    for item in Path(source_path).iterdir():
        dst = Path(output_path) / item.name
        if dst.exists():
            continue
        if item.is_file():
            # Skip large safetensors/bin files at root (shouldn't be any for pipelines)
            if item.suffix in (".safetensors", ".bin", ".pt", ".pth"):
                continue
            shutil.copy2(item, dst)
            print(f"  Copied {item.name}")
        # Don't copy subdirectories here — those are handled by component logic


def copy_non_quantizable_components(source_path, output_path, non_quantizable):
    """Copy non-quantizable component subdirectories (schedulers, tokenizers, etc.)."""
    for subfolder, class_name in non_quantizable:
        src_dir = os.path.join(source_path, subfolder)
        dst_dir = os.path.join(output_path, subfolder)
        if os.path.isdir(src_dir):
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print(f"  Copied {subfolder}/ ({class_name})")
        else:
            print(f"  WARNING: {subfolder}/ not found at {src_dir}")


def quantize_component(model_id, subfolder, class_name, output_path, cache_dir,
                        max_shard_size, weights_dtype, dynamic_loss_threshold, group_size):
    """Load, quantize with dynamic dtype selection, and save a single component."""
    from sdnq import SDNQConfig, save_sdnq_model
    from sdnq.quantizer import sdnq_post_load_quant

    print(f"\n{'='*60}")
    print(f"Quantizing {subfolder} ({class_name})")
    print(f"  min_dtype={weights_dtype}, threshold={dynamic_loss_threshold}, group_size={group_size}")
    print(f"{'='*60}")

    cls = resolve_class(class_name)
    model = cls.from_pretrained(
        model_id, subfolder=subfolder,
        torch_dtype=torch.bfloat16, cache_dir=cache_dir,
    )

    model = sdnq_post_load_quant(
        model,
        add_skip_keys=True,
        weights_dtype=weights_dtype,
        group_size=group_size,
        use_dynamic_quantization=True,
        dynamic_loss_threshold=dynamic_loss_threshold,
    )

    # Build config reflecting what was actually applied
    sdnq_config = SDNQConfig(
        weights_dtype=weights_dtype,
        group_size=group_size,
        use_dynamic_quantization=True,
        dynamic_loss_threshold=dynamic_loss_threshold,
    )
    # Copy the per-layer dtype decisions into the saved config
    if hasattr(model, "quantization_config"):
        mc = model.quantization_config
        sdnq_config.modules_to_not_convert = mc.modules_to_not_convert
        sdnq_config.modules_dtype_dict = mc.modules_dtype_dict

    component_path = os.path.join(output_path, subfolder)
    os.makedirs(component_path, exist_ok=True)
    save_sdnq_model(model, component_path, max_shard_size=max_shard_size,
                     is_pipeline=False, sdnq_config=sdnq_config)

    # Print dtype distribution
    if sdnq_config.modules_dtype_dict:
        print(f"\n  Dtype distribution for {subfolder}:")
        for dtype_name, layers in sorted(sdnq_config.modules_dtype_dict.items()):
            print(f"    {dtype_name}: {len(layers)} layers")
    if sdnq_config.modules_to_not_convert:
        print(f"    skipped (above threshold): {len(sdnq_config.modules_to_not_convert)} layers")

    print(f"  Saved to {component_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    import logging
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # 1. Resolve source path and load model_index.json
    source_path = resolve_model_path(args.model_id, args.cache_dir)
    print(f"Source: {source_path}")

    model_index_path = os.path.join(source_path, "model_index.json")
    if not os.path.exists(model_index_path):
        print("ERROR: model_index.json not found. Is this a diffusers pipeline?")
        sys.exit(1)

    with open(model_index_path, encoding="utf-8") as f:
        model_index = json.load(f)

    # 2. Discover components
    quantizable, non_quantizable = discover_components(model_index)
    print(f"\nQuantizable components: {[c[0] for c in quantizable]}")
    print(f"Non-quantizable components: {[c[0] for c in non_quantizable]}")

    # Filter to requested components
    if args.components:
        requested = set(args.components)
        quantizable = [(s, c) for s, c in quantizable if s in requested]
        print(f"Filtered to: {[c[0] for c in quantizable]}")

    # 3. Copy model_index.json
    shutil.copy2(model_index_path, os.path.join(output_path, "model_index.json"))
    print("\nCopied model_index.json")

    # 4. Copy all extra files (READMEs, images, configs, etc.)
    print("\nCopying extra files...")
    copy_all_extra_files(source_path, output_path)

    # 5. Copy non-quantizable components
    print("\nCopying non-quantizable components...")
    copy_non_quantizable_components(source_path, output_path, non_quantizable)

    # 6. Quantize each component
    for subfolder, class_name in quantizable:
        quantize_component(
            args.model_id, subfolder, class_name,
            output_path, args.cache_dir, args.max_shard_size,
            args.weights_dtype, args.dynamic_loss_threshold, args.group_size,
        )

    print(f"\n{'='*60}")
    print("Pipeline quantization complete!")
    print(f"Output: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
