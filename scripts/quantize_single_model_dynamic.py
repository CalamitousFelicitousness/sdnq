#!/usr/bin/env python3
"""Quantize a single-model HuggingFace repo with dynamic per-layer dtype selection.

For models that are NOT diffusers pipelines (no model_index.json), e.g. transformers
models loaded via AutoModelForCausalLM with trust_remote_code.

Uses streaming quantization: creates the model skeleton on meta device (no RAM),
then loads each layer from safetensors, quantizes it on GPU, and stores the
quantized result on CPU. Peak RAM is only the quantized model + one bf16 layer.

Usage:
    source /home/ohiom/sdnq/venv/bin/activate
    python scripts/quantize_single_model_dynamic.py \
        --model-id tencent/HunyuanImage-3.0-Instruct \
        --output-path /path/to/output \
        --modules-to-not-convert .vision_model.head .time_embed .final_layer .lm_head
"""

import argparse
import gc
import json
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
    p.add_argument("--modules-to-not-convert", nargs="*", default=None,
                    help="Layer name patterns to skip (e.g. .vision_model.head .lm_head)")
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
            shutil.copytree(item, dst)
            print(f"  Copied {item.name}/")


def detect_model_class(source_path):
    """Auto-detect the model class from config.json auto_map."""
    config_path = os.path.join(source_path, "config.json")
    if not os.path.exists(config_path):
        return "AutoModel"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    auto_map = config.get("auto_map", {})
    for key in ["AutoModelForCausalLM", "AutoModelForSeq2SeqLM",
                 "AutoModelForSequenceClassification", "AutoModel"]:
        if key in auto_map:
            return key
    return "AutoModel"


def get_weight_map(source_path):
    """Build mapping from parameter names to safetensors filenames."""
    source = Path(source_path)
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    for name in ["model.safetensors", "diffusion_pytorch_model.safetensors"]:
        st_path = source / name
        if st_path.exists():
            from safetensors.torch import safe_open
            with safe_open(st_path, framework="pt") as f:
                return {key: name for key in f.keys()}
    raise FileNotFoundError(f"No safetensors files found in {source_path}")


def load_tensor(source_path, weight_map, param_name, shard_cache):
    """Load a single tensor from the appropriate safetensors shard."""
    if param_name not in weight_map:
        return None
    from safetensors.torch import safe_open
    shard_file = weight_map[param_name]
    if shard_file not in shard_cache:
        shard_cache[shard_file] = safe_open(
            Path(source_path) / shard_file, framework="pt"
        )
    return shard_cache[shard_file].get_tensor(param_name)


def set_module_tensor(model, full_name, tensor):
    """Replace a meta parameter/buffer with a real tensor by dotted name."""
    parts = full_name.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    attr = parts[-1]
    old = getattr(obj, attr)
    if isinstance(old, torch.nn.Parameter):
        setattr(obj, attr, torch.nn.Parameter(tensor, requires_grad=False))
    else:
        setattr(obj, attr, tensor)


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

    # 3. Create empty model on meta device (no RAM for weights)
    model_class_name = args.model_class or detect_model_class(source_path)
    print(f"\nCreating model skeleton with {model_class_name} (no weight allocation)...")

    import transformers
    from accelerate import init_empty_weights

    model_cls = getattr(transformers, model_class_name)
    config = transformers.AutoConfig.from_pretrained(
        args.model_id, cache_dir=args.cache_dir, trust_remote_code=True
    )
    with init_empty_weights():
        model = model_cls.from_config(config)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.eval()

    # 4. Get safetensors weight map
    weight_map = get_weight_map(source_path)
    shard_cache = {}
    n_shards = len(set(weight_map.values()))
    print(f"  {len(weight_map)} parameters across {n_shards} shards")

    # 5. Prepare quantization
    from sdnq import SDNQConfig, save_sdnq_model
    from sdnq.common import allowed_types, conv_transpose_types, conv_types
    from sdnq.quantizer import (
        add_module_skip_keys,
        check_param_name_in,
        sdnq_quantize_layer,
    )

    modules_to_not_convert = list(args.modules_to_not_convert or [])
    modules_dtype_dict = {}
    model, modules_to_not_convert, modules_dtype_dict = add_module_skip_keys(
        model, modules_to_not_convert, modules_dtype_dict
    )

    quant_device = torch.device("cuda") if torch.cuda.is_available() else None
    return_device = torch.device("cpu")
    print(f"\nStreaming quantization (quantize on {quant_device or 'cpu'}, "
          f"store on {return_device}):")
    print(f"  min_dtype={args.weights_dtype}, "
          f"threshold={args.dynamic_loss_threshold}, group_size={args.group_size}")
    if modules_to_not_convert:
        print(f"  skip patterns: {len(modules_to_not_convert)}")

    # 6. Stream-load and quantize each module
    quantized_count = 0
    skipped_count = 0

    for module_name, module in model.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue

        param_name = module_name + ".weight"

        # Materialize weight from safetensors
        if module.weight.device.type == "meta":
            tensor = load_tensor(source_path, weight_map, param_name, shard_cache)
            if tensor is None:
                continue  # tied weight or not in checkpoint
            module.weight = torch.nn.Parameter(
                tensor.to(torch.bfloat16), requires_grad=False
            )
            del tensor

        # Materialize bias if present and still on meta
        if hasattr(module, "bias") and module.bias is not None and module.bias.device.type == "meta":
            bias_name = module_name + ".bias"
            tensor = load_tensor(source_path, weight_map, bias_name, shard_cache)
            if tensor is not None:
                module.bias = torch.nn.Parameter(tensor, requires_grad=False)
                del tensor

        # Check if this layer should be quantized
        if check_param_name_in(param_name, modules_to_not_convert):
            skipped_count += 1
            continue

        layer_class = module.__class__.__name__
        if layer_class not in allowed_types:
            continue
        if module.weight.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            continue
        if layer_class in conv_types or layer_class in conv_transpose_types:
            continue

        # Quantize this layer
        module, modules_to_not_convert, modules_dtype_dict = sdnq_quantize_layer(
            module,
            weights_dtype=args.weights_dtype,
            group_size=args.group_size,
            use_dynamic_quantization=True,
            dynamic_loss_threshold=args.dynamic_loss_threshold,
            quantization_device=quant_device,
            return_device=return_device,
            modules_to_not_convert=modules_to_not_convert,
            modules_dtype_dict=modules_dtype_dict,
            param_name=param_name,
        )

        # Replace in parent (sdnq_quantize_layer may wrap the module)
        parts = module_name.rsplit(".", 1)
        if len(parts) == 2:
            parent = model.get_submodule(parts[0])
            setattr(parent, parts[1], module)
        else:
            setattr(model, module_name, module)

        quantized_count += 1
        if quantized_count % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 7. Materialize any remaining meta parameters (norms, embeddings, buffers)
    remaining = 0
    for name, param in list(model.named_parameters()):
        if param.device.type == "meta":
            tensor = load_tensor(source_path, weight_map, name, shard_cache)
            if tensor is not None:
                set_module_tensor(model, name, tensor.to(torch.bfloat16))
                remaining += 1
    for name, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            tensor = load_tensor(source_path, weight_map, name, shard_cache)
            if tensor is not None:
                set_module_tensor(model, name, tensor)
                remaining += 1

    shard_cache.clear()
    gc.collect()

    # Check for any params still stuck on meta
    still_meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    still_meta += [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if still_meta:
        print(f"\n  WARNING: {len(still_meta)} params still on meta "
              f"(tied weights or not in checkpoint):")
        for name in still_meta[:10]:
            print(f"    {name}")

    print(f"\n  Quantized: {quantized_count} layers")
    print(f"  Skipped (by pattern): {skipped_count} layers")
    print(f"  Non-quantized params materialized: {remaining}")

    # 8. Build config and save
    sdnq_config = SDNQConfig(
        weights_dtype=args.weights_dtype,
        group_size=args.group_size,
        use_dynamic_quantization=True,
        dynamic_loss_threshold=args.dynamic_loss_threshold,
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict=modules_dtype_dict,
    )

    if modules_dtype_dict:
        print("\nDtype distribution:")
        for dtype_name, layers in sorted(modules_dtype_dict.items()):
            print(f"  {dtype_name}: {len(layers)} layers")

    model.quantization_config = sdnq_config
    if hasattr(model, "config"):
        try:
            model.config.quantization_config = sdnq_config
        except Exception:
            pass
        try:
            model.config["quantization_config"] = sdnq_config.to_dict()
        except Exception:
            pass

    print(f"\nSaving quantized model to {output_path}...")
    save_sdnq_model(model, output_path, max_shard_size=args.max_shard_size,
                     is_pipeline=False, sdnq_config=sdnq_config)

    print(f"\nQuantization complete! Output: {output_path}")


if __name__ == "__main__":
    main()
