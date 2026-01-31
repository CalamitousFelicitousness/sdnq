#!/usr/bin/env python3
"""Estimate theoretical memory footprint for SDNQ pipeline quantization configs.

Loads model component structures on meta device (no GPU/CPU memory required),
classifies every parameter per pipeline config JSON, and produces a
quality-vs-memory comparison table across all test configurations.

Usage:
    python scripts/estimate_config_memory.py
    python scripts/estimate_config_memory.py --model-id Tongyi-MAI/Z-Image
    python scripts/estimate_config_memory.py --output memory_comparison.csv
    python scripts/estimate_config_memory.py --configs ref2_int8.json ref2_uint4.json
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ---------------------------------------------------------------------------
# Dtype info for memory estimation (inlined to avoid heavy sdnq import)
# ---------------------------------------------------------------------------

DTYPE_BITS = {
    "int8": 8, "int7": 7, "int6": 6, "int5": 5, "int4": 4, "int3": 3, "int2": 2,
    "uint8": 8, "uint7": 7, "uint6": 6, "uint5": 5, "uint4": 4, "uint3": 3, "uint2": 2, "uint1": 1,
    "float8_e4m3fn": 8, "float8_e5m2": 8, "fp8": 8,
    "fp16": 16, "float16": 16, "bfloat16": 16, "bf16": 16,
    "fp32": 32, "float32": 32,
}

DTYPE_UNSIGNED = {
    "uint8", "uint7", "uint6", "uint5", "uint4", "uint3", "uint2", "uint1",
}

QUANTIZABLE_CLASSES = {"Linear", "Conv1d", "Conv2d", "Conv3d"}


# ---------------------------------------------------------------------------
# Layer inventory from meta-device models
# ---------------------------------------------------------------------------

def build_layer_inventory(model):
    """Walk model modules, collect info for every parameter.

    Returns list of dicts with: name, layer_class, shape, numel, is_quantizable.
    """
    inventory = []
    for module_name, module in model.named_modules():
        layer_class = module.__class__.__name__
        is_quantizable_type = layer_class in QUANTIZABLE_CLASSES
        for pname, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{pname}" if module_name else pname
            inventory.append({
                "name": full_name,
                "layer_class": layer_class,
                "shape": tuple(param.shape),
                "numel": math.prod(param.shape),
                "is_quantizable": is_quantizable_type and pname == "weight",
            })
    return inventory


def load_component_inventory(model_id, comp_name, lib_name, class_name, cache_dir=None):
    """Load a single pipeline component on meta device and return its inventory."""
    if lib_name == "transformers":
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(model_id, subfolder=comp_name, cache_dir=cache_dir)
        with init_empty_weights():
            model = AutoModel.from_config(config)
    else:
        config_path = hf_hub_download(model_id, f"{comp_name}/config.json", cache_dir=cache_dir)
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        lib = __import__(lib_name, fromlist=[class_name])
        cls = getattr(lib, class_name)
        with init_empty_weights():
            model = cls.from_config(config)

    inv = build_layer_inventory(model)
    del model
    return inv


def load_all_inventories(model_id, cache_dir=None):
    """Load model_index.json and build inventories for all model components."""
    index_path = hf_hub_download(model_id, "model_index.json", cache_dir=cache_dir)
    with open(index_path, encoding="utf-8") as f:
        model_index = json.load(f)

    # Class names that are not neural network models (schedulers, tokenizers, etc.)
    non_model_classes = {
        "FlowMatchEulerDiscreteScheduler", "EulerDiscreteScheduler",
        "DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler",
    }

    inventories = {}
    for key, value in model_index.items():
        if key.startswith("_") or not isinstance(value, list) or len(value) != 2:
            continue
        lib_name, class_name = value
        if not lib_name or not class_name:
            continue
        if lib_name not in ("diffusers", "transformers"):
            continue
        if class_name in non_model_classes:
            continue
        try:
            inv = load_component_inventory(model_id, key, lib_name, class_name, cache_dir)
            inventories[key] = inv
            total_params = sum(p["numel"] for p in inv)
            quantizable = sum(p["numel"] for p in inv if p["is_quantizable"])
            print(f"  {key} ({class_name}): {len(inv)} tensors, "
                  f"{total_params:,} params ({quantizable:,} quantizable)")
        except Exception as e:
            print(f"  Warning: could not load {key} ({class_name}): {e}")

    return inventories


# ---------------------------------------------------------------------------
# Parameter name matching (replicates quantizer.py:123 logic)
# ---------------------------------------------------------------------------

def check_param_name_in(param_name, param_list):
    """Check if param_name matches any pattern in param_list."""
    if not param_list:
        return False
    split_param_name = param_name.split(".")
    for param in param_list:
        if param.startswith("."):
            if param_name.startswith(param[1:]):
                return True
            continue
        if (
            param_name == param
            or param in split_param_name
            or param_name.endswith("." + param)
            or ("*" in param and re.match(
                param.replace(".*", r"\..*").replace("*", ".*"), param_name))
        ):
            return True
    return False


def get_effective_dtype(weights_dtype, param_name, modules_dtype_dict):
    """Determine effective dtype for a layer considering promotions."""
    if not modules_dtype_dict:
        return weights_dtype
    base_bits = DTYPE_BITS.get(weights_dtype, 16)
    for key, patterns in modules_dtype_dict.items():
        if check_param_name_in(param_name, patterns):
            key_lower = key.lower()
            if key_lower in {"8bit", "8bits"}:
                if base_bits != 8:
                    return "int8"
            elif key_lower.startswith("minimum_"):
                min_str = key_lower.removeprefix("minimum_").removesuffix("bits").removesuffix("bit")
                if min_str.startswith("uint"):
                    is_unsigned = True
                    min_str = min_str.removeprefix("uint")
                else:
                    is_unsigned = False
                    min_str = min_str.removeprefix("int")
                minimum_bits = int(min_str)
                if base_bits < minimum_bits:
                    return ("uint" if is_unsigned or minimum_bits <= 4 else "int") + min_str
            else:
                return key_lower
    return weights_dtype


def get_layer_svd(use_svd, svd_rank, param_name, modules_svd_dict):
    """Get per-layer SVD settings."""
    if modules_svd_dict:
        for rank, layer_names in modules_svd_dict.items():
            if check_param_name_in(param_name, layer_names):
                return True, int(rank)
    return use_svd, svd_rank


def get_layer_group_size(param_name, modules_group_size_dict, default_group_size):
    """Get per-layer group size."""
    if modules_group_size_dict:
        for gs, layer_names in modules_group_size_dict.items():
            if check_param_name_in(param_name, layer_names):
                return int(gs)
    return default_group_size


# ---------------------------------------------------------------------------
# Group size resolution (replicates quantizer.py auto group_size logic)
# ---------------------------------------------------------------------------

def resolve_group_size(group_size, weights_dtype, shape, use_svd, layer_class):
    """Resolve group_size=0 to actual value."""
    if group_size != 0:
        return group_size

    num_bits = DTYPE_BITS.get(weights_dtype, 16)
    is_linear = layer_class in {"Linear", "SDNQLinear"}

    if is_linear:
        group_size = 2 ** ((3 if use_svd else 2) + num_bits)
    else:
        group_size = 2 ** ((2 if use_svd else 1) + num_bits)

    # Determine channel_size from shape
    if len(shape) >= 2:
        channel_size = shape[1]
        for d in shape[2:]:
            channel_size *= d
    elif len(shape) == 1:
        channel_size = shape[0]
    else:
        return -1

    if group_size >= channel_size:
        return -1

    # Find divisible group_size
    num_of_groups = channel_size // group_size
    while num_of_groups > 1 and num_of_groups * group_size != channel_size:
        num_of_groups -= 1
        group_size = channel_size // num_of_groups
    if num_of_groups <= 1:
        return -1

    return group_size


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_layer_bytes(numel, shape, dtype_str, group_size, use_svd, svd_rank):
    """Estimate total storage bytes for one quantized layer."""
    num_bits = DTYPE_BITS.get(dtype_str, 16)
    is_unsigned = dtype_str in DTYPE_UNSIGNED

    weight_bytes = numel * num_bits / 8

    if group_size <= 0:
        scale_count = shape[0] if shape else 1
    else:
        scale_count = math.ceil(numel / group_size)
    scale_bytes = scale_count * 2  # fp16 scales

    zp_bytes = scale_count * 2 if is_unsigned else 0

    svd_bytes = 0
    if use_svd and len(shape) >= 2:
        out_features = shape[0]
        in_features = shape[1]
        for d in shape[2:]:
            in_features *= d
        svd_bytes = (out_features * svd_rank + svd_rank * in_features) * 2

    return int(weight_bytes + scale_bytes + zp_bytes + svd_bytes)


def estimate_component_memory(inventory, comp_config):
    """Estimate quantized and FP16 memory for one model component.

    Returns (quantized_bytes, fp16_bytes).
    """
    weights_dtype = comp_config.get("weights_dtype", "int8")
    group_size = comp_config.get("group_size", 0)
    use_svd = comp_config.get("use_svd", False)
    svd_rank = comp_config.get("svd_rank", 32)
    modules_to_not_convert = comp_config.get("modules_to_not_convert", [])
    modules_dtype_dict = comp_config.get("modules_dtype_dict", {})
    modules_svd_dict = comp_config.get("modules_svd_dict", {})
    modules_group_size_dict = comp_config.get("modules_group_size_dict", {})

    # JSON keys are strings; convert to int where needed
    if modules_svd_dict:
        modules_svd_dict = {int(k): v for k, v in modules_svd_dict.items()}
    if modules_group_size_dict:
        modules_group_size_dict = {int(k): v for k, v in modules_group_size_dict.items()}

    total_quant = 0
    total_fp16 = 0

    for layer in inventory:
        fp16_bytes = layer["numel"] * 2
        total_fp16 += fp16_bytes

        if not layer["is_quantizable"]:
            total_quant += fp16_bytes
            continue

        name = layer["name"]

        if check_param_name_in(name, modules_to_not_convert):
            total_quant += fp16_bytes
            continue

        eff_dtype = get_effective_dtype(weights_dtype, name, modules_dtype_dict)
        layer_use_svd, layer_svd_rank = get_layer_svd(use_svd, svd_rank, name, modules_svd_dict)
        layer_gs = get_layer_group_size(name, modules_group_size_dict, group_size)
        layer_gs = resolve_group_size(
            layer_gs, eff_dtype, layer["shape"], layer_use_svd, layer["layer_class"]
        )

        layer_bytes = estimate_layer_bytes(
            numel=layer["numel"],
            shape=layer["shape"],
            dtype_str=eff_dtype,
            group_size=layer_gs,
            use_svd=layer_use_svd,
            svd_rank=layer_svd_rank,
        )
        total_quant += layer_bytes

    return total_quant, total_fp16


def estimate_pipeline_memory(all_inventories, pipeline_config):
    """Estimate total memory for all components in a pipeline config.

    Components not listed in the config stay at FP16 (unquantized).
    """
    total_quant = 0
    total_fp16 = 0

    for comp_name, inventory in all_inventories.items():
        if comp_name in pipeline_config:
            q, fp = estimate_component_memory(inventory, pipeline_config[comp_name])
        else:
            fp = sum(layer["numel"] * 2 for layer in inventory)
            q = fp
        total_quant += q
        total_fp16 += fp

    return total_quant, total_fp16


# ---------------------------------------------------------------------------
# Quality metrics from report CSVs
# ---------------------------------------------------------------------------

def read_quality_metrics(report_csv_path):
    """Read mean PSNR, SSIM, LPIPS from a report CSV."""
    result = {"mean_psnr": None, "mean_ssim": None, "mean_lpips": None}
    if not report_csv_path or not os.path.exists(report_csv_path):
        return result

    psnrs, ssims, lpipss = [], [], []
    with open(report_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, dest in [("psnr", psnrs), ("ssim", ssims), ("lpips", lpipss)]:
                if row.get(key):
                    try:
                        dest.append(float(row[key]))
                    except ValueError:
                        pass

    if psnrs:
        result["mean_psnr"] = sum(psnrs) / len(psnrs)
    if ssims:
        result["mean_ssim"] = sum(ssims) / len(ssims)
    if lpipss:
        result["mean_lpips"] = sum(lpipss) / len(lpipss)
    return result


# ---------------------------------------------------------------------------
# Test configuration definitions
# ---------------------------------------------------------------------------

# Ordered list: (test_num, label, config_json_filename, report_csv_filename)
ALL_TESTS = [
    # Phase 1 (1-10)
    (1,  "int8 (no skip keys)",           "ref2_int8.json",                     "ref2_int8_report.csv"),
    (2,  "uint8 (no skip keys)",          "ref2_uint8.json",                    "ref2_uint8_report.csv"),
    (3,  "uint4 (no skip keys)",          "ref2_uint4.json",                    "ref2_uint4_report.csv"),
    (4,  "uint4 + skip adaLN",            "ref2_uint4_skip_adaln.json",         "ref2_uint4_skip_adaln_report.csv"),
    (5,  "uint4 WITH skip keys",          "ref2_uint4_with_skipkeys.json",      "ref2_uint4_skipkeys_report.csv"),
    (6,  "int8 WITH skip keys",           "ref2_int8_with_skipkeys.json",       "ref2_int8_skipkeys_report.csv"),
    (7,  "ONLY adaLN quantized",          "ref2_only_adaln.json",               "ref2_only_adaln_report.csv"),
    (8,  "ONLY attention quantized",      "ref2_only_attn.json",                "ref2_only_attn_report.csv"),
    (9,  "ONLY MLP quantized",            "ref2_only_mlp.json",                 "ref2_only_mlp_report.csv"),
    (10, "ONLY embed+final quantized",    "ref2_only_embed.json",               "ref2_only_embed_report.csv"),
    # Phase 2 (11-18)
    (11, "int8 transformer only",         "ref2_int8_transformer_only.json",    "ref2_int8_transformer_only_report.csv"),
    (12, "int8 text encoder only",        "ref2_int8_textenc_only.json",        "ref2_int8_textenc_only_report.csv"),
    (13, "int8 + skip adaLN",             "ref2_int8_skip_adaln.json",          "ref2_int8_skip_adaln_report.csv"),
    (14, "int8 + skip all sensitive",     "ref2_int8_skip_all_sensitive.json",   "ref2_int8_skip_all_sensitive_report.csv"),
    (15, "int8 + skip adaLN+attn",        "ref2_int8_skip_adaln_attn.json",     "ref2_int8_skip_adaln_attn_report.csv"),
    (16, "uint8 + skip all sensitive",    "ref2_uint8_skip_all_sensitive.json",  "ref2_uint8_skip_all_sensitive_report.csv"),
    (17, "uint8 + skip adaLN+attn",       "ref2_uint8_skip_adaln_attn.json",    "ref2_uint8_skip_adaln_attn_report.csv"),
    (18, "int8 auto-config",              "ref2_int8_autoconfig.json",           "ref2_int8_autoconfig_report.csv"),
    # Phase 3 (19-25)
    (19, "uint4 MLP-only",                "ref2_uint4_mlp_only.json",            "ref2_uint4_mlp_only_report.csv"),
    (20, "uint4 MLP-only + SVD",          "ref2_uint4_mlp_svd.json",             "ref2_uint4_mlp_svd_report.csv"),
    (21, "uint4 MLP+SVD+stoch+fp32",      "ref2_uint4_mlp_svd_stoch_fp32.json",  "ref2_uint4_mlp_svd_stoch_fp32_report.csv"),
    (22, "uint4 hybrid u6attn",           "ref2_uint4_hybrid_u6attn.json",       "ref2_uint4_hybrid_u6attn_report.csv"),
    (23, "uint4 hybrid u8attn+SVD",       "ref2_uint4_hybrid_u8attn_svd.json",   "ref2_uint4_hybrid_u8attn_svd_report.csv"),
    (24, "uint4 kitchen sink",            "ref2_uint4_kitchen_sink.json",        "ref2_uint4_kitchen_sink_report.csv"),
    (25, "uint4 auto-config enhanced",    "ref2_uint4_autoconfig_enhanced.json", "ref2_uint4_autoconfig_enhanced_report.csv"),
]


def get_test_configs(base_dir, filter_configs=None):
    """Build list of test config dicts, optionally filtered to specific JSON files."""
    configs = []
    for test_num, label, config_json, report_csv in ALL_TESTS:
        config_path = os.path.join(base_dir, config_json)
        report_path = os.path.join(base_dir, report_csv)

        if filter_configs and config_json not in filter_configs and config_path not in filter_configs:
            continue

        if not os.path.exists(config_path):
            print(f"  Warning: config not found: {config_json}")
            continue

        with open(config_path, encoding="utf-8") as f:
            pipeline_config = json.load(f)

        configs.append({
            "test_num": test_num,
            "label": label,
            "config_json": config_json,
            "report_csv": report_path,
            "pipeline_config": pipeline_config,
        })

    return configs


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def fmt_bytes(b):
    """Format bytes as human-readable string."""
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024:.0f} KB"


def output_rich_table(results, fp16_total):
    """Print results using rich tables."""
    console = Console()
    table = Table(title="Memory Footprint Comparison", show_lines=True)

    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Config", min_width=30)
    table.add_column("Quant Size", justify="right", min_width=10)
    table.add_column("FP16 Size", justify="right", min_width=10, style="dim")
    table.add_column("Savings", justify="right", min_width=8)
    table.add_column("PSNR", justify="right", min_width=7)
    table.add_column("SSIM", justify="right", min_width=7)
    table.add_column("LPIPS", justify="right", min_width=7)
    table.add_column("PSNR/GB", justify="right", min_width=8)

    for r in results:
        savings = (1 - r["quant_bytes"] / r["fp16_bytes"]) * 100 if r["fp16_bytes"] > 0 else 0
        quant_gb = r["quant_bytes"] / (1024**3)
        psnr_str = f"{r['mean_psnr']:.2f}" if r["mean_psnr"] is not None else "-"
        ssim_str = f"{r['mean_ssim']:.4f}" if r["mean_ssim"] is not None else "-"
        lpips_str = f"{r['mean_lpips']:.4f}" if r["mean_lpips"] is not None else "-"
        psnr_per_gb = (r["mean_psnr"] / quant_gb) if r["mean_psnr"] is not None and quant_gb > 0 else None
        psnr_gb_str = f"{psnr_per_gb:.2f}" if psnr_per_gb is not None else "-"

        # Color savings by magnitude
        if savings > 40:
            sav_style = "bold green"
        elif savings > 20:
            sav_style = "green"
        elif savings > 0:
            sav_style = "yellow"
        else:
            sav_style = "red"

        table.add_row(
            str(r["test_num"]),
            r["label"],
            fmt_bytes(r["quant_bytes"]),
            fmt_bytes(r["fp16_bytes"]),
            Text(f"{savings:.1f}%", style=sav_style),
            psnr_str,
            ssim_str,
            lpips_str,
            psnr_gb_str,
        )

    console.print(table)
    console.print(f"\nFP16 baseline total: {fmt_bytes(fp16_total)}")


def output_plain_table(results, fp16_total):
    """Print results as plain text table."""
    header = f"{'#':>3}  {'Config':<34} {'Quant':>10} {'FP16':>10} {'Save%':>7} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'PSNR/GB':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        savings = (1 - r["quant_bytes"] / r["fp16_bytes"]) * 100 if r["fp16_bytes"] > 0 else 0
        quant_gb = r["quant_bytes"] / (1024**3)
        psnr_str = f"{r['mean_psnr']:.2f}" if r["mean_psnr"] is not None else "-"
        ssim_str = f"{r['mean_ssim']:.4f}" if r["mean_ssim"] is not None else "-"
        lpips_str = f"{r['mean_lpips']:.4f}" if r["mean_lpips"] is not None else "-"
        psnr_per_gb = (r["mean_psnr"] / quant_gb) if r["mean_psnr"] is not None and quant_gb > 0 else None
        psnr_gb_str = f"{psnr_per_gb:.2f}" if psnr_per_gb is not None else "-"

        print(f"{r['test_num']:>3}  {r['label']:<34} "
              f"{fmt_bytes(r['quant_bytes']):>10} {fmt_bytes(r['fp16_bytes']):>10} "
              f"{savings:>6.1f}% {psnr_str:>7} {ssim_str:>7} {lpips_str:>7} {psnr_gb_str:>8}")

    print(sep)
    print(f"FP16 baseline total: {fmt_bytes(fp16_total)}")


def output_csv(results, output_path):
    """Write results to CSV."""
    fieldnames = [
        "test_num", "label", "config_json",
        "quant_bytes", "fp16_bytes", "savings_pct",
        "mean_psnr", "mean_ssim", "mean_lpips", "psnr_per_gb",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            savings = (1 - r["quant_bytes"] / r["fp16_bytes"]) * 100 if r["fp16_bytes"] > 0 else 0
            quant_gb = r["quant_bytes"] / (1024**3)
            psnr_per_gb = (r["mean_psnr"] / quant_gb) if r["mean_psnr"] is not None and quant_gb > 0 else None
            writer.writerow({
                "test_num": r["test_num"],
                "label": r["label"],
                "config_json": r["config_json"],
                "quant_bytes": r["quant_bytes"],
                "fp16_bytes": r["fp16_bytes"],
                "savings_pct": round(savings, 2),
                "mean_psnr": round(r["mean_psnr"], 4) if r["mean_psnr"] is not None else "",
                "mean_ssim": round(r["mean_ssim"], 4) if r["mean_ssim"] is not None else "",
                "mean_lpips": round(r["mean_lpips"], 4) if r["mean_lpips"] is not None else "",
                "psnr_per_gb": round(psnr_per_gb, 4) if psnr_per_gb is not None else "",
            })
    print(f"\nCSV written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Estimate memory footprint for SDNQ quantization configs"
    )
    p.add_argument("--model-id", default="Tongyi-MAI/Z-Image",
                    help="HuggingFace model ID (default: Tongyi-MAI/Z-Image)")
    p.add_argument("--cache-dir", default=None,
                    help="HuggingFace cache directory")
    p.add_argument("--config-dir", default=None,
                    help="Directory containing config JSONs and report CSVs "
                         "(default: repo root)")
    p.add_argument("--configs", nargs="*", default=None,
                    help="Specific config JSON filenames to process "
                         "(default: all 25 tests)")
    p.add_argument("--output", default=None,
                    help="Output CSV path (default: memory_comparison.csv in config-dir)")
    return p.parse_args()


def main():
    args = parse_args()

    # Determine base directory
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    base_dir = args.config_dir or str(repo_root)

    print(f"Model: {args.model_id}")
    print(f"Config directory: {base_dir}")
    print()

    # Step 1: Load model inventories on meta device
    print("Loading model structure on meta device...")
    inventories = load_all_inventories(args.model_id, cache_dir=args.cache_dir)
    if not inventories:
        print("Error: no model components loaded.")
        sys.exit(1)

    fp16_total = sum(
        sum(layer["numel"] * 2 for layer in inv)
        for inv in inventories.values()
    )
    print(f"\nTotal FP16 baseline: {fmt_bytes(fp16_total)}")
    print()

    # Step 2: Load test configs
    print("Loading test configurations...")
    configs = get_test_configs(base_dir, filter_configs=args.configs)
    if not configs:
        print("Error: no test configurations found.")
        sys.exit(1)
    print(f"  Found {len(configs)} configs\n")

    # Step 3: Estimate memory for each config
    print("Estimating memory footprints...")
    results = []
    for cfg in configs:
        quant_bytes, fp16_bytes = estimate_pipeline_memory(
            inventories, cfg["pipeline_config"]
        )
        quality = read_quality_metrics(cfg["report_csv"])
        results.append({
            "test_num": cfg["test_num"],
            "label": cfg["label"],
            "config_json": cfg["config_json"],
            "quant_bytes": quant_bytes,
            "fp16_bytes": fp16_bytes,
            **quality,
        })

    # Step 4: Output results
    print()
    if HAS_RICH:
        output_rich_table(results, fp16_total)
    else:
        output_plain_table(results, fp16_total)

    # Step 5: Write CSV
    csv_path = args.output or os.path.join(base_dir, "memory_comparison.csv")
    output_csv(results, csv_path)


if __name__ == "__main__":
    main()
