#!/usr/bin/env python3
"""Per-layer quantization sensitivity analysis for diffusers models.

Loads a model component, quantizes each layer individually to target dtypes,
measures reconstruction error, and outputs a ranked report identifying
sensitive layers with copy-pasteable SDNQConfig snippets.

Example usage:
    python scripts/analyze_quantization_sensitivity.py \
        --model-id Tongyi-MAI/Z-Image \
        --components transformer:ZImageTransformer2DModel \
        --dtypes uint4 uint6 uint8 int8 \
        --output quant_sensitivity_report.csv
"""

import argparse
import csv
import importlib
import math
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import weasyprint
    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False


def parse_args():
    p = argparse.ArgumentParser(description="Per-layer quantization sensitivity analysis")
    p.add_argument("--model-id", default="Tongyi-MAI/Z-Image",
                    help="HuggingFace model ID or local path")
    p.add_argument("--components", nargs="+", default=None,
                    help="subfolder:ClassName pairs to analyze. "
                         "If omitted, auto-discovered from model_index.json.")
    p.add_argument("--cache-dir", default="/home/ohiom/database/models/huggingface",
                    help="HuggingFace cache directory")
    p.add_argument("--output", default="quant_sensitivity_report.csv",
                    help="CSV output path")
    p.add_argument("--dtypes", nargs="+", default=["uint4", "uint6", "uint8", "int8"],
                    help="Quantization dtypes to test")
    p.add_argument("--group-size", type=int, nargs="+", default=[0],
                    help="Group size(s) to test (0=auto). Multiple values enables group-size comparison mode.")
    p.add_argument("--svd-rank", type=int, nargs="+", default=[32],
                    help="SVD rank(s) to test. Multiple values enables rank comparison in auto-config.")
    p.add_argument("--svd-steps", type=int, default=8,
                    help="SVD iteration steps")
    p.add_argument("--no-svd", action="store_true",
                    help="Disable SVD testing")
    p.add_argument("--no-raw", action="store_true",
                    help="Disable raw (no-SVD) testing")
    p.add_argument("--skip-threshold", type=float, default=0.1,
                    help="NMSE above which layer should be skipped entirely")
    p.add_argument("--promote-threshold", type=float, default=0.01,
                    help="NMSE above which layer needs higher precision")
    p.add_argument("--top-n", type=int, default=0,
                    help="Number of worst layers to show in summary (0=all)")
    p.add_argument("--device", default="cpu",
                    help="Computation device")
    p.add_argument("--report-format", choices=["html", "html-png", "pdf"], default="html-png",
                    help="Report output format (default: html-png)")
    p.add_argument("--report-output", default=None,
                    help="Report path (default: <output-stem>_report.html/.pdf)")
    p.add_argument("--auto-config", action="store_true",
                    help="Grid search: find cheapest config per layer under promote-threshold")
    p.add_argument("--pipeline-config-output", default=None,
                    help="JSON output path for per-component pipeline config "
                         "(default: <output-stem>_pipeline_config.json). Only with --auto-config.")
    args = p.parse_args()
    # Normalize: --group-size stores into group_size as a list; provide both names
    args.group_sizes = args.group_size  # list
    args.group_size = args.group_sizes[0]  # scalar for single-value code paths
    # Normalize: --svd-rank stores into svd_rank as a list; provide both names
    args.svd_ranks = args.svd_rank  # list
    args.svd_rank = args.svd_ranks[0]  # scalar for single-value code paths
    return args


def discover_components(model_id, cache_dir):
    """Auto-discover quantizable components from model_index.json.

    Returns a list of 'subfolder:ClassName' strings, or None if model_index.json
    is not available (e.g. single-model repos).
    """
    # Components that have no quantizable weights
    skip_types = {
        "FlowMatchEulerDiscreteScheduler", "EulerDiscreteScheduler",
        "DDIMScheduler", "PNDMScheduler", "DPMSolverMultistepScheduler",
        "LMSDiscreteScheduler", "UniPCMultistepScheduler",
        "Qwen2Tokenizer", "CLIPTokenizer", "T5Tokenizer",
        "T5TokenizerFast", "CLIPTokenizerFast", "LlamaTokenizer",
        "LlamaTokenizerFast", "PreTrainedTokenizerFast",
        "CLIPImageProcessor", "CLIPFeatureExtractor",
    }
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            model_id, "model_index.json",
            cache_dir=cache_dir, local_files_only=True,
        )
    except Exception:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                model_id, "model_index.json",
                cache_dir=cache_dir,
            )
        except Exception:
            return None

    import json
    with open(path) as f:
        index = json.load(f)

    components = []
    for key, value in index.items():
        if not isinstance(value, list) or len(value) < 2:
            continue
        _pkg, class_name = value[0], value[1]
        if class_name in skip_types:
            continue
        components.append(f"{key}:{class_name}")

    return components if components else None


def resolve_class(class_name: str):
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


def compute_weight_stats(weight: torch.Tensor):
    """Compute per-layer statistics of the original weight (independent of quantization)."""
    w = weight.float()
    mean = w.mean()
    centered = w - mean
    var = centered.var()

    # Kurtosis: E[(x-mu)^4] / E[(x-mu)^2]^2
    if var.item() > 0:
        kurtosis = (centered.pow(4).mean() / var.square()).item()
    else:
        kurtosis = float("inf")

    # Outlier ratio: fraction of weights > 3 sigma from mean
    std = var.sqrt()
    if std.item() > 0:
        outlier_ratio = ((centered.abs() > 3 * std).float().mean()).item()
    else:
        outlier_ratio = 0.0

    # Dynamic range ratio: max(|w|) / median(|w|)
    abs_w = w.abs()
    median_val = abs_w.median().item()
    if median_val > 0:
        dynamic_range_ratio = (abs_w.max().item() / median_val)
    else:
        dynamic_range_ratio = float("inf")

    return {
        "kurtosis": kurtosis,
        "outlier_ratio": outlier_ratio,
        "dynamic_range_ratio": dynamic_range_ratio,
    }


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor):
    """Compute reconstruction error metrics between original and reconstructed weights."""
    orig_f32 = original.float()
    recon_f32 = reconstructed.float()
    error = orig_f32 - recon_f32

    mse = F.mse_loss(orig_f32, recon_f32)
    variance = orig_f32.std().square()
    nmse = (mse / variance).item() if variance.item() > 0 else float("inf")

    # SQNR in dB: -10 * log10(NMSE). Higher = better.
    if nmse > 0 and not math.isinf(nmse):
        sqnr_db = -10.0 * math.log10(nmse)
    else:
        sqnr_db = float("-inf") if nmse == float("inf") else float("inf")

    cos_sim = F.cosine_similarity(orig_f32.flatten().unsqueeze(0),
                                   recon_f32.flatten().unsqueeze(0)).item()
    max_abs_err = error.abs().max().item()
    orig_norm = orig_f32.norm(2).item()
    relative_l2 = (error.norm(2).item() / orig_norm) if orig_norm > 0 else float("inf")

    return {
        "nmse": nmse,
        "sqnr_db": sqnr_db,
        "cosine_sim": cos_sim,
        "max_abs_err": max_abs_err,
        "relative_l2": relative_l2,
    }


def analyze_component(model_id, subfolder, class_name, cache_dir, args):
    """Load and analyze a single model component. Returns list of result dicts."""
    from sdnq.common import allowed_types
    from sdnq.quantizer import add_module_skip_keys, sdnq_quantize_layer_weight

    cls = resolve_class(class_name)
    print(f"\nLoading {class_name} from {model_id} (subfolder={subfolder})...")
    model = cls.from_pretrained(model_id, subfolder=subfolder,
                                torch_dtype=torch.bfloat16, cache_dir=cache_dir)
    model.eval()
    model = model.to(args.device)

    # Log skip keys for reference but don't filter — sensitivity analysis tests all layers
    _, skip_keys, _dtype_dict_override = add_module_skip_keys(model)
    skip_keys = skip_keys or []
    print(f"  Skip keys ({len(skip_keys)}, ignored for analysis): {skip_keys[:10]}{'...' if len(skip_keys) > 10 else ''}")

    # Collect quantizable layers (skip keys are intentionally not filtered)
    layers = []
    for name, module in model.named_modules():
        if type(module).__name__ not in allowed_types:
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if not module.weight.is_floating_point():
            continue
        param_name = f"{name}.weight"
        layers.append((param_name, name, module))

    print(f"  Found {len(layers)} quantizable layers")

    # Determine SVD configs to test: list of (use_svd, svd_rank) tuples
    svd_configs = []
    if not args.no_raw:
        svd_configs.append((False, args.svd_ranks[0]))  # raw: rank is irrelevant but needed for the tuple
    if not args.no_svd:
        svd_configs.extend((True, rank) for rank in args.svd_ranks)
    if not svd_configs:
        print("  ERROR: Both --no-svd and --no-raw specified, nothing to test")
        return []

    results = []
    total = len(layers) * len(args.dtypes) * len(svd_configs) * len(args.group_sizes)
    pbar = tqdm(total=total, desc=f"  Analyzing {class_name}", unit="layer-dtype")

    for param_name, _module_name, module in layers:
        original_weight = module.weight.data.float()
        layer_class = type(module).__name__
        shape = tuple(original_weight.shape)
        numel = original_weight.numel()

        # Compute weight stats once per layer
        weight_stats = compute_weight_stats(original_weight)

        for dtype_str in args.dtypes:
            for use_svd, svd_rank in svd_configs:
                for gs in args.group_sizes:
                    pbar.set_postfix_str(f"{param_name} {dtype_str} svd={use_svd} rank={svd_rank} gs={gs}", refresh=False)
                    try:
                        qw, scale, zp, svd_up, svd_down, dequantizer = sdnq_quantize_layer_weight(
                            weight=module.weight.data.clone(),
                            layer_class_name=layer_class,
                            weights_dtype=dtype_str,
                            group_size=gs,
                            svd_rank=svd_rank,
                            svd_steps=args.svd_steps,
                            use_svd=use_svd,
                            use_quantized_matmul=False,
                            dequantize_fp32=True,
                            param_name=param_name,
                        )
                        reconstructed = dequantizer(
                            qw, scale, zp, svd_up, svd_down,
                            skip_quantized_matmul=False,
                            dtype=torch.float32,
                            skip_compile=True,
                        )
                        metrics = compute_metrics(original_weight, reconstructed)
                        actual_group_size = dequantizer.group_size

                        # Clean up intermediates
                        del qw, scale, zp, svd_up, svd_down, dequantizer, reconstructed

                        results.append({
                            "component": f"{subfolder}:{class_name}",
                            "param_name": param_name,
                            "layer_class": layer_class,
                            "shape": str(shape),
                            "numel": numel,
                            "weights_dtype": dtype_str,
                            "use_svd": use_svd,
                            "svd_rank": svd_rank,
                            "group_size": actual_group_size,
                            "requested_group_size": gs,
                            **metrics,
                            **weight_stats,
                        })
                    except Exception as e:
                        print(f"\n  WARN: Failed {param_name} dtype={dtype_str} svd={use_svd} rank={svd_rank} gs={gs}: {e}")
                        traceback.print_exc()
                        results.append({
                            "component": f"{subfolder}:{class_name}",
                            "param_name": param_name,
                            "layer_class": layer_class,
                            "shape": str(shape),
                            "numel": numel,
                            "weights_dtype": dtype_str,
                            "use_svd": use_svd,
                            "svd_rank": svd_rank,
                            "group_size": -1,
                            "requested_group_size": gs,
                            "nmse": float("inf"),
                            "sqnr_db": float("-inf"),
                            "cosine_sim": 0.0,
                            "max_abs_err": float("inf"),
                            "relative_l2": float("inf"),
                            **weight_stats,
                        })
                    pbar.update(1)

    pbar.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def write_csv(results, output_path):
    """Write results to CSV."""
    if not results:
        print("No results to write.")
        return
    fieldnames = ["component", "param_name", "layer_class", "shape", "numel",
                  "weights_dtype", "use_svd", "svd_rank", "group_size", "requested_group_size",
                  "nmse", "sqnr_db", "cosine_sim", "max_abs_err", "relative_l2",
                  "kurtosis", "outlier_ratio", "dynamic_range_ratio"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV written to {output_path} ({len(results)} rows)")


def nmse_to_css_color(nmse, skip_thresh, promote_thresh):
    """Map NMSE value to (background_color, text_color) CSS pair."""
    if math.isinf(nmse):
        return ("#d62828", "#ffffff")
    if nmse <= promote_thresh * 0.1:
        return ("#2d6a4f", "#ffffff")
    if nmse <= promote_thresh:
        return ("#40916c", "#ffffff")
    if nmse <= skip_thresh * 0.5:
        return ("#b8860b", "#1a1a2e")
    if nmse <= skip_thresh:
        return ("#e76f51", "#1a1a2e")
    return ("#d62828", "#ffffff")


def nmse_to_chart_color(nmse, skip_thresh, promote_thresh):
    """Map NMSE value to a single matplotlib color."""
    if math.isinf(nmse):
        return "#d62828"
    if nmse <= promote_thresh * 0.1:
        return "#2d6a4f"
    if nmse <= promote_thresh:
        return "#40916c"
    if nmse <= skip_thresh * 0.5:
        return "#e9c46a"
    if nmse <= skip_thresh:
        return "#e76f51"
    return "#d62828"


def _build_auto_config_snippet(ok_layers_info, skip_layers, args, component=None):
    """Build an SDNQConfig snippet for auto-config mode.

    Args:
        ok_layers_info: list of (param_name, rec_dtype, rec_group_size, rec_svd, rec_svd_rank)
            for OK layers.
        skip_layers: list of param_name strings for SKIP layers.
        args: parsed CLI args.
        component: optional "subfolder:ClassName" string for per-component snippets.
    """
    from collections import Counter

    skip_layers_unique = sorted(set(skip_layers))

    if not ok_layers_info:
        lines = []
        if component:
            subfolder = component.split(":")[0]
            class_name = component.split(":", 1)[1] if ":" in component else component
            lines.append(f"# Component: {subfolder} ({class_name})")
        lines.append("# All layers failed — no quantization recommended")
        lines.append("modules_to_not_convert = [")
        lines.extend(f'    "{layer}",' for layer in skip_layers_unique)
        lines.append("]")
        return "\n".join(lines)

    # Find most common dtype and group_size across OK layers
    dtype_counter = Counter(info[1] for info in ok_layers_info)
    gs_counter = Counter(info[2] for info in ok_layers_info)
    base_dtype = dtype_counter.most_common(1)[0][0]
    base_gs = gs_counter.most_common(1)[0][0]

    lines = []
    if component:
        subfolder = component.split(":")[0]
        class_name = component.split(":", 1)[1] if ":" in component else component
        lines.append(f"# Component: {subfolder} ({class_name})")
    lines.append("# Layers to exclude from quantization entirely")
    lines.append("modules_to_not_convert = [")
    lines.extend(f'    "{layer}",' for layer in skip_layers_unique)
    lines.append("]")
    lines.append("")

    # Collect layers with non-base dtypes and non-base group sizes
    dtype_overrides = {}
    gs_overrides = {}
    for pname, rec_dtype, rec_gs, _rec_svd, _rec_svd_rank in ok_layers_info:
        if rec_dtype != base_dtype:
            dtype_overrides.setdefault(rec_dtype, []).append(pname)
        if rec_gs != base_gs:
            gs_overrides.setdefault(rec_gs, []).append(pname)

    if dtype_overrides:
        lines.append("# Layers that need a different dtype than the base")
        lines.append("modules_dtype_dict = {")
        for dtype, layer_list in sorted(dtype_overrides.items()):
            lines.append(f'    "{dtype}": [')
            for layer in sorted(layer_list):
                lines.append(f'        "{layer}",')
            lines.append("    ],")
        lines.append("}")
        lines.append("")

    # Collect layers that use SVD, grouped by per-layer rank
    svd_overrides = {}
    all_use_svd = all(info[3] for info in ok_layers_info)
    svd_ranks_used = set(info[4] for info in ok_layers_info if info[3])
    single_rank = len(svd_ranks_used) == 1
    for pname, _rec_dtype, _rec_gs, rec_svd, rec_svd_rank in ok_layers_info:
        if rec_svd:
            svd_overrides.setdefault(rec_svd_rank, []).append(pname)

    # Emit global use_svd only when all layers use SVD with the same rank
    emit_global_svd = all_use_svd and single_rank
    if gs_overrides:
        lines.append("# Layers that need a different group_size than the base")
        lines.append("modules_group_size_dict = {")
        for gs, layer_list in sorted(gs_overrides.items()):
            lines.append(f"    {gs}: [")
            for layer in sorted(layer_list):
                lines.append(f'        "{layer}",')
            lines.append("    ],")
        lines.append("}")
        lines.append("")

    if svd_overrides and not emit_global_svd:
        lines.append("# Layers that benefit from SVD compression (per-layer)")
        lines.append("modules_svd_dict = {")
        for rank, layer_list in sorted(svd_overrides.items()):
            lines.append(f"    {rank}: [")
            for layer in sorted(layer_list):
                lines.append(f'        "{layer}",')
            lines.append("    ],")
        lines.append("}")
        lines.append("")

    lines.append("# Usage:")
    lines.append("# from sdnq import SDNQConfig")
    lines.append(f'# config = SDNQConfig(weights_dtype="{base_dtype}",')
    lines.append(f"#                     group_size={base_gs},")
    if emit_global_svd:
        global_rank = next(iter(svd_ranks_used))
        lines.append(f"#                     use_svd=True, svd_rank={global_rank},")
    lines.append("#                     modules_to_not_convert=modules_to_not_convert,")
    if dtype_overrides:
        lines.append("#                     modules_dtype_dict=modules_dtype_dict,")
    if gs_overrides:
        lines.append("#                     modules_group_size_dict=modules_group_size_dict,")
    if svd_overrides and not emit_global_svd:
        lines.append("#                     modules_svd_dict=modules_svd_dict,")
    lines.append("#                     )")

    return "\n".join(lines)


def _classify_layers_for_auto_config(layers, args):
    """Classify layers for auto-config: find cheapest passing config per layer.

    Args:
        layers: dict of {param_name: {layer_class, shape, numel, results}}.
        args: parsed CLI args.

    Returns:
        (classified_rows, skip_layers, ok_layers_info) tuple.
    """
    import ast

    classified_rows = []
    skip_layers = []
    ok_layers_info = []  # (param_name, rec_dtype, rec_group_size, rec_svd, rec_svd_rank)

    for pname, layer_info in layers.items():
        shape_str = layer_info["shape"]
        try:
            shape = ast.literal_eval(shape_str) if isinstance(shape_str, str) else shape_str
        except (ValueError, SyntaxError):
            shape = ()
        numel = layer_info["numel"]
        fp16_bytes = numel * 2

        passing = []
        for r in layer_info["results"]:
            if r["nmse"] <= args.promote_threshold and not math.isinf(r["nmse"]):
                cost = estimate_layer_bytes(
                    numel=numel,
                    shape=shape,
                    dtype_str=r["weights_dtype"],
                    group_size=r["group_size"],
                    use_svd=r["use_svd"],
                    svd_rank=r["svd_rank"],
                )
                passing.append((r, cost))

        if not passing:
            skip_layers.append(pname)
            bg, fg = nmse_to_css_color(float("inf"), args.skip_threshold, args.promote_threshold)
            classified_rows.append({
                "param_name": pname,
                "truncated_name": truncate_layer_name(pname),
                "layer_class": layer_info["layer_class"],
                "shape": shape_str,
                "numel": numel,
                "rec_dtype": "fp16",
                "rec_group_size": "-",
                "rec_svd": False,
                "rec_svd_rank": "-",
                "rec_nmse": float("inf"),
                "rec_nmse_display": "N/A",
                "rec_size_bytes": fp16_bytes,
                "rec_size_display": _fmt_bytes(fp16_bytes),
                "fp16_size_bytes": fp16_bytes,
                "nmse_bg": bg,
                "nmse_fg": fg,
                "status": "SKIP",
                "component": layer_info.get("component", ""),
            })
        else:
            passing.sort(key=lambda x: (
                x[1],
                x[0]["use_svd"],
                -x[0]["group_size"],
                x[0]["weights_dtype"],
            ))
            best_r, best_cost = passing[0]
            bg, fg = nmse_to_css_color(best_r["nmse"], args.skip_threshold, args.promote_threshold)
            classified_rows.append({
                "param_name": pname,
                "truncated_name": truncate_layer_name(pname),
                "layer_class": layer_info["layer_class"],
                "shape": shape_str,
                "numel": numel,
                "rec_dtype": best_r["weights_dtype"],
                "rec_group_size": best_r["group_size"],
                "rec_svd": best_r["use_svd"],
                "rec_svd_rank": best_r["svd_rank"],
                "rec_nmse": best_r["nmse"],
                "rec_nmse_display": fmt_nmse(best_r["nmse"]),
                "rec_size_bytes": best_cost,
                "rec_size_display": _fmt_bytes(best_cost),
                "fp16_size_bytes": fp16_bytes,
                "nmse_bg": bg,
                "nmse_fg": fg,
                "status": "OK",
                "component": layer_info.get("component", ""),
            })
            ok_layers_info.append((pname, best_r["weights_dtype"], best_r["group_size"], best_r["use_svd"], best_r["svd_rank"]))

    return classified_rows, skip_layers, ok_layers_info


def _build_pipeline_config_entry(ok_layers_info, skip_layers):
    """Build a pipeline config dict entry (SDNQConfig kwargs) for one component.

    Returns a dict suitable for JSON serialization and later use as SDNQConfig(**entry).
    """
    from collections import Counter

    if not ok_layers_info:
        return {
            "weights_dtype": "int8",
            "group_size": 0,
            "modules_to_not_convert": sorted(set(skip_layers)),
        }

    dtype_counter = Counter(info[1] for info in ok_layers_info)
    gs_counter = Counter(info[2] for info in ok_layers_info)
    base_dtype = dtype_counter.most_common(1)[0][0]
    base_gs = gs_counter.most_common(1)[0][0]

    entry = {
        "weights_dtype": base_dtype,
        "group_size": base_gs,
    }

    skip_unique = sorted(set(skip_layers))
    if skip_unique:
        entry["modules_to_not_convert"] = skip_unique

    # Dtype overrides
    dtype_overrides = {}
    for pname, rec_dtype, _rec_gs, _rec_svd, _rec_svd_rank in ok_layers_info:
        if rec_dtype != base_dtype:
            dtype_overrides.setdefault(rec_dtype, []).append(pname)
    if dtype_overrides:
        entry["modules_dtype_dict"] = {k: sorted(v) for k, v in sorted(dtype_overrides.items())}

    # Group size overrides
    gs_overrides = {}
    for pname, _rec_dtype, rec_gs, _rec_svd, _rec_svd_rank in ok_layers_info:
        if rec_gs != base_gs:
            gs_overrides.setdefault(rec_gs, []).append(pname)
    if gs_overrides:
        entry["modules_group_size_dict"] = {k: sorted(v) for k, v in sorted(gs_overrides.items())}

    # SVD overrides
    svd_overrides = {}
    all_use_svd = all(info[3] for info in ok_layers_info)
    svd_ranks_used = set(info[4] for info in ok_layers_info if info[3])
    single_rank = len(svd_ranks_used) == 1
    for pname, _rec_dtype, _rec_gs, rec_svd, rec_svd_rank in ok_layers_info:
        if rec_svd:
            svd_overrides.setdefault(rec_svd_rank, []).append(pname)

    if all_use_svd and single_rank:
        entry["use_svd"] = True
        entry["svd_rank"] = next(iter(svd_ranks_used))
    elif svd_overrides:
        entry["modules_svd_dict"] = {k: sorted(v) for k, v in sorted(svd_overrides.items())}

    return entry


def _prepare_auto_config_data(results, args):
    """Prepare auto-config summary: grid search for cheapest passing config per layer."""
    import ast
    from collections import Counter

    # Group results by component, then by param_name
    by_component = {}
    for r in results:
        comp = r["component"]
        pname = r["param_name"]
        by_component.setdefault(comp, {}).setdefault(pname, {
            "layer_class": r["layer_class"],
            "shape": r["shape"],
            "numel": r["numel"],
            "component": comp,
            "results": [],
        })["results"].append(r)

    # Process each component independently
    all_classified_rows = []
    all_skip_layers = []
    all_ok_layers_info = []
    per_component_ok = {}  # comp -> ok_layers_info
    per_component_skip = {}  # comp -> skip_layers
    config_snippets = {}  # comp -> snippet string
    pipeline_config_json = {}  # subfolder -> SDNQConfig kwargs

    for comp, layers in sorted(by_component.items()):
        comp_rows, comp_skip, comp_ok = _classify_layers_for_auto_config(layers, args)
        all_classified_rows.extend(comp_rows)
        all_skip_layers.extend(comp_skip)
        all_ok_layers_info.extend(comp_ok)
        per_component_ok[comp] = comp_ok
        per_component_skip[comp] = comp_skip

        config_snippets[comp] = _build_auto_config_snippet(comp_ok, comp_skip, args, component=comp)

        subfolder = comp.split(":")[0]
        pipeline_config_json[subfolder] = _build_pipeline_config_entry(comp_ok, comp_skip)

    # Sort: SKIP layers first (worst cases up top), then by descending cost
    all_classified_rows.sort(key=lambda r: (
        0 if r["status"] == "SKIP" else 1,
        -r["rec_size_bytes"],
    ))

    # Add rank numbers
    for i, row in enumerate(all_classified_rows, 1):
        row["rank"] = i

    # Aggregate stats
    total_layers = len(all_classified_rows)
    total_params = sum(r["numel"] for r in all_classified_rows)
    ok_count = sum(1 for r in all_classified_rows if r["status"] == "OK")
    skip_count = sum(1 for r in all_classified_rows if r["status"] == "SKIP")

    fp16_total = sum(r["fp16_size_bytes"] for r in all_classified_rows)
    recommended_total = sum(r["rec_size_bytes"] for r in all_classified_rows)

    # Uniform total: first dtype + first group_size + no SVD for all layers
    uniform_dtype = args.dtypes[0]
    uniform_gs = args.group_sizes[0]
    uniform_total = 0
    for row in all_classified_rows:
        try:
            shape = ast.literal_eval(row["shape"]) if isinstance(row["shape"], str) else row["shape"]
        except (ValueError, SyntaxError):
            shape = ()
        eff_gs = uniform_gs if uniform_gs > 0 else 32
        uniform_total += estimate_layer_bytes(
            numel=row["numel"], shape=shape, dtype_str=uniform_dtype,
            group_size=eff_gs, use_svd=False, svd_rank=args.svd_rank,
        )

    savings_vs_fp16_pct = (1 - recommended_total / fp16_total) * 100 if fp16_total > 0 else 0
    savings_vs_uniform_pct = (1 - recommended_total / uniform_total) * 100 if uniform_total > 0 else 0

    # NMSE stats for OK layers
    ok_nmse_values = [r["rec_nmse"] for r in all_classified_rows
                      if r["status"] == "OK" and not math.isinf(r["rec_nmse"])]
    if ok_nmse_values:
        mean_nmse = sum(ok_nmse_values) / len(ok_nmse_values)
        sorted_nmse = sorted(ok_nmse_values)
        median_nmse = sorted_nmse[len(sorted_nmse) // 2]
        worst_nmse = sorted_nmse[-1]
    else:
        mean_nmse = median_nmse = worst_nmse = float("inf")

    ok_pct = ok_count / total_layers * 100 if total_layers else 0
    skip_pct = skip_count / total_layers * 100 if total_layers else 0

    # SVD rank distribution across OK layers
    rank_dist_counter = Counter()
    for row in all_classified_rows:
        if row["status"] == "OK":
            if row["rec_svd"]:
                rank_dist_counter[str(row["rec_svd_rank"])] += 1
            else:
                rank_dist_counter["no-SVD"] += 1
    rank_distribution = []
    if "no-SVD" in rank_dist_counter:
        rank_distribution.append(("no-SVD", rank_dist_counter.pop("no-SVD")))
    rank_distribution.extend((k, rank_dist_counter[k]) for k in sorted(rank_dist_counter, key=int))

    # Per-component size stats
    per_component_size = {}
    for comp in sorted(by_component.keys()):
        subfolder = comp.split(":")[0]
        comp_rows = [r for r in all_classified_rows if r.get("component") == comp]
        comp_fp16 = sum(r["fp16_size_bytes"] for r in comp_rows)
        comp_rec = sum(r["rec_size_bytes"] for r in comp_rows)
        comp_savings = (1 - comp_rec / comp_fp16) * 100 if comp_fp16 > 0 else 0
        per_component_size[subfolder] = {
            "fp16_total_display": _fmt_bytes(comp_fp16),
            "recommended_total_display": _fmt_bytes(comp_rec),
            "savings_vs_fp16_pct": f"{comp_savings:.1f}",
            "layer_count": len(comp_rows),
            "ok_count": sum(1 for r in comp_rows if r["status"] == "OK"),
            "skip_count": sum(1 for r in comp_rows if r["status"] == "SKIP"),
        }

    size_stats = {
        "fp16_total": fp16_total,
        "fp16_total_display": _fmt_bytes(fp16_total),
        "uniform_total": uniform_total,
        "uniform_total_display": _fmt_bytes(uniform_total),
        "uniform_dtype": uniform_dtype,
        "uniform_gs": uniform_gs,
        "recommended_total": recommended_total,
        "recommended_total_display": _fmt_bytes(recommended_total),
        "savings_vs_fp16_pct": f"{savings_vs_fp16_pct:.1f}",
        "savings_vs_uniform_pct": f"{savings_vs_uniform_pct:.1f}",
        "per_component": per_component_size,
    }

    # Build combined config snippet with per-component sections
    if len(config_snippets) == 1:
        config_snippet = next(iter(config_snippets.values()))
    else:
        sections = [config_snippets[comp] for comp in sorted(config_snippets.keys())]
        config_snippet = "\n\n" + ("\n\n" + "# " + "=" * 60 + "\n\n").join(sections)

    return {
        "primary_dtype": args.dtypes[0],
        "use_svd_filter": not args.no_svd,
        "column_keys": [],
        "column_mode": "auto_config",
        "primary_results": [],
        "top_n_results": [],
        "nmse_lookup": {},
        "stats": {
            "total_layers": total_layers,
            "total_params": total_params,
            "ok_count": ok_count,
            "promote_count": 0,
            "skip_count": skip_count,
            "ok_pct": ok_pct,
            "promote_pct": 0,
            "skip_pct": skip_pct,
            "mean_nmse": fmt_nmse(mean_nmse),
            "median_nmse": fmt_nmse(median_nmse),
            "worst_nmse": fmt_nmse(worst_nmse),
        },
        "classified_rows": all_classified_rows,
        "skip_layers": sorted(set(all_skip_layers)),
        "promote_layers": {},
        "svd_comparison": [],
        "svd_summary": None,
        "config_snippet": config_snippet,
        "config_snippets": config_snippets,
        "size_stats": size_stats,
        "rank_distribution": rank_distribution,
        "pipeline_config_json": pipeline_config_json,
    }


def prepare_summary_data(results, args):
    """Extract structured summary data from results for use by terminal output and reports."""
    if not results:
        return None

    if getattr(args, "auto_config", False):
        return _prepare_auto_config_data(results, args)

    primary_dtype = args.dtypes[0]
    use_svd_filter = args.no_raw
    multi_gs = len(args.group_sizes) > 1
    primary_gs = args.group_sizes[0]

    # Determine column mode
    if multi_gs:
        column_keys = [f"g{gs}" for gs in args.group_sizes]
        column_mode = "group_size"
    else:
        column_keys = list(args.dtypes)
        column_mode = "dtype"

    # Primary results: used for ranking and statistics
    if multi_gs:
        primary_results = [r for r in results
                           if r["weights_dtype"] == primary_dtype
                           and r["use_svd"] == use_svd_filter
                           and r.get("requested_group_size", r["group_size"]) == primary_gs]
    else:
        primary_results = [r for r in results
                           if r["weights_dtype"] == primary_dtype
                           and r["use_svd"] == use_svd_filter
                           and r.get("requested_group_size", args.group_size) == args.group_size]
    if not primary_results:
        return None

    # Deduplicate by param_name (keep first occurrence)
    seen = set()
    deduped = []
    for r in primary_results:
        if r["param_name"] not in seen:
            seen.add(r["param_name"])
            deduped.append(r)
    primary_results = deduped

    # Sort by NMSE descending
    primary_results.sort(key=lambda r: r["nmse"], reverse=True)
    top_n = args.top_n if args.top_n > 0 else len(primary_results)
    top_n_results = primary_results[:top_n]

    # Build lookups: param_name -> {column_key -> nmse}
    nmse_lookup = {}
    for r in results:
        if r["use_svd"] != use_svd_filter:
            continue
        key = r["param_name"]
        if key not in nmse_lookup:
            nmse_lookup[key] = {}
        if multi_gs:
            # Only include the primary dtype, keyed by group size
            if r["weights_dtype"] == primary_dtype:
                gs_key = f"g{r.get('requested_group_size', r['group_size'])}"
                nmse_lookup[key][gs_key] = r["nmse"]
        else:
            # Only include the primary group size, keyed by dtype
            rgs = r.get("requested_group_size", args.group_size)
            if rgs == args.group_size:
                nmse_lookup[key][r["weights_dtype"]] = r["nmse"]

    # Classify layers
    def _classify_layer(pname):
        """Classify a layer based on column_keys (dtypes or group sizes)."""
        nmse_values = nmse_lookup.get(pname, {})
        primary_key = column_keys[0]
        primary_nmse = nmse_values.get(primary_key, float("inf"))

        all_above_skip = all(nmse_values.get(ck, float("inf")) > args.skip_threshold
                            for ck in column_keys)

        if all_above_skip:
            return "SKIP", None
        elif primary_nmse > args.promote_threshold:
            best_key = None
            for ck in column_keys:
                if nmse_values.get(ck, float("inf")) <= args.promote_threshold:
                    best_key = ck
                    break
            if best_key:
                return f">{best_key}", best_key
            else:
                return "SKIP", None
        else:
            return "OK", None

    # Statistics — computed over ALL primary layers
    total_layers = len(primary_results)
    total_params = sum(r["numel"] for r in primary_results)
    ok_count = 0
    promote_count = 0
    skip_count = 0
    for r in primary_results:
        status, _ = _classify_layer(r["param_name"])
        if status == "OK":
            ok_count += 1
        elif status == "SKIP":
            skip_count += 1
        else:
            promote_count += 1

    nmse_values_all = [r["nmse"] for r in primary_results if not math.isinf(r["nmse"])]
    if nmse_values_all:
        mean_nmse = sum(nmse_values_all) / len(nmse_values_all)
        sorted_nmse = sorted(nmse_values_all)
        median_nmse = sorted_nmse[len(sorted_nmse) // 2]
        worst_nmse = sorted_nmse[-1]
    else:
        mean_nmse = median_nmse = worst_nmse = float("inf")

    ok_pct = ok_count / total_layers * 100 if total_layers else 0
    promote_pct = promote_count / total_layers * 100 if total_layers else 0
    skip_pct = skip_count / total_layers * 100 if total_layers else 0

    # Classify each top-N row for the detail table
    skip_layers = []
    promote_layers = {}  # pname -> best column key
    classified_rows = []

    for i, r in enumerate(top_n_results, 1):
        pname = r["param_name"]
        status, best_key = _classify_layer(pname)

        if status == "SKIP":
            skip_layers.append(pname)
        elif best_key:
            promote_layers[pname] = best_key

        kurt_val = r.get("kurtosis", 0)
        if math.isinf(kurt_val):
            kurt_str = "INF"
        else:
            kurt_str = f"{kurt_val:.1f}"

        nmse_by_column = {}
        for ck in column_keys:
            val = nmse_lookup.get(pname, {}).get(ck, float("nan"))
            bg, fg = nmse_to_css_color(val, args.skip_threshold, args.promote_threshold)
            nmse_by_column[ck] = {
                "value": val,
                "display": fmt_nmse(val),
                "bg_color": bg,
                "text_color": fg,
            }

        classified_rows.append({
            "rank": i,
            "param_name": pname,
            "truncated_name": truncate_layer_name(pname),
            "layer_class": r["layer_class"],
            "shape": r["shape"],
            "kurtosis": kurt_str,
            "nmse_by_column": nmse_by_column,
            "status": status,
            "promote_key": promote_layers.get(pname),
        })

    # SVD comparison (only in single group-size mode)
    svd_comparison = []
    svd_summary = None
    has_svd = not args.no_svd
    has_raw = not args.no_raw
    if has_svd and has_raw and not multi_gs:
        svd_lookup = {}  # param_name -> best SVD result (lowest NMSE across ranks)
        raw_lookup = {}
        for r in results:
            if r["weights_dtype"] != primary_dtype:
                continue
            rgs = r.get("requested_group_size", args.group_size)
            if rgs != args.group_size:
                continue
            if r["use_svd"]:
                pname = r["param_name"]
                prev = svd_lookup.get(pname)
                if prev is None or r["nmse"] < prev["nmse"]:
                    svd_lookup[pname] = r
            else:
                raw_lookup[r["param_name"]] = r

        # Compute SVD vs Raw summary over ALL matched layers
        improved = 0
        harmed = 0
        unchanged = 0
        raw_nmse_all = []
        svd_nmse_all = []
        for pname, raw_r in raw_lookup.items():
            svd_r = svd_lookup.get(pname)
            if svd_r is None:
                continue
            rn = raw_r["nmse"]
            sn = svd_r["nmse"]
            if math.isinf(rn) or math.isinf(sn):
                continue
            raw_nmse_all.append(rn)
            svd_nmse_all.append(sn)
            if sn < rn:
                improved += 1
            elif sn > rn:
                harmed += 1
            else:
                unchanged += 1

        if raw_nmse_all:
            raw_mean = sum(raw_nmse_all) / len(raw_nmse_all)
            svd_mean = sum(svd_nmse_all) / len(svd_nmse_all)
            raw_sorted_all = sorted(raw_nmse_all)
            svd_sorted_all = sorted(svd_nmse_all)
            raw_median = raw_sorted_all[len(raw_sorted_all) // 2]
            svd_median = svd_sorted_all[len(svd_sorted_all) // 2]
            total_compared = improved + harmed + unchanged

            if improved > harmed * 2:
                verdict = "beneficial"
            elif harmed > improved * 2:
                verdict = "harmful"
            else:
                verdict = "mixed"

            svd_summary = {
                "total_compared": total_compared,
                "improved": improved,
                "harmed": harmed,
                "unchanged": unchanged,
                "raw_mean_nmse": fmt_nmse(raw_mean),
                "svd_mean_nmse": fmt_nmse(svd_mean),
                "raw_median_nmse": fmt_nmse(raw_median),
                "svd_median_nmse": fmt_nmse(svd_median),
                "mean_change_pct": f"{(svd_mean - raw_mean) / raw_mean * 100:+.1f}%" if raw_mean > 0 else "N/A",
                "median_change_pct": f"{(svd_median - raw_median) / raw_median * 100:+.1f}%" if raw_median > 0 else "N/A",
                "verdict": verdict,
            }

        raw_sorted = sorted(raw_lookup.values(), key=lambda r: r["nmse"], reverse=True)
        svd_top = raw_sorted[:top_n]

        for raw_r in svd_top:
            pname = raw_r["param_name"]
            svd_r = svd_lookup.get(pname)

            raw_nmse = raw_r["nmse"]
            raw_sqnr = raw_r["sqnr_db"]
            svd_nmse = svd_r["nmse"] if svd_r else float("nan")
            svd_sqnr = svd_r["sqnr_db"] if svd_r else float("nan")

            if not math.isinf(raw_nmse) and raw_nmse > 0 and svd_r and not math.isinf(svd_nmse):
                improvement = (raw_nmse - svd_nmse) / raw_nmse * 100
                imp_str = f"{improvement:+.1f}%"
                imp_positive = improvement > 0
            else:
                imp_str = "N/A"
                imp_positive = False

            if not math.isinf(raw_sqnr) and svd_r and not math.isinf(svd_sqnr):
                sqnr_gain = svd_sqnr - raw_sqnr
                gain_str = f"{sqnr_gain:+.1f} dB"
                gain_positive = sqnr_gain > 0
            else:
                gain_str = "N/A"
                gain_positive = False

            raw_bg, raw_fg = nmse_to_css_color(raw_nmse, args.skip_threshold, args.promote_threshold)
            svd_bg_c, svd_fg_c = nmse_to_css_color(svd_nmse, args.skip_threshold, args.promote_threshold) if svd_r else ("#2a2a4a", "#8899aa")

            svd_comparison.append({
                "param_name": pname,
                "truncated_name": truncate_layer_name(pname),
                "raw_nmse": fmt_nmse(raw_nmse),
                "svd_nmse": fmt_nmse(svd_nmse) if svd_r else "N/A",
                "improvement_pct": imp_str,
                "improvement_positive": imp_positive,
                "raw_sqnr": fmt_sqnr(raw_sqnr),
                "svd_sqnr": fmt_sqnr(svd_sqnr) if svd_r else "N/A",
                "sqnr_gain": gain_str,
                "sqnr_gain_positive": gain_positive,
                "raw_bg": raw_bg,
                "raw_fg": raw_fg,
                "svd_bg": svd_bg_c,
                "svd_fg": svd_fg_c,
            })

    # Rank-vs-rank comparison (when multiple SVD ranks tested, non-auto-config)
    rank_comparison = []
    rank_comparison_ranks = []
    multi_rank = len(args.svd_ranks) > 1
    if multi_rank and not args.no_svd and not multi_gs:
        rank_comparison_ranks = args.svd_ranks
        # Build lookup: param_name -> {svd_rank -> result}
        rank_lookup = {}
        for r in results:
            if r["weights_dtype"] != primary_dtype:
                continue
            if not r["use_svd"]:
                continue
            rgs = r.get("requested_group_size", args.group_size)
            if rgs != args.group_size:
                continue
            pname = r["param_name"]
            if pname not in rank_lookup:
                rank_lookup[pname] = {}
            rank_lookup[pname][r["svd_rank"]] = r

        # Use the first rank as reference, sort by its NMSE descending
        ref_rank = args.svd_ranks[0]
        ref_results = []
        for pname, by_rank in rank_lookup.items():
            if ref_rank in by_rank:
                ref_results.append((pname, by_rank))
        ref_results.sort(key=lambda x: x[1].get(ref_rank, {}).get("nmse", 0), reverse=True)
        ref_results = ref_results[:top_n]

        for pname, by_rank in ref_results:
            nmse_by_rank = {}
            for rk in rank_comparison_ranks:
                r = by_rank.get(rk)
                if r is not None:
                    val = r["nmse"]
                    bg, fg = nmse_to_css_color(val, args.skip_threshold, args.promote_threshold)
                    nmse_by_rank[rk] = {
                        "value": val,
                        "display": fmt_nmse(val),
                        "bg_color": bg,
                        "text_color": fg,
                    }
                else:
                    nmse_by_rank[rk] = {
                        "value": float("nan"),
                        "display": "N/A",
                        "bg_color": "#2a2a4a",
                        "text_color": "#8899aa",
                    }
            # Best rank for this layer
            valid = [(rk, nmse_by_rank[rk]["value"]) for rk in rank_comparison_ranks
                     if not math.isnan(nmse_by_rank[rk]["value"]) and not math.isinf(nmse_by_rank[rk]["value"])]
            best_rank = min(valid, key=lambda x: x[1])[0] if valid else None

            rank_comparison.append({
                "param_name": pname,
                "truncated_name": truncate_layer_name(pname),
                "nmse_by_rank": nmse_by_rank,
                "best_rank": best_rank,
            })

    # Config snippet
    skip_layers_unique = sorted(set(skip_layers))

    snippet_lines = []
    snippet_lines.append("# Layers to exclude from quantization entirely")
    snippet_lines.append("modules_to_not_convert = [")
    snippet_lines.extend(f'    "{layer}",' for layer in skip_layers_unique)
    snippet_lines.append("]")
    snippet_lines.append("")

    if multi_gs:
        # Group-size comparison: recommend group_size overrides
        promote_by_gs = {}
        for pname, gs_key in promote_layers.items():
            promote_by_gs.setdefault(gs_key, []).append(pname)

        snippet_lines.append("# Layers that need a different group size than the primary")
        snippet_lines.append("modules_group_size_dict = {")
        for gs_key, layer_list in sorted(promote_by_gs.items()):
            # Extract numeric group size from "g32" -> 32
            gs_val = int(gs_key[1:])
            layer_list_sorted = sorted(layer_list)
            snippet_lines.append(f"    {gs_val}: [")
            snippet_lines.extend(f'        "{layer}",' for layer in layer_list_sorted)
            snippet_lines.append("    ],")
        snippet_lines.append("}")
        snippet_lines.append("")
        snippet_lines.append("# Usage:")
        snippet_lines.append("# from sdnq import SDNQConfig")
        snippet_lines.append(f'# config = SDNQConfig(weights_dtype="{primary_dtype}",')
        snippet_lines.append(f"#                     group_size={primary_gs},")
        snippet_lines.append("#                     modules_to_not_convert=modules_to_not_convert)")
    else:
        promote_by_dtype = {}
        for pname, dtype in promote_layers.items():
            promote_by_dtype.setdefault(dtype, []).append(pname)

        snippet_lines.append("# Layers that need higher precision than the primary dtype")
        snippet_lines.append("modules_dtype_dict = {")
        for dtype, layer_list in sorted(promote_by_dtype.items()):
            layer_list_sorted = sorted(layer_list)
            snippet_lines.append(f'    "{dtype}": [')
            snippet_lines.extend(f'        "{layer}",' for layer in layer_list_sorted)
            snippet_lines.append("    ],")
        snippet_lines.append("}")
        snippet_lines.append("")
        snippet_lines.append("# Usage:")
        snippet_lines.append("# from sdnq import SDNQConfig")
        snippet_lines.append(f'# config = SDNQConfig(weights_dtype="{primary_dtype}",')
        snippet_lines.append("#                     modules_to_not_convert=modules_to_not_convert,")
        snippet_lines.append("#                     modules_dtype_dict=modules_dtype_dict)")

    config_snippet = "\n".join(snippet_lines)

    return {
        "primary_dtype": primary_dtype,
        "use_svd_filter": use_svd_filter,
        "column_keys": column_keys,
        "column_mode": column_mode,
        "primary_results": primary_results,
        "top_n_results": top_n_results,
        "nmse_lookup": nmse_lookup,
        "stats": {
            "total_layers": total_layers,
            "total_params": total_params,
            "ok_count": ok_count,
            "promote_count": promote_count,
            "skip_count": skip_count,
            "ok_pct": ok_pct,
            "promote_pct": promote_pct,
            "skip_pct": skip_pct,
            "mean_nmse": fmt_nmse(mean_nmse),
            "median_nmse": fmt_nmse(median_nmse),
            "worst_nmse": fmt_nmse(worst_nmse),
        },
        "classified_rows": classified_rows,
        "skip_layers": skip_layers_unique,
        "promote_layers": promote_layers,
        "svd_comparison": svd_comparison,
        "svd_summary": svd_summary,
        "rank_comparison": rank_comparison,
        "rank_comparison_ranks": rank_comparison_ranks,
        "config_snippet": config_snippet,
    }


def build_nmse_chart_base64(summary, args):
    """Build a horizontal bar chart of top-N layers and return as base64 PNG string."""
    import base64
    import io

    rows = summary["classified_rows"]
    if not rows:
        return None

    # Cap at 50 bars
    rows = rows[:50]
    # Reverse so worst is at top visually (matplotlib plots bottom-up)
    rows = list(reversed(rows))

    is_auto_config = summary["column_mode"] == "auto_config"
    primary_dtype = summary["primary_dtype"]
    labels = [r["truncated_name"] for r in rows]
    values = []
    colors = []
    annotations = []

    skip_thresh = args.skip_threshold
    promote_thresh = args.promote_threshold

    for r in rows:
        if is_auto_config:
            val = r["rec_nmse"]
        else:
            primary_col = summary["column_keys"][0]
            val = r["nmse_by_column"][primary_col]["value"]
        if math.isinf(val):
            # Use 2x skip threshold for display, annotate as INF
            display_val = skip_thresh * 2
            annotations.append("INF")
        else:
            display_val = val
            annotations.append(None)
        values.append(display_val)
        colors.append(nmse_to_chart_color(val, skip_thresh, promote_thresh))

    n = len(rows)
    fig_height = max(6, n * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    y_pos = range(n)
    ax.barh(y_pos, values, color=colors, edgecolor="none", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace", color="#e0e0e0")
    if is_auto_config:
        ax.set_xlabel("Recommended NMSE", color="#e0e0e0", fontsize=10)
        ax.set_title(f"Auto-Config: Recommended NMSE per Layer (top {len(rows)})", color="#ffffff", fontsize=13, pad=12)
    elif summary["column_mode"] == "group_size":
        primary_col = summary["column_keys"][0]
        ax.set_xlabel(f"NMSE @ {primary_dtype} (group_size={primary_col})", color="#e0e0e0", fontsize=10)
        ax.set_title(f"Top {len(rows)} Most Sensitive Layers", color="#ffffff", fontsize=13, pad=12)
    else:
        ax.set_xlabel(f"NMSE @ {primary_dtype}", color="#e0e0e0", fontsize=10)
        ax.set_title(f"Top {len(rows)} Most Sensitive Layers", color="#ffffff", fontsize=13, pad=12)

    # Use log scale if range is wide
    finite_values = [v for v in values if v > 0]
    if finite_values and max(finite_values) / (min(finite_values) or 1) > 100:
        ax.set_xscale("log")

    # Reference lines
    ax.axvline(x=skip_thresh, color="#d62828", linestyle="--", linewidth=1, alpha=0.7, label=f"skip={skip_thresh}")
    ax.axvline(x=promote_thresh, color="#e9c46a", linestyle="--", linewidth=1, alpha=0.7, label=f"promote={promote_thresh}")

    # Annotate INF bars
    for i, ann in enumerate(annotations):
        if ann:
            ax.text(values[i] * 1.05, i, ann, va="center", fontsize=7, color="#ff6b6b", fontweight="bold")

    ax.legend(loc="lower right", fontsize=8, facecolor="#16213e", edgecolor="#2a2a4a",
              labelcolor="#e0e0e0")
    ax.tick_params(axis="x", colors="#e0e0e0", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2a2a4a")
    ax.spines["left"].set_color("#2a2a4a")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _highlight_config_snippet(snippet):
    """Simple syntax highlighting for the config snippet HTML output."""
    import html as html_mod
    lines = snippet.split("\n")
    highlighted = []
    for line in lines:
        escaped = html_mod.escape(line)
        if escaped.lstrip().startswith("#"):
            highlighted.append(f'<span class="comment">{escaped}</span>')
        else:
            # Highlight strings
            import re
            parts = re.split(r'(&quot;[^&]*?&quot;|"[^"]*?")', escaped)
            out = []
            for part in parts:
                if part.startswith(("&quot;", '"')):
                    out.append(f'<span class="string">{part}</span>')
                else:
                    out.append(part)
            highlighted.append("".join(out))
    return "\n".join(highlighted)


def build_html_report(summary, args, chart_base64=None):
    """Render the Jinja2 HTML report template with summary data."""
    from datetime import datetime

    template_path = Path(__file__).parent / "report_template.html"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
        autoescape=False,
    )
    template = env.get_template(template_path.name)

    config_snippet_html = _highlight_config_snippet(summary["config_snippet"])

    # Per-component snippet sections for HTML
    config_snippets = summary.get("config_snippets", {})
    config_snippets_html = {}
    for comp, snippet in config_snippets.items():
        config_snippets_html[comp] = _highlight_config_snippet(snippet)

    html = template.render(
        model_id=args.model_id,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        primary_dtype=summary["primary_dtype"],
        use_svd_filter=summary["use_svd_filter"],
        skip_threshold=args.skip_threshold,
        promote_threshold=args.promote_threshold,
        dtypes=summary["column_keys"],
        column_mode=summary["column_mode"],
        stats=summary["stats"],
        classified_rows=summary["classified_rows"],
        svd_comparison=summary["svd_comparison"],
        svd_summary=summary.get("svd_summary"),
        size_stats=summary.get("size_stats"),
        rank_distribution=summary.get("rank_distribution", []),
        rank_comparison=summary.get("rank_comparison", []),
        rank_comparison_ranks=summary.get("rank_comparison_ranks", []),
        config_snippet_html=config_snippet_html,
        config_snippets_html=config_snippets_html,
        chart_base64=chart_base64,
    )
    return html


def generate_report(results, args):
    """Generate an HTML or PDF report from analysis results."""
    if not HAS_JINJA2:
        print("WARNING: jinja2 not installed. Skipping report. Install: pip install Jinja2")
        return

    summary = prepare_summary_data(results, args)
    if summary is None:
        print("WARNING: No data available for report generation.")
        return

    fmt = args.report_format
    include_chart = fmt in ("html-png", "pdf") and HAS_MATPLOTLIB
    if fmt in ("html-png", "pdf") and not HAS_MATPLOTLIB:
        print("WARNING: matplotlib not installed. Chart omitted from report.")

    chart_b64 = build_nmse_chart_base64(summary, args) if include_chart else None
    html = build_html_report(summary, args, chart_b64)

    # Determine output path
    if args.report_output:
        report_path = Path(args.report_output)
    else:
        stem = Path(args.output).stem
        suffix = ".pdf" if fmt == "pdf" else ".html"
        report_path = Path(args.output).with_name(stem + "_report" + suffix)

    if fmt == "pdf":
        if HAS_WEASYPRINT:
            weasyprint.HTML(string=html).write_pdf(str(report_path))
        else:
            print("WARNING: weasyprint not installed. Falling back to HTML output. Install: pip install weasyprint")
            report_path = report_path.with_suffix(".html")
            report_path.write_text(html, encoding="utf-8")
    else:
        report_path.write_text(html, encoding="utf-8")

    print(f"Report written to {report_path}")


def nmse_to_style(nmse, skip_thresh, promote_thresh):
    """Map NMSE value to a rich style string for heatmap coloring."""
    if math.isinf(nmse):
        return "bold white on red"
    if nmse <= promote_thresh * 0.1:
        return "bold green"
    elif nmse <= promote_thresh:
        return "green"
    elif nmse <= skip_thresh * 0.5:
        return "yellow"
    elif nmse <= skip_thresh:
        return "dark_orange"
    else:
        return "bold red"


def fmt_nmse(val):
    """Format an NMSE value for display."""
    if math.isinf(val):
        return "INF"
    return f"{val:.6f}"


def fmt_sqnr(val):
    """Format an SQNR dB value for display."""
    if math.isinf(val):
        return "-INF" if val < 0 else "INF"
    return f"{val:.1f}"


def truncate_layer_name(name, max_len=50):
    """Truncate a layer name, keeping the rightmost (most specific) portion."""
    if len(name) <= max_len:
        return name
    return "..." + name[-(max_len - 3):]


def _fmt_bytes(n):
    """Format a byte count as a human-readable string."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 ** 3:
        return f"{n / 1024**2:.1f} MB"
    else:
        return f"{n / 1024**3:.2f} GB"


def estimate_layer_bytes(numel, shape, dtype_str, group_size, use_svd, svd_rank):
    """Estimate total storage bytes for one quantized layer.

    Components: quantized weight + scale + zero-point (unsigned) + SVD factors.
    """
    from sdnq.common import dtype_dict as dd

    info = dd.get(dtype_str, {})
    num_bits = info.get("num_bits", 16)
    is_unsigned = info.get("is_unsigned", False)

    # Quantized weight bytes
    weight_bytes = numel * num_bits / 8

    # Scale overhead
    if group_size <= 0:
        # Tensorwise: one scale per output feature
        scale_count = shape[0] if shape else 1
    else:
        scale_count = math.ceil(numel / group_size)
    scale_bytes = scale_count * 2  # fp16

    # Zero-point (unsigned dtypes only)
    zp_bytes = scale_count * 2 if is_unsigned else 0

    # SVD overhead (svd_up + svd_down in fp16)
    svd_bytes = 0
    if use_svd and len(shape) >= 2:
        out_features = shape[0]
        in_features = shape[1]
        for d in shape[2:]:
            in_features *= d
        svd_bytes = (out_features * svd_rank + svd_rank * in_features) * 2

    return int(weight_bytes + scale_bytes + zp_bytes + svd_bytes)


def _print_rich_auto_config(summary, args):
    """Print rich auto-config summary with stats panel, recommendation table, and config snippet."""
    console = Console()
    s = summary["stats"]
    ss = summary["size_stats"]

    # --- Statistics Panel ---
    stats_text = Text()
    stats_text.append(f"Layers analyzed: {s['total_layers']}    ", style="bold")
    stats_text.append(f"Total parameters: {s['total_params']:,}\n")
    stats_text.append(f"Thresholds: skip={args.skip_threshold}, promote={args.promote_threshold}\n\n")

    stats_text.append("OK", style="bold green")
    stats_text.append(f": {s['ok_count']} ({s['ok_pct']:.1f}%)    ")
    stats_text.append("SKIP", style="bold red")
    stats_text.append(f": {s['skip_count']} ({s['skip_pct']:.1f}%)\n\n")

    stats_text.append("NMSE (recommended):  ")
    stats_text.append(f"mean={s['mean_nmse']}  median={s['median_nmse']}  worst={s['worst_nmse']}\n\n")

    stats_text.append("Size Estimates:\n", style="bold underline")
    stats_text.append(f"  FP16 baseline:     {ss['fp16_total_display']}\n")
    stats_text.append(f"  Uniform ({ss['uniform_dtype']} g{ss['uniform_gs']}): {ss['uniform_total_display']}\n")
    stats_text.append(f"  Recommended:       {ss['recommended_total_display']}", style="bold green")
    stats_text.append(f"  ({ss['savings_vs_fp16_pct']}% vs FP16, {ss['savings_vs_uniform_pct']}% vs uniform)")

    rank_dist = summary.get("rank_distribution", [])
    if rank_dist:
        stats_text.append("\n\n")
        stats_text.append("SVD Rank Distribution:\n", style="bold underline")
        parts = [f"{label}: {count} layers" for label, count in rank_dist]
        stats_text.append("  " + ", ".join(parts))

    # Per-component size breakdown
    per_comp = ss.get("per_component", {})
    if len(per_comp) > 1:
        stats_text.append("\n\n")
        stats_text.append("Per-Component Breakdown:\n", style="bold underline")
        for subfolder, cs in per_comp.items():
            stats_text.append(f"  {subfolder}: {cs['recommended_total_display']}")
            stats_text.append(f" ({cs['savings_vs_fp16_pct']}% vs FP16, "
                              f"{cs['ok_count']} OK / {cs['skip_count']} SKIP)\n")

    console.print(Panel(stats_text, title="Auto-Config Grid Search Summary", border_style="bright_blue"))

    # --- Recommendation Table ---
    multi_comp = len(ss.get("per_component", {})) > 1
    table = Table(
        title=f"Per-Layer Optimal Configuration ({len(summary['classified_rows'])} layers)",
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("#", justify="right", style="dim", width=4)
    if multi_comp:
        table.add_column("Comp", width=14, no_wrap=True)
    table.add_column("Layer", max_width=50, no_wrap=True)
    table.add_column("Shape", width=18)
    table.add_column("Rec.Dtype", width=10)
    table.add_column("GroupSize", justify="right", width=10)
    table.add_column("SVD", width=5)
    table.add_column("NMSE", justify="right", width=11)
    table.add_column("Est.Size", justify="right", width=10)
    table.add_column("Status", justify="right", width=8)

    for row in summary["classified_rows"]:
        status = row["status"]
        if status == "SKIP":
            status_text = Text(status, style="bold red")
        else:
            status_text = Text(status, style="bold green")

        nmse_style = nmse_to_style(row["rec_nmse"], args.skip_threshold, args.promote_threshold)
        nmse_text = Text(row["rec_nmse_display"], style=nmse_style)

        comp_subfolder = row.get("component", "").split(":")[0] if row.get("component") else ""
        row_cells = [str(row["rank"])]
        if multi_comp:
            row_cells.append(comp_subfolder)
        row_cells.extend([
            row["truncated_name"],
            row["shape"],
            row["rec_dtype"],
            str(row["rec_group_size"]),
            str(row["rec_svd_rank"]) if row["rec_svd"] else "no",
            nmse_text,
            row["rec_size_display"],
            status_text,
        ])
        table.add_row(*row_cells)

    console.print()
    console.print(table)

    # --- Config Snippet ---
    snippet_code = summary["config_snippet"]
    syntax = Syntax(snippet_code, "python", theme="monokai", line_numbers=False)
    console.print()
    console.print(Panel(syntax, title="SDNQ Config Snippet", border_style="bright_green"))
    console.print()


def _print_plain_auto_config(summary, args):
    """Print plain-text auto-config summary."""
    s = summary["stats"]
    ss = summary["size_stats"]

    print("\n" + "=" * 100)
    print("AUTO-CONFIG GRID SEARCH SUMMARY")
    print("=" * 100)
    print(f"Layers: {s['total_layers']}  Parameters: {s['total_params']:,}")
    print(f"Thresholds: skip={args.skip_threshold}, promote={args.promote_threshold}")
    print(f"OK: {s['ok_count']} ({s['ok_pct']:.1f}%)  SKIP: {s['skip_count']} ({s['skip_pct']:.1f}%)")
    print(f"NMSE (recommended): mean={s['mean_nmse']}  median={s['median_nmse']}  worst={s['worst_nmse']}")
    print()
    print(f"FP16 baseline:     {ss['fp16_total_display']}")
    print(f"Uniform ({ss['uniform_dtype']} g{ss['uniform_gs']}): {ss['uniform_total_display']}")
    print(f"Recommended:       {ss['recommended_total_display']}  "
          f"({ss['savings_vs_fp16_pct']}% vs FP16, {ss['savings_vs_uniform_pct']}% vs uniform)")

    rank_dist = summary.get("rank_distribution", [])
    if rank_dist:
        parts = [f"{label}: {count} layers" for label, count in rank_dist]
        print(f"\nSVD rank distribution: {', '.join(parts)}")

    per_comp = ss.get("per_component", {})
    if len(per_comp) > 1:
        print("\nPer-Component Breakdown:")
        for subfolder, cs in per_comp.items():
            print(f"  {subfolder}: {cs['recommended_total_display']} "
                  f"({cs['savings_vs_fp16_pct']}% vs FP16, "
                  f"{cs['ok_count']} OK / {cs['skip_count']} SKIP)")

    multi_comp = len(per_comp) > 1
    comp_col = f"{'Comp':<14}  " if multi_comp else ""
    header = (f"{'#':>4}  {comp_col}{'Layer':<50}  {'Shape':<18}  {'Dtype':<10}  "
              f"{'GS':>6}  {'SVD':>3}  {'NMSE':>10}  {'Size':>10}  {'Status':>6}")
    print()
    print(header)
    print("-" * len(header))

    for row in summary["classified_rows"]:
        svd_str = str(row["rec_svd_rank"]) if row["rec_svd"] else "no"
        comp_subfolder = row.get("component", "").split(":")[0] if row.get("component") else ""
        comp_part = f"{comp_subfolder:<14}  " if multi_comp else ""
        line = (f"{row['rank']:>4}  {comp_part}{row['truncated_name']:<50}  {row['shape']:<18}  "
                f"{row['rec_dtype']:<10}  {row['rec_group_size']!s:>6}  {svd_str:>3}  "
                f"{row['rec_nmse_display']:>10}  {row['rec_size_display']:>10}  {row['status']:>6}")
        print(line)

    print()
    print("=" * 100)
    print("SDNQ CONFIG SNIPPET")
    print("=" * 100)
    print()
    print(summary["config_snippet"])
    print()


def print_rich_summary(results, args):
    """Print rich console summary with heatmap tables, SVD comparison, and config snippets."""
    console = Console()

    summary = prepare_summary_data(results, args)
    if summary is None:
        console.print("[bold red]No results to display.[/]")
        return

    if summary["column_mode"] == "auto_config":
        _print_rich_auto_config(summary, args)
        return

    primary_dtype = summary["primary_dtype"]
    use_svd_filter = summary["use_svd_filter"]
    s = summary["stats"]

    # --- Statistics Panel ---
    stats_text = Text()
    stats_text.append(f"Layers analyzed: {s['total_layers']}    ", style="bold")
    stats_text.append(f"Total parameters: {s['total_params']:,}\n")
    stats_text.append(f"Primary dtype: {primary_dtype}    ", style="bold")
    stats_text.append(f"SVD: {'off' if not use_svd_filter else 'on'}    ")
    stats_text.append(f"Thresholds: skip={args.skip_threshold}, promote={args.promote_threshold}\n\n")

    stats_text.append("OK", style="bold green")
    stats_text.append(f": {s['ok_count']} ({s['ok_pct']:.1f}%)    ")
    stats_text.append("PROMOTE", style="bold yellow")
    stats_text.append(f": {s['promote_count']} ({s['promote_pct']:.1f}%)    ")
    stats_text.append("SKIP", style="bold red")
    stats_text.append(f": {s['skip_count']} ({s['skip_pct']:.1f}%)\n\n")

    stats_text.append(f"NMSE @ {primary_dtype}:  ")
    stats_text.append(f"mean={s['mean_nmse']}  median={s['median_nmse']}  worst={s['worst_nmse']}")

    svd_s = summary.get("svd_summary")
    if svd_s:
        stats_text.append("\n\n")
        stats_text.append("SVD vs Raw", style="bold underline")
        stats_text.append(f" ({svd_s['total_compared']} layers compared)\n")
        stats_text.append(f"  Mean NMSE:   raw={svd_s['raw_mean_nmse']}  svd={svd_s['svd_mean_nmse']}  ({svd_s['mean_change_pct']})\n")
        stats_text.append(f"  Median NMSE: raw={svd_s['raw_median_nmse']}  svd={svd_s['svd_median_nmse']}  ({svd_s['median_change_pct']})\n")
        stats_text.append("  Layers: ")
        stats_text.append(f"{svd_s['improved']} improved", style="green")
        stats_text.append(f"  {svd_s['harmed']} harmed", style="red")
        stats_text.append(f"  {svd_s['unchanged']} unchanged\n", style="dim")
        verdict = svd_s["verdict"]
        if verdict == "beneficial":
            stats_text.append("  Verdict: ", style="bold")
            stats_text.append("SVD is beneficial", style="bold green")
        elif verdict == "harmful":
            stats_text.append("  Verdict: ", style="bold")
            stats_text.append("SVD is harmful", style="bold red")
        else:
            stats_text.append("  Verdict: ", style="bold")
            stats_text.append("Mixed results", style="bold yellow")

    console.print(Panel(stats_text, title="Quantization Sensitivity Summary", border_style="bright_blue"))

    # --- Section A: Heatmap Table ---
    column_keys = summary["column_keys"]
    table = Table(
        title=f"Top {len(summary['classified_rows'])} Most Sensitive Layers (sorted by {primary_dtype} NMSE, svd={use_svd_filter})",
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Layer", max_width=50, no_wrap=True)
    table.add_column("Class", width=12)
    table.add_column("Shape", width=18)
    table.add_column("Kurtosis", justify="right", width=9)
    for ck in column_keys:
        table.add_column(ck, justify="right", width=11)
    table.add_column("Status", justify="right", width=10)

    for row in summary["classified_rows"]:
        status = row["status"]

        # Status styling
        if status == "SKIP":
            status_text = Text(status, style="bold red")
        elif status == "OK":
            status_text = Text(status, style="bold green")
        else:
            status_text = Text(status, style="bold yellow")

        # Build column cells with heatmap coloring
        col_cells = []
        for ck in column_keys:
            val = row["nmse_by_column"][ck]["value"]
            style = nmse_to_style(val, args.skip_threshold, args.promote_threshold)
            col_cells.append(Text(fmt_nmse(val), style=style))

        table.add_row(
            str(row["rank"]),
            row["truncated_name"],
            row["layer_class"],
            row["shape"],
            row["kurtosis"],
            *col_cells,
            status_text,
        )

    console.print()
    console.print(table)

    # --- Section B: SVD vs Raw Side-by-Side ---
    if summary["svd_comparison"]:
        svd_table = Table(
            title=f"SVD vs Raw Comparison @ {primary_dtype} (top {len(summary['svd_comparison'])} by raw NMSE)",
            show_lines=False,
            pad_edge=True,
        )
        svd_table.add_column("#", justify="right", style="dim", width=4)
        svd_table.add_column("Layer", max_width=50, no_wrap=True)
        svd_table.add_column("Raw NMSE", justify="right", width=11)
        svd_table.add_column("SVD NMSE", justify="right", width=11)
        svd_table.add_column("Improvement", justify="right", width=12)
        svd_table.add_column("Raw SQNR", justify="right", width=10)
        svd_table.add_column("SVD SQNR", justify="right", width=10)
        svd_table.add_column("SQNR Gain", justify="right", width=10)

        for i, svd_row in enumerate(summary["svd_comparison"], 1):
            raw_nmse_val = svd_row["raw_nmse"]
            svd_nmse_val = svd_row["svd_nmse"]

            imp_str = svd_row["improvement_pct"]
            imp_style = "green" if svd_row["improvement_positive"] else ("red" if imp_str != "N/A" else "dim")

            gain_str = svd_row["sqnr_gain"]
            gain_style = "green" if svd_row["sqnr_gain_positive"] else ("red" if gain_str != "N/A" else "dim")

            svd_table.add_row(
                str(i),
                svd_row["truncated_name"],
                Text(raw_nmse_val),
                Text(svd_nmse_val),
                Text(imp_str, style=imp_style),
                svd_row["raw_sqnr"],
                svd_row["svd_sqnr"],
                Text(gain_str, style=gain_style),
            )

        console.print()
        console.print(svd_table)

    # --- Section B2: Rank-vs-Rank Comparison ---
    rank_comp = summary.get("rank_comparison", [])
    rank_comp_ranks = summary.get("rank_comparison_ranks", [])
    if rank_comp and rank_comp_ranks:
        rank_table = Table(
            title=f"SVD Rank Comparison @ {primary_dtype} (top {len(rank_comp)} layers)",
            show_lines=False,
            pad_edge=True,
        )
        rank_table.add_column("#", justify="right", style="dim", width=4)
        rank_table.add_column("Layer", max_width=50, no_wrap=True)
        for rk in rank_comp_ranks:
            rank_table.add_column(f"rank={rk}", justify="right", width=11)
        rank_table.add_column("Best", justify="right", width=6)

        for i, rc_row in enumerate(rank_comp, 1):
            cells = []
            for rk in rank_comp_ranks:
                val = rc_row["nmse_by_rank"][rk]["value"]
                style = nmse_to_style(val, args.skip_threshold, args.promote_threshold)
                cells.append(Text(rc_row["nmse_by_rank"][rk]["display"], style=style))
            best_str = str(rc_row["best_rank"]) if rc_row["best_rank"] is not None else "-"
            rank_table.add_row(
                str(i),
                rc_row["truncated_name"],
                *cells,
                Text(best_str, style="bold"),
            )

        console.print()
        console.print(rank_table)

    # --- Section C: Config Snippet ---
    snippet_code = summary["config_snippet"]
    syntax = Syntax(snippet_code, "python", theme="monokai", line_numbers=False)

    console.print()
    console.print(Panel(syntax, title="SDNQ Config Snippet", border_style="bright_green"))
    console.print()


def print_plain_summary(results, args):
    """Fallback plain-text summary when rich is not available."""
    summary = prepare_summary_data(results, args)
    if summary is None:
        print("No results for primary dtype to summarize.")
        return

    if summary["column_mode"] == "auto_config":
        _print_plain_auto_config(summary, args)
        return

    primary_dtype = summary["primary_dtype"]
    use_svd_filter = summary["use_svd_filter"]
    s = summary["stats"]

    column_keys = summary["column_keys"]
    header = f"{'Rank':>4}  {'Layer':<60}  {'Class':<12}"
    for ck in column_keys:
        header += f"  {ck:>10}"
    header += f"  {'Status':>8}"

    print("\n" + "=" * len(header))
    print(f"TOP {len(summary['classified_rows'])} MOST SENSITIVE LAYERS (sorted by {primary_dtype} NMSE, svd={use_svd_filter})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for row in summary["classified_rows"]:
        line = f"{row['rank']:>4}  {row['param_name']:<60}  {row['layer_class']:<12}"
        for ck in column_keys:
            val = row["nmse_by_column"][ck]["value"]
            if math.isinf(val):
                line += f"  {'INF':>10}"
            elif math.isnan(val):
                line += f"  {'NaN':>10}"
            else:
                line += f"  {val:>10.6f}"
        line += f"  {row['status']:>8}"
        print(line)

    print("-" * len(header))
    print(f"Thresholds: skip={args.skip_threshold}, promote={args.promote_threshold}")
    print(f"\nOverall: {s['total_layers']} layers -- {s['ok_count']} OK, {s['promote_count']} PROMOTE, {s['skip_count']} SKIP")

    svd_s = summary.get("svd_summary")
    if svd_s:
        print(f"\nSVD vs Raw ({svd_s['total_compared']} layers compared):")
        print(f"  Mean NMSE:   raw={svd_s['raw_mean_nmse']}  svd={svd_s['svd_mean_nmse']}  ({svd_s['mean_change_pct']})")
        print(f"  Median NMSE: raw={svd_s['raw_median_nmse']}  svd={svd_s['svd_median_nmse']}  ({svd_s['median_change_pct']})")
        print(f"  Layers: {svd_s['improved']} improved, {svd_s['harmed']} harmed, {svd_s['unchanged']} unchanged")
        print(f"  Verdict: {svd_s['verdict'].upper()}")

    # Rank-vs-rank comparison (plain text)
    rank_comp = summary.get("rank_comparison", [])
    rank_comp_ranks = summary.get("rank_comparison_ranks", [])
    if rank_comp and rank_comp_ranks:
        print()
        rk_header = f"{'#':>4}  {'Layer':<60}"
        for rk in rank_comp_ranks:
            rk_header += f"  {'rank=' + str(rk):>10}"
        rk_header += f"  {'Best':>6}"
        print("=" * len(rk_header))
        print(f"SVD RANK COMPARISON @ {summary['primary_dtype']} (top {len(rank_comp)} layers)")
        print("=" * len(rk_header))
        print(rk_header)
        print("-" * len(rk_header))
        for i, rc_row in enumerate(rank_comp, 1):
            line = f"{i:>4}  {rc_row['truncated_name']:<60}"
            for rk in rank_comp_ranks:
                val = rc_row["nmse_by_rank"][rk]["display"]
                line += f"  {val:>10}"
            best_str = str(rc_row["best_rank"]) if rc_row["best_rank"] is not None else "-"
            line += f"  {best_str:>6}"
            print(line)

    # Config snippet (plain text)
    print("\n" + "=" * 80)
    print("SDNQ CONFIG SNIPPET")
    print("=" * 80)
    print()
    print(summary["config_snippet"])
    print()


def print_summary(results, args):
    """Print summary, using rich output if available."""
    if HAS_RICH:
        print_rich_summary(results, args)
    else:
        print("[WARNING] 'rich' not installed — falling back to plain text output. "
              "Install with: pip install rich")
        print_plain_summary(results, args)


def main():
    args = parse_args()

    # Auto-discover components if not specified
    if args.components is None:
        discovered = discover_components(args.model_id, args.cache_dir)
        if discovered:
            args.components = discovered
            print(f"Auto-discovered {len(discovered)} components from model_index.json")
        else:
            print("ERROR: No --components specified and model_index.json not found.")
            print("       Specify components explicitly, e.g.: --components transformer:MyModelClass")
            sys.exit(1)

    print("Quantization Sensitivity Analysis")
    print(f"  Model: {args.model_id}")
    print(f"  Components: {args.components}")
    print(f"  Dtypes: {args.dtypes}")
    print(f"  SVD modes: raw={'yes' if not args.no_raw else 'no'}, svd={'yes' if not args.no_svd else 'no'}")
    print(f"  SVD rank(s): {args.svd_ranks}")
    print(f"  Group size(s): {args.group_sizes}")
    print(f"  Device: {args.device}")

    all_results = []
    for comp_spec in args.components:
        if ":" not in comp_spec:
            print(f"ERROR: Component spec must be 'subfolder:ClassName', got '{comp_spec}'")
            sys.exit(1)
        subfolder, class_name = comp_spec.split(":", 1)
        results = analyze_component(args.model_id, subfolder, class_name,
                                     args.cache_dir, args)
        all_results.extend(results)

    write_csv(all_results, args.output)
    print_summary(all_results, args)
    generate_report(all_results, args)

    if args.auto_config:
        summary = prepare_summary_data(all_results, args)
        if summary and summary.get("pipeline_config_json"):
            import json as _json

            config_path = args.pipeline_config_output
            if config_path is None:
                config_path = str(Path(args.output).with_name(
                    Path(args.output).stem + "_pipeline_config.json"
                ))
            with open(config_path, "w", encoding="utf-8") as f:
                _json.dump(summary["pipeline_config_json"], f, indent=2)
            print(f"Pipeline config written to {config_path}")


if __name__ == "__main__":
    main()
