#!/usr/bin/env python3
"""Measure quantization impact on diffusion pipeline outputs.

Compares FP16 baseline and quantized pipeline outputs at two levels:
per-step latent divergence and final image quality metrics (PSNR, SSIM, LPIPS).

Only one pipeline is in VRAM at a time. Intermediate latents and images are
saved to disk so the comparison phase runs on CPU with no model loaded.

Example usage:
    # Pre-quantized pipeline
    python scripts/benchmark_inference_quality.py \
        --model-id Tongyi-MAI/Z-Image \
        --quantized-path /path/to/quantized

    # On-the-fly quantization with uniform config
    python scripts/benchmark_inference_quality.py \
        --model-id Tongyi-MAI/Z-Image \
        --weights-dtype int8 --use-quantized-matmul

    # On-the-fly quantization with per-component pipeline config
    python scripts/benchmark_inference_quality.py \
        --model-id Tongyi-MAI/Z-Image \
        --pipeline-config quant_sensitivity_report_pipeline_config.json
"""

import argparse
import base64
import csv
import importlib

import sdnq  # noqa: F401 - registers SDNQ quantizer with diffusers/transformers
import inspect
import io
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

try:
    import lpips as lpips_module
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

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
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Default prompts when none provided
DEFAULT_PROMPTS = [
    "A photo of an astronaut riding a horse on the moon",
    "A serene landscape with mountains reflected in a crystal-clear lake at sunset",
    "A futuristic city skyline with flying cars and neon lights at night",
    "A close-up portrait of a cat wearing a tiny top hat, digital art",
    "An oil painting of a medieval castle surrounded by a moat, dramatic lighting",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark inference quality: compare FP16 baseline vs quantized pipeline outputs"
    )

    # Model source
    p.add_argument("--model-id", required=True,
                    help="HuggingFace model ID or local path")
    p.add_argument("--quantized-path", default=None,
                    help="Pre-quantized pipeline path (if omitted, quantize on-the-fly)")

    # On-the-fly quantization (ignored when --quantized-path set)
    p.add_argument("--weights-dtype", default="int8",
                    help="Quantization dtype for on-the-fly quantization")
    p.add_argument("--group-size", type=int, default=0,
                    help="Group size (0=auto, -1=tensorwise)")
    p.add_argument("--use-svd", action="store_true",
                    help="Enable SVD compression")
    p.add_argument("--svd-rank", type=int, default=32,
                    help="SVD rank")
    p.add_argument("--use-quantized-matmul", action="store_true",
                    help="Enable quantized matmul kernels")
    p.add_argument("--pipeline-config", default=None,
                    help="JSON config from analyze script (per-component SDNQConfig)")

    # Inference
    p.add_argument("--prompts", default=None,
                    help="Text file with one prompt per line (uses built-in defaults if omitted)")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                    help="Random seeds for reproducibility")
    p.add_argument("--num-inference-steps", type=int, default=20,
                    help="Number of denoising steps")
    p.add_argument("--guidance-scale", type=float, default=7.5,
                    help="Guidance scale (passed only if pipeline accepts it)")
    p.add_argument("--height", type=int, default=1024,
                    help="Output image height")
    p.add_argument("--width", type=int, default=1024,
                    help="Output image width")
    p.add_argument("--pipeline-class", default=None,
                    help="Explicit pipeline class name (auto-detected if omitted)")
    p.add_argument("--torch-dtype", choices=["float16", "bfloat16"], default="bfloat16",
                    help="Torch dtype for pipeline loading")

    # Environment
    p.add_argument("--cache-dir", default="/home/ohiom/database/models/huggingface",
                    help="HuggingFace cache directory")
    p.add_argument("--device", default="cuda",
                    help="Device to run inference on")
    p.add_argument("--cpu-offload", action="store_true",
                    help="Enable model CPU offloading (moves submodels to CPU when not in use)")
    p.add_argument("--output-dir", default="inference_quality_results",
                    help="Directory for intermediate latents and images")

    # Output
    p.add_argument("--output", default="inference_quality_report.csv",
                    help="CSV output path")
    p.add_argument("--report-format", choices=["html-png", "html", "none"], default="html-png",
                    help="Report format (default: html-png)")
    p.add_argument("--report-output", default=None,
                    help="Custom report output path")
    p.add_argument("--compare-only", action="store_true",
                    help="Skip inference, recompute metrics from saved data in --output-dir")

    return p.parse_args()


def load_prompts(path):
    """Load prompts from a text file (one per line) or return built-in defaults."""
    if path is None:
        return DEFAULT_PROMPTS
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    if not prompts:
        print("WARNING: Prompt file is empty, using built-in defaults.")
        return DEFAULT_PROMPTS
    return prompts


def resolve_pipeline_class(model_id, cache_dir, explicit_class=None):
    """Resolve the diffusers pipeline class for the given model.

    If explicit_class is provided, import it directly. Otherwise, read
    model_index.json to determine the class.
    """
    if explicit_class:
        mod = importlib.import_module("diffusers")
        cls = getattr(mod, explicit_class, None)
        if cls is None:
            raise ValueError(f"Could not find class '{explicit_class}' in diffusers")
        return cls

    # Try to read model_index.json
    try:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(
                model_id, "model_index.json",
                cache_dir=cache_dir, local_files_only=True,
            )
        except Exception:
            path = hf_hub_download(
                model_id, "model_index.json",
                cache_dir=cache_dir,
            )
        with open(path, encoding="utf-8") as f:
            index = json.load(f)
        class_name = index.get("_class_name")
        if class_name:
            mod = importlib.import_module("diffusers")
            cls = getattr(mod, class_name, None)
            if cls is not None:
                return cls
    except Exception:
        pass

    # Check if it's a local directory
    local_index = Path(model_id) / "model_index.json"
    if local_index.exists():
        with open(local_index, encoding="utf-8") as f:
            index = json.load(f)
        class_name = index.get("_class_name")
        if class_name:
            mod = importlib.import_module("diffusers")
            cls = getattr(mod, class_name, None)
            if cls is not None:
                return cls

    # Fallback to DiffusionPipeline
    from diffusers import DiffusionPipeline
    return DiffusionPipeline


def load_fp16_pipeline(model_id, cache_dir, device, torch_dtype, pipeline_cls,
                       cpu_offload=False):
    """Load the FP16 baseline pipeline."""
    print(f"\nLoading FP16 baseline pipeline ({pipeline_cls.__name__})...")
    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
        print("  FP16 pipeline loaded (CPU offloading enabled).")
    else:
        pipe = pipe.to(device)
        print("  FP16 pipeline loaded.")
    return pipe


def load_quantized_pipeline(model_id, quantized_path, cache_dir, device,
                            torch_dtype, pipeline_cls, args):
    """Load the quantized pipeline (pre-quantized or on-the-fly)."""
    cpu_offload = getattr(args, "cpu_offload", False)
    if quantized_path:
        print(f"\nLoading pre-quantized pipeline from {quantized_path}...")
        pipe = pipeline_cls.from_pretrained(
            quantized_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        if cpu_offload:
            pipe.enable_model_cpu_offload()
            print("  Pre-quantized pipeline loaded (CPU offloading enabled).")
        else:
            pipe = pipe.to(device)
            print("  Pre-quantized pipeline loaded.")
        return pipe

    # On-the-fly quantization
    print(f"\nLoading pipeline for on-the-fly quantization ({pipeline_cls.__name__})...")
    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # Load per-component config if provided
    per_comp_config = None
    if args.pipeline_config:
        with open(args.pipeline_config, encoding="utf-8") as f:
            per_comp_config = json.load(f)
        print(f"  Loaded per-component config from {args.pipeline_config}")

    # Discover and quantize components
    if per_comp_config:
        _quantize_pipeline_components(pipe, per_comp_config, device)
    else:
        _quantize_pipeline_uniform(pipe, args, device)

    print("  On-the-fly quantization complete.")
    return pipe


def _quantize_pipeline_components(pipe, per_comp_config, device):
    """Quantize pipeline components using per-component config."""
    from sdnq.quantizer import sdnq_post_load_quant

    valid_keys = {
        "weights_dtype", "group_size", "svd_rank", "svd_steps",
        "use_svd", "use_quantized_matmul", "dequantize_fp32",
        "modules_to_not_convert", "modules_dtype_dict", "modules_svd_dict",
        "modules_group_size_dict",
    }

    for comp_name, config in per_comp_config.items():
        component = getattr(pipe, comp_name, None)
        if component is None or not isinstance(component, torch.nn.Module):
            continue
        quant_kwargs = {k: v for k, v in config.items() if k in valid_keys}
        print(f"  Quantizing {comp_name} (dtype={config.get('weights_dtype', 'int8')})...")
        sdnq_post_load_quant(component, add_skip_keys=True, **quant_kwargs)


def _quantize_pipeline_uniform(pipe, args, device):
    """Quantize all quantizable pipeline components with uniform config."""
    from sdnq.quantizer import sdnq_post_load_quant

    # Find model components (nn.Module subclasses that aren't the pipe itself)
    for attr_name in dir(pipe):
        if attr_name.startswith("_"):
            continue
        component = getattr(pipe, attr_name, None)
        if component is None or component is pipe:
            continue
        if not isinstance(component, torch.nn.Module):
            continue
        # Skip schedulers, tokenizers, etc. (they don't have parameters)
        if not any(True for _ in component.parameters()):
            continue
        print(f"  Quantizing {attr_name}...")
        quant_kwargs = {
            "weights_dtype": args.weights_dtype,
            "group_size": args.group_size,
            "use_quantized_matmul": args.use_quantized_matmul,
        }
        if args.use_svd:
            quant_kwargs["use_svd"] = True
            quant_kwargs["svd_rank"] = args.svd_rank
        sdnq_post_load_quant(component, add_skip_keys=True, **quant_kwargs)


def unload_pipeline(pipe):
    """Unload a pipeline and free VRAM."""
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


class LatentCollector:
    """callback_on_step_end handler that clones latents to CPU."""

    def __init__(self):
        self.latents = []

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        self.latents.append(callback_kwargs["latents"].detach().cpu().float().clone())
        return callback_kwargs


class LegacyLatentCollector:
    """Legacy callback handler for older pipeline APIs."""

    def __init__(self):
        self.latents = []

    def __call__(self, step, timestep, latents):
        self.latents.append(latents.detach().cpu().float().clone())


def _build_call_kwargs(pipe, args, seed, collector, legacy_collector):
    """Build kwargs for pipeline __call__ based on its signature."""
    sig = inspect.signature(pipe.__call__)
    call_kwargs = {
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
    }

    if "guidance_scale" in sig.parameters:
        call_kwargs["guidance_scale"] = args.guidance_scale

    # Latent capture callback
    has_callback = False
    if "callback_on_step_end" in sig.parameters:
        call_kwargs["callback_on_step_end"] = collector
        if "callback_on_step_end_tensor_inputs" in sig.parameters:
            call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        has_callback = True
    elif "callback" in sig.parameters:
        call_kwargs["callback"] = legacy_collector
        if "callback_steps" in sig.parameters:
            call_kwargs["callback_steps"] = 1
        has_callback = True

    # Deterministic generator
    device = args.device
    if device.startswith("cuda"):
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = torch.Generator().manual_seed(seed)
    if "generator" in sig.parameters:
        call_kwargs["generator"] = gen

    return call_kwargs, has_callback


def run_pipeline_inference(pipe, prompts, seeds, args, output_dir, label):
    """Run inference for all prompt x seed combos, saving latents and images.

    Returns dict mapping (prompt_idx, seed) -> {"latents": list[path], "final_image": path}
    """
    results = {}
    total = len(prompts) * len(seeds)
    print(f"\nRunning {label} inference ({total} combinations)...")

    for p_idx, prompt in enumerate(prompts):
        for seed in seeds:
            combo_dir = Path(output_dir) / label / f"prompt_{p_idx}_seed_{seed}"
            combo_dir.mkdir(parents=True, exist_ok=True)

            collector = LatentCollector()
            legacy_collector = LegacyLatentCollector()
            call_kwargs, has_callback = _build_call_kwargs(
                pipe, args, seed, collector, legacy_collector,
            )

            if not has_callback and p_idx == 0 and seed == seeds[0]:
                    print("  WARNING: Pipeline does not support latent callbacks. "
                          "Only final images will be compared.")

            output = pipe(prompt=prompt, **call_kwargs)

            # Extract final image
            if hasattr(output, "images") and output.images:
                image = output.images[0]
            else:
                print(f"  WARNING: No images in output for prompt {p_idx}, seed {seed}")
                continue

            # Save final image
            image_path = combo_dir / "final_image.png"
            image.save(str(image_path))

            # Save per-step latents
            latents = collector.latents if collector.latents else legacy_collector.latents
            latent_paths = []
            for step_i, lat in enumerate(latents):
                lat_path = combo_dir / f"step_{step_i:02d}.pt"
                torch.save(lat, str(lat_path))
                latent_paths.append(str(lat_path))

            results[(p_idx, seed)] = {
                "latents": latent_paths,
                "final_image": str(image_path),
            }
            print(f"  [{label}] prompt {p_idx}, seed {seed}: "
                  f"{len(latent_paths)} steps captured, image saved")

    return results


def compute_latent_metrics(fp16_latent_paths, quant_latent_paths):
    """Compare per-step latents and return list of {step, mse, cosine_sim}."""
    n_steps = min(len(fp16_latent_paths), len(quant_latent_paths))
    step_metrics = []
    for i in range(n_steps):
        fp16_lat = torch.load(fp16_latent_paths[i], weights_only=True).float()
        quant_lat = torch.load(quant_latent_paths[i], weights_only=True).float()
        mse = F.mse_loss(fp16_lat, quant_lat).item()
        cos = F.cosine_similarity(
            fp16_lat.flatten().unsqueeze(0),
            quant_lat.flatten().unsqueeze(0),
            dim=1,
        ).item()
        step_metrics.append({"step": i, "mse": mse, "cosine_sim": cos})
    return step_metrics


def compute_psnr(img1, img2):
    """Compute PSNR between two PIL Images (uint8)."""
    if not HAS_NUMPY:
        return float("nan")
    a1 = np.array(img1, dtype=np.float64)
    a2 = np.array(img2, dtype=np.float64)
    mse = np.mean((a1 - a2) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def compute_image_metrics(fp16_img_path, quant_img_path):
    """Compute image quality metrics between two images.

    Returns dict with psnr (always), ssim (if torchmetrics), lpips (if lpips).
    """
    img1 = Image.open(fp16_img_path).convert("RGB")
    img2 = Image.open(quant_img_path).convert("RGB")

    metrics = {}

    # PSNR (always available)
    metrics["psnr"] = compute_psnr(img1, img2)

    # Convert to tensors for SSIM/LPIPS: [1, C, H, W] float [0, 1]
    if (HAS_TORCHMETRICS or HAS_LPIPS) and HAS_NUMPY:
            t1 = torch.from_numpy(np.array(img1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t2 = torch.from_numpy(np.array(img2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # SSIM
    if HAS_TORCHMETRICS and HAS_NUMPY:
        try:
            ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)
            metrics["ssim"] = ssim_fn(t1, t2).item()
        except Exception as e:
            print(f"  WARNING: SSIM computation failed: {e}")
            metrics["ssim"] = None
    else:
        metrics["ssim"] = None

    # LPIPS
    if HAS_LPIPS and HAS_NUMPY:
        try:
            # LPIPS expects input in [-1, 1]
            lp1 = t1 * 2.0 - 1.0
            lp2 = t2 * 2.0 - 1.0
            lpips_fn = lpips_module.LPIPS(net="alex", verbose=False)
            metrics["lpips"] = lpips_fn(lp1, lp2).item()
        except Exception as e:
            print(f"  WARNING: LPIPS computation failed: {e}")
            metrics["lpips"] = None
    else:
        metrics["lpips"] = None

    return metrics


def compare_results(fp16_results, quant_results, prompts, seeds):
    """Compare FP16 and quantized results for all prompt x seed combos.

    Returns (step_results, image_results) where:
    - step_results: list of {prompt_idx, seed, step, mse, cosine_sim}
    - image_results: list of {prompt_idx, seed, prompt, psnr, ssim, lpips,
                              mean_step_mse, max_step_mse, final_step_mse,
                              mean_cosine_sim, min_cosine_sim}
    """
    print("\nComparing results (CPU)...")
    step_results = []
    image_results = []

    for p_idx, prompt in enumerate(prompts):
        for seed in seeds:
            key = (p_idx, seed)
            if key not in fp16_results or key not in quant_results:
                continue

            fp16 = fp16_results[key]
            quant = quant_results[key]

            # Per-step latent metrics
            latent_metrics = []
            if fp16["latents"] and quant["latents"]:
                latent_metrics = compute_latent_metrics(
                    fp16["latents"], quant["latents"],
                )
                step_results.extend({
                    "prompt_idx": p_idx,
                    "seed": seed,
                    "step": m["step"],
                    "mse": m["mse"],
                    "cosine_sim": m["cosine_sim"],
                } for m in latent_metrics)

            # Image metrics
            img_metrics = compute_image_metrics(
                fp16["final_image"], quant["final_image"],
            )

            # Aggregate step metrics
            if latent_metrics:
                mses = [m["mse"] for m in latent_metrics]
                cosines = [m["cosine_sim"] for m in latent_metrics]
                mean_step_mse = sum(mses) / len(mses)
                max_step_mse = max(mses)
                final_step_mse = mses[-1]
                mean_cos = sum(cosines) / len(cosines)
                min_cos = min(cosines)
            else:
                mean_step_mse = None
                max_step_mse = None
                final_step_mse = None
                mean_cos = None
                min_cos = None

            image_results.append({
                "prompt_idx": p_idx,
                "seed": seed,
                "prompt": prompt,
                "psnr": img_metrics["psnr"],
                "ssim": img_metrics["ssim"],
                "lpips": img_metrics["lpips"],
                "mean_step_mse": mean_step_mse,
                "max_step_mse": max_step_mse,
                "final_step_mse": final_step_mse,
                "mean_cosine_sim": mean_cos,
                "min_cosine_sim": min_cos,
            })

    return step_results, image_results


def write_summary_csv(image_results, output_path):
    """Write the main summary CSV (one row per prompt x seed)."""
    if not image_results:
        return
    fieldnames = [
        "prompt_idx", "seed", "prompt", "psnr", "ssim", "lpips",
        "mean_step_mse", "max_step_mse", "final_step_mse",
        "mean_cosine_sim", "min_cosine_sim",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in image_results:
            writer.writerow(row)
    print(f"Summary CSV written to {output_path}")


def write_step_csv(step_results, output_path):
    """Write the per-step CSV (one row per prompt x seed x step)."""
    if not step_results:
        return
    stem = Path(output_path).stem
    step_path = Path(output_path).with_name(stem + "_steps.csv")
    fieldnames = ["prompt_idx", "seed", "step", "mse", "cosine_sim"]
    with open(step_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in step_results:
            writer.writerow(row)
    print(f"Steps CSV written to {step_path}")


def _classify_error_trend(step_results, prompts, seeds):
    """Classify error accumulation pattern across all runs.

    Returns one of: "bounded", "accumulating", "self-correcting", "insufficient-data"
    """
    if not step_results:
        return "insufficient-data"

    # Group by (prompt_idx, seed)
    runs = {}
    for sr in step_results:
        key = (sr["prompt_idx"], sr["seed"])
        runs.setdefault(key, []).append(sr)

    trends = []
    for steps in runs.values():
        steps_sorted = sorted(steps, key=lambda s: s["step"])
        if len(steps_sorted) < 3:
            continue
        mses = [s["mse"] for s in steps_sorted]
        # Compare first third, middle third, last third
        n = len(mses)
        t1 = sum(mses[: n // 3]) / max(1, n // 3)
        t3 = sum(mses[2 * n // 3:]) / max(1, n - 2 * n // 3)
        if t1 == 0:
            continue
        ratio = t3 / t1
        if ratio > 2.0:
            trends.append("accumulating")
        elif ratio < 0.5:
            trends.append("self-correcting")
        else:
            trends.append("bounded")

    if not trends:
        return "insufficient-data"

    # Majority vote
    from collections import Counter
    counts = Counter(trends)
    return counts.most_common(1)[0][0]


def _fmt_val(val, fmt=".4f"):
    """Format a metric value, handling None and special floats."""
    if val is None:
        return "N/A"
    if math.isinf(val):
        return "INF"
    if math.isnan(val):
        return "N/A"
    return f"{val:{fmt}}"


def print_rich_summary(step_results, image_results, args):
    """Print summary tables using rich."""
    console = Console()

    # Table 1: Image quality per prompt/seed
    t1 = Table(title="Image Quality Metrics", show_lines=True)
    t1.add_column("Prompt", style="cyan", max_width=40)
    t1.add_column("Seed", style="dim")
    t1.add_column("PSNR (dB)", justify="right")
    t1.add_column("SSIM", justify="right")
    t1.add_column("LPIPS", justify="right")

    for row in image_results:
        prompt_short = row["prompt"][:37] + "..." if len(row["prompt"]) > 40 else row["prompt"]
        psnr = row["psnr"]
        psnr_style = "green" if psnr and psnr > 30 else "yellow" if psnr and psnr > 20 else "red"
        psnr_text = Text(_fmt_val(psnr, ".2f"), style=psnr_style) if psnr is not None else Text("N/A")

        t1.add_row(
            prompt_short,
            str(row["seed"]),
            psnr_text,
            _fmt_val(row["ssim"]),
            _fmt_val(row["lpips"]),
        )

    console.print(t1)

    # Table 2: Error accumulation
    if step_results:
        t2 = Table(title="Error Accumulation", show_lines=True)
        t2.add_column("Prompt", style="cyan", max_width=30)
        t2.add_column("Seed", style="dim")
        t2.add_column("First Step MSE", justify="right")
        t2.add_column("Mid Step MSE", justify="right")
        t2.add_column("Final Step MSE", justify="right")
        t2.add_column("Mean Cosine", justify="right")

        # Group steps by run
        runs = {}
        for sr in step_results:
            key = (sr["prompt_idx"], sr["seed"])
            runs.setdefault(key, []).append(sr)

        for row in image_results:
            key = (row["prompt_idx"], row["seed"])
            steps = sorted(runs.get(key, []), key=lambda s: s["step"])
            if not steps:
                continue
            prompt_short = row["prompt"][:27] + "..." if len(row["prompt"]) > 30 else row["prompt"]
            first_mse = steps[0]["mse"]
            mid_mse = steps[len(steps) // 2]["mse"]
            final_mse = steps[-1]["mse"]
            t2.add_row(
                prompt_short,
                str(row["seed"]),
                _fmt_val(first_mse, ".6f"),
                _fmt_val(mid_mse, ".6f"),
                _fmt_val(final_mse, ".6f"),
                _fmt_val(row["mean_cosine_sim"]),
            )

        console.print(t2)

    # Aggregate panel
    if image_results:
        psnrs = [r["psnr"] for r in image_results if r["psnr"] is not None and not math.isinf(r["psnr"]) and not math.isnan(r["psnr"])]
        ssims = [r["ssim"] for r in image_results if r["ssim"] is not None]
        lpipss = [r["lpips"] for r in image_results if r["lpips"] is not None]

        lines = []
        if psnrs:
            lines.append(f"Mean PSNR: {sum(psnrs) / len(psnrs):.2f} dB")
        if ssims:
            lines.append(f"Mean SSIM: {sum(ssims) / len(ssims):.4f}")
        if lpipss:
            lines.append(f"Mean LPIPS: {sum(lpipss) / len(lpipss):.4f}")

        trend = _classify_error_trend(step_results, None, None)
        trend_labels = {
            "bounded": "[green]Bounded[/green] — error stays stable across steps",
            "accumulating": "[red]Accumulating[/red] — error grows with denoising steps",
            "self-correcting": "[cyan]Self-correcting[/cyan] — error decreases in later steps",
            "insufficient-data": "[dim]Insufficient data[/dim]",
        }
        lines.append(f"Error trend: {trend_labels.get(trend, trend)}")

        console.print(Panel("\n".join(lines), title="Aggregate Metrics"))


def print_plain_summary(step_results, image_results, args):
    """Print summary tables in plain text."""
    print("\n" + "=" * 80)
    print("IMAGE QUALITY METRICS")
    print("=" * 80)
    header = f"{'Prompt':<40} {'Seed':<8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}"
    print(header)
    print("-" * len(header))
    for row in image_results:
        prompt_short = row["prompt"][:37] + "..." if len(row["prompt"]) > 40 else row["prompt"]
        print(f"{prompt_short:<40} {row['seed']:<8} "
              f"{_fmt_val(row['psnr'], '.2f'):>8} "
              f"{_fmt_val(row['ssim']):>8} "
              f"{_fmt_val(row['lpips']):>8}")

    if step_results:
        print("\n" + "=" * 80)
        print("ERROR ACCUMULATION")
        print("=" * 80)
        runs = {}
        for sr in step_results:
            key = (sr["prompt_idx"], sr["seed"])
            runs.setdefault(key, []).append(sr)

        header2 = f"{'Prompt':<30} {'Seed':<8} {'1st MSE':>12} {'Mid MSE':>12} {'Last MSE':>12} {'Mean Cos':>10}"
        print(header2)
        print("-" * len(header2))
        for row in image_results:
            key = (row["prompt_idx"], row["seed"])
            steps = sorted(runs.get(key, []), key=lambda s: s["step"])
            if not steps:
                continue
            prompt_short = row["prompt"][:27] + "..." if len(row["prompt"]) > 30 else row["prompt"]
            print(f"{prompt_short:<30} {row['seed']:<8} "
                  f"{_fmt_val(steps[0]['mse'], '.6f'):>12} "
                  f"{_fmt_val(steps[len(steps) // 2]['mse'], '.6f'):>12} "
                  f"{_fmt_val(steps[-1]['mse'], '.6f'):>12} "
                  f"{_fmt_val(row['mean_cosine_sim']):>10}")

    # Aggregate
    if image_results:
        print("\n" + "=" * 80)
        print("AGGREGATE METRICS")
        print("=" * 80)
        psnrs = [r["psnr"] for r in image_results if r["psnr"] is not None and not math.isinf(r["psnr"]) and not math.isnan(r["psnr"])]
        ssims = [r["ssim"] for r in image_results if r["ssim"] is not None]
        lpipss = [r["lpips"] for r in image_results if r["lpips"] is not None]
        if psnrs:
            print(f"  Mean PSNR: {sum(psnrs) / len(psnrs):.2f} dB")
        if ssims:
            print(f"  Mean SSIM: {sum(ssims) / len(ssims):.4f}")
        if lpipss:
            print(f"  Mean LPIPS: {sum(lpipss) / len(lpipss):.4f}")
        trend = _classify_error_trend(step_results, None, None)
        print(f"  Error trend: {trend}")


def print_summary(step_results, image_results, args):
    """Print summary using rich if available, otherwise plain text."""
    if HAS_RICH:
        print_rich_summary(step_results, image_results, args)
    else:
        print_plain_summary(step_results, image_results, args)


# ── HTML Report ──────────────────────────────────────────────────────────────

REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Inference Quality Report — {{ model_id }}</title>
<style>
:root {
    --bg: #1a1a2e; --surface: #16213e; --surface2: #0f3460;
    --text: #e0e0e0; --text-dim: #a0a0a0; --accent: #00b4d8;
    --green: #40916c; --yellow: #b8860b; --orange: #e76f51; --red: #d62828;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 2rem; }
.container { max-width: 1400px; margin: 0 auto; }
h1 { color: var(--accent); margin-bottom: 0.5rem; font-size: 1.8rem; }
h2 { color: var(--accent); margin: 2rem 0 1rem; font-size: 1.3rem; }
.meta { color: var(--text-dim); margin-bottom: 2rem; font-size: 0.9rem; }
.stats-card {
    background: var(--surface); border-radius: 8px; padding: 1.5rem;
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;
    margin-bottom: 2rem;
}
.stat { text-align: center; }
.stat-value { font-size: 1.6rem; font-weight: bold; color: var(--accent); font-family: monospace; }
.stat-label { font-size: 0.85rem; color: var(--text-dim); margin-top: 0.25rem; }
.trend-bounded { color: var(--green); }
.trend-accumulating { color: var(--red); }
.trend-self-correcting { color: var(--accent); }
table { width: 100%; border-collapse: collapse; margin-bottom: 2rem; background: var(--surface); border-radius: 8px; overflow: hidden; }
th { background: var(--surface2); color: var(--accent); padding: 0.75rem 1rem; text-align: left; position: sticky; top: 0; font-size: 0.85rem; }
td { padding: 0.6rem 1rem; border-top: 1px solid rgba(255,255,255,0.05); font-family: monospace; font-size: 0.85rem; }
tr:hover { background: rgba(255,255,255,0.03); }
.psnr-excellent { background: rgba(64,145,108,0.3); }
.psnr-good { background: rgba(64,145,108,0.15); }
.psnr-fair { background: rgba(184,134,11,0.2); }
.psnr-poor { background: rgba(214,40,40,0.2); }
.chart-container { background: var(--surface); border-radius: 8px; padding: 1rem; margin-bottom: 2rem; text-align: center; }
.chart-container img { max-width: 100%; height: auto; }
.images-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
.image-pair { background: var(--surface); border-radius: 8px; padding: 1rem; }
.image-pair h3 { font-size: 0.9rem; color: var(--text-dim); margin-bottom: 0.5rem; }
.image-pair .imgs { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
.image-pair img { width: 100%; border-radius: 4px; }
.image-pair .label { text-align: center; font-size: 0.75rem; color: var(--text-dim); margin-top: 0.25rem; }
@media print {
    :root { --bg: #fff; --surface: #f5f5f5; --surface2: #e0e0e0; --text: #1a1a1a; --text-dim: #666; }
    body { padding: 1rem; }
}
</style>
</head>
<body>
<div class="container">
<h1>Inference Quality Report</h1>
<p class="meta">Model: <strong>{{ model_id }}</strong> | Generated: {{ timestamp }} | Steps: {{ num_steps }} | {{ num_combos }} prompt x seed combinations</p>

<div class="stats-card">
    <div class="stat">
        <div class="stat-value">{{ mean_psnr }}</div>
        <div class="stat-label">Mean PSNR (dB)</div>
    </div>
    <div class="stat">
        <div class="stat-value">{{ mean_ssim }}</div>
        <div class="stat-label">Mean SSIM</div>
    </div>
    <div class="stat">
        <div class="stat-value">{{ mean_lpips }}</div>
        <div class="stat-label">Mean LPIPS</div>
    </div>
    <div class="stat">
        <div class="stat-value trend-{{ trend_class }}">{{ error_trend }}</div>
        <div class="stat-label">Error Trend</div>
    </div>
</div>

<h2>Image Quality</h2>
<div style="overflow-x:auto;">
<table>
<thead><tr>
    <th>Prompt</th><th>Seed</th><th>PSNR (dB)</th><th>SSIM</th><th>LPIPS</th>
    <th>Mean Step MSE</th><th>Final Step MSE</th><th>Mean Cosine</th>
</tr></thead>
<tbody>
{% for row in image_rows %}
<tr>
    <td>{{ row.prompt_short }}</td>
    <td>{{ row.seed }}</td>
    <td class="{{ row.psnr_class }}">{{ row.psnr }}</td>
    <td>{{ row.ssim }}</td>
    <td>{{ row.lpips }}</td>
    <td>{{ row.mean_step_mse }}</td>
    <td>{{ row.final_step_mse }}</td>
    <td>{{ row.mean_cosine_sim }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

{% if chart_base64 %}
<h2>Error Accumulation</h2>
<div class="chart-container">
    <img src="data:image/png;base64,{{ chart_base64 }}" alt="Error accumulation chart">
</div>
{% endif %}

{% if image_pairs %}
<h2>Side-by-Side Comparisons</h2>
<div class="images-grid">
{% for pair in image_pairs %}
<div class="image-pair">
    <h3>Prompt {{ pair.prompt_idx }}, Seed {{ pair.seed }}</h3>
    <div class="imgs">
        <div>
            <img src="data:image/png;base64,{{ pair.fp16_b64 }}" alt="FP16">
            <div class="label">FP16 Baseline</div>
        </div>
        <div>
            <img src="data:image/png;base64,{{ pair.quant_b64 }}" alt="Quantized">
            <div class="label">Quantized</div>
        </div>
    </div>
</div>
{% endfor %}
</div>
{% endif %}

</div>
</body>
</html>
"""


def _psnr_class(psnr):
    """Map PSNR to a CSS class."""
    if psnr is None or math.isnan(psnr) or math.isinf(psnr):
        return ""
    if psnr >= 35:
        return "psnr-excellent"
    if psnr >= 28:
        return "psnr-good"
    if psnr >= 20:
        return "psnr-fair"
    return "psnr-poor"


def _image_to_base64(img_path, max_size=256):
    """Load an image, resize for thumbnail, and return base64-encoded PNG."""
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_error_chart_base64(step_results):
    """Build a matplotlib chart of MSE vs step, one line per run."""
    if not HAS_MATPLOTLIB or not step_results:
        return None

    # Group by (prompt_idx, seed)
    runs = {}
    for sr in step_results:
        key = (sr["prompt_idx"], sr["seed"])
        runs.setdefault(key, []).append(sr)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for key, steps in sorted(runs.items()):
        steps_sorted = sorted(steps, key=lambda s: s["step"])
        x = [s["step"] for s in steps_sorted]
        y = [s["mse"] for s in steps_sorted]
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.2,
                label=f"p{key[0]}/s{key[1]}", alpha=0.8)

    ax.set_xlabel("Denoising Step", color="#e0e0e0")
    ax.set_ylabel("MSE (latent space)", color="#e0e0e0")
    ax.set_title("Error Accumulation Across Denoising Steps", color="#00b4d8")
    ax.tick_params(colors="#a0a0a0")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.5,
              labelcolor="#e0e0e0", facecolor="#16213e", edgecolor="#333")

    # Use log scale if range is large
    y_vals = [s["mse"] for s in step_results if s["mse"] > 0]
    if y_vals and max(y_vals) / max(min(y_vals), 1e-12) > 100:
        ax.set_yscale("log")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_report(step_results, image_results, fp16_results, quant_results, args):
    """Generate an HTML report with metrics tables, chart, and image thumbnails."""
    if args.report_format == "none":
        return
    if not HAS_JINJA2:
        print("WARNING: jinja2 not installed. Skipping report. Install: pip install Jinja2")
        return

    from datetime import datetime

    # Compute aggregates
    psnrs = [r["psnr"] for r in image_results if r["psnr"] is not None and not math.isinf(r["psnr"]) and not math.isnan(r["psnr"])]
    ssims = [r["ssim"] for r in image_results if r["ssim"] is not None]
    lpipss = [r["lpips"] for r in image_results if r["lpips"] is not None]

    mean_psnr = f"{sum(psnrs) / len(psnrs):.2f}" if psnrs else "N/A"
    mean_ssim = f"{sum(ssims) / len(ssims):.4f}" if ssims else "N/A"
    mean_lpips = f"{sum(lpipss) / len(lpipss):.4f}" if lpipss else "N/A"

    trend = _classify_error_trend(step_results, None, None)
    trend_class = trend.replace("-", "_") if "-" in trend else trend

    # Build image rows
    image_rows = []
    for row in image_results:
        prompt_short = row["prompt"][:50] + "..." if len(row["prompt"]) > 53 else row["prompt"]
        image_rows.append({
            "prompt_short": prompt_short,
            "seed": row["seed"],
            "psnr": _fmt_val(row["psnr"], ".2f"),
            "psnr_class": _psnr_class(row["psnr"]),
            "ssim": _fmt_val(row["ssim"]),
            "lpips": _fmt_val(row["lpips"]),
            "mean_step_mse": _fmt_val(row["mean_step_mse"], ".6f"),
            "final_step_mse": _fmt_val(row["final_step_mse"], ".6f"),
            "mean_cosine_sim": _fmt_val(row["mean_cosine_sim"]),
        })

    # Chart
    include_chart = args.report_format == "html-png" and HAS_MATPLOTLIB
    if args.report_format == "html-png" and not HAS_MATPLOTLIB:
        print("WARNING: matplotlib not installed. Chart omitted from report.")
    chart_b64 = build_error_chart_base64(step_results) if include_chart else None

    # Image pairs (thumbnails)
    image_pairs = []
    for row in image_results:
        key = (row["prompt_idx"], row["seed"])
        if key in fp16_results and key in quant_results:
            try:
                fp16_b64 = _image_to_base64(fp16_results[key]["final_image"])
                quant_b64 = _image_to_base64(quant_results[key]["final_image"])
                image_pairs.append({
                    "prompt_idx": row["prompt_idx"],
                    "seed": row["seed"],
                    "fp16_b64": fp16_b64,
                    "quant_b64": quant_b64,
                })
            except Exception:
                pass

    # Render
    env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=False)
    template = env.from_string(REPORT_TEMPLATE)
    html = template.render(
        model_id=args.model_id,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_steps=args.num_inference_steps,
        num_combos=len(image_results),
        mean_psnr=mean_psnr,
        mean_ssim=mean_ssim,
        mean_lpips=mean_lpips,
        error_trend=trend.replace("-", " ").title(),
        trend_class=trend_class,
        image_rows=image_rows,
        chart_base64=chart_b64,
        image_pairs=image_pairs,
    )

    # Write
    if args.report_output:
        report_path = Path(args.report_output)
    else:
        stem = Path(args.output).stem
        report_path = Path(args.output).with_name(stem + "_report.html")

    report_path.write_text(html, encoding="utf-8")
    print(f"Report written to {report_path}")


def load_results_from_disk(output_dir, label, prompts, seeds):
    """Reload previously saved inference results from disk."""
    results = {}
    for p_idx in range(len(prompts)):
        for seed in seeds:
            combo_dir = Path(output_dir) / label / f"prompt_{p_idx}_seed_{seed}"
            image_path = combo_dir / "final_image.png"
            if not image_path.exists():
                print(f"  WARNING: Missing {image_path}, skipping")
                continue
            latent_paths = sorted(combo_dir.glob("step_*.pt"))
            results[(p_idx, seed)] = {
                "latents": [str(p) for p in latent_paths],
                "final_image": str(image_path),
            }
    print(f"  Loaded {len(results)} {label} results from {output_dir}/{label}/")
    return results


def main():
    args = parse_args()

    # Resolve torch dtype
    torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16

    # Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Prompts: {len(prompts)}, Seeds: {args.seeds}, Steps: {args.num_inference_steps}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare_only:
        print("\n--compare-only: skipping inference, loading saved data...")
        fp16_results = load_results_from_disk(
            args.output_dir, "fp16", prompts, args.seeds,
        )
        quant_results = load_results_from_disk(
            args.output_dir, "quantized", prompts, args.seeds,
        )
    else:
        # Resolve pipeline class
        pipeline_cls = resolve_pipeline_class(
            args.model_id, args.cache_dir, args.pipeline_class,
        )
        print(f"Pipeline class: {pipeline_cls.__name__}")

        # Phase 1: FP16 baseline
        fp16_pipe = load_fp16_pipeline(
            args.model_id, args.cache_dir, args.device, torch_dtype, pipeline_cls,
            cpu_offload=args.cpu_offload,
        )
        fp16_results = run_pipeline_inference(
            fp16_pipe, prompts, args.seeds, args, args.output_dir, "fp16",
        )
        unload_pipeline(fp16_pipe)

        # Phase 2: Quantized
        quant_pipe = load_quantized_pipeline(
            args.model_id, args.quantized_path, args.cache_dir,
            args.device, torch_dtype, pipeline_cls, args,
        )
        quant_results = run_pipeline_inference(
            quant_pipe, prompts, args.seeds, args, args.output_dir, "quantized",
        )
        unload_pipeline(quant_pipe)

    # Phase 3: Compare (CPU only)
    step_results, image_results = compare_results(
        fp16_results, quant_results, prompts, args.seeds,
    )

    # Phase 4: Output
    write_summary_csv(image_results, args.output)
    write_step_csv(step_results, args.output)
    print_summary(step_results, image_results, args)
    generate_report(step_results, image_results, fp16_results, quant_results, args)


if __name__ == "__main__":
    main()
