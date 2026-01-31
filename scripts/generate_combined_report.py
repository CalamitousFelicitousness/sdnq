#!/usr/bin/env python3
"""Generate a combined HTML report from all ref2 ablation test results.

Reads all ref2_*_report.csv and ref2_*_report_steps.csv files,
embeds side-by-side images (FP16 vs quantized) for all tests,
and produces an interactive HTML report with charts.

Optionally reads memory_comparison.csv (from estimate_config_memory.py)
to include memory footprint and PSNR/GB efficiency data.

Usage:
    source /home/ohiom/sdnq/venv/bin/activate
    python scripts/generate_combined_report.py
"""

import base64
import csv
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Test definitions ─────────────────────────────────────────────────────────

TESTS = [
    # (short_name, csv_stem, output_dir, label, category)
    ("int8 (no skip keys)", "ref2_int8_report", "ref2_results", "Reference Configs", "ref"),
    ("uint8 (no skip keys)", "ref2_uint8_report", "ref2_uint8_results", "Reference Configs", "ref"),
    ("uint4 (no skip keys)", "ref2_uint4_report", "ref2_uint4_results", "Reference Configs", "ref"),
    ("uint4 + skip adaLN", "ref2_uint4_skip_adaln_report", "ref2_uint4_skip_adaln_results", "Reference Configs", "ref"),
    ("uint4 WITH skip keys", "ref2_uint4_skipkeys_report", "ref2_uint4_skipkeys_results", "Skip Keys Comparison", "skip"),
    ("int8 WITH skip keys", "ref2_int8_skipkeys_report", "ref2_int8_skipkeys_results", "Skip Keys Comparison", "skip"),
    ("ONLY adaLN quantized", "ref2_only_adaln_report", "ref2_only_adaln_results", "Forward Ablation", "fwd"),
    ("ONLY attention quantized", "ref2_only_attn_report", "ref2_only_attn_results", "Forward Ablation", "fwd"),
    ("ONLY MLP quantized", "ref2_only_mlp_report", "ref2_only_mlp_results", "Forward Ablation", "fwd"),
    ("ONLY embed+final quantized", "ref2_only_embed_report", "ref2_only_embed_results", "Forward Ablation", "fwd"),
    # Phase 2: Combination tests
    ("int8 transformer only", "ref2_int8_transformer_only_report", "ref2_int8_transformer_only_results", "Component Isolation", "iso"),
    ("int8 text encoder only", "ref2_int8_textenc_only_report", "ref2_int8_textenc_only_results", "Component Isolation", "iso"),
    ("int8 + skip adaLN", "ref2_int8_skip_adaln_report", "ref2_int8_skip_adaln_results", "Combined Skip Strategies", "combo"),
    ("int8 + skip embed+adaLN+final", "ref2_int8_skip_all_sensitive_report", "ref2_int8_skip_all_sensitive_results", "Combined Skip Strategies", "combo"),
    ("int8 + skip adaLN+attn", "ref2_int8_skip_adaln_attn_report", "ref2_int8_skip_adaln_attn_results", "Combined Skip Strategies", "combo"),
    ("uint8 + skip embed+adaLN+final", "ref2_uint8_skip_all_sensitive_report", "ref2_uint8_skip_all_sensitive_results", "uint8 with Skips", "u8skip"),
    ("uint8 + skip embed+adaLN+attn+final", "ref2_uint8_skip_adaln_attn_report", "ref2_uint8_skip_adaln_attn_results", "uint8 with Skips", "u8skip"),
    ("int8 auto-config (sensitivity)", "ref2_int8_autoconfig_report", "ref2_int8_autoconfig_results", "Auto-Config", "auto"),
    # Phase 3: uint4 Maximum Quality
    ("uint4 MLP-only (aggressive skips)", "ref2_uint4_mlp_only_report", "ref2_uint4_mlp_only_results", "uint4 Maximum Quality", "u4max"),
    ("uint4 MLP-only + SVD rank 64", "ref2_uint4_mlp_svd_report", "ref2_uint4_mlp_svd_results", "uint4 Maximum Quality", "u4max"),
    ("uint4 MLP-only + SVD + stoch + FP32", "ref2_uint4_mlp_svd_stoch_fp32_report", "ref2_uint4_mlp_svd_stoch_fp32_results", "uint4 Maximum Quality", "u4max"),
    ("uint4 hybrid uint6 attn", "ref2_uint4_hybrid_u6attn_report", "ref2_uint4_hybrid_u6attn_results", "uint4 Hybrid Configs", "u4hyb"),
    ("uint4 hybrid uint8 attn + SVD", "ref2_uint4_hybrid_u8attn_svd_report", "ref2_uint4_hybrid_u8attn_svd_results", "uint4 Hybrid Configs", "u4hyb"),
    ("uint4 kitchen sink (all levers)", "ref2_uint4_kitchen_sink_report", "ref2_uint4_kitchen_sink_results", "uint4 Hybrid Configs", "u4hyb"),
    ("uint4 auto-config enhanced", "ref2_uint4_autoconfig_enhanced_report", "ref2_uint4_autoconfig_enhanced_results", "uint4 Auto-Config", "u4auto"),
    # Phase 4: Runtime & Dtype Ablation
    ("int8 + quantized matmul", "ablation_int8_quantized_matmul_report", "ablation_int8_quantized_matmul_results", "Runtime Ablation", "runtime"),
    ("int8 + dequantize FP32", "ablation_int8_dequant_fp32_report", "ablation_int8_dequant_fp32_results", "Runtime Ablation", "runtime"),
    ("int8 + stochastic rounding", "ablation_int8_stochastic_rounding_report",
     "ablation_int8_stochastic_rounding_results", "Runtime Ablation", "runtime"),
    ("uint4 MLP+SVD+stoch+FP32 (fixed)", "ablation_uint4_stochastic_rounding_report",
     "ablation_uint4_stochastic_rounding_results", "Runtime Ablation", "runtime"),
    ("float8_e4m3fn", "ablation_fp8_e4m3fn_report", "ablation_fp8_e4m3fn_results", "FP8 Dtype", "fp8"),
    ("float8_e4m3fn + quantized matmul", "ablation_fp8_e4m3fn_quantized_matmul_report",
     "ablation_fp8_e4m3fn_quantized_matmul_results", "FP8 Dtype", "fp8"),
    ("int8 group_size=64", "ablation_int8_group64_report", "ablation_int8_group64_results", "Group Size", "grp"),
    ("int8 tensorwise (g=-1)", "ablation_int8_tensorwise_report", "ablation_int8_tensorwise_results", "Group Size", "grp"),
    # Phase 4b: uint4 Runtime & Group Size Ablation
    ("uint4 + quantized matmul", "ablation_uint4_quantized_matmul_report",
     "ablation_uint4_quantized_matmul_results", "uint4 Runtime Ablation", "u4runtime"),
    ("uint4 + dequantize FP32", "ablation_uint4_dequant_fp32_report",
     "ablation_uint4_dequant_fp32_results", "uint4 Runtime Ablation", "u4runtime"),
    ("uint4 group_size=64", "ablation_uint4_group64_report",
     "ablation_uint4_group64_results", "uint4 Group Size", "u4grp"),
    ("uint4 tensorwise (g=-1)", "ablation_uint4_tensorwise_report",
     "ablation_uint4_tensorwise_results", "uint4 Group Size", "u4grp"),
]

BASE_DIR = Path(__file__).resolve().parent.parent
FP16_DIR = BASE_DIR / "ref2_results"  # FP16 images shared across all tests


def read_csv(path):
    """Read a CSV file and return list of dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def image_to_base64(img_path, max_size=384):
    """Load image, resize for thumbnail, return base64 PNG."""
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def psnr_class(val):
    if val is None or math.isnan(val) or math.isinf(val):
        return ""
    if val >= 35:
        return "excellent"
    if val >= 25:
        return "good"
    if val >= 18:
        return "fair"
    if val >= 12:
        return "poor"
    return "terrible"


def psnr_color(val):
    """Return inline color for PSNR bar."""
    if val >= 35:
        return "#40916c"
    if val >= 25:
        return "#6aaa64"
    if val >= 18:
        return "#b8860b"
    if val >= 12:
        return "#e76f51"
    return "#d62828"


def build_error_chart(all_steps_data):
    """Build MSE-vs-step chart for all tests, return base64 PNG."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    colors = plt.cm.tab20.colors

    for ci, (test_name, steps) in enumerate(all_steps_data.items()):
        # Group by prompt_idx, seed — average across prompts for cleaner chart
        by_step = {}
        for s in steps:
            step = int(s["step"])
            mse = float(s["mse"])
            by_step.setdefault(step, []).append(mse)

        avg_by_step = {k: sum(v) / len(v) for k, v in sorted(by_step.items())}
        x = list(avg_by_step.keys())
        y = list(avg_by_step.values())

        color = colors[ci % len(colors)]
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.5,
                label=test_name, alpha=0.85, color=color)

    ax.set_xlabel("Denoising Step", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("MSE (latent space)", color="#e0e0e0", fontsize=11)
    ax.set_title("Error Accumulation Across Denoising Steps — All Tests",
                 color="#00b4d8", fontsize=14, pad=12)
    ax.tick_params(colors="#a0a0a0")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.7,
              labelcolor="#e0e0e0", facecolor="#16213e", edgecolor="#444",
              ncol=2)
    ax.grid(True, alpha=0.15, color="#555")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_bar_chart(tests_data, metric, label, higher_is_better=True):
    """Build horizontal bar chart for a single metric across all tests."""
    names = []
    values = []
    for name, rows in tests_data.items():
        vals = [float(r[metric]) for r in rows if r.get(metric)]
        if vals:
            names.append(name)
            values.append(sum(vals) / len(vals))

    fig, ax = plt.subplots(figsize=(12, max(4, len(names) * 0.5)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    colors = []
    for v in values:
        if metric == "psnr":
            colors.append(psnr_color(v))
        elif metric == "ssim":
            if v >= 0.85:
                colors.append("#40916c")
            elif v >= 0.6:
                colors.append("#b8860b")
            else:
                colors.append("#d62828")
        elif metric == "lpips":
            if v <= 0.15:
                colors.append("#40916c")
            elif v <= 0.5:
                colors.append("#b8860b")
            else:
                colors.append("#d62828")
        else:
            colors.append("#00b4d8")

    y_pos = range(len(names))
    ax.barh(y_pos, values, color=colors, height=0.6, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color="#e0e0e0", fontsize=9)
    ax.set_xlabel(label, color="#e0e0e0", fontsize=11)
    ax.set_title(f"{label} — All Configurations", color="#00b4d8", fontsize=13, pad=10)
    ax.tick_params(colors="#a0a0a0")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, axis="x", alpha=0.15, color="#555")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        fmt = f"{v:.2f}" if metric == "psnr" else f"{v:.4f}"
        ax.text(v + max(values) * 0.01, i, fmt, va="center",
                color="#e0e0e0", fontsize=8, fontfamily="monospace")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_memory_data():
    """Load memory_comparison.csv and return dict keyed by test label.

    Keys by both the CSV's own label and the config_json filename (without .json,
    with _report suffix matching) so HTML report names can look up by csv_stem.
    """
    mem_csv = BASE_DIR / "memory_comparison.csv"
    if not mem_csv.exists():
        print(f"  Memory data not found at {mem_csv} — skipping memory columns")
        return {}
    mem = {}
    with open(mem_csv, newline="") as f:
        for row in csv.DictReader(f):
            entry = {
                "quant_bytes": int(row["quant_bytes"]),
                "fp16_bytes": int(row["fp16_bytes"]),
                "savings_pct": float(row["savings_pct"]),
                "psnr_per_gb": float(row["psnr_per_gb"]),
            }
            # Key by label
            mem[row["label"]] = entry
            # Also key by config_json stem for csv_stem-based lookup
            # e.g. "ref2_int8.json" -> key "ref2_int8"
            cfg = row.get("config_json", "")
            if cfg.endswith(".json"):
                mem[cfg[:-5]] = entry
    return mem


def build_efficiency_chart(memory_data, tests_summary):
    """Build scatter plot: memory savings % (x) vs PSNR (y), with PSNR/GB as bubble size."""
    names = []
    savings = []
    psnrs = []
    efficiencies = []

    for name, csv_stem, _output_dir, _section, _cat in TESTS:
        if name not in tests_summary:
            continue
        rows = tests_summary[name]
        csv_key = csv_stem.replace("_report", "")
        mem = memory_data.get(name) or memory_data.get(csv_key)
        if not mem:
            continue
        vals = [float(r["psnr"]) for r in rows if r.get("psnr")]
        if not vals:
            continue
        names.append(name)
        savings.append(mem["savings_pct"])
        psnrs.append(sum(vals) / len(vals))
        efficiencies.append(mem["psnr_per_gb"])

    if not names:
        return None

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Bubble size proportional to PSNR/GB
    max_eff = max(efficiencies) if efficiencies else 1
    sizes = [max(40, (e / max_eff) * 400) for e in efficiencies]

    # Color by PSNR
    colors = [psnr_color(p) for p in psnrs]

    ax.scatter(savings, psnrs, s=sizes, c=colors, alpha=0.75, edgecolors="#555", linewidths=0.5)

    for i, name in enumerate(names):
        short = name[:25] + "..." if len(name) > 28 else name
        ax.annotate(short, (savings[i], psnrs[i]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=6.5, color="#c0c0c0", alpha=0.9)

    ax.set_xlabel("Memory Savings (%)", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("Mean PSNR (dB)", color="#e0e0e0", fontsize=11)
    ax.set_title("Quality vs Memory Savings — Bubble Size = PSNR/GB Efficiency",
                 color="#00b4d8", fontsize=13, pad=12)
    ax.tick_params(colors="#a0a0a0")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, alpha=0.15, color="#555")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_memory_bar_chart(memory_data, tests_summary):
    """Build horizontal bar chart showing quantized size vs FP16 size for all tests."""
    names = []
    quant_gb = []
    fp16_gb = []

    for name, csv_stem, _output_dir, _section, _cat in TESTS:
        if name not in tests_summary:
            continue
        csv_key = csv_stem.replace("_report", "")
        mem = memory_data.get(name) or memory_data.get(csv_key)
        if not mem:
            continue
        names.append(name)
        quant_gb.append(mem["quant_bytes"] / (1024**3))
        fp16_gb.append(mem["fp16_bytes"] / (1024**3))

    if not names:
        return None

    fig, ax = plt.subplots(figsize=(12, max(4, len(names) * 0.5)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    y_pos = range(len(names))
    # FP16 bars (background, lighter)
    ax.barh(y_pos, fp16_gb, color="#333355", height=0.6, alpha=0.5, label="FP16 Baseline")
    # Quantized bars (foreground)
    bar_colors = []
    for qg, fg in zip(quant_gb, fp16_gb):
        savings = (1 - qg / fg) * 100
        if savings >= 60:
            bar_colors.append("#40916c")
        elif savings >= 40:
            bar_colors.append("#6aaa64")
        elif savings >= 20:
            bar_colors.append("#b8860b")
        else:
            bar_colors.append("#e76f51")
    ax.barh(y_pos, quant_gb, color=bar_colors, height=0.6, alpha=0.85, label="Quantized")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color="#e0e0e0", fontsize=9)
    ax.set_xlabel("Size (GB)", color="#e0e0e0", fontsize=11)
    ax.set_title("Memory Footprint — Quantized vs FP16 Baseline", color="#00b4d8", fontsize=13, pad=10)
    ax.tick_params(colors="#a0a0a0")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, axis="x", alpha=0.15, color="#555")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right", framealpha=0.7,
              labelcolor="#e0e0e0", facecolor="#16213e", edgecolor="#444")

    for i, (qg, fg) in enumerate(zip(quant_gb, fp16_gb)):
        savings = (1 - qg / fg) * 100
        ax.text(qg + 0.1, i, f"{qg:.1f} GB ({savings:.0f}% saved)",
                va="center", color="#e0e0e0", fontsize=7.5, fontfamily="monospace")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    from datetime import datetime

    # ── Load memory data ─────────────────────────────────────────────────────
    memory_data = load_memory_data()

    # ── Load all data ────────────────────────────────────────────────────────
    tests_summary = {}  # name -> list of row dicts
    tests_steps = {}    # name -> list of step row dicts
    tests_images = {}   # name -> [(prompt_idx, seed, fp16_b64, quant_b64)]

    for name, csv_stem, output_dir, _section, _cat in TESTS:
        csv_path = BASE_DIR / f"{csv_stem}.csv"
        steps_path = BASE_DIR / f"{csv_stem}_steps.csv"
        quant_dir = BASE_DIR / output_dir / "quantized"

        if not csv_path.exists():
            print(f"  SKIP {name}: {csv_path} not found")
            continue

        rows = read_csv(csv_path)
        tests_summary[name] = rows

        if steps_path.exists():
            tests_steps[name] = read_csv(steps_path)

        # Collect image pairs
        pairs = []
        for row in rows:
            p_idx = int(row["prompt_idx"])
            seed = int(row["seed"])
            fp16_img = FP16_DIR / "fp16" / f"prompt_{p_idx}_seed_{seed}" / "final_image.png"
            quant_img = quant_dir / f"prompt_{p_idx}_seed_{seed}" / "final_image.png"
            if fp16_img.exists() and quant_img.exists():
                pairs.append({
                    "prompt_idx": p_idx,
                    "seed": seed,
                    "prompt": row["prompt"][:80] + "..." if len(row["prompt"]) > 83 else row["prompt"],
                    "fp16_b64": image_to_base64(fp16_img),
                    "quant_b64": image_to_base64(quant_img),
                    "psnr": float(row["psnr"]),
                    "ssim": float(row.get("ssim", 0) or 0),
                })
        tests_images[name] = pairs

    print(f"Loaded {len(tests_summary)} tests")

    # ── Build charts ─────────────────────────────────────────────────────────
    error_chart_b64 = build_error_chart(tests_steps) if tests_steps else None
    psnr_chart_b64 = build_bar_chart(tests_summary, "psnr", "Mean PSNR (dB)")
    ssim_chart_b64 = build_bar_chart(tests_summary, "ssim", "Mean SSIM")
    lpips_chart_b64 = build_bar_chart(tests_summary, "lpips", "Mean LPIPS")
    efficiency_chart_b64 = build_efficiency_chart(memory_data, tests_summary) if memory_data else None
    memory_chart_b64 = build_memory_bar_chart(memory_data, tests_summary) if memory_data else None

    # ── Compute summary table ────────────────────────────────────────────────
    summary_rows = []
    for name, csv_stem, _output_dir, section, cat in TESTS:
        if name not in tests_summary:
            continue
        rows = tests_summary[name]
        psnrs = [float(r["psnr"]) for r in rows]
        ssims = [float(r["ssim"]) for r in rows if r.get("ssim")]
        lpipss = [float(r["lpips"]) for r in rows if r.get("lpips")]
        mean_psnr = sum(psnrs) / len(psnrs) if psnrs else 0
        mean_ssim = sum(ssims) / len(ssims) if ssims else 0
        mean_lpips = sum(lpipss) / len(lpipss) if lpipss else 0
        # Look up memory data by display name or csv_stem-derived config key
        csv_key = csv_stem.replace("_report", "")
        mem = memory_data.get(name) or memory_data.get(csv_key) or {}
        quant_gb = mem.get("quant_bytes", 0) / (1024**3) if mem else None
        psnr_per_gb = mem.get("psnr_per_gb") if mem else None
        ssim_per_gb = (mean_ssim / quant_gb) if (quant_gb and quant_gb > 0) else None
        lpips_per_gb = (1.0 / (mean_lpips * quant_gb)) if (quant_gb and quant_gb > 0 and mean_lpips > 0) else None
        summary_rows.append({
            "name": name,
            "section": section,
            "cat": cat,
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
            "mean_lpips": mean_lpips,
            "psnr_class": psnr_class(mean_psnr),
            "quant_gb": quant_gb,
            "savings_pct": mem.get("savings_pct") if mem else None,
            "psnr_per_gb": psnr_per_gb,
            "ssim_per_gb": ssim_per_gb,
            "lpips_per_gb": lpips_per_gb,
        })

    # ── Render HTML ──────────────────────────────────────────────────────────
    html = render_html(
        summary_rows=summary_rows,
        tests_summary=tests_summary,
        tests_images=tests_images,
        error_chart_b64=error_chart_b64,
        psnr_chart_b64=psnr_chart_b64,
        ssim_chart_b64=ssim_chart_b64,
        lpips_chart_b64=lpips_chart_b64,
        efficiency_chart_b64=efficiency_chart_b64,
        memory_chart_b64=memory_chart_b64,
        has_memory=bool(memory_data),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    out_path = BASE_DIR / "ref2_combined_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path}")


def build_findings_html(summary_rows):
    """Auto-generate per-metric key findings from summary data.

    Returns a dict mapping metric name ('psnr', 'ssim', 'lpips') to an HTML
    string of <li> items.
    """
    metrics_cfg = {
        "psnr": {"key": "mean_psnr", "fmt": ".2f", "unit": "dB", "higher_better": True},
        "ssim": {"key": "mean_ssim", "fmt": ".4f", "unit": "", "higher_better": True},
        "lpips": {"key": "mean_lpips", "fmt": ".4f", "unit": "", "higher_better": False},
    }
    findings = {}

    for metric, cfg in metrics_cfg.items():
        key = cfg["key"]
        hb = cfg["higher_better"]
        unit = (" " + cfg["unit"]) if cfg["unit"] else ""
        f = cfg["fmt"]
        items = []

        valid = [r for r in summary_rows if r.get(key) is not None]
        if not valid:
            findings[metric] = ""
            continue

        best_fn = max if hb else min
        worst_fn = min if hb else max

        # Overall best/worst
        best = best_fn(valid, key=lambda r: r[key])
        worst = worst_fn(valid, key=lambda r: r[key])
        items.append(
            f'<li><span class="highlight">Best overall:</span> '
            f'<span class="ok">{best["name"]}</span> at '
            f'<span class="ok">{format(best[key], f)}{unit}</span></li>'
        )
        items.append(
            f'<li><span class="highlight">Worst overall:</span> '
            f'<span class="bad">{worst["name"]}</span> at '
            f'<span class="bad">{format(worst[key], f)}{unit}</span></li>'
        )

        # Forward ablation: most/least sensitive layer
        fwd = [r for r in valid if r.get("cat") == "fwd"]
        if fwd:
            most_sensitive = worst_fn(fwd, key=lambda r: r[key])
            least_sensitive = best_fn(fwd, key=lambda r: r[key])
            items.append(
                f'<li><span class="highlight">Most sensitive layer:</span> '
                f'<span class="bad">{most_sensitive["name"]}</span> '
                f'({format(most_sensitive[key], f)}{unit})</li>'
            )
            items.append(
                f'<li><span class="highlight">Least sensitive layer:</span> '
                f'<span class="ok">{least_sensitive["name"]}</span> '
                f'({format(least_sensitive[key], f)}{unit})</li>'
            )

        # Skip keys effect: compare int8/uint4 with vs without
        def find_row(substr):
            return next((r for r in valid if substr in r["name"].lower()), None)

        int8_no = find_row("int8 (no skip")
        int8_sk = find_row("int8 with skip")
        if int8_no and int8_sk:
            delta = int8_sk[key] - int8_no[key]
            sign = "+" if delta > 0 else ""
            better = (delta > 0) == hb
            cls = "ok" if better else "bad"
            items.append(
                f'<li><span class="highlight">Skip keys on int8:</span> '
                f'<span class="{cls}">{sign}{format(delta, f)}{unit}</span> '
                f'({format(int8_no[key], f)} → {format(int8_sk[key], f)})</li>'
            )

        uint4_no = find_row("uint4 (no skip")
        uint4_sk = find_row("uint4 with skip")
        if uint4_no and uint4_sk:
            delta = uint4_sk[key] - uint4_no[key]
            sign = "+" if delta > 0 else ""
            better = (delta > 0) == hb
            cls = "ok" if better else "bad"
            items.append(
                f'<li><span class="highlight">Skip keys on uint4:</span> '
                f'<span class="{cls}">{sign}{format(delta, f)}{unit}</span> '
                f'({format(uint4_no[key], f)} → {format(uint4_sk[key], f)})</li>'
            )

        # Component isolation: transformer-only vs text-encoder-only
        iso = [r for r in valid if r.get("cat") == "iso"]
        if len(iso) >= 2:
            trans = find_row("transformer only")
            tenc = find_row("text encoder only")
            if trans and tenc:
                t_better = (trans[key] > tenc[key]) == hb
                cls_t = "ok" if t_better else "bad"
                cls_e = "bad" if t_better else "ok"
                items.append(
                    f'<li><span class="highlight">Component isolation:</span> '
                    f'transformer-only <span class="{cls_t}">{format(trans[key], f)}{unit}</span> vs '
                    f'text-encoder-only <span class="{cls_e}">{format(tenc[key], f)}{unit}</span></li>'
                )

        findings[metric] = "\n        ".join(items)

    return findings


def render_html(*, summary_rows, tests_summary, tests_images,
                error_chart_b64, psnr_chart_b64, ssim_chart_b64,
                lpips_chart_b64, efficiency_chart_b64, memory_chart_b64,
                has_memory, timestamp):
    """Build the full HTML string."""

    # ── Summary table rows ───────────────────────────────────────────────────
    def fmt(v, f=".2f"):
        return format(v, f)

    # Group by section
    sections = {}
    for r in summary_rows:
        sections.setdefault(r["section"], []).append(r)

    mem_colspan = 8 if has_memory else 5
    summary_html = ""
    for section, rows in sections.items():
        summary_html += f'<tr class="section-header"><td colspan="{mem_colspan}">{section}</td></tr>\n'
        for r in rows:
            pc = r["psnr_class"]
            mem_cells = ""
            # Build data attributes for JS metric switcher
            qgb = r.get("quant_gb")
            sav = r.get("savings_pct")
            ppg = r.get("psnr_per_gb")
            spg = r.get("ssim_per_gb")
            lpg = r.get("lpips_per_gb")
            data_attrs = (
                f'data-psnr="{r["mean_psnr"]:.4f}" '
                f'data-ssim="{r["mean_ssim"]:.6f}" '
                f'data-lpips="{r["mean_lpips"]:.6f}"'
            )
            if ppg is not None:
                data_attrs += f' data-psnr-per-gb="{ppg:.4f}"'
            if spg is not None:
                data_attrs += f' data-ssim-per-gb="{spg:.6f}"'
            if lpg is not None:
                data_attrs += f' data-lpips-per-gb="{lpg:.4f}"'
            data_attrs += f' data-name="{esc(r["name"])}"'
            if has_memory:
                if qgb is not None:
                    sav_class = "excellent" if sav >= 60 else "good" if sav >= 40 else "fair" if sav >= 20 else "poor"
                    eff_val = ppg if ppg is not None else 0
                    mem_cells = f"""
    <td class="metric" data-col="quant_gb">{qgb:.1f}</td>
    <td class="metric {sav_class}" data-col="savings">{sav:.1f}%</td>
    <td class="metric" data-col="eff_per_gb">{eff_val:.2f}</td>"""
                else:
                    mem_cells = (
                        '<td class="metric" data-col="quant_gb">—</td>'
                        '<td class="metric" data-col="savings">—</td>'
                        '<td class="metric" data-col="eff_per_gb">—</td>'
                    )
            summary_html += f"""<tr {data_attrs}>
    <td>{r['name']}</td>
    <td class="metric {pc}" data-col="psnr">{fmt(r['mean_psnr'])}</td>
    <td class="metric" data-col="ssim">{fmt(r['mean_ssim'], '.4f')}</td>
    <td class="metric" data-col="lpips">{fmt(r['mean_lpips'], '.4f')}</td>{mem_cells}
    <td data-col="bar"><div class="bar-cell"><div class="bar" style="width:{min(r['mean_psnr']/35*100, 100):.0f}%;background:{psnr_color(r['mean_psnr'])}"></div></div></td>
</tr>\n"""

    # ── Per-test detail + images ─────────────────────────────────────────────
    details_html = ""
    for name, _csv_stem, _output_dir, _section, _cat in TESTS:
        if name not in tests_summary:
            continue
        rows = tests_summary[name]
        pairs = tests_images.get(name, [])

        details_html += f'<div class="test-block" id="test-{css_id(name)}">\n'
        details_html += f'<h3>{name}</h3>\n'
        details_html += '<table class="detail-table"><thead><tr>'
        details_html += '<th>Prompt</th><th>Seed</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th>'
        details_html += '<th>Mean Step MSE</th><th>Final MSE</th><th>Cosine Sim</th>'
        details_html += '</tr></thead><tbody>\n'
        for r in rows:
            psnr_v = float(r["psnr"])
            pc = psnr_class(psnr_v)
            prompt_short = r["prompt"][:60] + "..." if len(r["prompt"]) > 63 else r["prompt"]
            details_html += f"""<tr>
    <td class="prompt-cell" title="{esc(r['prompt'])}">{esc(prompt_short)}</td>
    <td>{r['seed']}</td>
    <td class="metric {pc}">{float(r['psnr']):.2f}</td>
    <td class="metric">{float(r.get('ssim', 0) or 0):.4f}</td>
    <td class="metric">{float(r.get('lpips', 0) or 0):.4f}</td>
    <td class="metric">{float(r.get('mean_step_mse', 0)):.6f}</td>
    <td class="metric">{float(r.get('final_step_mse', 0)):.6f}</td>
    <td class="metric">{float(r.get('mean_cosine_sim', 0)):.4f}</td>
</tr>\n"""
        details_html += '</tbody></table>\n'

        if pairs:
            details_html += '<div class="images-grid">\n'
            for p in pairs:
                details_html += f"""<div class="image-pair">
    <div class="pair-header">Prompt {p['prompt_idx']}, Seed {p['seed']} — PSNR {p['psnr']:.2f} dB</div>
    <div class="imgs">
        <div><img src="data:image/png;base64,{p['fp16_b64']}" alt="BF16"><div class="img-label">BF16 Baseline</div></div>
        <div><img src="data:image/png;base64,{p['quant_b64']}" alt="Quantized"><div class="img-label">Quantized</div></div>
    </div>
</div>\n"""
            details_html += '</div>\n'

        details_html += '</div>\n'

    # ── Best / worst stats ───────────────────────────────────────────────────
    best = max(summary_rows, key=lambda r: r["mean_psnr"])
    worst = min(summary_rows, key=lambda r: r["mean_psnr"])
    best_eff = max(summary_rows, key=lambda r: r.get("psnr_per_gb") or 0)
    fp16_gb = summary_rows[0].get("quant_gb")  # placeholder, overridden below
    for r in summary_rows:
        if r.get("savings_pct") is not None:
            fp16_gb = (r["quant_gb"] / (1 - r["savings_pct"] / 100)) if r["savings_pct"] < 100 else 0
            break

    # ── Build conditional HTML fragments ─────────────────────────────────────
    memory_nav_link = '<a href="#memory-charts">Memory</a>' if has_memory else ""
    memory_th = '<th>Size (GB)</th><th>Savings</th><th id="eff-header">PSNR/GB</th>' if has_memory else ""

    if has_memory:
        memory_stat_cards = (
            '<div class="stat-card">'
            f'<div class="value accent">{fp16_gb:.1f} GB</div>'
            '<div class="label">FP16 Baseline Size</div>'
            '<div class="sub">Full precision model</div>'
            '</div>'
            '<div class="stat-card">'
            f'<div class="value best">{fmt(best_eff.get("psnr_per_gb", 0))} dB/GB</div>'
            '<div class="label">Best PSNR/GB Efficiency</div>'
            f'<div class="sub">{best_eff["name"]}</div>'
            '</div>'
        )
    else:
        memory_stat_cards = ""

    if memory_chart_b64:
        memory_chart_html = (
            '<h2 id="memory-charts">Memory Footprint Analysis</h2>'
            f'<div class="chart-section"><img src="data:image/png;base64,{memory_chart_b64}" alt="Memory footprint chart"></div>'
        )
    else:
        memory_chart_html = ""

    if efficiency_chart_b64:
        efficiency_chart_html = (
            '<h2>Quality vs Memory Efficiency</h2>'
            f'<div class="chart-section"><img src="data:image/png;base64,{efficiency_chart_b64}" alt="Quality vs memory efficiency scatter"></div>'
        )
    else:
        efficiency_chart_html = ""

    if error_chart_b64:
        error_chart_html = (
            '<h2 id="error-chart">Error Accumulation Across Denoising Steps</h2>'
            f'<div class="chart-section"><img src="data:image/png;base64,{error_chart_b64}" alt="Error accumulation chart"></div>'
        )
    else:
        error_chart_html = ""

    # ── Build per-metric findings ────────────────────────────────────────────
    findings_data = build_findings_html(summary_rows)

    findings_divs = ""
    for m_key, _m_label in [("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS")]:
        display = "" if m_key == "psnr" else "display:none"
        content = findings_data.get(m_key, "")
        findings_divs += (
            f'<div id="findings-{m_key}" class="findings-content" style="{display}">\n'
            f'    <ul>\n        {content}\n    </ul>\n'
            f'</div>\n'
        )

    # ── Assemble ─────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Z-Image Quantization Ablation Report</title>
<style>
:root {{
    --bg: #0f0f1a; --surface: #1a1a2e; --surface2: #16213e; --surface3: #0f3460;
    --text: #e0e0e0; --text-dim: #8a8a9a; --accent: #00b4d8; --accent2: #48cae4;
    --green: #40916c; --yellow: #b8860b; --orange: #e76f51; --red: #d62828;
    --border: rgba(255,255,255,0.06);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 2rem; line-height: 1.6; }}
.container {{ max-width: 1600px; margin: 0 auto; }}

/* Header */
.header {{ margin-bottom: 2.5rem; }}
h1 {{ color: var(--accent); font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; }}
h2 {{ color: var(--accent); margin: 2.5rem 0 1rem; font-size: 1.4rem; font-weight: 600; }}
h3 {{ color: var(--accent2); margin: 1.5rem 0 0.75rem; font-size: 1.15rem; }}
.meta {{ color: var(--text-dim); font-size: 0.9rem; margin-top: 0.5rem; }}

/* Stats cards */
.stats-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
.stat-card {{ background: var(--surface); border-radius: 10px; padding: 1.25rem 1.5rem; border: 1px solid var(--border); }}
.stat-card .value {{ font-size: 1.8rem; font-weight: 700; font-family: "JetBrains Mono", "Fira Code", monospace; }}
.stat-card .label {{ font-size: 0.82rem; color: var(--text-dim); margin-top: 0.2rem; }}
.stat-card .sub {{ font-size: 0.78rem; color: var(--text-dim); margin-top: 0.25rem; }}
.stat-card .value.best {{ color: var(--green); }}
.stat-card .value.worst {{ color: var(--red); }}
.stat-card .value.accent {{ color: var(--accent); }}

/* Summary table */
.summary-table {{ width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 10px; overflow: hidden; margin-bottom: 2rem; border: 1px solid var(--border); }}
.summary-table th {{ background: var(--surface3); color: var(--accent); padding: 0.8rem 1rem; text-align: left; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
.summary-table td {{ padding: 0.65rem 1rem; border-top: 1px solid var(--border); font-size: 0.88rem; }}
.summary-table tr:hover {{ background: rgba(255,255,255,0.02); }}
.section-header td {{ background: var(--surface2); color: var(--accent2); font-weight: 600; font-size: 0.85rem; padding: 0.5rem 1rem; letter-spacing: 0.03em; }}
.metric {{ font-family: "JetBrains Mono", "Fira Code", monospace; text-align: right; }}
.excellent {{ color: var(--green); font-weight: 600; }}
.good {{ color: #6aaa64; }}
.fair {{ color: var(--yellow); }}
.poor {{ color: var(--orange); }}
.terrible {{ color: var(--red); font-weight: 600; }}
.bar-cell {{ width: 120px; }}
.bar {{ height: 6px; border-radius: 3px; min-width: 2px; }}

/* Charts */
.chart-section {{ background: var(--surface); border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid var(--border); text-align: center; }}
.chart-section img {{ max-width: 100%; height: auto; border-radius: 6px; }}
.charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
@media (max-width: 1200px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}

/* Test detail blocks */
.test-block {{ background: var(--surface); border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid var(--border); }}
.detail-table {{ width: 100%; border-collapse: collapse; margin-bottom: 1rem; }}
.detail-table th {{ background: var(--surface2); color: var(--accent); padding: 0.6rem 0.8rem; text-align: left; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
.detail-table td {{ padding: 0.5rem 0.8rem; border-top: 1px solid var(--border); font-size: 0.82rem; }}
.prompt-cell {{ max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

/* Image pairs */
.images-grid {{ display: grid; grid-template-columns: 1fr; gap: 1.5rem; }}
.image-pair {{ background: var(--surface2); border-radius: 8px; padding: 0.75rem; }}
.pair-header {{ font-size: 0.82rem; color: var(--text-dim); margin-bottom: 0.5rem; font-family: monospace; }}
.imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }}
.imgs img {{ width: 100%; border-radius: 4px; display: block; }}
.img-label {{ text-align: center; font-size: 0.72rem; color: var(--text-dim); margin-top: 0.25rem; }}

/* Key findings */
.findings {{ background: var(--surface); border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid var(--border); }}
.findings ul {{ margin: 0.5rem 0 0 1.5rem; }}
.findings li {{ margin-bottom: 0.4rem; font-size: 0.92rem; }}
.findings .highlight {{ color: var(--accent); font-weight: 600; }}
.findings .bad {{ color: var(--red); font-weight: 600; }}
.findings .ok {{ color: var(--green); font-weight: 600; }}

/* Nav */
.nav {{ position: sticky; top: 0; z-index: 100; background: var(--bg); padding: 0.75rem 0; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }}
.nav a {{ color: var(--accent); text-decoration: none; font-size: 0.82rem; margin-right: 1.2rem; }}
.nav a:hover {{ text-decoration: underline; }}

/* Metric selector */
.metric-selector {{ display: flex; gap: 0.25rem; margin-bottom: 1rem; }}
.metric-selector button {{ background: var(--surface); color: var(--text-dim); border: 1px solid var(--border); border-radius: 20px; padding: 0.4rem 1.1rem; font-size: 0.82rem; font-weight: 500; cursor: pointer; transition: all 0.15s ease; font-family: inherit; }}
.metric-selector button:hover {{ background: var(--surface2); color: var(--text); }}
.metric-selector button.active {{ background: var(--accent); color: #0f0f1a; border-color: var(--accent); font-weight: 600; }}
.metric-selector .sep {{ width: 1px; background: var(--border); margin: 0 0.5rem; align-self: stretch; }}

/* Print */
@media print {{
    :root {{ --bg: #fff; --surface: #f8f8f8; --surface2: #eee; --surface3: #ddd; --text: #1a1a1a; --text-dim: #666; --border: #ddd; }}
    body {{ padding: 1rem; }}
    .nav {{ display: none; }}
}}
</style>
</head>
<body>
<div class="container">

<div class="nav">
    <a href="#overview">Overview</a>
    <a href="#summary">Summary Table</a>
    <a href="#charts">Charts</a>
    {memory_nav_link}
    <a href="#error-chart">Error Accumulation</a>
    <a href="#details">Detailed Results</a>
    <a href="#findings">Key Findings</a>
</div>

<div class="header" id="overview">
    <h1>Z-Image Quantization Ablation Report</h1>
    <p class="meta">Model: <strong>Tongyi-MAI/Z-Image</strong> | Generated: {timestamp} | {len(summary_rows)} configurations | 2 prompts &times; seed 42 | 20 steps | bf16 baseline | CPU offload | No skip keys</p>
</div>

<div class="stats-row">
    <div class="stat-card" id="best-card">
        <div class="value best"><span id="best-val">{fmt(best['mean_psnr'])} dB</span></div>
        <div class="label"><span id="best-label">Best PSNR</span></div>
        <div class="sub"><span id="best-name">{best['name']}</span></div>
    </div>
    <div class="stat-card" id="worst-card">
        <div class="value worst"><span id="worst-val">{fmt(worst['mean_psnr'])} dB</span></div>
        <div class="label"><span id="worst-label">Worst PSNR</span></div>
        <div class="sub"><span id="worst-name">{worst['name']}</span></div>
    </div>
    <div class="stat-card">
        <div class="value accent">{len(summary_rows)}</div>
        <div class="label">Configurations Tested</div>
        <div class="sub">Reference + skip keys + forward ablation</div>
    </div>
    <div class="stat-card">
        <div class="value accent">uint4 g32</div>
        <div class="label">Base Quantization</div>
        <div class="sub">Forward ablation dtype</div>
    </div>
    {memory_stat_cards}
</div>

<div class="findings" id="findings">
    <h2 style="margin-top:0">Key Findings</h2>
{findings_divs}
</div>

<h2 id="summary">Summary — All Configurations</h2>
<div class="metric-selector">
    <button class="active" data-metric="psnr">PSNR</button>
    <button data-metric="ssim">SSIM</button>
    <button data-metric="lpips">LPIPS</button>
    {'<div class="sep"></div><button data-metric="psnr_per_gb">PSNR/GB</button><button data-metric="ssim_per_gb">SSIM/GB</button><button data-metric="lpips_per_gb">1/(LPIPS·GB)</button>' if has_memory else ""}
</div>
<table class="summary-table" id="summary-table">
<thead><tr>
    <th>Configuration</th><th>Mean PSNR (dB)</th><th>Mean SSIM</th><th>Mean LPIPS</th>{memory_th}<th>PSNR</th>
</tr></thead>
<tbody>
{summary_html}
</tbody>
</table>

<h2 id="charts">Metric Comparison Charts</h2>
<div class="charts-grid">
    <div class="chart-section"><img src="data:image/png;base64,{psnr_chart_b64}" alt="PSNR chart"></div>
    <div class="chart-section"><img src="data:image/png;base64,{ssim_chart_b64}" alt="SSIM chart"></div>
</div>
<div class="chart-section"><img src="data:image/png;base64,{lpips_chart_b64}" alt="LPIPS chart"></div>

{memory_chart_html}

{efficiency_chart_html}

{error_chart_html}

<h2 id="details">Detailed Results Per Configuration</h2>
{details_html}

<div style="text-align:center;color:var(--text-dim);font-size:0.78rem;margin-top:3rem;padding:1rem;">
    Generated by SDNQ quality benchmark | {timestamp}
</div>

</div>
<script>
(function() {{
    const METRICS = {{
        psnr:        {{ attr: "psnr", label: "PSNR", unit: "dB", higherBetter: true,
                        fmt: v => v.toFixed(2), baseMetric: "psnr",
                        thresholds: [[35,"excellent"],[25,"good"],[18,"fair"],[12,"poor"]] }},
        ssim:        {{ attr: "ssim", label: "SSIM", unit: "", higherBetter: true,
                        fmt: v => v.toFixed(4), baseMetric: "ssim",
                        thresholds: [[0.95,"excellent"],[0.85,"good"],[0.6,"fair"],[0.3,"poor"]] }},
        lpips:       {{ attr: "lpips", label: "LPIPS", unit: "", higherBetter: false,
                        fmt: v => v.toFixed(4), baseMetric: "lpips",
                        thresholds: [[0.1,"excellent"],[0.2,"good"],[0.5,"fair"],[0.8,"poor"]] }},
        psnr_per_gb: {{ attr: "psnr-per-gb", label: "PSNR/GB", unit: "dB/GB", higherBetter: true,
                        fmt: v => v.toFixed(2), baseMetric: "psnr", isEfficiency: true,
                        thresholds: [[2.2,"excellent"],[1.8,"good"],[1.3,"fair"],[0.8,"poor"]] }},
        ssim_per_gb: {{ attr: "ssim-per-gb", label: "SSIM/GB", unit: "/GB", higherBetter: true,
                        fmt: v => v.toFixed(4), baseMetric: "ssim", isEfficiency: true,
                        thresholds: [[0.085,"excellent"],[0.07,"good"],[0.05,"fair"],[0.02,"poor"]] }},
        lpips_per_gb: {{ attr: "lpips-per-gb", label: "1/(LPIPS\u00b7GB)", unit: "", higherBetter: true,
                        fmt: v => v.toFixed(3), baseMetric: "lpips", isEfficiency: true,
                        thresholds: [[0.45,"excellent"],[0.30,"good"],[0.20,"fair"],[0.10,"poor"]] }},
    }};

    function classify(val, metric) {{
        const m = METRICS[metric];
        const t = m.thresholds;
        if (m.higherBetter) {{
            for (const [thresh, cls] of t) if (val >= thresh) return cls;
        }} else {{
            for (const [thresh, cls] of t) if (val <= thresh) return cls;
        }}
        return "terrible";
    }}

    const colorMap = {{
        excellent: "var(--green)", good: "#6aaa64", fair: "var(--yellow)",
        poor: "var(--orange)", terrible: "var(--red)"
    }};

    function switchMetric(metric) {{
        const m = METRICS[metric];
        if (!m) return;

        const table = document.getElementById("summary-table");
        const tbody = table.querySelector("tbody");
        const allRows = Array.from(tbody.querySelectorAll("tr"));

        // Group rows by sections: each section starts with a section-header row
        const groups = [];
        let currentGroup = null;
        for (const row of allRows) {{
            if (row.classList.contains("section-header")) {{
                currentGroup = {{ header: row, dataRows: [] }};
                groups.push(currentGroup);
            }} else if (currentGroup) {{
                currentGroup.dataRows.push(row);
            }}
        }}

        // Sort data rows within each group
        for (const g of groups) {{
            g.dataRows.sort((a, b) => {{
                const va = parseFloat(a.dataset[toCamel(m.attr)]) || 0;
                const vb = parseFloat(b.dataset[toCamel(m.attr)]) || 0;
                return m.higherBetter ? vb - va : va - vb;
            }});
        }}

        // Collect all data row values for bar scaling
        const allVals = [];
        for (const g of groups) {{
            for (const row of g.dataRows) {{
                const v = parseFloat(row.dataset[toCamel(m.attr)]);
                if (!isNaN(v)) allVals.push(v);
            }}
        }}
        const maxVal = Math.max(...allVals, 1e-9);

        // Rebuild tbody in sorted order and update cells
        tbody.innerHTML = "";
        for (const g of groups) {{
            tbody.appendChild(g.header);
            for (const row of g.dataRows) {{
                const val = parseFloat(row.dataset[toCamel(m.attr)]);
                const cls = isNaN(val) ? "" : classify(val, metric);

                // Update the primary metric column coloring (all metric cells)
                for (const td of row.querySelectorAll("td[data-col]")) {{
                    const col = td.dataset.col;
                    // Remove old quality classes
                    td.classList.remove("excellent", "good", "fair", "poor", "terrible");
                    // Apply class to the column matching the active metric
                    if (col === metric.replace("_per_gb", "_per_gb") && !isNaN(val)) {{
                        // Color the cell of the selected metric
                    }}
                    if (col === metric && !isNaN(val)) {{
                        td.classList.add(cls);
                    }} else if (col === "psnr" && metric !== "psnr") {{
                        // Keep psnr uncolored when not selected
                    }}
                }}

                // Update bar
                const barTd = row.querySelector("td[data-col='bar']");
                if (barTd && !isNaN(val)) {{
                    const pct = Math.min((val / maxVal) * 100, 100);
                    const barDiv = barTd.querySelector(".bar");
                    if (barDiv) {{
                        // For LPIPS, invert: best (lowest) gets longest bar
                        const barPct = m.higherBetter ? pct : Math.min(((maxVal - val + allVals.reduce((a,b)=>Math.min(a,b), Infinity)) / maxVal) * 100, 100);
                        const displayPct = m.higherBetter
                            ? Math.min((val / maxVal) * 100, 100)
                            : Math.min(((1 - val / maxVal) + 0.02) * 100, 100);
                        barDiv.style.width = displayPct.toFixed(0) + "%";
                        barDiv.style.background = colorMap[cls] || colorMap.fair;
                    }}
                }}

                tbody.appendChild(row);
            }}
        }}

        // Update best/worst stat cards
        if (allVals.length > 0) {{
            let bestRow = null, worstRow = null;
            let bestVal = m.higherBetter ? -Infinity : Infinity;
            let worstVal = m.higherBetter ? Infinity : -Infinity;
            for (const g of groups) {{
                for (const row of g.dataRows) {{
                    const v = parseFloat(row.dataset[toCamel(m.attr)]);
                    if (isNaN(v)) continue;
                    if (m.higherBetter ? v > bestVal : v < bestVal) {{
                        bestVal = v;
                        bestRow = row;
                    }}
                    if (m.higherBetter ? v < worstVal : v > worstVal) {{
                        worstVal = v;
                        worstRow = row;
                    }}
                }}
            }}
            const unit = m.unit ? " " + m.unit : "";
            const bv = document.getElementById("best-val");
            const bl = document.getElementById("best-label");
            const bn = document.getElementById("best-name");
            const wv = document.getElementById("worst-val");
            const wl = document.getElementById("worst-label");
            const wn = document.getElementById("worst-name");
            if (bv && bestRow) {{
                bv.textContent = m.fmt(bestVal) + unit;
                bl.textContent = "Best " + m.label;
                bn.textContent = bestRow.dataset.name || "";
            }}
            if (wv && worstRow) {{
                wv.textContent = m.fmt(worstVal) + unit;
                wl.textContent = "Worst " + m.label;
                wn.textContent = worstRow.dataset.name || "";
            }}
        }}

        // Update efficiency column header + cells
        const effHeader = document.getElementById("eff-header");
        if (effHeader) {{
            if (m.isEfficiency) {{
                effHeader.textContent = m.label;
                for (const g of groups) {{
                    for (const row of g.dataRows) {{
                        const effTd = row.querySelector("td[data-col='eff_per_gb']");
                        if (effTd) {{
                            const v = parseFloat(row.dataset[toCamel(m.attr)]);
                            effTd.textContent = isNaN(v) ? "\u2014" : m.fmt(v);
                        }}
                    }}
                }}
            }} else {{
                effHeader.textContent = "PSNR/GB";
                for (const g of groups) {{
                    for (const row of g.dataRows) {{
                        const effTd = row.querySelector("td[data-col='eff_per_gb']");
                        if (effTd) {{
                            const v = parseFloat(row.dataset[toCamel("psnr-per-gb")]);
                            effTd.textContent = isNaN(v) ? "\u2014" : v.toFixed(2);
                        }}
                    }}
                }}
            }}
        }}

        // Update findings visibility
        const baseMet = m.baseMetric || metric;
        document.querySelectorAll(".findings-content").forEach(div => {{
            div.style.display = div.id === "findings-" + baseMet ? "" : "none";
        }});

        // Update active button
        document.querySelectorAll(".metric-selector button").forEach(btn => {{
            btn.classList.toggle("active", btn.dataset.metric === metric);
        }});
    }}

    function toCamel(s) {{
        return s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    }}

    // Bind buttons
    document.querySelectorAll(".metric-selector button").forEach(btn => {{
        btn.addEventListener("click", () => switchMetric(btn.dataset.metric));
    }});

    // Apply initial coloring for PSNR (already set server-side, but ensures consistency)
    // No action needed — server-side rendering handles initial state.
}})();
</script>
</body>
</html>"""


def css_id(name):
    return name.lower().replace(" ", "-").replace("+", "").replace("(", "").replace(")", "")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    main()
