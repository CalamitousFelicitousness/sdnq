# Sensitivity Analysis Report Generation

The `scripts/analyze_quantization_sensitivity.py` script can generate HTML and PDF reports in addition to CSV output and terminal summaries.

## Quick Start

```bash
# Default: HTML with embedded chart
python scripts/analyze_quantization_sensitivity.py \
    --model-id Tongyi-MAI/Z-Image \
    --components transformer:ZImageTransformer2DModel

# Plain HTML (no chart)
python scripts/analyze_quantization_sensitivity.py \
    --model-id Tongyi-MAI/Z-Image \
    --components transformer:ZImageTransformer2DModel \
    --report-format html

# PDF output
python scripts/analyze_quantization_sensitivity.py \
    --model-id Tongyi-MAI/Z-Image \
    --components transformer:ZImageTransformer2DModel \
    --report-format pdf

# Custom output path
python scripts/analyze_quantization_sensitivity.py \
    --model-id Tongyi-MAI/Z-Image \
    --components transformer:ZImageTransformer2DModel \
    --report-output my_report.html
```

## CLI Arguments

### Model & Input

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-id` | `Tongyi-MAI/Z-Image` | HuggingFace model ID or local path to the model |
| `--components` | `transformer:ZImageTransformer2DModel` | One or more `subfolder:ClassName` pairs to analyze. The class is resolved from `diffusers` or `transformers` |
| `--cache-dir` | `~/database/models/huggingface` | HuggingFace cache directory for downloading/loading models |
| `--device` | `cpu` | Computation device (e.g. `cpu`, `cuda`, `cuda:0`) |

### Quantization

| Argument | Default | Description |
|----------|---------|-------------|
| `--dtypes` | `uint4 uint6 uint8 int8` | Space-separated list of quantization dtypes to test. The first dtype is treated as the "primary" dtype for sorting and classification |
| `--group-size` | `0` | One or more group sizes to test. `0` = auto (defaults to 32), `-1` = disabled (tensorwise). Multiple values (e.g. `--group-size 16 32 64`) enables group-size comparison mode |
| `--svd-rank` | `32` | SVD rank(s) to test. Multiple values (e.g. `--svd-rank 16 32 64`) enables rank comparison in auto-config grid search |
| `--svd-steps` | `8` | Number of iteration steps for SVD computation |
| `--no-svd` | off | Disable SVD mode testing (only test raw quantization) |
| `--no-raw` | off | Disable raw mode testing (only test SVD quantization) |

At least one of raw or SVD must be enabled. If both `--no-svd` and `--no-raw` are passed, the script exits with an error.

#### Group Size Comparison Mode

When multiple values are passed to `--group-size`, the script switches from dtype-comparison mode to group-size comparison mode:

```bash
python scripts/analyze_quantization_sensitivity.py \
    --model-id Tongyi-MAI/Z-Image \
    --components transformer:ZImageTransformer2DModel \
    --dtypes uint4 \
    --group-size 16 32 64
```

In this mode:

- The **heatmap table columns** become `g16`, `g32`, `g64` (one per group size) instead of dtype columns.
- The **primary dtype** (first in `--dtypes`) is fixed for all comparisons.
- Layer classification checks whether any tested group size brings NMSE under the promote threshold.
- The **config snippet** outputs a `modules_group_size_dict` (mapping group sizes to layer lists) instead of `modules_dtype_dict`.
- The **SVD comparison table** is omitted in group-size comparison mode.

When a single group size is passed (the default), behavior is unchanged — dtypes are the column axis.

### Analysis Mode

| Argument | Default | Description |
|----------|---------|-------------|
| `--auto-config` | off | Run grid search across all tested dtypes, group sizes, and SVD modes to find the cheapest quantization configuration per layer that keeps NMSE under the promote threshold |

### Classification Thresholds

| Argument | Default | Description |
|----------|---------|-------------|
| `--skip-threshold` | `0.1` | NMSE above which a layer should be excluded from quantization entirely |
| `--promote-threshold` | `0.01` | NMSE above which a layer needs a higher-precision dtype than the primary |

Layers are classified using multi-dtype logic:
- **OK** — primary dtype NMSE is at or below the promote threshold
- **>dtype** (PROMOTE) — primary dtype NMSE exceeds promote threshold, but another tested dtype brings it under
- **SKIP** — all tested dtypes exceed the skip threshold, or no tested dtype can bring NMSE under the promote threshold

### Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | `quant_sensitivity_report.csv` | CSV output path for the full per-layer results |
| `--top-n` | `30` | Number of worst layers to display in the summary table and report |
| `--report-format` | `html-png` | Report format: `html` (no chart), `html-png` (with embedded chart), or `pdf` |
| `--report-output` | auto | Report file path. Defaults to `<output-stem>_report.html` or `.pdf` |
| `--pipeline-config-output` | auto | JSON output path for per-component pipeline config. Defaults to `<output-stem>_pipeline_config.json`. Only used with `--auto-config` |

The report is generated automatically after the CSV and terminal summary. The output path is derived from the `--output` CSV path unless `--report-output` is specified.

## Report Formats

### `html` — Plain HTML

Generates a self-contained HTML file with:
- Statistics summary card (layer counts, OK/PROMOTE/SKIP badges, NMSE stats)
- Heatmap table of most sensitive layers with per-cell color coding
- SVD vs Raw comparison table (when both modes are tested)
- Copy-pasteable SDNQ config snippet with syntax highlighting

No external dependencies beyond Jinja2.

### `html-png` — HTML with Chart (default)

Same as `html` plus an embedded matplotlib bar chart showing the NMSE distribution of the top-N most sensitive layers. The chart uses:
- Horizontal bars colored by NMSE severity (green/yellow/orange/red)
- Dashed reference lines at skip and promote thresholds
- Logarithmic X-axis when the NMSE range exceeds 100x
- INF values rendered at 2x skip threshold with annotation
- Dark theme matching the HTML report

Requires matplotlib.

### `pdf` — PDF Document

Renders the `html-png` content to PDF via weasyprint. Uses `@media print` CSS rules that switch to a white background with dark text for print readability.

Requires both matplotlib and weasyprint.

## Dependencies

Install with pip extras:

```bash
# HTML + chart support
pip install sdnq[report]

# Full PDF support
pip install sdnq[report-pdf]
```

Or install individually:

```bash
pip install Jinja2 matplotlib        # html / html-png
pip install Jinja2 matplotlib weasyprint  # pdf
```

### Fallback Behavior

The report system degrades gracefully when optional dependencies are missing:

| Missing | Behavior |
|---------|----------|
| Jinja2 | Report generation skipped entirely with a warning |
| matplotlib | `html-png` falls back to `html` (no chart) with a warning |
| weasyprint | `pdf` falls back to `html-png` or `html` with a warning |

Terminal output (rich/plain) and CSV output are never affected.

## Template

The HTML template is at `scripts/report_template.html`. It uses Jinja2 templating with all CSS inlined. Key design choices:

- **Dark theme** with CSS custom properties (`--bg`, `--surface`, `--text`, etc.)
- **Sticky table headers** for scrolling large tables
- **Monospace numbers** for alignment in data cells
- **NMSE color mapping**: green (low error) through yellow/orange to red (high error / INF)
- **Print override**: `@media print` block switches to white background with dark text
- **Responsive**: `max-width: 1400px` container, `overflow-x: auto` on tables

### Color Scale

NMSE values are mapped to background colors:

| NMSE Range | Color | Meaning |
|-----------|-------|---------|
| `<= promote * 0.1` | Dark green `#2d6a4f` | Excellent |
| `<= promote` | Green `#40916c` | Good |
| `<= skip * 0.5` | Dark yellow `#b8860b` | Marginal |
| `<= skip` | Orange `#e76f51` | Poor |
| `> skip` or INF | Red `#d62828` | Unacceptable |

## Auto-Config Mode

When `--auto-config` is passed, the script performs a grid search across all tested dtypes, group sizes, and SVD modes to find the cheapest quantization configuration per layer that keeps NMSE under the promote threshold.

### Usage

```bash
# Search across two dtypes and two group sizes
python scripts/analyze_quantization_sensitivity.py \
    --model-id Tongyi-MAI/Z-Image \
    --components transformer:ZImageTransformer2DModel \
    --dtypes uint4 uint8 \
    --group-size 0 32 \
    --auto-config

# Wider search with more dtypes and group sizes
python scripts/analyze_quantization_sensitivity.py \
    --model-id Tongyi-MAI/Z-Image \
    --components transformer:ZImageTransformer2DModel \
    --dtypes uint4 uint6 uint8 int8 \
    --group-size 0 32 64 \
    --auto-config

# Minimal grid (single dtype, SVD is the only variable)
python scripts/analyze_quantization_sensitivity.py \
    --model-id Tongyi-MAI/Z-Image \
    --components transformer:ZImageTransformer2DModel \
    --dtypes uint4 \
    --group-size 0 \
    --auto-config

# Exclude SVD from the grid
python scripts/analyze_quantization_sensitivity.py \
    --model-id Tongyi-MAI/Z-Image \
    --components transformer:ZImageTransformer2DModel \
    --dtypes uint4 uint8 int8 \
    --group-size 0 32 \
    --no-svd \
    --auto-config

# Tighter quality threshold (only accept very low error)
python scripts/analyze_quantization_sensitivity.py \
    --model-id Tongyi-MAI/Z-Image \
    --components transformer:ZImageTransformer2DModel \
    --dtypes uint4 uint8 int8 \
    --group-size 0 32 \
    --promote-threshold 0.001 \
    --auto-config
```

The grid searched is every combination of `dtypes × group_sizes × {raw, svd@rank1, svd@rank2, ...}`. When multiple SVD ranks are passed via `--svd-rank`, each rank is tested as a separate grid axis. For each layer, the cheapest combination that keeps NMSE under `--promote-threshold` is selected. Layers where nothing passes are marked SKIP and counted at FP16 cost.

### How It Works

For each layer, auto-config:
1. Collects all result combinations (dtype × group_size × SVD mode) from the analysis
2. Filters to combinations where NMSE ≤ promote threshold ("passing")
3. Estimates storage cost for each passing combination using a cost model
4. Picks the cheapest combination. Tie-break: prefer no SVD, then larger group size, then alphabetical dtype
5. If no combination passes, marks the layer as SKIP (kept at FP16)

### Cost Model

The `estimate_layer_bytes()` function estimates total storage for one quantized layer:

```
total = quantized_weight + scale_overhead + zero_point_overhead + svd_overhead
```

- **Quantized weight**: `numel × num_bits / 8`
- **Scale**: `ceil(numel / group_size) × 2` bytes (FP16). Tensorwise (group_size ≤ 0): `out_features × 2`
- **Zero-point** (unsigned dtypes only): same shape as scale
- **SVD factors**: `(out_features × rank + rank × in_features) × 2` bytes (FP16)

### Output

The auto-config mode produces:

- **Terminal table**: Per-layer rows with recommended dtype, group size, SVD, NMSE, estimated size, and status (OK/SKIP)
- **Size estimates**: FP16 baseline, uniform config total, recommended total, and savings percentages
- **Config snippet**: Ready-to-use `SDNQConfig` with the most common dtype/group_size as base, `modules_to_not_convert` for SKIP layers, `modules_dtype_dict` for layers needing a different dtype, `modules_group_size_dict` for layers needing a different group size, and `modules_svd_dict` for layers benefiting from SVD
- **HTML report**: Auto-config table replaces the heatmap table; size stats appear in the stats card
- **Chart**: Horizontal bar chart of recommended NMSE per layer

### Config Snippet Per-Layer Overrides

`SDNQConfig` supports per-layer dtype overrides (`modules_dtype_dict`), per-layer SVD overrides (`modules_svd_dict`), and per-layer group size overrides (`modules_group_size_dict`). The snippet uses the most common values as the base configuration and emits override dicts for layers that need different dtype, group size, or SVD settings.

## Architecture

Report generation uses a shared data pipeline:

```
results (list[dict])
    │
    ▼
prepare_summary_data(results, args) → summary dict
    │
    ├──► print_rich_summary() / print_plain_summary()  (terminal)
    │
    └──► generate_report()
            │
            ├── build_nmse_chart_base64()  (matplotlib → base64 PNG)
            ├── build_html_report()        (Jinja2 template rendering)
            └── weasyprint.HTML().write_pdf()  (PDF only)
```

`prepare_summary_data()` is the single source of truth. Both terminal output and HTML/PDF reports consume the same structured dict, ensuring consistency.

## Per-Component Auto-Config and Pipeline Quantization

When `--auto-config` is used with a multi-component pipeline, the analysis produces:

1. **Per-component config snippets** — Each component gets its own SDNQConfig recommendation with component-specific base dtype, skip layers, dtype overrides, and SVD overrides.
2. **Pipeline config JSON** — A machine-readable JSON file mapping subfolder names to SDNQConfig kwargs. Written to `<output-stem>_pipeline_config.json` by default (override with `--pipeline-config-output`).
3. **Component column** — The terminal table and HTML report include a "Comp" column when multiple components are analyzed.
4. **Per-component size breakdown** — The stats card shows size estimates per component.

### Pipeline Config JSON Format

```json
{
  "transformer": {
    "weights_dtype": "uint4",
    "group_size": 32,
    "modules_to_not_convert": ["proj_out.weight"],
    "modules_dtype_dict": {"uint8": ["sensitive_layer.weight"]},
    "modules_svd_dict": {32: ["layer_with_svd.weight"]}
  },
  "text_encoder": {
    "weights_dtype": "int8",
    "group_size": 0,
    "use_svd": true,
    "svd_rank": 32
  }
}
```

Each entry contains the kwargs to pass to `SDNQConfig(...)` for that component.

### Pipeline Quantize Script

`scripts/quantize_pipeline.py` consumes the pipeline config JSON to quantize and save a full diffusers pipeline. It processes each component individually to keep memory usage low.

#### Usage

```bash
# Using generated per-component config
python scripts/quantize_pipeline.py \
    --model-id Tongyi-MAI/Z-Image \
    --output-path /path/to/quantized \
    --pipeline-config quant_sensitivity_report_pipeline_config.json

# Uniform config for all components
python scripts/quantize_pipeline.py \
    --model-id Tongyi-MAI/Z-Image \
    --output-path /path/to/quantized \
    --weights-dtype int8

# Quantize only specific components
python scripts/quantize_pipeline.py \
    --model-id Tongyi-MAI/Z-Image \
    --output-path /path/to/quantized \
    --pipeline-config config.json \
    --components transformer text_encoder
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-id` | required | HuggingFace model ID or local path |
| `--output-path` | required | Output directory for the quantized pipeline |
| `--pipeline-config` | `None` | JSON config from analyze script (per-component SDNQConfig params) |
| `--cache-dir` | `~/database/models/huggingface` | HuggingFace cache directory |
| `--device` | `cpu` | Device for quantization computation |
| `--max-shard-size` | `5GB` | Max shard size for saved model files |
| `--components` | `None` | Subset of components to quantize (subfolder names) |
| `--weights-dtype` | `int8` | Quantization dtype (fallback when no `--pipeline-config`) |
| `--group-size` | `0` | Group size (fallback when no `--pipeline-config`) |
| `--use-svd` | off | Enable SVD (fallback when no `--pipeline-config`) |
| `--svd-rank` | `32` | SVD rank (fallback when no `--pipeline-config`) |
| `--use-quantized-matmul` | off | Enable quantized matmul (fallback when no `--pipeline-config`) |

#### What It Does

1. **Discovers components** from `model_index.json`
2. **Copies non-quantizable components** (schedulers, tokenizers, processors) and `model_index.json` to the output directory
3. **Quantizes each model component** individually using `sdnq_post_load_quant()` and saves with `save_sdnq_model()`
4. Each component subfolder gets its own `quantization_config.json`

#### End-to-End Workflow

```bash
# Step 1: Analyze sensitivity
python scripts/analyze_quantization_sensitivity.py \
    --model-id Tongyi-MAI/Z-Image --auto-config

# Step 2: Quantize pipeline
python scripts/quantize_pipeline.py \
    --model-id Tongyi-MAI/Z-Image \
    --output-path /path/to/quantized \
    --pipeline-config quant_sensitivity_report_pipeline_config.json

# Step 3: Load in code
import sdnq
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("/path/to/quantized")
```

## Output-Level Validation

To verify that a quantized pipeline produces acceptable output quality (not just per-layer weight error), use `scripts/benchmark_inference_quality.py`. It compares FP16 baseline and quantized pipeline outputs at two levels: per-step latent divergence and final image quality metrics (PSNR, SSIM, LPIPS).

```bash
# Compare against a pre-quantized pipeline
python scripts/benchmark_inference_quality.py \
    --model-id Tongyi-MAI/Z-Image \
    --quantized-path /path/to/quantized

# Or use the pipeline config for on-the-fly quantization
python scripts/benchmark_inference_quality.py \
    --model-id Tongyi-MAI/Z-Image \
    --pipeline-config quant_sensitivity_report_pipeline_config.json
```

See `python scripts/benchmark_inference_quality.py --help` for full CLI reference.
