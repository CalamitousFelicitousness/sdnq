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

### INT8 Pre-Quant Sensitivity Checks

These two opt-in checks supplement the weight reconstruction NMSE with axis-specific INT8 sensitivity measurements. See the dedicated [INT8 Pre-Quant Sensitivity Checks](#int8-pre-quant-sensitivity-checks) section below for the full behavior; this table is the CLI reference.

| Argument | Default | Description |
|----------|---------|-------------|
| `--check-weights` | `none` | When set to `dynamic`, runs `sdnq_quantize_layer_weight_dynamic` per layer and records the dtype it picked. Layers where INT8 weight quantization fails the threshold and the function falls through to a finer dtype are flagged. Reproduces the signal SDNQ's dynamic quantization uses (Disty-style hand-curated skip lists) |
| `--weight-threshold` | `1e-2` | Std-normalized weight MSE threshold for `--check-weights`. Mirrors SDNQ's `dynamic_loss_threshold` default. For tightly-quantizable models like Anima where the default flags nothing, drop to `1e-4` to surface relative outliers |
| `--check-matmul` | `none` | When set to `int8`, runs an extra per-layer pass that compares the FP32 reference output to the actual SDNQ INT8 GEMM output, using activations captured from a short calibration pipeline run. Required: also pass `--prompt-set`. Detects layers where `int8_matmul` corrupts output due to activation outliers, a failure mode invisible to weight-only analysis |
| `--prompt-set` | required with `--check-matmul` | `booru` or `natural`. Selects the calibration prompt file at `prompts/<set>.txt`. No default because the wrong distribution produces misleading flags |
| `--matmul-steps` | `2` | Inference steps for the calibration pipeline run |
| `--matmul-max-tokens` | `256` | Cap on captured activation token dimension to bound memory |

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
| `--pipeline-config-output` | auto | JSON output path for per-component pipeline config. Defaults to `<output-stem>_pipeline_config.json`. Written when any of `--auto-config`, `--check-weights`, or `--check-matmul` produce findings. When multiple sources contribute, skip-config (dyn + matmul findings) takes precedence over auto-config grid-search recommendations on conflict |

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

## INT8 Pre-Quant Sensitivity Checks

Auto-config measures **weight reconstruction error**: how far `dequantize(quantize(weight))` drifts from the original. That signal is necessary but not sufficient: a layer can have low weight reconstruction error and still produce visibly corrupted output at inference time, either because (a) the chosen INT8 weight dtype is the wrong fit for that layer's distribution, or (b) the INT8 GEMM kernel itself accumulates error past acceptable bounds on the actual activations the layer sees. These two failure modes are orthogonal (a single layer can fail on one axis, the other, or both) and require different SDNQConfig knobs to fix.

The two checks below each target one axis. They run independently and can be combined.

### `--check-weights dynamic`: Weight-Side Sensitivity

Reuses `sdnq_quantize_layer_weight_dynamic` from `src/sdnq/quantizer.py:432-479` per layer to detect layers where INT8 weight quantization fails. The dynamic function walks `weights_dtype_order` from coarsest to finest, picks the first dtype whose **std-normalized weight MSE** falls under `dynamic_loss_threshold`, and returns the chosen dtype. Layers where INT8 was insufficient and the walk fell through to a finer dtype get flagged with the chosen alternative recorded.

This reproduces the signal SDNQ's existing dynamic quantization uses internally to assemble per-architecture skip lists. Hand-curated lists like the one Disty maintains for Anima are the result of running this same procedure and pruning to a manageable subset.

The check is component-scoped to transformer and text-encoder subfolders (where Linear layers dominate). It does not require a pipeline calibration run because it operates purely on the weight tensors.

#### Usage

```bash
python scripts/analyze_quantization_sensitivity.py \
    --model-id CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers \
    --components transformer:CosmosTransformer3DModel text_encoder:Qwen3Model \
    --dtypes int8 \
    --no-svd \
    --check-weights dynamic \
    --weight-threshold 1e-4 \
    --device cuda
```

#### Output

Adds these CSV columns (populated for INT8 rows in target components, blank otherwise):

| Column | Type | Description |
|----|----|----|
| `dyn_chosen_dtype` | str | The dtype dynamic quant picked (e.g. `int8`, `uint8`, `int9`, `float8_e4m3fn`). Empty when no dtype passed |
| `dyn_int8_safe` | bool | True when chosen dtype was INT8 (layer is safe). False when fell through |
| `dyn_int8_loss` | float | Std-normalized weight MSE of INT8 specifically, as a diagnostic value |
| `dyn_status` | str | `ok` (chose a dtype) or `no_dtype_passed` (full-skip required) |

The HTML report adds a **Dyn Choice** column showing the chosen dtype: green for INT8 (safe), amber for any finer dtype, red for `no_dtype_passed`.

#### Threshold tuning

The default `--weight-threshold 1e-2` matches SDNQ's `dynamic_loss_threshold` default. For models with broadly well-behaved weights (Anima, Z-Image), every layer's INT8 loss falls under this threshold and nothing gets flagged. To surface **relative** outliers (the layers most likely to benefit from a finer dtype), drop the threshold:

| Threshold | Anima Preview-3 weight-flagged | Notes |
|----|----|----|
| `1e-2` (default) | 0 / 650 layers | Generous; appropriate when you want only catastrophic weight-dtype mismatches |
| `1e-4` | 63 / 650 layers | Captures 100% of Disty's hand-curated 33-layer Anima skip list |
| `1e-8` | 650 / 650 layers | Useless; flags everything |

Inspect the `dyn_int8_loss` distribution in the CSV to pick a threshold appropriate for your model.

### `--check-matmul int8`: Matmul-Side Sensitivity

Measures per-layer INT8 GEMM output divergence against the FP32 reference using **real captured activations**. Loads the full pipeline, hooks `register_forward_pre_hook` on every Linear in the target subfolders, runs a short calibration generation with the prompt from `--prompt-set`, then per-layer compares `F.linear(x, W_fp32)` against `int8_matmul(x, W_int8, scale, ...)` from `src/sdnq/layers/linear/linear_int8.py`. The GEMM kernel is the production one; we measure exactly what the layer would output at inference time.

The check is required because of dynamic per-token activation quantization: `int8_matmul` symmetric-quantizes each input row to INT8 internally (`quantize_int_mm_input` at `src/sdnq/layers/linear/linear_int8.py:11`). When activations have outliers (common in cross-attention K/V/O projections that receive text-encoder hidden states), those outliers dominate the per-row scale and crush the rest of the row into noise. The result is large visible artifacts in generated images, and weight-only analysis is structurally blind to it.

#### Calibration prompts

Calibration prompts live in `prompts/booru.txt` and `prompts/natural.txt` at the repo root. Each file is a list of newline-separated prompts; `#` comments and blank lines are skipped. The matmul check uses line 0 of the chosen file. Future multi-sample tests can consume all entries from the same file.

`--prompt-set` is **required** when `--check-matmul` is set, since the wrong distribution produces misleading flags:

- **`booru`** for tag-trained models (Anima, Pony, Illustrious, NoobAI, ...). The text encoder activation distribution is conditioned on comma-separated tags with parenthesised emphasis and weight modifiers.
- **`natural`** for caption-trained models (Z-Image, SD3.5, Flux, Cosmos2 base, Stable Cascade, ...). The text encoder activation distribution is conditioned on full English sentences.

Both prompt files are tuned to be ≥50 BPE tokens so text-encoder Linear layers clear the 32-token gate that `int8_matmul` short-circuits below (see eligibility section).

#### Usage

```bash
python scripts/analyze_quantization_sensitivity.py \
    --model-id CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers \
    --components transformer:CosmosTransformer3DModel text_encoder:Qwen3Model \
    --dtypes int8 \
    --no-svd \
    --check-matmul int8 \
    --prompt-set booru \
    --device cuda
```

#### Output

Adds these CSV columns (populated for INT8 rows in target components, blank otherwise):

| Column | Type | Description |
|----|----|----|
| `mm_nmse` | float | NMSE between FP32 reference output and INT8 matmul output |
| `mm_sqnr_db` | float | Signal-to-quantization-noise ratio in dB |
| `mm_cosine_sim` | float | Cosine similarity between reference and INT8 outputs |
| `mm_max_abs_err` | float | Max element-wise absolute error |
| `mm_relative_l2` | float | Relative L2 error |
| `mm_eligible` | bool | False if the layer was excluded (see eligibility) |
| `mm_status` | str | `ok` or one of the eligibility codes below |

The HTML report adds an **INT8 MM NMSE** column color-coded the same way as the weight-NMSE columns.

#### Eligibility gates

Not every Linear layer can run through INT8 matmul at inference. The analyzer mirrors the production gates exactly to avoid measuring under a fallback path that would never run in practice. Excluded layers report a status code rather than a measurement:

| `mm_status` | Meaning |
|----|----|
| `ok` | Measurement ran |
| `dim_gate` | Weight dimensions don't both pass `≥ 32` and `divisible by 16` (gate at `src/sdnq/quantizer.py:305-307`). The layer never enters `int8_matmul` even at inference |
| `too_few_tokens` | Captured input has `numel/last_dim < 32`, below the threshold at `src/sdnq/layers/linear/linear_int8.py:50`. Matches the production short-circuit that bypasses `int8_matmul` and falls back to `dequantize → F.linear` for tiny inputs. Conditioning Linears (time embeddings, adaLN modulation, per-head norm projections) commonly hit this |
| `unsupported_class:<name>` | Defensive; should not fire for diffusers/transformers models |
| `no_activation` | Layer wasn't called during calibration (conditional/MoE branch). Worth investigating if it appears |
| `ref_failed` / `quantize_failed` / `mm_failed` | An exception in the FP32 reference, in `sdnq_quantize_layer_weight`, or in `int8_matmul` itself. Rare; indicates a bug or incompatible weight |

The terminal output prints the `Matmul ineligibility breakdown` counter at the end of the suggestion section so you can sanity-check totals.

### Combined Output: Structured Pre-Quant Config

When both checks are enabled (or even one), the suggestion printer emits **three orthogonal SDNQConfig knobs** rather than a single `modules_to_not_convert` list. Each maps to a distinct failure mode:

| Knob | Failure mode | Cost trade-off |
|----|----|----|
| `modules_to_not_convert` | Dynamic quant returned `no_dtype_passed` (no dtype in the walk passed). Layer must stay in fp16 | Highest memory cost (full fp16). Only used when dynamic quant indicates nothing else works |
| `modules_dtype_dict[<chosen>]` | Dynamic quant rejected INT8 but found a working finer dtype. Layer keeps quantization at the right precision | Memory savings preserved at the layer's actual safe precision (e.g. UINT8 same size as INT8) |
| `modules_quant_config[<layer>]: {"use_quantized_matmul": False}` | Matmul-side flagged. Layer keeps INT8 weight; only the per-layer GEMM dispatch is disabled | INT8 weight memory savings preserved; per-layer fp16 dequant + `F.linear` fallback at inference |

A single layer can land in `modules_dtype_dict` and `modules_quant_config` simultaneously, since they're orthogonal axes (weight precision vs. kernel choice). The terminal output prints each section separately with paste-ready Python:

```text
# transformer:CosmosTransformer3DModel
#   weight-side flagged: 56  matmul-side flagged: 84
#   weight threshold: 0.0001    (int8_loss > threshold => layer rejected)
#   matmul threshold: 0.1    (mm_nmse > threshold => INT8 GEMM corrupts output)

# Layers requiring a finer weight dtype than int8 (56 layers across 4 dtype(s)).
modules_dtype_dict = {
    "float9_e3m5fn": [
        "time_embed.t_embedder.linear_2.weight",
    ],
    "int10": [...],
    "int9": [...],
    "uint8": [...],
}

# Layers where INT8 GEMM corrupts output (84 layers): keep INT8 weight, disable per-layer matmul.
modules_quant_config = {
    "transformer_blocks.0.attn2.to_k.weight": {"use_quantized_matmul": False},
    ...
}
```

The same data is also written as JSON to `<output-stem>_pipeline_config.json` (see [Pipeline Config JSON Format](#pipeline-config-json-format) for the schema). The JSON is directly consumable as `SDNQConfig(**entry)` for each subfolder; no glue code needed.

### Why TE skips matter despite running once

The text encoder runs once per generation, but its output is the conditioning signal feeding **every cross-attention layer in the transformer at every step**. With 30 inference steps × 28 transformer blocks, text-encoder errors are read 840× per generation. Errors that originate in the text encoder propagate through the entire denoising trajectory and don't get washed out by subsequent norm layers the way transformer-internal errors do. This makes TE weight-side fixes high-leverage even when the absolute error magnitudes are smaller than transformer flags.

The cost side is also asymmetric: skipping a TE layer from INT8 has near-zero latency cost (TE runs once and the absolute time is small). Skipping a transformer layer from INT8 has per-step cost on every Linear-forward. So TE flags should be acted on; transformer flags need to weigh quality against per-generation latency.

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

When `--auto-config`, `--check-weights`, or `--check-matmul` runs on a multi-component pipeline, the analysis produces:

1. **Per-component config snippets**: each component gets its own SDNQConfig recommendation with component-specific base dtype, skip layers, dtype overrides, and SVD overrides.
2. **Pipeline config JSON**: a machine-readable JSON file mapping subfolder names to SDNQConfig kwargs. Written to `<output-stem>_pipeline_config.json` by default (override with `--pipeline-config-output`). Triggered by any of `--auto-config`, `--check-weights`, or `--check-matmul` producing findings.
3. **Component column**: the terminal table and HTML report include a "Comp" column when multiple components are analyzed.
4. **Per-component size breakdown**: the stats card shows size estimates per component (auto-config only).

### Pipeline Config JSON Format

The JSON is a dict mapping subfolder name to `SDNQConfig` kwargs. Each subfolder entry can include any combination of fields depending on which sources contributed (auto-config grid search, `--check-weights`, `--check-matmul`):

```json
{
  "transformer": {
    "weights_dtype": "int8",
    "use_quantized_matmul": true,
    "group_size": 32,
    "modules_to_not_convert": ["layer_with_no_passing_dtype.weight"],
    "modules_dtype_dict": {
      "uint8": ["sensitive_layer.weight"],
      "int9": ["finer_precision_layer.weight"],
      "float9_e3m5fn": ["time_embedder_linear.weight"]
    },
    "modules_svd_dict": {"32": ["layer_with_svd.weight"]},
    "modules_quant_config": {
      "transformer_blocks.0.attn2.to_k.weight": {"use_quantized_matmul": false}
    }
  },
  "text_encoder": {
    "weights_dtype": "int8",
    "use_quantized_matmul": true,
    "modules_dtype_dict": {
      "uint8": ["layers.16.mlp.down_proj.weight"]
    }
  }
}
```

Field provenance:

| Field | Source |
|----|----|
| `weights_dtype`, `group_size`, `use_svd`, `svd_rank` | Auto-config grid search (or defaults when only `--check-weights`/`--check-matmul` are active) |
| `modules_to_not_convert` | Auto-config (grid-search SKIP) + `--check-weights` (`no_dtype_passed`). Union, deduped |
| `modules_dtype_dict` | Auto-config (per-dtype overrides from grid search) + `--check-weights` (per-layer chosen dtype). Union per dtype key |
| `modules_group_size_dict`, `modules_svd_dict` | Auto-config only |
| `modules_quant_config` | `--check-matmul` only |

When sources conflict on a field, skip-config (`--check-weights`/`--check-matmul`) takes precedence over auto-config because per-layer dynamic quantization and per-layer matmul measurement are more rigorous than the threshold-based grid-search heuristic.

Each entry is directly consumable as `SDNQConfig(**entry)` with no transformations needed.

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
