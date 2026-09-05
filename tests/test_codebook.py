"""Codebook dtype suite: fit invariants, packing, dequant paths, layer forwards, ladder, config and state dict.

Runs standalone (python tests/test_codebook.py) or under pytest. Imports only sdnq; set PYTHONPATH=src
when the package is not installed.
"""

import sys
import time

import torch

from sdnq.common import dtype_dict, weights_dtype_order, sdnq_keys
from sdnq.quant_utils import fit_codebook, quantize_weight_shared_codebook, quantize_weight
from sdnq.dequantizer import dequantize_weight
from sdnq.packed_int import pack_int, unpack_int
from sdnq.quantizer import sdnq_quantize_layer, sdnq_quantize_layer_weight_dynamic, SDNQConfig, SDNQQuantizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_F, IN_F = 512, 1024
CB_DTYPES = ("cb2", "cb3", "cb4", "cb5", "cb6")

floor_cache: dict[str, float] = {}


def weight_distributions(seed: int = 0) -> dict[str, torch.Tensor]:
    # spread of shapes the fit must hold on: symmetric, skewed, heavy-tailed, bimodal
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(OUT_F, IN_F, generator=g) * 0.04
    return {
        "gaussian": base.clone(),
        "skewed": (base.abs() ** 1.5) * torch.sign(base + 0.3),
        "heavy_tailed": base * (1.0 + base.abs() * 25.0),
        "bimodal": base + torch.sign(base) * 0.05,
    }


def normalized_values(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.abs().amax(dim=-1, keepdim=True).clamp_(min=torch.finfo(torch.float32).tiny)
    return weight.float() / scale


def assignment_mse(values: torch.Tensor, levels: torch.Tensor) -> float:
    midpoints = levels[1:].add(levels[:-1]).mul(0.5)
    picked = levels[torch.bucketize(values, midpoints)]
    return float(torch.mean((values - picked) ** 2))


def fit_floor() -> float:
    # noise floor of the fit itself: scatter_add on CUDA is non-deterministic, so the same input can
    # produce slightly different levels; every fitted-MSE comparison carries this measured floor
    if "fit" not in floor_cache:
        values = normalized_values(weight_distributions(seed=7)["gaussian"]).to(DEVICE)
        mses = [assignment_mse(values, fit_codebook(values, 16)) for _ in range(2)]
        floor_cache["fit"] = abs(mses[0] - mses[1])
    return floor_cache["fit"]


def build_layer(weights_dtype="cb4", use_quantized_matmul=False, use_hadamard=False, use_svd=False, seed=0, group_size=0, quantized_matmul_dtype=None, use_codebook=False):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(IN_F, OUT_F, bias=False, dtype=torch.bfloat16, device=DEVICE)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(OUT_F, IN_F, device=DEVICE) * 0.04)
    reference = lin.weight.detach().clone()
    cfg = SDNQConfig(weights_dtype=weights_dtype, quantized_matmul_dtype=quantized_matmul_dtype, group_size=group_size, hadamard_group_size=256,
                     use_hadamard=use_hadamard, use_svd=use_svd, svd_rank=32, use_quantized_matmul=use_quantized_matmul, use_codebook=use_codebook,
                     dequantize_fp32=False, quantization_device=str(DEVICE), return_device=str(DEVICE))
    layer, _ = sdnq_quantize_layer(lin, cfg, torch_dtype=torch.bfloat16, param_name="test.weight")
    return layer, reference


def dequantized_weight(layer) -> torch.Tensor:
    return layer.sdnq_dequantizer(
        layer.weight, layer.scale, zero_point=layer.zero_point,
        svd_up=layer.svd_up, svd_down=layer.svd_down, codebook=layer.codebook,
        skip_quantized_matmul=layer.sdnq_dequantizer.use_quantized_matmul, skip_compile=True,
    )


# --- fit ---

def test_fit_deterministic_on_cpu():
    values = normalized_values(weight_distributions(seed=1)["gaussian"])
    assert torch.equal(fit_codebook(values, 16), fit_codebook(values, 16)), "cpu fit is not deterministic"


def test_fit_zero_steps_is_affine_grid():
    values = normalized_values(weight_distributions(seed=2)["skewed"])
    levels = fit_codebook(values, 16, steps=0)
    w_min, w_max = values.min(), values.max()
    grid = torch.arange(16, dtype=values.dtype).div(15).mul(w_max - w_min).add(w_min)
    assert torch.equal(levels, grid), "zero-step fit must equal the uniform min/max grid"


def test_fit_beats_affine_grid():
    # the invariant the dtype is built on: init == affine grid + monotone iterations
    for name, weight in weight_distributions(seed=3).items():
        values = normalized_values(weight).to(DEVICE)
        for num_levels in (4, 16, 64):
            mse_init = assignment_mse(values, fit_codebook(values, num_levels, steps=0))
            mse_fit = assignment_mse(values, fit_codebook(values, num_levels))
            assert mse_fit <= mse_init + fit_floor(), f"{name} n={num_levels}: fitted {mse_fit:.3e} > affine {mse_init:.3e} + floor {fit_floor():.1e}"


def test_quantize_beats_uint4_end_to_end():
    # full path incl. the int8 level snap; the snap moves each level by at most 0.5/127 of the level
    # range, far below 4-bit quantization noise, so a 1.05 margin over the pre-snap invariant covers it
    ratios = []
    for name, weight in weight_distributions(seed=4).items():
        w = weight.to(DEVICE).float()
        q, scale, codebook = quantize_weight_shared_codebook(w, -1, "cb4")
        cb_mse = float(torch.mean((w - codebook.to(torch.float32)[q.to(torch.int32)] * scale) ** 2))
        aq, ascale, azp = quantize_weight(w, -1, "uint4")
        affine_mse = float(torch.mean((w - (aq.to(torch.float32) * ascale + azp)) ** 2))
        ratios.append((name, cb_mse / affine_mse))
        assert cb_mse <= affine_mse * 1.05, f"{name}: cb4 mse {cb_mse:.3e} vs uint4 {affine_mse:.3e}"
    print("  cb4/uint4 mse ratios: " + ", ".join(f"{n}={r:.3f}" for n, r in ratios))


def test_zero_row_does_not_poison_codebook():
    w = weight_distributions(seed=5)["gaussian"].to(DEVICE).float()
    w[0].zero_()
    q, scale, codebook = quantize_weight_shared_codebook(w, -1, "cb4")
    assert torch.isfinite(codebook.to(torch.float32)).all(), "codebook has non-finite levels"
    assert torch.isfinite(scale).all(), "scale has non-finite values"
    dq = codebook.to(torch.float32)[q.to(torch.int32)] * scale
    # the clamped scale bounds the row at |level| * tiny * level_absmax / 127 <= tiny, which flushes to zero
    assert float(dq[0].abs().max()) <= torch.finfo(torch.float32).tiny, f"zero row dequant {float(dq[0].abs().max()):.3e} above the clamp bound"
    err = float(torch.mean((w[1:] - dq[1:]) ** 2)) / float(w[1:].square().mean())
    assert err < 0.05, f"other rows degraded: relative mse {err:.3e}"


def test_nan_weight_does_not_poison_shared_codebook():
    # the NaN row itself dequantizes to NaN, as it does under affine quantization; the shared book and every other row stay intact
    w = weight_distributions(seed=5)["gaussian"].to(DEVICE).float()
    w[3, 17] = float("nan")
    q, scale, codebook = quantize_weight_shared_codebook(w, -1, "cb4")
    assert torch.isfinite(codebook.to(torch.float32)).all(), "a NaN weight leaked into the shared codebook"
    dq = codebook.to(torch.float32)[q.to(torch.int32)] * scale
    others = torch.arange(OUT_F, device=DEVICE) != 3
    assert torch.isfinite(dq[others]).all(), "a NaN weight leaked into other rows"
    err = float(torch.mean((w[others] - dq[others]) ** 2)) / float(w[others].square().mean())
    assert err < 0.05, f"other rows degraded: relative mse {err:.3e}"


def test_fit_survives_clusters_past_fp32_count_limit():
    # a float32 count accumulator stops incrementing at 2^24; per-channel normalization of outlier-heavy
    # weights concentrates a 100M-element tensor into a few central clusters that stay past that cliff
    # for every iteration, so the skewed means never heal; 16384x6144 with boosted input channels is
    # the smallest real geometry that crosses it, and the affine invariant must hold there too
    g = torch.Generator(device="cpu").manual_seed(11)
    w = torch.randn(16384, 6144, generator=g) * 0.02
    w[:, [1, 3072, 6142]] *= 8.0
    w = w.to(DEVICE)
    q, scale, codebook = quantize_weight_shared_codebook(w, -1, "cb4")
    cb_mse = float(torch.mean((w - codebook.to(torch.float32)[q.to(torch.int32)] * scale) ** 2))
    del q, scale, codebook
    aq, ascale, azp = quantize_weight(w, -1, "uint4")
    affine_mse = float(torch.mean((w - (aq.to(torch.float32) * ascale + azp)) ** 2))
    assert cb_mse <= affine_mse * 1.05, f"cb4 mse {cb_mse:.3e} vs uint4 {affine_mse:.3e} at 16384x6144"


# --- pack and dequant ---

def test_pack_roundtrip_all_widths():
    g = torch.Generator(device="cpu").manual_seed(6)
    for wdt in CB_DTYPES:
        num_levels = dtype_dict[wdt]["max"] + 1
        for shape in ((32, 64), (16, 8, 24)): # 2-D per-channel and 3-D grouped layouts
            idx = torch.randint(0, num_levels, shape, generator=g, dtype=torch.uint8)
            unpacked = unpack_int(pack_int(idx.clone(), wdt), wdt, torch.Size(shape))
            assert torch.equal(unpacked.to(torch.uint8), idx), f"{wdt} {shape}: pack/unpack mismatch"


def test_dequant_matches_reference():
    w = weight_distributions(seed=8)["gaussian"].to(DEVICE).float()
    q, scale, codebook = quantize_weight_shared_codebook(w, -1, "cb4")
    ref = codebook.to(torch.float32)[q.to(torch.int32)] * scale
    out = dequantize_weight("cb4", pack_int(q.clone(), "cb4"), scale, codebook=codebook, quantized_weight_shape=q.shape, dtype=torch.float32)
    assert torch.equal(out, ref), f"weight-only dequant differs from reference by {float((out - ref).abs().max()):.3e}"


def test_dequant_grouped_matches_reference():
    groups, group_size = 8, IN_F // 8
    w = weight_distributions(seed=9)["heavy_tailed"].to(DEVICE).float()
    q, scale, codebook = quantize_weight_shared_codebook(w.unflatten(-1, (groups, group_size)), -1, "cb4")
    ref = (codebook.to(torch.float32)[q.to(torch.int32)] * scale).view(OUT_F, IN_F)
    out = dequantize_weight("cb4", pack_int(q.clone(), "cb4"), scale, codebook=codebook, quantized_weight_shape=q.shape, result_shape=torch.Size((OUT_F, IN_F)), dtype=torch.float32)
    assert torch.equal(out, ref), "grouped dequant differs from reference"


# --- layer forwards ---

def test_layer_forward_matches_dequant_linear():
    layer, _ = build_layer("cb4")
    x = torch.randn(4, IN_F, device=DEVICE, dtype=torch.bfloat16)
    out = layer(x)
    ref = torch.nn.functional.linear(x, dequantized_weight(layer))
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-3), f"weight-only forward differs from dequantized linear by {float((out - ref).abs().max()):.3e}"


def test_auto_group_rule():
    # zero_point-free storage buys groups half the affine size at equal bytes; 6 bits and above stay row-wise
    assert build_layer("uint4")[0].sdnq_dequantizer.group_size == 64, "affine uint4 auto group moved"
    assert build_layer("cb4")[0].sdnq_dequantizer.group_size == 32, "cb4 auto group must be half of uint4's"
    assert build_layer("cb2")[0].sdnq_dequantizer.group_size == 8, "cb2 auto group must be 8"
    assert build_layer("cb4", use_svd=True)[0].sdnq_dequantizer.group_size == 64, "svd doubles the cb4 auto group"
    assert build_layer("cb6")[0].sdnq_dequantizer.group_size == -1, "cb6 auto must be row-wise"
    assert build_layer("cb4", group_size=-1)[0].sdnq_dequantizer.group_size == -1, "explicit row-wise must be honored"


def test_matmul_bypass_matches_dequant_linear():
    default_layer, _ = build_layer("cb4", use_quantized_matmul=True)
    assert default_layer.sdnq_dequantizer.re_quantize_for_matmul, "grouped cb4 must take the requant matmul path"
    layer, _ = build_layer("cb4", use_quantized_matmul=True, group_size=-1)
    assert layer.sdnq_dequantizer.use_quantized_matmul, "matmul not enabled on the layer"
    assert not layer.sdnq_dequantizer.re_quantize_for_matmul, "row-wise cb4 must take the direct matmul path"
    x = torch.randn(4, IN_F, device=DEVICE, dtype=torch.bfloat16) # < 32 rows: the small-batch bypass
    out = layer(x)
    ref = torch.nn.functional.linear(x, dequantized_weight(layer))
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-3), "small-batch bypass differs from dequantized linear"


def gemm_vs_dequant_relerr(weights_dtype: str, group_size: int = 0) -> float:
    layer, _ = build_layer(weights_dtype, use_quantized_matmul=True, group_size=group_size)
    x = torch.randn(64, IN_F, device=DEVICE, dtype=torch.bfloat16)
    out = layer(x).float()
    ref = torch.nn.functional.linear(x, dequantized_weight(layer)).float()
    return float((out - ref).norm() / ref.norm().clamp(min=1e-12))


def test_gemm_path_error_within_control_band():
    # the GEMM-vs-dequant gap is dominated by dynamic int8 activation quantization, identical machinery
    # for every integer dtype; int8 and uint4 are the controls and 2x their band guards the cb-specific
    # step (unpack + gather), where a wrong nibble order or book gives ~100% relative error
    controls = {d: gemm_vs_dequant_relerr(d) for d in ("int8", "uint4")}
    direct = gemm_vs_dequant_relerr("cb4", group_size=-1)
    grouped = gemm_vs_dequant_relerr("cb4")
    bound = max(controls.values()) * 2.0
    print(f"  gemm relerr: cb4 direct={direct:.4f} grouped={grouped:.4f} controls={ {k: round(v, 4) for k, v in controls.items()} } bound={bound:.4f}")
    assert direct <= bound, f"cb4 direct gemm relerr {direct:.4f} above control band {bound:.4f}"
    assert grouped <= bound, f"cb4 requant gemm relerr {grouped:.4f} above control band {bound:.4f}"


def test_matmul_with_hadamard_and_svd():
    for kwargs in (dict(use_hadamard=True), dict(use_svd=True), dict(use_hadamard=True, use_svd=True)):
        layer, reference = build_layer("cb4", use_quantized_matmul=True, **kwargs)
        x = torch.randn(64, IN_F, device=DEVICE, dtype=torch.bfloat16)
        out = layer(x).float()
        ref = torch.nn.functional.linear(x, reference).float()
        relerr = float((out - ref).norm() / ref.norm())
        assert relerr < 0.2, f"{kwargs}: relative error vs bf16 reference {relerr:.3f}"


def test_non_int8_matmul_dtype_requantizes():
    # the int8 book feeds int8 kernels only; any other matmul dtype must go through the requant path
    layer, reference = build_layer("cb4", use_quantized_matmul=True, group_size=-1, quantized_matmul_dtype="uint8")
    assert layer.sdnq_dequantizer.re_quantize_for_matmul, "cb4 with a uint8 matmul must requantize"
    x = torch.randn(64, IN_F, device=DEVICE, dtype=torch.bfloat16)
    out = layer(x).float()
    ref = torch.nn.functional.linear(x, reference).float()
    relerr = float((out - ref).norm() / ref.norm())
    assert relerr < 0.2, f"uint8 requant path relative error vs bf16 reference {relerr:.3f}"


def test_cb4_beats_int4_against_reference():
    # control comparison on identical weights and inputs: the codebook grid must not lose to the uniform int4 grid
    errs = {}
    for wdt in ("cb4", "int4"):
        layer, reference = build_layer(wdt, seed=11)
        x = torch.randn(64, IN_F, device=DEVICE, dtype=torch.bfloat16)
        out = layer(x).float()
        ref = torch.nn.functional.linear(x, reference).float()
        errs[wdt] = float((out - ref).norm() / ref.norm())
    print(f"  output relerr vs bf16: cb4={errs['cb4']:.4f} int4={errs['int4']:.4f}")
    assert errs["cb4"] <= errs["int4"], f"cb4 {errs['cb4']:.4f} worse than int4 {errs['int4']:.4f}"


def test_conv_forward_matches_dequant():
    torch.manual_seed(12)
    conv = torch.nn.Conv2d(16, 32, 3, padding=1, bias=False, dtype=torch.bfloat16, device=DEVICE)
    cfg = SDNQConfig(weights_dtype="cb4", quant_conv=True, dequantize_fp32=False, quantization_device=str(DEVICE), return_device=str(DEVICE))
    layer, _ = sdnq_quantize_layer(conv, cfg, torch_dtype=torch.bfloat16, param_name="test.weight")
    assert layer.codebook is not None, "conv layer has no codebook"
    x = torch.randn(1, 16, 8, 8, device=DEVICE, dtype=torch.bfloat16)
    out = layer(x)
    dq = layer.sdnq_dequantizer(layer.weight, layer.scale, zero_point=layer.zero_point, svd_up=layer.svd_up, svd_down=layer.svd_down, codebook=layer.codebook, skip_compile=True)
    ref = torch.nn.functional.conv2d(x, dq, padding=1)
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-3), "conv forward differs from dequantized conv"


def test_embedding_forward_matches_dequant():
    torch.manual_seed(13)
    emb = torch.nn.Embedding(64, 128, dtype=torch.bfloat16, device=DEVICE)
    cfg = SDNQConfig(weights_dtype="cb4", quant_embedding=True, dequantize_fp32=False, quantization_device=str(DEVICE), return_device=str(DEVICE))
    layer, _ = sdnq_quantize_layer(emb, cfg, torch_dtype=torch.bfloat16, param_name="test.weight")
    assert layer.codebook is not None, "embedding layer has no codebook"
    ids = torch.tensor([[0, 5, 63], [7, 7, 1]], device=DEVICE)
    out = layer(ids)
    dq = layer.sdnq_dequantizer(layer.weight, layer.scale, zero_point=layer.zero_point, svd_up=layer.svd_up, svd_down=layer.svd_down, codebook=layer.codebook, skip_compile=True)
    ref = torch.nn.functional.embedding(ids, dq)
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-3), "embedding forward differs from dequantized lookup"


def test_layer_dequantize_unwraps_codebook():
    layer, _ = build_layer("cb4")
    expected = dequantized_weight(layer)
    plain = layer.dequantize()
    assert plain.__class__ is torch.nn.Linear, f"dequantize must restore the original class, got {plain.__class__}"
    assert not hasattr(plain, "codebook") and not hasattr(plain, "sdnq_dequantizer"), "quant state must be gone after dequantize"
    assert torch.equal(plain.weight.detach(), expected), "dequantized weight differs from the dequantizer output"


def test_use_codebook_flag_path_unchanged():
    # the per-row levels-in-scale mode keeps working next to the cb dtypes
    layer, reference = build_layer("uint4", use_codebook=True)
    assert layer.codebook is None and layer.sdnq_dequantizer.use_codebook, "use_codebook layer must not grow a shared codebook"
    assert layer.scale.shape[-1] == 16, f"use_codebook keeps 16 levels in the scale slot, got {tuple(layer.scale.shape)}"
    x = torch.randn(64, IN_F, device=DEVICE, dtype=torch.bfloat16)
    relerr = float((layer(x).float() - torch.nn.functional.linear(x, reference).float()).norm() / torch.nn.functional.linear(x, reference).float().norm())
    assert relerr < 0.2, f"use_codebook forward relative error {relerr:.3f}"


# --- ladder, config, state ---

def test_ladder_contains_cb_rungs_in_order():
    for wdt in CB_DTYPES:
        assert weights_dtype_order.count(wdt) == 1, f"{wdt} not exactly once in weights_dtype_order"
        nbits = dtype_dict[wdt]["num_bits"]
        assert weights_dtype_order.index(wdt) < weights_dtype_order.index(f"int{nbits + 1}"), f"{wdt} must precede int{nbits + 1}"
        assert weights_dtype_order.index(wdt) > weights_dtype_order.index(f"uint{nbits}"), f"{wdt} must follow uint{nbits}"


def test_dynamic_from_int4_never_lands_on_cb():
    torch.manual_seed(14)
    weight = torch.randn(OUT_F, IN_F, device=DEVICE) * 0.04
    result = sdnq_quantize_layer_weight_dynamic(weight, layer_class_name="Linear", weights_dtype="int4", dynamic_loss_threshold=1e-9, torch_dtype=torch.bfloat16)
    if result is not None:
        deq, _ = result
        assert not dtype_dict[deq.weights_dtype].get("is_codebook", False), f"int4 dynamic landed on {deq.weights_dtype}"


def test_dynamic_from_cb4_walks_cb():
    torch.manual_seed(15)
    weight = torch.randn(OUT_F, IN_F, device=DEVICE) * 0.04
    result = sdnq_quantize_layer_weight_dynamic(weight, layer_class_name="Linear", weights_dtype="cb4", dynamic_loss_threshold=1.0, torch_dtype=torch.bfloat16)
    assert result is not None, "cb4 dynamic returned nothing"
    deq, weight_data = result
    assert deq.weights_dtype == "cb4", f"loose threshold must accept the first rung, got {deq.weights_dtype}"
    assert weight_data.get("codebook") is not None, "dynamic result carries no codebook"


def test_config_rejects_flag_and_training_with_cb():
    for kwargs, what in ((dict(use_codebook=True), "use_codebook"), (dict(is_training=True), "training")):
        try:
            SDNQConfig(weights_dtype="cb4", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"SDNQConfig must reject {what} with a cb dtype")
    SDNQConfig(weights_dtype="uint4", use_codebook=True) # the flag itself stays valid on affine dtypes


def test_state_dict_carries_codebook():
    layer, _ = build_layer("cb4")
    sd = layer.state_dict()
    assert "codebook" in sd, f"codebook missing from state_dict keys {sorted(sd.keys())}"
    assert sd["codebook"].dtype == torch.int8, f"fresh codebook stored as {sd['codebook'].dtype}"
    assert "codebook" in sdnq_keys, "codebook missing from sdnq_keys"


def test_meta_quantization_yields_codebook_placeholder():
    # pre-quantized loads quantize a meta model first so every stored key has a placeholder to land on
    lin = torch.nn.Linear(IN_F, OUT_F, bias=False, dtype=torch.bfloat16, device="meta")
    cfg = SDNQConfig(weights_dtype="cb4", dequantize_fp32=True, quantization_device="meta", return_device="meta")
    layer, _ = sdnq_quantize_layer(lin, cfg, torch_dtype=torch.bfloat16, param_name="test.weight")
    assert layer.codebook is not None and layer.codebook.device.type == "meta", "meta quantization produced no codebook placeholder"
    assert tuple(layer.codebook.shape) == (16,) and layer.codebook.dtype == torch.int8, f"placeholder is {layer.codebook.dtype}{tuple(layer.codebook.shape)}"


def test_prequantized_load_keeps_codebook_dtype():
    # the HF pre-quantized path fp32-upcasts float side tensors under dequantize_fp32; the codebook
    # keeps its stored dtype in both the fresh (int8) and ingested (fp32) forms
    layer, _ = build_layer("cb4")
    model = torch.nn.Module()
    model.lin = layer
    quantizer = SDNQQuantizer(SDNQConfig(weights_dtype="cb4", dequantize_fp32=True), pre_quantized=True)
    quantizer.torch_dtype = torch.bfloat16
    assert quantizer.check_if_quantized_param(model, layer.codebook.data, "lin.codebook"), "codebook not claimed as a quantized param"
    quantizer.create_quantized_param(model, torch.arange(16, dtype=torch.int8), "lin.codebook", DEVICE)
    assert model.lin.codebook.dtype == torch.int8, f"int8 codebook became {model.lin.codebook.dtype}"
    quantizer.create_quantized_param(model, torch.linspace(-1.0, 1.0, 16, dtype=torch.float32), "lin.codebook", DEVICE)
    assert model.lin.codebook.dtype == torch.float32, f"fp32 codebook became {model.lin.codebook.dtype}"
    quantizer.create_quantized_param(model, layer.scale.data.to(torch.bfloat16), "lin.scale", DEVICE)
    assert model.lin.scale.dtype == torch.float32, "scale must still upcast under dequantize_fp32"


SUITES = {
    "fit": [test_fit_deterministic_on_cpu, test_fit_zero_steps_is_affine_grid, test_fit_beats_affine_grid, test_quantize_beats_uint4_end_to_end,
            test_zero_row_does_not_poison_codebook, test_nan_weight_does_not_poison_shared_codebook, test_fit_survives_clusters_past_fp32_count_limit],
    "pack-dequant": [test_pack_roundtrip_all_widths, test_dequant_matches_reference, test_dequant_grouped_matches_reference],
    "forward": [test_layer_forward_matches_dequant_linear, test_auto_group_rule, test_matmul_bypass_matches_dequant_linear, test_gemm_path_error_within_control_band,
                test_matmul_with_hadamard_and_svd, test_non_int8_matmul_dtype_requantizes, test_cb4_beats_int4_against_reference, test_conv_forward_matches_dequant,
                test_embedding_forward_matches_dequant, test_layer_dequantize_unwraps_codebook, test_use_codebook_flag_path_unchanged],
    "ladder-config-state": [test_ladder_contains_cb_rungs_in_order, test_dynamic_from_int4_never_lands_on_cb, test_dynamic_from_cb4_walks_cb,
                            test_config_rejects_flag_and_training_with_cb, test_state_dict_carries_codebook, test_meta_quantization_yields_codebook_placeholder,
                            test_prequantized_load_keeps_codebook_dtype],
}


def run_all() -> bool:
    started = time.time()
    print(f"device={DEVICE} fit noise floor={fit_floor():.3e}")
    passed = failed = 0
    with torch.inference_mode():
        for cat, tests in SUITES.items():
            print(f"=== {cat} ===")
            for fn in tests:
                try:
                    fn()
                    print(f"  PASS: {fn.__name__}")
                    passed += 1
                except Exception as e: # pylint: disable=broad-except
                    print(f"  FAIL: {fn.__name__} ({type(e).__name__}: {e})")
                    failed += 1
    print(f"Total: {passed} passed, {failed} failed in {time.time() - started:.1f}s")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
