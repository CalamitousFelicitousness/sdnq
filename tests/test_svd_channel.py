"""Rowwise-int8 svd factor channel: merged behind the full-precision pair at dequant and matmul time.

Runs standalone (python tests/test_svd_channel.py) or under pytest; set PYTHONPATH=src when the package is not installed.
"""

import sys
import time

import torch

from sdnq.quantizer import sdnq_quantize_layer, SDNQConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_F, IN_F, RANK = 512, 1024, 32
CHANNEL_KEYS = ("svd_up_q", "svd_up_scale", "svd_down_q", "svd_down_scale")


def quantize_rowwise(t):
    t32 = t.detach().to(torch.float32)
    scale = t32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    return (t32 / scale).round().clamp(-127, 127).to(torch.int8), scale


def dequantize_rowwise(q, scale):
    return q.to(torch.float32) * scale


def build_layer(weights_dtype="uint4", use_quantized_matmul=False, group_size=0, use_svd=False, use_codebook=False, seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(IN_F, OUT_F, bias=True, dtype=torch.bfloat16, device=DEVICE)
    cfg = SDNQConfig(weights_dtype=weights_dtype, group_size=group_size, use_svd=use_svd, svd_rank=32, use_codebook=use_codebook, use_quantized_matmul=use_quantized_matmul,
                     dequantize_fp32=False, quantization_device=str(DEVICE), return_device=str(DEVICE))
    layer, _ = sdnq_quantize_layer(lin, cfg, torch_dtype=torch.bfloat16, param_name="test.weight")
    return layer


def hosted_pair(seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    up = (torch.randn(OUT_F, RANK, generator=g) * 0.02).to(DEVICE, torch.bfloat16)
    down = (torch.randn(RANK, IN_F, generator=g) * 0.02).to(DEVICE, torch.bfloat16)
    return up, down


def attach_channel(layer, up, down):
    # the layer stores the channel in its own layout: transposed with oriented scales under quantized matmul;
    # returns what the merge reconstructs, in that layout and in fp32; the merge casts to the full-precision pair dtype when one exists, else to the output dtype
    up_q, up_s = quantize_rowwise(up)
    down_q, down_s = quantize_rowwise(down)
    if layer.sdnq_dequantizer.use_quantized_matmul:
        up_q, up_s, down_q, down_s = up_q.t().contiguous(), up_s.t().contiguous(), down_q.t().contiguous(), down_s.t().contiguous()
    layer.svd_up_q = torch.nn.Parameter(up_q, requires_grad=False)
    layer.svd_up_scale = torch.nn.Parameter(up_s, requires_grad=False)
    layer.svd_down_q = torch.nn.Parameter(down_q, requires_grad=False)
    layer.svd_down_scale = torch.nn.Parameter(down_s, requires_grad=False)
    return dequantize_rowwise(up_q, up_s), dequantize_rowwise(down_q, down_s)


def clear_channel(layer):
    for key in CHANNEL_KEYS:
        setattr(layer, key, None)


def manual_concat(layer, up_m, down_m, dtype=torch.float32):
    if layer.svd_up is None:
        return up_m.to(dtype), down_m.to(dtype)
    up_m, down_m = up_m.to(layer.svd_up.dtype), down_m.to(layer.svd_down.dtype)
    if layer.sdnq_dequantizer.use_quantized_matmul:
        return torch.cat([layer.svd_up, up_m], dim=0), torch.cat([layer.svd_down, down_m], dim=1)
    return torch.cat([layer.svd_up, up_m], dim=1), torch.cat([layer.svd_down, down_m], dim=0)


def dequant(layer, svd_up, svd_down, with_channel, dtype=torch.float32):
    deq = layer.sdnq_dequantizer
    kwargs = {key: getattr(layer, key) for key in CHANNEL_KEYS} if with_channel else {}
    return deq(layer.weight, layer.scale, zero_point=layer.zero_point, svd_up=svd_up, svd_down=svd_down, codebook=layer.codebook,
               skip_quantized_matmul=deq.use_quantized_matmul, dtype=dtype, skip_compile=True, **kwargs)


def test_merge_matches_manual_concat_on_every_dequant_path():
    cases = [dict(weights_dtype="int8"), dict(weights_dtype="uint4"), dict(weights_dtype="uint4", use_svd=True), dict(weights_dtype="cb4"), dict(weights_dtype="uint4", use_codebook=True)]
    for case in cases:
        layer = build_layer(**case)
        up_m, down_m = attach_channel(layer, *hosted_pair())
        merged = dequant(layer, layer.svd_up, layer.svd_down, with_channel=True)
        up_c, down_c = manual_concat(layer, up_m, down_m)
        manual = dequant(layer, up_c, down_c, with_channel=False)
        assert torch.equal(merged, manual), f"{case}: merged channel differs from the manual concat by {float((merged - manual).abs().max()):.3e}"
        base = dequant(layer, layer.svd_up, layer.svd_down, with_channel=False)
        assert not torch.equal(merged, base), f"{case}: the channel must act on the dequantized weight"


def test_matmul_layout_forward_matches_concat():
    # matmul layout stores the channel transposed; the direct int8 GEMM (int8 and cb4 at group -1) and the requant path
    # (grouped uint4) all take it through the input-side factor term
    for case in (dict(weights_dtype="int8", group_size=-1), dict(weights_dtype="cb4", group_size=-1), dict(weights_dtype="uint4")):
        layer = build_layer(use_quantized_matmul=True, **case)
        x = torch.randn(64, IN_F, device=DEVICE, dtype=torch.bfloat16)
        up_m, down_m = attach_channel(layer, *hosted_pair())
        out_channel = layer(x).float()
        clear_channel(layer)
        layer.svd_up = torch.nn.Parameter(up_m.to(torch.bfloat16), requires_grad=False) # the matmul paths merge in the input dtype
        layer.svd_down = torch.nn.Parameter(down_m.to(torch.bfloat16), requires_grad=False)
        out_concat = layer(x).float()
        assert torch.allclose(out_channel, out_concat, rtol=1e-3, atol=1e-3), f"{case}: channel forward differs from the concat forward by {float((out_channel - out_concat).abs().max()):.3e}"
        layer.svd_up = layer.svd_down = None
        assert not torch.allclose(layer(x).float(), out_channel, rtol=1e-3, atol=1e-3), f"{case}: the channel must change the matmul output"


def test_small_batch_bypass_uses_channel():
    layer = build_layer("uint4", use_quantized_matmul=True)
    x = torch.randn(4, IN_F, device=DEVICE, dtype=torch.bfloat16) # below 32 rows the matmul forward dequantizes instead
    up_m, down_m = attach_channel(layer, *hosted_pair())
    out_channel = layer(x).float()
    clear_channel(layer)
    layer.svd_up = torch.nn.Parameter(up_m.to(torch.bfloat16), requires_grad=False)
    layer.svd_down = torch.nn.Parameter(down_m.to(torch.bfloat16), requires_grad=False)
    out_concat = layer(x).float()
    assert torch.allclose(out_channel, out_concat, rtol=1e-2, atol=1e-3), f"bypass forward differs from the concat forward by {float((out_channel - out_concat).abs().max()):.3e}"


def test_layer_dequantize_consumes_channel():
    layer = build_layer("uint4")
    attach_channel(layer, *hosted_pair())
    expected = dequant(layer, layer.svd_up, layer.svd_down, with_channel=True, dtype=torch.bfloat16)
    plain = layer.dequantize()
    assert plain.__class__ is torch.nn.Linear, f"dequantize must restore the original class, got {plain.__class__}"
    assert not any(hasattr(plain, key) for key in CHANNEL_KEYS), "the channel must be gone after dequantize"
    assert torch.allclose(plain.weight.detach(), expected, rtol=1e-2, atol=1e-3), "dequantized weight must include the channel"


def test_absent_channel_is_identity():
    layer = build_layer("uint4", use_svd=True)
    a = dequant(layer, layer.svd_up, layer.svd_down, with_channel=False)
    b = dequant(layer, layer.svd_up, layer.svd_down, with_channel=True) # every channel attribute is None on a fresh layer
    assert torch.equal(a, b), "an empty channel must not change the dequantized weight"


SUITES = {
    "channel": [test_merge_matches_manual_concat_on_every_dequant_path, test_matmul_layout_forward_matches_concat, test_small_batch_bypass_uses_channel,
                test_layer_dequantize_consumes_channel, test_absent_channel_is_identity],
}


def run_all() -> bool:
    started = time.time()
    print(f"device={DEVICE}")
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
