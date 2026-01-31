#!/bin/bash
set -e
source /home/ohiom/sdnq/venv/bin/activate

COMMON="--model-id Tongyi-MAI/Z-Image --prompts ablation_prompts_short.txt --seeds 42 --cpu-offload --no-skip-keys --report-format none"
FP16="--fp16-from ref2_results"

# Phase 4: Runtime & Dtype Ablation tests (all reuse FP16 baseline)

echo "======= TEST 26/33: int8 + quantized matmul ======="
python scripts/benchmark_inference_quality.py $COMMON $FP16 \
    --pipeline-config ablation_int8_quantized_matmul.json \
    --output ablation_int8_quantized_matmul_report.csv --output-dir ablation_int8_quantized_matmul_results

echo "======= TEST 27/33: int8 + dequantize FP32 ======="
python scripts/benchmark_inference_quality.py $COMMON $FP16 \
    --pipeline-config ablation_int8_dequant_fp32.json \
    --output ablation_int8_dequant_fp32_report.csv --output-dir ablation_int8_dequant_fp32_results

echo "======= TEST 28/33: int8 + stochastic rounding ======="
python scripts/benchmark_inference_quality.py $COMMON $FP16 \
    --pipeline-config ablation_int8_stochastic_rounding.json \
    --output ablation_int8_stochastic_rounding_report.csv --output-dir ablation_int8_stochastic_rounding_results

echo "======= TEST 29/33: uint4 MLP+SVD+stoch+FP32 (fixed) ======="
python scripts/benchmark_inference_quality.py $COMMON $FP16 \
    --pipeline-config ablation_uint4_stochastic_rounding.json \
    --output ablation_uint4_stochastic_rounding_report.csv --output-dir ablation_uint4_stochastic_rounding_results

# FP8 tests require Hopper (SM 9.0+) — skip on Ampere/Ada
echo "======= TEST 30/33: float8_e4m3fn (SKIPPED — requires Hopper GPU) ======="
# python scripts/benchmark_inference_quality.py $COMMON $FP16 \
#     --pipeline-config ablation_fp8_e4m3fn.json \
#     --output ablation_fp8_e4m3fn_report.csv --output-dir ablation_fp8_e4m3fn_results

echo "======= TEST 31/33: float8_e4m3fn + quantized matmul (SKIPPED — requires Hopper GPU) ======="
# python scripts/benchmark_inference_quality.py $COMMON $FP16 \
#     --pipeline-config ablation_fp8_e4m3fn_quantized_matmul.json \
#     --output ablation_fp8_e4m3fn_quantized_matmul_report.csv --output-dir ablation_fp8_e4m3fn_quantized_matmul_results

echo "======= TEST 32/33: int8 group_size=64 ======="
python scripts/benchmark_inference_quality.py $COMMON $FP16 \
    --pipeline-config ablation_int8_group64.json \
    --output ablation_int8_group64_report.csv --output-dir ablation_int8_group64_results

echo "======= TEST 33/33: int8 tensorwise (group_size=-1) ======="
python scripts/benchmark_inference_quality.py $COMMON $FP16 \
    --pipeline-config ablation_int8_tensorwise.json \
    --output ablation_int8_tensorwise_report.csv --output-dir ablation_int8_tensorwise_results

echo "======= TEST 34/37: uint4 + quantized matmul ======="
python scripts/benchmark_inference_quality.py $COMMON $FP16 \
    --pipeline-config ablation_uint4_quantized_matmul.json \
    --output ablation_uint4_quantized_matmul_report.csv --output-dir ablation_uint4_quantized_matmul_results

echo "======= TEST 35/37: uint4 + dequantize FP32 ======="
python scripts/benchmark_inference_quality.py $COMMON $FP16 \
    --pipeline-config ablation_uint4_dequant_fp32.json \
    --output ablation_uint4_dequant_fp32_report.csv --output-dir ablation_uint4_dequant_fp32_results

echo "======= TEST 36/37: uint4 group_size=64 ======="
python scripts/benchmark_inference_quality.py $COMMON $FP16 \
    --pipeline-config ablation_uint4_group64.json \
    --output ablation_uint4_group64_report.csv --output-dir ablation_uint4_group64_results

echo "======= TEST 37/37: uint4 tensorwise (group_size=-1) ======="
python scripts/benchmark_inference_quality.py $COMMON $FP16 \
    --pipeline-config ablation_uint4_tensorwise.json \
    --output ablation_uint4_tensorwise_report.csv --output-dir ablation_uint4_tensorwise_results

echo "======= ALL PHASE 4 TESTS COMPLETE ======="
