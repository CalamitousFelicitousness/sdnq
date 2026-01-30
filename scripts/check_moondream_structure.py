#!/usr/bin/env python3
"""Check the actual parameter structure of moondream3."""

import os

import torch
from transformers import AutoModelForCausalLM


def main():
    model_path = os.path.expanduser(
        "~/database/models/huggingface/models--moondream--moondream3-preview/snapshots/e86382f00368618bfbbef8026cb606e9c0e3cd0e"
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    # Exclusion patterns from our script
    exclusion_patterns = [
        "model.vision",
        ".vision",
        "model.region",
        ".region",
        ".ln",
        ".post_ln",
        ".lm_head",
        ".mlp.router",
        ".mlp.fc1.weight",
        ".mlp.fc2.weight",
        ".wte",
        ".freqs_cis",
        ".tau",
        ".kv_cache",
        ".coord_features",
        ".size_features",
    ]

    # Count parameters
    linear_weights = []
    excluded_linear = []
    other_params = []

    for name, param in model.named_parameters():
        # Check if excluded
        is_excluded = False
        for pattern in exclusion_patterns:
            if pattern.startswith("."):
                if pattern[1:] in name:
                    is_excluded = True
                    break
            elif pattern in name:
                is_excluded = True
                break

        # Check if it's a linear layer weight
        is_linear_weight = name.endswith('.weight') and param.ndim == 2

        if is_linear_weight:
            if is_excluded:
                excluded_linear.append((name, param.shape, param.numel()))
            else:
                linear_weights.append((name, param.shape, param.numel()))
        else:
            other_params.append((name, param.shape, param.numel()))

    total_linear_params = sum(p[2] for p in linear_weights)
    total_excluded_params = sum(p[2] for p in excluded_linear)
    total_other_params = sum(p[2] for p in other_params)
    total_params = total_linear_params + total_excluded_params + total_other_params

    print(f"\n{'='*80}")
    print("PARAMETER CATEGORIZATION")
    print(f"{'='*80}")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"\nLinear weights TO BE QUANTIZED: {len(linear_weights)} layers, {total_linear_params:,} params ({total_linear_params/total_params*100:.1f}%)")
    print(f"Linear weights EXCLUDED: {len(excluded_linear)} layers, {total_excluded_params:,} params ({total_excluded_params/total_params*100:.1f}%)")
    print(f"Other parameters: {len(other_params)} params, {total_other_params:,} total ({total_other_params/total_params*100:.1f}%)")

    if linear_weights:
        print(f"\n{'='*80}")
        print("LINEAR WEIGHTS TO BE QUANTIZED (first 30):")
        print(f"{'='*80}")
        for name, shape, numel in linear_weights[:30]:
            print(f"  {name:80s} {shape!s:30s} {numel:>15,} params")
        if len(linear_weights) > 30:
            print(f"  ... and {len(linear_weights) - 30} more")

    if excluded_linear:
        print(f"\n{'='*80}")
        print("EXCLUDED LINEAR WEIGHTS (first 30):")
        print(f"{'='*80}")
        for name, shape, numel in excluded_linear[:30]:
            print(f"  {name:80s} {shape!s:30s} {numel:>15,} params")
        if len(excluded_linear) > 30:
            print(f"  ... and {len(excluded_linear) - 30} more")


if __name__ == "__main__":
    main()
