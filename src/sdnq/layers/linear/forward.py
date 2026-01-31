# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import Tuple

import torch

from ...common import use_contiguous_mm


def check_mats(input: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    input = input.contiguous()
    if use_contiguous_mm:
        weight = weight.contiguous()
    elif weight.is_contiguous():
        weight = weight.t().contiguous().t()
    return input, weight


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    weight = self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, self.svd_up, self.svd_down)
    return torch.nn.functional.linear(input.to(dtype=weight.dtype), weight, self.bias)
