"""
Benchmark workloads. Every input is generated from a fixed seed.

W0  model forward only on a random tensor (per-tile cost, VRAM).
W1  full inference loop (run_inference) on a synthetic uint8 volume with an
    empty band and a partially covered band (plumbing cost, empty-tile
    handling). No download needed.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKLOADS = ("W0", "W1")


def make_w0_input(batch_size: int, in_chans: int, tile_size: int, seed: int) -> torch.Tensor:
    """Random (B, 1, C, H, W) float32 in [0, 1], like the DataLoader output after ToFloat."""
    gen = torch.Generator().manual_seed(seed)
    return torch.rand((batch_size, 1, in_chans, tile_size, tile_size), generator=gen, dtype=torch.float32)


def make_w1_layers(size: int, in_chans: int, seed: int) -> np.ndarray:
    """
    Synthetic (H, W, C) uint8 surface volume, H = W = size.

    Row bands, top to bottom:
      [0, size/4)        empty: every voxel zero (whole tiles skip-eligible)
      [size/4, size/2)   partially covered: left half zero, right half data
      [size/2, size)     fully covered

    Data voxels are uniform in [1, 255] so every covered pixel is valid.
    """
    if size % 4:
        raise ValueError(f"W1 size must be a multiple of 4, got {size}")
    rng = np.random.default_rng(seed)
    layers = rng.integers(1, 256, size=(size, size, in_chans), dtype=np.uint8)
    q = size // 4
    layers[:q] = 0
    layers[q:2 * q, : size // 2] = 0
    return layers


def valid_pixel_count(layers: np.ndarray) -> int:
    """Pixels with at least one nonzero layer (the pipeline's validity rule)."""
    return int(np.count_nonzero(np.any(layers != 0, axis=-1)))


class StubModel(nn.Module):
    """
    Cheap stand-in with the decoder's input/output contract:
    (B, 1, C, H, W) -> (B, 1, H/4, W/4). For plumbing tests only.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.head = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x[:, None]
        x = self.conv(x).mean(dim=2)
        return self.head(F.avg_pool2d(x, 4))

    def get_output_scale_factor(self) -> int:
        return 4


def describe(workload: str, size: int) -> Dict[str, str]:
    if workload == "W0":
        return {"workload": "W0", "input": "random tensor"}
    return {"workload": "W1", "input": f"synthetic {size}x{size}, empty band + partial band"}
