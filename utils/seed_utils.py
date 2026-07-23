"""Seed utilities for reproducibility."""

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Fix the random seed across all random generators for reproducibility.

    Sets the seed for Python's built-in random module, NumPy, PyTorch (CPU and
    all CUDA devices). RNG seeding (init/data order) is reproducible regardless
    of `deterministic`; what `deterministic` controls is only cuDNN's kernel
    selection.

    Args:
        seed: Integer seed value to use across all generators.
        deterministic: If True, forces bitwise-deterministic cuDNN kernels
            (`cudnn.deterministic=True`, `cudnn.benchmark=False`), which
            disables the tensor-core-optimized convolution algorithms and is
            substantially slower. If False (default), cuDNN autotunes the
            fastest kernel for each input shape (`benchmark=True`) -- runs
            stay statistically reproducible across seeds, just not bit-for-bit
            identical to a `deterministic=True` run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def configure_backends() -> None:
    """Enable H100 tensor-core math paths (TF32 matmul/conv) for fp32 ops.

    Safe to call unconditionally, including on CPU-only/CI environments -- these
    flags are inert without CUDA. Call once at process startup, before any model
    is built or seed_everything() (whose `deterministic` flag independently
    controls cuDNN's autotuning/determinism trade-off).
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
