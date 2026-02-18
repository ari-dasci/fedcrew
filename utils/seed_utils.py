"""Seed utilities for reproducibility."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fix the random seed across all random generators for reproducibility.

    Sets the seed for Python's built-in random module, NumPy, PyTorch (CPU and
    all CUDA devices), and configures cuDNN for deterministic behaviour.

    Args:
        seed: Integer seed value to use across all generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
