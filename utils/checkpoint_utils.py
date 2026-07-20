"""Checkpoint saving utilities for the final training round."""

import os

import torch
from flex.data import Dataset
from flex.model import FlexModel

from config import ExperimentConfig


def save_checkpoint(
    server_flex_model: FlexModel,
    _: Dataset,
    config: ExperimentConfig,
    round_number: int,
) -> None:
    """Save the global model's state_dict to disk.

    Args:
        server_flex_model: Server model containing the trained model.
        _: Unused server dataset parameter (for FlexPool.map compatibility).
        config: Experiment configuration.
        round_number: Current federation round.
    """
    path = config.get_checkpoint_path(round_number)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(server_flex_model["model"].state_dict(), path)
