import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExperimentConfig
from utils.logging_utils import save_predictions


def test_wandb_run_name_includes_causal_metadata():
    config = ExperimentConfig(
        dataset="celeba",
        clients=30,
        fedcrew=True,
        lognum=0,
        epochs=5,
        batchsize=64,
        clipgradients=False,
        samples=3,
        no_log=True,
        fedprox=0.0,
        fednova=False,
        rounds=100,
        l1=0.01,
        l2=0.0,
        alpha=0.65,
        l2_fc=0.0,
        seed=7,
        causal_mode="crp_only",
        causal_crp=True,
        causal_logits=False,
        anchor_selection="random",
        anchor_seed=11,
        wandb_group="celeba-causal-ablation",
    )

    run_name = config.get_wandb_run_name()

    assert "crp_only" in run_name
    assert "random" in run_name
    assert "anchor11" in run_name


def test_save_predictions_round_trips(tmp_path):
    config = ExperimentConfig(
        dataset="celeba",
        clients=30,
        fedcrew=True,
        lognum=0,
        epochs=5,
        batchsize=64,
        clipgradients=False,
        samples=3,
        no_log=True,
        fedprox=0.0,
        fednova=False,
        rounds=100,
        l1=0.01,
        l2=0.0,
        alpha=0.65,
        l2_fc=0.0,
        seed=7,
        predictions_dir=str(tmp_path),
    )

    predictions = {
        "logits": torch.tensor([[0.1, 0.9]]),
        "preds": torch.tensor([1]),
        "targets": torch.tensor([1]),
    }

    save_predictions(predictions, config, round_number=42)

    loaded = torch.load(config.get_predictions_path(42))
    assert torch.equal(loaded["logits"], predictions["logits"])
    assert torch.equal(loaded["preds"], predictions["preds"])
    assert torch.equal(loaded["targets"], predictions["targets"])
