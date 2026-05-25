import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExperimentConfig


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
