import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExperimentConfig
from utils.checkpoint_utils import save_checkpoint


def test_save_checkpoint_round_trips_state_dict(tmp_path):
    model = torch.nn.Linear(2, 1)
    server_flex_model = {"model": model}

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
        checkpoint_dir=str(tmp_path),
    )

    save_checkpoint(server_flex_model, None, config, round_number=99)

    path = config.get_checkpoint_path(99)
    loaded = torch.load(path)

    for key, value in model.state_dict().items():
        assert torch.equal(loaded[key], value)
