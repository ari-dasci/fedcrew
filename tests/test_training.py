import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExperimentConfig
from utils import training


class FakeDataset:
    def __init__(self, samples):
        self.samples = samples

    def to_torchvision_dataset(self, transform=None):
        return self.samples


def test_obtain_metrics_with_predictions_returns_stable_per_sample_data(monkeypatch):
    monkeypatch.setattr(training, "get_transforms", lambda *a, **k: None)
    monkeypatch.setattr(training, "get_num_classes", lambda *a, **k: 2)

    samples = [
        (torch.tensor([1.0, 0.0]), 0),
        (torch.tensor([0.0, 1.0]), 1),
        (torch.tensor([2.0, 0.0]), 0),
    ]

    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))

    server_flex_model = {"model": model, "criterion": torch.nn.CrossEntropyLoss()}

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
    )

    loss, acc, confusion_matrix, predictions = training.obtain_metrics_with_predictions(
        server_flex_model, FakeDataset(samples), config, device="cpu"
    )

    assert predictions["logits"].shape == (3, 2)
    assert predictions["preds"].shape == (3,)
    assert predictions["targets"].shape == (3,)
    assert torch.equal(predictions["targets"], torch.tensor([0, 1, 0]))
    assert torch.equal(predictions["preds"], predictions["logits"].argmax(dim=1))
    assert acc == 1.0
