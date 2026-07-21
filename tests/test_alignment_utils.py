import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExperimentConfig
from utils import alignment_utils
from utils.alignment_utils import (
    compute_client_alignment,
    get_layer_activations,
    linear_cka,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Linear(3, 4, bias=False)
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.fc(self.feat(x))


def test_get_layer_activations_captures_named_layer_output_and_removes_hook():
    model = TinyNet()
    images = torch.randn(5, 3)

    activation = get_layer_activations(model, "feat", images)

    assert activation.shape == (5, 4)
    assert torch.equal(activation, model.feat(images))
    assert len(model.feat._forward_hooks) == 0


def test_linear_cka_identical_inputs_score_near_one():
    x = torch.randn(20, 6)
    score = linear_cka(x, x.clone())
    assert abs(score.item() - 1.0) < 1e-4


def test_linear_cka_uncorrelated_inputs_score_lower_than_identical():
    torch.manual_seed(0)
    x = torch.randn(50, 8)
    y = torch.randn(50, 8)

    same_score = linear_cka(x, x.clone())
    diff_score = linear_cka(x, y)

    assert diff_score.item() < same_score.item()


def test_compute_client_alignment_returns_expected_shapes(monkeypatch):
    n_clients = 3

    def fake_load_client_model(original_model, collected_weights, client_id, device):
        return TinyNet()

    monkeypatch.setattr(alignment_utils, "load_client_model", fake_load_client_model)
    monkeypatch.setattr(alignment_utils, "get_transforms", lambda *a, **k: None)
    monkeypatch.setattr(alignment_utils, "get_relevance_layer", lambda *a, **k: "feat")

    class FakeDataset:
        def to_torchvision_dataset(self, transform=None):
            return [(torch.randn(3), 0) for _ in range(4)]

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

    server_flex_model = {
        "model": TinyNet(),
        "weights": [torch.zeros(1) for _ in range(n_clients)],
    }

    result = compute_client_alignment(
        server_flex_model, None, FakeDataset(), config, device="cpu"
    )

    assert result["cka_matrix"].shape == (n_clients, n_clients)
    assert result["fc_divergence_matrix"].shape == (n_clients, n_clients)
    assert torch.allclose(
        torch.diagonal(result["cka_matrix"]), torch.ones(n_clients), atol=1e-4
    )
    assert torch.allclose(
        torch.diagonal(result["fc_divergence_matrix"]), torch.zeros(n_clients), atol=1e-4
    )
