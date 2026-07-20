import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExperimentConfig
from utils import crp_utils


def test_compute_features_weights_per_client_skips_crp_when_disabled(monkeypatch):
    calls = {"logits": 0, "crp": 0}

    def fake_load_client_model(*args, **kwargs):
        return object()

    def fake_group_correct_class_probs(*args, **kwargs):
        calls["logits"] += 1
        return {0: [0.9]}

    def fake_get_crp_attribution(*args, **kwargs):
        calls["crp"] += 1
        return {0: torch.ones(1, 2)}

    monkeypatch.setattr(crp_utils, "load_client_model", fake_load_client_model)
    monkeypatch.setattr(
        crp_utils, "group_correct_class_probs", fake_group_correct_class_probs
    )
    monkeypatch.setattr(crp_utils, "get_crp_attribution", fake_get_crp_attribution)

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
        causal_mode="logits_only",
        causal_crp=False,
        causal_logits=True,
        anchor_selection="first",
        anchor_seed=None,
        wandb_group=None,
    )

    class ModelStub:
        def __init__(self):
            self.fc = torch.nn.Linear(2, 1, bias=False)

    server_model = {"model": ModelStub(), "weights": [torch.zeros(1)]}

    result = crp_utils.compute_features_weights_per_client(
        server_model, None, None, config, device="cpu"
    )

    assert calls["logits"] == 1
    assert calls["crp"] == 0
    assert result.shape == torch.Size([1, 1, 2])


def test_client_crp_saves_raw_tensor_only_when_requested(monkeypatch, tmp_path):
    saved = {}

    class FakeDataset:
        def to_torchvision_dataset(self, *args, **kwargs):
            return [(object(), 0)]

    def fake_get_transforms(*args, **kwargs):
        return None

    def fake_get_relevance_layer(*args, **kwargs):
        return "layer"

    def fake_extract_heatmap(*args, **kwargs):
        return torch.ones(1, 2), 1

    def fake_imgify(*args, **kwargs):
        return object()

    def fake_log_crp_heatmap(*args, **kwargs):
        pass

    def fake_torch_save(obj, path):
        saved["obj"] = obj
        saved["path"] = path

    monkeypatch.setattr(crp_utils, "get_transforms", fake_get_transforms)
    monkeypatch.setattr(crp_utils, "get_relevance_layer", fake_get_relevance_layer)
    monkeypatch.setattr(crp_utils, "extract_heatmap", fake_extract_heatmap)
    monkeypatch.setattr(crp_utils, "imgify", fake_imgify)
    monkeypatch.setattr(crp_utils, "log_crp_heatmap", fake_log_crp_heatmap)
    monkeypatch.setattr(crp_utils.torch, "save", fake_torch_save)

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
        crp_maps_dir=str(tmp_path),
    )

    crp_utils.client_crp(
        object(), "server", FakeDataset(), config, logger=None, round_number=0,
        device="cpu", save_raw=False,
    )
    assert "obj" not in saved

    crp_utils.client_crp(
        object(), "server", FakeDataset(), config, logger=None, round_number=0,
        device="cpu", save_raw=True,
    )
    assert torch.equal(saved["obj"], torch.ones(1, 2))
    assert saved["path"] == config.get_crp_map_path(0, "server", 0)
