import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import parse_args


def test_causal_alias_enables_full_mode_and_metadata_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--dataset",
            "celeba",
            "--clients",
            "30",
            "--rounds",
            "100",
            "--epochs",
            "5",
            "--samples",
            "3",
            "--alpha",
            "0.65",
            "--l1",
            "0.01",
            "--causal",
            "--seed",
            "7",
        ],
    )

    config = parse_args()

    assert config.fedcrew is True
    assert config.causal_mode == "full"
    assert config.causal_crp is True
    assert config.causal_logits is True
    assert config.anchor_selection == "first"
    assert config.anchor_seed is None
