import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExperimentConfig, parse_args


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


def test_final_artifact_paths_encode_dataset_method_seed_and_round():
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

    checkpoint_path = config.get_checkpoint_path(99)
    predictions_path = config.get_predictions_path(99)
    crp_map_path = config.get_crp_map_path(99, "server", 0)

    for path in (checkpoint_path, predictions_path, crp_map_path):
        assert config.dataset in path
        assert "seeded" in path
        assert "round99" in path
        assert "fedcrew" in path

    assert crp_map_path.endswith("client_server/sample0.pt")


def test_final_artifact_paths_use_configured_directories():
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
        checkpoint_dir="ckpt_root",
        predictions_dir="preds_root",
        crp_maps_dir="crp_root",
    )

    assert config.get_checkpoint_path(0).startswith("ckpt_root/")
    assert config.get_predictions_path(0).startswith("preds_root/")
    assert config.get_crp_map_path(0, 1, 2).startswith("crp_root/")


def test_no_final_artifacts_flag_disables_saving(monkeypatch):
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
            "--no-final-artifacts",
        ],
    )

    config = parse_args()

    assert config.save_final_artifacts is False


def test_moon_flags_parse_and_are_reflected_in_names(monkeypatch):
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
            "--moon",
            "--moon-mu",
            "2.0",
            "--moon-tau",
            "0.1",
        ],
    )

    config = parse_args()

    assert config.moon is True
    assert config.moon_mu == 2.0
    assert config.moon_tau == 0.1
    assert config.get_aggregator_name() == "moon"
    assert "moon" in config.get_wandb_run_name()


def test_wandb_tags_always_include_instrumentation_version_and_custom_tags():
    from config import INSTRUMENTATION_VERSION

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
        wandb_tags=["anchor-sweep"],
    )

    tags = config.get_wandb_tags()

    assert tags == [INSTRUMENTATION_VERSION, "anchor-sweep"]


def test_wandb_tags_default_to_just_instrumentation_version(monkeypatch):
    from config import INSTRUMENTATION_VERSION

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
        ],
    )

    config = parse_args()

    assert config.get_wandb_tags() == [INSTRUMENTATION_VERSION]


def test_causal_mode_uniform_disables_crp_and_logits(monkeypatch):
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
            "--fedcrew",
            "--causal-mode",
            "uniform",
        ],
    )

    config = parse_args()

    assert config.causal_mode == "uniform"
    assert config.causal_crp is False
    assert config.causal_logits is False
