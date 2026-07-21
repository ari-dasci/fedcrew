from pathlib import Path


def test_anchor_sensitivity_sweep_script_exists_and_sweeps_samples_and_anchor_seed():
    script = Path("scripts/run_anchor_sensitivity_sweep.sh")

    assert script.exists()
    content = script.read_text()

    assert "--dataset celeba" in content
    assert "--clients 30" in content
    assert "--rounds 100" in content
    assert "--epochs 5" in content
    assert "--samples \"$k\"" in content
    assert "--anchor-selection random" in content
    assert "--anchor-seed \"$anchor_seed\"" in content
    assert "--fedcrew" in content
    assert "--causal-mode full" in content
