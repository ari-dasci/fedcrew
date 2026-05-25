from pathlib import Path


def test_ablation_runner_script_exists_and_uses_requested_baseline():
    script = Path("scripts/run_celeba_causal_ablation.sh")

    assert script.exists()
    content = script.read_text()

    assert "--dataset celeba" in content
    assert "--clients 30" in content
    assert "--rounds 100" in content
    assert "--epochs 5" in content
    assert "--samples 3" in content
    assert "--alpha 0.65" in content
    assert "--l1 0.01" in content
    assert "--causal" in content
