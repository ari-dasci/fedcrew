# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working style

Do NOT use subagents (the `Agent` tool — Explore, Plan, general-purpose, forks, etc.) in this
repository, including during plan-mode exploration. They consume excessive usage; do
exploration and implementation work directly instead.

## Project

FedCReW: a federated learning approach that uses Concept Relevance Propagation (CRP) to weight
client contributions during aggregation, instead of plain FedAvg. See README.md for the full
algorithm description (client training → CRP relevance analysis on an anchor set → weighted
aggregation of the classifier head, plain averaging of the backbone).

Built on top of the [FLEX](https://github.com/FLEXible-FL/FLEX-framework) federated learning
framework (`flexible-fl` package, imported as `flex`), plus Zennit/CRP
(`zennit-crp` package, imported as `crp`) for concept relevance propagation.

## Commands

This project uses `uv` for dependency management — always prefix commands with `uv run`, never
call `python`/`pytest`/`ruff`/`ty` directly (they're not on PATH outside the venv).

```bash
# Run an experiment
uv run python main.py --dataset cifar_10 --fedcrew --rounds 100 --clients 25 --epochs 10

# Tests — pytest is NOT a project dependency (only ruff/ty are in the dev group),
# so it must be pulled in ad hoc:
uv run --with pytest pytest                      # run all tests
uv run --with pytest pytest tests/test_config.py -k test_name  # run a single test

# Lint / type-check
uv run ruff check .
uv run ty check
```

Running `main.py` requires CUDA (`setup_logging` asserts `torch.cuda.is_available()`); it cannot
be smoke-tested on a CPU-only box. The test suite mocks around the dataset/CRP/CUDA dependencies
instead of exercising the full pipeline.

`scripts/run_celeba_causal_ablation.sh` sweeps `--causal-mode` × `--seed` × `--anchor-seed` for the
CelebA causal ablation; it only prints the commands unless run with `EXECUTE=1`.

## Architecture

The training loop (`main.py: train_pool`) drives a FLEX `FlexPool` (server + clients) through
federation rounds. Per round: copy server weights to selected clients → local `train` →
aggregate. Aggregation is chosen once per run based on config flags, in this priority:
`fedcrew` > `fednova` > plain `fed_avg` (FedProx isn't a separate aggregator — it adds a
regularization term inside client-side `train`, see `utils/fedprox.py`, and can combine with
FedCReW/FedNova since it's independent of the aggregation function).

**FedCReW aggregation** (`utils/flex_boilerplate.py: fedcrew_weighted_average`) treats the model
as backbone + head: all non-final layers are FedAvg'd; the final (`fc`) layer is combined via a
per-class, per-client `ponderation_tensor` of shape `(n_labels, n_clients, n_features)`, then
renormalized so per-class weight norms match their mean. That tensor is computed once per round in
`utils/crp_utils.py: compute_features_weights_per_client`, which for each client model:
- runs CRP attribution on a fixed anchor subsample (`get_crp_attribution`, via `zennit`/`crp` on
  the layer named by `models.py: fetch_relevance_layer` — `layer4.1.conv2` for ResNet-18,
  `fc_mid` for `MNISTNet`), and/or
- computes correct-class softmax confidence on the same anchor subsample
  (`group_correct_class_probs`), zeroed out per-class if any sample scores below 0.5.

Which of these two signals are used is controlled by `config.causal_crp` / `config.causal_logits`
(derived from `--causal-mode {full,crp_only,logits_only}` or the individual
`--causal-crp`/`--causal-logits` flags — see the reconciliation logic in `config.py: parse_args`).
`--causal` is a deprecated alias for `--fedcrew`, kept only for backward compatibility.

The anchor set ("$D_S$" in the paper) is a small per-class sample drawn once from the server's
test data at round 0 (`utils/data_utils.py: select_subsample_server_data`), then held fixed and
excluded from the remaining test set for the rest of the run. Selection is deterministic
(`--anchor-selection first`) or seeded-random (`--anchor-selection random --anchor-seed N`).

**Model/dataset registries are config-driven, not polymorphic**: `models.py: DATASET_CONFIG` and
`datasets.py: DATASET_CONFIG` are separate dicts keyed by the same dataset name strings
(`cifar_10`, `cifar_10_non_iid`, `cifar_100_non_iid`, `celeba`, `celeba_a`, `celeba_m`,
`mnist_non_iid`), each mapping to transforms/model-factory or a FLEX dataset loader respectively.
Adding a dataset means adding an entry to both, plus a relevance-layer case in
`models.py: fetch_relevance_layer` if it uses a new architecture. Non-IID
CIFAR-10/CIFAR-100/CelebA/EMNIST loaders cache their FLEX-partitioned dataset to disk (`*.pck`
files via `dill`) so re-runs skip the expensive partitioning step.

Other aggregation variants live alongside FedCReW: `utils/fednova.py` (iteration-count-weighted
softmax aggregation), `utils/fedprox.py` (proximal regularization term added to client loss),
and `utils/feddyn.py` (FedDyn's linear + quadratic dynamic-regularization term, plus a
per-client persisted gradient state updated after each round).

Logging (`utils/logging_utils.py`) writes to both TensorBoard (`runs/{dataset}/...`, filename
encodes most config flags via `ExperimentConfig.get_summary_writer_filename`) and Weights & Biases
(project `crp_aggregation`, run name via `get_wandb_run_name`); both can be disabled with
`--no_log`. Client-level metrics log every 5 rounds; CRP heatmaps for the server model log every 5
rounds after round 0.

## Current work context

The paper (FedCReW, FGCS) is under major revision, due September 2026 — see `docs/plan_review.md`
for the reviewer-driven work plan (new baselines FedPer/MOON, CIFAR-100, CKA/head-divergence
analysis, checkpoint/logit/CRP-map instrumentation, 4-way ablation, anchor-sensitivity sweep).
Before the cluster re-run, the FLEX-based train/eval loop needs instrumentation added: per-run
checkpoints, per-sample logits at the final round, and CRP relevance maps at the final round —
nothing from the original experiments was saved, so this re-run must be fully instrumented.
