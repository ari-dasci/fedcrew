"""Print run_ids of finished, current-instrumentation wandb runs.

Used by scripts/run_cluster_matrix.sh to seed its local .done markers from
wandb, so re-runs on a fresh cluster (with no local state) still skip jobs
that already completed successfully in a previous session -- wandb is the
durable source of truth, the local .done files are just a fast cache of it.

Output: one "{dataset}.{aggregator}.seed{seed}" per line (matching the
run_id format run_cluster_matrix.sh uses for its .done filenames), for every
run in the "crp_aggregation" project that is state=="finished" and tagged
with the current INSTRUMENTATION_VERSION.
"""

import sys

import wandb

from config import INSTRUMENTATION_VERSION

WANDB_PROJECT = "crp_aggregation"


def main() -> int:
    api = wandb.Api()
    runs = api.runs(
        WANDB_PROJECT,
        filters={"tags": INSTRUMENTATION_VERSION, "state": "finished"},
    )

    for run in runs:
        dataset = run.config.get("dataset")
        aggregator = run.config.get("aggregator")
        seed = run.config.get("seed")
        if dataset is None or aggregator is None or seed is None:
            continue
        print(f"{dataset}.{aggregator}.seed{seed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
