#!/usr/bin/env bash
# Cluster re-run matrix (docs/plan_review.md, item 4): every dataset x every
# implemented aggregation method x N seeds, load-balanced across a shared
# GPU set with multiple concurrent jobs per GPU.
#
# Usage:
#   GPUS=4,6,7 JOBS_PER_GPU=3 EXECUTE=1 ./scripts/run_cluster_matrix.sh
#
# Without EXECUTE=1 the script only prints the job matrix (dry run), same
# convention as scripts/run_celeba_causal_ablation.sh.
#
# Re-invoking the script (e.g. after fixing a bug) skips runs that already
# completed -- only missing/failed runs are (re)launched. Completion is
# tracked with local .done marker files, but those don't survive moving to a
# new cluster, so at startup the script first syncs them from wandb (the
# durable source of truth) via scripts/fetch_wandb_done_runs.py -- any run
# already finished under the current INSTRUMENTATION_VERSION tag, on any
# cluster, is seeded into DONE_DIR before the job matrix is built.
set -euo pipefail

# ---------------------------------------------------------------------------
# Config -- review/edit before the real cluster run.
# ---------------------------------------------------------------------------

GPUS="${GPUS:-0}"                    # comma-separated GPU ids, e.g. GPUS=4,6,7
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"    # concurrent jobs per GPU

# Each job spawns its own DataLoader worker processes (main.py --dataloader-workers,
# default 4) *and* its own torch CPU thread pool. With JOBS_PER_GPU concurrent
# python processes per GPU, both must be capped so total CPU usage across all
# concurrent jobs doesn't oversubscribe the host's cores.
IFS=',' read -ra _GPU_LIST_FOR_CPU_CALC <<< "$GPUS"
N_GPUS_FOR_CPU_CALC=${#_GPU_LIST_FOR_CPU_CALC[@]}
N_CONCURRENT_JOBS=$((N_GPUS_FOR_CPU_CALC * JOBS_PER_GPU))
HOST_CORES=$(nproc)
CORES_PER_JOB=$(( HOST_CORES / N_CONCURRENT_JOBS ))
[[ "$CORES_PER_JOB" -lt 1 ]] && CORES_PER_JOB=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CORES_PER_JOB}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-$CORES_PER_JOB}"

SEEDS=(0 1 2 3 4)
DATASETS=(cifar_10_non_iid cifar_100_non_iid celeba celeba_a celeba_m)
METHODS=(fedavg fedprox fednova fedcrew moon feddyn)   # fedper: add once implemented

FEDPROX_MU=0.01
MOON_TAU=1.0    # paper-stated default; overrides main.py's own --moon-tau default (0.5)
FEDDYN_ALPHA=0.01

# Per-dataset hyperparameters. Pre-filled from README.md / the existing
# ablation scripts as a starting point -- REVIEW/EDIT before the real run.
declare -A HP_CLIENTS=(
  [cifar_10_non_iid]=15 [cifar_100_non_iid]=15
  [celeba]=30 [celeba_a]=30 [celeba_m]=30
)
declare -A HP_ROUNDS=(
  [cifar_10_non_iid]=100 [cifar_100_non_iid]=100
  [celeba]=100 [celeba_a]=100 [celeba_m]=100
)
declare -A HP_EPOCHS=(
  [cifar_10_non_iid]=10 [cifar_100_non_iid]=10
  [celeba]=5 [celeba_a]=5 [celeba_m]=5
)
declare -A HP_SAMPLES=(
  [cifar_10_non_iid]=3 [cifar_100_non_iid]=3
  [celeba]=3 [celeba_a]=3 [celeba_m]=3
)
declare -A HP_ALPHA=(
  [cifar_10_non_iid]=0.5 [cifar_100_non_iid]=0.5
  [celeba]=0.5 [celeba_a]=0.5 [celeba_m]=0.5
)
declare -A HP_L1=(
  [cifar_10_non_iid]=0.01 [cifar_100_non_iid]=0.01
  [celeba]=0.01 [celeba_a]=0.01 [celeba_m]=0.01
)

LOG_DIR="logs/cluster_matrix"
DONE_DIR="${LOG_DIR}/.done"
FAILED_LOG="${LOG_DIR}/failed_runs.tsv"
LOCK_FILE="${LOG_DIR}/.failed_runs.lock"
QUEUE_LOCK="${LOG_DIR}/.queue.lock"
WANDB_GROUP="${WANDB_GROUP:-cluster-matrix-$(date +%Y%m%d)}"

mkdir -p "$DONE_DIR"
[[ -f "$FAILED_LOG" ]] || printf 'timestamp\trun_id\tdataset\tmethod\tseed\tgpu\texit_code\tlog\n' > "$FAILED_LOG"

# ---------------------------------------------------------------------------
# Sync completed-run state from wandb (source of truth across clusters) into
# the local .done markers, so a fresh cluster with no local state still
# skips already-finished runs. A transient wandb/network failure here should
# not abort the whole matrix -- fall back to whatever local .done state
# already exists. Skipped on dry runs (EXECUTE!=1), which don't consult
# .done markers, so a dry run doesn't need wandb access.
# ---------------------------------------------------------------------------

if [[ "${EXECUTE:-0}" == "1" ]]; then
  echo "Syncing completed-run state from wandb..."
  if ! uv run python scripts/fetch_wandb_done_runs.py > "${LOG_DIR}/.wandb_done_runs.tmp"; then
    echo "warning: wandb sync failed, falling back to local .done state only" >&2
  else
    while IFS= read -r run_id; do
      [[ -n "$run_id" ]] && touch "${DONE_DIR}/${run_id}"
    done < "${LOG_DIR}/.wandb_done_runs.tmp"
  fi
  rm -f "${LOG_DIR}/.wandb_done_runs.tmp"
fi

# ---------------------------------------------------------------------------
# Method -> extra main.py flags
# ---------------------------------------------------------------------------

method_flags() {
  local method="$1"
  case "$method" in
    fedavg)  echo ;;
    fedprox) echo "--fedprox $FEDPROX_MU" ;;
    fednova) echo "--fednova" ;;
    fedcrew) echo "--fedcrew --causal-mode full" ;;
    moon)    echo "--moon --moon-tau $MOON_TAU" ;;
    feddyn)  echo "--feddyn $FEDDYN_ALPHA" ;;
    *)
      echo "Unknown method: $method" >&2
      exit 1
      ;;
  esac
}

# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

run_job() {
  local gpu="$1" dataset="$2" method="$3" seed="$4"
  local run_id="${dataset}.${method}.seed${seed}"
  local done_marker="${DONE_DIR}/${run_id}"
  local log_file="${LOG_DIR}/${run_id}.log"

  if [[ -f "$done_marker" ]]; then
    echo "[gpu $gpu] skip (already done): $run_id"
    return
  fi

  local extra_flags
  read -r -a extra_flags <<< "$(method_flags "$method")"

  local cmd=(uv run python main.py
    --dataset "$dataset"
    --clients "${HP_CLIENTS[$dataset]}"
    --rounds "${HP_ROUNDS[$dataset]}"
    --epochs "${HP_EPOCHS[$dataset]}"
    --samples "${HP_SAMPLES[$dataset]}"
    --alpha "${HP_ALPHA[$dataset]}"
    --l1 "${HP_L1[$dataset]}"
    --seed "$seed"
    --dataloader-workers "$DATALOADER_WORKERS"
    --wandb-group "$WANDB_GROUP"
    --wandb-tags cluster-matrix
    "${extra_flags[@]}")

  echo "[gpu $gpu] start: $run_id"

  local ec=0
  if CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1; then
    ec=0
  else
    ec=$?
  fi

  if [[ "$ec" -eq 0 ]]; then
    touch "$done_marker"
    echo "[gpu $gpu] done: $run_id"
  else
    {
      flock 200
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Iseconds)" "$run_id" "$dataset" "$method" "$seed" "$gpu" "$ec" "$log_file" \
        >> "$FAILED_LOG"
    } 200>"$LOCK_FILE"
    echo "[gpu $gpu] FAILED (exit $ec): $run_id -- see $log_file" >&2
  fi
}

worker() {
  local gpu="$1"
  local job
  # The FIFO is a shared byte stream: without serializing the dequeue,
  # concurrent `read -u3` calls from multiple workers can interleave bytes
  # from different lines and corrupt job specs. flock (held on this worker's
  # own fd 9, opened once for the worker's lifetime) makes "read one line"
  # atomic across workers.
  exec 9>"$QUEUE_LOCK"
  while true; do
    flock 9
    if ! IFS= read -r -u 3 job; then
      flock -u 9
      break
    fi
    flock -u 9
    [[ "$job" == "__STOP__" ]] && break
    IFS=$'\t' read -r dataset method seed <<< "$job"
    run_job "$gpu" "$dataset" "$method" "$seed"
  done
  exec 9>&-
}

# ---------------------------------------------------------------------------
# Build job matrix
# ---------------------------------------------------------------------------

JOBS=()
for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOBS+=("${dataset}"$'\t'"${method}"$'\t'"${seed}")
    done
  done
done

TOTAL_JOBS=${#JOBS[@]}
echo "Total jobs in matrix: $TOTAL_JOBS"

if [[ "${EXECUTE:-0}" != "1" ]]; then
  echo "Dry run (set EXECUTE=1 to launch). Job list:"
  for job in "${JOBS[@]}"; do
    IFS=$'\t' read -r dataset method seed <<< "$job"
    flags_str=$(method_flags "$method")
    echo "dataset=$dataset method=$method seed=$seed" \
      "--clients ${HP_CLIENTS[$dataset]} --rounds ${HP_ROUNDS[$dataset]}" \
      "--epochs ${HP_EPOCHS[$dataset]} --samples ${HP_SAMPLES[$dataset]}" \
      "--alpha ${HP_ALPHA[$dataset]} --l1 ${HP_L1[$dataset]} $flags_str"
  done
  exit 0
fi

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPUS=${#GPU_LIST[@]}
N_WORKERS=$((N_GPUS * JOBS_PER_GPU))
echo "Launching $N_WORKERS workers across $N_GPUS GPU(s) (${GPUS}), $JOBS_PER_GPU per GPU"

QUEUE_FIFO=$(mktemp -u "${LOG_DIR}/queue.XXXXXX")
mkfifo "$QUEUE_FIFO"
exec 3<>"$QUEUE_FIFO"
rm -f "$QUEUE_FIFO"

WORKER_PIDS=()
for gpu in "${GPU_LIST[@]}"; do
  for _ in $(seq 1 "$JOBS_PER_GPU"); do
    worker "$gpu" &
    WORKER_PIDS+=("$!")
  done
done

for job in "${JOBS[@]}"; do
  echo "$job" >&3
done
for _ in $(seq 1 "$N_WORKERS"); do
  echo "__STOP__" >&3
done

wait "${WORKER_PIDS[@]}"
exec 3>&-

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

N_DONE=$(find "$DONE_DIR" -type f | wc -l)
# failed_runs.tsv accumulates one line per failed attempt across reruns, so a
# job retried and failed twice would double-count here -- report distinct
# run_ids that are still failing (logged as failed, no .done marker) instead.
N_STILL_FAILING=$(
  tail -n +2 "$FAILED_LOG" | cut -f2 | sort -u | while read -r run_id; do
    [[ -f "${DONE_DIR}/${run_id}" ]] || echo "$run_id"
  done | wc -l
)

echo ""
echo "=== Cluster matrix summary ==="
echo "Total jobs:          $TOTAL_JOBS"
echo "Completed:           $N_DONE"
echo "Still failing:       $N_STILL_FAILING"
if [[ "$N_STILL_FAILING" -gt 0 ]]; then
  echo "See $FAILED_LOG for failure details (rerun this script to retry; completed runs are skipped)."
fi
