#!/bin/bash
# Run one method on a specified subset of datasets sequentially for one model.
# Used by check_and_rerun.sh for consolidated reruns (<PFX><MIDX>r<N> jobs).
#
# Usage (via sbatch):
#   sbatch ... scripts/run_method_model_datasets.sh <method> <model_idx> <batch_size> <ds_idx...>
#
# Examples:
#   run_method_model_datasets.sh DPO 4 4 2 3 5 6   # llama7b DPO on shp/pku/ultrallama/ultragemma, bs=4
#   run_method_model_datasets.sh SimPO 2 8 1 3      # tinyllama11b SimPO on hh/pku, bs=8

#SBATCH --partition=gpu-large
#SBATCH --cpus-per-gpu=8
#SBATCH --qos=priority
#SBATCH --time=120:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -uo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p runs logs

method="${1:?method argument required}"
model_idx="${2:?model_idx argument required}"
batch_size="${3:-8}"
shift 3
dataset_indices=("$@")

if [ ${#dataset_indices[@]} -eq 0 ]; then
    echo "ERROR: no dataset indices provided" >&2
    exit 1
fi

dataset_names=(hh shp pku ultrabin ultrallama ultragemma)
model_names=(qwen05b tinyllama11b qwen3b llama7b llama70b)
model_name="${model_names[$((model_idx - 1))]}"

ds_labels=()
for d in "${dataset_indices[@]}"; do
    ds_labels+=("${dataset_names[$((d - 1))]}")
done

echo "================================================================"
echo " MPO – method=${method}  model_idx=${model_idx} (${model_name})"
echo " SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo " datasets: ${ds_labels[*]}"
echo " batch_size=${batch_size}"
echo " start: $(date +'%Y-%m-%d %H:%M:%S')"
echo "================================================================"

job_start=$(date +%s)
n_ok=0
n_fail=0
n_total=${#dataset_indices[@]}

for d in "${dataset_indices[@]}"; do
    dname="${dataset_names[$((d - 1))]}"
    echo ""
    echo "--- [$(date +'%H:%M:%S')] dataset ${d} (${dname}) ---"
    if bash "${PROJECT_ROOT}/run_all.sh" "${model_idx}" "${d}" "${method}" 0 "${batch_size}"; then
        echo "--- [$(date +'%H:%M:%S')] dataset ${d} (${dname}): DONE ---"
        n_ok=$((n_ok + 1))
    else
        rc=$?
        echo "--- [$(date +'%H:%M:%S')] dataset ${d} (${dname}): FAILED (exit ${rc}) ---"
        n_fail=$((n_fail + 1))
    fi
done

job_end=$(date +%s)
elapsed=$((job_end - job_start))

echo ""
echo "================================================================"
echo " Summary: ${n_ok}/${n_total} datasets succeeded, ${n_fail}/${n_total} failed"
printf " Total elapsed: %dh %dm %ds\n" \
    "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
echo " end: $(date +'%Y-%m-%d %H:%M:%S')"
echo "================================================================"

if [ "${n_fail}" -gt 0 ]; then
    exit 1
fi
