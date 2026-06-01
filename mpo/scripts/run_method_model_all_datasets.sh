#!/bin/bash
# SLURM job script: run one method on all 6 datasets sequentially for one model.
# Submitted by submit_full_reproduction.sh with an afterok dependency on that
# model's 6 SFT jobs.
#
# Usage (via sbatch):
#   sbatch --job-name=DPO_M1 \
#          --dependency=afterok:<sft_jid1>:...<sft_jid6> \
#          scripts/run_method_model_all_datasets.sh DPO 1
#
# Positional args (passed through sbatch):
#   $1  method     – one of the 14 public method names
#   $2  model_idx  – 1..4

#SBATCH --partition=gpu-large
#SBATCH --cpus-per-gpu=8
#SBATCH --time=120:00:00
#SBATCH --mem=128G
# GPU selection policy:
# Priority order: H200 > H100 > A100
# Use A100 only if no H200/H100 GPUs are available
#SBATCH --qos=priority
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=thin.nguyen@deakin.edu.au

set -uo pipefail

# SLURM copies scripts to its spool dir, so BASH_SOURCE[0] points there, not here.
# SLURM_SUBMIT_DIR is set to the directory where sbatch was called (always PROJECT_ROOT).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p runs logs

method="${1:?method argument required}"
model_idx="${2:?model_idx argument required}"
batch_size="${3:-8}"   # optional: override batch size (e.g. 4 for OOM-prone methods)

datasets=(1 2 3 4 5 6)
dataset_names=(hh shp pku ultrabin ultrallama ultragemma)
model_names=(qwen05b tinyllama11b qwen3b llama7b llama70b)
model_name="${model_names[$((model_idx - 1))]}"

echo "================================================================"
echo " MPO – method=${method}  model_idx=${model_idx} (${model_name})"
echo " SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo " datasets: all 6 sequentially  batch_size=${batch_size}"
echo " start: $(date +'%Y-%m-%d %H:%M:%S')"
echo "================================================================"

job_start=$(date +%s)
n_ok=0
n_fail=0

for d in "${datasets[@]}"; do
    dname="${dataset_names[$((d - 1))]}"
    echo ""
    echo "--- [$(date +'%H:%M:%S')] dataset ${d} (${dname}) ---"

    # run_all.sh handles env setup (slurm_env.sh), checkpoint discovery, and training.
    # Its exec redirect sends detailed output to logs/<model>_<dataset>_<method>_<jid>.out.
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
echo " Summary: ${n_ok}/6 datasets succeeded, ${n_fail}/6 failed"
printf " Total elapsed: %dh %dm %ds\n" \
    "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
echo " end: $(date +'%Y-%m-%d %H:%M:%S')"
echo "================================================================"

if [ "${n_fail}" -gt 0 ]; then
    exit 1
fi
