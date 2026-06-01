#!/bin/bash
# Rerun SimPO on qwen3b + ultrallama only.
# Wraps run_all.sh with targeted SBATCH overrides.
# SFT checkpoint verified present before submission.
#
# Usage: bash scripts/rerun_qwen3b_ultrallama_simpo.sh [--dry-run]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

SFT_CKP="$PROJECT_ROOT/.cache/thinng/ultrallama_qwen3b_sft_2026-05-27_12-51-52_950825/LATEST/policy.pt"
if [ ! -f "$SFT_CKP" ]; then
    echo "ERROR: SFT checkpoint missing: $SFT_CKP" >&2
    echo "Run SFT first before rerunning SimPO." >&2
    exit 1
fi
echo "SFT checkpoint OK: $SFT_CKP"

CMD=(sbatch
    --partition=gpu-large
    --gpus=h100:1
    --cpus-per-gpu=8
    --time=24:00:00
    --mem=80G
    --qos=batch-short
    --job-name=SIM3UL
    --output=runs/slurm-%j_%x.out
    --error=runs/slurm-%j_%x.err
    "$PROJECT_ROOT/run_all.sh" 3 5 SimPO 0
)

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] ${CMD[*]}"
else
    JID=$("${CMD[@]}" | awk '{print $NF}')
    echo "Submitted: job $JID"
    echo "  out : runs/slurm-${JID}_SIM3UL.out"
    echo "  err : runs/slurm-${JID}_SIM3UL.err"
fi
