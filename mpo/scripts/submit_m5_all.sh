#!/bin/bash
# Submit M5 SFT jobs (6) + 14 methods × 3 pair-jobs (2 datasets each = 42 method jobs).
# QLoRA: SFT uses h100:1 (1 model), methods use h100:2 (policy + frozen reference).
#
# Usage:
#   bash scripts/submit_m5_all.sh                             # submit SFT + methods
#   bash scripts/submit_m5_all.sh --skip-sft JID1:...:JID6   # methods only, given SFT JIDs
#
# Run from PROJECT_ROOT.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SBATCH_SFT=(--partition=gpu-large --gpus=h100:1 --mem=80G --qos=priority --time=120:00:00 --cpus-per-gpu=8)
SBATCH_METHOD=(--nodes=1 --partition=gpu-large --gpus=h100:2 --mem=80G --qos=priority --time=120:00:00 --cpus-per-gpu=8)

declare -A DS_IDX=([hh]=1 [shp]=2 [pku]=3 [ultrabin]=4 [ultrallama]=5 [ultragemma]=6)
DATASETS=(hh shp pku ultrabin ultrallama ultragemma)

METHODS=(DPO betaDPO SimPO TDPO TIS CDPO CW MPO-TS MPO-Dual MPO-EMA MPO-LN MPO-Safe MPO-Conf MPO-ConfSafe)
declare -A METHOD_PFX=(
  [DPO]=DPO [betaDPO]=BDPO [SimPO]=SIM [TDPO]=TDPO [TIS]=TIS
  [CDPO]=CDPO [CW]=CW [MPO-TS]=MTS [MPO-Dual]=MDL [MPO-EMA]=MEMA
  [MPO-LN]=MLN [MPO-Safe]=MSFE [MPO-Conf]=MCF [MPO-ConfSafe]=MCSF
)

# Parse --skip-sft flag
SKIP_SFT=0
if [[ "${1:-}" == "--skip-sft" ]]; then
  SKIP_SFT=1
  SFT_DEP_STR="${2:?'--skip-sft requires JID1:JID2:...:JID6 as second argument'}"
fi

if [[ "$SKIP_SFT" == "0" ]]; then
  SFT_JIDS=()
  echo "=== Submitting M5 SFT jobs (QLoRA, h100:1) ==="
  for ds in "${DATASETS[@]}"; do
    di="${DS_IDX[$ds]}"
    jname="S5D${di}"
    jid=$(sbatch "${SBATCH_SFT[@]}" \
      --job-name="$jname" \
      --output="runs/slurm-%j_${jname}.out" \
      --error="runs/slurm-%j_${jname}.err" \
      run_all.sh 5 "$di" DPO 1 16 | awk '{print $NF}')
    SFT_JIDS+=("$jid")
    echo "  $jname ($ds, D${di}) → job $jid"
  done
  DEP="afterok:$(IFS=:; echo "${SFT_JIDS[*]}")"
  echo "SFT job IDs: ${SFT_JIDS[*]}"
else
  DEP="afterok:${SFT_DEP_STR}"
  echo "Skipping SFT submission; using dependency: $DEP"
fi

# Dataset pairs: 3 jobs per method, 2 datasets each
declare -A PAIR_DS=([1]="1 2" [2]="3 4" [3]="5 6")

echo ""
echo "=== Submitting 14×3=42 M5 method pair-jobs (2 datasets each, h100:2) ==="
echo "    Dependency: $DEP"
for method in "${METHODS[@]}"; do
  pfx="${METHOD_PFX[$method]}"
  for pair in 1 2 3; do
    ds_pair="${PAIR_DS[$pair]}"
    jname="${pfx}5r${pair}"   # e.g. DPO5r1/r2/r3 — matches is_covered() pattern
    jid=$(sbatch "${SBATCH_METHOD[@]}" \
      --dependency="$DEP" \
      --job-name="$jname" \
      --output="runs/slurm-%j_${jname}.out" \
      --error="runs/slurm-%j_${jname}.err" \
      scripts/run_method_model_datasets.sh "$method" 5 16 $ds_pair | awk '{print $NF}')
    echo "  $jname ($method, datasets $ds_pair) → job $jid"
  done
done

echo ""
echo "Done. $(date)"
