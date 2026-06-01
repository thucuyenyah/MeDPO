#!/bin/bash
# Scan all training logs, report status, and auto-rerun any FAILED/FAIL-ARCH
# combinations that are not already covered by a running or pending SLURM job.
#
# Usage:
#   cd /weka/thinng/2026/icdm/mpo
#   bash scripts/check_and_rerun.sh [--dry-run]
#
# FAIL-ARCH: training completed (FINAL eval written) but Done: line missing
#            → archive step failed, typically because SFT ckpt was absent.
# FAILED:    training did not complete (no FINAL eval) → OOM, cancelled, etc.
#
# Rerun strategy: failures for a (model, method) are grouped into ONE consolidated
# job using scripts/run_method_model_datasets.sh, named <PFX><MIDX>r<N> where
# N = number of failed datasets (e.g. DPO4r4, MCSF3r2). Max name length = 7 chars.
#
# Covering rule: a failure is already covered if squeue contains any job whose
# name matches the patterns in is_covered() below.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS="$PROJECT_ROOT/logs"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── mappings ────────────────────────────────────────────────────────────────
declare -A MODEL_IDX=([qwen05b]=1 [tinyllama11b]=2 [qwen3b]=3 [llama7b]=4 [llama70b]=5)
declare -A MODEL_MN=( [qwen05b]=M1 [tinyllama11b]=M2 [qwen3b]=M3 [llama7b]=M4 [llama70b]=M5)
declare -A DS_IDX=([hh]=1 [shp]=2 [pku]=3 [ultrabin]=4 [ultrallama]=5 [ultragemma]=6)
declare -A DS_ABB=([hh]=HH [shp]=SH [pku]=PK [ultrabin]=UB [ultrallama]=UL [ultragemma]=UG)
declare -A METHOD_PFX=(
  [DPO]=DPO [betaDPO]=BDPO [SimPO]=SIM [TDPO]=TDPO [TIS]=TIS
  [CDPO]=CDPO [CW]=CW [MPO-TS]=MTS [MPO-Dual]=MDL [MPO-EMA]=MEMA
  [MPO-LN]=MLN [MPO-Safe]=MSFE [MPO-Conf]=MCF [MPO-ConfSafe]=MCSF
)

MODELS=(qwen05b tinyllama11b qwen3b llama7b llama70b)
DATASETS=(hh shp pku ultrabin ultrallama ultragemma)
METHODS=(DPO betaDPO SimPO TDPO TIS CDPO CW MPO-TS MPO-Dual MPO-EMA MPO-LN MPO-Safe MPO-Conf MPO-ConfSafe)

# ── GPU resources per model ──────────────────────────────────────────────────
gpu_flags() {
  local model=$1
  case "$model" in
    # llama70b: QLoRA — policy + frozen reference each 35GB; 2×H100 same node via device_map=balanced
    llama70b) echo "--nodes=1 --partition=gpu-large --gpus=h100:2 --mem=80G --qos=priority --time=120:00:00" ;;
    # llama7b: H200, 5-day wall, priority QOS — consolidated reruns can run 6 datasets × ~12h
    llama7b)  echo "--partition=gpu-large --gpus=h200:1 --mem=128G --qos=priority --time=120:00:00" ;;
    # smaller models: H100, 12h is enough for any single-model consolidated rerun
    *)        echo "--partition=gpu-large --gpus=h100:1 --mem=80G  --qos=batch-short --time=12:00:00" ;;
  esac
}

# ── snapshot of queued job names ─────────────────────────────────────────────
RUNNING_JOBS=$(squeue -u "$(whoami)" -h -o "%i" 2>/dev/null | tr '\n' ' ')
RUNNING_NAMES=$(squeue -u "$(whoami)" -h -o "%j" 2>/dev/null | tr '\n' ' ')

is_running_jid() { echo "$RUNNING_JOBS" | grep -qw "$1"; }

# is_covered: returns 0 if this (model, method) already has a queued job covering it.
# Args: pfx mn ds midxn   (ds and midxn are optional for legacy patterns)
is_covered() {
  local pfx=$1 mn=$2 ds="${3:-}" midxn="${4:-}"
  # original all-datasets job names: DPO_M3, betaDPO_M1, etc.
  echo "$RUNNING_NAMES" | grep -qw "${pfx}_${mn}" && return 0
  # original targeted per-dataset jobs: DPO_M3_pku, etc.
  echo "$RUNNING_NAMES" | grep -qE "(^| )${pfx}_${mn}_" && return 0
  # legacy alias: SIM3UL covers qwen3b/ultrallama/SimPO
  [[ "$pfx" == "SIM" && "$mn" == "M3" ]] && echo "$RUNNING_NAMES" | grep -qw "SIM3UL" && return 0
  # compact all-datasets jobs: TDPOMx / TISMx / BDPOMx
  local compact="${pfx}${mn}"
  echo "$RUNNING_NAMES" | grep -qw "$compact" && return 0
  # consolidated rerun jobs: DPO4r4, MCSF3r2, etc.
  if [ -n "$midxn" ]; then
    echo "$RUNNING_NAMES" | grep -qE "(^| )${pfx}${midxn}r[0-9]" && return 0
  fi
  # legacy compact per-dataset reruns: DPO4SH, MCSF4UG, etc.
  if [ -n "$ds" ] && [ -n "$midxn" ]; then
    local ds2="${DS_ABB[$ds]:-${ds:0:2}}"
    echo "$RUNNING_NAMES" | grep -qw "${pfx}${midxn}${ds2}" && return 0
  fi
  return 1
}

# ── scan + act ───────────────────────────────────────────────────────────────
ok=0; running=0; missing=0; fail=0; resubmitted=0
# note: use ok=$((ok+1)) not ((ok++)) — the latter exits 1 when result=0 under set -e

printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
       model dataset method job_id status accuracy
printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
       --------------- ------------ -------------- -------- ---------- ----------

# Loop order: model → method → dataset
# Failures for each (model, method) are accumulated and submitted as one job.
for model in "${MODELS[@]}"; do
  mn="${MODEL_MN[$model]}"
  midx="${MODEL_IDX[$model]}"
  for method in "${METHODS[@]}"; do
    pfx="${METHOD_PFX[$method]}"
    pending_ds=()    # dataset names for uncovered failures
    pending_didx=()  # dataset indices for uncovered failures

    for dataset in "${DATASETS[@]}"; do
      didx="${DS_IDX[$dataset]}"

      latest=$(ls "$LOGS"/${model}_${dataset}_${method}_*.out 2>/dev/null | \
               sed 's/.*_\([0-9]*\)\.out/\1 &/' | sort -n | tail -1 | awk '{print $2}' || true)

      if [ -z "$latest" ]; then
        # For llama70b: also submit MISSING if the SFT checkpoint exists
        # (SFT checkpoint signals the model is ready for method training)
        if [[ "$model" == "llama70b" ]]; then
          sft_dir=$(find "$PROJECT_ROOT/.cache/thinng" -maxdepth 1 -type d \
                    -name "${dataset}_${model}_sft*" 2>/dev/null | sort | tail -1 || true)
          sft_ckpt="${sft_dir}/LATEST/policy.pt"
          if [[ -n "$sft_dir" && -f "$sft_ckpt" ]]; then
            if is_covered "$pfx" "$mn" "$dataset" "$midx"; then
              printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
                     "$model" "$dataset" "$method" "-" "MISSING(cov)" "-"
              missing=$((missing+1))
            else
              printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
                     "$model" "$dataset" "$method" "-" "MISSING(pend)" "-"
              missing=$((missing+1))
              pending_ds+=("$dataset")
              pending_didx+=("$didx")
            fi
          else
            printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
                   "$model" "$dataset" "$method" "-" "MISSING(noSFT)" "-"
            missing=$((missing+1))
          fi
        else
          printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
                 "$model" "$dataset" "$method" "-" "MISSING" "-"
          missing=$((missing+1))
        fi
        continue
      fi

      jid=$(basename "$latest" .out | sed 's/.*_//')

      if is_running_jid "$jid"; then
        printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
               "$model" "$dataset" "$method" "$jid" "RUNNING" "-"
        running=$((running+1))
        continue
      fi

      final_line=$(grep "^FINAL eval:" "$latest" 2>/dev/null | tail -1 || true)
      # Accept both canonical "Done:" and legacy "betaDPO done." (pre-fix logs)
      done_line=$(grep -E "^Done:|^betaDPO done\." "$latest" 2>/dev/null | tail -1 || true)

      if [ -n "$final_line" ] && [ -n "$done_line" ]; then
        acc=$(echo "$final_line" | grep -oP "(?<='rewards_eval/accuracies': ')[0-9.]+" || echo "0")
        acc_pct=$(printf "%.2f%%" "$(echo "$acc * 100" | bc -l)")
        printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
               "$model" "$dataset" "$method" "$jid" "SUCCESS" "$acc_pct"
        ok=$((ok+1))
        continue
      fi

      # Failure — determine status
      if [ -n "$final_line" ]; then
        status="FAIL-ARCH"
      else
        status="FAILED"
      fi

      if is_covered "$pfx" "$mn" "$dataset" "$midx"; then
        printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
               "$model" "$dataset" "$method" "$jid" "${status}(cov)" "-"
        fail=$((fail+1))
        continue
      fi

      # For llama70b failures, only resubmit once the SFT checkpoint exists.
      # Old method logs from pre-NCCL-fix runs must not trigger premature resubmission.
      if [[ "$model" == "llama70b" ]]; then
        sft_dir=$(find "$PROJECT_ROOT/.cache/thinng" -maxdepth 1 -type d \
                  -name "${dataset}_${model}_sft*" 2>/dev/null | sort | tail -1 || true)
        sft_ckpt="${sft_dir}/LATEST/policy.pt"
        if [[ -z "$sft_dir" || ! -f "$sft_ckpt" ]]; then
          printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
                 "$model" "$dataset" "$method" "$jid" "${status}(noSFT)" "-"
          fail=$((fail+1))
          continue
        fi
      fi

      # Accumulate for consolidated rerun
      printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
             "$model" "$dataset" "$method" "$jid" "${status}(pend)" "-"
      pending_ds+=("$dataset")
      pending_didx+=("$didx")
      fail=$((fail+1))
    done

    # Submit pending datasets of this (model, method).
    # llama70b: 1 dataset per job (each takes ~41h; 2+ would exceed 120h wall time).
    # Others: one consolidated job for all pending datasets.
    if [ ${#pending_ds[@]} -gt 0 ]; then
      bs=8
      [[ "$model" == "llama7b"  ]] && bs=4
      [[ "$model" == "llama70b" ]] && bs=16

      read -ra gflags <<< "$(gpu_flags "$model")"
      qos_flag="--qos=batch-short"
      [[ "$model" == "llama70b" || "$model" == "llama7b" ]] && qos_flag=""

      if [[ "$model" == "llama70b" ]]; then
        # Submit one job per dataset to stay within the 120h wall time limit
        for i in "${!pending_ds[@]}"; do
          ds="${pending_ds[$i]}"
          di="${pending_didx[$i]}"
          # Shorten ds abbreviation for job name (≤2 chars) to stay ≤8 chars total
          ds_ab="${DS_ABB[$ds]:-${ds:0:2}}"
          jname="${pfx}5${ds_ab}"   # e.g. DPO5HH, SIM5SH — always ≤8 chars
          cmd=(sbatch "${gflags[@]}"
               --cpus-per-gpu=8
               --job-name="$jname"
               --output="$PROJECT_ROOT/runs/slurm-%j_${jname}.out"
               --error="$PROJECT_ROOT/runs/slurm-%j_${jname}.err"
               "$PROJECT_ROOT/scripts/run_method_model_datasets.sh"
               "$method" "$midx" "$bs" "$di")
          if [ "$DRY_RUN" = "1" ]; then
            echo "  [DRY] $jname ($ds): ${cmd[*]}"
          else
            new_jid=$("${cmd[@]}" | awk '{print $NF}')
            echo "  submitted $jname ($ds) → job $new_jid"
            resubmitted=$((resubmitted + 1))
          fi
        done
      else
        n=${#pending_ds[@]}
        jname="${pfx}${midx}r${n}"   # e.g. DPO4r4, MCSF3r2 — always ≤8 chars
        cmd=(sbatch "${gflags[@]}"
             --cpus-per-gpu=8
             ${qos_flag:+"$qos_flag"}
             --job-name="$jname"
             --output="$PROJECT_ROOT/runs/slurm-%j_${jname}.out"
             --error="$PROJECT_ROOT/runs/slurm-%j_${jname}.err"
             "$PROJECT_ROOT/scripts/run_method_model_datasets.sh"
             "$method" "$midx" "$bs" "${pending_didx[@]}")
        if [ "$DRY_RUN" = "1" ]; then
          echo "  [DRY] $jname (${pending_ds[*]}): ${cmd[*]}"
        else
          new_jid=$("${cmd[@]}" | awk '{print $NF}')
          echo "  submitted $jname (${pending_ds[*]}) → job $new_jid"
          resubmitted=$((resubmitted + n))
        fi
      fi
    fi
  done
done

echo ""
echo "=== SUMMARY $(date '+%Y-%m-%d %H:%M') ==="
printf "  SUCCESS     : %d / 420\n" "$ok"
printf "  RUNNING     : %d\n"       "$running"
printf "  FAIL(cov)   : %d\n"       "$((fail - resubmitted))"
printf "  RESUBMITTED : %d\n"       "$resubmitted"
printf "  MISSING     : %d\n"       "$missing"
