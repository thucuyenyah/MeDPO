#!/bin/bash
# Scan all training logs and report SUCCESS/RUNNING/FAILED/MISSING per (model,dataset,method).
#
# Usage:
#   cd /weka/thinng/2026/icdm/mpo
#   bash scripts/scan_results.sh
#
# Output: table of all 336 runs (4 models × 6 datasets × 14 methods)
# with status and accuracy (% where available).
#
# Success criteria: log must contain both
#   "FINAL eval:" line (with rewards_eval/accuracies)  AND  "Done:" line.
# FAIL-ARCH: has FINAL eval but no Done: (archive/post-training step failed).
# FAILED: no FINAL eval (training did not complete).
# MISSING: no log file exists for this combo.
# RUNNING: latest log's job ID is currently in squeue.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS="$PROJECT_ROOT/logs"

MODELS=(qwen05b tinyllama11b qwen3b llama7b)
DATASETS=(hh shp pku ultrabin ultrallama ultragemma)
METHODS=(DPO betaDPO SimPO TDPO TIS CDPO CW MPO-TS MPO-Dual MPO-EMA MPO-LN MPO-Safe MPO-Conf MPO-ConfSafe)

# Snapshot of currently running/pending job IDs
RUNNING_JOBS=$(squeue -u "$(whoami)" -h -o "%i" 2>/dev/null | tr '\n' ' ')

printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" "model" "dataset" "method" "job_id" "status" "accuracy"
printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" "---------------" "------------" "--------------" "--------" "----------" "----------"

ok=0; fail=0; running=0; missing=0
fail_list=""

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
      # Latest log for this (model, dataset, method) — highest numeric job ID
      latest=$(ls "$LOGS"/${model}_${dataset}_${method}_*.out 2>/dev/null | \
               sed 's/.*_\([0-9]*\)\.out/\1 &/' | sort -n | tail -1 | awk '{print $2}' || true)

      if [ -z "$latest" ]; then
        printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
               "$model" "$dataset" "$method" "-" "MISSING" "-"
        ((missing++)) || true
        continue
      fi

      jid=$(basename "$latest" .out | sed 's/.*_//')

      # Currently in the scheduler?
      if echo "$RUNNING_JOBS" | grep -qw "$jid"; then
        printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
               "$model" "$dataset" "$method" "$jid" "RUNNING" "-"
        ((running++)) || true
        continue
      fi

      final_line=$(grep "^FINAL eval:" "$latest" 2>/dev/null | tail -1 || true)
      done_line=$(grep  -E "^Done:|^betaDPO done\." "$latest" 2>/dev/null | tail -1 || true)

      if [ -n "$final_line" ] && [ -n "$done_line" ]; then
        acc=$(echo "$final_line" | grep -oP "(?<='rewards_eval/accuracies': ')[0-9.]+" || true)
        acc_pct=$(printf "%.2f%%" "$(echo "$acc * 100" | bc -l 2>/dev/null || echo 0)")
        printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
               "$model" "$dataset" "$method" "$jid" "SUCCESS" "$acc_pct"
        ((ok++)) || true
      elif [ -n "$final_line" ]; then
        printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
               "$model" "$dataset" "$method" "$jid" "FAIL-ARCH" "-"
        ((fail++)) || true
        fail_list+="  $model/$dataset/$method  [FAIL-ARCH, jid=$jid]\n"
      else
        printf "%-15s %-12s %-14s %-8s %-10s %-10s\n" \
               "$model" "$dataset" "$method" "$jid" "FAILED" "-"
        ((fail++)) || true
        fail_list+="  $model/$dataset/$method  [FAILED, jid=$jid]\n"
      fi
    done
  done
done

echo ""
echo "=== SUMMARY ==="
printf "  SUCCESS : %d / 336\n" "$ok"
printf "  RUNNING : %d\n"       "$running"
printf "  FAILED  : %d\n"       "$fail"
printf "  MISSING : %d\n"       "$missing"

if [ -n "$fail_list" ]; then
  echo ""
  echo "=== FAILED / FAIL-ARCH (may need resubmission) ==="
  echo -e "$fail_list"
fi
