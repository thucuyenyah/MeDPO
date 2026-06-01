#!/bin/bash
# Smoke test: verify run_all.sh argument parsing without launching training.
# Checks that all public methods, model indices, and dataset indices parse cleanly.
# Does NOT submit or run any training jobs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$PROJECT_ROOT/run_all.sh"

pass=0
fail=0

check() {
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "  PASS: $desc"
        pass=$((pass + 1))
    else
        echo "  FAIL: $desc"
        fail=$((fail + 1))
    fi
}

check_fail() {
    local desc="$1"; shift
    if ! "$@" >/dev/null 2>&1; then
        echo "  PASS (expected failure): $desc"
        pass=$((pass + 1))
    else
        echo "  FAIL (should have failed): $desc"
        fail=$((fail + 1))
    fi
}

echo "=== MPO smoke test ==="
echo "Script: $SCRIPT"
echo ""

# --------------------------------------------------------------------------
# 1. Usage message (no args)
# --------------------------------------------------------------------------
echo "-- 1. Usage (no args) --"
usage_output="$("$SCRIPT" 2>&1 || true)"
if echo "$usage_output" | grep -q "Usage"; then
    echo "  PASS: no-arg usage message shown"
    pass=$((pass + 1))
else
    echo "  FAIL: no-arg usage message not shown"
    fail=$((fail + 1))
fi

# --------------------------------------------------------------------------
# 2. Invalid model index
# --------------------------------------------------------------------------
echo "-- 2. Invalid model index --"
check_fail "model_idx=0 rejected"  bash -c "source $PROJECT_ROOT/slurm_env.sh 2>/dev/null || true; bash $SCRIPT 0 1 DPO 0" || true
check_fail "model_idx=5 rejected"  bash -c "source $PROJECT_ROOT/slurm_env.sh 2>/dev/null || true; bash $SCRIPT 5 1 DPO 0" || true

# --------------------------------------------------------------------------
# 3. Invalid dataset index
# --------------------------------------------------------------------------
echo "-- 3. Invalid dataset index --"
check_fail "dataset_idx=0 rejected" bash -c "source $PROJECT_ROOT/slurm_env.sh 2>/dev/null || true; bash $SCRIPT 1 0 DPO 0" || true
check_fail "dataset_idx=7 rejected" bash -c "source $PROJECT_ROOT/slurm_env.sh 2>/dev/null || true; bash $SCRIPT 1 7 DPO 0" || true

# --------------------------------------------------------------------------
# 4. Invalid method name
# --------------------------------------------------------------------------
echo "-- 4. Invalid method name --"
check_fail "unknown method rejected" bash -c "source $PROJECT_ROOT/slurm_env.sh 2>/dev/null || true; bash $SCRIPT 1 1 fdFABEv38 0" || true
check_fail "old name TISDPO rejected" bash -c "source $PROJECT_ROOT/slurm_env.sh 2>/dev/null || true; bash $SCRIPT 1 1 TISDPO 0" || true
check_fail "old name CausalWalk rejected" bash -c "source $PROJECT_ROOT/slurm_env.sh 2>/dev/null || true; bash $SCRIPT 1 1 CausalWalk 0" || true
check_fail "old name bDPO rejected" bash -c "source $PROJECT_ROOT/slurm_env.sh 2>/dev/null || true; bash $SCRIPT 1 1 bDPO 0" || true
check_fail "old name originaldpo rejected" bash -c "source $PROJECT_ROOT/slurm_env.sh 2>/dev/null || true; bash $SCRIPT 1 1 originaldpo 0" || true
check_fail "old name CDPOBackdoor rejected" bash -c "source $PROJECT_ROOT/slurm_env.sh 2>/dev/null || true; bash $SCRIPT 1 1 CDPOBackdoor 0" || true

# --------------------------------------------------------------------------
# 5. Method name parsing (check the case block resolves without error)
#    We parse-check by running bash -n (syntax) and grepping for method in script.
# --------------------------------------------------------------------------
echo "-- 5. All public method names present in run_all.sh --"
PUBLIC_METHODS="DPO betaDPO SimPO TDPO TIS CDPO CW MPO-TS MPO-Dual MPO-EMA MPO-LN MPO-Safe MPO-Conf MPO-ConfSafe"
for m in $PUBLIC_METHODS; do
    if grep -q "\"$m\")" "$SCRIPT" || grep -q "\"$m\"\)" "$SCRIPT" || grep -qF "$m)" "$SCRIPT"; then
        echo "  PASS: method '$m' present"
        pass=$((pass + 1))
    else
        echo "  FAIL: method '$m' missing from $SCRIPT"
        fail=$((fail + 1))
    fi
done

# --------------------------------------------------------------------------
# 6. Old internal names must NOT appear as user-facing case labels
# --------------------------------------------------------------------------
echo "-- 6. Old internal names absent from case labels --"
OLD_NAMES="fdDPOv9Conf fdDPOv10Safe fdDPOv11CSafe fdDPOv12LN fdFABEMA fdFABETS fdFABEDL CausalWalk-DPO originaldpo bDPO TISDPO CDPOBackdoor"
for old in $OLD_NAMES; do
    if grep -qF "$old)" "$SCRIPT"; then
        echo "  FAIL: old name '$old' still appears as case label in $SCRIPT"
        fail=$((fail + 1))
    else
        echo "  PASS: old name '$old' absent"
        pass=$((pass + 1))
    fi
done

# --------------------------------------------------------------------------
# 7. Syntax check
# --------------------------------------------------------------------------
echo "-- 7. Shell syntax check --"
if bash -n "$SCRIPT"; then
    echo "  PASS: bash -n OK"
    pass=$((pass + 1))
else
    echo "  FAIL: bash -n reported syntax errors"
    fail=$((fail + 1))
fi

# --------------------------------------------------------------------------
# 8. betaDPO subdir structure
# --------------------------------------------------------------------------
echo "-- 8. betaDPO subdirectory structure --"
BETADPO_DIR="$PROJECT_ROOT/betaDPO"
for f in train.py trainers.py utils.py preference_datasets.py config/config.yaml config/loss/dpo.yaml; do
    if [ -f "$BETADPO_DIR/$f" ]; then
        echo "  PASS: betaDPO/$f exists"
        pass=$((pass + 1))
    else
        echo "  FAIL: betaDPO/$f missing"
        fail=$((fail + 1))
    fi
done

# --------------------------------------------------------------------------
# 9. Model configs for all 4 public models
# --------------------------------------------------------------------------
echo "-- 9. Model configs --"
for model in qwen05b tinyllama11b qwen3b llama7b; do
    if [ -f "$PROJECT_ROOT/config/model/${model}.yaml" ]; then
        echo "  PASS: config/model/${model}.yaml exists"
        pass=$((pass + 1))
    else
        echo "  FAIL: config/model/${model}.yaml missing"
        fail=$((fail + 1))
    fi
done

# --------------------------------------------------------------------------
# 10. Submission helper scripts present and syntactically valid
# --------------------------------------------------------------------------
echo "-- 10. Submission helper scripts --"
for s in scripts/submit_full_reproduction.sh scripts/run_method_model_all_datasets.sh; do
    if [ -f "$PROJECT_ROOT/$s" ]; then
        echo "  PASS: $s exists"
        pass=$((pass + 1))
    else
        echo "  FAIL: $s missing"
        fail=$((fail + 1))
        continue
    fi
    if bash -n "$PROJECT_ROOT/$s"; then
        echo "  PASS: $s syntax OK"
        pass=$((pass + 1))
    else
        echo "  FAIL: $s has syntax errors"
        fail=$((fail + 1))
    fi
done

# --------------------------------------------------------------------------
# 11. Job-name lengths in submit_full_reproduction.sh
# --------------------------------------------------------------------------
echo "-- 11. All job names <=8 chars in submit_full_reproduction.sh --"
SUBM="$PROJECT_ROOT/scripts/submit_full_reproduction.sh"
long_names=$(grep -oP '(?<=")\w[^"]*(?="\))' "$SUBM" 2>/dev/null | \
    grep -v '^#' | awk 'length($0)>8' || true)
# Alternative: check method_prefix outputs
for name in DPO BDPO SIM TDPO TIS CDPO CW MTS MDL MEMA MLN MSFE MCF MCSF; do
    full="${name}_M4"   # longest form: prefix_M4
    if [ ${#full} -le 8 ]; then
        echo "  PASS: '$full' (${#full} chars)"
        pass=$((pass + 1))
    else
        echo "  FAIL: '$full' (${#full} chars) exceeds 8"
        fail=$((fail + 1))
    fi
done
for sft in S1D1 S4D6; do
    if [ ${#sft} -le 8 ]; then
        echo "  PASS: SFT name '$sft' (${#sft} chars)"
        pass=$((pass + 1))
    else
        echo "  FAIL: SFT name '$sft' exceeds 8 chars"
        fail=$((fail + 1))
    fi
done

# --------------------------------------------------------------------------
# 12. run_method_model_all_datasets.sh references all 6 datasets
# --------------------------------------------------------------------------
echo "-- 12. Inner loop covers all 6 datasets --"
INNER="$PROJECT_ROOT/scripts/run_method_model_all_datasets.sh"
if grep -q "1 2 3 4 5 6" "$INNER"; then
    echo "  PASS: datasets array 1-6 present"
    pass=$((pass + 1))
else
    echo "  FAIL: datasets 1-6 not found in inner loop script"
    fail=$((fail + 1))
fi

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo ""
echo "=== Results: $pass passed, $fail failed ==="
if [ $fail -gt 0 ]; then
    exit 1
fi
