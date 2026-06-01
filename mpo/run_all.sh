#!/bin/bash
set -e

# ===============================
# Environment
# ===============================
source /raid/nhdang01/miniconda3/etc/profile.d/conda.sh
conda activate dpo

export PYTHONNOUSERSITE=1
export WANDB_API_KEY=8f17474bb5e6fbb39a20e2e78dac373f97f339e6

# ---------------------------------------------------------------------------
# MPO – run_all.sh
#
# Usage:
#   ./run_all.sh <model_idx> <dataset_idx> <method> <run_sft>
#
# model_idx:
#   1 = qwen05b    (Qwen2.5-0.5B)
#   2 = tinyllama11b  (TinyLlama-1.1B)
#   3 = qwen3b     (Qwen2.5-3B)
#   4 = llama7b    (LLaMA-7B)
#
# dataset_idx:
#   1 = hh         (Anthropic HH-RLHF)
#   2 = shp        (Stanford Human Preferences)
#   3 = pku        (PKU-SafeRLHF)
#   4 = ultrabin   (UltraFeedback binarized)
#   5 = ultrallama (UltraFeedback Llama-scored)
#   6 = ultragemma (UltraFeedback Gemma-scored)
#
# method:
#   Baselines : DPO betaDPO SimPO TDPO TIS CDPO CW
#   Proposed  : MPO-TS MPO-Dual MPO-EMA MPO-LN MPO-Safe MPO-Conf MPO-ConfSafe
#
# run_sft:
#   1 = run SFT first, then exit (run once per model×dataset)
#   0 = skip SFT; load the existing SFT checkpoint and run the preference method
#
# Examples:
#   # Step 1 – SFT (once per model/dataset pair):
#   sbatch --job-name=sft11 run_all.sh 1 1 DPO 1
#
#   # Step 2 – preference-optimisation methods:
#   sbatch --job-name=dpo11  run_all.sh 1 1 DPO       0
#   sbatch --job-name=mpo11  run_all.sh 1 1 MPO-Dual  0
# ---------------------------------------------------------------------------

# SLURM copies scripts to its spool dir, so BASH_SOURCE[0] points there, not here.
# SLURM_SUBMIT_DIR is set to the directory where sbatch was called (always PROJECT_ROOT).
# if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
#     PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
# else
#     PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# fi
# cd "$PROJECT_ROOT"

# mkdir -p "$PROJECT_ROOT/runs" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "Usage: $0 <model_idx> <dataset_idx> <method> <run_sft> [batch_size]"
    echo ""
    echo "  model_idx  : 1=qwen05b  2=tinyllama11b  3=qwen3b  4=llama7b  5=llama70b"
    echo "  dataset_idx: 1=hh  2=shp  3=pku  4=ultrabin  5=ultrallama  6=ultragemma"
    echo "  method     : DPO betaDPO SimPO TDPO TIS CDPO CW"
    echo "               MPO-TS MPO-Dual MPO-EMA MPO-LN MPO-Safe MPO-Conf MPO-ConfSafe"
    echo "               MPO-NoProj MPO-LengthDiag MPO-LengthOnly MPO-NormOnly"
    echo "  run_sft    : 1=run SFT first  0=use existing SFT checkpoint"
    echo "  batch_size : optional override (default: 8 for M1-M3, 4 for M4, 16 for M5)"
    exit 1
fi

model_name_idx=$1
dataset_idx=$2
method=$3
run_sft=$4

# ---------------------------------------------------------------------------
# Model mapping
# ---------------------------------------------------------------------------
model_names=("qwen05b" "tinyllama11b" "qwen3b" "llama7b" "llama70b")
if ((model_name_idx < 1 || model_name_idx > 5)); then
    echo "❌ Invalid model_idx: $model_name_idx (must be 1-5)"
    exit 1
fi
model_name="${model_names[$((model_name_idx - 1))]}"

# ---------------------------------------------------------------------------
# Dataset mapping
# ---------------------------------------------------------------------------
datasets=("hh" "shp" "pku" "ultrabin" "ultrallama" "ultragemma")
if ((dataset_idx < 1 || dataset_idx > 6)); then
    echo "❌ Invalid dataset_idx: $dataset_idx (must be 1-6)"
    exit 1
fi
dataset="${datasets[$((dataset_idx - 1))]}"

# ---------------------------------------------------------------------------
# Method → internal configuration
# ---------------------------------------------------------------------------
# fd_dpo_alpha defaults to 0.10 when alpha_code=0 (the internal default).
# Frontdoor methods (variants 1-22) use frontdoor.alpha; those are not
# exposed in the public interface.

variant=""
variant_name=""
frontdoor_mode="frontdoor.enabled=false"
frontdoor_layer=""
frontdoor_alpha=""
frontdoor_runtime_flags=""
fd_dpo_mode="fd_dpo.enabled=false"
fd_dpo_layer=""
fd_dpo_alpha=""
fd_dpo_runtime_flags=""
baseline_loss_flags=""
is_betadpo=0

# Fixed hyperparameters (tuned per paper)
fd_dpo_alpha_value="0.10"
fd_dpo_walk_tau_value="${FD_DPO_WALK_TAU:-0.25}"
fd_dpo_fabe_clip_value="${FD_DPO_FABE_CLIP:-0.05}"
fd_dpo_confidence_k_value="${FD_DPO_CONFIDENCE_K:-1.0}"
fd_dpo_fabe_ema_momentum_value="${FD_DPO_FABE_EMA_MOMENTUM:-0.99}"
fd_dpo_fabe_token_temperature_value="${FD_DPO_FABE_TOKEN_TEMPERATURE:-1.0}"
simpo_beta_value="${SIMPO_BETA:-2.0}"
simpo_gamma_value="${SIMPO_GAMMA:-1.0}"
tdpo_alpha_value="${TDPO_ALPHA:-0.5}"
tdpo2_value="${TDPO2:-true}"
tisdpo_alpha_value="${TISDPO_ALPHA:-0.5}"
tisdpo_token_level_value="${TISDPO_TOKEN_LEVEL:-true}"
tisdpo_weight_mode_value="${TISDPO_WEIGHT_MODE:-margin_proxy}"
cdpo_confounder_proxy_value="${CDPO_CONFOUNDER_PROXY:-length_bin}"
cdpo_num_bins_value="${CDPO_NUM_BINS:-5}"
cdpo_weight_clip_value="${CDPO_WEIGHT_CLIP:-5.0}"
cdpo_normalize_weights_value="${CDPO_NORMALIZE_WEIGHTS:-true}"

# Layer selection: proportional to model depth for hidden-state mediator.
# qwen05b(24L)/tinyllama11b(22L) → layer 20 (~85%)
# qwen3b(36L)/llama7b(32L)      → layer 31 (~86-97%)
# llama70b(80L)                  → layer 60 (~75%)
case "$model_name" in
    qwen05b|tinyllama11b) fd_dpo_default_layer=20 ;;
    qwen3b|llama7b)       fd_dpo_default_layer=31 ;;
    llama70b)             fd_dpo_default_layer=60 ;;
esac

case "$method" in
    DPO)
        variant=9
        variant_name="DPO"
        ;;
    betaDPO)
        variant=1
        variant_name="betaDPO"
        is_betadpo=1
        ;;
    SimPO)
        variant=60
        variant_name="SimPO"
        baseline_loss_flags="loss.objective=simpo loss.beta=${simpo_beta_value} loss.gamma=${simpo_gamma_value}"
        ;;
    TDPO)
        variant=61
        variant_name="TDPO"
        baseline_loss_flags="loss.objective=tdpo loss.beta=0.1 loss.tdpo_alpha=${tdpo_alpha_value} loss.tdpo2=${tdpo2_value}"
        ;;
    TIS)
        variant=62
        variant_name="TIS"
        baseline_loss_flags="loss.objective=tisdpo loss.beta=0.1 loss.tisdpo_alpha=${tisdpo_alpha_value} loss.tisdpo_token_level=${tisdpo_token_level_value} loss.tisdpo_weight_mode=${tisdpo_weight_mode_value}"
        ;;
    CDPO)
        variant=65
        variant_name="CDPO"
        baseline_loss_flags="loss.objective=cdpo loss.beta=0.1 cdpo.enabled=true cdpo.backdoor_weight=true cdpo.confounder_proxy=${cdpo_confounder_proxy_value} cdpo.num_bins=${cdpo_num_bins_value} cdpo.weight_clip=${cdpo_weight_clip_value} cdpo.normalize_weights=${cdpo_normalize_weights_value}"
        ;;
    CW)
        variant=29
        variant_name="CW"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_layer="fd_dpo.layer=${fd_dpo_default_layer}"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=response_mean fd_dpo.contrast=causal_walk fd_dpo.detach_mediator=true fd_dpo.walk_tau=${fd_dpo_walk_tau_value}"
        ;;
    MPO-EMA)
        variant=35
        variant_name="MPO-EMA"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_layer="fd_dpo.layer=${fd_dpo_default_layer}"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=response_mean fd_dpo.contrast=fabe fd_dpo.detach_mediator=true fd_dpo.fabe_variant=ema fd_dpo.fabe_variant_id=35 fd_dpo.fabe_ema_momentum=${fd_dpo_fabe_ema_momentum_value}"
        ;;
    MPO-TS)
        variant=36
        variant_name="MPO-TS"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_layer="fd_dpo.layer=${fd_dpo_default_layer}"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=token_selective fd_dpo.contrast=fabe fd_dpo.detach_mediator=true fd_dpo.fabe_variant=token_selective fd_dpo.fabe_variant_id=36 fd_dpo.fabe_token_temperature=${fd_dpo_fabe_token_temperature_value}"
        ;;
    MPO-Dual)
        variant=37
        variant_name="MPO-Dual"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=token_selective fd_dpo.contrast=fabe fd_dpo.detach_mediator=true fd_dpo.fabe_variant=dual_layer fd_dpo.fabe_variant_id=37 fd_dpo.fabe_token_temperature=${fd_dpo_fabe_token_temperature_value} fd_dpo.fabe_dual_lower_weight=0.3 fd_dpo.fabe_dual_upper_weight=0.7"
        case "$model_name" in
            qwen05b|tinyllama11b)
                fd_dpo_layer="fd_dpo.low_layer=12 fd_dpo.layer=20"
                ;;
            qwen3b|llama7b)
                fd_dpo_layer="fd_dpo.low_layer=20 fd_dpo.layer=31"
                ;;
            llama70b)
                fd_dpo_layer="fd_dpo.low_layer=40 fd_dpo.layer=60"
                ;;
        esac
        ;;
    MPO-LN)
        variant=34
        variant_name="MPO-LN"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_layer="fd_dpo.layer=${fd_dpo_default_layer}"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=response_mean fd_dpo.contrast=fabe fd_dpo.detach_mediator=true fd_dpo.fabe_layernorm=true"
        ;;
    MPO-Safe)
        variant=32
        variant_name="MPO-Safe"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_layer="fd_dpo.layer=${fd_dpo_default_layer}"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=response_mean fd_dpo.contrast=fabe fd_dpo.detach_mediator=true fd_dpo.fabe_correction_clip=${fd_dpo_fabe_clip_value}"
        ;;
    MPO-Conf)
        variant=31
        variant_name="MPO-Conf"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_layer="fd_dpo.layer=${fd_dpo_default_layer}"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=response_mean fd_dpo.contrast=fabe fd_dpo.detach_mediator=true fd_dpo.fabe_confidence_weight=true fd_dpo.fabe_confidence_k=${fd_dpo_confidence_k_value}"
        ;;
    MPO-ConfSafe)
        variant=33
        variant_name="MPO-ConfSafe"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_layer="fd_dpo.layer=${fd_dpo_default_layer}"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=response_mean fd_dpo.contrast=fabe fd_dpo.detach_mediator=true fd_dpo.fabe_confidence_weight=true fd_dpo.fabe_confidence_k=${fd_dpo_confidence_k_value} fd_dpo.fabe_correction_clip=${fd_dpo_fabe_clip_value}"
        ;;
    MPO-NoProj)
        # Experiment 1 – Artifact-projection ablation.
        # Uses raw cosine similarity on unprojected (mean-centred) mediator
        # representations as the contrast score φ. Trains with DPO+φ_noproj loss
        # so we can compare final-checkpoint metrics vs MPO-EMA and plain DPO.
        # The companion full-FABE φ is also computed and logged each step
        # (keys: noproj_phi_mean, noproj_fabe_ref_mean, noproj_vs_fabe_corr).
        variant=38
        variant_name="MPO-NoProj"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_layer="fd_dpo.layer=${fd_dpo_default_layer}"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=response_mean fd_dpo.contrast=fabe fd_dpo.detach_mediator=true fd_dpo.fabe_variant=noproj fd_dpo.fabe_variant_id=38"
        ;;
    MPO-LengthDiag)
        # Experiment 2 – Length-bias diagnostic.
        # Trains identically to MPO-EMA (full FABE with projection) but logs rich
        # length-bias statistics every step:
        #   length_diag_phi_fabe_mean, length_diag_phi_noproj_mean,
        #   length_diag_corr_phi_fabe_len, length_diag_corr_phi_noproj_len,
        #   length_diag_corr_dpo_margin_len, length_diag_fabe_len_bias_reduction
        # These scalars let you verify that projection substantially reduces
        # length correlation while preserving useful preference information.
        variant=39
        variant_name="MPO-LengthDiag"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_layer="fd_dpo.layer=${fd_dpo_default_layer}"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=response_mean fd_dpo.contrast=fabe fd_dpo.detach_mediator=true fd_dpo.fabe_variant=length_diag fd_dpo.fabe_variant_id=39 fd_dpo.fabe_ema_momentum=${fd_dpo_fabe_ema_momentum_value}"
        ;;
    MPO-LengthOnly)
        # Experiment 3 – Artifact-component ablation: length-only artifact direction.
        # Uses artifact_score = standardize(log(length)) only — no norm component.
        # Trains with DPO+φ_length_only loss. Compare against MPO-EMA (full FABE)
        # and MPO-NormOnly to isolate which component drives bias reduction.
        # Logged keys: length_only_phi_mean, length_only_fabe_ref_mean, length_only_vs_fabe_corr.
        variant=40
        variant_name="MPO-LengthOnly"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_layer="fd_dpo.layer=${fd_dpo_default_layer}"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=response_mean fd_dpo.contrast=fabe fd_dpo.detach_mediator=true fd_dpo.fabe_variant=length_only fd_dpo.fabe_variant_id=40"
        ;;
    MPO-NormOnly)
        # Experiment 4 – Artifact-component ablation: norm-only artifact direction.
        # Uses artifact_score = standardize(norm) only — no log-length component.
        # Trains with DPO+φ_norm_only loss. Compare against MPO-EMA (full FABE)
        # and MPO-LengthOnly to isolate which component drives bias reduction.
        # Logged keys: norm_only_phi_mean, norm_only_fabe_ref_mean, norm_only_vs_fabe_corr.
        variant=41
        variant_name="MPO-NormOnly"
        fd_dpo_mode="fd_dpo.enabled=true"
        fd_dpo_layer="fd_dpo.layer=${fd_dpo_default_layer}"
        fd_dpo_alpha="fd_dpo.alpha=${fd_dpo_alpha_value}"
        fd_dpo_runtime_flags="fd_dpo.mode=hidden_state fd_dpo.pool=response_mean fd_dpo.contrast=fabe fd_dpo.detach_mediator=true fd_dpo.fabe_variant=norm_only fd_dpo.fabe_variant_id=41"
        ;;
    *)
        echo "❌ Unknown method: $method"
        echo "   Valid methods: DPO betaDPO SimPO TDPO TIS CDPO CW"
        echo "                  MPO-TS MPO-Dual MPO-EMA MPO-LN MPO-Safe MPO-Conf MPO-ConfSafe"
        echo "                  MPO-NoProj MPO-LengthDiag MPO-LengthOnly MPO-NormOnly"
        exit 1
        ;;
esac

# Non-SFT methods must use run_sft=0
if [ "$run_sft" = "1" ] && [[ "$method" != "DPO" && "$method" != "betaDPO" ]]; then
    echo "❌ run_sft=1 is only used for DPO/betaDPO (they share the SFT step)."
    echo "   For method=$method, always use run_sft=0."
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate model config exists
# ---------------------------------------------------------------------------
# if [ "$is_betadpo" = "1" ]; then
#     TRAIN_DIR="$PROJECT_ROOT/betaDPO"
# else
#     TRAIN_DIR="$PROJECT_ROOT"
# fi
# if [ ! -f "$TRAIN_DIR/config/model/${model_name}.yaml" ]; then
#     echo "❌ Model config not found: $TRAIN_DIR/config/model/${model_name}.yaml"
#     exit 1
# fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
echo "pwd: $(pwd)"
echo "whoami: $(whoami)"
echo "hostname: $(hostname)"


python -c "import hydra, tensor_parallel; print('deps OK')"

export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_dir="logs"
mkdir -p "$log_dir"
job_id=$(date +%Y%m%d_%H%M%S)

log_file="${log_dir}/${model_name}_${dataset}_${variant_name}_${job_id}.out"
err_file="${log_dir}/${model_name}_${dataset}_${variant_name}_${job_id}.err"
exec > >(tee "$log_file") 2> >(tee "$err_file" >&2)
echo "==================================================================="
echo " MPO – method=$method  model=$model_name  dataset=$dataset  run_sft=$run_sft"
echo "==================================================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}  cuda={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

start_ts=$(date +%s)
echo "Start: $(date +'%Y-%m-%d %H:%M:%S')"

# Trainer and training flags — M5 (llama70b) uses QLoRA (4-bit NF4 + LoRA r=64) on
# BasicTrainer. 4-bit base (35GB) + LoRA adapters (~1.6GB) fits on 1 H100 for SFT;
# method training (policy + frozen reference) uses 2 H100s via device_map="balanced".
# All other models use BasicTrainer (single GPU, full precision).
trainer_name="BasicTrainer"
extra_train_flags=""
fsdp_world_size=1
# n_examples_flag="n_examples=null"
# if [[ "$model_name" == "llama70b" ]]; then
#     trainer_name="BasicTrainer"
#     extra_train_flags="model.use_qlora=true"
#     fsdp_world_size=1
#     n_examples_flag="n_examples=20000"
# fi

# Model-specific batch size defaults; 5th arg overrides.
# llama7b: bs=4 (H200 OOMs at bs=8 on long-sequence datasets)
# llama70b: FSDPTrainer training loop double-slices:
#   local_microbatch = batch / grad_accum / world_size  → must be ≥ 1
#   batch_size = grad_accum × world_size × per_rank_bs
#   With per_rank_bs=1, world_size=4, effective_batch=16: grad_accum=4, batch_size=16
case "$model_name" in
    llama7b)  default_bs=4 ;;
    llama70b) default_bs=16 ;;
    *)        default_bs=8 ;;
esac
batch_size="${5:-$default_bs}"

if [[ "$model_name" == "llama70b" ]]; then
    grad_accum=16  # batch_size(16) / grad_accum(16) / world_size(1) = 1 example per step
else
    grad_accum=$(( 16 / batch_size ))
fi

# eval also slices by world_size: eval_batch_size / world_size must be ≥ 1
# llama70b: training micro-batch=1 (batch 16 / grad_accum 16); eval at 16 OOMs when
# TDPO/TIS/CW/MTS compute per-token quantities on top of ~35GB policy + ~35GB reference.
if [[ "$model_name" == "llama70b" ]]; then
    eval_batch_size=1
else
    eval_batch_size=$(( batch_size < fsdp_world_size ? fsdp_world_size : batch_size ))
fi

# ---------------------------------------------------------------------------
# SFT step (run_sft=1)
# ---------------------------------------------------------------------------
if [ "$run_sft" = "1" ]; then
    echo "--- SFT ---"
    # cd "$TRAIN_DIR"
    python -u train.py \
        model="$model_name" \
        datasets=["$dataset"] \
        loss=sft \
        frontdoor.enabled=false \
        exp_name="${dataset}_${model_name}_sft" \
        gradient_accumulation_steps=$grad_accum \
        n_examples=100 \
        batch_size=$batch_size \
        eval_batch_size=$eval_batch_size \
        trainer=$trainer_name \
        $extra_train_flags \
        sample_during_eval=false
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    echo "SFT done. Elapsed: ${elapsed}s"
    exit 0
fi

# ---------------------------------------------------------------------------
# Find the latest SFT checkpoint
# ---------------------------------------------------------------------------
BASE_DIR=".cache/nhdang01"
PREFIX="${dataset}_${model_name}_sft"

latest_suffix=$(find "$BASE_DIR" -maxdepth 1 -type d -name "${PREFIX}*" 2>/dev/null | \
    sed -E "s|.*/${PREFIX}||" | sort | tail -n 1)

if [ -z "$latest_suffix" ]; then
    echo "❌ No SFT checkpoint found for prefix '${PREFIX}' in $BASE_DIR"
    echo "   Run with run_sft=1 first: ./run_all.sh $model_name_idx $dataset_idx $method 1"
    exit 1
fi
ckpt_path="$BASE_DIR/${PREFIX}${latest_suffix}/LATEST/policy.pt"
echo "SFT checkpoint: $ckpt_path"

# ---------------------------------------------------------------------------
# betaDPO – runs from betaDPO/ subdirectory with its own train.py
# ---------------------------------------------------------------------------
if [ "$is_betadpo" = "1" ]; then
    echo "--- betaDPO ---"
    # cd "$PROJECT_ROOT/betaDPO"
    python -u train.py \
        model="$model_name" \
        datasets=["$dataset"] \
        loss=dpo \
        loss.beta=0.1 \
        loss.mode_loss=beta_DPO \
        loss.mode_weight=0.2 \
        loss.a=0.6 \
        exp_name="${dataset}_${model_name}_betaDPO" \
        gradient_accumulation_steps=$grad_accum \
        batch_size=$batch_size \
        eval_batch_size=$eval_batch_size \
        trainer=$trainer_name \
        $extra_train_flags \
        sample_during_eval=false \
        model.archive="$ckpt_path"
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    echo "betaDPO done. Elapsed: ${elapsed}s"
    echo "Done: $method on $dataset with $model_name. Elapsed: ${elapsed}s ($(($elapsed/3600))h $((($elapsed%3600)/60))m $(($elapsed%60))s)"
    exit 0
fi

# ---------------------------------------------------------------------------
# All other methods – run from project root
# ---------------------------------------------------------------------------
echo "--- $method ---"
# cd "$PROJECT_ROOT"
python -u train.py \
    model="$model_name" \
    datasets=["$dataset"] \
    loss=dpo \
    loss.beta=0.1 \
    $baseline_loss_flags \
    $frontdoor_mode \
    $frontdoor_layer \
    $frontdoor_alpha \
    $frontdoor_runtime_flags \
    $fd_dpo_mode \
    $fd_dpo_layer \
    $fd_dpo_alpha \
    $fd_dpo_runtime_flags \
    n_epochs=1 \
    n_eval_examples=256 \
    do_first_eval=true \
    eval_every=20000 \
    n_examples=100 \
    exp_name="${dataset}_${model_name}_${variant_name}" \
    gradient_accumulation_steps=$grad_accum \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    trainer=$trainer_name \
    $extra_train_flags \
    sample_during_eval=false \
    model.archive="$ckpt_path"

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
echo "Done: $method on $dataset with $model_name. Elapsed: ${elapsed}s ($(($elapsed/3600))h $((($elapsed%3600)/60))m $(($elapsed%60))s)"
