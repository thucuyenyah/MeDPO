# MPO: Preference Optimisation via Hidden-State Mediation

Code for the paper **"Preference Optimisation via Hidden-State Mediation"** (ICDM 2026).

MPO corrects for hidden-state confounding in pairwise preference data by extracting a
mediator signal from intermediate transformer layers and using it to adjust the DPO
margin at training time.

---

## Supported methods

| Name | Type | Description |
|------|------|-------------|
| DPO | Baseline | Standard Direct Preference Optimisation |
| betaDPO | Baseline | Beta-DPO with adaptive beta scheduling |
| SimPO | Baseline | Reference-free average-log-prob reward |
| TDPO | Baseline | Token-level KL-corrected DPO |
| TIS | Baseline | Token-importance-sampled DPO |
| CDPO | Baseline | Causal DPO with proxy confounder reweighting |
| CW | Baseline | CausalWalk hidden-state path scoring |
| MPO-TS | Proposed | Token-selective hidden-state mediation |
| MPO-Dual | Proposed | Dual-layer token-selective mediation |
| MPO-EMA | Proposed (ablation) | EMA-smoothed mediator signal |
| MPO-LN | Proposed (ablation) | Layer-normalised mediator |
| MPO-Safe | Proposed (ablation) | Clipped mediator correction |
| MPO-Conf | Proposed (ablation) | Confidence-weighted mediator |
| MPO-ConfSafe | Proposed (ablation) | Confidence-weighted + clipped mediator |

---

## Supported models

| idx | Model |
|-----|-------|
| 1 | `Qwen/Qwen2.5-0.5B` (qwen05b) |
| 2 | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (tinyllama11b) |
| 3 | `Qwen/Qwen2.5-3B` (qwen3b) |
| 4 | `huggyllama/llama-7b` (llama7b) |

## Supported datasets

| idx | Dataset |
|-----|---------|
| 1 | `hh` — Anthropic HH-RLHF |
| 2 | `shp` — Stanford Human Preferences |
| 3 | `pku` — PKU-SafeRLHF |
| 4 | `ultrabin` — UltraFeedback (binarized) |
| 5 | `ultrallama` — UltraFeedback (Llama-scored) |
| 6 | `ultragemma` — UltraFeedback (Gemma-scored) |

---

## Hardware requirements

| GPU | Models | Partition | Notes |
|-----|--------|-----------|-------|
| H200 | all 4 | `gpu-large` | Preferred — largest HBM3e capacity |
| H100 | all 4 | `gpu-large` | Fully supported |
| A100 | qwen05b, tinyllama11b only | `gpu` | qwen3b and llama7b risk OOM |

H200 is preferred when available. A100 can be used for the two smallest models
when H100/H200 nodes are fully occupied.

The submission script selects the GPU automatically and restricts model scope for A100:

```bash
bash scripts/submit_full_reproduction.sh           # auto: H200 if available, else H100
bash scripts/submit_full_reproduction.sh h200      # force H200, all 4 models
bash scripts/submit_full_reproduction.sh h100      # force H100, all 4 models
bash scripts/submit_full_reproduction.sh a100      # A100, qwen05b + tinyllama11b only
bash scripts/submit_full_reproduction.sh --dry-run # preview commands, no submission
```

When only A100 is available, reproduce in two waves:

```bash
bash scripts/submit_full_reproduction.sh a100   # submit models 1-2 now
bash scripts/submit_full_reproduction.sh h100   # submit models 3-4 when H100 frees up
```

---

## Environment setup

```bash
conda create -n mpo python=3.11
conda activate mpo
pip install -r requirements.txt
```

Key dependencies: `torch==2.0.1`, `transformers==4.29.2`, `datasets==2.20.0`,
`hydra-core==1.3.2`, `tensor-parallel==1.2.4`, `wandb==0.15.3`.

Set `WANDB_API_KEY` in your environment (or in `slurm_env.sh`) before training.

---

## Running experiments

All experiments use a two-step workflow:

### Step 1: SFT (once per model × dataset)

```bash
# Local (single GPU)
./run_all.sh <model_idx> <dataset_idx> DPO 1

# SLURM (H100)
sbatch --partition=gpu-large --gpus=h100:1 --job-name=sft11 run_all.sh 1 1 DPO 1

# SLURM (H200)
sbatch --partition=gpu-large --gpus=h200:1 --job-name=sft11 run_all.sh 1 1 DPO 1
```

### Step 2: Preference optimisation

```bash
# Local
./run_all.sh <model_idx> <dataset_idx> <method> 0

# SLURM (H100)
sbatch --partition=gpu-large --gpus=h100:1 --job-name=mpo11 run_all.sh 1 1 MPO-Dual 0
```

### Interface

```
./run_all.sh <model_idx> <dataset_idx> <method> <run_sft>

model_idx:   1=qwen05b  2=tinyllama11b  3=qwen3b  4=llama7b
dataset_idx: 1=hh  2=shp  3=pku  4=ultrabin  5=ultrallama  6=ultragemma
method:      DPO betaDPO SimPO TDPO TIS CDPO CW
             MPO-TS MPO-Dual MPO-EMA MPO-LN MPO-Safe MPO-Conf MPO-ConfSafe
run_sft:     1=run SFT first  0=use existing SFT checkpoint
```

---

## Reproducing paper tables

See `commands.txt` for the complete list of `sbatch` commands covering all
4 models × 6 datasets × 14 methods.

**Quick start (qwen05b on HH-RLHF, H100):**

```bash
GPU="--partition=gpu-large --gpus=h100:1"
sbatch $GPU --job-name=sft11   run_all.sh 1 1 DPO      1   # SFT
sbatch $GPU --job-name=dpo11   run_all.sh 1 1 DPO      0
sbatch $GPU --job-name=mpodl11 run_all.sh 1 1 MPO-Dual 0
sbatch $GPU --job-name=mpots11 run_all.sh 1 1 MPO-TS   0
```

---

## Output files

Training results (checkpoints and evaluation logs) are written to:

```
.cache/thinng/<exp_name><timestamp>/
```

Evaluation metrics (reward accuracy, reward margin) are logged to Weights & Biases
and also printed at the end of each job's log file in `logs/`.

---

## Project structure

```
mpo/
  run_all.sh            Main entry point
  commands.txt          Full command list for reproduction
  train.py              Training script (Hydra-based)
  trainers.py           Trainer classes + loss implementations
  utils.py              Utility functions
  preference_datasets.py Dataset loading
  frontdoor.py          Frontdoor mediator capture/steering (legacy)
  fddpo.py              Hidden-state mediation losses (MPO core)
  config/               Hydra configuration
    config.yaml
    loss/
    model/
  betaDPO/              Beta-DPO variant (separate trainer)
    train.py
    trainers.py
    utils.py
    preference_datasets.py
    config/
  scripts/
    smoke_test.sh       Argument-parsing smoke test
  results/              Output directory (gitignored)
  LICENSE
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mpo2026icdm,
  title     = {Preference Optimisation via Hidden-State Mediation},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year      = {2026},
}
```
