# Reproducibility Gaps

This document records any methods or results from the paper that cannot be fully
reproduced from this codebase, along with the reason.

---

## Win-rate evaluation

**Status: not included in this release.**

The paper reports win-rate results evaluated with an LLM judge (e.g. GPT-4 or a reward
model). The win-rate evaluation pipeline was excluded from this release because it
depends on an external judge API, and the specific prompts/judge configuration are
being finalised. Win-rate code will be added in a follow-up release.

Reward accuracy and reward margin evaluation (the primary metrics in Tables 1–3) are
fully reproducible from this codebase via the standard evaluation loop in `train.py`.

---

## LLaMA-7B (llama7b) results

**Status: code included; large model may require >128 GB RAM.**

The llama7b model config (`config/model/llama7b.yaml`) is included and the model is
supported in `run_all.sh`. However, LLaMA-7B requires FSDP or a >40 GB GPU for full
training. The paper results for llama7b were obtained on a single H100 (80 GB) with
`BasicTrainer`. Users with smaller GPUs should switch to `FSDPTrainer`.

---

## betaDPO – alpha tuning

**Status: fixed hyperparameters used.**

The paper's betaDPO results were obtained with `mode_weight=0.2` and `a=0.6` (the
values in `betaDPO/config/loss/dpo.yaml`). These were tuned on a held-out split.
To reproduce exactly, use these fixed values (already set as defaults).

---

## No known missing methods

All methods reported in the paper are implemented and runnable:

| Paper name | Code name | run_all.sh method |
|-----------|-----------|-------------------|
| DPO | DPO | `DPO` |
| betaDPO | betaDPO | `betaDPO` |
| SimPO | SimPO | `SimPO` |
| TDPO | TDPO | `TDPO` |
| TIS-DPO | TIS | `TIS` |
| CDPO | CDPO | `CDPO` |
| CausalWalk / CW | CW | `CW` |
| MPO-TS | MPO-TS | `MPO-TS` |
| MPO-Dual | MPO-Dual | `MPO-Dual` |
| MPO-EMA | MPO-EMA | `MPO-EMA` |
| MPO-LN | MPO-LN | `MPO-LN` |
| MPO-Safe | MPO-Safe | `MPO-Safe` |
| MPO-Conf | MPO-Conf | `MPO-Conf` |
| MPO-ConfSafe | MPO-ConfSafe | `MPO-ConfSafe` |
