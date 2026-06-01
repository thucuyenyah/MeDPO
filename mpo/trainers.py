import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
from fddpo import (
    compute_fd_dpo_contrast,
    compute_fddpo_fabe_score_with_stats,
    compute_fddpo_fabe_noproj_score_with_stats,
    compute_fddpo_fabe_length_only_score_with_stats,
    compute_fddpo_fabe_norm_only_score_with_stats,
    compute_length_bias_diagnostics,
    compute_logprob_distance_contrast,
    fd_dpo_external_score_preference_loss,
    fd_dpo_preference_loss,
    fd_dpo_v2_preference_loss,
    fd_dpo_v3_preference_loss,
    fd_dpo_weighted_clipped_preference_loss,
)
from frontdoor import FrontdoorMediatorCapture, FrontdoorSteering


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def simpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    chosen_lengths: torch.FloatTensor,
    rejected_lengths: torch.FloatTensor,
    beta: float,
    gamma: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    chosen_scores = beta * (policy_chosen_logps / chosen_lengths.clamp_min(1.0))
    rejected_scores = beta * (policy_rejected_logps / rejected_lengths.clamp_min(1.0))
    logits = chosen_scores - rejected_scores - gamma
    losses = -F.logsigmoid(logits)
    return losses, chosen_scores.detach(), rejected_scores.detach()


def tdpo_loss(
    chosen_logps_margin: torch.FloatTensor,
    rejected_logps_margin: torch.FloatTensor,
    chosen_position_kl: torch.FloatTensor,
    rejected_position_kl: torch.FloatTensor,
    beta: float,
    alpha: float = 0.5,
    tdpo2: bool = True,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Token-level DPO loss from the official TDPO decomposition."""
    chosen_values = chosen_logps_margin + chosen_position_kl
    rejected_values = rejected_logps_margin + rejected_position_kl
    logps_margin = chosen_logps_margin - rejected_logps_margin

    if tdpo2:
        logits = logps_margin - float(alpha) * (rejected_position_kl - chosen_position_kl.detach())
    else:
        logits = logps_margin - (rejected_position_kl - chosen_position_kl)

    losses = -F.logsigmoid(float(beta) * logits)
    chosen_rewards = float(beta) * chosen_values.detach()
    rejected_rewards = float(beta) * rejected_values.detach()
    return losses, chosen_rewards, rejected_rewards


def tisdpo_loss(
    chosen_logps_margin: torch.FloatTensor,
    rejected_logps_margin: torch.FloatTensor,
    chosen_position_kl: torch.FloatTensor,
    rejected_position_kl: torch.FloatTensor,
    beta: float,
    alpha: float = 0.5,
    token_level: bool = True,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """TIS-DPO loss over weighted token-level policy/reference margins."""
    if token_level:
        chosen_values = chosen_logps_margin - chosen_position_kl
        rejected_values = rejected_logps_margin - rejected_position_kl
        logits = (
            chosen_logps_margin
            - rejected_logps_margin
            - float(alpha) * (chosen_position_kl - rejected_position_kl)
        )
    else:
        chosen_values = chosen_logps_margin
        rejected_values = rejected_logps_margin
        logits = chosen_logps_margin - rejected_logps_margin

    losses = -F.logsigmoid(float(beta) * logits)
    chosen_rewards = float(beta) * chosen_values.detach()
    rejected_rewards = float(beta) * rejected_values.detach()
    return losses, chosen_rewards, rejected_rewards


def cdpo_proxy_preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    backdoor_weights: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Backdoor-weighted DPO used by the local CDPO proxy implementation."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if reference_free:
        ref_logratios = 0.0

    logits = pi_logratios - ref_logratios
    per_example_losses = (
        -F.logsigmoid(float(beta) * logits) * (1 - label_smoothing)
        - F.logsigmoid(-float(beta) * logits) * label_smoothing
    )
    weights = backdoor_weights.to(device=per_example_losses.device, dtype=per_example_losses.dtype)
    weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    losses = per_example_losses * weights

    chosen_rewards = float(beta) * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = float(beta) * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def compute_cdpo_proxy_weights(
    batch: Dict[str, Union[List, torch.LongTensor]],
    confounder_proxy: str = "length_bin",
    num_bins: int = 5,
    weight_clip: float = 5.0,
    normalize_weights: bool = True,
    backdoor_weight: bool = True,
    max_length: int = 512,
    max_prompt_length: int = 256,
) -> Tuple[torch.FloatTensor, torch.LongTensor, Dict[str, torch.Tensor]]:
    """Estimate inverse-stratum proxy weights for CDPO-proxy-backdoor."""
    if num_bins <= 0:
        raise ValueError("cdpo.num_bins must be positive")

    chosen_labels = batch["chosen_labels"]
    rejected_labels = batch["rejected_labels"]
    device = chosen_labels.device
    dtype = torch.float32
    chosen_lengths = (chosen_labels[:, 1:] != -100).sum(dim=-1).to(device=device, dtype=dtype)
    rejected_lengths = (rejected_labels[:, 1:] != -100).sum(dim=-1).to(device=device, dtype=dtype)
    prompt_attention_mask = batch.get("prompt_attention_mask")
    if prompt_attention_mask is None:
        prompt_lengths = torch.zeros_like(chosen_lengths)
    else:
        prompt_lengths = prompt_attention_mask.sum(dim=-1).to(device=device, dtype=dtype)

    proxy = str(confounder_proxy)
    response_capacity = max(1.0, float(max_length - max_prompt_length))
    if proxy in {"length_bin", "total_length_bin"}:
        values = prompt_lengths + chosen_lengths + rejected_lengths
        max_value = float(max_prompt_length) + 2.0 * response_capacity
    elif proxy == "prompt_length_bin":
        values = prompt_lengths
        max_value = float(max_prompt_length)
    elif proxy == "response_length_bin":
        values = chosen_lengths + rejected_lengths
        max_value = 2.0 * response_capacity
    elif proxy == "chosen_length_bin":
        values = chosen_lengths
        max_value = response_capacity
    elif proxy == "rejected_length_bin":
        values = rejected_lengths
        max_value = response_capacity
    elif proxy == "length_gap_bin":
        values = (chosen_lengths - rejected_lengths).abs()
        max_value = response_capacity
    elif proxy in {"dataset_source", "model_dataset_group"}:
        values = torch.zeros_like(chosen_lengths)
        max_value = 1.0
    else:
        raise ValueError(
            f"unknown cdpo.confounder_proxy: {proxy} "
            "(expected length_bin, total_length_bin, prompt_length_bin, "
            "response_length_bin, chosen_length_bin, rejected_length_bin, "
            "length_gap_bin, dataset_source, or model_dataset_group)"
        )

    scaled = values.clamp_min(0.0) / max(max_value, 1.0)
    bins = torch.floor(scaled * int(num_bins)).to(device=device, dtype=torch.long)
    bins = bins.clamp(min=0, max=int(num_bins) - 1)

    batch_size = max(int(chosen_labels.shape[0]), 1)
    counts = torch.bincount(bins, minlength=int(num_bins)).to(device=device, dtype=dtype)
    observed = counts > 0
    observed_count = observed.sum().to(dtype=dtype).clamp_min(1.0)
    bin_counts = counts.gather(0, bins).clamp_min(1.0)

    if backdoor_weight:
        weights = float(batch_size) / (observed_count * bin_counts)
    else:
        weights = torch.ones_like(values)

    if normalize_weights:
        weights = weights / weights.mean().clamp_min(1e-6)
    if weight_clip is not None and float(weight_clip) > 0:
        weights = weights.clamp(max=float(weight_clip))
    weights = torch.nan_to_num(weights, nan=1.0, posinf=float(weight_clip), neginf=0.0).detach()

    stats = {
        "observed_bins": observed_count.detach(),
        "weight_mean": weights.mean().detach(),
        "weight_std": weights.std(unbiased=False).detach(),
        "weight_min": weights.min().detach(),
        "weight_max": weights.max().detach(),
    }
    return weights, bins.detach(), stats


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def _get_batch_token_logps(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    """Return per-token log-probs aligned to ``labels[:, 1:]`` for response weighting."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"logits shape {logits.shape[:-1]} does not match labels {labels.shape}")

    shifted_labels = labels[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    loss_mask = shifted_labels != -100
    shifted_labels[shifted_labels == -100] = 0
    per_token_logps = torch.gather(
        shifted_logits.log_softmax(-1),
        dim=2,
        index=shifted_labels.unsqueeze(2),
    ).squeeze(2)
    per_token_logps = per_token_logps * loss_mask.to(dtype=per_token_logps.dtype)
    return torch.nan_to_num(per_token_logps, nan=0.0, posinf=0.0, neginf=0.0)


def _get_weighted_token_stats(
    logits: torch.FloatTensor,
    reference_logits: torch.FloatTensor,
    labels: torch.LongTensor,
    weights: Optional[torch.FloatTensor] = None,
    average_log_prob: bool = False,
    kl_direction: str = "ref_to_policy",
    tisdpo_weight_mode: str = "uniform",
    tisdpo_min_weight: float = 0.25,
    tisdpo_max_weight: float = 2.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Return weighted token log-ratio, token KL, policy logp, and mean weight."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"logits shape {logits.shape[:-1]} does not match labels {labels.shape}")
    if reference_logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"reference logits shape {reference_logits.shape[:-1]} does not match labels {labels.shape}"
        )

    shifted_labels = labels[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    shifted_reference_logits = reference_logits[:, :-1, :]
    loss_mask = shifted_labels != -100
    shifted_labels[shifted_labels == -100] = 0

    vocab_logps = shifted_logits.log_softmax(-1)
    reference_vocab_logps = shifted_reference_logits.log_softmax(-1)
    if kl_direction == "ref_to_policy":
        reference_vocab_ps = shifted_reference_logits.softmax(-1)
        per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    elif kl_direction == "policy_to_ref":
        vocab_ps = shifted_logits.softmax(-1)
        per_position_kl = (vocab_ps * (vocab_logps - reference_vocab_logps)).sum(-1)
    else:
        raise ValueError(f"unknown token KL direction: {kl_direction}")

    per_token_logps = torch.gather(vocab_logps, dim=2, index=shifted_labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(
        reference_vocab_logps, dim=2, index=shifted_labels.unsqueeze(2)
    ).squeeze(2)
    logps_margin = per_token_logps - per_reference_token_logps

    if weights is not None:
        if weights.shape != labels.shape:
            raise ValueError(f"token weights shape {weights.shape} does not match labels {labels.shape}")
        token_weights = weights[:, 1:].to(device=logps_margin.device, dtype=logps_margin.dtype)
    elif tisdpo_weight_mode == "uniform":
        token_weights = torch.ones_like(logps_margin)
    elif tisdpo_weight_mode == "margin_proxy":
        source = logps_margin.detach().abs()
        denom = (source * loss_mask).sum(-1, keepdim=True) / loss_mask.sum(-1, keepdim=True).clamp_min(1)
        token_weights = source / denom.clamp_min(1e-6)
        token_weights = token_weights.clamp(float(tisdpo_min_weight), float(tisdpo_max_weight))
    else:
        raise ValueError(
            f"unknown TIS-DPO weight mode: {tisdpo_weight_mode} "
            "(expected uniform or margin_proxy)"
        )

    token_weights = torch.nan_to_num(token_weights, nan=1.0, posinf=float(tisdpo_max_weight), neginf=0.0)
    token_weights = token_weights * loss_mask.to(dtype=token_weights.dtype)
    denom = loss_mask.sum(-1).clamp_min(1)

    logps_margin_sum = (logps_margin * token_weights).sum(-1)
    position_kl_sum = (per_position_kl * token_weights).sum(-1)
    policy_logps_sum = (per_token_logps * token_weights).sum(-1)
    weight_mean = token_weights.sum(-1) / denom

    if average_log_prob:
        logps_margin_sum = logps_margin_sum / denom
        position_kl_sum = position_kl_sum / denom
        policy_logps_sum = policy_logps_sum / denom

    return logps_margin_sum, position_kl_sum, policy_logps_sum, weight_mean.detach()


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model
        self.uses_reference_model = config.loss.name in {'dpo', 'ipo'} and getattr(config.loss, 'objective', 'dpo') != 'simpo'
        self.dynamic_alpha_state = None
        self.runtime_budget_exceeded = False

        train_n_examples = config.n_examples
        eval_n_examples = config.n_eval_examples
        if config.runtime_budget.enabled and not config.runtime_budget.full_data:
            if config.runtime_budget.max_train_samples is not None and train_n_examples is None:
                train_n_examples = int(config.runtime_budget.max_train_samples)
            if config.runtime_budget.max_eval_samples is not None:
                eval_n_examples = min(int(eval_n_examples), int(config.runtime_budget.max_eval_samples))

        if config.frontdoor.enabled:
            print("Frontdoor Steering enabled")
            self.frontdoor = FrontdoorSteering(
                policy,
                layer=config.frontdoor.layer,
                layers=config.frontdoor.layers,
                alpha=config.frontdoor.alpha,
                dynamic_alpha=config.frontdoor.dynamic_alpha,
                apply_layer=config.frontdoor.apply_layer,
                normalize_direction=config.frontdoor.normalize_direction,
                aggregate_mode=config.frontdoor.aggregate_mode,
                confidence_gate_mode=config.frontdoor.confidence_gate_mode,
                confidence_gate_scale=config.frontdoor.confidence_gate_scale,
                token_weight_mode=config.frontdoor.token_weight_mode,
                layer_weights=config.frontdoor.layer_weights,
                debug=config.frontdoor.debug,
                optimization_mode=config.frontdoor.optimization_mode,
                detach_statistics=config.frontdoor.detach_statistics,
                token_saliency_dim_stride=config.frontdoor.token_saliency_dim_stride,
                token_topk_fraction=config.frontdoor.token_topk_fraction,
                sparse_layer_count=config.frontdoor.sparse_layer_count,
                sparse_layer_schedule_bins=config.frontdoor.sparse_layer_schedule_bins,
            )
        else:
            self.frontdoor = None

        self.fd_dpo_config = getattr(config, 'fd_dpo', None)
        self.fd_dpo_enabled = bool(
            self.fd_dpo_config is not None and getattr(self.fd_dpo_config, 'enabled', False)
        )
        self.fd_dpo_mode = None
        self.fd_dpo_gate_mode = "none"
        self.fd_dpo_capture = None
        self.fd_dpo_capture_low = None
        self.fd_dpo_layer = None
        self.fd_dpo_low_layer = None
        self._fd_dpo_hidden_states = None
        self._fd_dpo_hidden_states_low = None
        self._fd_dpo_concatenated_labels = None
        self._fd_dpo_concatenated_attention_mask = None
        self._fd_dpo_reference_token_logps = None
        self._fd_dpo_batch_size = None
        self._fd_dpo_last_contrast_stats = {}
        self._fd_dpo_fabe_artifact_dir_ema = None
        self.cdpo_config = getattr(config, 'cdpo', None)
        self.cdpo_enabled = bool(
            self.cdpo_config is not None and getattr(self.cdpo_config, 'enabled', False)
        )

        if self.fd_dpo_enabled:
            objective = getattr(config.loss, 'objective', 'dpo')
            if config.loss.name != 'dpo' or objective != 'dpo':
                raise ValueError("fdDPO requires loss=dpo and loss.objective=dpo")
            if self.frontdoor is not None:
                raise ValueError("fdDPO is a reward correction and cannot be combined with frontdoor steering")

            self.fd_dpo_mode = getattr(self.fd_dpo_config, 'mode', 'hidden_state')
            if self.fd_dpo_mode not in {'hidden_state', 'logprob_distance'}:
                raise ValueError(
                    f"unknown fdDPO mode: {self.fd_dpo_mode} "
                    "(expected hidden_state or logprob_distance)"
                )
            self.fd_dpo_gate_mode = getattr(self.fd_dpo_config, 'gate_mode', 'none')
            if self.fd_dpo_gate_mode not in {
                'none',
                'margin_confidence',
                'adaptive_mixture_batchnorm_clipped',
                'confidence_weight',
                'clipped',
                'confidence_weight_clipped',
            }:
                raise ValueError(
                    f"unknown fdDPO gate_mode: {self.fd_dpo_gate_mode} "
                    "(expected none, margin_confidence, adaptive_mixture_batchnorm_clipped, "
                    "confidence_weight, clipped, or confidence_weight_clipped)"
                )
            self.fd_dpo_fabe_variant = str(getattr(self.fd_dpo_config, 'fabe_variant', 'base'))
            if self.fd_dpo_fabe_variant not in {
                'base',
                'ema',
                'token_selective',
                'dual_layer',
                'counterfactual_approx',
                'noproj',        # artifact-projection ablation (Experiment 1)
                'length_only',   # artifact-component ablation: length signal only (Experiment 3)
                'norm_only',     # artifact-component ablation: norm signal only (Experiment 4)
                'length_diag',   # length-bias diagnostic run (Experiment 2)
            }:
                raise ValueError(
                    f"unknown fdDPO FABE variant: {self.fd_dpo_fabe_variant} "
                    "(expected base, ema, token_selective, dual_layer, counterfactual_approx, "
                    "noproj, length_only, norm_only, or length_diag)"
                )

            configured_layer = getattr(self.fd_dpo_config, 'layer', None)
            if self.fd_dpo_mode == 'hidden_state' or configured_layer is not None:
                self.fd_dpo_layer = self._resolve_fd_dpo_layer()
            if self.fd_dpo_mode == 'hidden_state' and self.fd_dpo_fabe_variant == 'dual_layer':
                self.fd_dpo_low_layer = self._resolve_fd_dpo_low_layer()
            print(
                "fdDPO enabled "
                f"(mode={self.fd_dpo_mode}, alpha={float(self.fd_dpo_config.alpha):.4f}, "
                f"layer={self.fd_dpo_layer}, "
                f"pool={self.fd_dpo_config.pool}, contrast={self.fd_dpo_config.contrast}, "
                f"gate_mode={self.fd_dpo_gate_mode})"
            )
            if self.fd_dpo_gate_mode == 'margin_confidence':
                rank0_print(
                    "fdDPO_v2 confidence gate active: "
                    f"k={float(getattr(self.fd_dpo_config, 'k', 5.0)):.4f}, "
                    f"tau={float(getattr(self.fd_dpo_config, 'tau', 0.05)):.4f}, "
                    f"detach_gate={bool(getattr(self.fd_dpo_config, 'detach_gate', True))}"
                )
            if self.fd_dpo_gate_mode == 'adaptive_mixture_batchnorm_clipped':
                rank0_print(
                    "fdDPO_v3 adaptive mixture active: "
                    f"alpha_v1={float(getattr(self.fd_dpo_config, 'alpha_v1', 0.05)):.4f}, "
                    f"alpha_v2={float(getattr(self.fd_dpo_config, 'alpha_v2', 0.10)):.4f}, "
                    f"alpha_norm={float(getattr(self.fd_dpo_config, 'alpha_norm', 0.02)):.4f}, "
                    f"k={float(getattr(self.fd_dpo_config, 'k', 5.0)):.4f}, "
                    f"tau={float(getattr(self.fd_dpo_config, 'tau', 0.05)):.4f}, "
                    f"mix_k={float(getattr(self.fd_dpo_config, 'mix_k', 10.0)):.4f}, "
                    f"mix_tau={float(getattr(self.fd_dpo_config, 'mix_tau', 0.05)):.4f}, "
                    f"clip={float(getattr(self.fd_dpo_config, 'clip', 0.05)):.4f}, "
                    f"detach_gate={bool(getattr(self.fd_dpo_config, 'detach_gate', True))}"
                )
            if self.fd_dpo_gate_mode in {'confidence_weight', 'clipped', 'confidence_weight_clipped'}:
                rank0_print(
                    "fdDPO next lightweight correction active: "
                    f"mode={self.fd_dpo_gate_mode}, "
                    f"confidence_k={float(getattr(self.fd_dpo_config, 'confidence_k', 1.0)):.4f}, "
                    f"clip={getattr(self.fd_dpo_config, 'correction_clip', None)}, "
                    f"detach_gate={bool(getattr(self.fd_dpo_config, 'detach_gate', True))}"
                )
            if self.fd_dpo_mode == 'hidden_state':
                if self.fd_dpo_fabe_variant == 'dual_layer':
                    self.fd_dpo_capture_low = FrontdoorMediatorCapture(
                        policy,
                        layer=self.fd_dpo_low_layer,
                        detach_mediator=self.fd_dpo_config.detach_mediator,
                        debug=getattr(self.fd_dpo_config, 'debug', False),
                    )
                self.fd_dpo_capture = FrontdoorMediatorCapture(
                    policy,
                    layer=self.fd_dpo_layer,
                    detach_mediator=self.fd_dpo_config.detach_mediator,
                    debug=getattr(self.fd_dpo_config, 'debug', False),
                )
                if getattr(self.fd_dpo_config, 'contrast', None) == 'fabe':
                    rank0_print(
                        "FABE-DPO correction config: "
                        f"confidence_weight={bool(getattr(self.fd_dpo_config, 'fabe_confidence_weight', False))}, "
                        f"confidence_k={float(getattr(self.fd_dpo_config, 'fabe_confidence_k', 1.0)):.4f}, "
                        f"clip={getattr(self.fd_dpo_config, 'fabe_correction_clip', None)}, "
                        f"layernorm={bool(getattr(self.fd_dpo_config, 'fabe_layernorm', False))}, "
                        f"variant={self.fd_dpo_fabe_variant}, "
                        f"low_layer={self.fd_dpo_low_layer if self.fd_dpo_low_layer is not None else 'none'}, "
                        f"token_temperature={float(getattr(self.fd_dpo_config, 'fabe_token_temperature', 1.0)):.4f}, "
                        f"ema_momentum={float(getattr(self.fd_dpo_config, 'fabe_ema_momentum', 0.99)):.4f}"
                    )
                    if self.fd_dpo_fabe_variant == 'counterfactual_approx':
                        rank0_print(
                            "fdDPO_v38 counterfactual mediator substitution is approximate: "
                            "uses a no-extra-forward linearized mediator-effect proxy, not activation injection."
                        )
            else:
                rank0_print(
                    "fdDPO lightweight mode active: "
                    "using existing policy/reference log-probs; hidden hook disabled; "
                    "forward passes per batch=2"
                )

        if self.cdpo_enabled:
            objective = getattr(config.loss, 'objective', 'dpo')
            if config.loss.name != 'dpo' or objective != 'cdpo':
                raise ValueError("CDPO requires loss=dpo, loss.objective=cdpo, and cdpo.enabled=true")
            if self.frontdoor is not None or self.fd_dpo_enabled:
                raise ValueError("CDPO baseline must not be combined with frontdoor or fdDPO corrections")
            rank0_print(
                "CDPO-proxy-backdoor enabled "
                f"(proxy={getattr(self.cdpo_config, 'confounder_proxy', 'length_bin')}, "
                f"num_bins={int(getattr(self.cdpo_config, 'num_bins', 5))}, "
                f"weight_clip={float(getattr(self.cdpo_config, 'weight_clip', 5.0))}, "
                f"normalize_weights={bool(getattr(self.cdpo_config, 'normalize_weights', True))})"
            )
            
        self.train_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split='train',
            n_epochs=config.n_epochs,
            n_examples=train_n_examples,
            batch_size=config.batch_size,
            seed=self.seed,
            silent=rank != 0,
            cache_dir=get_local_dir(config.local_dirs),
        )
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split='test',
            n_examples=eval_n_examples,
            batch_size=config.eval_batch_size,
            seed=self.seed,
            silent=rank != 0,
            cache_dir=get_local_dir(config.local_dirs),
        )
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')
        if config.runtime_budget.enabled:
            rank0_print(
                'Runtime budget config: '
                f'enabled={config.runtime_budget.enabled} '
                f'full_data={config.runtime_budget.full_data} '
                f'max_hours_per_job={config.runtime_budget.max_hours_per_job} '
                f'max_steps={config.runtime_budget.max_steps} '
                f'max_train_samples={train_n_examples} '
                f'max_eval_samples={eval_n_examples}'
            )

    def _resolve_fd_dpo_layer(self) -> int:
        configured_layer = getattr(self.fd_dpo_config, 'layer', None)
        if configured_layer is not None:
            return int(configured_layer)

        num_layers = len(self.policy.model.layers)
        model_id = str(getattr(self.config.model, 'name_or_path', '')).lower()
        if 'qwen2.5-3b' in model_id or 'qwen3b' in model_id:
            default_layer = 31
        elif 'tinyllama' in model_id or 'qwen2.5-0.5b' in model_id or 'qwen05b' in model_id:
            default_layer = 20
        else:
            default_layer = 20
        return min(default_layer, num_layers - 1)

    def _resolve_fd_dpo_low_layer(self) -> int:
        configured_layer = getattr(self.fd_dpo_config, 'low_layer', None)
        if configured_layer is not None:
            return int(configured_layer)

        num_layers = len(self.policy.model.layers)
        model_id = str(getattr(self.config.model, 'name_or_path', '')).lower()
        if 'qwen2.5-3b' in model_id or 'qwen3b' in model_id:
            default_layer = 20
        elif 'tinyllama' in model_id or 'qwen2.5-0.5b' in model_id or 'qwen05b' in model_id:
            default_layer = 12
        else:
            default_layer = max(0, min(12, num_layers - 1))
        return min(default_layer, num_layers - 1)

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.uses_reference_model:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.uses_reference_model:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
        
    def concatenated_forward(
        self, 
        model: nn.Module, 
        batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        """Run the given model on the concatenated chosen + rejected inputs.
        Steering must happen ONLY for the POLICY model.
        """

        B = batch['chosen_input_ids'].shape[0]
        concatenated_batch = concatenated_inputs(batch)
        use_fd_dpo_capture = model is self.policy and self.fd_dpo_capture is not None
        capture_reference_token_logps = (
            model is self.reference_model
            and self.fd_dpo_enabled
            and getattr(self.fd_dpo_config, 'contrast', None) == 'fabe'
            and str(getattr(self.fd_dpo_config, 'pool', 'response_mean')) == 'token_selective'
        )

        # --- ONLY steer policy model ---
        if model is self.policy and self.frontdoor is not None:
            self.frontdoor.begin_batch(B, labels=concatenated_batch['concatenated_labels'])
        else:
            # disable steering for reference
            if hasattr(self, "frontdoor") and self.frontdoor is not None:
                self.frontdoor.disable()

        if model is self.policy and self.frontdoor is not None:
            self.frontdoor.set_context(alpha_scale=self.frontdoor.alpha_scale)

        if use_fd_dpo_capture:
            self._fd_dpo_hidden_states = None
            self._fd_dpo_hidden_states_low = None
            self._fd_dpo_concatenated_labels = None
            self._fd_dpo_concatenated_attention_mask = None
            self._fd_dpo_batch_size = None
            self._fd_dpo_last_contrast_stats = {}
            if self.fd_dpo_capture_low is not None:
                self.fd_dpo_capture_low.begin()
            self.fd_dpo_capture.begin()
        elif self.fd_dpo_capture is not None:
            self.fd_dpo_capture.disable()
            if self.fd_dpo_capture_low is not None:
                self.fd_dpo_capture_low.disable()
        if capture_reference_token_logps:
            self._fd_dpo_reference_token_logps = None

        # Forward pass
        outputs = model(
            concatenated_batch['concatenated_input_ids'],
            attention_mask=concatenated_batch['concatenated_attention_mask']
        )

        logits = outputs.logits.to(torch.float32)

        # Compute logps
        all_logps = _get_batch_logps(
            logits,
            concatenated_batch['concatenated_labels'],
            average_log_prob=False
        )
        if capture_reference_token_logps:
            self._fd_dpo_reference_token_logps = _get_batch_token_logps(
                logits,
                concatenated_batch['concatenated_labels'],
            ).detach()

        # Split chosen/rejected
        chosen_logps   = all_logps[:B]
        rejected_logps = all_logps[B:]

        if model is self.policy and self.frontdoor is not None:
            self.frontdoor.clear_context()
        if use_fd_dpo_capture:
            hidden_states = self.fd_dpo_capture.hidden_states
            hidden_states_low = self.fd_dpo_capture_low.hidden_states if self.fd_dpo_capture_low is not None else None
            self.fd_dpo_capture.clear()
            if self.fd_dpo_capture_low is not None:
                self.fd_dpo_capture_low.clear()
            if hidden_states is None:
                raise RuntimeError(f"fdDPO capture hook did not run for layer {self.fd_dpo_layer}")
            if self.fd_dpo_capture_low is not None and hidden_states_low is None:
                raise RuntimeError(f"fdDPO low-layer capture hook did not run for layer {self.fd_dpo_low_layer}")
            self._fd_dpo_hidden_states = hidden_states
            self._fd_dpo_hidden_states_low = hidden_states_low
            self._fd_dpo_concatenated_labels = concatenated_batch['concatenated_labels']
            self._fd_dpo_concatenated_attention_mask = concatenated_batch['concatenated_attention_mask']
            self._fd_dpo_batch_size = B
        return chosen_logps, rejected_logps


    def token_contrast_forward(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        kl_direction: str,
        weights: Optional[torch.FloatTensor] = None,
        tisdpo_weight_mode: str = "uniform",
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """Run policy/reference once and compute token-level preference statistics."""
        if self.reference_model is None:
            raise ValueError("token-level DPO objectives require a reference model")

        B = batch['chosen_input_ids'].shape[0]
        concatenated_batch = concatenated_inputs(batch)
        if self.frontdoor is not None:
            self.frontdoor.disable()
        if self.fd_dpo_capture is not None:
            self.fd_dpo_capture.disable()

        policy_outputs = self.policy(
            concatenated_batch['concatenated_input_ids'],
            attention_mask=concatenated_batch['concatenated_attention_mask'],
        )
        policy_logits = policy_outputs.logits.to(torch.float32)
        with torch.no_grad():
            reference_outputs = self.reference_model(
                concatenated_batch['concatenated_input_ids'],
                attention_mask=concatenated_batch['concatenated_attention_mask'],
            )
            reference_logits = reference_outputs.logits.to(torch.float32)

        if weights is None:
            concatenated_weights = None
        else:
            concatenated_weights = pad_to_length(
                weights,
                concatenated_batch['concatenated_labels'].shape[1],
                pad_value=0,
            )

        all_logps_margin, all_position_kl, all_logps, all_weight_mean = _get_weighted_token_stats(
            policy_logits,
            reference_logits,
            concatenated_batch['concatenated_labels'],
            weights=concatenated_weights,
            average_log_prob=False,
            kl_direction=kl_direction,
            tisdpo_weight_mode=tisdpo_weight_mode,
            tisdpo_min_weight=getattr(self.config.loss, 'tisdpo_min_weight', 0.25),
            tisdpo_max_weight=getattr(self.config.loss, 'tisdpo_max_weight', 2.0),
        )

        chosen_logps_margin = all_logps_margin[:B]
        rejected_logps_margin = all_logps_margin[B:]
        chosen_position_kl = all_position_kl[:B]
        rejected_position_kl = all_position_kl[B:]
        chosen_logps = all_logps[:B].detach()
        rejected_logps = all_logps[B:].detach()
        chosen_weight_mean = all_weight_mean[:B]
        rejected_weight_mean = all_weight_mean[B:]
        return (
            chosen_logps_margin,
            rejected_logps_margin,
            chosen_position_kl,
            rejected_position_kl,
            chosen_logps,
            rejected_logps,
            chosen_weight_mean,
            rejected_weight_mean,
        )


    def _fabe_reference_token_logps(self) -> Optional[torch.Tensor]:
        if str(getattr(self.fd_dpo_config, 'pool', 'response_mean')) != 'token_selective':
            return None
        if self._fd_dpo_reference_token_logps is None:
            raise RuntimeError("fdDPO token-selective FABE requires reference token log-probs")
        return self._fd_dpo_reference_token_logps

    def _update_fabe_ema_direction(self, batch_dir: torch.Tensor) -> torch.Tensor:
        momentum = float(getattr(self.fd_dpo_config, 'fabe_ema_momentum', 0.99))
        direction = batch_dir.detach().float()
        if self._fd_dpo_fabe_artifact_dir_ema is None:
            self._fd_dpo_fabe_artifact_dir_ema = direction
        else:
            self._fd_dpo_fabe_artifact_dir_ema = (
                momentum * self._fd_dpo_fabe_artifact_dir_ema.to(device=direction.device)
                + (1.0 - momentum) * direction
            )
        self._fd_dpo_fabe_artifact_dir_ema = F.normalize(
            self._fd_dpo_fabe_artifact_dir_ema,
            dim=0,
            eps=float(getattr(self.fd_dpo_config, 'eps', 1e-6)),
        )
        self._fd_dpo_fabe_artifact_dir_ema = torch.nan_to_num(
            self._fd_dpo_fabe_artifact_dir_ema,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).detach()
        return self._fd_dpo_fabe_artifact_dir_ema

    def _compute_single_fabe_score(
        self,
        hidden_states: torch.Tensor,
        train: bool,
        artifact_direction: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return compute_fddpo_fabe_score_with_stats(
            hidden_states,
            batch_size=self._fd_dpo_batch_size,
            labels=self._fd_dpo_concatenated_labels,
            attention_mask=self._fd_dpo_concatenated_attention_mask,
            pool=self.fd_dpo_config.pool,
            detach_mediator=self.fd_dpo_config.detach_mediator,
            layernorm_mediator=getattr(self.fd_dpo_config, 'fabe_layernorm', False),
            reference_token_logps=self._fabe_reference_token_logps(),
            token_temperature=getattr(self.fd_dpo_config, 'fabe_token_temperature', 1.0),
            artifact_direction=artifact_direction,
            eps=getattr(self.fd_dpo_config, 'eps', 1e-6),
        )

    def _compute_fd_dpo_contrast(self, train: bool = True) -> torch.Tensor:
        if self._fd_dpo_hidden_states is None:
            raise RuntimeError("fdDPO hidden states are missing; policy forward must run before correction")
        self._fd_dpo_last_contrast_stats = {}
        if getattr(self.fd_dpo_config, 'contrast', None) == 'fabe':
            fabe_variant = str(getattr(self.fd_dpo_config, 'fabe_variant', 'base'))
            if fabe_variant == 'dual_layer':
                if self._fd_dpo_hidden_states_low is None:
                    raise RuntimeError("fdDPO dual-layer FABE requires low-layer hidden states")
                lower_score, lower_stats = self._compute_single_fabe_score(
                    self._fd_dpo_hidden_states_low,
                    train=train,
                )
                upper_score, upper_stats = self._compute_single_fabe_score(
                    self._fd_dpo_hidden_states,
                    train=train,
                )
                lower_weight = float(getattr(self.fd_dpo_config, 'fabe_dual_lower_weight', 0.3))
                upper_weight = float(getattr(self.fd_dpo_config, 'fabe_dual_upper_weight', 0.7))
                score = lower_weight * lower_score + upper_weight * upper_score
                score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
                self._fd_dpo_last_contrast_stats = {
                    "lower_score_mean": lower_score.detach().mean(),
                    "upper_score_mean": upper_score.detach().mean(),
                }
                if "token_entropy" in upper_stats:
                    self._fd_dpo_last_contrast_stats["token_entropy"] = upper_stats["token_entropy"].detach()
                self._fd_dpo_last_contrast_stats["artifact_dir_used_norm"] = upper_stats[
                    "artifact_dir_used_norm"
                ].detach()
                return score

            if fabe_variant == 'ema':
                warmup_score, warmup_stats = self._compute_single_fabe_score(
                    self._fd_dpo_hidden_states,
                    train=train,
                    artifact_direction=self._fd_dpo_fabe_artifact_dir_ema,
                )
                if train:
                    ema_dir = self._update_fabe_ema_direction(warmup_stats["artifact_dir_batch"])
                    score, stats = self._compute_single_fabe_score(
                        self._fd_dpo_hidden_states,
                        train=train,
                        artifact_direction=ema_dir,
                    )
                else:
                    score, stats = warmup_score, warmup_stats
                self._fd_dpo_last_contrast_stats = {
                    key: value.detach() for key, value in stats.items() if key != "artifact_dir_batch"
                }
                if self._fd_dpo_fabe_artifact_dir_ema is not None:
                    self._fd_dpo_last_contrast_stats["ema_norm"] = self._fd_dpo_fabe_artifact_dir_ema.norm().detach()
                else:
                    self._fd_dpo_last_contrast_stats["ema_norm"] = stats["artifact_dir_used_norm"].detach()
                return score

            # --- Experiment 1: Artifact-Projection Ablation ---
            # Use raw cosine similarity on unprojected (but mean-centred) mediator
            # representations as the contrast score φ. No artifact direction is
            # estimated or projected out. This verifies the contribution of the
            # projection step itself.
            if fabe_variant == 'noproj':
                score, stats = compute_fddpo_fabe_noproj_score_with_stats(
                    self._fd_dpo_hidden_states,
                    batch_size=self._fd_dpo_batch_size,
                    labels=self._fd_dpo_concatenated_labels,
                    attention_mask=self._fd_dpo_concatenated_attention_mask,
                    pool=getattr(self.fd_dpo_config, 'pool', 'response_mean'),
                    detach_mediator=getattr(self.fd_dpo_config, 'detach_mediator', True),
                    layernorm_mediator=getattr(self.fd_dpo_config, 'fabe_layernorm', False),
                    eps=getattr(self.fd_dpo_config, 'eps', 1e-6),
                )
                self._fd_dpo_last_contrast_stats = {
                    key: value.detach() for key, value in stats.items()
                }
                return score

            # --- Experiment 3: Artifact-Component Ablation (length-only) ---
            # Projects out only the log-length direction.
            # artifact_score = standardize(log(length))
            if fabe_variant == 'length_only':
                score, stats = compute_fddpo_fabe_length_only_score_with_stats(
                    self._fd_dpo_hidden_states,
                    batch_size=self._fd_dpo_batch_size,
                    labels=self._fd_dpo_concatenated_labels,
                    attention_mask=self._fd_dpo_concatenated_attention_mask,
                    pool=getattr(self.fd_dpo_config, 'pool', 'response_mean'),
                    detach_mediator=getattr(self.fd_dpo_config, 'detach_mediator', True),
                    layernorm_mediator=getattr(self.fd_dpo_config, 'fabe_layernorm', False),
                    eps=getattr(self.fd_dpo_config, 'eps', 1e-6),
                )
                self._fd_dpo_last_contrast_stats = {
                    key: value.detach() for key, value in stats.items() if key != "artifact_dir_batch"
                }
                return score

            # --- Experiment 4: Artifact-Component Ablation (norm-only) ---
            # Projects out only the embedding-norm direction.
            # artifact_score = standardize(norm)
            if fabe_variant == 'norm_only':
                score, stats = compute_fddpo_fabe_norm_only_score_with_stats(
                    self._fd_dpo_hidden_states,
                    batch_size=self._fd_dpo_batch_size,
                    labels=self._fd_dpo_concatenated_labels,
                    attention_mask=self._fd_dpo_concatenated_attention_mask,
                    pool=getattr(self.fd_dpo_config, 'pool', 'response_mean'),
                    detach_mediator=getattr(self.fd_dpo_config, 'detach_mediator', True),
                    layernorm_mediator=getattr(self.fd_dpo_config, 'fabe_layernorm', False),
                    eps=getattr(self.fd_dpo_config, 'eps', 1e-6),
                )
                self._fd_dpo_last_contrast_stats = {
                    key: value.detach() for key, value in stats.items() if key != "artifact_dir_batch"
                }
                return score

            # --- Experiment 2: Length-Bias Diagnostic ---
            # During training, use the full FABE score as the contrast signal
            # (so the model trains identically to MPO-Safe/base) but also compute
            # and cache rich length-bias statistics that are then emitted as
            # additional metrics in get_batch_metrics.
            if fabe_variant == 'length_diag':
                score, stats = self._compute_single_fabe_score(
                    self._fd_dpo_hidden_states,
                    train=train,
                )
                self._fd_dpo_last_contrast_stats = {
                    key: value.detach() for key, value in stats.items() if key != "artifact_dir_batch"
                }
                # Stash the hidden states reference for diagnostic computation in
                # get_batch_metrics (will be consumed there before being cleared).
                self._fd_dpo_last_contrast_stats["_length_diag_pending"] = score.new_tensor(1.0)
                return score

            score, stats = self._compute_single_fabe_score(
                self._fd_dpo_hidden_states,
                train=train,
            )
            self._fd_dpo_last_contrast_stats = {
                key: value.detach() for key, value in stats.items() if key != "artifact_dir_batch"
            }
            return score
        return compute_fd_dpo_contrast(
            self._fd_dpo_hidden_states,
            batch_size=self._fd_dpo_batch_size,
            labels=self._fd_dpo_concatenated_labels,
            attention_mask=self._fd_dpo_concatenated_attention_mask,
            pool=self.fd_dpo_config.pool,
            contrast=self.fd_dpo_config.contrast,
            detach_mediator=self.fd_dpo_config.detach_mediator,
            walk_tau=getattr(self.fd_dpo_config, 'walk_tau', 0.25),
            fabe_layernorm=getattr(self.fd_dpo_config, 'fabe_layernorm', False),
            eps=getattr(self.fd_dpo_config, 'eps', 1e-6),
        )


    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name in {'dpo', 'ipo'}:
            objective = getattr(loss_config, 'objective', 'dpo')
            reference_chosen_logps, reference_rejected_logps = None, None
            margin_signal = None

            if objective not in {'simpo', 'tdpo', 'tisdpo'}:
                with torch.no_grad():
                    reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)
                    margin_signal = reference_chosen_logps - reference_rejected_logps

                if self.frontdoor is not None and self.config.frontdoor.dynamic_alpha:
                    # Lightweight dynamic-alpha: reuse the batch reward-gap
                    # signal that DPO already computes, compress it to a single
                    # scalar, and update it with an EMA at low frequency. This
                    # keeps the extension adaptive without any extra forward
                    # passes or dataset scans.
                    update_alpha = (
                        not train
                        or self.dynamic_alpha_state is None
                        or self.config.frontdoor.alpha_update_interval <= 1
                        or self.batch_counter % self.config.frontdoor.alpha_update_interval == 0
                    )
                    if update_alpha:
                        mean_gap = margin_signal.mean()
                        scaled_gap = self.config.frontdoor.confidence_scale * mean_gap + self.config.frontdoor.confidence_bias
                        alpha_scale = torch.sigmoid(scaled_gap).detach()
                        if train and self.dynamic_alpha_state is not None:
                            ema = self.config.frontdoor.alpha_ema_decay
                            alpha_scale = ema * self.dynamic_alpha_state + (1.0 - ema) * alpha_scale
                        self.dynamic_alpha_state = alpha_scale
                    self.frontdoor.set_context(alpha_scale=self.dynamic_alpha_state)
                elif self.frontdoor is not None:
                    gate_mode = getattr(self.config.frontdoor, 'confidence_gate_mode', 'none')
                    gate_scale = float(getattr(self.config.frontdoor, 'confidence_gate_scale', 1.0))
                    if gate_mode == 'batch_margin':
                        batch_gate = torch.sigmoid(-gate_scale * margin_signal.mean()).detach()
                        self.frontdoor.set_context(alpha_scale=batch_gate)
                    elif gate_mode == 'example_margin':
                        example_gate = torch.sigmoid(gate_scale * margin_signal).detach()
                        self.frontdoor.set_context(alpha_scale=example_gate)

            if objective in {'tdpo', 'tisdpo'}:
                if self.fd_dpo_enabled or self.frontdoor is not None:
                    raise ValueError("external TDPO/TIS-DPO baselines must not be combined with fdDPO/frontdoor")

                if objective == 'tdpo':
                    (
                        chosen_logps_margin,
                        rejected_logps_margin,
                        chosen_position_kl,
                        rejected_position_kl,
                        policy_chosen_logps,
                        policy_rejected_logps,
                        chosen_weight_mean,
                        rejected_weight_mean,
                    ) = self.token_contrast_forward(
                        batch,
                        kl_direction="ref_to_policy",
                        tisdpo_weight_mode="uniform",
                    )
                    losses, chosen_rewards, rejected_rewards = tdpo_loss(
                        chosen_logps_margin,
                        rejected_logps_margin,
                        chosen_position_kl,
                        rejected_position_kl,
                        beta=loss_config.beta,
                        alpha=getattr(loss_config, 'tdpo_alpha', 0.5),
                        tdpo2=bool(getattr(loss_config, 'tdpo2', True)),
                    )
                    metrics[f'tdpo_alpha/{train_test}'] = [float(getattr(loss_config, 'tdpo_alpha', 0.5))]
                    metrics[f'tdpo2/{train_test}'] = [1.0 if bool(getattr(loss_config, 'tdpo2', True)) else 0.0]
                else:
                    (
                        chosen_logps_margin,
                        rejected_logps_margin,
                        chosen_position_kl,
                        rejected_position_kl,
                        policy_chosen_logps,
                        policy_rejected_logps,
                        chosen_weight_mean,
                        rejected_weight_mean,
                    ) = self.token_contrast_forward(
                        batch,
                        kl_direction="policy_to_ref",
                        tisdpo_weight_mode=getattr(loss_config, 'tisdpo_weight_mode', 'uniform'),
                    )
                    losses, chosen_rewards, rejected_rewards = tisdpo_loss(
                        chosen_logps_margin,
                        rejected_logps_margin,
                        chosen_position_kl,
                        rejected_position_kl,
                        beta=loss_config.beta,
                        alpha=getattr(loss_config, 'tisdpo_alpha', 0.5),
                        token_level=bool(getattr(loss_config, 'tisdpo_token_level', True)),
                    )
                    metrics[f'tisdpo_alpha/{train_test}'] = [float(getattr(loss_config, 'tisdpo_alpha', 0.5))]
                    metrics[f'tisdpo_token_level/{train_test}'] = [1.0 if bool(getattr(loss_config, 'tisdpo_token_level', True)) else 0.0]
                    metrics[f'tisdpo_weight_chosen/{train_test}'] = chosen_weight_mean.cpu().numpy().tolist()
                    metrics[f'tisdpo_weight_rejected/{train_test}'] = rejected_weight_mean.cpu().numpy().tolist()

                chosen_kl = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
                rejected_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)
                metrics[f'kl_{train_test}/chosen'] = chosen_kl.cpu().numpy().tolist()
                metrics[f'kl_{train_test}/rejected'] = rejected_kl.cpu().numpy().tolist()
                metrics[f'kl_{train_test}/margin'] = (chosen_kl - rejected_kl).cpu().numpy().tolist()
            elif objective == 'simpo':
                policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
                # Lightweight SimPO-frontdoor variant: reuse the chosen/rejected
                # policy log-probs from the standard pass and normalize them by
                # cached response lengths. No reference model or auxiliary pass.
                chosen_lengths = batch.get('_chosen_lengths')
                rejected_lengths = batch.get('_rejected_lengths')
                if chosen_lengths is None or rejected_lengths is None:
                    chosen_lengths = (batch['chosen_labels'][:, 1:] != -100).sum(dim=-1).to(torch.float32)
                    rejected_lengths = (batch['rejected_labels'][:, 1:] != -100).sum(dim=-1).to(torch.float32)
                    batch['_chosen_lengths'] = chosen_lengths
                    batch['_rejected_lengths'] = rejected_lengths
                losses, chosen_rewards, rejected_rewards = simpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    chosen_lengths,
                    rejected_lengths,
                    beta=loss_config.beta,
                    gamma=loss_config.gamma,
                )
            elif objective == 'cdpo':
                if not self.cdpo_enabled:
                    raise ValueError("loss.objective=cdpo requires cdpo.enabled=true")
                if self.fd_dpo_enabled or self.frontdoor is not None:
                    raise ValueError("CDPO baseline must not be combined with fdDPO/frontdoor")
                policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
                cdpo_weights, cdpo_bins, cdpo_stats = compute_cdpo_proxy_weights(
                    batch,
                    confounder_proxy=getattr(self.cdpo_config, 'confounder_proxy', 'length_bin'),
                    num_bins=int(getattr(self.cdpo_config, 'num_bins', 5)),
                    weight_clip=float(getattr(self.cdpo_config, 'weight_clip', 5.0)),
                    normalize_weights=bool(getattr(self.cdpo_config, 'normalize_weights', True)),
                    backdoor_weight=bool(getattr(self.cdpo_config, 'backdoor_weight', True)),
                    max_length=int(getattr(self.config, 'max_length', 512)),
                    max_prompt_length=int(getattr(self.config, 'max_prompt_length', 256)),
                )
                losses, chosen_rewards, rejected_rewards = cdpo_proxy_preference_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    backdoor_weights=cdpo_weights,
                    beta=loss_config.beta,
                    label_smoothing=loss_config.label_smoothing,
                    reference_free=loss_config.reference_free,
                )
                metrics[f'cdpo_weight_mean/{train_test}'] = [cdpo_stats['weight_mean'].item()]
                metrics[f'cdpo_weight_std/{train_test}'] = [cdpo_stats['weight_std'].item()]
                metrics[f'cdpo_weight_min/{train_test}'] = [cdpo_stats['weight_min'].item()]
                metrics[f'cdpo_weight_max/{train_test}'] = [cdpo_stats['weight_max'].item()]
                metrics[f'cdpo_observed_bins/{train_test}'] = [cdpo_stats['observed_bins'].item()]
                metrics[f'cdpo_num_bins/{train_test}'] = [float(getattr(self.cdpo_config, 'num_bins', 5))]
                metrics[f'cdpo_proxy_length_bin/{train_test}'] = [
                    1.0 if str(getattr(self.cdpo_config, 'confounder_proxy', 'length_bin')) == 'length_bin' else 0.0
                ]
                metrics[f'cdpo_bins/{train_test}'] = cdpo_bins.cpu().numpy().tolist()
            else:
                policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
                if loss_config.name == 'dpo':
                    loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
                elif loss_config.name == 'ipo':
                    loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
                else:
                    raise ValueError(f'unknown loss {loss_config.name}')

                if self.fd_dpo_enabled:
                    if loss_config.name != 'dpo':
                        raise ValueError("fdDPO correction is only defined for DPO, not IPO")
                    if self.fd_dpo_mode == 'hidden_state':
                        fd_contrast = self._compute_fd_dpo_contrast(train=train)
                    elif self.fd_dpo_mode == 'logprob_distance':
                        chosen_lengths = batch.get('_chosen_lengths')
                        rejected_lengths = batch.get('_rejected_lengths')
                        if chosen_lengths is None or rejected_lengths is None:
                            chosen_lengths = (batch['chosen_labels'][:, 1:] != -100).sum(dim=-1)
                            rejected_lengths = (batch['rejected_labels'][:, 1:] != -100).sum(dim=-1)
                            batch['_chosen_lengths'] = chosen_lengths
                            batch['_rejected_lengths'] = rejected_lengths
                        fd_contrast = compute_logprob_distance_contrast(
                            policy_chosen_logps,
                            policy_rejected_logps,
                            reference_chosen_logps,
                            reference_rejected_logps,
                            chosen_lengths=chosen_lengths,
                            rejected_lengths=rejected_lengths,
                            beta=loss_config.beta,
                        )
                    else:
                        raise ValueError(f"unknown fdDPO mode: {self.fd_dpo_mode}")
                    fd_contrast_name = getattr(self.fd_dpo_config, 'contrast', 'one_minus_cosine')
                    fabe_variant_name = str(getattr(self.fd_dpo_config, 'fabe_variant', 'base'))
                    if fd_contrast_name == 'fabe' and fabe_variant_name == 'counterfactual_approx':
                        dpo_margin_proxy = loss_config.beta * (
                            (policy_chosen_logps - policy_rejected_logps)
                            - (reference_chosen_logps - reference_rejected_logps)
                        )
                        cf_effect = fd_contrast * torch.tanh(dpo_margin_proxy.detach().abs())
                        cf_effect = torch.nan_to_num(cf_effect, nan=0.0, posinf=0.0, neginf=0.0)
                        fd_contrast = cf_effect
                        self._fd_dpo_last_contrast_stats["cf_effect_mean"] = cf_effect.detach().mean()
                    if fd_contrast_name in {'causal_walk', 'fabe'}:
                        (
                            losses,
                            chosen_rewards,
                            rejected_rewards,
                            fd_correction,
                            fd_external_stats,
                        ) = fd_dpo_external_score_preference_loss(
                            policy_chosen_logps,
                            policy_rejected_logps,
                            reference_chosen_logps,
                            reference_rejected_logps,
                            mediator_score=fd_contrast,
                            beta=loss_config.beta,
                            alpha=self.fd_dpo_config.alpha,
                            label_smoothing=loss_config.label_smoothing,
                            reference_free=loss_config.reference_free,
                            confidence_weight=(
                                fd_contrast_name == 'fabe'
                                and bool(getattr(self.fd_dpo_config, 'fabe_confidence_weight', False))
                            ),
                            confidence_k=getattr(self.fd_dpo_config, 'fabe_confidence_k', 1.0),
                            correction_clip=(
                                getattr(self.fd_dpo_config, 'fabe_correction_clip', None)
                                if fd_contrast_name == 'fabe'
                                else None
                            ),
                            detach_gate=getattr(self.fd_dpo_config, 'detach_gate', True),
                            layernorm_enabled=(
                                fd_contrast_name == 'fabe'
                                and bool(getattr(self.fd_dpo_config, 'fabe_layernorm', False))
                            ),
                        )
                    elif self.fd_dpo_gate_mode in {'confidence_weight', 'clipped', 'confidence_weight_clipped'}:
                        use_confidence_weight = self.fd_dpo_gate_mode in {'confidence_weight', 'confidence_weight_clipped'}
                        use_clip = self.fd_dpo_gate_mode in {'clipped', 'confidence_weight_clipped'}
                        (
                            losses,
                            chosen_rewards,
                            rejected_rewards,
                            fd_correction,
                            fd_conf_weight,
                            fd_next_stats,
                        ) = fd_dpo_weighted_clipped_preference_loss(
                            policy_chosen_logps,
                            policy_rejected_logps,
                            reference_chosen_logps,
                            reference_rejected_logps,
                            mediator_contrast=fd_contrast,
                            beta=loss_config.beta,
                            alpha=self.fd_dpo_config.alpha,
                            confidence_weight=use_confidence_weight,
                            confidence_k=getattr(self.fd_dpo_config, 'confidence_k', 1.0),
                            correction_clip=(
                                getattr(self.fd_dpo_config, 'correction_clip', None)
                                if use_clip
                                else None
                            ),
                            detach_gate=getattr(self.fd_dpo_config, 'detach_gate', True),
                            label_smoothing=loss_config.label_smoothing,
                            reference_free=loss_config.reference_free,
                        )
                    elif self.fd_dpo_gate_mode == 'adaptive_mixture_batchnorm_clipped':
                        (
                            losses,
                            chosen_rewards,
                            rejected_rewards,
                            fd_correction,
                            fd_v3_stats,
                        ) = fd_dpo_v3_preference_loss(
                            policy_chosen_logps,
                            policy_rejected_logps,
                            reference_chosen_logps,
                            reference_rejected_logps,
                            mediator_contrast=fd_contrast,
                            beta=loss_config.beta,
                            alpha_v1=getattr(self.fd_dpo_config, 'alpha_v1', 0.05),
                            alpha_v2=getattr(self.fd_dpo_config, 'alpha_v2', 0.10),
                            alpha_norm=getattr(self.fd_dpo_config, 'alpha_norm', 0.02),
                            k=getattr(self.fd_dpo_config, 'k', 5.0),
                            tau=getattr(self.fd_dpo_config, 'tau', 0.05),
                            mix_k=getattr(self.fd_dpo_config, 'mix_k', 10.0),
                            mix_tau=getattr(self.fd_dpo_config, 'mix_tau', 0.05),
                            clip=getattr(self.fd_dpo_config, 'clip', 0.05),
                            eps=getattr(self.fd_dpo_config, 'eps', 1e-6),
                            detach_gate=getattr(self.fd_dpo_config, 'detach_gate', True),
                            label_smoothing=loss_config.label_smoothing,
                            reference_free=loss_config.reference_free,
                        )
                    elif self.fd_dpo_gate_mode == 'margin_confidence':
                        (
                            losses,
                            chosen_rewards,
                            rejected_rewards,
                            fd_gate,
                            fd_correction,
                        ) = fd_dpo_v2_preference_loss(
                            policy_chosen_logps,
                            policy_rejected_logps,
                            reference_chosen_logps,
                            reference_rejected_logps,
                            mediator_contrast=fd_contrast,
                            beta=loss_config.beta,
                            alpha=self.fd_dpo_config.alpha,
                            k=getattr(self.fd_dpo_config, 'k', 5.0),
                            tau=getattr(self.fd_dpo_config, 'tau', 0.05),
                            detach_gate=getattr(self.fd_dpo_config, 'detach_gate', True),
                            label_smoothing=loss_config.label_smoothing,
                            reference_free=loss_config.reference_free,
                        )
                    else:
                        losses, chosen_rewards, rejected_rewards = fd_dpo_preference_loss(
                            policy_chosen_logps,
                            policy_rejected_logps,
                            reference_chosen_logps,
                            reference_rejected_logps,
                            mediator_contrast=fd_contrast,
                            beta=loss_config.beta,
                            alpha=self.fd_dpo_config.alpha,
                            label_smoothing=loss_config.label_smoothing,
                            reference_free=loss_config.reference_free,
                        )
                    fd_contrast_detached = fd_contrast.detach()
                    metrics[f'fd_contrast_mean/{train_test}'] = [fd_contrast_detached.mean().item()]
                    metrics[f'fd_contrast_std/{train_test}'] = [fd_contrast_detached.std(unbiased=False).item()]
                    metrics[f'fd_alpha/{train_test}'] = [float(self.fd_dpo_config.alpha)]
                    metrics[f'fd_layer/{train_test}'] = [float(self.fd_dpo_layer) if self.fd_dpo_layer is not None else -1.0]
                    metrics[f'fd_mode_logprob_distance/{train_test}'] = [1.0 if self.fd_dpo_mode == 'logprob_distance' else 0.0]
                    if self.fd_dpo_gate_mode == 'margin_confidence':
                        fd_gate_detached = fd_gate.detach()
                        fd_correction_detached = fd_correction.detach()
                        metrics[f'fd_gate_mean/{train_test}'] = [fd_gate_detached.mean().item()]
                        metrics[f'fd_gate_std/{train_test}'] = [fd_gate_detached.std(unbiased=False).item()]
                        metrics[f'fd_correction_mean/{train_test}'] = [fd_correction_detached.mean().item()]
                        metrics[f'fd_tau/{train_test}'] = [float(getattr(self.fd_dpo_config, 'tau', 0.05))]
                        metrics[f'fd_k/{train_test}'] = [float(getattr(self.fd_dpo_config, 'k', 5.0))]
                    if self.fd_dpo_gate_mode == 'adaptive_mixture_batchnorm_clipped':
                        metrics[f'fd_v3_c_mean/{train_test}'] = [fd_v3_stats['c_mean'].item()]
                        metrics[f'fd_v3_c_std/{train_test}'] = [fd_v3_stats['c_std'].item()]
                        metrics[f'fd_v3_gate_mean/{train_test}'] = [fd_v3_stats['gate_mean'].item()]
                        metrics[f'fd_v3_gate_std/{train_test}'] = [fd_v3_stats['gate_std'].item()]
                        metrics[f'fd_v3_mix/{train_test}'] = [fd_v3_stats['mix'].item()]
                        metrics[f'fd_v3_corr_v1_mean/{train_test}'] = [fd_v3_stats['corr_v1_mean'].item()]
                        metrics[f'fd_v3_corr_v2_mean/{train_test}'] = [fd_v3_stats['corr_v2_mean'].item()]
                        metrics[f'fd_v3_corr_norm_mean/{train_test}'] = [fd_v3_stats['corr_norm_mean'].item()]
                        metrics[f'fd_v3_correction_mean/{train_test}'] = [fd_v3_stats['correction_mean'].item()]
                        metrics[f'fd_v3_correction_std/{train_test}'] = [fd_v3_stats['correction_std'].item()]
                        metrics[f'fd_v3_correction_abs_mean/{train_test}'] = [fd_v3_stats['correction_abs_mean'].item()]
                        metrics[f'fd_v3_clip_frac/{train_test}'] = [fd_v3_stats['clip_frac'].item()]
                        metrics[f'fd_v3_alpha_v1/{train_test}'] = [float(getattr(self.fd_dpo_config, 'alpha_v1', 0.05))]
                        metrics[f'fd_v3_alpha_v2/{train_test}'] = [float(getattr(self.fd_dpo_config, 'alpha_v2', 0.10))]
                        metrics[f'fd_v3_alpha_norm/{train_test}'] = [float(getattr(self.fd_dpo_config, 'alpha_norm', 0.02))]
                        metrics[f'fd_v3_tau/{train_test}'] = [float(getattr(self.fd_dpo_config, 'tau', 0.05))]
                        metrics[f'fd_v3_k/{train_test}'] = [float(getattr(self.fd_dpo_config, 'k', 5.0))]
                        metrics[f'fd_v3_mix_tau/{train_test}'] = [float(getattr(self.fd_dpo_config, 'mix_tau', 0.05))]
                        metrics[f'fd_v3_clip/{train_test}'] = [float(getattr(self.fd_dpo_config, 'clip', 0.05))]
                    if self.fd_dpo_gate_mode in {'confidence_weight', 'clipped', 'confidence_weight_clipped'}:
                        metrics[f'fddpo_correction_mean/{train_test}'] = [fd_next_stats['correction_mean'].item()]
                        metrics[f'fddpo_correction_std/{train_test}'] = [fd_next_stats['correction_std'].item()]
                        metrics[f'fddpo_correction_abs_mean/{train_test}'] = [fd_next_stats['correction_abs_mean'].item()]
                        metrics[f'fddpo_conf_weight_mean/{train_test}'] = [fd_next_stats['conf_weight_mean'].item()]
                        metrics[f'fddpo_clip_value/{train_test}'] = [fd_next_stats['clip_value'].item()]
                    if fd_contrast_name == 'causal_walk':
                        metrics[f'fddpo_v7_path_score_mean/{train_test}'] = [fd_external_stats['score_mean'].item()]
                        metrics[f'fddpo_v7_path_score_std/{train_test}'] = [fd_external_stats['score_std'].item()]
                        metrics[f'correction_mean/{train_test}'] = [fd_external_stats['correction_mean'].item()]
                        metrics[f'correction_abs_mean/{train_test}'] = [fd_external_stats['correction_abs_mean'].item()]
                        metrics[f'alpha/{train_test}'] = [fd_external_stats['alpha'].item()]
                    if fd_contrast_name == 'fabe':
                        metrics[f'fddpo_v8_fabe_score_mean/{train_test}'] = [fd_external_stats['score_mean'].item()]
                        metrics[f'fddpo_v8_fabe_score_std/{train_test}'] = [fd_external_stats['score_std'].item()]
                        metrics[f'correction_mean/{train_test}'] = [fd_external_stats['correction_mean'].item()]
                        metrics[f'correction_abs_mean/{train_test}'] = [fd_external_stats['correction_abs_mean'].item()]
                        variant_code = float(getattr(self.fd_dpo_config, 'fabe_variant_id', 30.0))
                        metrics[f'fabe_variant/{train_test}'] = [variant_code]
                        metrics[f'fabe_score_mean/{train_test}'] = [fd_external_stats['score_mean'].item()]
                        metrics[f'fabe_score_std/{train_test}'] = [fd_external_stats['score_std'].item()]
                        metrics[f'fabe_correction_mean/{train_test}'] = [fd_external_stats['margin_correction_mean'].item()]
                        metrics[f'fabe_correction_abs_mean/{train_test}'] = [fd_external_stats['margin_correction_abs_mean'].item()]
                        metrics[f'fabe_conf_mean/{train_test}'] = [fd_external_stats['conf_mean'].item()]
                        metrics[f'fabe_clip_value/{train_test}'] = [fd_external_stats['clip_value'].item()]
                        metrics[f'fabe_layernorm_enabled/{train_test}'] = [fd_external_stats['layernorm_enabled'].item()]
                        fabe_stats = self._fd_dpo_last_contrast_stats
                        if 'ema_norm' in fabe_stats:
                            metrics[f'fabe_ema_norm/{train_test}'] = [fabe_stats['ema_norm'].item()]
                        if 'token_entropy' in fabe_stats:
                            metrics[f'fabe_token_entropy/{train_test}'] = [fabe_stats['token_entropy'].item()]
                        if 'lower_score_mean' in fabe_stats:
                            metrics[f'fabe_lower_score_mean/{train_test}'] = [fabe_stats['lower_score_mean'].item()]
                        if 'upper_score_mean' in fabe_stats:
                            metrics[f'fabe_upper_score_mean/{train_test}'] = [fabe_stats['upper_score_mean'].item()]
                        if 'cf_effect_mean' in fabe_stats:
                            metrics[f'fabe_cf_effect_mean/{train_test}'] = [fabe_stats['cf_effect_mean'].item()]
                        metrics[f'alpha/{train_test}'] = [fd_external_stats['alpha'].item()]
                        # --- Experiment 2: Length-bias diagnostic (length_diag variant only) ---
                        # Compute and log the full length-bias diagnostic block while the
                        # hidden states are still available (they are cleared below).
                        if (
                            getattr(self.fd_dpo_config, 'fabe_variant', 'base') == 'length_diag'
                            and self._fd_dpo_hidden_states is not None
                        ):
                            try:
                                _diag = compute_length_bias_diagnostics(
                                    self._fd_dpo_hidden_states,
                                    batch_size=self._fd_dpo_batch_size,
                                    labels=self._fd_dpo_concatenated_labels,
                                    attention_mask=self._fd_dpo_concatenated_attention_mask,
                                    pool=getattr(self.fd_dpo_config, 'pool', 'response_mean'),
                                    detach_mediator=getattr(self.fd_dpo_config, 'detach_mediator', True),
                                    eps=getattr(self.fd_dpo_config, 'eps', 1e-6),
                                    reference_chosen_logps=reference_chosen_logps,
                                    reference_rejected_logps=reference_rejected_logps,
                                    beta=float(loss_config.beta),
                                    policy_chosen_logps=policy_chosen_logps.detach() if policy_chosen_logps is not None else None,
                                    policy_rejected_logps=policy_rejected_logps.detach() if policy_rejected_logps is not None else None,
                                )
                                for _k, _v in _diag.items():
                                    metrics[f'length_diag_{_k}/{train_test}'] = [float(_v.item())]
                            except Exception as _diag_exc:
                                rank0_print(f"[length_diag] diagnostic computation failed: {_diag_exc}")
                    # --- Experiment 1: Artifact-projection ablation (noproj variant) ---
                    # The score was already computed as fabe_noproj; log extra stats to
                    # mirror the standard fabe block so downstream analysis scripts find
                    # the same key names prefixed with noproj_.
                    if (
                        getattr(self.fd_dpo_config, 'contrast', 'one_minus_cosine') == 'fabe'
                        and getattr(self.fd_dpo_config, 'fabe_variant', 'base') == 'noproj'
                    ):
                        noproj_contrast_detached = fd_contrast.detach()
                        metrics[f'noproj_phi_mean/{train_test}'] = [noproj_contrast_detached.mean().item()]
                        metrics[f'noproj_phi_std/{train_test}'] = [noproj_contrast_detached.std(unbiased=False).item()]
                        # Also compute and log a companion full-FABE score so the two
                        # curves can be compared in the same run.
                        if self._fd_dpo_hidden_states is not None:
                            try:
                                _ref_score, _ref_stats = compute_fddpo_fabe_score_with_stats(
                                    self._fd_dpo_hidden_states,
                                    batch_size=self._fd_dpo_batch_size,
                                    labels=self._fd_dpo_concatenated_labels,
                                    attention_mask=self._fd_dpo_concatenated_attention_mask,
                                    pool=getattr(self.fd_dpo_config, 'pool', 'response_mean'),
                                    detach_mediator=getattr(self.fd_dpo_config, 'detach_mediator', True),
                                    eps=getattr(self.fd_dpo_config, 'eps', 1e-6),
                                )
                                metrics[f'noproj_fabe_ref_mean/{train_test}'] = [_ref_score.detach().mean().item()]
                                metrics[f'noproj_fabe_ref_std/{train_test}'] = [_ref_score.detach().std(unbiased=False).item()]
                                # Pearson r between noproj-φ and full-FABE-φ for sanity.
                                _xc = noproj_contrast_detached - noproj_contrast_detached.mean()
                                _yc = _ref_score.detach() - _ref_score.detach().mean()
                                _num = (_xc * _yc).sum()
                                _den = (_xc.pow(2).sum() * _yc.pow(2).sum()).sqrt().clamp_min(1e-8)
                                metrics[f'noproj_vs_fabe_corr/{train_test}'] = [float((_num / _den).item())]
                            except Exception as _noproj_exc:
                                rank0_print(f"[noproj] companion FABE computation failed: {_noproj_exc}")
                    # --- Experiment 3: Artifact-component ablation (length_only variant) ---
                    if (
                        getattr(self.fd_dpo_config, 'contrast', 'one_minus_cosine') == 'fabe'
                        and getattr(self.fd_dpo_config, 'fabe_variant', 'base') == 'length_only'
                    ):
                        length_only_contrast_detached = fd_contrast.detach()
                        metrics[f'length_only_phi_mean/{train_test}'] = [length_only_contrast_detached.mean().item()]
                        metrics[f'length_only_phi_std/{train_test}'] = [length_only_contrast_detached.std(unbiased=False).item()]
                        if self._fd_dpo_hidden_states is not None:
                            try:
                                _ref_score, _ref_stats = compute_fddpo_fabe_score_with_stats(
                                    self._fd_dpo_hidden_states,
                                    batch_size=self._fd_dpo_batch_size,
                                    labels=self._fd_dpo_concatenated_labels,
                                    attention_mask=self._fd_dpo_concatenated_attention_mask,
                                    pool=getattr(self.fd_dpo_config, 'pool', 'response_mean'),
                                    detach_mediator=getattr(self.fd_dpo_config, 'detach_mediator', True),
                                    eps=getattr(self.fd_dpo_config, 'eps', 1e-6),
                                )
                                metrics[f'length_only_fabe_ref_mean/{train_test}'] = [_ref_score.detach().mean().item()]
                                metrics[f'length_only_fabe_ref_std/{train_test}'] = [_ref_score.detach().std(unbiased=False).item()]
                                _xc = length_only_contrast_detached - length_only_contrast_detached.mean()
                                _yc = _ref_score.detach() - _ref_score.detach().mean()
                                _num = (_xc * _yc).sum()
                                _den = (_xc.pow(2).sum() * _yc.pow(2).sum()).sqrt().clamp_min(1e-8)
                                metrics[f'length_only_vs_fabe_corr/{train_test}'] = [float((_num / _den).item())]
                            except Exception as _length_only_exc:
                                rank0_print(f"[length_only] companion FABE computation failed: {_length_only_exc}")
                    # --- Experiment 4: Artifact-component ablation (norm_only variant) ---
                    if (
                        getattr(self.fd_dpo_config, 'contrast', 'one_minus_cosine') == 'fabe'
                        and getattr(self.fd_dpo_config, 'fabe_variant', 'base') == 'norm_only'
                    ):
                        norm_only_contrast_detached = fd_contrast.detach()
                        metrics[f'norm_only_phi_mean/{train_test}'] = [norm_only_contrast_detached.mean().item()]
                        metrics[f'norm_only_phi_std/{train_test}'] = [norm_only_contrast_detached.std(unbiased=False).item()]
                        if self._fd_dpo_hidden_states is not None:
                            try:
                                _ref_score, _ref_stats = compute_fddpo_fabe_score_with_stats(
                                    self._fd_dpo_hidden_states,
                                    batch_size=self._fd_dpo_batch_size,
                                    labels=self._fd_dpo_concatenated_labels,
                                    attention_mask=self._fd_dpo_concatenated_attention_mask,
                                    pool=getattr(self.fd_dpo_config, 'pool', 'response_mean'),
                                    detach_mediator=getattr(self.fd_dpo_config, 'detach_mediator', True),
                                    eps=getattr(self.fd_dpo_config, 'eps', 1e-6),
                                )
                                metrics[f'norm_only_fabe_ref_mean/{train_test}'] = [_ref_score.detach().mean().item()]
                                metrics[f'norm_only_fabe_ref_std/{train_test}'] = [_ref_score.detach().std(unbiased=False).item()]
                                _xc = norm_only_contrast_detached - norm_only_contrast_detached.mean()
                                _yc = _ref_score.detach() - _ref_score.detach().mean()
                                _num = (_xc * _yc).sum()
                                _den = (_xc.pow(2).sum() * _yc.pow(2).sum()).sqrt().clamp_min(1e-8)
                                metrics[f'norm_only_vs_fabe_corr/{train_test}'] = [float((_num / _den).item())]
                            except Exception as _norm_only_exc:
                                rank0_print(f"[norm_only] companion FABE computation failed: {_norm_only_exc}")
                    self._fd_dpo_hidden_states = None
                    self._fd_dpo_hidden_states_low = None
                    self._fd_dpo_concatenated_labels = None
                    self._fd_dpo_concatenated_attention_mask = None
                    self._fd_dpo_reference_token_logps = None
                    self._fd_dpo_batch_size = None
                    self._fd_dpo_last_contrast_stats = {}
                else:
                    losses, chosen_rewards, rejected_rewards = preference_loss(
                        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.uses_reference_model:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None
        target_examples = self.config.n_examples
        if self.config.runtime_budget.enabled and not self.config.runtime_budget.full_data:
            if self.config.runtime_budget.max_train_samples is not None:
                target_examples = int(self.config.runtime_budget.max_train_samples)

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.uses_reference_model:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.uses_reference_model:
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                if self.config.sample_during_eval:                    
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.uses_reference_model:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.uses_reference_model:
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

                if self.example_counter > 0  and self.config.save_mid_epoch:
                    if self.config.debug:
                        rank0_print('skipping save in debug mode')
                    else:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                if target_examples is not None:
                    projected_total_hours = target_examples / max(examples_per_second, 1e-6) / 3600.0
                    mean_train_metrics['runtime/projected_total_hours'] = projected_total_hours
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')

            if self.config.runtime_budget.enabled:
                if self.config.runtime_budget.max_steps is not None and self.batch_counter >= int(self.config.runtime_budget.max_steps):
                    rank0_print(
                        f'RUNTIME_STEP_CAP_REACHED max_steps={self.config.runtime_budget.max_steps} '
                        f'examples={self.example_counter} updates={self.batch_counter}'
                    )
                    return

                if (
                    not self.config.runtime_budget.full_data
                    and target_examples is not None
                    and self.example_counter >= int(self.config.runtime_budget.watchdog_min_examples)
                ):
                    projected_total_hours = target_examples / max(examples_per_second, 1e-6) / 3600.0
                    if projected_total_hours > float(self.config.runtime_budget.max_hours_per_job):
                        self.runtime_budget_exceeded = True
                        rank0_print(
                            'RUNTIME_BUDGET_EXCEEDED '
                            f'projected_total_hours={projected_total_hours:.2f} '
                            f'limit_hours={self.config.runtime_budget.max_hours_per_job} '
                            f'examples={self.example_counter} updates={self.batch_counter} '
                            f'examples_per_second={examples_per_second:.6f}'
                        )
                        if self.config.runtime_budget.save_partial_on_exceed and not self.config.debug:
                            output_dir = os.path.join(self.run_dir, 'BUDGET_EXCEEDED')
                            self.save(output_dir, {
                                'runtime_budget_exceeded': True,
                                'projected_total_hours': projected_total_hours,
                                'examples': self.example_counter,
                                'updates': self.batch_counter,
                            })
                        return
            #### END TRAINING ####
        #### FINAL EVALUATION AFTER TRAINING ####
        rank0_print(f'Final evaluation after training {self.example_counter} examples')
        self.policy.eval()

        all_eval_metrics = defaultdict(list)
        if self.config.sample_during_eval:
            all_policy_samples, all_reference_samples = [], []
            policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
            if self.uses_reference_model:
                reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

        for eval_batch in tqdm.tqdm(self.eval_batches, desc='Final eval metrics') if self.rank == 0 else self.eval_batches:
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)
            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        if self.config.sample_during_eval:
            if self.config.n_eval_model_samples < self.config.eval_batch_size:
                rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                sample_batches = self.eval_batches[:1]
            else:
                n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                sample_batches = self.eval_batches[:n_sample_batches]

            for eval_batch in tqdm.tqdm(sample_batches, desc='Final samples') if self.rank == 0 else sample_batches:
                local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)
                all_policy_samples.extend(policy_samples)
                all_reference_samples.extend(reference_samples)

                for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                    policy_text_table.add_data(self.example_counter, prompt, sample)
                if self.uses_reference_model:
                    for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                        reference_text_table.add_data(self.example_counter, prompt, sample)

        mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
        rank0_print(f'FINAL eval: {formatted_dict(mean_eval_metrics)}')

        if self.config.sample_during_eval:
            rank0_print(json.dumps(all_policy_samples[:10], indent=2))
            if self.uses_reference_model:
                rank0_print(json.dumps(all_reference_samples[:10], indent=2))

        if self.config.wandb.enabled and self.rank == 0:
            wandb.log({f"final/{k}": v for k, v in mean_eval_metrics.items()}, step=self.example_counter)
            if self.config.sample_during_eval:
                wandb.log({"final_policy_samples": policy_text_table}, step=self.example_counter)
                if self.uses_reference_model:
                    wandb.log({"final_reference_samples": reference_text_table}, step=self.example_counter)


    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=getattr(config.model, 'fsdp_cpu_offload', False)),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if self.uses_reference_model:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)
        
        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
    
    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()
        

class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        
        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if self.uses_reference_model:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
    
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        
