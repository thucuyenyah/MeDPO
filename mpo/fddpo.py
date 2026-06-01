from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def dpo_loss_from_margin(
    reward_margin: torch.FloatTensor,
    label_smoothing: float = 0.0,
) -> torch.FloatTensor:
    """DPO logistic loss from an already beta-scaled reward margin."""
    return (
        -F.logsigmoid(reward_margin) * (1.0 - label_smoothing)
        - F.logsigmoid(-reward_margin) * label_smoothing
    )


def fd_dpo_loss_from_rewards(
    chosen_rewards: torch.FloatTensor,
    rejected_rewards: torch.FloatTensor,
    mediator_contrast: torch.FloatTensor,
    alpha: float,
    label_smoothing: float = 0.0,
) -> torch.FloatTensor:
    """fdDPO loss from beta-scaled rewards and a per-example mediator contrast."""
    if chosen_rewards.shape != rejected_rewards.shape:
        raise ValueError(
            f"chosen/rejected reward shape mismatch: {chosen_rewards.shape} vs {rejected_rewards.shape}"
        )
    if mediator_contrast.shape != chosen_rewards.shape:
        raise ValueError(
            f"mediator contrast shape {mediator_contrast.shape} does not match rewards {chosen_rewards.shape}"
        )

    contrast = mediator_contrast.to(device=chosen_rewards.device, dtype=chosen_rewards.dtype)
    fd_margin = chosen_rewards - rejected_rewards + 2.0 * float(alpha) * contrast
    return dpo_loss_from_margin(fd_margin, label_smoothing=label_smoothing)


def compute_margin_confidence_gate(
    dpo_margin: torch.FloatTensor,
    k: float = 5.0,
    tau: float = 0.05,
    detach_gate: bool = True,
) -> torch.FloatTensor:
    """Confidence gate from the beta-scaled DPO reward margin."""
    gate_source = dpo_margin.detach() if detach_gate else dpo_margin
    gate = torch.sigmoid(float(k) * (gate_source.abs() - float(tau)))
    return torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0)


def compute_simple_confidence_weight(
    dpo_margin: torch.FloatTensor,
    k: float = 1.0,
    detach_gate: bool = True,
) -> torch.FloatTensor:
    """Lightweight confidence weight sigmoid(k * |DPO margin|)."""
    weight_source = dpo_margin.detach() if detach_gate else dpo_margin
    weight = torch.sigmoid(float(k) * weight_source.abs())
    return torch.nan_to_num(weight, nan=0.0, posinf=1.0, neginf=0.0)


def fd_dpo_v2_loss_from_rewards(
    chosen_rewards: torch.FloatTensor,
    rejected_rewards: torch.FloatTensor,
    mediator_contrast: torch.FloatTensor,
    alpha: float,
    k: float = 5.0,
    tau: float = 0.05,
    detach_gate: bool = True,
    label_smoothing: float = 0.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Gated fdDPO loss from beta-scaled rewards.

    Returns per-example losses, gates, and reward corrections.
    """
    if chosen_rewards.shape != rejected_rewards.shape:
        raise ValueError(
            f"chosen/rejected reward shape mismatch: {chosen_rewards.shape} vs {rejected_rewards.shape}"
        )
    if mediator_contrast.shape != chosen_rewards.shape:
        raise ValueError(
            f"mediator contrast shape {mediator_contrast.shape} does not match rewards {chosen_rewards.shape}"
        )

    dpo_margin = chosen_rewards - rejected_rewards
    contrast = mediator_contrast.to(device=dpo_margin.device, dtype=dpo_margin.dtype)
    contrast = torch.nan_to_num(contrast, nan=0.0, posinf=0.0, neginf=0.0)
    gate = compute_margin_confidence_gate(dpo_margin, k=k, tau=tau, detach_gate=detach_gate)
    correction = float(alpha) * gate * contrast
    correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    fd_margin = dpo_margin + 2.0 * correction
    return dpo_loss_from_margin(fd_margin, label_smoothing=label_smoothing), gate, correction


def fd_dpo_weighted_clipped_loss_from_rewards(
    chosen_rewards: torch.FloatTensor,
    rejected_rewards: torch.FloatTensor,
    mediator_contrast: torch.FloatTensor,
    alpha: float,
    confidence_weight: bool = False,
    confidence_k: float = 1.0,
    correction_clip: Optional[float] = None,
    detach_gate: bool = True,
    label_smoothing: float = 0.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, Dict[str, torch.Tensor]]:
    """fdDPO next-gen correction: optional confidence weighting and clipping."""
    if chosen_rewards.shape != rejected_rewards.shape:
        raise ValueError(
            f"chosen/rejected reward shape mismatch: {chosen_rewards.shape} vs {rejected_rewards.shape}"
        )
    if mediator_contrast.shape != chosen_rewards.shape:
        raise ValueError(
            f"mediator contrast shape {mediator_contrast.shape} does not match rewards {chosen_rewards.shape}"
        )

    dpo_margin = chosen_rewards - rejected_rewards
    contrast = mediator_contrast.to(device=dpo_margin.device, dtype=dpo_margin.dtype)
    contrast = torch.nan_to_num(contrast, nan=0.0, posinf=0.0, neginf=0.0)

    if confidence_weight:
        weight = compute_simple_confidence_weight(
            dpo_margin,
            k=confidence_k,
            detach_gate=detach_gate,
        )
    else:
        weight = torch.ones_like(dpo_margin)

    correction = float(alpha) * weight * contrast
    correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)

    clip_tensor = dpo_margin.new_tensor(-1.0)
    if correction_clip is not None:
        clip_value = abs(float(correction_clip))
        correction = correction.clamp(min=-clip_value, max=clip_value)
        clip_tensor = dpo_margin.new_tensor(clip_value)

    correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    fd_margin = dpo_margin + 2.0 * correction
    stats = {
        "correction_mean": correction.detach().mean(),
        "correction_std": correction.detach().std(unbiased=False),
        "correction_abs_mean": correction.detach().abs().mean(),
        "conf_weight_mean": weight.detach().mean(),
        "clip_value": clip_tensor.detach(),
    }
    return dpo_loss_from_margin(fd_margin, label_smoothing=label_smoothing), correction, weight, stats


def fd_dpo_v3_loss_from_rewards(
    chosen_rewards: torch.FloatTensor,
    rejected_rewards: torch.FloatTensor,
    mediator_contrast: torch.FloatTensor,
    alpha_v1: float = 0.05,
    alpha_v2: float = 0.10,
    alpha_norm: float = 0.02,
    k: float = 5.0,
    tau: float = 0.05,
    mix_k: float = 10.0,
    mix_tau: float = 0.05,
    clip: float = 0.05,
    eps: float = 1e-6,
    detach_gate: bool = True,
    label_smoothing: float = 0.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, Dict[str, torch.Tensor]]:
    """Adaptive mixture fdDPO loss from beta-scaled rewards.

    The returned correction is the clipped per-example reward correction.
    """
    if chosen_rewards.shape != rejected_rewards.shape:
        raise ValueError(
            f"chosen/rejected reward shape mismatch: {chosen_rewards.shape} vs {rejected_rewards.shape}"
        )
    if mediator_contrast.shape != chosen_rewards.shape:
        raise ValueError(
            f"mediator contrast shape {mediator_contrast.shape} does not match rewards {chosen_rewards.shape}"
        )

    dpo_margin = chosen_rewards - rejected_rewards
    contrast = mediator_contrast.to(device=dpo_margin.device, dtype=dpo_margin.dtype)
    contrast = torch.nan_to_num(contrast, nan=0.0, posinf=0.0, neginf=0.0)

    gate = compute_margin_confidence_gate(dpo_margin, k=k, tau=tau, detach_gate=detach_gate)
    c_mean = contrast.mean()
    c_std = contrast.std(unbiased=False)
    c_norm = (contrast - c_mean) / (c_std + float(eps))
    c_norm = torch.nan_to_num(c_norm, nan=0.0, posinf=0.0, neginf=0.0)

    mix_source = contrast.detach().std(unbiased=False)
    mix = torch.sigmoid(float(mix_k) * (mix_source - float(mix_tau))).detach()

    corr_v1 = float(alpha_v1) * contrast
    corr_v2 = float(alpha_v2) * gate * contrast
    corr_norm = float(alpha_norm) * gate * c_norm
    preclip_correction = (1.0 - mix) * corr_v1 + mix * corr_v2 + corr_norm
    preclip_correction = torch.nan_to_num(preclip_correction, nan=0.0, posinf=0.0, neginf=0.0)

    clip_value = abs(float(clip))
    correction = preclip_correction.clamp(min=-clip_value, max=clip_value)
    correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    clip_frac = (preclip_correction.abs() > clip_value).to(dtype=dpo_margin.dtype).mean()

    fd_margin = dpo_margin + 2.0 * correction
    stats = {
        "c_mean": c_mean.detach(),
        "c_std": c_std.detach(),
        "gate_mean": gate.detach().mean(),
        "gate_std": gate.detach().std(unbiased=False),
        "mix": mix.detach(),
        "corr_v1_mean": corr_v1.detach().mean(),
        "corr_v2_mean": corr_v2.detach().mean(),
        "corr_norm_mean": corr_norm.detach().mean(),
        "correction_mean": correction.detach().mean(),
        "correction_std": correction.detach().std(unbiased=False),
        "correction_abs_mean": correction.detach().abs().mean(),
        "clip_frac": clip_frac.detach(),
    }
    return dpo_loss_from_margin(fd_margin, label_smoothing=label_smoothing), correction, stats


def fd_dpo_preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    mediator_contrast: torch.FloatTensor,
    beta: float,
    alpha: float,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute fdDPO loss and corrected implicit rewards.

    This mirrors standard DPO when alpha is zero. The mediator contrast can be
    detached by the caller; the log-probability margin remains differentiable.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if reference_free:
        ref_logratios = 0

    dpo_margin = beta * (pi_logratios - ref_logratios)
    contrast = mediator_contrast.to(device=dpo_margin.device, dtype=dpo_margin.dtype)
    if contrast.shape != dpo_margin.shape:
        raise ValueError(
            f"mediator contrast shape {contrast.shape} does not match DPO margin {dpo_margin.shape}"
        )

    fd_margin = dpo_margin + 2.0 * float(alpha) * contrast
    losses = dpo_loss_from_margin(fd_margin, label_smoothing=label_smoothing)

    reward_contrast = mediator_contrast.detach().to(device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach() + float(alpha) * reward_contrast
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach() - float(alpha) * reward_contrast

    return losses, chosen_rewards, rejected_rewards


def fd_dpo_v2_preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    mediator_contrast: torch.FloatTensor,
    beta: float,
    alpha: float,
    k: float = 5.0,
    tau: float = 0.05,
    detach_gate: bool = True,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute confidence-gated fdDPO loss and corrected implicit rewards."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if reference_free:
        ref_logratios = 0

    dpo_margin = beta * (pi_logratios - ref_logratios)
    contrast = mediator_contrast.to(device=dpo_margin.device, dtype=dpo_margin.dtype)
    if contrast.shape != dpo_margin.shape:
        raise ValueError(
            f"mediator contrast shape {contrast.shape} does not match DPO margin {dpo_margin.shape}"
        )
    contrast = torch.nan_to_num(contrast, nan=0.0, posinf=0.0, neginf=0.0)

    gate = compute_margin_confidence_gate(dpo_margin, k=k, tau=tau, detach_gate=detach_gate)
    correction = float(alpha) * gate * contrast
    correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    fd_margin = dpo_margin + 2.0 * correction
    losses = dpo_loss_from_margin(fd_margin, label_smoothing=label_smoothing)

    reward_correction = correction.detach().to(device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach() + reward_correction
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach() - reward_correction

    return losses, chosen_rewards, rejected_rewards, gate.detach(), reward_correction


def fd_dpo_weighted_clipped_preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    mediator_contrast: torch.FloatTensor,
    beta: float,
    alpha: float,
    confidence_weight: bool = False,
    confidence_k: float = 1.0,
    correction_clip: Optional[float] = None,
    detach_gate: bool = True,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, Dict[str, torch.Tensor]]:
    """Compute fdDPO v4-v6 loss and corrected implicit rewards."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if reference_free:
        ref_logratios = 0

    dpo_margin = beta * (pi_logratios - ref_logratios)
    contrast = mediator_contrast.to(device=dpo_margin.device, dtype=dpo_margin.dtype)
    if contrast.shape != dpo_margin.shape:
        raise ValueError(
            f"mediator contrast shape {contrast.shape} does not match DPO margin {dpo_margin.shape}"
        )

    dpo_chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    dpo_rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    losses, correction, weight, stats = fd_dpo_weighted_clipped_loss_from_rewards(
        dpo_chosen_rewards,
        dpo_rejected_rewards,
        contrast,
        alpha=alpha,
        confidence_weight=confidence_weight,
        confidence_k=confidence_k,
        correction_clip=correction_clip,
        detach_gate=detach_gate,
        label_smoothing=label_smoothing,
    )

    reward_correction = correction.detach().to(device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)
    chosen_rewards = dpo_chosen_rewards.detach() + reward_correction
    rejected_rewards = dpo_rejected_rewards.detach() - reward_correction

    return losses, chosen_rewards, rejected_rewards, reward_correction, weight.detach(), stats


def fd_dpo_external_score_preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    mediator_score: torch.FloatTensor,
    beta: float,
    alpha: float,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
    confidence_weight: bool = False,
    confidence_k: float = 1.0,
    correction_clip: Optional[float] = None,
    detach_gate: bool = True,
    layernorm_enabled: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, Dict[str, torch.Tensor]]:
    """Compute a DPO loss corrected by a signed external-front-door score.

    Hidden-state variants use a margin correction. For v30 this is exactly
    ``dpo_margin + alpha * score``. Later FABE variants optionally apply a
    confidence weight and/or clip to that margin correction. Rewards are
    adjusted symmetrically by half the margin correction so alpha=0 is exactly
    DPO.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if reference_free:
        ref_logratios = 0

    dpo_margin = beta * (pi_logratios - ref_logratios)
    score = mediator_score.to(device=dpo_margin.device, dtype=dpo_margin.dtype)
    if score.shape != dpo_margin.shape:
        raise ValueError(
            f"mediator score shape {score.shape} does not match DPO margin {dpo_margin.shape}"
        )
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    if confidence_weight:
        conf = compute_simple_confidence_weight(
            dpo_margin,
            k=confidence_k,
            detach_gate=detach_gate,
        )
    else:
        conf = torch.ones_like(dpo_margin)
    conf = torch.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0)

    margin_correction = float(alpha) * conf * score
    margin_correction = torch.nan_to_num(margin_correction, nan=0.0, posinf=0.0, neginf=0.0)

    clip_tensor = dpo_margin.new_tensor(-1.0)
    if correction_clip is not None:
        clip_value = abs(float(correction_clip))
        margin_correction = margin_correction.clamp(min=-clip_value, max=clip_value)
        clip_tensor = dpo_margin.new_tensor(clip_value)
    margin_correction = torch.nan_to_num(margin_correction, nan=0.0, posinf=0.0, neginf=0.0)

    correction = 0.5 * margin_correction
    correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    fd_margin = dpo_margin + margin_correction
    losses = dpo_loss_from_margin(fd_margin, label_smoothing=label_smoothing)

    dpo_chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    dpo_rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    reward_correction = correction.detach().to(device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)
    chosen_rewards = dpo_chosen_rewards.detach() + reward_correction
    rejected_rewards = dpo_rejected_rewards.detach() - reward_correction
    stats = {
        "score_mean": score.detach().mean(),
        "score_std": score.detach().std(unbiased=False),
        "correction_mean": reward_correction.detach().mean(),
        "correction_abs_mean": reward_correction.detach().abs().mean(),
        "margin_correction_mean": margin_correction.detach().mean(),
        "margin_correction_abs_mean": margin_correction.detach().abs().mean(),
        "conf_mean": conf.detach().mean(),
        "clip_value": clip_tensor.detach(),
        "layernorm_enabled": dpo_margin.new_tensor(1.0 if layernorm_enabled else 0.0).detach(),
        "alpha": dpo_margin.new_tensor(float(alpha)).detach(),
    }

    return losses, chosen_rewards, rejected_rewards, reward_correction, stats


def fd_dpo_v3_preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    mediator_contrast: torch.FloatTensor,
    beta: float,
    alpha_v1: float = 0.05,
    alpha_v2: float = 0.10,
    alpha_norm: float = 0.02,
    k: float = 5.0,
    tau: float = 0.05,
    mix_k: float = 10.0,
    mix_tau: float = 0.05,
    clip: float = 0.05,
    eps: float = 1e-6,
    detach_gate: bool = True,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, Dict[str, torch.Tensor]]:
    """Compute adaptive-mixture fdDPO_v3 loss and corrected implicit rewards."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if reference_free:
        ref_logratios = 0

    dpo_margin = beta * (pi_logratios - ref_logratios)
    contrast = mediator_contrast.to(device=dpo_margin.device, dtype=dpo_margin.dtype)
    if contrast.shape != dpo_margin.shape:
        raise ValueError(
            f"mediator contrast shape {contrast.shape} does not match DPO margin {dpo_margin.shape}"
        )

    dpo_chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    dpo_rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    losses, correction, stats = fd_dpo_v3_loss_from_rewards(
        dpo_chosen_rewards,
        dpo_rejected_rewards,
        contrast,
        alpha_v1=alpha_v1,
        alpha_v2=alpha_v2,
        alpha_norm=alpha_norm,
        k=k,
        tau=tau,
        mix_k=mix_k,
        mix_tau=mix_tau,
        clip=clip,
        eps=eps,
        detach_gate=detach_gate,
        label_smoothing=label_smoothing,
    )

    reward_correction = correction.detach().to(device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)
    chosen_rewards = dpo_chosen_rewards.detach() + reward_correction
    rejected_rewards = dpo_rejected_rewards.detach() - reward_correction

    return losses, chosen_rewards, rejected_rewards, reward_correction, stats


def compute_logprob_distance_contrast(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    chosen_lengths: Optional[torch.FloatTensor] = None,
    rejected_lengths: Optional[torch.FloatTensor] = None,
    beta: float = 1.0,
) -> torch.FloatTensor:
    """Hook-free fdDPO contrast from existing policy/reference log-probs.

    The scalar mediator proxy is the length-normalized implicit DPO reward for
    each response. This preserves the front-door reward-correction form without
    any hidden-state hook or additional model forward pass.
    """
    if policy_chosen_logps.shape != policy_rejected_logps.shape:
        raise ValueError(
            f"chosen/rejected logp shape mismatch: {policy_chosen_logps.shape} vs {policy_rejected_logps.shape}"
        )
    if reference_chosen_logps.shape != policy_chosen_logps.shape:
        raise ValueError(
            f"reference chosen shape {reference_chosen_logps.shape} does not match policy chosen {policy_chosen_logps.shape}"
        )
    if reference_rejected_logps.shape != policy_rejected_logps.shape:
        raise ValueError(
            f"reference rejected shape {reference_rejected_logps.shape} does not match policy rejected {policy_rejected_logps.shape}"
        )

    if chosen_lengths is None:
        chosen_lengths = torch.ones_like(policy_chosen_logps)
    if rejected_lengths is None:
        rejected_lengths = torch.ones_like(policy_rejected_logps)

    chosen_lengths = chosen_lengths.to(device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype).clamp_min(1.0)
    rejected_lengths = rejected_lengths.to(device=policy_rejected_logps.device, dtype=policy_rejected_logps.dtype).clamp_min(1.0)

    chosen_proxy = beta * (policy_chosen_logps - reference_chosen_logps) / chosen_lengths
    rejected_proxy = beta * (policy_rejected_logps - reference_rejected_logps) / rejected_lengths
    contrast = (chosen_proxy - rejected_proxy).abs().detach()
    contrast = torch.nan_to_num(contrast, nan=0.0, posinf=0.0, neginf=0.0)
    if contrast.shape != policy_chosen_logps.shape:
        raise ValueError(
            f"fdDPO logprob contrast shape {contrast.shape} does not match logps {policy_chosen_logps.shape}"
        )
    return contrast


def _select_pool_mask(
    hidden_states: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    pool: str,
) -> torch.Tensor:
    batch_seq_shape = hidden_states.shape[:2]
    if labels is not None and tuple(labels.shape[:2]) != tuple(batch_seq_shape):
        raise ValueError(
            f"fdDPO labels shape {labels.shape[:2]} does not match hidden shape {batch_seq_shape}"
        )
    if attention_mask is not None and tuple(attention_mask.shape[:2]) != tuple(batch_seq_shape):
        raise ValueError(
            f"fdDPO attention mask shape {attention_mask.shape[:2]} does not match hidden shape {batch_seq_shape}"
        )

    if pool in {"response_mean", "token_selective"}:
        if labels is not None:
            response_mask = labels.to(device=hidden_states.device) != -100
            if attention_mask is not None:
                valid_mask = attention_mask.to(device=hidden_states.device).bool()
                empty_response = response_mask.sum(dim=1, keepdim=True) == 0
                response_mask = torch.where(empty_response, valid_mask, response_mask)
            return response_mask
        if attention_mask is not None:
            return attention_mask.to(device=hidden_states.device).bool()
        return hidden_states.new_ones(batch_seq_shape, dtype=torch.bool)

    if pool == "attention_mean":
        if attention_mask is not None:
            return attention_mask.to(device=hidden_states.device).bool()
        return hidden_states.new_ones(batch_seq_shape, dtype=torch.bool)

    raise ValueError(f"unknown fdDPO pool mode: {pool}")


def _masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (hidden_states * weights.unsqueeze(-1)).sum(dim=1) / denom


def _align_token_logps_to_hidden(
    reference_token_logps: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if reference_token_logps.dim() != 2:
        raise ValueError("reference_token_logps must be rank-2 [2B, T] or [2B, T-1]")
    if reference_token_logps.size(0) != hidden_states.size(0):
        raise ValueError(
            f"reference token logp batch {reference_token_logps.size(0)} "
            f"does not match hidden batch {hidden_states.size(0)}"
        )

    seq_len = hidden_states.size(1)
    token_logps = reference_token_logps.to(device=hidden_states.device, dtype=hidden_states.dtype)
    if token_logps.size(1) == seq_len:
        return token_logps
    if token_logps.size(1) == seq_len - 1:
        aligned = hidden_states.new_zeros((hidden_states.size(0), seq_len), dtype=hidden_states.dtype)
        aligned[:, 1:] = token_logps
        return aligned
    raise ValueError(
        f"reference token logp length {token_logps.size(1)} is incompatible with hidden length {seq_len}"
    )


def _token_selective_weights(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    reference_token_logps: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    aligned_logps = _align_token_logps_to_hidden(reference_token_logps, hidden_states)
    valid_mask = mask.to(device=hidden_states.device).bool()
    surprisal = torch.nan_to_num(-aligned_logps.float(), nan=0.0, posinf=0.0, neginf=0.0)
    logits = surprisal / max(float(temperature), float(eps))
    logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits, dim=1)
    weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
    denom = weights.sum(dim=1, keepdim=True)
    fallback = valid_mask.to(dtype=weights.dtype)
    fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(float(eps))
    weights = torch.where(denom > float(eps), weights / denom.clamp_min(float(eps)), fallback)
    entropy = -(weights.clamp_min(float(eps)).log() * weights).sum(dim=1)
    return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), entropy.detach()


def _pool_fabe_mediator(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    pool: str,
    reference_token_logps: Optional[torch.Tensor] = None,
    token_temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if pool == "token_selective":
        if reference_token_logps is None:
            raise ValueError("fdDPO token_selective pooling requires reference token log-probs")
        weights, entropy = _token_selective_weights(
            hidden_states,
            mask,
            reference_token_logps,
            temperature=token_temperature,
            eps=eps,
        )
        pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, entropy
    return _masked_mean(hidden_states, mask), None


def _masked_mean_scalar(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype)
    denom = weights.sum(dim=1).clamp_min(float(eps))
    return (values * weights).sum(dim=1) / denom


def _validate_hidden_score_inputs(
    hidden_states: torch.Tensor,
    batch_size: int,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    pool: str,
    detach_mediator: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 3:
        raise ValueError("fdDPO hidden_states must be a rank-3 tensor [2B, T, H]")
    expected = 2 * int(batch_size)
    if hidden_states.size(0) != expected:
        raise ValueError(
            f"fdDPO expected hidden batch dimension {expected}, got {hidden_states.size(0)}"
        )

    mediator_hidden = hidden_states.detach() if detach_mediator else hidden_states
    mediator_hidden = torch.nan_to_num(mediator_hidden.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mask = _select_pool_mask(mediator_hidden, labels, attention_mask, pool)
    if tuple(mask.shape) != tuple(mediator_hidden.shape[:2]):
        raise ValueError(
            f"fdDPO mask shape {mask.shape} does not match hidden shape {mediator_hidden.shape[:2]}"
        )
    return mediator_hidden, mask


def _causal_walk_sequence_strength(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    walk_tau: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    mask = mask.to(device=hidden_states.device).bool()
    unit_hidden = F.normalize(hidden_states.float(), dim=-1, eps=float(eps))

    pooled = _masked_mean(unit_hidden, mask)
    pooled = F.normalize(pooled, dim=-1, eps=float(eps))
    token_cos = (unit_hidden * pooled.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)
    token_score = _masked_mean_scalar(0.5 * (token_cos + 1.0), mask, eps=eps)

    if hidden_states.size(1) < 2:
        return torch.nan_to_num(token_score, nan=0.0, posinf=0.0, neginf=0.0)

    pair_mask = mask[:, 1:] & mask[:, :-1]
    edge_cos = (unit_hidden[:, 1:] * unit_hidden[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
    edge_prob = torch.sigmoid(edge_cos / max(float(walk_tau), float(eps))).clamp_min(float(eps))
    edge_log_mean = _masked_mean_scalar(edge_prob.log(), pair_mask, eps=eps)
    edge_walk = torch.exp(edge_log_mean)
    edge_walk = torch.where(pair_mask.any(dim=1), edge_walk, token_score)

    strength = 0.5 * token_score + 0.5 * edge_walk
    return torch.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0)


def compute_fddpo_v7_causal_walk_score(
    hidden_states: torch.Tensor,
    batch_size: int,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pool: str = "response_mean",
    detach_mediator: bool = True,
    walk_tau: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CausalWalk-DPO score from response-token hidden-state paths.

    The sequence score is a lightweight random-walk proxy: adjacent response
    token cosine similarities define first-order transition probabilities, and
    their geometric mean is combined with token-to-sequence coherence.
    """
    mediator_hidden, mask = _validate_hidden_score_inputs(
        hidden_states, batch_size, labels, attention_mask, pool, detach_mediator
    )
    expected = 2 * int(batch_size)
    chosen_strength = _causal_walk_sequence_strength(
        mediator_hidden[:batch_size],
        mask[:batch_size],
        walk_tau=walk_tau,
        eps=eps,
    )
    rejected_strength = _causal_walk_sequence_strength(
        mediator_hidden[batch_size:expected],
        mask[batch_size:expected],
        walk_tau=walk_tau,
        eps=eps,
    )
    score = (chosen_strength - rejected_strength).clamp(min=-1.0, max=1.0)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if score.shape != (batch_size,):
        raise ValueError(f"fdDPO_v7 score shape {score.shape} does not match batch size {batch_size}")
    return score


def _standardize(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = values - values.mean()
    scale = centered.std(unbiased=False).clamp_min(float(eps))
    return torch.nan_to_num(centered / scale, nan=0.0, posinf=0.0, neginf=0.0)


def _estimate_fabe_artifact_direction(
    pooled: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_center = pooled.mean(dim=0, keepdim=True)
    residual = pooled - batch_center

    lengths = mask.to(device=pooled.device, dtype=pooled.dtype).sum(dim=1).clamp_min(1.0)
    norms = pooled.norm(dim=-1)
    artifact_score = _standardize(lengths.log(), eps=eps) + _standardize(norms, eps=eps)
    spurious_den = artifact_score.abs().sum().clamp_min(float(eps))
    spurious_direction = (residual * artifact_score.unsqueeze(1)).sum(dim=0, keepdim=True) / spurious_den
    spurious_unit = F.normalize(spurious_direction, dim=-1, eps=float(eps))
    spurious_unit = torch.nan_to_num(spurious_unit, nan=0.0, posinf=0.0, neginf=0.0)
    return residual, spurious_unit, lengths, norms


def _fabe_score_from_pooled(
    pooled: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    artifact_direction: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    expected = 2 * int(batch_size)
    if pooled.dim() != 2 or pooled.size(0) != expected:
        raise ValueError(f"FABE pooled mediator shape {pooled.shape} does not match [2B, H]")

    residual, batch_unit, lengths, norms = _estimate_fabe_artifact_direction(pooled, mask, eps=eps)
    if artifact_direction is None:
        spurious_unit = batch_unit
    else:
        spurious_unit = artifact_direction.to(device=pooled.device, dtype=pooled.dtype)
        if spurious_unit.dim() == 1:
            spurious_unit = spurious_unit.unsqueeze(0)
        if spurious_unit.shape != batch_unit.shape:
            raise ValueError(
                f"FABE artifact direction shape {spurious_unit.shape} does not match expected {batch_unit.shape}"
            )
        spurious_unit = F.normalize(spurious_unit.float(), dim=-1, eps=float(eps))
        spurious_unit = torch.nan_to_num(spurious_unit, nan=0.0, posinf=0.0, neginf=0.0)

    projection = (residual * spurious_unit).sum(dim=-1, keepdim=True) * spurious_unit
    causal = residual - projection
    raw_norm = residual.norm(dim=-1).clamp_min(float(eps))
    causal_norm = causal.norm(dim=-1)
    semantic_weight = (causal_norm / raw_norm).clamp(min=0.0, max=1.0)

    chosen_causal = causal[:batch_size]
    rejected_causal = causal[batch_size:expected]
    causal_contrast = 1.0 - F.cosine_similarity(chosen_causal, rejected_causal, dim=-1, eps=float(eps))
    pair_weight = 0.5 * (semantic_weight[:batch_size] + semantic_weight[batch_size:expected])
    score = (causal_contrast * pair_weight).clamp(min=0.0, max=1.0)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if score.shape != (batch_size,):
        raise ValueError(f"FABE score shape {score.shape} does not match batch size {batch_size}")

    stats = {
        "artifact_dir_batch": batch_unit.detach().squeeze(0),
        "artifact_dir_used_norm": spurious_unit.detach().norm(),
        "length_mean": lengths.detach().mean(),
        "mediator_norm_mean": norms.detach().mean(),
    }
    return score, stats


def compute_fddpo_fabe_score_with_stats(
    hidden_states: torch.Tensor,
    batch_size: int,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pool: str = "response_mean",
    detach_mediator: bool = True,
    layernorm_mediator: bool = False,
    reference_token_logps: Optional[torch.Tensor] = None,
    token_temperature: float = 1.0,
    artifact_direction: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """FABE score plus diagnostics and reusable artifact direction.

    ``pool=token_selective`` uses already-computed reference token log-probs and
    masks non-response tokens before softmax. Passing ``artifact_direction``
    lets v35 project with a trainer-level EMA direction while still estimating
    the current batch direction for the next EMA update.
    """
    mediator_hidden, mask = _validate_hidden_score_inputs(
        hidden_states, batch_size, labels, attention_mask, pool, detach_mediator
    )
    pooled, token_entropy = _pool_fabe_mediator(
        mediator_hidden,
        mask,
        pool=pool,
        reference_token_logps=reference_token_logps,
        token_temperature=token_temperature,
        eps=eps,
    )
    pooled = torch.nan_to_num(pooled.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if layernorm_mediator:
        pooled_mean = pooled.mean(dim=-1, keepdim=True)
        pooled_std = pooled.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(eps))
        pooled = (pooled - pooled_mean) / pooled_std
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

    score, stats = _fabe_score_from_pooled(
        pooled,
        mask,
        batch_size=batch_size,
        artifact_direction=artifact_direction,
        eps=eps,
    )
    if token_entropy is not None:
        stats["token_entropy"] = token_entropy.mean()
    return score, stats


def compute_fddpo_v8_fabe_score(
    hidden_states: torch.Tensor,
    batch_size: int,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pool: str = "response_mean",
    detach_mediator: bool = True,
    layernorm_mediator: bool = False,
    reference_token_logps: Optional[torch.Tensor] = None,
    token_temperature: float = 1.0,
    artifact_direction: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """FABE-DPO score after suppressing a batch-level spurious direction.

    Response mediators are pooled hidden states. A lightweight spurious
    direction is estimated from mediator residuals correlated with response
    length and mediator norm, then projected out before computing semantic
    chosen/rejected contrast.
    """
    score, _ = compute_fddpo_fabe_score_with_stats(
        hidden_states,
        batch_size=batch_size,
        labels=labels,
        attention_mask=attention_mask,
        pool=pool,
        detach_mediator=detach_mediator,
        layernorm_mediator=layernorm_mediator,
        reference_token_logps=reference_token_logps,
        token_temperature=token_temperature,
        artifact_direction=artifact_direction,
        eps=eps,
    )
    return score


def compute_fddpo_fabe_noproj_score_with_stats(
    hidden_states: torch.Tensor,
    batch_size: int,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pool: str = "response_mean",
    detach_mediator: bool = True,
    layernorm_mediator: bool = False,
    reference_token_logps: Optional[torch.Tensor] = None,
    token_temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Artifact-projection ablation: raw cosine similarity on unprojected mediators.

    Identical to the full FABE pipeline up to pooling, but skips the artifact
    direction estimation and projection step. The contrast score φ is therefore:

        φ_i = 1 - cosine(pool(m_chosen_i), pool(m_rejected_i))

    on the mean-centred residuals (so it is still centred, but *not* projected).
    This isolates the contribution of artifact removal from the contribution of
    adding a representation-based signal at all.
    """
    mediator_hidden, mask = _validate_hidden_score_inputs(
        hidden_states, batch_size, labels, attention_mask, pool, detach_mediator
    )
    pooled, token_entropy = _pool_fabe_mediator(
        mediator_hidden,
        mask,
        pool=pool,
        reference_token_logps=reference_token_logps,
        token_temperature=token_temperature,
        eps=eps,
    )
    pooled = torch.nan_to_num(pooled.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if layernorm_mediator:
        pooled_mean = pooled.mean(dim=-1, keepdim=True)
        pooled_std = pooled.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(eps))
        pooled = (pooled - pooled_mean) / pooled_std
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

    expected = 2 * int(batch_size)
    # Mean-centre only (no artifact projection).
    batch_center = pooled.mean(dim=0, keepdim=True)
    residual = pooled - batch_center

    chosen_residual = residual[:batch_size]
    rejected_residual = residual[batch_size:expected]

    lengths = mask.to(device=pooled.device, dtype=pooled.dtype).sum(dim=1).clamp_min(1.0)
    norms = pooled.norm(dim=-1)

    score = 1.0 - F.cosine_similarity(chosen_residual, rejected_residual, dim=-1, eps=float(eps))
    score = score.clamp(min=0.0, max=2.0)  # cosine ∈ [-1,1] → 1-cos ∈ [0,2]
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if score.shape != (batch_size,):
        raise ValueError(f"FABE-NoProj score shape {score.shape} does not match batch size {batch_size}")

    stats: Dict[str, torch.Tensor] = {
        "length_mean": lengths.detach().mean(),
        "mediator_norm_mean": norms.detach().mean(),
    }
    if token_entropy is not None:
        stats["token_entropy"] = token_entropy.mean()
    return score, stats


def _estimate_fabe_artifact_direction_length_only(
    pooled: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Artifact direction estimated from length signal only.

    artifact_score = standardize(log(length))
    """
    batch_center = pooled.mean(dim=0, keepdim=True)
    residual = pooled - batch_center

    lengths = mask.to(device=pooled.device, dtype=pooled.dtype).sum(dim=1).clamp_min(1.0)
    norms = pooled.norm(dim=-1)
    artifact_score = _standardize(lengths.log(), eps=eps)
    spurious_den = artifact_score.abs().sum().clamp_min(float(eps))
    spurious_direction = (residual * artifact_score.unsqueeze(1)).sum(dim=0, keepdim=True) / spurious_den
    spurious_unit = F.normalize(spurious_direction, dim=-1, eps=float(eps))
    spurious_unit = torch.nan_to_num(spurious_unit, nan=0.0, posinf=0.0, neginf=0.0)
    return residual, spurious_unit, lengths, norms


def _estimate_fabe_artifact_direction_norm_only(
    pooled: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Artifact direction estimated from embedding norm signal only.

    artifact_score = standardize(norm)
    """
    batch_center = pooled.mean(dim=0, keepdim=True)
    residual = pooled - batch_center

    lengths = mask.to(device=pooled.device, dtype=pooled.dtype).sum(dim=1).clamp_min(1.0)
    norms = pooled.norm(dim=-1)
    artifact_score = _standardize(norms, eps=eps)
    spurious_den = artifact_score.abs().sum().clamp_min(float(eps))
    spurious_direction = (residual * artifact_score.unsqueeze(1)).sum(dim=0, keepdim=True) / spurious_den
    spurious_unit = F.normalize(spurious_direction, dim=-1, eps=float(eps))
    spurious_unit = torch.nan_to_num(spurious_unit, nan=0.0, posinf=0.0, neginf=0.0)
    return residual, spurious_unit, lengths, norms


def _fabe_score_from_pooled_length_only(
    pooled: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """FABE score using length-only artifact direction."""
    expected = 2 * int(batch_size)
    if pooled.dim() != 2 or pooled.size(0) != expected:
        raise ValueError(f"FABE pooled mediator shape {pooled.shape} does not match [2B, H]")

    residual, spurious_unit, lengths, norms = _estimate_fabe_artifact_direction_length_only(pooled, mask, eps=eps)

    projection = (residual * spurious_unit).sum(dim=-1, keepdim=True) * spurious_unit
    causal = residual - projection
    raw_norm = residual.norm(dim=-1).clamp_min(float(eps))
    causal_norm = causal.norm(dim=-1)
    semantic_weight = (causal_norm / raw_norm).clamp(min=0.0, max=1.0)

    chosen_causal = causal[:batch_size]
    rejected_causal = causal[batch_size:expected]
    causal_contrast = 1.0 - F.cosine_similarity(chosen_causal, rejected_causal, dim=-1, eps=float(eps))
    pair_weight = 0.5 * (semantic_weight[:batch_size] + semantic_weight[batch_size:expected])
    score = (causal_contrast * pair_weight).clamp(min=0.0, max=1.0)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if score.shape != (batch_size,):
        raise ValueError(f"FABE-LengthOnly score shape {score.shape} does not match batch size {batch_size}")

    stats = {
        "artifact_dir_batch": spurious_unit.detach().squeeze(0),
        "artifact_dir_used_norm": spurious_unit.detach().norm(),
        "length_mean": lengths.detach().mean(),
        "mediator_norm_mean": norms.detach().mean(),
    }
    return score, stats


def _fabe_score_from_pooled_norm_only(
    pooled: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """FABE score using norm-only artifact direction."""
    expected = 2 * int(batch_size)
    if pooled.dim() != 2 or pooled.size(0) != expected:
        raise ValueError(f"FABE pooled mediator shape {pooled.shape} does not match [2B, H]")

    residual, spurious_unit, lengths, norms = _estimate_fabe_artifact_direction_norm_only(pooled, mask, eps=eps)

    projection = (residual * spurious_unit).sum(dim=-1, keepdim=True) * spurious_unit
    causal = residual - projection
    raw_norm = residual.norm(dim=-1).clamp_min(float(eps))
    causal_norm = causal.norm(dim=-1)
    semantic_weight = (causal_norm / raw_norm).clamp(min=0.0, max=1.0)

    chosen_causal = causal[:batch_size]
    rejected_causal = causal[batch_size:expected]
    causal_contrast = 1.0 - F.cosine_similarity(chosen_causal, rejected_causal, dim=-1, eps=float(eps))
    pair_weight = 0.5 * (semantic_weight[:batch_size] + semantic_weight[batch_size:expected])
    score = (causal_contrast * pair_weight).clamp(min=0.0, max=1.0)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if score.shape != (batch_size,):
        raise ValueError(f"FABE-NormOnly score shape {score.shape} does not match batch size {batch_size}")

    stats = {
        "artifact_dir_batch": spurious_unit.detach().squeeze(0),
        "artifact_dir_used_norm": spurious_unit.detach().norm(),
        "length_mean": lengths.detach().mean(),
        "mediator_norm_mean": norms.detach().mean(),
    }
    return score, stats


def compute_fddpo_fabe_length_only_score_with_stats(
    hidden_states: torch.Tensor,
    batch_size: int,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pool: str = "response_mean",
    detach_mediator: bool = True,
    layernorm_mediator: bool = False,
    reference_token_logps: Optional[torch.Tensor] = None,
    token_temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Artifact-component ablation: length-only artifact direction.

    Identical to the full FABE pipeline but estimates the artifact direction
    using only the log-length signal:

        artifact_score = standardize(log(length))

    This isolates whether length alone drives the bias reduction, without
    the embedding-norm component.
    """
    mediator_hidden, mask = _validate_hidden_score_inputs(
        hidden_states, batch_size, labels, attention_mask, pool, detach_mediator
    )
    pooled, token_entropy = _pool_fabe_mediator(
        mediator_hidden,
        mask,
        pool=pool,
        reference_token_logps=reference_token_logps,
        token_temperature=token_temperature,
        eps=eps,
    )
    pooled = torch.nan_to_num(pooled.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if layernorm_mediator:
        pooled_mean = pooled.mean(dim=-1, keepdim=True)
        pooled_std = pooled.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(eps))
        pooled = (pooled - pooled_mean) / pooled_std
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

    score, stats = _fabe_score_from_pooled_length_only(pooled, mask, batch_size, eps=eps)
    if token_entropy is not None:
        stats["token_entropy"] = token_entropy.mean()
    return score, stats


def compute_fddpo_fabe_norm_only_score_with_stats(
    hidden_states: torch.Tensor,
    batch_size: int,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pool: str = "response_mean",
    detach_mediator: bool = True,
    layernorm_mediator: bool = False,
    reference_token_logps: Optional[torch.Tensor] = None,
    token_temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Artifact-component ablation: norm-only artifact direction.

    Identical to the full FABE pipeline but estimates the artifact direction
    using only the embedding-norm signal:

        artifact_score = standardize(norm)

    This isolates whether embedding norm alone drives the bias reduction,
    without the log-length component.
    """
    mediator_hidden, mask = _validate_hidden_score_inputs(
        hidden_states, batch_size, labels, attention_mask, pool, detach_mediator
    )
    pooled, token_entropy = _pool_fabe_mediator(
        mediator_hidden,
        mask,
        pool=pool,
        reference_token_logps=reference_token_logps,
        token_temperature=token_temperature,
        eps=eps,
    )
    pooled = torch.nan_to_num(pooled.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if layernorm_mediator:
        pooled_mean = pooled.mean(dim=-1, keepdim=True)
        pooled_std = pooled.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(eps))
        pooled = (pooled - pooled_mean) / pooled_std
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

    score, stats = _fabe_score_from_pooled_norm_only(pooled, mask, batch_size, eps=eps)
    if token_entropy is not None:
        stats["token_entropy"] = token_entropy.mean()
    return score, stats


def compute_length_bias_diagnostics(
    hidden_states: torch.Tensor,
    batch_size: int,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pool: str = "response_mean",
    detach_mediator: bool = True,
    eps: float = 1e-6,
    reference_chosen_logps: Optional[torch.Tensor] = None,
    reference_rejected_logps: Optional[torch.Tensor] = None,
    beta: float = 0.1,
    policy_chosen_logps: Optional[torch.Tensor] = None,
    policy_rejected_logps: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute per-batch length-bias diagnostic statistics.

    Returns a dict of scalar tensors suitable for direct logging to W&B/console:

    - ``phi_fabe``          : MeDPO φ (with artifact projection)
    - ``phi_noproj``        : MeDPO φ without artifact projection (raw cosine)
    - ``dpo_margin``        : DPO reward margin β(log π_c/π_r − log ref_c/ref_r)
                              (only when reference log-probs are provided)
    - ``length_diff``       : chosen_len − rejected_len (scalar, signed)
    - ``corr_phi_fabe_len`` : Pearson r between φ_fabe and |length_diff|
    - ``corr_phi_noproj_len``: Pearson r between φ_noproj and |length_diff|
    - ``corr_dpo_margin_len``: Pearson r between dpo_margin and |length_diff|
    - ``fabe_len_bias_reduction``: |corr_noproj| − |corr_fabe| (positive = projection helps)

    All correlations are batch-level Pearson coefficients; they are noisy for
    small batches but accumulate meaningfully when logged over many steps.
    """
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 3:
        raise ValueError("length_bias hidden_states must be a rank-3 tensor [2B, T, H]")
    expected = 2 * int(batch_size)
    if hidden_states.size(0) != expected:
        raise ValueError(
            f"length_bias expected hidden batch dimension {expected}, got {hidden_states.size(0)}"
        )

    mediator_hidden, mask = _validate_hidden_score_inputs(
        hidden_states, batch_size, labels, attention_mask, pool, detach_mediator
    )

    # Compute both φ variants.
    phi_fabe, fabe_stats = compute_fddpo_fabe_score_with_stats(
        hidden_states,
        batch_size=batch_size,
        labels=labels,
        attention_mask=attention_mask,
        pool=pool,
        detach_mediator=detach_mediator,
        eps=eps,
    )
    phi_noproj, noproj_stats = compute_fddpo_fabe_noproj_score_with_stats(
        hidden_states,
        batch_size=batch_size,
        labels=labels,
        attention_mask=attention_mask,
        pool=pool,
        detach_mediator=detach_mediator,
        eps=eps,
    )

    # Response lengths from the mask.
    full_lengths = mask.to(device=mediator_hidden.device, dtype=torch.float32).sum(dim=1).clamp_min(1.0)
    chosen_lengths = full_lengths[:batch_size]
    rejected_lengths = full_lengths[batch_size:expected]
    length_diff = chosen_lengths - rejected_lengths  # signed per-example
    abs_length_diff = length_diff.abs()

    def _pearson(x: torch.Tensor, y: torch.Tensor, eps_val: float = 1e-8) -> torch.Tensor:
        if x.numel() < 2:
            return x.new_tensor(0.0)
        xc = x - x.mean()
        yc = y - y.mean()
        num = (xc * yc).sum()
        den = (xc.pow(2).sum() * yc.pow(2).sum()).sqrt().clamp_min(eps_val)
        return torch.nan_to_num(num / den, nan=0.0, posinf=0.0, neginf=0.0)

    corr_fabe = _pearson(phi_fabe, abs_length_diff)
    corr_noproj = _pearson(phi_noproj, abs_length_diff)

    result: Dict[str, torch.Tensor] = {
        "phi_fabe_mean": phi_fabe.detach().mean(),
        "phi_fabe_std": phi_fabe.detach().std(unbiased=False),
        "phi_noproj_mean": phi_noproj.detach().mean(),
        "phi_noproj_std": phi_noproj.detach().std(unbiased=False),
        "length_diff_mean": length_diff.detach().mean(),
        "abs_length_diff_mean": abs_length_diff.detach().mean(),
        "corr_phi_fabe_len": corr_fabe.detach(),
        "corr_phi_noproj_len": corr_noproj.detach(),
        "fabe_len_bias_reduction": (corr_noproj.abs() - corr_fabe.abs()).detach(),
    }

    if (
        reference_chosen_logps is not None
        and reference_rejected_logps is not None
        and policy_chosen_logps is not None
        and policy_rejected_logps is not None
    ):
        dpo_margin = float(beta) * (
            (policy_chosen_logps - policy_rejected_logps)
            - (reference_chosen_logps - reference_rejected_logps)
        )
        dpo_margin = torch.nan_to_num(dpo_margin.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        corr_dpo = _pearson(dpo_margin, abs_length_diff)
        result["dpo_margin_mean"] = dpo_margin.mean()
        result["dpo_margin_std"] = dpo_margin.std(unbiased=False)
        result["corr_dpo_margin_len"] = corr_dpo.detach()

    return result


def compute_fd_dpo_contrast(
    hidden_states: torch.Tensor,
    batch_size: int,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pool: str = "response_mean",
    contrast: str = "one_minus_cosine",
    detach_mediator: bool = True,
    walk_tau: float = 0.25,
    fabe_layernorm: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute c_i = 1 - cosine(m_chosen, m_rejected) from captured policy hiddens."""
    if contrast == "causal_walk":
        return compute_fddpo_v7_causal_walk_score(
            hidden_states,
            batch_size=batch_size,
            labels=labels,
            attention_mask=attention_mask,
            pool=pool,
            detach_mediator=detach_mediator,
            walk_tau=walk_tau,
            eps=eps,
        )
    if contrast == "fabe":
        return compute_fddpo_v8_fabe_score(
            hidden_states,
            batch_size=batch_size,
            labels=labels,
            attention_mask=attention_mask,
            pool=pool,
            detach_mediator=detach_mediator,
            layernorm_mediator=fabe_layernorm,
            eps=eps,
        )
    if contrast == "fabe_noproj":
        # Artifact-projection ablation: raw cosine on unprojected (but mean-centred)
        # mediator representations. Use this to isolate the contribution of
        # artifact removal from simply adding a representation-based signal.
        score, _ = compute_fddpo_fabe_noproj_score_with_stats(
            hidden_states,
            batch_size=batch_size,
            labels=labels,
            attention_mask=attention_mask,
            pool=pool,
            detach_mediator=detach_mediator,
            layernorm_mediator=fabe_layernorm,
            eps=eps,
        )
        return score

    if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 3:
        raise ValueError("fdDPO hidden_states must be a rank-3 tensor [2B, T, H]")
    expected = 2 * int(batch_size)
    if hidden_states.size(0) != expected:
        raise ValueError(
            f"fdDPO expected hidden batch dimension {expected}, got {hidden_states.size(0)}"
        )

    mediator_hidden = hidden_states.detach() if detach_mediator else hidden_states
    mask = _select_pool_mask(mediator_hidden, labels, attention_mask, pool)
    if tuple(mask.shape) != tuple(mediator_hidden.shape[:2]):
        raise ValueError(
            f"fdDPO mask shape {mask.shape} does not match hidden shape {mediator_hidden.shape[:2]}"
        )

    chosen_hidden = mediator_hidden[:batch_size]
    rejected_hidden = mediator_hidden[batch_size:expected]
    chosen_mask = mask[:batch_size]
    rejected_mask = mask[batch_size:expected]

    chosen_mediator = _masked_mean(chosen_hidden, chosen_mask).float()
    rejected_mediator = _masked_mean(rejected_hidden, rejected_mask).float()

    if contrast != "one_minus_cosine":
        raise ValueError(f"unknown fdDPO contrast mode: {contrast}")

    mediator_contrast = 1.0 - F.cosine_similarity(chosen_mediator, rejected_mediator, dim=-1)
    mediator_contrast = torch.nan_to_num(mediator_contrast, nan=0.0, posinf=0.0, neginf=0.0)
    if mediator_contrast.shape != (batch_size,):
        raise ValueError(
            f"fdDPO contrast shape {mediator_contrast.shape} does not match batch size {batch_size}"
        )
    return mediator_contrast
