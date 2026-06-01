import math

import torch


class FrontdoorMediatorCapture:
    """Capture one policy hidden layer for fdDPO without modifying activations."""

    def __init__(self, model, layer, detach_mediator=True, debug=False):
        self.model = model
        self.layer = int(layer)
        self.detach_mediator = bool(detach_mediator)
        self.debug = debug
        self.active = False
        self.hidden_states = None

        num_layers = len(model.model.layers)
        if self.layer < 0 or self.layer >= num_layers:
            raise ValueError(
                f"fdDPO layer {self.layer} is out of range for model with {num_layers} layers"
            )

        target_layer = model.model.layers[self.layer]
        target_layer_cls = type(target_layer)
        self.hook_handle = target_layer.register_forward_hook(
            self._make_capture_hook(target_layer_cls)
        )
        if self.debug:
            print(f"[fdDPO] Capture hook attached -> Layer[{self.layer}]")

    def begin(self):
        self.active = True
        self.hidden_states = None

    def disable(self):
        self.active = False
        self.hidden_states = None

    def clear(self):
        self.active = False

    def remove(self):
        self.hook_handle.remove()

    def _make_capture_hook(self, target_layer_cls):
        def hook(module, inputs, outputs):
            if not self.active or not isinstance(module, target_layer_cls):
                return outputs

            out_is_tuple = isinstance(outputs, (tuple, list))
            hidden_states = outputs[0] if out_is_tuple else outputs
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 3:
                if self.debug:
                    print("[fdDPO] skip capture (invalid output shape)")
                return outputs

            self.hidden_states = hidden_states.detach() if self.detach_mediator else hidden_states
            if self.debug:
                print(f"[fdDPO] captured hidden shape={tuple(hidden_states.shape)}")
            return outputs

        return hook


class FrontdoorSteering:
    def __init__(
        self,
        model,
        layer=None,
        layers=None,
        apply_layer=None,
        alpha=0.1,
        dynamic_alpha=False,
        normalize_direction=False,
        aggregate_mode="single",
        confidence_gate_mode="none",
        confidence_gate_scale=1.0,
        token_weight_mode="uniform",
        layer_weights=None,
        debug=False,
        optimization_mode="fast",
        detach_statistics=True,
        token_saliency_dim_stride=4,
        token_topk_fraction=0.25,
        sparse_layer_count=1,
        sparse_layer_schedule_bins=10,
    ):
        self.model = model
        self.alpha = alpha
        self.dynamic_alpha = dynamic_alpha
        self.normalize_direction = normalize_direction
        self.aggregate_mode = aggregate_mode
        self.confidence_gate_mode = confidence_gate_mode
        self.confidence_gate_scale = confidence_gate_scale
        self.token_weight_mode = token_weight_mode
        self.debug = debug
        # "full" keeps the earlier exploratory implementation; all other modes
        # are lightweight variants intended to stay close to original MeDPO
        # cost by reusing tensors from the same forward pass.
        self.optimization_mode = optimization_mode
        self.detach_statistics = detach_statistics
        self.token_saliency_dim_stride = max(1, int(token_saliency_dim_stride))
        self.token_topk_fraction = float(token_topk_fraction)
        self.sparse_layer_count = max(1, int(sparse_layer_count))
        self.sparse_layer_schedule_bins = max(1, int(sparse_layer_schedule_bins))
        self.full_mode = optimization_mode == "full"

        self.batch_size = None
        self.labels = None
        self.alpha_scale = None
        self._cached_token_weights = None
        self._active_layer_ids = set()
        self._forward_counter = 0
        self._layer_directions = {}

        if layers is None:
            if layer is None:
                raise ValueError("either layer or layers must be provided")
            layers = [layer]
        if len(layers) == 0:
            raise ValueError("layers must not be empty")

        self.layer_ids = list(layers)
        self.apply_layer = max(self.layer_ids) if apply_layer is None else int(apply_layer)
        if self.apply_layer not in self.layer_ids:
            raise ValueError("apply_layer must be one of the configured frontdoor layers")
        if layer_weights is None:
            layer_weights = [1.0] * len(self.layer_ids)
        if len(layer_weights) != len(self.layer_ids):
            raise ValueError("layer_weights must have the same length as layers")

        weights = torch.tensor(layer_weights, dtype=torch.float32)
        weights = weights / weights.sum()
        self.layer_weights = {layer_id: weight.item() for layer_id, weight in zip(self.layer_ids, weights)}
        self._sorted_layers = [layer_id for layer_id, _ in sorted(self.layer_weights.items(), key=lambda item: item[1], reverse=True)]
        self._layer_schedule = self._build_layer_schedule()

        self.hook_handles = []
        num_layers = len(model.model.layers)
        for layer_id in self.layer_ids:
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(
                    f"frontdoor layer {layer_id} is out of range for model with {num_layers} layers"
                )
            target_layer = model.model.layers[layer_id]
            target_layer_cls = type(target_layer)
            handle = target_layer.register_forward_hook(self._make_block_only_hook(layer_id, target_layer_cls))
            self.hook_handles.append(handle)
            if debug:
                print(f"[Frontdoor] Hook attached -> Layer[{layer_id}] weight={self.layer_weights[layer_id]:.3f}")

    def _build_layer_schedule(self):
        if self.full_mode or len(self.layer_ids) <= self.sparse_layer_count:
            return self.layer_ids

        schedule = []
        for layer_id in self.layer_ids:
            slots = max(1, int(round(self.layer_weights[layer_id] * self.sparse_layer_schedule_bins)))
            schedule.extend([layer_id] * slots)
        return schedule or self.layer_ids

    def _select_active_layers(self):
        if self.full_mode or len(self.layer_ids) <= self.sparse_layer_count:
            return set(self.layer_ids)

        if self.sparse_layer_count == 1:
            layer_id = self._layer_schedule[self._forward_counter % len(self._layer_schedule)]
            return {layer_id}

        return set(self._sorted_layers[: self.sparse_layer_count])

    def begin_batch(self, batch_size, labels=None):
        self.batch_size = batch_size
        if labels is not None:
            self.labels = labels
        self._cached_token_weights = None
        self._active_layer_ids = self._select_active_layers()
        self._forward_counter += 1
        self._layer_directions = {}
        if self.debug:
            print(f"[Frontdoor] batch_size = {batch_size}")
            print(f"[Frontdoor] active_layers = {sorted(self._active_layer_ids)}")

    def disable(self):
        self.batch_size = None
        self._cached_token_weights = None
        self._active_layer_ids = set()
        self._layer_directions = {}
        self.clear_context()
        if self.debug:
            print("[Frontdoor] disabled")

    def set_context(self, labels=None, alpha_scale=None):
        if labels is not None:
            self.labels = labels
            self._cached_token_weights = None
        if alpha_scale is not None:
            self.alpha_scale = alpha_scale

    def clear_context(self):
        self.labels = None
        self.alpha_scale = None
        self._cached_token_weights = None
        self._layer_directions = {}

    def _make_block_only_hook(self, layer_id, target_layer_cls):
        def hook(module, inputs, outputs):
            if not isinstance(module, target_layer_cls):
                return outputs
            return self._hook_fn(layer_id, module, inputs, outputs)

        return hook

    def _token_weights(self, hidden_states):
        if self._cached_token_weights is not None:
            return self._cached_token_weights

        batch_size = self.batch_size
        seq_len = hidden_states.size(1)
        weights = hidden_states.new_ones((2 * batch_size, seq_len))

        if self.labels is None:
            self._cached_token_weights = weights
            return weights

        response_mask = (self.labels[: 2 * batch_size] != -100).to(hidden_states.dtype)

        if self.token_weight_mode == "uniform":
            self._cached_token_weights = weights
            return weights
        if self.token_weight_mode == "response_mask":
            self._cached_token_weights = response_mask
            return response_mask
        if self.token_weight_mode == "response_position":
            positions = torch.cumsum(response_mask, dim=1)
            counts = response_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            # Lightweight token-selective steering: deterministic position
            # weights over the response span, with slightly higher weight on
            # later response tokens and no auxiliary scoring pass.
            weights = response_mask * (1.0 + positions / counts)
            self._cached_token_weights = weights
            return weights
        if self.token_weight_mode == "response_norm":
            if not self.full_mode:
                self._cached_token_weights = response_mask
                return response_mask
            stats_hidden = hidden_states.detach() if self.detach_statistics else hidden_states
            if not self.full_mode and self.token_saliency_dim_stride > 1:
                stats_hidden = stats_hidden[..., :: self.token_saliency_dim_stride]
            saliency = stats_hidden.norm(dim=-1)
            weights = response_mask * saliency
            if not self.full_mode and 0.0 < self.token_topk_fraction < 1.0:
                k = max(1, min(seq_len, int(math.ceil(seq_len * self.token_topk_fraction))))
                topk_values, topk_indices = weights.topk(k, dim=1)
                sparse_weights = torch.zeros_like(weights)
                sparse_weights.scatter_(1, topk_indices, topk_values)
                weights = sparse_weights
            self._cached_token_weights = weights
            return weights

        raise ValueError(f"unknown token_weight_mode: {self.token_weight_mode}")

    @staticmethod
    def _weighted_mean(hidden_states, weights):
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (hidden_states * weights.unsqueeze(-1)).sum(dim=1) / denom

    def _normalize_direction(self, direction):
        if not self.normalize_direction:
            return direction
        norms = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return direction / norms

    def _resolve_direction(self, layer_id, direction):
        direction = self._normalize_direction(direction)
        if self.aggregate_mode != "layer_average":
            return direction

        self._layer_directions[layer_id] = direction
        if layer_id != self.apply_layer:
            return None

        ordered = [self._layer_directions[curr] for curr in self.layer_ids if curr in self._layer_directions]
        if not ordered:
            return None
        return torch.stack(ordered, dim=0).mean(dim=0)

    def _hook_fn(self, layer_id, module, inputs, outputs):
        if self.batch_size is None or layer_id not in self._active_layer_ids:
            return outputs

        out_is_tuple = isinstance(outputs, (tuple, list))
        hidden_states = outputs[0] if out_is_tuple else outputs

        if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 3:
            if self.debug:
                print("[Frontdoor] skip (invalid output shape)")
            return outputs

        batch_size = self.batch_size
        if hidden_states.size(0) < 2 * batch_size:
            if self.debug:
                print("[Frontdoor] skip (batch too small)")
            return outputs

        chosen_hidden = hidden_states[:batch_size]
        rejected_hidden = hidden_states[batch_size : 2 * batch_size]
        stats_hidden = hidden_states.detach() if self.detach_statistics else hidden_states

        weights = self._token_weights(stats_hidden)
        chosen_weights = weights[:batch_size]
        rejected_weights = weights[batch_size : 2 * batch_size]

        chosen_vector = self._weighted_mean(stats_hidden[:batch_size], chosen_weights)
        rejected_vector = self._weighted_mean(stats_hidden[batch_size : 2 * batch_size], rejected_weights)
        direction = torch.clamp(chosen_vector - rejected_vector, min=-30, max=30)
        direction = self._resolve_direction(layer_id, direction)
        if direction is None:
            return outputs

        alpha_scale = chosen_hidden.new_ones((batch_size, 1))
        if self.alpha_scale is not None:
            alpha_source = self.alpha_scale
            if not isinstance(alpha_source, torch.Tensor):
                alpha_source = torch.tensor(alpha_source, device=chosen_hidden.device, dtype=chosen_hidden.dtype)
            else:
                alpha_source = alpha_source.to(device=chosen_hidden.device, dtype=chosen_hidden.dtype)

            if alpha_source.dim() == 0:
                alpha_scale = alpha_source.reshape(1, 1).expand(batch_size, 1)
            else:
                alpha_scale = alpha_source[:batch_size].reshape(batch_size, 1)

        layer_weight = self.layer_weights[layer_id]
        scaled_direction = (self.alpha * layer_weight) * alpha_scale * direction.to(chosen_hidden.dtype)
        scaled_direction = scaled_direction.unsqueeze(1)

        if self.debug:
            print(f"[Frontdoor] HOOK @ layer {layer_id}")
            print(f"  u_norm = {direction.norm(dim=1).mean():.4f}")

        steered_hidden = hidden_states.clone()
        steered_hidden[:batch_size] = chosen_hidden + scaled_direction
        steered_hidden[batch_size : 2 * batch_size] = rejected_hidden - scaled_direction

        if out_is_tuple:
            return (steered_hidden, *outputs[1:])
        return steered_hidden
