#!/usr/bin/env python3
"""Paper-aligned GlucoFM-like reimplementation; not Google-released code."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


LENGTH = 288
PATCH = 12
PATCHES = 24
DYNAMICS_TARGET_LAGS = (1, 3, 6, 12, 24)


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor) -> torch.Tensor:
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (-gradient,)


def reverse_gradient(values: torch.Tensor) -> torch.Tensor:
    return _ReverseGradient.apply(values)


def masked_mean_std(
    x: torch.Tensor, mask: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = mask.to(x.dtype)
    count = weight.sum(dim=dim, keepdim=True)
    mean = (x * weight).sum(dim=dim, keepdim=True) / count.clamp_min(1.0)
    var = ((x - mean).square() * weight).sum(dim=dim, keepdim=True) / count.clamp_min(
        1.0
    )
    valid = count.gt(0)
    return mean * valid, torch.sqrt(var + 1e-6) * valid


def scale_invariant_dynamics_targets(
    glucose: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return masked multiscale difference energy and autocorrelation targets."""
    observed = mask.to(glucose.dtype) * torch.isfinite(glucose).to(glucose.dtype)
    values = torch.nan_to_num(glucose)
    mean, std = masked_mean_std(values, observed, dim=1)
    normalized = ((values - mean) / std.clamp_min(1e-3)) * observed
    energy = []
    correlation = []
    for lag in DYNAMICS_TARGET_LAGS:
        valid = observed[:, lag:] * observed[:, :-lag]
        count = valid.sum(dim=1).clamp_min(1.0)
        difference = normalized[:, lag:] - normalized[:, :-lag]
        energy.append((difference.square() * valid).sum(dim=1) / count)
        correlation.append(
            (normalized[:, lag:] * normalized[:, :-lag] * valid).sum(dim=1) / count
        )
    return torch.stack(energy + correlation, dim=1)


class DualStreamEmbedder(nn.Module):
    def __init__(
        self,
        *,
        balance_components: bool = False,
        calibrate_physiological_units: bool = False,
        relative_auxiliary_features: bool = False,
        dual_reference_features: bool = False,
        absolute_reference_weight: float = 1.0,
        glucose_center: float = 0.0,
        glucose_scale: float = 1.0,
        rate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if glucose_scale <= 0 or rate_scale <= 0:
            raise ValueError("physiological feature scales must be positive")
        self.calibrate_physiological_units = calibrate_physiological_units
        self.relative_auxiliary_features = relative_auxiliary_features
        self.dual_reference_features = dual_reference_features
        if relative_auxiliary_features and dual_reference_features:
            raise ValueError(
                "relative-only and dual-reference auxiliary modes are exclusive"
            )
        if dual_reference_features and not calibrate_physiological_units:
            raise ValueError(
                "dual-reference features require physiological unit calibration"
            )
        if not 0.0 <= absolute_reference_weight <= 1.0:
            raise ValueError("absolute reference weight must be in [0, 1]")
        self.absolute_reference_weight = absolute_reference_weight
        self.register_buffer(
            "glucose_center", torch.tensor(glucose_center), persistent=False
        )
        self.register_buffer(
            "glucose_scale", torch.tensor(glucose_scale), persistent=False
        )
        self.register_buffer("rate_scale", torch.tensor(rate_scale), persistent=False)
        self.sigma_rho = nn.Parameter(
            torch.tensor(math.log((6.0 - 2.0) / (12.0 - 6.0)))
        )
        self.state_wave = nn.Linear(12, 64)
        self.state_diff = nn.Linear(11, 16)
        self.state_stats = nn.Linear(4 if dual_reference_features else 2, 48)
        self.state_token = nn.Linear(128, 64)
        self.event_wave = nn.Linear(12, 48)
        self.event_rate = nn.Linear(24 if dual_reference_features else 12, 48)
        self.event_stats = nn.Linear(4 if dual_reference_features else 2, 32)
        normalizer = nn.LayerNorm if balance_components else nn.Identity
        self.state_wave_norm = normalizer(64) if balance_components else normalizer()
        self.state_diff_norm = normalizer(16) if balance_components else normalizer()
        self.state_stats_norm = normalizer(48) if balance_components else normalizer()
        self.event_wave_norm = normalizer(48) if balance_components else normalizer()
        self.event_rate_norm = normalizer(48) if balance_components else normalizer()
        self.event_stats_norm = normalizer(32) if balance_components else normalizer()
        self.event_token = nn.Linear(128, 64)
        self.fusion = nn.Linear(128, 128)
        self.time_projection = nn.Linear(2, 128)
        self.position = nn.Parameter(torch.zeros(1, PATCHES, 128))
        self.time_gate = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.position, std=0.02)

    @property
    def sigma(self) -> torch.Tensor:
        return 2.0 + 10.0 * torch.sigmoid(self.sigma_rho)

    def causal_state(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lag = torch.arange(37, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-lag.square() / (2.0 * self.sigma.square()))
        kernel = kernel / kernel.sum()
        weight = kernel.flip(0).reshape(1, 1, -1)
        numerator = F.conv1d(F.pad((x * mask).unsqueeze(1), (36, 0)), weight).squeeze(1)
        denominator = F.conv1d(F.pad(mask.unsqueeze(1), (36, 0)), weight).squeeze(1)
        return numerator / denominator.clamp_min(1e-6)

    @staticmethod
    def rates(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rate = torch.zeros_like(x)
        rate_mask = torch.zeros_like(mask, dtype=torch.bool)
        observed = mask.bool()
        for lag in range(1, 10):
            eligible = observed[:, lag:] & observed[:, :-lag] & ~rate_mask[:, lag:]
            current = (x[:, lag:] - x[:, :-lag]) / float(lag)
            rate[:, lag:] = torch.where(eligible, current, rate[:, lag:])
            rate_mask[:, lag:] |= eligible
        return rate, rate_mask.to(x.dtype)

    def forward(
        self, glucose: torch.Tensor, mask: torch.Tensor, circadian_start: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = mask.to(glucose.dtype) * torch.isfinite(glucose).to(glucose.dtype)
        glucose = torch.nan_to_num(glucose)
        mean, std = masked_mean_std(glucose, mask, dim=1)
        normalized = ((glucose - mean) / std.clamp_min(1e-3)) * mask
        state = self.causal_state(normalized, mask) * mask
        event = (normalized - state) * mask
        # Appendix equations define physiological summary/rate features on the
        # aligned glucose signal in mg/dL, while only the Gaussian decomposition
        # is applied after mask-aware normalization.
        raw_rate, rate_mask = self.rates(glucose, mask)
        relative_rate, relative_rate_mask = self.rates(normalized, mask)

        state_patch = state.reshape(-1, PATCHES, PATCH)
        event_patch = event.reshape(-1, PATCHES, PATCH)
        raw_patch = glucose.reshape(-1, PATCHES, PATCH)
        mask_patch = mask.reshape(-1, PATCHES, PATCH)
        rate_patch = raw_rate.reshape(-1, PATCHES, PATCH)
        rate_mask_patch = rate_mask.reshape(-1, PATCHES, PATCH)
        relative_rate_patch = relative_rate.reshape(-1, PATCHES, PATCH)
        relative_rate_mask_patch = relative_rate_mask.reshape(-1, PATCHES, PATCH)
        relative_mean, relative_std = masked_mean_std(
            normalized.reshape(-1, PATCHES, PATCH), mask_patch, dim=2
        )
        relative_rate_mean, relative_rate_std = masked_mean_std(
            relative_rate_patch, relative_rate_mask_patch, dim=2
        )
        state_mean, state_std = masked_mean_std(raw_patch, mask_patch, dim=2)
        rate_mean, rate_std = masked_mean_std(rate_patch, rate_mask_patch, dim=2)
        if self.calibrate_physiological_units:
            state_mean = (state_mean - self.glucose_center) / self.glucose_scale
            state_std = state_std / self.glucose_scale
            rate_patch = rate_patch / self.rate_scale
            rate_mean = rate_mean / self.rate_scale
            rate_std = rate_std / self.rate_scale
        if self.dual_reference_features:
            state_stats_input = torch.cat(
                [
                    relative_mean,
                    relative_std,
                    self.absolute_reference_weight * state_mean,
                    self.absolute_reference_weight * state_std,
                ],
                dim=2,
            )
            event_rate_input = torch.cat(
                [relative_rate_patch, self.absolute_reference_weight * rate_patch],
                dim=2,
            )
            event_stats_input = torch.cat(
                [
                    relative_rate_mean,
                    relative_rate_std,
                    self.absolute_reference_weight * rate_mean,
                    self.absolute_reference_weight * rate_std,
                ],
                dim=2,
            )
        elif self.relative_auxiliary_features:
            state_stats_input = torch.cat([relative_mean, relative_std], dim=2)
            event_rate_input = relative_rate_patch
            event_stats_input = torch.cat(
                [relative_rate_mean, relative_rate_std], dim=2
            )
        else:
            state_stats_input = torch.cat([state_mean, state_std], dim=2)
            event_rate_input = rate_patch
            event_stats_input = torch.cat([rate_mean, rate_std], dim=2)
        state_aux = torch.cat(
            [
                self.state_wave_norm(self.state_wave(state_patch)),
                self.state_diff_norm(self.state_diff(torch.diff(state_patch, dim=2))),
                self.state_stats_norm(self.state_stats(state_stats_input)),
            ],
            dim=2,
        )
        event_aux = torch.cat(
            [
                self.event_wave_norm(self.event_wave(event_patch)),
                self.event_rate_norm(self.event_rate(event_rate_input)),
                self.event_stats_norm(self.event_stats(event_stats_input)),
            ],
            dim=2,
        )
        state_token = self.state_token(state_aux)
        event_token = self.event_token(event_aux)
        fused = self.fusion(torch.cat([state_token, event_token], dim=2))

        patch_index = torch.arange(PATCHES, device=glucose.device) * PATCH
        absolute = (circadian_start[:, None] + patch_index[None, :]) % LENGTH
        angle = 2.0 * math.pi * absolute.to(glucose.dtype) / LENGTH
        circular = torch.stack([torch.sin(angle), torch.cos(angle)], dim=2)
        temporal = self.position + torch.sigmoid(self.time_gate) * self.time_projection(
            circular
        )
        density = mask_patch.mean(dim=2)
        return fused + temporal, state_token, event_token, density


def transformer(layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(128))


class OnlineEncoder(nn.Module):
    def __init__(
        self,
        *,
        balance_components: bool = False,
        calibrate_physiological_units: bool = False,
        relative_auxiliary_features: bool = False,
        dual_reference_features: bool = False,
        absolute_reference_weight: float = 1.0,
        glucose_center: float = 0.0,
        glucose_scale: float = 1.0,
        rate_scale: float = 1.0,
        factorized_output: bool = False,
        normalize_dynamic_output: bool = False,
        cadence_adapter: bool = False,
        absolute_feature_center: torch.Tensor | None = None,
        absolute_feature_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.embedder = DualStreamEmbedder(
            balance_components=balance_components,
            calibrate_physiological_units=calibrate_physiological_units,
            relative_auxiliary_features=relative_auxiliary_features,
            dual_reference_features=dual_reference_features,
            absolute_reference_weight=absolute_reference_weight,
            glucose_center=glucose_center,
            glucose_scale=glucose_scale,
            rate_scale=rate_scale,
        )
        self.context = transformer(3)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 128))
        nn.init.normal_(self.mask_token, std=0.02)
        self.factorized_output = factorized_output
        self.normalize_dynamic_output = normalize_dynamic_output
        self.cadence_adapter_enabled = cadence_adapter
        if factorized_output:
            if absolute_feature_center is None:
                absolute_feature_center = torch.zeros(48)
            if absolute_feature_scale is None:
                absolute_feature_scale = torch.ones(48)
            if absolute_feature_center.shape != (
                48,
            ) or absolute_feature_scale.shape != (48,):
                raise ValueError("absolute feature calibration must have shape [48]")
            if (
                not torch.isfinite(absolute_feature_center).all()
                or not torch.isfinite(absolute_feature_scale).all()
            ):
                raise ValueError("absolute feature calibration must be finite")
            if not absolute_feature_scale.gt(0).all():
                raise ValueError("absolute feature scales must be positive")
            self.register_buffer(
                "absolute_feature_center",
                absolute_feature_center.to(torch.float32).clone(),
            )
            self.register_buffer(
                "absolute_feature_scale",
                absolute_feature_scale.to(torch.float32).clone(),
            )
            self.dynamic_encoder = nn.Linear(128, 127)
            self.dynamic_decoder = nn.Linear(127, 128)
            self.absolute_encoder = nn.Sequential(
                nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 1)
            )
            self.absolute_decoder = nn.Sequential(
                nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 48)
            )
            if cadence_adapter:
                self.cadence_adapter = nn.Sequential(
                    nn.Linear(127, 127), nn.GELU(), nn.Linear(127, 127)
                )
                nn.init.zeros_(self.cadence_adapter[-1].weight)
                nn.init.zeros_(self.cadence_adapter[-1].bias)
            with torch.no_grad():
                self.dynamic_encoder.weight.zero_()
                self.dynamic_encoder.weight[:, :127].copy_(torch.eye(127))
                self.dynamic_encoder.bias.zero_()
                self.dynamic_decoder.weight.zero_()
                self.dynamic_decoder.weight[:127].copy_(torch.eye(127))
                self.dynamic_decoder.bias.zero_()

    def absolute_features(
        self, glucose: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        values = torch.nan_to_num(glucose).reshape(-1, PATCHES, PATCH)
        observed = (
            mask.to(glucose.dtype) * torch.isfinite(glucose).to(glucose.dtype)
        ).reshape(-1, PATCHES, PATCH)
        mean, std = masked_mean_std(values, observed, dim=2)
        valid = observed.sum(dim=2).gt(0).to(glucose.dtype)
        features = torch.cat([mean.squeeze(2) * valid, std.squeeze(2) * valid], dim=1)
        return (features - self.absolute_feature_center) / self.absolute_feature_scale

    def factorized_embedding(
        self,
        context: torch.Tensor,
        glucose: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        dynamic = self.dynamic_embedding(context, mask)
        absolute = self.absolute_encoder(self.absolute_features(glucose, mask))
        return torch.cat([dynamic, absolute], dim=1)

    def dynamic_embedding(
        self, context: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        dynamic = self.dynamic_encoder(context.mean(dim=1))
        if self.normalize_dynamic_output:
            dynamic = F.layer_norm(dynamic, (dynamic.shape[1],))
        if self.cadence_adapter_enabled:
            if mask is None:
                raise ValueError("cadence adapter requires an observation mask")
            coverage = mask.to(dynamic.dtype).mean(dim=1, keepdim=True)
            sparse_gate = ((0.8 - coverage) / 0.4).clamp(0.0, 1.0)
            detached = dynamic.detach()
            adapted = detached + self.cadence_adapter(detached)
            dynamic = dynamic * (1.0 - sparse_gate) + adapted * sparse_gate
            if self.normalize_dynamic_output:
                dynamic = F.layer_norm(dynamic, (dynamic.shape[1],))
        return dynamic

    def forward(
        self,
        glucose: torch.Tensor,
        mask: torch.Tensor,
        circadian_start: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_mask = mask
        if patch_mask is not None:
            visible_points = (
                (~patch_mask).unsqueeze(2).expand(-1, -1, PATCH).reshape(-1, LENGTH)
            )
            branch_mask = mask * visible_points
        tokens, state, event, density = self.embedder(
            glucose, branch_mask, circadian_start
        )
        if patch_mask is not None:
            tokens = torch.where(patch_mask.unsqueeze(2), self.mask_token, tokens)
        return self.context(tokens), state, event, density

    def embed(
        self, glucose: torch.Tensor, mask: torch.Tensor, circadian_start: torch.Tensor
    ) -> torch.Tensor:
        encoded, _, _, _ = self(glucose, mask, circadian_start)
        if self.factorized_output:
            return self.factorized_embedding(encoded, glucose, mask)
        return encoded.mean(dim=1)


class GlucoFMLike(nn.Module):
    def __init__(
        self,
        *,
        balance_components: bool = False,
        calibrate_physiological_units: bool = False,
        relative_auxiliary_features: bool = False,
        dual_reference_features: bool = False,
        absolute_reference_weight: float = 1.0,
        glucose_center: float = 0.0,
        glucose_scale: float = 1.0,
        rate_scale: float = 1.0,
        variance_weight: float = 0.0,
        covariance_weight: float = 0.0,
        mean_shortcut_weight: float = 0.0,
        absolute_subspace_dim: int = 8,
        factorized_output: bool = False,
        normalize_dynamic_output: bool = False,
        cadence_adapter: bool = False,
        factorized_dynamic_weight: float = 0.0,
        factorized_absolute_weight: float = 0.0,
        absolute_feature_center: torch.Tensor | None = None,
        absolute_feature_scale: torch.Tensor | None = None,
        sparse_aware_masking: bool = False,
    ) -> None:
        super().__init__()
        if variance_weight < 0 or covariance_weight < 0 or mean_shortcut_weight < 0:
            raise ValueError(
                "representation regularization weights must be non-negative"
            )
        if not 0 <= absolute_subspace_dim < 128:
            raise ValueError("absolute subspace dimension must be in [0, 127]")
        if factorized_dynamic_weight < 0 or factorized_absolute_weight < 0:
            raise ValueError("factorized objective weights must be non-negative")
        if factorized_output and (
            factorized_dynamic_weight == 0 or factorized_absolute_weight == 0
        ):
            raise ValueError("factorized output requires both factorized objectives")
        if normalize_dynamic_output and not factorized_output:
            raise ValueError("dynamic output normalization requires factorized output")
        if cadence_adapter and not factorized_output:
            raise ValueError("cadence adapter requires factorized output")
        self.sparse_aware_masking = sparse_aware_masking
        self.balance_components = balance_components
        self.calibrate_physiological_units = calibrate_physiological_units
        self.relative_auxiliary_features = relative_auxiliary_features
        self.dual_reference_features = dual_reference_features
        self.absolute_reference_weight = absolute_reference_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.mean_shortcut_weight = mean_shortcut_weight
        self.absolute_subspace_dim = absolute_subspace_dim
        self.factorized_output = factorized_output
        self.normalize_dynamic_output = normalize_dynamic_output
        self.cadence_adapter = cadence_adapter
        self.factorized_dynamic_weight = factorized_dynamic_weight
        self.factorized_absolute_weight = factorized_absolute_weight
        self.online = OnlineEncoder(
            balance_components=balance_components,
            calibrate_physiological_units=calibrate_physiological_units,
            relative_auxiliary_features=relative_auxiliary_features,
            dual_reference_features=dual_reference_features,
            absolute_reference_weight=absolute_reference_weight,
            glucose_center=glucose_center,
            glucose_scale=glucose_scale,
            rate_scale=rate_scale,
            factorized_output=factorized_output,
            normalize_dynamic_output=normalize_dynamic_output,
            cadence_adapter=cadence_adapter,
            absolute_feature_center=absolute_feature_center,
            absolute_feature_scale=absolute_feature_scale,
        )
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        self.predictor = transformer(1)
        self.state_transition = nn.Sequential(
            nn.Linear(131, 128), nn.GELU(), nn.Linear(128, 64)
        )
        self.event_transition = nn.Sequential(
            nn.Linear(131, 128), nn.GELU(), nn.Linear(128, 64)
        )

    @staticmethod
    def sample_patch_mask(
        batch: int, device: torch.device, *, observed_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ratios = torch.empty(batch, device=device).uniform_(0.5, 0.6)
        counts = torch.ceil(ratios * PATCHES).to(torch.int64)
        order = torch.rand(batch, PATCHES, device=device).argsort(dim=1)
        mask = torch.zeros(batch, PATCHES, dtype=torch.bool, device=device)
        for row in range(batch):
            mask[row, order[row, : counts[row]]] = True
        if observed_mask is not None:
            # Keep the original dense-row policy and RNG draws. For sparse
            # rows, truncate the same random patch order by real-point budget,
            # not by the number of grid cells (most may already be missing).
            if observed_mask.shape != (batch, LENGTH):
                raise ValueError("expected [batch,288] physical observation mask")
            points = observed_mask.bool().reshape(batch, PATCHES, PATCH).sum(2)
            total = points.sum(1)
            ordered_points = points.gather(1, order)
            budget = total // 5  # remove at most 20% of real observations
            chosen = (
                (ordered_points.cumsum(1) <= budget[:, None])
                & (torch.arange(PATCHES, device=device)[None] < counts[:, None])
                & (ordered_points > 0)
            )
            sparse_mask = torch.zeros_like(mask).scatter(1, order, chosen)
            mask = torch.where((total <= 200)[:, None], sparse_mask, mask)
        return mask

    def pretraining_loss(
        self, glucose: torch.Tensor, mask: torch.Tensor, circadian_start: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        patch_mask = self.sample_patch_mask(
            len(glucose),
            glucose.device,
            observed_mask=mask if self.sparse_aware_masking else None,
        )
        physical_density = (
            mask.reshape(-1, PATCHES, PATCH).to(glucose.dtype).mean(dim=2)
        )
        online_context, online_state, online_event, _ = self.online(
            glucose, mask, circadian_start, patch_mask
        )
        with torch.no_grad():
            target_context, target_state, target_event, _ = self.target(
                glucose, mask, circadian_start
            )
        predicted_context = self.predictor(online_context)
        mcr_weight = patch_mask.to(glucose.dtype) * physical_density
        mcr_point = F.smooth_l1_loss(
            predicted_context, target_context, reduction="none"
        ).mean(dim=2)
        mcr = (mcr_point * mcr_weight).sum() / mcr_weight.sum().clamp_min(1e-6)

        patch_position = torch.arange(
            PATCHES, device=glucose.device, dtype=glucose.dtype
        )
        position = (
            (patch_position / max(1, PATCHES - 1))
            .reshape(1, PATCHES, 1)
            .expand(len(glucose), -1, -1)
        )
        absolute = (circadian_start[:, None] + patch_position[None, :] * PATCH) % LENGTH
        angle = 2.0 * math.pi * absolute / LENGTH
        time = torch.stack([torch.sin(angle), torch.cos(angle)], dim=2)
        state_input = torch.cat([online_state, online_event, position, time], dim=2)
        event_input = torch.cat([online_event, online_state, position, time], dim=2)
        state_prediction = online_state[:, :-1] + self.state_transition(
            state_input[:, :-1]
        )
        event_prediction = online_event[:, :-1] + self.event_transition(
            event_input[:, :-1]
        )
        transition_weight = (
            (~patch_mask[:, :-1]).to(glucose.dtype)
            * physical_density[:, :-1]
            * physical_density[:, 1:]
        )
        state_loss = F.smooth_l1_loss(
            state_prediction, target_state[:, 1:], reduction="none"
        ).mean(dim=2)
        event_loss = F.smooth_l1_loss(
            event_prediction, target_event[:, 1:], reduction="none"
        ).mean(dim=2)
        denominator = transition_weight.sum().clamp_min(1e-6)
        td = 0.5 * (
            (state_loss * transition_weight).sum() / denominator
            + (event_loss * transition_weight).sum() / denominator
        )
        pooled = online_context.mean(dim=1)
        dynamic_code = (
            self.online.dynamic_embedding(online_context, mask)
            if self.factorized_output
            else pooled
        )
        centered = dynamic_code - dynamic_code.mean(dim=0, keepdim=True)
        feature_std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
        variance = F.relu(0.2 - feature_std).mean()
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        covariance = covariance.fill_diagonal_(0).square().sum() / covariance.shape[0]
        finite_mask = mask.to(glucose.dtype) * torch.isfinite(glucose).to(glucose.dtype)
        daily_mean, _ = masked_mean_std(torch.nan_to_num(glucose), finite_mask, dim=1)
        daily_mean = daily_mean.squeeze(1)
        daily_mean = daily_mean - daily_mean.mean()
        dynamic = (
            centered
            if self.factorized_output
            else centered[:, self.absolute_subspace_dim :]
        )
        dynamic_std = torch.sqrt(dynamic.square().mean(dim=0) + 1e-4)
        daily_mean_std = torch.sqrt(daily_mean.square().mean() + 1e-4)
        dynamic_mean_correlation = (dynamic * daily_mean[:, None]).mean(dim=0) / (
            dynamic_std * daily_mean_std
        )
        mean_shortcut = dynamic_mean_correlation.square().mean()
        factorized_dynamic = torch.zeros((), device=glucose.device)
        factorized_absolute = torch.zeros((), device=glucose.device)
        if self.factorized_output:
            factorized_dynamic = F.smooth_l1_loss(
                self.online.dynamic_decoder(dynamic_code), target_context.mean(dim=1)
            )
            absolute_features = self.online.absolute_features(glucose, mask)
            absolute_code = self.online.absolute_encoder(absolute_features)
            factorized_absolute = F.mse_loss(
                self.online.absolute_decoder(absolute_code), absolute_features
            )
        total = (
            mcr
            + td
            + self.variance_weight * variance
            + self.covariance_weight * covariance
            + self.mean_shortcut_weight * mean_shortcut
            + self.factorized_dynamic_weight * factorized_dynamic
            + self.factorized_absolute_weight * factorized_absolute
        )
        return total, {
            "mcr": mcr.detach(),
            "td": td.detach(),
            "variance": variance.detach(),
            "covariance": covariance.detach(),
            "mean_shortcut": mean_shortcut.detach(),
            "factorized_dynamic": factorized_dynamic.detach(),
            "factorized_absolute": factorized_absolute.detach(),
        }

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for online, target in zip(self.online.parameters(), self.target.parameters()):
            target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)


def augment_cgm(
    glucose: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the paper's value and structural augmentations in randomized order."""
    output = glucose.clone()
    observed = mask.bool().clone()
    operations = torch.randperm(4).tolist()
    probability_scale = torch.ones(len(output), device=output.device)
    for operation in operations:
        base_probability = (0.25, 0.10, 0.40, 0.05)[operation]
        selected = (
            torch.rand(len(output), device=output.device)
            < base_probability * probability_scale
        )
        if operation == 2:
            selected &= observed.sum(dim=1) > 200
        for row in selected.nonzero(as_tuple=False).flatten().tolist():
            if operation == 0:
                amplitude = float(torch.empty(1, device=output.device).uniform_(5, 15))
                cycles = float(torch.empty(1, device=output.device).uniform_(0.5, 2.0))
                phase = float(
                    torch.empty(1, device=output.device).uniform_(0, 2 * math.pi)
                )
                grid = torch.arange(LENGTH, device=output.device, dtype=output.dtype)
                output[row] += (
                    amplitude
                    * torch.sin(2 * math.pi * cycles * grid / LENGTH + phase)
                    * observed[row]
                )
            elif operation == 1:
                width = int(torch.randint(6, 13, (1,), device=output.device))
                start = int(
                    torch.randint(0, LENGTH - width + 1, (1,), device=output.device)
                )
                minimum = float(torch.empty(1, device=output.device).uniform_(0.4, 0.7))
                position = torch.linspace(-1, 1, width, device=output.device).abs()
                multiplier = minimum + (1.0 - minimum) * position
                output[row, start : start + width] *= multiplier
            elif operation == 2:
                offset = int(torch.randint(0, 3, (1,), device=output.device))
                keep = torch.zeros(LENGTH, dtype=torch.bool, device=output.device)
                keep[offset::3] = True
                observed[row] &= keep
            else:
                blocks = int(torch.randint(1, 4, (1,), device=output.device))
                for _ in range(blocks):
                    width = int(torch.randint(2, 13, (1,), device=output.device))
                    start = int(
                        torch.randint(0, LENGTH - width + 1, (1,), device=output.device)
                    )
                    observed[row, start : start + width] = False
            probability_scale[row] *= 0.25
    output *= observed
    return output, observed.to(mask.dtype)


def counterfactual_sensor_view(
    glucose: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate sensor nuisance changes without pairing different people."""
    observed = mask.bool().clone()
    gain = torch.empty(len(glucose), 1, device=glucose.device).uniform_(0.95, 1.05)
    bias = torch.empty(len(glucose), 1, device=glucose.device).uniform_(-8.0, 8.0)
    noise_scale = torch.empty(len(glucose), 1, device=glucose.device).uniform_(0.0, 2.0)
    output = glucose * gain + bias + torch.randn_like(glucose) * noise_scale
    decimated = torch.rand(len(glucose), device=glucose.device) < 0.5
    offsets = torch.randint(0, 3, (len(glucose),), device=glucose.device)
    grid = torch.arange(LENGTH, device=glucose.device)
    for row in decimated.nonzero(as_tuple=False).flatten().tolist():
        observed[row] &= grid.remainder(3).eq(offsets[row])
    output *= observed
    return output, observed.to(mask.dtype)


def counterfactual_cadence_view(
    glucose: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decimate every trace from the 5-minute grid to one random 15-minute phase."""
    observed = mask.bool().clone()
    offsets = torch.randint(0, 3, (len(glucose),), device=glucose.device)
    grid = torch.arange(LENGTH, device=glucose.device)
    for row in range(len(glucose)):
        observed[row] &= grid.remainder(3).eq(offsets[row])
    return glucose * observed, observed.to(mask.dtype)
