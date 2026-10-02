"""Loss functions for DRC-DualNet training."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def compute_hoc_reliability_target(
    meta: torch.Tensor,
    a_gamma: float = 1.0,
    a_s: float = 1.0,
    a_v: float = 1.0,
) -> torch.Tensor:
    """Compute physics-guided expert reliability target R_hoc.

    Args:
        meta: Standardized metadata tensor [B, 3]:
            meta[:, 0] = gamma_hat
            meta[:, 1] = S_peak
            meta[:, 2] = V_hoc

    Returns:
        R_hoc: [B, 1], in [0, 1]

    Formula:
        R_gamma = sigmoid(a_gamma * gamma_hat)
        R_conf  = sigmoid(a_s * S_peak - a_v * V_hoc)
        R_hoc   = R_gamma * R_conf
    """

    if meta.ndim != 2 or meta.size(-1) < 3:
        raise ValueError(f"meta must have shape [B, >=3], got {tuple(meta.shape)}")

    gamma_hat = meta[:, 0:1]
    S_peak = meta[:, 1:2]
    V_hoc = meta[:, 2:3]

    R_gamma = torch.sigmoid(a_gamma * gamma_hat)
    R_conf = torch.sigmoid(a_s * S_peak - a_v * V_hoc)
    return R_gamma * R_conf


class GateConsistencyLoss(nn.Module):
    """Physics-guided gate consistency loss.

    If gate is vector-valued [B, D], its mean over feature dimensions is matched
    to R_hoc. If gate is scalar [B, 1], it is used directly.
    """

    def __init__(
        self,
        a_gamma: float = 1.0,
        a_s: float = 1.0,
        a_v: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.a_gamma = a_gamma
        self.a_s = a_s
        self.a_v = a_v
        self.reduction = reduction

    def forward(self, gate: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        if gate.ndim != 2:
            raise ValueError(f"gate must have shape [B, D] or [B, 1], got {gate.shape}")

        gate_mean = gate.mean(dim=-1, keepdim=True)
        target = compute_hoc_reliability_target(
            meta,
            a_gamma=self.a_gamma,
            a_s=self.a_s,
            a_v=self.a_v,
        ).detach()

        loss = (gate_mean - target) ** 2

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError("reduction must be 'mean', 'sum' or 'none'.")


class DRCTrainingLoss(nn.Module):
    """Combined classification and optional gate-consistency loss.

    Expected model output:
        logits, aux = model(..., return_aux=True)
        aux should contain key "gate" when lambda_gate > 0.
    """

    def __init__(
        self,
        lambda_gate: float = 0.0,
        class_weights: torch.Tensor | None = None,
        a_gamma: float = 1.0,
        a_s: float = 1.0,
        a_v: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_gate = float(lambda_gate)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.gate_loss = GateConsistencyLoss(a_gamma=a_gamma, a_s=a_s, a_v=a_v)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
        meta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cls_loss = self.ce(logits, labels)
        total = cls_loss
        logs: dict[str, torch.Tensor] = {"loss_cls": cls_loss.detach()}

        if self.lambda_gate > 0:
            if aux is None or "gate" not in aux:
                raise ValueError("aux['gate'] is required when lambda_gate > 0.")
            if meta is None:
                raise ValueError("meta is required when lambda_gate > 0.")

            lg = self.gate_loss(aux["gate"], meta)
            total = total + self.lambda_gate * lg
            logs["loss_gate"] = lg.detach()
        else:
            logs["loss_gate"] = torch.zeros((), device=logits.device)

        logs["loss_total"] = total.detach()
        return total, logs


class STARNetTrainingLoss(nn.Module):
    """Official STARNet multi-task weighting: 0.6 reconstruction + 0.4 CE."""

    def __init__(self, reconstruction_weight: float = 0.6) -> None:
        super().__init__()
        self.reconstruction_weight = float(reconstruction_weight)

    def forward(self, logits, labels, aux=None, meta=None):
        if aux is None or "reconstruction" not in aux or "reconstruction_target" not in aux:
            raise ValueError("STARNet loss requires reconstruction tensors in model aux output.")
        loss_cls = F.cross_entropy(logits, labels)
        loss_reconstruction = F.mse_loss(aux["reconstruction"], aux["reconstruction_target"])
        total = self.reconstruction_weight * loss_reconstruction + (1.0 - self.reconstruction_weight) * loss_cls
        return total, {
            "loss_cls": loss_cls.detach(),
            "loss_gate": loss_reconstruction.detach(),
            "loss_total": total.detach(),
        }


class CenterLoss(nn.Module):
    """Center loss for optional feature compactness experiments."""

    def __init__(self, num_classes: int, feature_dim: int, alpha: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers_batch = self.centers[labels]
        return 0.5 * torch.mean(torch.sum((features - centers_batch) ** 2, dim=1))


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Supervised contrastive loss for optional experiments."""

    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, features.T) / temperature

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)

    pos_count = mask.sum(dim=1)
    loss = -(mask * log_prob).sum(dim=1) / (pos_count + eps)
    valid = pos_count > 0
    if valid.any():
        return loss[valid].mean()
    return torch.zeros((), device=features.device)
