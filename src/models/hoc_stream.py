"""DRC-HOC expert feature stream."""

from __future__ import annotations

import torch
from torch import nn


class HOCStream(nn.Module):
    """MLP encoder for DRC-HOC expert features.

    Input:
        hoc: [B, D_hoc]

    Output:
        f_hoc: [B, feature_dim]
    """

    def __init__(
        self,
        hoc_dim: int,
        feature_dim: int = 128,
        hidden_dims: tuple[int, ...] = (128, 128),
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if hoc_dim <= 0:
            raise ValueError("hoc_dim must be positive.")

        layers: list[nn.Module] = []
        in_dim = hoc_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden

        layers.extend(
            [
                nn.Linear(in_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, hoc: torch.Tensor) -> torch.Tensor:
        return self.net(hoc)
