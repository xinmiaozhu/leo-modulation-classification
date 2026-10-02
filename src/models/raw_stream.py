"""Raw I/Q stream encoders.

Input convention:
    iq: Tensor with shape [B, 2, N]

The stream outputs a fixed-dimensional representation f_raw with shape [B, D].
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock1D(nn.Module):
    """Conv1d + BatchNorm + activation + optional pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        pool: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool is not None and pool > 1:
            layers.append(nn.MaxPool1d(pool))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RawIQStream(nn.Module):
    """Lightweight CNN stream for raw I/Q sequences.

    This is the default raw stream used by the paper receiver. It is deliberately
    lightweight enough for edge-oriented experiments, while still stronger than
    a very shallow baseline.
    """

    def __init__(
        self,
        in_channels: int = 2,
        feature_dim: int = 128,
        channels: tuple[int, ...] = (32, 64, 128),
        kernel_sizes: tuple[int, ...] = (7, 5, 3),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if len(channels) != len(kernel_sizes):
            raise ValueError("channels and kernel_sizes must have the same length.")

        blocks: list[nn.Module] = []
        c_in = in_channels
        for idx, (c_out, k) in enumerate(zip(channels, kernel_sizes)):
            blocks.append(
                ConvBlock1D(
                    c_in,
                    c_out,
                    kernel_size=k,
                    pool=2 if idx < len(channels) - 1 else None,
                    dropout=dropout if idx > 0 else 0.0,
                )
            )
            c_in = c_out

        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        """Encode I/Q sequence.

        Args:
            iq: [B, 2, N]

        Returns:
            f_raw: [B, feature_dim]
        """

        z = self.encoder(iq)
        z = self.pool(z).squeeze(-1)
        return self.proj(z)


class ResidualBlock1D(nn.Module):
    """Basic 1D residual block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        y = self.dropout(y)
        return self.act(x + y)


class RawIQResNetStream(nn.Module):
    """Stronger raw-IQ 1D-ResNet stream for baseline comparisons."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_blocks: int = 4,
        feature_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock1D(base_channels, kernel_size=3, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(base_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        z = self.stem(iq)
        z = self.blocks(z)
        z = self.pool(z).squeeze(-1)
        return self.proj(z)
