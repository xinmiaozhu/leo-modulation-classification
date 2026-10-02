"""Symbol-level constellation feature stream."""

from __future__ import annotations

import torch
from torch import nn


class ConstellationImageStream(nn.Module):
    """Small 2D CNN encoder for constellation density images."""

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 128,
        channels: tuple[int, ...] = (16, 32, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers.extend(
                [
                    nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            c_in = c_out

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(c_in, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.encoder(image)
        return self.head(x)


class EVMFeatureStream(nn.Module):
    """MLP encoder for all-candidate EVM and radial statistics."""

    def __init__(
        self,
        evm_dim: int,
        feature_dim: int = 64,
        hidden_dims: tuple[int, ...] = (64, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(evm_dim)
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden)),
                    nn.BatchNorm1d(int(hidden)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
            in_dim = int(hidden)
        layers.append(nn.Linear(in_dim, feature_dim))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, evm_features: torch.Tensor) -> torch.Tensor:
        return self.net(evm_features)


class SymbolConstellationStream(nn.Module):
    """Joint encoder for constellation image and all-candidate EVM features."""

    def __init__(
        self,
        evm_dim: int,
        feature_dim: int = 128,
        image_channels: tuple[int, ...] = (16, 32, 64),
        evm_hidden_dims: tuple[int, ...] = (64, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        image_dim = feature_dim
        evm_out_dim = max(32, feature_dim // 2)
        self.image_stream = ConstellationImageStream(
            in_channels=1,
            feature_dim=image_dim,
            channels=image_channels,
            dropout=dropout,
        )
        self.evm_stream = EVMFeatureStream(
            evm_dim=evm_dim,
            feature_dim=evm_out_dim,
            hidden_dims=evm_hidden_dims,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + evm_out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor, evm_features: torch.Tensor) -> torch.Tensor:
        f_img = self.image_stream(image)
        f_evm = self.evm_stream(evm_features)
        return self.fusion(torch.cat([f_img, f_evm], dim=-1))
