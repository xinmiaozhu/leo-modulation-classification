"""DRC network for compensated I/Q and constellation descriptors."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
from torch import nn

from .constellation_stream import SymbolConstellationStream
from .constellation_stream import ConstellationImageStream, EVMFeatureStream
from .raw_stream import RawIQResNetStream, RawIQStream


@dataclass
class DRCDualNetConfig:
    hoc_dim: int
    evm_dim: int
    num_classes: int
    feature_dim: int = 128
    meta_dim: int = 3

    raw_stream_type: str = "cnn"
    raw_channels: tuple[int, ...] = (32, 64, 128)
    raw_kernel_sizes: tuple[int, ...] = (7, 5, 3)
    raw_resnet_blocks: int = 4

    hoc_hidden_dims: tuple[int, ...] = (128, 128)
    constellation_channels: tuple[int, ...] = (16, 32, 64)
    evm_hidden_dims: tuple[int, ...] = (64, 64)

    fusion_hidden_dim: int = 256
    fusion_type: str = "mlp"
    use_iq_stream: bool = True
    use_hoc_stream: bool = False
    use_metadata: bool = False
    use_constellation_image: bool = False
    use_evm_features: bool = True
    dropout: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DRCDualNet(nn.Module):
    """Configurable concat-fusion network for compensated I/Q and auxiliary features."""

    def __init__(
        self,
        hoc_dim: int,
        evm_dim: int,
        num_classes: int,
        feature_dim: int = 128,
        meta_dim: int = 3,
        raw_stream_type: str = "cnn",
        raw_channels: tuple[int, ...] = (32, 64, 128),
        raw_kernel_sizes: tuple[int, ...] = (7, 5, 3),
        raw_resnet_blocks: int = 4,
        hoc_hidden_dims: tuple[int, ...] = (128, 128),
        constellation_channels: tuple[int, ...] = (16, 32, 64),
        evm_hidden_dims: tuple[int, ...] = (64, 64),
        fusion_hidden_dim: int = 256,
        fusion_type: str = "mlp",
        use_iq_stream: bool = True,
        use_hoc_stream: bool = False,
        use_metadata: bool = False,
        use_constellation_image: bool = False,
        use_evm_features: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_iq_stream = bool(use_iq_stream)
        if use_hoc_stream:
            raise ValueError("DRCDualNet supports only I/Q and constellation streams.")
        if not (
            self.use_iq_stream
            or use_constellation_image
            or use_evm_features
        ):
            raise ValueError("At least one input stream must be enabled.")

        raw_stream_type = raw_stream_type.lower()
        if not self.use_iq_stream:
            self.raw_stream = None
        elif raw_stream_type == "cnn":
            self.raw_stream = RawIQStream(
                in_channels=2,
                feature_dim=feature_dim,
                channels=raw_channels,
                kernel_sizes=raw_kernel_sizes,
                dropout=dropout,
            )
        elif raw_stream_type == "resnet":
            self.raw_stream = RawIQResNetStream(
                in_channels=2,
                base_channels=raw_channels[-1] if len(raw_channels) > 0 else 64,
                num_blocks=raw_resnet_blocks,
                feature_dim=feature_dim,
                dropout=dropout,
            )
        else:
            raise ValueError("raw_stream_type must be 'cnn' or 'resnet'.")

        self.use_constellation_image = bool(use_constellation_image)
        self.use_evm_features = bool(use_evm_features)
        if self.use_constellation_image and self.use_evm_features:
            self.constellation_stream = SymbolConstellationStream(
                evm_dim=evm_dim,
                feature_dim=feature_dim,
                image_channels=constellation_channels,
                evm_hidden_dims=evm_hidden_dims,
                dropout=dropout,
            )
        elif self.use_constellation_image:
            self.constellation_stream = ConstellationImageStream(
                in_channels=1,
                feature_dim=feature_dim,
                channels=constellation_channels,
                dropout=dropout,
            )
        elif self.use_evm_features:
            self.constellation_stream = EVMFeatureStream(
                evm_dim=evm_dim,
                feature_dim=feature_dim,
                hidden_dims=evm_hidden_dims,
                dropout=dropout,
            )
        else:
            self.constellation_stream = None

        num_feature_streams = (
            int(self.use_iq_stream)
            + int(self.use_constellation_image or self.use_evm_features)
        )
        fusion_in = num_feature_streams * feature_dim + (meta_dim if use_metadata else 0)
        fusion_type = fusion_type.lower()
        if fusion_type == "mlp":
            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, fusion_hidden_dim),
                nn.BatchNorm1d(fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(fusion_hidden_dim, feature_dim),
                nn.ReLU(inplace=True),
            )
        elif fusion_type == "linear":
            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, feature_dim),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError("fusion_type must be 'mlp' or 'linear'.")
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feature_dim, num_classes),
        )

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.use_metadata = bool(use_metadata)

    def encode(
        self,
        iq: torch.Tensor | None,
        hoc: torch.Tensor | None,
        meta: torch.Tensor | None,
        constellation: torch.Tensor | None = None,
        evm_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        parts: list[torch.Tensor] = []
        out: dict[str, torch.Tensor] = {}
        if self.raw_stream is not None:
            if iq is None:
                raise ValueError("iq input is required when use_iq_stream=True.")
            f_raw = self.raw_stream(iq)
            parts.append(f_raw)
            out["f_raw"] = f_raw
        if self.constellation_stream is not None:
            if self.use_constellation_image and constellation is None:
                raise ValueError("constellation input is required when use_constellation_image=True.")
            if self.use_evm_features and evm_features is None:
                raise ValueError("evm_features input is required when use_evm_features=True.")
            if self.use_constellation_image and self.use_evm_features:
                f_const = self.constellation_stream(constellation, evm_features)
            elif self.use_constellation_image:
                f_const = self.constellation_stream(constellation)
            else:
                f_const = self.constellation_stream(evm_features)
            parts.append(f_const)
            out["f_constellation"] = f_const

        if self.use_metadata:
            if meta is None:
                raise ValueError("meta is required when use_metadata=True.")
            parts.append(meta)
        f_out = self.fusion(torch.cat(parts, dim=-1))
        out["f_out"] = f_out
        return out

    def forward(
        self,
        iq: torch.Tensor | None,
        hoc: torch.Tensor | None,
        meta: torch.Tensor | None = None,
        *,
        constellation: torch.Tensor | None = None,
        evm_features: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        aux = self.encode(iq, hoc, meta, constellation, evm_features)
        logits = self.classifier(aux["f_out"])
        if return_aux:
            aux["logits"] = logits
            return logits, aux
        return logits


def build_drc_dualnet(config: DRCDualNetConfig | dict[str, Any]) -> DRCDualNet:
    if isinstance(config, DRCDualNetConfig):
        kwargs = config.to_dict()
    else:
        kwargs = dict(config)
    return DRCDualNet(**kwargs)
