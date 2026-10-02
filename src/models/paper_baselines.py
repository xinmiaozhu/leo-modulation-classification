"""Paper-aligned baselines for the external AMC comparison.

The source papers use shorter observations than this project's 8192-sample
frames.  The temporal models therefore operate on their native window length
and aggregate window-level logits.  This preserves local waveform structure;
it does not resize the full frame with average pooling.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class _PaperBaseline(nn.Module):
    """Unified forward/diagnostic interface used by Trainer and Evaluator."""

    @staticmethod
    def _result(
        logits: torch.Tensor,
        features: torch.Tensor,
        return_aux: bool,
        **diagnostics: torch.Tensor,
    ):
        if return_aux:
            aux = {"f_out": features, "logits": logits}
            aux.update(diagnostics)
            return logits, aux
        return logits


class _WindowedIQBaseline(_PaperBaseline):
    """Apply a native-length AMC model to an entire long observation."""

    def __init__(
        self,
        input_length: int,
        train_windows: int = 4,
        window_batch_size: int = 512,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.train_windows = int(train_windows)
        self.window_batch_size = max(int(window_batch_size), 1)
        if self.input_length <= 0:
            raise ValueError("input_length must be positive.")

    def _make_windows(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected compensated I/Q [B, 2, N], got {tuple(iq.shape)}")

        length = self.input_length
        n = int(iq.shape[-1])
        if n < length:
            iq = F.pad(iq, (0, length - n))
            n = length

        starts = list(range(0, n - length + 1, length))
        final_start = n - length
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        windows = torch.stack([iq[..., start : start + length] for start in starts], dim=1)

        if self.training and self.train_windows > 0 and windows.shape[1] > self.train_windows:
            selected = torch.randperm(windows.shape[1], device=iq.device)[: self.train_windows]
            selected, _ = torch.sort(selected)
            windows = windows.index_select(1, selected)
        return windows

    def _window_forward(
        self,
        iq: torch.Tensor,
        encoder: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        windows = self._make_windows(iq)
        batch_size, num_windows = windows.shape[:2]
        flat = windows.reshape(batch_size * num_windows, 2, self.input_length)

        logits_parts: list[torch.Tensor] = []
        feature_parts: list[torch.Tensor] = []
        for start in range(0, flat.shape[0], self.window_batch_size):
            features, logits = encoder(flat[start : start + self.window_batch_size])
            feature_parts.append(features)
            logits_parts.append(logits)

        window_features = torch.cat(feature_parts, dim=0).reshape(batch_size, num_windows, -1)
        window_logits = torch.cat(logits_parts, dim=0).reshape(batch_size, num_windows, -1)
        return (
            window_logits.mean(dim=1),
            window_features.mean(dim=1),
            torch.tensor(num_windows, device=iq.device),
        )


class CNN2Baseline(_WindowedIQBaseline):
    """O'Shea et al. CNN2 with its original 2 x 128 input geometry."""

    def __init__(
        self,
        num_classes: int,
        input_length: int = 128,
        feature_dim: int = 256,
        dropout: float = 0.5,
        train_windows: int = 4,
        window_batch_size: int = 512,
    ) -> None:
        if int(input_length) != 128:
            raise ValueError("CNN2's paper-aligned input_length must be 128.")
        super().__init__(input_length, train_windows, window_batch_size)
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3))
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(80 * 132, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _encode_window(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = iq.unsqueeze(1)
        x = F.relu(self.conv1(F.pad(x, (2, 2, 0, 0))), inplace=True)
        x = self.dropout(x)
        x = F.relu(self.conv2(F.pad(x, (2, 2, 0, 0))), inplace=True)
        x = self.dropout(x)
        features = F.relu(self.dense(x.flatten(1)), inplace=True)
        features = self.dropout(features)
        return features, self.classifier(features)

    def forward(self, iq: torch.Tensor, *args, return_aux: bool = False, **kwargs):
        logits, features, num_windows = self._window_forward(iq, self._encode_window)
        return self._result(logits, features, return_aux, num_windows=num_windows)


def _time_pool(x: torch.Tensor, mode: str = "max") -> torch.Tensor:
    if mode == "avg":
        return F.avg_pool2d(x, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
    return F.max_pool2d(x, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))


def _mcnet_residual_downsample(x: torch.Tensor) -> torch.Tensor:
    # MATLAB's asymmetric [1 0 0 0] padding preserves the two-row I/Q axis.
    x = F.pad(x, (0, 0, 0, 1))
    return F.max_pool2d(x, kernel_size=(2, 2), stride=(1, 2))


class _MCNetMBlock(nn.Module):
    """MCNet M-block with optional temporal downsampling."""

    def __init__(
        self,
        in_channels: int = 128,
        downsample: bool = False,
        residual: bool = True,
    ) -> None:
        super().__init__()
        stride = (1, 2) if downsample else (1, 1)
        self.downsample = bool(downsample)
        self.residual = bool(residual)
        self.reduce = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.vertical = nn.Conv2d(32, 48, kernel_size=(3, 1), padding=(1, 0))
        self.horizontal = nn.Conv2d(
            32,
            48,
            kernel_size=(1, 3),
            stride=stride,
            padding=(0, 1),
        )
        self.pointwise = nn.Conv2d(32, 32, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = F.relu(self.reduce(x), inplace=True)
        vertical = F.relu(self.vertical(reduced), inplace=True)
        if self.downsample:
            vertical = _time_pool(vertical, mode="avg")
        horizontal = F.relu(self.horizontal(reduced), inplace=True)
        pointwise = F.relu(self.pointwise(reduced), inplace=True)
        mixed = torch.cat([vertical, horizontal, pointwise], dim=1)
        if not self.residual:
            return mixed
        residual = _mcnet_residual_downsample(x) if self.downsample else x
        return F.relu(mixed + residual, inplace=True)


class _MCNetFinalBlock(nn.Module):
    """Final wider MCNet block (96 + 96 + 64 = 256 channels)."""

    def __init__(self) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(128, 32, kernel_size=1)
        self.vertical = nn.Conv2d(32, 96, kernel_size=(3, 1), padding=(1, 0))
        self.horizontal = nn.Conv2d(32, 96, kernel_size=(1, 3), padding=(0, 1))
        self.pointwise = nn.Conv2d(32, 64, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.reduce(x), inplace=True)
        return torch.cat(
            [
                F.relu(self.vertical(x), inplace=True),
                F.relu(self.horizontal(x), inplace=True),
                F.relu(self.pointwise(x), inplace=True),
            ],
            dim=1,
        )


class MCNetBaseline(_WindowedIQBaseline):
    """PyTorch reconstruction of the authors' MCNet layer graph."""

    def __init__(
        self,
        num_classes: int,
        input_length: int = 1024,
        feature_dim: int = 384,
        dropout: float = 0.0,
        train_windows: int = 4,
        window_batch_size: int = 256,
        **unused,
    ) -> None:
        if int(input_length) != 1024:
            raise ValueError("MCNet's paper-aligned input_length must be 1024.")
        if int(feature_dim) != 384:
            raise ValueError("MCNet's final concatenated feature dimension is 384.")
        super().__init__(input_length, train_windows, window_batch_size)
        self.stem = nn.Conv2d(
            1,
            64,
            kernel_size=(3, 7),
            stride=(1, 2),
            padding=(1, 3),
        )
        self.pre_vertical = nn.Conv2d(64, 32, kernel_size=(3, 1), padding=(1, 0))
        self.pre_horizontal = nn.Conv2d(
            64,
            32,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
        )
        self.jump = nn.Conv2d(64, 128, kernel_size=1, stride=(1, 2))
        self.block_a = _MCNetMBlock(
            in_channels=64,
            downsample=True,
            residual=False,
        )
        self.block_b = _MCNetMBlock(downsample=False)
        self.block_c = _MCNetMBlock(downsample=True)
        self.block_d = _MCNetMBlock(downsample=False)
        self.block_e = _MCNetMBlock(downsample=True)
        self.block_f = _MCNetFinalBlock()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(384, num_classes)

    def _encode_window(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.stem(iq.unsqueeze(1)), inplace=True)
        x = _time_pool(x, mode="max")

        vertical = _time_pool(F.relu(self.pre_vertical(x), inplace=True), mode="avg")
        horizontal = F.relu(self.pre_horizontal(x), inplace=True)
        pre = torch.cat([vertical, horizontal], dim=1)

        jump = _time_pool(F.relu(self.jump(pre), inplace=True), mode="max")
        main = self.block_a(_time_pool(pre, mode="max"))
        main = F.relu(main + jump, inplace=True)
        main = self.block_b(main)
        main = self.block_c(main)
        main = self.block_d(main)
        block_e = self.block_e(main)
        block_f = self.block_f(block_e)

        merged = torch.cat([block_e, block_f], dim=1)
        features = F.adaptive_avg_pool2d(merged, output_size=(1, 1)).flatten(1)
        features = self.dropout(features)
        return features, self.classifier(features)

    def forward(self, iq: torch.Tensor, *args, return_aux: bool = False, **kwargs):
        logits, features, num_windows = self._window_forward(iq, self._encode_window)
        return self._result(logits, features, return_aux, num_windows=num_windows)


class _CNNLSTMRepresentationStream(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = 64 + hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.cnn(x)
        spatial = temporal.mean(dim=-1)
        _, (hidden, _) = self.lstm(temporal.transpose(1, 2))
        return self.dropout(torch.cat([spatial, hidden[-1]], dim=1))


class CNNLSTMDualStreamBaseline(_WindowedIQBaseline):
    """I/Q and amplitude/phase CNN-LSTM streams with pairwise interaction."""

    def __init__(
        self,
        num_classes: int,
        input_length: int = 128,
        feature_dim: int = 128,
        dropout: float = 0.2,
        lstm_hidden: int = 64,
        train_windows: int = 4,
        window_batch_size: int = 512,
    ) -> None:
        super().__init__(input_length, train_windows, window_batch_size)
        self.iq_stream = _CNNLSTMRepresentationStream(lstm_hidden, dropout)
        self.ap_stream = _CNNLSTMRepresentationStream(lstm_hidden, dropout)
        stream_dim = self.iq_stream.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(4 * stream_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    @staticmethod
    def _amplitude_phase(iq: torch.Tensor) -> torch.Tensor:
        i = iq[:, 0:1]
        q = iq[:, 1:2]
        amplitude = torch.sqrt(i.square() + q.square() + 1e-12)
        amplitude = amplitude / (amplitude.square().mean(dim=-1, keepdim=True).sqrt() + 1e-8)
        phase = torch.atan2(q, i) / torch.pi
        return torch.cat([amplitude, phase], dim=1)

    def _encode_window(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_iq = self.iq_stream(iq)
        f_ap = self.ap_stream(self._amplitude_phase(iq))
        interaction = torch.cat(
            [f_iq, f_ap, f_iq * f_ap, torch.abs(f_iq - f_ap)],
            dim=1,
        )
        features = self.fusion(interaction)
        return features, self.classifier(features)

    def forward(self, iq: torch.Tensor, *args, return_aux: bool = False, **kwargs):
        logits, features, num_windows = self._window_forward(iq, self._encode_window)
        return self._result(logits, features, return_aux, num_windows=num_windows)


class SatelliteLightCNNBaseline(_WindowedIQBaseline):
    """Satellite AMC CNN with learned I/Q correlation and temporal features."""

    def __init__(
        self,
        num_classes: int,
        input_length: int = 1024,
        feature_dim: int = 96,
        dropout: float = 0.2,
        train_windows: int = 4,
        window_batch_size: int = 512,
    ) -> None:
        super().__init__(input_length, train_windows, window_batch_size)
        self.iq_correlation = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 7), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Linear(96, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _encode_window(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        correlation = self.iq_correlation(iq.unsqueeze(1)).squeeze(2)
        temporal = self.temporal(correlation)
        features = self.project(temporal.mean(dim=-1))
        return features, self.classifier(features)

    def forward(self, iq: torch.Tensor, *args, return_aux: bool = False, **kwargs):
        logits, features, num_windows = self._window_forward(iq, self._encode_window)
        return self._result(logits, features, return_aux, num_windows=num_windows)


class _SameConv2d(nn.Module):
    """TensorFlow/Keras ``padding='same'`` convolution, including stride > 1."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, **kwargs) -> None:
        super().__init__()
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.conv = nn.Conv2d(
            in_channels, out_channels, self.kernel_size, stride=self.stride, padding=0, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        out_h = (height + self.stride[0] - 1) // self.stride[0]
        out_w = (width + self.stride[1] - 1) // self.stride[1]
        pad_h = max((out_h - 1) * self.stride[0] + self.kernel_size[0] - height, 0)
        pad_w = max((out_w - 1) * self.stride[1] + self.kernel_size[1] - width, 0)
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        return self.conv(x)


class _SeparableConv2d(nn.Module):
    def __init__(self, channels: int, kernel_size=(1, 3)) -> None:
        super().__init__()
        self.depthwise = _SameConv2d(channels, channels, kernel_size, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _STARAttention(nn.Module):
    """Channel and temporal attention used by the official attentive-ghost blocks."""

    def __init__(self, channels: int, ratio: int = 128) -> None:
        super().__init__()
        hidden = max(1, channels // ratio)
        self.channel = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False), nn.Sigmoid(),
        )
        self.spatial = _SameConv2d(2, 1, kernel_size=(1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.channel(x.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
        x = x * weights
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(pooled))


class STARNetBaseline(_WindowedIQBaseline):
    """PyTorch port of the authors' 2024 TWC STARNet implementation.

    The official network consumes 128-point amplitude/phase records and jointly
    reconstructs the unmasked record and classifies it.  Here an 8192-point
    compensated-I/Q frame is handled by the same window/logit aggregation rule
    used for the other native-length paper baselines.
    """

    def __init__(
        self,
        num_classes: int,
        input_length: int = 128,
        train_windows: int = 4,
        window_batch_size: int = 512,
        mask_probability: float = 0.1,
        **unused,
    ) -> None:
        if int(input_length) != 128:
            raise ValueError("STARNet's official input_length is 128.")
        super().__init__(input_length, train_windows, window_batch_size)
        self.mask_probability = float(mask_probability)
        self.input_bn = nn.BatchNorm1d(2)
        self.stem = _SameConv2d(1, 4, kernel_size=(2, 7), stride=2)

        self.skip1 = _SameConv2d(4, 8, 1, stride=(1, 2))
        self.reduce1 = nn.Conv2d(4, 4, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.sep1 = _SeparableConv2d(4)
        self.attn1 = _STARAttention(4)
        self.down1 = _SameConv2d(4, 4, 1, stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(4)
        self.sep2 = _SeparableConv2d(4)

        self.skip2 = _SameConv2d(8, 16, 1, stride=(1, 2))
        self.reduce2 = nn.Conv2d(8, 8, 1)
        self.bn3 = nn.BatchNorm2d(8)
        self.sep3 = _SeparableConv2d(8)
        self.attn2 = _STARAttention(8)
        self.down2 = _SameConv2d(8, 8, 1, stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(8)
        self.sep4 = _SeparableConv2d(8)

        self.skip3 = _SameConv2d(16, 32, 1, stride=(1, 2))
        self.reduce3 = nn.Conv2d(16, 8, 1)
        self.bn5 = nn.BatchNorm2d(8)
        self.sep5 = _SeparableConv2d(8)
        self.attn3 = _STARAttention(8)
        self.down3 = _SameConv2d(8, 16, 1, stride=(1, 2))
        self.bn6 = nn.BatchNorm2d(16)
        self.sep6 = _SeparableConv2d(16)

        self.gru1 = nn.GRU(2, 32, batch_first=True)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        self.reconstruction = nn.Conv2d(1, 4, kernel_size=1)
        self.classifier_bn = nn.BatchNorm1d(64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(inplace=True), nn.BatchNorm1d(32), nn.Dropout(0.01),
            nn.Linear(32, 16), nn.ReLU(inplace=True), nn.BatchNorm1d(16), nn.Dropout(0.01),
            nn.Linear(16, num_classes),
        )

    @staticmethod
    def _amplitude_phase(iq: torch.Tensor) -> torch.Tensor:
        amplitude = torch.linalg.vector_norm(iq, dim=1)
        amplitude = amplitude / torch.linalg.vector_norm(amplitude, dim=1, keepdim=True).clamp_min(1e-8)
        phase = torch.atan2(iq[:, 1], iq[:, 0]) / torch.pi
        return torch.stack([amplitude, phase], dim=-1)  # [B, T, 2]

    def _cnn_branch(self, ap: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(ap.transpose(1, 2).unsqueeze(1)), inplace=True)
        skip = self.skip1(x)
        x = F.relu(self.reduce1(x), inplace=True)
        residual = self.bn1(x)
        x = F.relu(self.sep1(x) + residual, inplace=True)
        x = self.attn1(x)
        x = self.down1(x)
        residual = self.bn2(x)
        x = torch.cat([self.sep2(x), residual], dim=1) + skip

        skip = self.skip2(x)
        x = F.relu(self.reduce2(x), inplace=True)
        residual = self.bn3(x)
        x = F.relu(self.sep3(x) + residual, inplace=True)
        x = self.attn2(x)
        x = self.down2(x)
        residual = self.bn4(x)
        x = torch.cat([self.sep4(x), residual], dim=1) + skip

        skip = self.skip3(x)
        x = F.relu(self.reduce3(x), inplace=True)
        residual = self.bn5(x)
        x = F.relu(self.sep5(x) + residual, inplace=True)
        x = self.attn3(x)
        x = self.down3(x)
        residual = self.bn6(x)
        x = torch.cat([self.sep6(x), residual], dim=1) + skip
        return x.mean(dim=(-2, -1))

    def _encode_windows(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self._amplitude_phase(iq)
        model_input = target
        if self.training and self.mask_probability > 0:
            mask = torch.rand(target.shape[:2], device=target.device) < self.mask_probability
            model_input = target.masked_fill(mask.unsqueeze(-1), 0.0)
        normalized = self.input_bn(model_input.transpose(1, 2)).transpose(1, 2)
        cnn_features = self._cnn_branch(normalized)
        sequence, _ = self.gru1(normalized)
        _, state = self.gru2(sequence)
        features = torch.cat([0.7 * state[-1], 0.3 * cnn_features], dim=1)
        logits = self.classifier(self.classifier_bn(features))
        reconstruction = self.reconstruction(features.reshape(-1, 1, 64, 1))
        reconstruction = reconstruction.flatten(1).reshape(-1, 128, 2)
        return features, logits, reconstruction, target

    def forward(self, iq: torch.Tensor, *args, return_aux: bool = False, **kwargs):
        windows = self._make_windows(iq)
        batch_size, num_windows = windows.shape[:2]
        flat = windows.reshape(batch_size * num_windows, 2, self.input_length)
        feature_parts, logit_parts, recon_parts, target_parts = [], [], [], []
        for start in range(0, flat.shape[0], self.window_batch_size):
            features, logits, reconstruction, target = self._encode_windows(
                flat[start : start + self.window_batch_size]
            )
            feature_parts.append(features)
            logit_parts.append(logits)
            recon_parts.append(reconstruction)
            target_parts.append(target)
        features = torch.cat(feature_parts).reshape(batch_size, num_windows, -1).mean(dim=1)
        logits = torch.cat(logit_parts).reshape(batch_size, num_windows, -1).mean(dim=1)
        if return_aux:
            return logits, {
                "f_out": features,
                "logits": logits,
                "reconstruction": torch.cat(recon_parts),
                "reconstruction_target": torch.cat(target_parts),
                "num_windows": torch.tensor(num_windows, device=iq.device),
            }
        return logits


class NASAHOCNetBaseline(_PaperBaseline):
    """NASA 10-40-40-40-7 cumulant/SNR multilayer perceptron."""

    def __init__(
        self,
        hoc_dim: int,
        num_classes: int,
        feature_dim: int = 40,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(hoc_dim) != 10:
            raise ValueError(f"NASA HOC-NN requires 10 input features, got {hoc_dim}.")
        if int(feature_dim) != 40:
            raise ValueError("NASA HOC-NN uses three hidden layers of width 40.")
        layers: list[nn.Module] = [
            nn.Linear(10, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(40, num_classes)

    def forward(
        self,
        iq: torch.Tensor,
        hoc: torch.Tensor,
        *args,
        return_aux: bool = False,
        **kwargs,
    ):
        if hoc is None:
            raise ValueError("NASAHOCNetBaseline requires NASA's 10-D HOC/SNR vector.")
        features = self.encoder(hoc)
        return self._result(self.classifier(features), features, return_aux)


PAPER_BASELINE_MODELS = {
    "paper_cnn2": CNN2Baseline,
    "paper_mcnet": MCNetBaseline,
    "paper_cnn_lstm_dual": CNNLSTMDualStreamBaseline,
    "paper_satellite_cnn": SatelliteLightCNNBaseline,
    "paper_starnet": STARNetBaseline,
    "paper_nasa_hoc_nn": NASAHOCNetBaseline,
}


def build_paper_baseline(
    model_name: str,
    *,
    num_classes: int,
    hoc_dim: int | None = None,
    **model_config,
) -> nn.Module:
    """Build one of the registered external-comparison baselines."""

    if model_name not in PAPER_BASELINE_MODELS:
        raise ValueError(f"Unknown paper baseline: {model_name}")
    cls = PAPER_BASELINE_MODELS[model_name]
    if model_name == "paper_nasa_hoc_nn":
        if hoc_dim is None:
            raise ValueError("paper_nasa_hoc_nn requires hoc_dim.")
        return cls(hoc_dim=hoc_dim, num_classes=num_classes, **model_config)
    return cls(num_classes=num_classes, **model_config)
