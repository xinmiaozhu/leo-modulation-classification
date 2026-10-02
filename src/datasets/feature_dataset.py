"""PyTorch dataset for raw I/Q + DRC-HOC feature samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.split import load_split_indices


def compute_metadata_stats(
    feature_h5_path: str | Path,
    indices: Sequence[int] | np.ndarray | None = None,
    keys: tuple[str, ...] = ("gamma_hat", "S_peak", "V_hoc"),
    eps: float = 1e-8,
) -> dict[str, tuple[float, float]]:
    """Compute train-set mean/std for metadata standardization."""

    with h5py.File(feature_h5_path, "r") as f:
        if indices is None:
            indices = np.arange(len(f[keys[0]]), dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)

        stats = {}
        for key in keys:
            x = np.asarray(f[key][indices], dtype=np.float64)
            stats[key] = (float(np.mean(x)), float(np.std(x) + eps))

    return stats


def _hoc_transform(x: np.ndarray, transform: str) -> np.ndarray:
    transform = transform.lower()
    x = np.asarray(x, dtype=np.float32)
    if transform == "none":
        return x
    if transform == "signed_log1p":
        return np.sign(x) * np.log1p(np.abs(x))
    if transform == "log1p_abs":
        return np.log1p(np.abs(x))
    raise ValueError("hoc_transform must be one of: none, signed_log1p, log1p_abs.")


def compute_hoc_stats(
    feature_h5_path: str | Path,
    indices: Sequence[int] | np.ndarray | None = None,
    transform: str = "signed_log1p",
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute train-set HOC scaling statistics after the selected transform."""

    with h5py.File(feature_h5_path, "r") as f:
        if indices is None:
            indices = np.arange(len(f["h_drc"]), dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)

        x = np.asarray(f["h_drc"][indices], dtype=np.float32)

    x = _hoc_transform(x, transform)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = (np.nanstd(x, axis=0) + eps).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > eps), std, 1.0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    return {
        "transform": transform,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def compute_evm_stats(
    symbol_h5_path: str | Path,
    indices: Sequence[int] | np.ndarray | None = None,
    key: str = "evm_features",
    feature_indices: Sequence[int] | np.ndarray | None = None,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute train-set EVM-feature scaling statistics for a selected subset."""

    with h5py.File(symbol_h5_path, "r") as f:
        if indices is None:
            indices = np.arange(len(f[key]), dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)

        selected = _resolve_feature_indices(f[key].shape[1], feature_indices, key)
        x = np.asarray(f[key][indices], dtype=np.float32)[:, selected]

    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = (np.nanstd(x, axis=0) + eps).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > eps), std, 1.0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    return {
        "key": key,
        "feature_indices": selected.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def _resolve_feature_indices(
    feature_dim: int,
    feature_indices: Sequence[int] | np.ndarray | None,
    key: str,
) -> np.ndarray:
    """Validate an ordered feature subset and expand ``None`` to all columns."""

    if feature_indices is None:
        return np.arange(feature_dim, dtype=np.int64)

    selected = np.asarray(feature_indices, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise ValueError(f"{key} feature_indices must not be empty.")
    if np.any(selected < 0) or np.any(selected >= feature_dim):
        raise ValueError(
            f"{key} feature_indices must lie in [0, {feature_dim - 1}], got {selected.tolist()}."
        )
    if np.unique(selected).size != selected.size:
        raise ValueError(f"{key} feature_indices must not contain duplicates.")
    return selected.astype(np.int64, copy=False)


def _preprocess_iq(
    iq: np.ndarray,
    representation: str = "iq",
    normalize: str = "zscore",
    eps: float = 1e-8,
) -> np.ndarray:
    """Convert I/Q sample to the real tensor consumed by the 1D raw stream."""

    representation = representation.lower()
    normalize = normalize.lower()
    arr = np.asarray(iq, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected I/Q matrix with shape [2,N] or [N,2], got {arr.shape}.")
    if arr.shape[0] == 2:
        i_part = arr[0]
        q_part = arr[1]
    elif arr.shape[1] == 2:
        i_part = arr[:, 0]
        q_part = arr[:, 1]
    else:
        raise ValueError(f"Expected I/Q matrix with shape [2,N] or [N,2], got {arr.shape}.")

    if representation == "iq":
        out = np.stack([i_part, q_part], axis=0).astype(np.float32)
    elif representation == "ap":
        z = i_part.astype(np.float64) + 1j * q_part.astype(np.float64)
        out = np.stack([np.abs(z), np.angle(z)], axis=0).astype(np.float32)
    else:
        raise ValueError("iq_representation must be one of: iq, ap.")

    if normalize == "none":
        return out.astype(np.float32)
    if normalize == "zscore":
        mean = float(np.mean(out))
        std = float(np.std(out))
        return ((out - mean) / (std + eps)).astype(np.float32)
    if normalize == "power":
        if representation != "iq":
            raise ValueError("iq_normalize='power' is only valid for iq_representation='iq'.")
        power = float(np.mean(i_part.astype(np.float64) ** 2 + q_part.astype(np.float64) ** 2))
        return (out / np.sqrt(power + eps)).astype(np.float32)
    raise ValueError("iq_normalize must be one of: none, zscore, power.")


class LEOSignalFeatureDataset(Dataset):
    """Load raw I/Q, DRC-HOC features and physical metadata.

    Expected raw HDF5 fields:
        iq, label, snr_db, fd0, mu, gamma, domain_id, ...
    Expected feature HDF5 fields:
        h_drc, mu_hat, gamma_hat, S_peak, V_hoc, ...
    """

    def __init__(
        self,
        raw_h5_path: str | Path,
        feature_h5_path: str | Path,
        symbol_h5_path: str | Path | None = None,
        indices: Sequence[int] | np.ndarray | None = None,
        split_npz: str | Path | None = None,
        split_name: str | None = None,
        load_into_memory: bool = False,
        cache_iq: bool = False,
        standardize_metadata: bool = True,
        metadata_stats: dict[str, tuple[float, float]] | None = None,
        iq_source: str = "auto",
        iq_representation: str = "iq",
        iq_normalize: str = "zscore",
        standardize_hoc: bool = True,
        hoc_transform: str = "signed_log1p",
        hoc_stats: dict[str, Any] | None = None,
        standardize_evm: bool = True,
        evm_stats: dict[str, Any] | None = None,
        constellation_key: str = "constellation_img",
        evm_key: str = "evm_features",
        evm_feature_indices: Sequence[int] | np.ndarray | None = None,
        include_metadata: bool = True,
        include_iq: bool = True,
        include_hoc: bool = True,
        include_constellation: bool | None = None,
        include_evm_features: bool | None = None,
        return_diagnostics: bool = True,
    ) -> None:
        self.raw_h5_path = Path(raw_h5_path)
        self.feature_h5_path = Path(feature_h5_path)
        self.symbol_h5_path = Path(symbol_h5_path) if symbol_h5_path is not None else None
        self.load_into_memory = load_into_memory
        self.cache_iq = bool(cache_iq)
        self.include_metadata = bool(include_metadata)
        self.include_iq = bool(include_iq)
        self.include_hoc = bool(include_hoc)
        self.include_constellation = (
            self.symbol_h5_path is not None if include_constellation is None else bool(include_constellation)
        )
        self.include_evm_features = (
            self.symbol_h5_path is not None if include_evm_features is None else bool(include_evm_features)
        )
        self.return_diagnostics = bool(return_diagnostics)
        if (self.include_constellation or self.include_evm_features) and self.symbol_h5_path is None:
            raise ValueError(
                "symbol_h5_path is required when constellation or EVM features are enabled."
            )
        self.standardize_metadata = bool(standardize_metadata) and self.include_metadata
        self.iq_source = iq_source.lower()
        if self.iq_source not in {"auto", "raw", "comp"}:
            raise ValueError("iq_source must be one of: auto, raw, comp.")
        self.iq_representation = iq_representation.lower()
        if self.iq_representation not in {"iq", "ap"}:
            raise ValueError("iq_representation must be one of: iq, ap.")
        self.iq_normalize = iq_normalize.lower()
        if self.iq_normalize not in {"none", "zscore", "power"}:
            raise ValueError("iq_normalize must be one of: none, zscore, power.")
        self.standardize_hoc = bool(standardize_hoc)
        self.hoc_transform = hoc_transform.lower()
        self.standardize_evm = bool(standardize_evm)
        self.constellation_key = constellation_key
        self.evm_key = evm_key
        self.evm_feature_indices: np.ndarray | None = None

        if split_npz is not None:
            if split_name is None:
                raise ValueError("split_name must be provided when split_npz is used.")
            splits = load_split_indices(split_npz)
            indices = splits[split_name]

        if load_into_memory:
            with h5py.File(self.raw_h5_path, "r") as f:
                self.raw = {key: f[key][()] for key in f.keys() if key != "meta"}
            with h5py.File(self.feature_h5_path, "r") as f:
                self.feat = {key: f[key][()] for key in f.keys() if key != "meta"}
            if self.symbol_h5_path is not None:
                with h5py.File(self.symbol_h5_path, "r") as f:
                    requested = set()
                    if self.include_constellation:
                        requested.add(self.constellation_key)
                    if self.include_evm_features:
                        requested.add(self.evm_key)
                    if self.return_diagnostics:
                        requested.update(
                            {
                                "symbol_valid",
                                "symbol_count",
                                "timing_phase",
                                "pilot_evm",
                                "evm_best_index",
                                "evm_margin",
                            }
                        )
                    self.sym = {key: f[key][()] for key in requested if key in f}
            else:
                self.sym = None
            n = len(self.raw["label"])
            self.iq_feature_key = self._resolve_iq_feature_key("iq_comp" in self.feat)
        else:
            self.raw_file = None
            self.feat_file = None
            self.sym_file = None
            with h5py.File(self.raw_h5_path, "r") as f:
                n = len(f["label"])
            with h5py.File(self.feature_h5_path, "r") as f:
                self.iq_feature_key = self._resolve_iq_feature_key("iq_comp" in f)
            if self.symbol_h5_path is not None:
                with h5py.File(self.symbol_h5_path, "r") as f:
                    required_keys = []
                    if self.include_constellation:
                        required_keys.append(self.constellation_key)
                    if self.include_evm_features:
                        required_keys.append(self.evm_key)
                    for key in required_keys:
                        if key not in f:
                            raise KeyError(f"Required symbol feature dataset is missing: {key}")
                        if len(f[key]) != n:
                            raise ValueError(
                                f"symbol feature length mismatch: raw has {n}, "
                                f"{key} has {len(f[key])}"
                            )

        if self.include_evm_features:
            with h5py.File(self.symbol_h5_path, "r") as f:
                self.evm_feature_indices = _resolve_feature_indices(
                    int(f[self.evm_key].shape[1]),
                    evm_feature_indices,
                    self.evm_key,
                )

        self.iq_cache: np.ndarray | None = None
        if self.cache_iq and self.include_iq and not self.load_into_memory:
            if self.iq_feature_key is not None:
                with h5py.File(self.feature_h5_path, "r") as f:
                    self.iq_cache = np.asarray(f[self.iq_feature_key][:], dtype=np.float32)
            else:
                with h5py.File(self.raw_h5_path, "r") as f:
                    self.iq_cache = np.asarray(f["iq"][:], dtype=np.float32)

        if indices is None:
            self.indices = np.arange(n, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        if metadata_stats is None and self.standardize_metadata:
            metadata_stats = compute_metadata_stats(
                self.feature_h5_path,
                indices=self.indices,
            )
        self.metadata_stats = metadata_stats

        if hoc_stats is None and self.standardize_hoc:
            hoc_stats = compute_hoc_stats(
                self.feature_h5_path,
                indices=self.indices,
                transform=self.hoc_transform,
            )
        self.hoc_stats = hoc_stats

        if self.include_evm_features and evm_stats is None and self.standardize_evm:
            evm_stats = compute_evm_stats(
                self.symbol_h5_path,
                indices=self.indices,
                key=self.evm_key,
                feature_indices=self.evm_feature_indices,
            )
        self.evm_stats = evm_stats

    def _resolve_iq_feature_key(self, has_iq_comp: bool) -> str | None:
        if self.iq_source == "raw":
            return None
        if self.iq_source == "comp":
            if not has_iq_comp:
                raise ValueError(
                    "iq_source='comp' was requested, but the feature HDF5 file "
                    "does not contain an 'iq_comp' dataset."
                )
            return "iq_comp"
        return "iq_comp" if has_iq_comp else None

    def _raw_open(self) -> h5py.File:
        if self.load_into_memory:
            raise RuntimeError("_raw_open should not be called in memory mode.")
        if getattr(self, "raw_file", None) is None:
            self.raw_file = h5py.File(self.raw_h5_path, "r")
        return self.raw_file

    def _feat_open(self) -> h5py.File:
        if self.load_into_memory:
            raise RuntimeError("_feat_open should not be called in memory mode.")
        if getattr(self, "feat_file", None) is None:
            self.feat_file = h5py.File(self.feature_h5_path, "r")
        return self.feat_file

    def _sym_open(self) -> h5py.File:
        if self.load_into_memory:
            raise RuntimeError("_sym_open should not be called in memory mode.")
        if self.symbol_h5_path is None:
            raise RuntimeError("No symbol feature HDF5 path was provided.")
        if getattr(self, "sym_file", None) is None:
            self.sym_file = h5py.File(self.symbol_h5_path, "r")
        return self.sym_file

    def _raw_get(self, key: str, i: int):
        if self.load_into_memory:
            return self.raw[key][i]
        return self._raw_open()[key][i]

    def _feat_get(self, key: str, i: int):
        if self.load_into_memory:
            return self.feat[key][i]
        return self._feat_open()[key][i]

    def _sym_get(self, key: str, i: int):
        if self.load_into_memory:
            return self.sym[key][i]
        return self._sym_open()[key][i]

    def __len__(self) -> int:
        return int(len(self.indices))

    def _build_meta(self, i: int) -> np.ndarray:
        values = np.array(
            [
                float(self._feat_get("gamma_hat", i)),
                float(self._feat_get("S_peak", i)),
                float(self._feat_get("V_hoc", i)),
            ],
            dtype=np.float32,
        )

        if self.standardize_metadata and self.metadata_stats is not None:
            keys = ["gamma_hat", "S_peak", "V_hoc"]
            for j, key in enumerate(keys):
                mean, std = self.metadata_stats[key]
                values[j] = (values[j] - mean) / std

        return values.astype(np.float32)

    def _build_iq(self, i: int) -> np.ndarray:
        if self.iq_cache is not None:
            iq = self.iq_cache[i]
        elif self.iq_feature_key is not None:
            iq = self._feat_get(self.iq_feature_key, i)
        else:
            iq = self._raw_get("iq", i)
        return _preprocess_iq(
            iq,
            representation=self.iq_representation,
            normalize=self.iq_normalize,
        )

    def _build_hoc(self, i: int) -> np.ndarray:
        hoc = self._feat_get("h_drc", i).astype(np.float32)
        hoc = _hoc_transform(hoc, self.hoc_transform).astype(np.float32)
        if self.standardize_hoc and self.hoc_stats is not None:
            mean = np.asarray(self.hoc_stats["mean"], dtype=np.float32)
            std = np.asarray(self.hoc_stats["std"], dtype=np.float32)
            hoc = (hoc - mean) / std
        return np.nan_to_num(hoc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _build_evm_features(self, i: int) -> np.ndarray:
        evm = self._sym_get(self.evm_key, i).astype(np.float32)
        if self.evm_feature_indices is not None:
            evm = evm[self.evm_feature_indices]
        if self.standardize_evm and self.evm_stats is not None:
            mean = np.asarray(self.evm_stats["mean"], dtype=np.float32)
            std = np.asarray(self.evm_stats["std"], dtype=np.float32)
            evm = (evm - mean) / std
        return np.nan_to_num(evm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        i = int(self.indices[idx])

        label = int(self._raw_get("label", i))

        item: dict[str, Any] = {
            "label": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(i, dtype=torch.long),
        }
        if self.include_iq:
            item["iq"] = torch.from_numpy(self._build_iq(i))
        if self.include_hoc:
            item["hoc"] = torch.from_numpy(self._build_hoc(i))
        if self.include_metadata:
            item["meta"] = torch.from_numpy(self._build_meta(i))

        if self.include_constellation:
            constellation = self._sym_get(self.constellation_key, i).astype(np.float32)
            item["constellation"] = torch.from_numpy(
                np.nan_to_num(constellation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            )
        if self.include_evm_features:
            item["evm_features"] = torch.from_numpy(self._build_evm_features(i))

        if self.return_diagnostics:
            # Raw metadata for grouped evaluation.
            for key in ["snr_db", "gamma", "mu", "fd0", "domain_id"]:
                exists = key in self.raw if self.load_into_memory else key in self._raw_open()
                if exists:
                    value = self._raw_get(key, i)
                    dtype = torch.long if key == "domain_id" else torch.float32
                    item[key] = torch.tensor(int(value) if key == "domain_id" else float(value), dtype=dtype)

            # Feature metadata remains available for analysis even when it is not a model input.
            for key in ["gamma_hat", "S_peak", "V_hoc", "mu_hat", "selected_k"]:
                exists = key in self.feat if self.load_into_memory else key in self._feat_open()
                if exists:
                    value = self._feat_get(key, i)
                    dtype = torch.long if key == "selected_k" else torch.float32
                    item[key] = torch.tensor(int(value) if key == "selected_k" else float(value), dtype=dtype)

        if self.return_diagnostics and self.symbol_h5_path is not None:
            for key in ["symbol_valid", "symbol_count", "timing_phase", "pilot_evm", "evm_best_index", "evm_margin"]:
                exists = key in self.sym if self.load_into_memory else key in self._sym_open()
                if exists:
                    value = self._sym_get(key, i)
                    dtype = torch.long if key in {"symbol_count", "timing_phase", "evm_best_index"} else torch.float32
                    if key == "symbol_valid":
                        item[key] = torch.tensor(bool(value), dtype=torch.bool)
                    elif dtype is torch.long:
                        item[key] = torch.tensor(int(value), dtype=dtype)
                    else:
                        item[key] = torch.tensor(float(value), dtype=dtype)

        return item

    def close(self) -> None:
        if getattr(self, "raw_file", None) is not None:
            self.raw_file.close()
            self.raw_file = None
        if getattr(self, "feat_file", None) is not None:
            self.feat_file.close()
            self.feat_file = None
        if getattr(self, "sym_file", None) is not None:
            self.sym_file.close()
            self.sym_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
