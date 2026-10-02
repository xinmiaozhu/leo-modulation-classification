"""Input/output helpers for JSON, pickle, NumPy and HDF5 files."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: str | Path) -> Path:
    """Create parent directory for a file path and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    path = ensure_parent(path)
    with Path(path).open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_npz(path: str | Path, **arrays: Any) -> None:
    path = ensure_parent(path)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def save_h5(path: str | Path, data: Mapping[str, Any], compression: str | None = "gzip") -> None:
    """Save a nested dictionary to HDF5."""

    path = ensure_parent(path)

    def write_item(group: h5py.Group, key: str, value: Any) -> None:
        if isinstance(value, Mapping):
            subgroup = group.create_group(key)
            for sub_key, sub_value in value.items():
                write_item(subgroup, sub_key, sub_value)
            return

        if isinstance(value, str):
            group.create_dataset(key, data=np.bytes_(value))
            return

        arr = np.asarray(value)
        if arr.shape == ():
            group.create_dataset(key, data=arr)
        else:
            group.create_dataset(key, data=arr, compression=compression)

    with h5py.File(path, "w") as f:
        for key, value in data.items():
            write_item(f, key, value)


def load_h5(path: str | Path, as_dict: bool = True) -> dict[str, Any] | h5py.File:
    """Load HDF5 file.

    If ``as_dict`` is False, an open h5py.File object is returned and must be
    closed by the caller.
    """

    path = Path(path)

    if not as_dict:
        return h5py.File(path, "r")

    def read_item(obj: h5py.Dataset | h5py.Group) -> Any:
        if isinstance(obj, h5py.Group):
            return {key: read_item(obj[key]) for key in obj.keys()}

        value = obj[()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    with h5py.File(path, "r") as f:
        return {key: read_item(f[key]) for key in f.keys()}


def append_h5_dataset(
    path: str | Path,
    key: str,
    array: np.ndarray,
    maxshape_first_dim: bool = True,
    compression: str | None = "gzip",
) -> None:
    """Append array to an HDF5 dataset along the first dimension."""

    path = ensure_parent(path)
    array = np.asarray(array)

    with h5py.File(path, "a") as f:
        if key not in f:
            maxshape = (None,) + array.shape[1:] if maxshape_first_dim else array.shape
            f.create_dataset(
                key,
                data=array,
                maxshape=maxshape,
                chunks=True,
                compression=compression,
            )
            return

        dset = f[key]
        old_size = dset.shape[0]
        new_size = old_size + array.shape[0]
        dset.resize((new_size,) + dset.shape[1:])
        dset[old_size:new_size] = array


def list_files(root: str | Path, suffix: str | tuple[str, ...] | None = None) -> list[Path]:
    """Recursively list files under root."""

    root = Path(root)
    if suffix is None:
        return sorted([p for p in root.rglob("*") if p.is_file()])
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix in suffix])
