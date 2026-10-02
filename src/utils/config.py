"""Configuration utilities.

This module provides a compact configuration system based on YAML.
It supports:
    - loading one or more YAML files;
    - recursive dictionary merging;
    - command-line style overrides such as ["train.epochs=100"];
    - dot-access through the Config class.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml


class Config(dict):
    """Dictionary with recursive dot access."""

    def __init__(self, data: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        data = {} if data is None else dict(data)
        data.update(kwargs)
        for key, value in data.items():
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Config):
            return value
        if isinstance(value, Mapping):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Config has no attribute '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def to_dict(self) -> dict[str, Any]:
        """Convert recursively to a plain Python dict."""

        def unwrap(obj: Any) -> Any:
            if isinstance(obj, Config):
                return {k: unwrap(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [unwrap(v) for v in obj]
            return obj

        return unwrap(self)

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(self.to_dict()))


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict. Empty files return an empty dict."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return data


def save_config(config: Mapping[str, Any] | Config, path: str | Path) -> None:
    """Save config as YAML."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict() if isinstance(config, Config) else dict(config)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def recursive_update(
    base: MutableMapping[str, Any],
    update: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Recursively update ``base`` with ``update``."""

    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            recursive_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def merge_configs(*configs: Mapping[str, Any] | Config) -> Config:
    """Merge multiple configs from left to right."""

    merged: dict[str, Any] = {}
    for cfg in configs:
        cfg_dict = cfg.to_dict() if isinstance(cfg, Config) else dict(cfg)
        recursive_update(merged, cfg_dict)
    return Config(merged)


def _parse_override_value(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()

    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def apply_overrides(config: MutableMapping[str, Any], overrides: Sequence[str] | None) -> None:
    """Apply command-line style overrides in-place.

    Example:
        overrides = ["train.epochs=50", "model.feature_dim=256"]
    """

    if not overrides:
        return

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override '{override}'. Expected format: key.subkey=value"
            )

        key_path, raw_value = override.split("=", 1)
        keys = [k.strip() for k in key_path.split(".") if k.strip()]
        if not keys:
            raise ValueError(f"Invalid override key in '{override}'")

        value = _parse_override_value(raw_value)

        cursor: MutableMapping[str, Any] = config
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], MutableMapping):
                cursor[key] = {}
            cursor = cursor[key]  # type: ignore[assignment]

        cursor[keys[-1]] = value


def load_config(
    paths: str | Path | Sequence[str | Path],
    overrides: Sequence[str] | None = None,
) -> Config:
    """Load one or more YAML files and apply optional overrides."""

    if isinstance(paths, (str, Path)):
        paths = [paths]

    merged: dict[str, Any] = {}
    for path in paths:
        recursive_update(merged, read_yaml(path))

    apply_overrides(merged, overrides)
    return Config(merged)


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a path relative to root if it is not absolute."""

    path = Path(path)
    if path.is_absolute():
        return path
    if root is None:
        root = Path.cwd()
    return Path(root).resolve() / path
