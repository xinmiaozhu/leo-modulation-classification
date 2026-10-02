"""Training callbacks and checkpoint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.utils.io import ensure_parent, save_json


@dataclass
class EarlyStopping:
    """Simple early stopping helper.

    Args:
        patience: Number of epochs without improvement before stopping.
        mode: "min" or "max".
        min_delta: Required improvement margin.
    """

    patience: int = 15
    mode: str = "max"
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        self.best: float | None = None
        self.num_bad_epochs: int = 0
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """Update state. Returns True if value improved."""

        if self.best is None:
            self.best = value
            self.num_bad_epochs = 0
            return True

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        self.should_stop = self.num_bad_epochs >= self.patience
        return improved


class CheckpointManager:
    """Save and load model checkpoints."""

    def __init__(self, output_dir: str | Path, monitor: str = "val_acc", mode: str = "max") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value: float | None = None
        self.best_path = self.output_dir / "best.pt"
        self.last_path = self.output_dir / "last.pt"

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value
        if self.mode == "min":
            return value < self.best_value
        raise ValueError("mode must be 'min' or 'max'.")

    def save(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int | None = None,
        metrics: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        path = ensure_parent(path)
        state = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
            "config": config or {},
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(state, path)

    def step(
        self,
        model: torch.nn.Module,
        monitor_value: float,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int | None = None,
        metrics: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """Save last checkpoint and best checkpoint if improved."""

        self.save(
            self.last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )

        improved = self._is_better(float(monitor_value))
        if improved:
            self.best_value = float(monitor_value)
            self.save(
                self.best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
        return improved


class HistoryLogger:
    """Store epoch metrics and export CSV/JSON."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(dict(record))

    def load_csv(self, path: str | Path) -> None:
        """Restore previously saved epoch records for resumed training."""

        path = Path(path)
        if path.exists():
            self.records = pd.read_csv(path).to_dict(orient="records")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save_csv(self, path: str | Path) -> None:
        path = ensure_parent(path)
        self.to_dataframe().to_csv(path, index=False)

    def save_json(self, path: str | Path) -> None:
        save_json(self.records, path)
