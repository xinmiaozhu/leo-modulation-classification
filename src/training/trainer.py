"""Training loop for DRC-DualNet and baselines."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.losses import DRCTrainingLoss, STARNetTrainingLoss
from src.training.callbacks import CheckpointManager, EarlyStopping, HistoryLogger
from src.training.metrics import accuracy_from_logits
from src.utils.logger import get_logger


@dataclass
class TrainConfig:
    """Training configuration."""

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    early_stop_patience: int = 15
    lambda_gate: float = 0.0
    grad_clip_norm: float | None = None
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_optimizer(
    model: torch.nn.Module,
    cfg: TrainConfig,
) -> torch.optim.Optimizer:
    """Build optimizer."""

    name = cfg.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> Any | None:
    """Build learning-rate scheduler."""

    name = cfg.scheduler.lower()
    if name in {"none", "null", ""}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, cfg.epochs // 3),
            gamma=0.3,
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")


class Trainer:
    """Generic trainer for DRC-DualNet-style models."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: TrainConfig,
        output_dir: str | Path = "outputs/checkpoints/run",
        criterion: torch.nn.Module | None = None,
        config_for_checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self.logger = get_logger()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        requested_device = torch.device(cfg.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for training, but this PyTorch installation has no "
                "CUDA support. Install a CUDA-enabled wheel or pass --device cpu explicitly."
            )
        self.device = requested_device
        self.model.to(self.device)

        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        self.criterion = criterion or DRCTrainingLoss(lambda_gate=cfg.lambda_gate)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt = CheckpointManager(self.output_dir, monitor="val_acc", mode="max")
        self.early = EarlyStopping(patience=cfg.early_stop_patience, mode="max")
        self.history = HistoryLogger()
        self.config_for_checkpoint = config_for_checkpoint or {}

    def _resume(self, checkpoint_path: str | Path) -> tuple[int, float]:
        """Restore training state and return (start_epoch, best_metric)."""

        checkpoint_path = Path(checkpoint_path)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])

        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            for optimizer_state in self.optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(self.device)
        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        last_epoch = int(state.get("epoch") or 0)
        self.history.load_csv(self.output_dir / "history.csv")
        history = self.history.records
        if history:
            monitor_values = [
                float(record.get("val_acc", record["train_acc"]))
                for record in history
                if record.get("val_acc", record.get("train_acc")) is not None
            ]
        else:
            metrics = state.get("metrics", {})
            monitor = metrics.get("val_acc", metrics.get("train_acc"))
            monitor_values = [float(monitor)] if monitor is not None else []

        self.early = EarlyStopping(patience=self.cfg.early_stop_patience, mode="max")
        for value in monitor_values:
            self.early.step(value)
        self.early.should_stop = False

        best_metric = max(monitor_values, default=-1.0)
        self.ckpt.best_value = best_metric if monitor_values else None
        self.logger.info(
            f"Resumed training from {checkpoint_path} at epoch {last_epoch}; "
            f"next epoch={last_epoch + 1}, best monitored value={best_metric:.4f}"
        )
        return last_epoch + 1, best_metric

    def _forward_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict]:
        iq = batch.get("iq", None)
        if iq is not None:
            iq = iq.to(self.device, non_blocking=True)
        label = batch["label"].to(self.device, non_blocking=True)

        hoc = batch.get("hoc", None)
        meta = batch.get("meta", None)
        if hoc is not None:
            hoc = hoc.to(self.device, non_blocking=True)
        if meta is not None:
            meta = meta.to(self.device, non_blocking=True)
        constellation = batch.get("constellation", None)
        evm_features = batch.get("evm_features", None)
        if constellation is not None:
            constellation = constellation.to(self.device, non_blocking=True)
        if evm_features is not None:
            evm_features = evm_features.to(self.device, non_blocking=True)

        # Most models in this project support the unified signature
        # model(iq, hoc, meta, return_aux=True). Raw-only models may ignore hoc/meta.
        try:
            if constellation is not None or evm_features is not None:
                logits, aux = self.model(
                    iq,
                    hoc,
                    meta,
                    constellation=constellation,
                    evm_features=evm_features,
                    return_aux=True,
                )
            else:
                logits, aux = self.model(iq, hoc, meta, return_aux=True)
        except TypeError:
            try:
                logits, aux = self.model(iq, hoc, meta, return_aux=True)
            except TypeError:
                logits, aux = self.model(iq, return_aux=True)

        if isinstance(self.criterion, (DRCTrainingLoss, STARNetTrainingLoss)):
            loss, logs = self.criterion(logits, label, aux=aux, meta=meta)
        else:
            loss = self.criterion(logits, label)
            logs = {
                "loss_cls": loss.detach(),
                "loss_gate": torch.zeros((), device=self.device),
                "loss_total": loss.detach(),
            }

        return loss, logits, logs

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_cls = 0.0
        total_gate = 0.0
        total_correct = 0
        total_count = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for batch in pbar:
            loss, logits, logs = self._forward_loss(batch)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

            self.optimizer.step()

            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            total_cls += float(logs["loss_cls"].item()) * bs
            total_gate += float(logs["loss_gate"].item()) * bs
            total_correct += int((torch.argmax(logits, dim=-1) == labels).sum().item())
            total_count += bs

            pbar.set_postfix(
                loss=total_loss / max(total_count, 1),
                acc=total_correct / max(total_count, 1),
            )

        return {
            "train_loss": total_loss / total_count,
            "train_loss_cls": total_cls / total_count,
            "train_loss_gate": total_gate / total_count,
            "train_acc": total_correct / total_count,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_cls = 0.0
        total_gate = 0.0
        total_correct = 0
        total_count = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [val]", leave=False)
        for batch in pbar:
            loss, logits, logs = self._forward_loss(batch)
            labels = batch["label"].to(self.device, non_blocking=True)

            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            total_cls += float(logs["loss_cls"].item()) * bs
            total_gate += float(logs["loss_gate"].item()) * bs
            total_correct += int((torch.argmax(logits, dim=-1) == labels).sum().item())
            total_count += bs

        return {
            "val_loss": total_loss / total_count,
            "val_loss_cls": total_cls / total_count,
            "val_loss_gate": total_gate / total_count,
            "val_acc": total_correct / total_count,
        }

    def fit(self, resume_from: str | Path | None = None) -> dict[str, Any]:
        """Run full training."""

        self.logger.info(f"Training on device: {self.device}")
        best_acc = -1.0

        start_epoch = 1
        if resume_from is not None:
            start_epoch, best_acc = self._resume(resume_from)

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)
            metrics = {"epoch": epoch, **train_metrics, **val_metrics}

            if self.scheduler is not None:
                if self.cfg.scheduler.lower() == "plateau":
                    self.scheduler.step(metrics.get("val_acc", train_metrics["train_acc"]))
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            metrics["lr"] = current_lr
            self.history.append(metrics)

            monitor = metrics.get("val_acc", metrics["train_acc"])
            best_this_epoch = self.ckpt.step(
                self.model,
                monitor_value=monitor,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=metrics,
                config=self.config_for_checkpoint,
            )
            if best_this_epoch:
                best_acc = monitor

            self.history.save_csv(self.output_dir / "history.csv")
            self.history.save_json(self.output_dir / "history.json")

            self.logger.info(
                f"Epoch {epoch:03d}: "
                f"train_loss={metrics['train_loss']:.4f}, "
                f"train_acc={metrics['train_acc']:.4f}, "
                f"val_acc={metrics.get('val_acc', float('nan')):.4f}, "
                f"lr={current_lr:.3e}"
            )

            improved = self.early.step(monitor)
            if self.early.should_stop:
                self.logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best monitored value: {self.early.best:.4f}"
                )
                break

        return {
            "best_acc": float(best_acc),
            "best_path": str(self.ckpt.best_path),
            "last_path": str(self.ckpt.last_path),
            "history_path": str(self.output_dir / "history.csv"),
        }
