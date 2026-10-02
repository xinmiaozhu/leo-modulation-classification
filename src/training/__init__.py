"""Training and evaluation utilities."""

from .metrics import (
    accuracy_from_logits,
    confusion_matrix_np,
    classification_report_dict,
    grouped_accuracy,
    snr_gamma_grid_accuracy,
)
from .trainer import Trainer, TrainConfig
from .evaluator import Evaluator, evaluate_checkpoint
from .callbacks import EarlyStopping, CheckpointManager, HistoryLogger

__all__ = [
    "accuracy_from_logits",
    "confusion_matrix_np",
    "classification_report_dict",
    "grouped_accuracy",
    "snr_gamma_grid_accuracy",
    "Trainer",
    "TrainConfig",
    "Evaluator",
    "evaluate_checkpoint",
    "EarlyStopping",
    "CheckpointManager",
    "HistoryLogger",
]
