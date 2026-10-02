"""Metrics and grouped analysis helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy from logits."""

    pred = torch.argmax(logits, dim=-1)
    return float((pred == labels).float().mean().item())


def topk_accuracy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 3,
) -> float:
    """Compute top-k accuracy."""

    topk = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1).indices
    correct = (topk == labels.unsqueeze(-1)).any(dim=-1)
    return float(correct.float().mean().item())


def confusion_matrix_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> np.ndarray:
    """Return confusion matrix."""

    labels = None if num_classes is None else list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> dict[str, float]:
    """Return common classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def grouped_accuracy(
    df: pd.DataFrame,
    group_col: str,
    label_col: str = "label",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Compute accuracy grouped by a column."""

    if group_col not in df.columns:
        raise KeyError(f"group_col '{group_col}' not found in DataFrame.")

    rows = []
    for value, g in df.groupby(group_col):
        rows.append(
            {
                group_col: value,
                "count": int(len(g)),
                "accuracy": float((g[label_col] == g[pred_col]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def grouped_mean(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Compute mean/std/count of value_col grouped by group_col."""

    rows = []
    for value, g in df.groupby(group_col):
        rows.append(
            {
                group_col: value,
                "count": int(len(g)),
                "mean": float(g[value_col].mean()),
                "std": float(g[value_col].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def add_binned_column(
    df: pd.DataFrame,
    source_col: str,
    bins: np.ndarray,
    out_col: str,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of df with a binned categorical column."""

    out = df.copy()
    out[out_col] = pd.cut(out[source_col], bins=bins, labels=labels, include_lowest=True)
    return out


def snr_gamma_grid_accuracy(
    df: pd.DataFrame,
    snr_bins: np.ndarray,
    gamma_bins: np.ndarray,
    label_col: str = "label",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Compute accuracy on SNR-gamma grid."""

    if "snr_db" not in df.columns or "gamma" not in df.columns:
        raise KeyError("DataFrame must contain 'snr_db' and 'gamma' columns.")

    work = df.copy()
    work["snr_bin"] = pd.cut(work["snr_db"], bins=snr_bins, include_lowest=True)
    work["gamma_bin"] = pd.cut(work["gamma"], bins=gamma_bins, include_lowest=True)

    rows = []
    grouped = work.groupby(["snr_bin", "gamma_bin"], observed=False)
    for (snr_bin, gamma_bin), g in grouped:
        if len(g) == 0:
            acc = np.nan
        else:
            acc = float((g[label_col] == g[pred_col]).mean())
        rows.append(
            {
                "snr_bin": str(snr_bin),
                "gamma_bin": str(gamma_bin),
                "count": int(len(g)),
                "accuracy": acc,
            }
        )
    return pd.DataFrame(rows)


def summarize_eval_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize evaluator DataFrame."""

    report = classification_report_dict(
        df["label"].to_numpy(),
        df["pred"].to_numpy(),
    )
    report["num_samples"] = int(len(df))

    if "snr_db" in df.columns:
        report["snr_db_min"] = float(df["snr_db"].min())
        report["snr_db_max"] = float(df["snr_db"].max())
    if "gamma" in df.columns:
        report["gamma_min"] = float(df["gamma"].min())
        report["gamma_max"] = float(df["gamma"].max())
    if "gate_mean" in df.columns:
        report["gate_mean"] = float(df["gate_mean"].mean())
        report["gate_std"] = float(df["gate_mean"].std(ddof=0))

    return report
