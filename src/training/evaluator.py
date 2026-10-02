"""Model evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import classification_report_dict, summarize_eval_dataframe


class Evaluator:
    """Evaluate a model and return per-sample predictions as a DataFrame."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for evaluation, but this PyTorch installation has no "
                "CUDA support. Install a CUDA-enabled wheel or request CPU explicitly."
            )
        self.device = requested_device
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: DataLoader,
        return_aux: bool = True,
        desc: str = "Evaluating",
    ) -> pd.DataFrame:
        self.model.eval()
        records: list[dict[str, Any]] = []

        for batch in tqdm(loader, desc=desc, leave=False):
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

            # Support raw-only models, hoc-only models and full dual models.
            try:
                if return_aux:
                    if constellation is not None or evm_features is not None:
                        output = self.model(
                            iq,
                            hoc,
                            meta,
                            constellation=constellation,
                            evm_features=evm_features,
                            return_aux=True,
                        )
                    else:
                        output = self.model(iq, hoc, meta, return_aux=True)
                else:
                    if constellation is not None or evm_features is not None:
                        output = self.model(
                            iq,
                            hoc,
                            meta,
                            constellation=constellation,
                            evm_features=evm_features,
                        )
                    else:
                        output = self.model(iq, hoc, meta)
            except TypeError:
                try:
                    if return_aux:
                        output = self.model(iq, hoc, meta, return_aux=True)
                    else:
                        output = self.model(iq, hoc, meta)
                except TypeError:
                    if return_aux:
                        output = self.model(iq, return_aux=True)
                    else:
                        output = self.model(iq)

            if isinstance(output, tuple):
                logits, aux = output
            else:
                logits, aux = output, {}

            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            conf = torch.max(probs, dim=-1).values

            B = label.size(0)
            for i in range(B):
                rec: dict[str, Any] = {
                    "label": int(label[i].item()),
                    "pred": int(pred[i].item()),
                    "confidence": float(conf[i].item()),
                    "correct": int(pred[i].item() == label[i].item()),
                }

                # Pass through common metadata if present.
                for key in [
                    "index",
                    "snr_db",
                    "gamma",
                    "mu",
                    "fd0",
                    "domain_id",
                    "gamma_hat",
                    "S_peak",
                    "V_hoc",
                    "mu_hat",
                    "selected_k",
                    "symbol_valid",
                    "symbol_count",
                    "timing_phase",
                    "pilot_evm",
                    "evm_best_index",
                    "evm_margin",
                ]:
                    if key in batch:
                        value = batch[key][i]
                        if torch.is_tensor(value):
                            value = value.item()
                        if key in {"index", "domain_id", "selected_k", "symbol_count", "timing_phase", "evm_best_index"}:
                            rec[key] = int(value)
                        elif key == "symbol_valid":
                            rec[key] = bool(value)
                        else:
                            rec[key] = float(value)

                # Gate diagnostics.
                if return_aux and isinstance(aux, dict) and "gate" in aux:
                    gate = aux["gate"].detach().cpu()
                    rec["gate_mean"] = float(gate[i].mean().item())
                    rec["gate_std"] = float(gate[i].std(unbiased=False).item())
                    rec["gate_min"] = float(gate[i].min().item())
                    rec["gate_max"] = float(gate[i].max().item())

                records.append(rec)

        return pd.DataFrame(records)

    def evaluate_and_summarize(
        self,
        loader: DataLoader,
        return_aux: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = self.evaluate_loader(loader, return_aux=return_aux)
        summary = summarize_eval_dataframe(df)
        return df, summary


def evaluate_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    loader: DataLoader,
    device: str | torch.device = "cuda",
    strict: bool = True,
    return_aux: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a checkpoint into model and evaluate."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=strict)

    evaluator = Evaluator(model, device=device)
    return evaluator.evaluate_and_summarize(loader, return_aux=return_aux)
