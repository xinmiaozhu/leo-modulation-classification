#!/usr/bin/env python
"""Audit the promoted Protocol-B hybrid-DFRFT result lineage.

The audit fails closed: a report is written only when every requested
matched-seed checkpoint and test CSV exists and the Proposed checkpoints point
to the final 48-D exact-mixture constellation artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (41, 73, 107, 149, 211)
METHODS = ("proposed", "mcnet", "starnet")
EXPECTED_RAW = "data/processed/leo_7mods_joint_practical.h5"
EXPECTED_IQ = "data/features/hybrid_dfrft/protocol_b_iqcomp.h5"
EXPECTED_SYMBOL = "data/features/hybrid_dfrft/protocol_b_symbol_exact.h5"
EXPECTED_PILOT = "data/features/hybrid_dfrft/protocol_b_pilot_hybrid.h5"
EXPECTED_SPLITS = "data/splits/leo_7mods_joint_practical_splits.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--output",
        default="outputs/results/hybrid_dfrft/protocol_b/artifact_audit.json",
    )
    parser.add_argument(
        "--require-ablation",
        action="store_true",
        help="Also require complete I/Q-only and constellation-only five-seed outputs.",
    )
    return parser.parse_args()


def normalized(value: object) -> str:
    return str(value).replace("\\", "/")


def accuracy(path: Path) -> float:
    frame = pd.read_csv(path)
    if "correct" in frame:
        return float(frame["correct"].astype(float).mean())
    pred = "pred" if "pred" in frame else "predicted"
    return float((frame["label"].to_numpy() == frame[pred].to_numpy()).mean())


def require_equal(config: dict, key: str, expected: str, context: str) -> None:
    actual = normalized(config.get(key, ""))
    if actual != expected:
        raise ValueError(f"{context}: {key}={actual!r}, expected {expected!r}")


def main() -> None:
    args = parse_args()
    pilot_path = ROOT / EXPECTED_PILOT
    with h5py.File(pilot_path, "r") as handle:
        if handle.attrs.get("pilot_estimator_type") != "hybrid_dfrft":
            raise ValueError("Promoted pilot artifact is not marked as hybrid_dfrft.")
        if normalized(handle.attrs.get("raw_data", "")) != EXPECTED_RAW:
            raise ValueError("Promoted pilot artifact points to the wrong raw data.")

    iq_path = ROOT / EXPECTED_IQ
    with h5py.File(iq_path, "r") as handle:
        if normalized(handle.attrs.get("mu_feature_data", "")) != EXPECTED_PILOT:
            raise ValueError("Compensated-I/Q artifact does not use the promoted pilot estimates.")
        if normalized(handle.attrs.get("raw_data", "")) != EXPECTED_RAW:
            raise ValueError("Compensated-I/Q artifact points to the wrong raw data.")

    symbol_path = ROOT / EXPECTED_SYMBOL
    with h5py.File(symbol_path, "r") as handle:
        if handle["evm_features"].shape[1] != 48:
            raise ValueError("Promoted constellation artifact is not 48-D.")
        feature_config = json.loads(handle.attrs["config_json"])
    expected_feature_settings = {
        "candidate_score_mode": "exact_mixture",
        "center": True,
        "power_normalize": True,
        "phase_grid_size": 1,
        "complexity_penalty_lambda": 0.0,
        "apsk_ring_penalty_lambda": 0.0,
    }
    for key, expected in expected_feature_settings.items():
        if feature_config.get(key) != expected:
            raise ValueError(
                f"Promoted constellation artifact has {key}={feature_config.get(key)!r}; "
                f"expected {expected!r}."
            )

    rows: list[dict[str, object]] = []
    for method in args.methods:
        for seed in args.seeds:
            checkpoint_dir = (
                ROOT / "outputs/checkpoints/hybrid_dfrft/protocol_b" / method / f"seed_{seed}"
            )
            result_dir = ROOT / "outputs/results/hybrid_dfrft/protocol_b" / method
            required = [
                checkpoint_dir / "best.pt",
                checkpoint_dir / "train_result.json",
                checkpoint_dir / "run_config.json",
                result_dir / f"seed_{seed}_test.csv",
                result_dir / f"seed_{seed}_test.summary.json",
            ]
            missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(f"{method} seed {seed} incomplete: {missing}")
            config = json.loads((checkpoint_dir / "run_config.json").read_text(encoding="utf-8"))
            context = f"{method} seed {seed}"
            require_equal(config, "raw_data", EXPECTED_RAW, context)
            require_equal(config, "feature_data", EXPECTED_IQ, context)
            require_equal(config, "splits", EXPECTED_SPLITS, context)
            if config.get("iq_source") != "comp":
                raise ValueError(f"{context}: iq_source is not compensated I/Q")
            if method == "proposed":
                require_equal(config, "symbol_feature_data", EXPECTED_SYMBOL, context)
                if int(config.get("evm_dim", -1)) != 48:
                    raise ValueError(f"{context}: descriptor dimension is not 48")
                model_config = config.get("model_config", {})
                if not model_config.get("use_iq_stream") or not model_config.get("use_evm_features"):
                    raise ValueError(f"{context}: Proposed is not the fused two-stream model")
                if model_config.get("use_hoc_stream"):
                    raise ValueError(f"{context}: deployed Proposed unexpectedly enables HOC")
            result_csv = result_dir / f"seed_{seed}_test.csv"
            rows.append({"method": method, "seed": seed, "accuracy": accuracy(result_csv)})

    ablation_rows: list[dict[str, object]] = []
    if args.require_ablation:
        ablation_root = ROOT / "outputs/results/hybrid_dfrft/representation_ablation/protocol_b"
        for method in ("iq_only", "constellation_only"):
            for seed in args.seeds:
                path = ablation_root / method / f"seed_{seed}_test.csv"
                summary_path = path.with_suffix(".summary.json")
                checkpoint_dir = (
                    ROOT
                    / "outputs/checkpoints/hybrid_dfrft/representation_ablation/protocol_b"
                    / method
                    / f"seed_{seed}"
                )
                required = [
                    path,
                    summary_path,
                    checkpoint_dir / "best.pt",
                    checkpoint_dir / "train_result.json",
                    checkpoint_dir / "run_config.json",
                ]
                missing = [str(item.relative_to(ROOT)) for item in required if not item.exists()]
                if missing:
                    raise FileNotFoundError(
                        f"Representation ablation incomplete: {missing}"
                    )
                config = json.loads(
                    (checkpoint_dir / "run_config.json").read_text(encoding="utf-8")
                )
                context = f"{method} seed {seed}"
                require_equal(config, "raw_data", EXPECTED_RAW, context)
                require_equal(config, "feature_data", EXPECTED_IQ, context)
                require_equal(config, "symbol_feature_data", EXPECTED_SYMBOL, context)
                require_equal(config, "splits", EXPECTED_SPLITS, context)
                model_config = config.get("model_config", {})
                expected_streams = {
                    "iq_only": (True, False),
                    "constellation_only": (False, True),
                }
                expected_iq, expected_constellation = expected_streams[method]
                if bool(model_config.get("use_iq_stream")) != expected_iq:
                    raise ValueError(f"{context}: unexpected I/Q-stream setting")
                if bool(model_config.get("use_evm_features")) != expected_constellation:
                    raise ValueError(f"{context}: unexpected constellation-stream setting")
                if model_config.get("use_hoc_stream"):
                    raise ValueError(f"{context}: HOC must be disabled")
                ablation_rows.append(
                    {"method": method, "seed": seed, "accuracy": accuracy(path)}
                )

    frame = pd.DataFrame(rows)
    aggregate = (
        frame.groupby("method", sort=False)["accuracy"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    payload = {
        "status": "PASS",
        "protocol": "B",
        "front_end": "hybrid_dfrft",
        "feature_lineage": {
            "pilot_estimates": EXPECTED_PILOT,
            "compensated_iq": EXPECTED_IQ,
            "constellation": EXPECTED_SYMBOL,
            "dimension": 48,
            **expected_feature_settings,
        },
        "runs": rows,
        "aggregate": aggregate.to_dict(orient="records"),
        "ablation_runs": ablation_rows,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(f"PASS: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
