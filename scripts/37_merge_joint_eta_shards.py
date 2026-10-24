#!/usr/bin/env python
"""Merge independently evaluated SNR shards of the joint-eta calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--accuracy-shards", nargs="+", required=True)
    p.add_argument("--detail-shards", nargs="+", required=True)
    p.add_argument("--metadata-shards", nargs="+", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--detail-output", required=True)
    p.add_argument("--metadata-output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (
        len(args.accuracy_shards)
        == len(args.detail_shards)
        == len(args.metadata_shards)
    ):
        raise ValueError("Accuracy, detail, and metadata shard counts must match.")

    accuracy = pd.concat(
        [pd.read_csv(path) for path in args.accuracy_shards], ignore_index=True
    )
    detail = pd.concat(
        [pd.read_csv(path) for path in args.detail_shards], ignore_index=True
    )
    # Extension shards repeat eta=0 to satisfy the calibrator's input contract.
    # Remove only byte-equivalent CSV rows; conflicting outcomes still fail the
    # key-uniqueness checks below.
    accuracy = accuracy.drop_duplicates().reset_index(drop=True)
    detail = detail.drop_duplicates().reset_index(drop=True)
    accuracy_key = ["snr_db", "eta_res", "direction_id"]
    detail_key = ["snr_db", "label", "frame_index", "eta_res", "direction_id"]
    if accuracy.duplicated(accuracy_key).any():
        raise ValueError("Duplicate SNR/eta/direction rows across accuracy shards.")
    if detail.duplicated(detail_key).any():
        raise ValueError("Duplicate paired outcomes across detail shards.")

    metadata = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.metadata_shards]
    invariant_keys = (
        "raw_data",
        "feature_data",
        "split_file",
        "split",
        "checkpoint",
        "allocation_grid",
        "direction_definition",
        "envelope_definition",
        "frame_duration_s",
        "sample_rate_hz",
        "samples_per_class_snr",
        "seed",
        "device",
    )
    for key in invariant_keys:
        values = [item[key] for item in metadata]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Metadata mismatch for {key}: {values}")

    output_csv = Path(args.output_csv)
    detail_output = Path(args.detail_output)
    metadata_output = Path(args.metadata_output)
    for path in (output_csv, detail_output, metadata_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    accuracy.sort_values(accuracy_key).to_csv(output_csv, index=False)
    detail.sort_values(detail_key).to_csv(detail_output, index=False)

    payload = {key: metadata[0][key] for key in invariant_keys}
    payload.update(
        {
            "snr_db": sorted(accuracy["snr_db"].astype(float).unique().tolist()),
            "eta_grid_by_snr": {
                f"{float(snr):g}": sorted(group["eta_res"].astype(float).unique().tolist())
                for snr, group in accuracy.groupby("snr_db", sort=True)
            },
            "base_frames": int(detail["frame_index"].nunique()),
            "paired_outcomes": int(len(detail)),
            "direction_evaluations": int(len(accuracy)),
            "elapsed_seconds_sum": float(sum(item["elapsed_seconds"] for item in metadata)),
            "source_metadata": args.metadata_shards,
        }
    )
    metadata_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_csv} ({len(accuracy)} rows)")
    print(f"Wrote {detail_output} ({len(detail)} rows)")


if __name__ == "__main__":
    main()
