#!/usr/bin/env python
"""Generate the formal five-seed Protocol-A baseline figure."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "matched_protocol_plot", PROJECT_ROOT / "scripts/47_plot_protocol_b_accuracy_vs_snr.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the matched-seed plotting module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.plot_five_seed("A", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
