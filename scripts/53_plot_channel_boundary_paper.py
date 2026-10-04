#!/usr/bin/env python
"""Render existing channel-boundary results without regenerating data or training."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.plotting.channel_boundary import plot_channel_boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="outputs/channel_boundary_full/neural_accuracy.csv")
    parser.add_argument("--output", default="outputs/figures/paper/channel_boundary_representative_accuracy_vs_snr.pdf")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    output = plot_channel_boundary(pd.read_csv(args.results), args.output, dpi=args.dpi)
    print(f"Saved figure and curve data: {output}")


if __name__ == "__main__":
    main()
