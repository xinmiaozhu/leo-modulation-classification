#!/usr/bin/env python
"""Final Protocol-B DCFT representation/HOC ablations (historical filename).

Share the final A/B runner so model settings, HOC preparation, matched seeds,
and paired statistics cannot drift between the two protocols.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main():
    path = Path(__file__).with_name("17_run_representation_ablation_seeds.py")
    spec = importlib.util.spec_from_file_location("final_representation_ablation", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    arguments = sys.argv[1:]
    if "--protocols" not in arguments:
        arguments = ["--protocols", "B", *arguments]
    runner.main(arguments)


if __name__ == "__main__":
    main()
