#!/usr/bin/env python
"""Run the independent four-length, five-seed Protocol-B experiment."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.frame_length import main

if __name__ == '__main__':
    main()
