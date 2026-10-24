#!/usr/bin/env python
"""Publish the frame-length table and reference-matched figure after all 60 runs pass validation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.frame_length_report import main

if __name__ == '__main__':
    main()
