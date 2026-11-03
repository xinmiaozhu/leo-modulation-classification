#!/usr/bin/env python
"""Evaluate four front ends; timing is a separate stage."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiments.acquisition_paper import estimate_test, timing
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['estimates','timing','all'],default='estimates')
    args=p.parse_args()
    if args.stage in ('estimates','all'): estimate_test()
    if args.stage in ('timing','all'): timing()
