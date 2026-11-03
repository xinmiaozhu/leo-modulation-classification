#!/usr/bin/env python
"""Build acquisition features or summarize frozen-classifier attribution."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiments.acquisition_paper import METHODS, build_features, summary, stratified_paired_accuracy_interval
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['features','summary'],default='summary')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    args=p.parse_args()
    if args.stage=='features':
        for method in args.methods: build_features(method)
    else: summary()
