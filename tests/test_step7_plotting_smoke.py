"""Smoke test for the current paper's channel-mismatch figure entry point."""

from pathlib import Path
import subprocess
import sys
import pandas as pd

def test_plot_channel_mismatch(tmp_path):
    df = pd.DataFrame([
        {"condition": condition, "snr_db": snr, "n": 100, "accuracy": accuracy}
        for condition in ("unequalized", "pilot_ls", "oracle")
        for snr, accuracy in ((-5, 0.4), (0, 0.7), (5, 0.9))
    ])
    csv = tmp_path / "eval.csv"
    fig = tmp_path / "fig.pdf"
    df.to_csv(csv, index=False)
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "scripts/50_plot_channel_mismatch_trajectory_generalization.py"),
         "--mismatch-csv", str(csv), "--mismatch-output", str(fig)],
        cwd=root, check=True, capture_output=True, text=True, timeout=60,
    )
    assert fig.read_bytes().startswith(b"%PDF-")
