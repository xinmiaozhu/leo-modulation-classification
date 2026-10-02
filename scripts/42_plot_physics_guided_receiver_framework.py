#!/usr/bin/env python
"""Draw the physics-guided compensation-first framework used as paper Fig. 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting.common import IEEE_TRANS_PALETTE, set_ieee_trans_style


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/figures/paper/fig1_receiver_framework.pdf")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def box(ax, xy, wh, label, *, face="white", edge="0.25", size=6.0, weight="normal", style="solid"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        facecolor=face, edgecolor=edge, linewidth=0.8, linestyle=style, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size,
            fontweight=weight, linespacing=1.05, zorder=4)
    return patch


def arrow(ax, start, end, *, color="0.25", width=1.0, style="solid", connection="arc3"):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=width,
                        linestyle=style, mutation_scale=7.5,
                        connectionstyle=connection, shrinkA=1, shrinkB=1),
        zorder=2,
    )


def main() -> None:
    cfg = args()
    set_ieee_trans_style()
    blue = IEEE_TRANS_PALETTE["blue"]
    green = IEEE_TRANS_PALETTE["green"]
    purple = IEEE_TRANS_PALETTE["purple"]
    orange = IEEE_TRANS_PALETTE["orange"]
    red = IEEE_TRANS_PALETTE["red"]

    fig, ax = plt.subplots(figsize=(7.16, 4.15))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    ax.text(7.6, 8.55, "Controlled flat-channel Protocols A and B after coarse synchronization",
            ha="center", va="center", fontsize=7.2, fontweight="bold")

    # Compensation-first deployed path.
    box(ax, (0.25, 5.65), (1.35, 1.05), "Received frame\n$y[n]$\n8192 samples", face="#F3F3F3", weight="bold")
    box(ax, (2.00, 6.35), (1.55, 0.82), "Known comb pilots\nRRC + de-rotation", face="#FFF1D6", edge=orange, size=5.0)
    box(ax, (3.95, 6.20), (1.65, 1.12), "A: coherent rate\nB: DFRFT/FFT + local\nCFO--rate refinement", face="#FFF1D6", edge=orange, weight="bold", size=4.55)
    box(ax, (3.95, 4.76), (1.65, 1.00), "Carrier-phase\ncompensation\nA: rate; B: CFO + rate", face="#FFF1D6", edge=orange, weight="bold", size=4.35)
    box(ax, (6.05, 5.10), (1.45, 1.12), "Recovered frame\n$\\widetilde y[n]$\ncarrier residual reduced", face="#E8F2FA", edge=blue, weight="bold", size=4.75)
    arrow(ax, (1.60, 6.18), (2.00, 6.76))
    arrow(ax, (3.55, 6.76), (3.95, 6.76), color=orange, width=1.25)
    arrow(ax, (4.78, 6.20), (4.78, 5.74), color=orange, width=1.25)
    arrow(ax, (1.60, 6.02), (3.95, 5.26), connection="arc3,rad=0.05")
    arrow(ax, (5.60, 5.26), (6.05, 5.66), color=orange, width=1.25)

    # Two representations, visually subordinate to the recovery front end.
    box(ax, (8.05, 6.30), (1.52, 0.90), "Standardized I/Q\n$2\\times8192$", face="#DCEAF7", edge=blue)
    box(ax, (9.95, 6.30), (1.35, 0.90), "I/Q encoder\n32/64/128", face="#DCEAF7", edge=blue)
    box(ax, (8.05, 4.55), (1.52, 1.02), "Exact-mixture +\ngeometry descriptors\n48-D", face="#DFEFD9", edge=green, size=5.25)
    box(ax, (9.95, 4.62), (1.35, 0.88), "Descriptor MLP\n48/64/64/128", face="#DFEFD9", edge=green, size=5.45)
    box(ax, (11.78, 5.15), (1.35, 1.12), "Compact fusion\n256$\\rightarrow$64\n$\\rightarrow$128", face="#E9E0F0", edge=purple, weight="bold")
    box(ax, (13.62, 5.30), (1.25, 0.82), "7-way\nsoftmax", face="#F5DEDC", edge=red, weight="bold")
    arrow(ax, (7.50, 5.82), (8.05, 6.75), color=blue, width=1.2)
    arrow(ax, (7.50, 5.48), (8.05, 5.06), color=green, width=1.2)
    arrow(ax, (9.57, 6.75), (9.95, 6.75), color=blue, width=1.2)
    arrow(ax, (9.57, 5.06), (9.95, 5.06), color=green, width=1.2)
    arrow(ax, (11.30, 6.75), (11.78, 5.92), color=blue, width=1.2)
    arrow(ax, (11.30, 5.06), (11.78, 5.48), color=green, width=1.2)
    arrow(ax, (13.13, 5.71), (13.62, 5.71), color=purple, width=1.25)
    ax.text(11.42, 7.45, "Deployed AMC path", ha="center", fontsize=6.2, color="0.25")

    # Physics/diagnostic loop. It never feeds the classifier.
    ax.plot([0.25, 14.87], [3.70, 3.70], color="0.55", linewidth=0.65, linestyle="--")
    ax.text(0.30, 3.48, "Physics-guided diagnostic and design path (not classifier input)",
            ha="left", va="top", fontsize=6.0, color="0.35", fontstyle="italic")
    box(ax, (0.30, 1.43), (1.75, 1.18), "Physical severity\n$(\\gamma,\\xi)$\n$\\gamma=\\pi|\\mu|T_{\\rm f}^{2}$",
        face="#FAFAFA", edge="0.40", style="dashed", size=5.45)
    box(ax, (2.45, 1.36), (2.05, 1.32), "Signed Fresnel response\n$A_{k,\\xi}(\\gamma)$\nfinite-$N$/attenuation bounds",
        face="#FAFAFA", edge="0.40", style="dashed", size=5.15)
    box(ax, (4.90, 1.36), (1.95, 1.32), "Conditional feature/error\nenvelope: $d_{cd}(\\gamma),P_e$\nsample radius $r(\\gamma;\\mathbf{s})$",
        face="#FAFAFA", edge="0.40", style="dashed", size=4.95)
    box(ax, (7.25, 1.32), (2.10, 1.40), "Explanatory local margin\n+ measured worst-case\naccuracy response",
        face="#FAFAFA", edge="0.40", style="dashed", size=5.15)
    box(ax, (9.75, 1.40), (2.05, 1.24), "Calibration-defined\ntolerances\n$\\gamma_{\\rm AMC}^{\\star}$, $\\eta_{\\rm AMC}^{\\star}$",
        face="#FAFAFA", edge="0.40", style="dashed", size=5.25)
    box(ax, (12.20, 1.27), (2.65, 1.50), "Estimator requirements\nrate: $Q(|e_\\mu|)$ bound\njoint: $Q(\\eta)\\leq\\eta_{\\rm AMC}^{\\star}$",
        face="#FFF1D6", edge=orange, style="dashed", size=5.15, weight="bold")
    for x0, x1 in ((2.05, 2.45), (4.50, 4.90), (6.85, 7.25), (9.35, 9.75), (11.80, 12.20)):
        arrow(ax, (x0, 2.02), (x1, 2.02), color="0.42", width=0.9, style="dashed")
    arrow(ax, (4.78, 6.20), (8.10, 2.72), color="0.48", width=0.8, style="dashed", connection="arc3,rad=0.13")
    arrow(ax, (6.78, 5.10), (8.55, 2.72), color="0.48", width=0.8, style="dashed", connection="arc3,rad=-0.08")

    ax.text(7.6, 0.52,
            "Analytical relations explain the mechanism; independent calibration defines the receiver-specific operating requirements.",
            ha="center", va="center", fontsize=5.7, color="0.30")

    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
