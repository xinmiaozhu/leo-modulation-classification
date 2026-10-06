"""Common plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


# Shared visual language for the paper figures.  The settings intentionally
# follow the compact, line-driven convention used by the reference AMC paper:
# white canvas, high-contrast primary colors, outward ticks, fine dash-dot
# grids, and line/marker pairs that remain distinguishable in grayscale.
IEEE_TRANS_PALETTE = {
    "black": "#000000",
    "gray": "#666666",
    "blue": "#1677C8",
    "orange": "#F08A00",
    "green": "#2E9F3E",
    "purple": "#7E3FB2",
    "red": "#E32624",
    "cyan": "#00A6C8",
    "magenta": "#C73EAE",
}


def set_ieee_trans_style() -> None:
    """Apply a consistent compact style to non-confusion paper figures."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            # Match the typography used by manuscript Fig. 2.
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "lines.markersize": 2.8,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def format_ieee_axis(ax, *, grid: bool = True) -> None:
    """Apply shared axes, tick, and grid styling without changing data limits."""

    # Keep the enclosing frame but reserve ticks for the actual left and bottom axes.
    ax.tick_params(direction="out", top=False, right=False, width=0.7, length=3.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    if grid:
        ax.grid(True, color="0.72", linestyle="-.", linewidth=0.4)


def set_panel_aspect_10_9(ax) -> None:
    """Set a 10:9 width-to-height plotting region for paper result figures."""

    ax.set_box_aspect(0.9)


def boxed_legend(ax, **kwargs):
    """Create the square opaque legends used consistently in the manuscript."""

    defaults = {
        "frameon": True,
        "fancybox": False,
        "edgecolor": "black",
        "facecolor": "white",
        "framealpha": 1.0,
        "borderpad": 0.28,
        "handlelength": 1.7,
        "handletextpad": 0.45,
        "labelspacing": 0.3,
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


def add_panel_label(
    ax,
    label: str,
    *,
    y: float = -0.28,
    fontsize: float = 8.0,
) -> None:
    """Place a compact IEEE-style panel identifier below an axes."""

    ax.text(
        0.5,
        y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="normal",
    )


def save_axes_panels(
    fig: plt.Figure,
    axes,
    output: str | Path,
    *,
    suffixes: Iterable[str] | None = None,
    pad_inches: float = 0.02,
) -> list[Path]:
    """Save each axes of a multi-panel figure as a separate panel file.

    The manuscript can include the returned ``*_a.pdf``, ``*_b.pdf``, ... files individually.
    Inset axes attached to a parent axes are included in the parent panel crop.
    """

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    flat_axes = np.asarray(axes, dtype=object).ravel().tolist()
    if suffixes is None:
        suffixes = [chr(ord("a") + i) for i in range(len(flat_axes))]
    suffixes = list(suffixes)
    if len(suffixes) != len(flat_axes):
        raise ValueError("suffixes must have the same length as axes")

    saved: list[Path] = []
    for suffix, ax in zip(suffixes, flat_axes):
        # Hide the other top-level axes while exporting this panel.  Merely
        # cropping to ``ax`` is insufficient when a neighbouring y-label or
        # legend extends into the target crop.
        visibility = {other: other.get_visible() for other in flat_axes}
        try:
            for other in flat_axes:
                if other is not ax:
                    other.set_visible(False)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bboxes = [ax.get_tightbbox(renderer)]
            for child in getattr(ax, "child_axes", []):
                bboxes.append(child.get_tightbbox(renderer))
            bbox = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
            panel_path = output.with_name(f"{output.stem}_{suffix}{output.suffix}")
            fig.savefig(panel_path, bbox_inches=bbox, pad_inches=pad_inches)
            saved.append(panel_path)
        finally:
            for other, was_visible in visibility.items():
                other.set_visible(was_visible)
    fig.canvas.draw()
    return saved
