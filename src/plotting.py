"""Shared matplotlib style for this repo's figures.

Plain matplotlib only, no seaborn -- keeps the dependency list short and every figure
looks like it came from the same repo.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: these scripts run from the CLI, never in a GUI session
import matplotlib.pyplot as plt

COLORS = {
    "honest": "#1b7837",
    "proxy": "#c2410c",
    "neutral": "#374151",
    "muted": "#9ca3af",
    "grid": "#d9d9d9",
}


def set_style() -> None:
    """Apply the repo-wide rcParams. Call once at the top of a script's __main__ block."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.5, 4.8),
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
        }
    )


def save_figure(fig: "plt.Figure", path: str | Path) -> None:
    """Write `fig` to `path`, creating the parent directory if needed, then close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
