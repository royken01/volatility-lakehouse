"""I define the color and styling used in for the plots to keep it consisent throughout the thesis
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRID = "#e4e2dc"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#6250d6"
RED = "#e34948"
CATEGORY_COLORS = [BLUE, ORANGE, AQUA, VIOLET, YELLOW, MAGENTA, GREEN, RED]


def event_type_colors() -> dict[str, str]:
    """One colour per configured event type label (``[events] types``), in the configured order."""
    from lakehouse.settings import get_settings

    labels = [t.label for t in get_settings().resolved_event_types]
    return {label: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, label in enumerate(labels)}



def apply_default_style(font_size: float = 9.5) -> None:
    """Light surface, recessive axes, no frame on the top / right, legends without a box."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": SECONDARY_INK, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK, "font.size": font_size,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "legend.frameon": False, "lines.solid_capstyle": "round"})


def clean_axes(axis, grid: str | None = "x") -> None:
    """Remove tick marks and such"""
    axis.tick_params(length=0)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    if grid:
        axis.grid(axis=grid, color=GRID, lw=0.7)
        axis.set_axisbelow(True)


def save_figure(figure, path, dpi: int = 200) -> None:
    """Used to save the figure to a specificed path"""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {path}")
