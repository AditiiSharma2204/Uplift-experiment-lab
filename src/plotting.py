"""Plain matplotlib defaults shared by all figures (white background, tab10 colours, light grid)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = [c["color"] for c in matplotlib.rcParamsDefault["axes.prop_cycle"]]  # tab10
NEUTRAL = "gray"
TEXT_SECONDARY = "#444444"


def apply_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
