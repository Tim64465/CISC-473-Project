"""One figure style for every P24 figure, and reproducible output.

Colours: ink and grid of fig_ablation (Figure 1 of the midterm) and the first three slots of the
validated categorical palette it uses (blue, orange, aqua; at most three hues per figure, each
series also carries its own marker so the figures read in greyscale). Sequential maps use one
hue (blue), light to dark.

Reproducible files: PDFs are written without a creation date (matplotlib otherwise stores the
build time), so rebuilding from the same results with the same matplotlib version gives the same
bytes. PNG copies (for slides) are written next to the PDFs and are git-ignored.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ_BLUE = LinearSegmentedColormap.from_list("p24_blue", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
    "#1c5cab", "#184f95", "#104281"])
FIG_DIR = Path(os.environ.get("P24_FIGURES", Path(__file__).resolve().parent / "figures"))  # tests write elsewhere
FULL_W = 7.2  # inches, as fig_ablation


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
                         "pdf.compression": 6, "svg.hashsalt": "p24"})


def tidy(ax, grid_axis="y"):
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name, out=None):
    """figures/<name>.pdf (no creation date) and a git-ignored PNG copy."""
    out = Path(out) if out else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(out / f"{name}.png", dpi=200, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {out / (name + '.pdf')}")
    return out / f"{name}.pdf"
