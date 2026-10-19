"""Figure 1 of the midterm report: PatchTST patch-length and look-back ablation on ETTh1.

    python plot_ablation.py results            # writes figures/fig_ablation.pdf

Mean +- sample std over seeds (lines, error bars), each seed as a faint dot,
published PatchTST/42 (Nie et al., Table 3) as a star. All runs are scored with
PatchTST's official test loader, as the published numbers were.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_tables import ablation_cells, load  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {96: ("#2a78d6", "o"), 720: ("#eb6834", "s")}  # validated categorical slots 1-2
PUBLISHED = {96: 0.375, 720: 0.449}  # PatchTST/42, ETTh1 (Nie et al. 2023, Table 3)


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else "results"
    cells = ablation_cells(load(results))
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    panels = (("P", (8, 16, 32, 64), 16, "patch length $P$ (stride $P/2$), $L=336$", "(a) Patch length"),
              ("L", (96, 336, 720), 336, "look-back $L$ (steps), $P=16$", "(b) Look-back"))
    for ax, (kind, xs, default, xlabel, title) in zip(axes, panels):
        pos = np.arange(len(xs))
        for H, (col, mk) in SERIES.items():
            vals = [cells[(kind, x, H)] for x in xs]
            mean = np.array([v.mean() for v in vals])
            sd = np.array([v.std(ddof=1) for v in vals])
            for i, v in enumerate(vals):
                ax.scatter(np.full(len(v), pos[i]) + np.array([-0.06, 0, 0.06]), v, s=9, color=col,
                           alpha=0.28, linewidths=0, zorder=2)
            ax.errorbar(pos, mean, yerr=sd, color=col, marker=mk, ms=5.5, lw=2, capsize=2.5,
                        elinewidth=1, zorder=3, label=f"$H={H}$")
            ax.annotate(f"H={H}", (pos[-1], mean[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
            ax.scatter([pos[xs.index(default)]], [PUBLISHED[H]], marker="*", s=90, facecolor="white",
                       edgecolor=INK, linewidths=0.9, zorder=4)
        ax.set_xticks(pos, [str(x) for x in xs])
        ax.set_xlim(-0.35, len(xs) - 0.35)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("test MSE ($z$-scored)")
    handles, labels = axes[0].get_legend_handles_labels()
    star = plt.Line2D([], [], marker="*", ls="", ms=10, markerfacecolor="white", markeredgecolor=INK)
    fig.legend(handles + [star], labels + ["published PatchTST/42 (Nie et al., Table 3)"],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig_ablation.pdf")
    fig.savefig(out / "fig_ablation.png", dpi=200)
    print(f"wrote {out / 'fig_ablation.pdf'}")


if __name__ == "__main__":
    main()
