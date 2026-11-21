"""Final-report figures from the run logs (patch 0006).

    python plot_final.py results        # writes figures/fig_*.pdf (+ git-ignored PNG copies)

  fig_pxl_heatmap     PatchTST P x L grid on ETTh1 at H=336 (every test window): mean MSE per cell
  fig_rq2_forest      RQ2 shift effects (paired) and D (Welch) with 95% intervals, H=96 and H=720
  fig_cal_ablation    PatchTST+Cal on ON-Zonal: paired differences vs PatchTST and vs the full model
  fig_mse_vs_latency  test MSE (every window) against median forward latency at batch 32 (job T),
                      ETTh1 H=336 and ON-Zonal H=96; the Pareto set is marked and labelled

Every number plotted comes from final_tables.py (the same functions that write the tables), so a
figure and its table cannot disagree; each fig_* function returns what it plotted (used by the tests).
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_tables as FT  # noqa: E402
from p24_plotstyle import AQUA, BLUE, FULL_W, GRID, INK, INK2, MUTED, ORANGE, SEQ_BLUE, plt, save, setup, tidy  # noqa: E402
from report_tables import load  # noqa: E402


# ------------------------------------------------------------------ P x L heatmap
def fig_pxl_heatmap(df, out=None):
    g, seeds = FT.grid_cells(df, "all")
    mean = np.array([[g[(p, L)].mean() for L in FT.GRID_L] for p in FT.GRID_P])
    sd = np.array([[g[(p, L)].std(ddof=1) for L in FT.GRID_L] for p in FT.GRID_P])
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    # pcolormesh (not imshow): one vector rectangle per cell, so the PDF has no embedded raster image
    im = ax.pcolormesh(np.arange(len(FT.GRID_L) + 1) - 0.5, np.arange(len(FT.GRID_P) + 1) - 0.5, mean, cmap=SEQ_BLUE,
                       edgecolors="white", linewidth=1.0)
    ax.invert_yaxis()
    lo, hi = mean.min(), mean.max()
    best = np.unravel_index(np.argmin(mean), mean.shape)
    for i in range(len(FT.GRID_P)):
        for j in range(len(FT.GRID_L)):
            dark = (mean[i, j] - lo) / (hi - lo) > 0.55
            ax.text(j, i, f"{mean[i, j]:.4f}\n$\\pm${sd[i, j]:.4f}", ha="center", va="center", fontsize=7.5,
                    color="white" if dark else INK, fontweight="bold" if (i, j) == best else "normal")
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, ec=INK, lw=1.8))
    d = (FT.GRID_P.index(16), FT.GRID_L.index(336))
    ax.add_patch(plt.Rectangle((d[1] - 0.46, d[0] - 0.46), 0.92, 0.92, fill=False, ec=INK2, lw=1.0, ls=(0, (3, 2))))
    ax.set_xticks(range(len(FT.GRID_L)), [str(L) for L in FT.GRID_L])
    ax.set_yticks(range(len(FT.GRID_P)), [f"{p} ({p // 2})" for p in FT.GRID_P])
    ax.set_xlabel("look-back $L$ (steps)")
    ax.set_ylabel("patch length $P$ (stride $S$)")
    ax.set_title(f"PatchTST on ETTh1, $H=336$: test MSE\n(every window, mean $\\pm$ std, {len(seeds)} seeds)",
                 loc="left", fontsize=8.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)  # matplotlib rasterizes long colorbars by default; keep the PDF all-vector
    cb.ax.tick_params(labelsize=7.5, color=INK2)
    fig.text(0.01, 0.01, "solid box: lowest mean; dashed: PatchTST default ($P{=}16$, $L{=}336$)", fontsize=7,
             color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_pxl_heatmap", out)
    return dict(mean=mean, sd=sd, seeds=seeds, best=(FT.GRID_P[best[0]], FT.GRID_L[best[1]]))


# ------------------------------------------------------------------ RQ2 forest plot
FOREST_ROWS = (  # (kind, key, label)
    ("delta", ("iTransformer", ""), "$\\Delta$ iTransformer"),
    ("delta", ("iTransformer", "nots"), "$\\Delta$ iTransformer, no tokens"),
    ("delta", ("PatchTST", ""), "$\\Delta$ PatchTST"),
    ("delta", ("DLinear", ""), "$\\Delta$ DLinear"),
    ("delta", ("iTransformer", "ci"), "$\\Delta$ Control 2"),
    ("D", (("iTransformer", ""), ("PatchTST", "")), "$D$ vs PatchTST"),
    ("D", (("iTransformer", ""), ("DLinear", "")), "$D$ vs DLinear"),
    ("D", (("iTransformer", ""), ("iTransformer", "ci")), "$D$ vs Control 2"),
    ("D", (("iTransformer", "nots"), ("PatchTST", "")), "$D$ (no tokens) vs PatchTST"),
    ("D", (("iTransformer", "nots"), ("DLinear", "")), "$D$ (no tokens) vs DLinear"),
)


def fig_rq2_forest(df, out=None):
    st = {H: FT.rq2_stats(df, H) for H in (96, 720)}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.3), sharey=True)
    plotted = {}
    y = np.arange(len(FOREST_ROWS))[::-1].astype(float)
    y[5:] -= 0.6  # a gap between the shift effects and the D rows
    for ax, H in zip(axes, (96, 720)):
        s = st[H]
        for yy, (kind, key, lbl) in zip(y, FOREST_ROWS):
            if kind == "delta":
                r = s["rows"].get(key)
                res, n = (r["delta"], len(r["seeds"])) if r else (None, 0)
            else:
                d = s["D"].get(key)
                res, n = (d[0], len(d[1])) if d else (None, 0)
            if res is None:
                ax.text(1.02, yy, "not run", transform=ax.get_yaxis_transform(), ha="left", va="center",
                        fontsize=6.5, color=MUTED, clip_on=False)
                continue
            m, lo, hi = res[:3]
            col, mk = (INK2, "o") if kind == "delta" else (BLUE, "D")
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.6, solid_capstyle="butt", zorder=2)
            ax.plot([m], [yy], marker=mk, ms=6, color=col, mec="white", mew=0.8, zorder=3, ls="")
            ax.text(1.02, yy, f"n={n}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5,
                    color=MUTED, clip_on=False)
            plotted[(H, kind, key)] = (m, lo, hi, n)
        ax.axvline(0, color=INK, lw=0.8, zorder=1)
        ax.set_title(f"$H={H}$", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("MSE difference (95% CI)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}".replace("-", "\u2212") if v else "0"))
        tidy(ax, "x")
        ax.margins(x=0.08)
    axes[0].set_yticks(y, [r[2] for r in FOREST_ROWS])
    axes[0].tick_params(axis="y", length=0)
    h1 = plt.Line2D([], [], color=INK2, marker="o", lw=1.6, mec="white", label="$\\Delta$ = shifted $-$ real (paired)")
    h2 = plt.Line2D([], [], color=BLUE, marker="D", lw=1.6, mec="white",
                    label="$D=\\Delta_{\\mathrm{iTransformer}}-\\Delta_{\\mathrm{control}}$ (Welch)")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=4)
    save(fig, "fig_rq2_forest", out)
    return plotted


# ------------------------------------------------------------------ PatchTST+Cal ablation
def fig_cal_ablation(df, out=None):
    st, seeds = FT.cal_stats(df)
    pt = st[""]["mse"]
    full = st[FT.CAL_FULL]["mse"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4), gridspec_kw=dict(width_ratios=(4, 3)))
    plotted = {}
    panels = ((axes[0], [(n, v) for n, v, _ in FT.CAL_VARIANTS], "vs_pt", pt,
               "(a) PatchTST+Cal $-$ PatchTST", "MSE difference (paired 95% CI)"),
              (axes[1], [(n, v) for n, v, _ in FT.CAL_VARIANTS[1:]], "vs_full", full,
               "(b) cost of removing a term: variant $-$ full", "MSE difference (paired 95% CI)"))
    for ax, items, key, base, title, xlabel in panels:
        ys = np.arange(len(items))[::-1]
        for yy, (name, v) in zip(ys, items):
            m, lo, hi = st[v][key][:3]
            per_seed = st[v]["mse"] - base
            col = BLUE if key == "vs_pt" else ORANGE
            ax.scatter(per_seed, np.full(len(per_seed), yy) + np.linspace(-0.12, 0.12, len(per_seed)), s=10,
                       color=col, alpha=0.35, linewidths=0, zorder=2)
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.8, solid_capstyle="butt", zorder=3)
            ax.plot([m], [yy], marker="o" if key == "vs_pt" else "s", ms=6.5, color=col, mec="white", mew=0.8, ls="",
                    zorder=4)
            ax.annotate(f"{m:+.4f}".replace("-", "−"), (hi, yy), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK)
            plotted[(key, v)] = (m, lo, hi, per_seed)
        ax.axvline(0, color=INK, lw=0.8, zorder=1)
        ax.set_yticks(ys, [n.replace("$-$", "−") for n, _ in items])
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel(xlabel)
        ax.margins(x=0.25, y=0.15)
        tidy(ax, "x")
    fig.text(0.01, 0.01, f"ON-Zonal, $L=96$, $H=96$, every window; dots: the {len(seeds)} seeds "
             f"({FT.seeds_txt(seeds)}); bar: 95% CI", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_cal_ablation", out)
    return plotted


# ------------------------------------------------------------------ MSE vs latency
FAMILY = {"PatchTST": (BLUE, "o"), "PatchTST+Cal": (BLUE, "D"), "iTransformer": (ORANGE, "s"),
          "DLinear": (AQUA, "^"), "Autoformer": (INK2, "P")}


def place_labels(ax, items, avoid):
    """Greedy label placement. Each label goes to the first free spot among 8 directions at growing
    distances from its point, else to the nearest free spot of a grid over the axes (with a leader
    line). Free: inside the axes and overlapping no marker and no earlier label. items: (x, y, text)."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(r)
    taken = [tuple(ax.transData.transform(p)) for p in avoid]
    boxes = []
    pt2px = fig.dpi / 72
    dirs = ((1, 0, "left", "center"), (1, 1, "left", "bottom"), (0, 1, "center", "bottom"), (-1, 1, "right", "bottom"),
            (-1, 0, "right", "center"), (-1, -1, "right", "top"), (0, -1, "center", "top"), (1, -1, "left", "top"))

    def free(bb, own):
        inside = bb.x0 >= axbox.x0 and bb.x1 <= axbox.x1 and bb.y0 >= axbox.y0 and bb.y1 <= axbox.y1
        return inside and not any(bb.overlaps(b) for b in boxes) and not any(
            bb.x0 - 4 <= px <= bb.x1 + 4 and bb.y0 - 4 <= py <= bb.y1 + 4 for px, py in taken if (px, py) != own)

    def put(x, y, t, dx, dy, ha, va, line):
        return ax.annotate(t, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7, color=INK,
                           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=3) if line else None)

    for x, y, t in items:
        own = tuple(ax.transData.transform((x, y)))
        best = None
        for rad in (7, 16, 26):
            for dx, dy, ha, va in dirs:
                ann = put(x, y, t, dx * rad, dy * rad, ha, va, False)  # measure the text only
                bb = ann.get_window_extent(r).expanded(1.04, 1.2)
                ann.remove()
                if free(bb, own):
                    best = (dx * rad, dy * rad, ha, va, rad > 7 or (dx, dy) != (1, 0), bb)
                    break
            if best:
                break
        if best is None:  # grid fallback: nearest free spot, label centred there
            for gx in np.arange(axbox.x0 + 2, axbox.x1, 6):
                for gy in np.arange(axbox.y0 + 2, axbox.y1, 4):
                    d = np.hypot(gx - own[0], gy - own[1])
                    if best is not None and d >= best[-1]:
                        continue
                    ann = put(x, y, t, (gx - own[0]) / pt2px, (gy - own[1]) / pt2px, "left", "bottom", False)
                    bb = ann.get_window_extent(r).expanded(1.04, 1.2)
                    ann.remove()
                    if free(bb, own):
                        best = ((gx - own[0]) / pt2px, (gy - own[1]) / pt2px, "left", "bottom", True, bb, d)
        assert best is not None, f"no free spot for label {t!r}"
        put(x, y, t, *best[:5])
        boxes.append(best[5])


def short(label):
    """Point labels: the marker gives the family (legend), the label the setting."""
    return (label.replace("PatchTST+Cal ", "Cal ").replace("PatchTST ", "").replace("iTransformer", "iTrans.")
            .replace(", no tokens", " (no tok.)"))


def fig_mse_vs_latency(df, results, out=None, x="b32"):
    pts = FT.latency_points(df, results)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.3))
    plotted = {}
    titles = {"ETTh1": f"(a) ETTh1, $H={FT.LAT_ETTH1_H}$", "ONZ": "(b) ON-Zonal (real zones), $H=96$"}
    for ax, ds in zip(axes, ("ETTh1", "ONZ")):
        ps = pts[ds]
        front = FT.pareto(ps, x)
        for p in ps:
            col, mk = FAMILY[p["family"]]
            on = p["label"] in front
            m, s = p["mse"].mean(), p["mse"].std(ddof=1)
            q25, q75 = p["q_" + x]
            ax.errorbar([p[x]], [m], yerr=[s], xerr=[[p[x] - q25], [q75 - p[x]]], color=col, lw=0.8, capsize=1.2,
                        zorder=2, ls="")
            ax.plot([p[x]], [m], marker=mk, ms=7 if on else 5.5, ls="", zorder=4 if on else 3,
                    mfc=col if on else "white", mec=INK if on else col, mew=1.1)
            plotted[(ds, p["label"])] = (p[x], m, s, on)
        fr = sorted((p for p in ps if p["label"] in front), key=lambda p: p[x])
        ax.step([p[x] for p in fr], [p["mse"].mean() for p in fr], where="post", color=INK, lw=0.9, ls=(0, (3, 2)),
                zorder=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.margins(x=0.12, y=0.12)
        ax.set_xlabel(f"median forward latency, batch {x[1:]} (ms, log)")
        ax.set_title(titles[ds], loc="left", fontsize=9, color=INK)
        tidy(ax, "both")
    axes[0].set_ylabel("test MSE, every window")
    handles = [plt.Line2D([], [], marker=mk, ls="", mfc="white", mec=col, mew=1.1, ms=6, label=f)
               for f, (col, mk) in FAMILY.items()]
    handles.append(plt.Line2D([], [], marker="o", ls=(0, (3, 2)), color=INK, mfc=INK2, mec=INK, ms=6,
                              label="Pareto set (filled)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.5)
    fig.text(0.01, 0.105, "bars: std over seeds (MSE), interquartile range of 50 timed passes (latency)", fontsize=6.5,
             color=INK2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ax, ds in zip(axes, ("ETTh1", "ONZ")):
        ps = pts[ds]
        front = FT.pareto(ps, x)
        fr = sorted((p for p in ps if p["label"] in front), key=lambda p: -p["mse"].mean())
        path = []  # points along the dashed Pareto line, so labels avoid it too
        srt = sorted(fr, key=lambda p: p[x])
        for a, b in zip(srt, srt[1:]):
            path += [(v, a["mse"].mean()) for v in np.geomspace(a[x], b[x], 25)]
            path += [(b[x], v) for v in np.linspace(a["mse"].mean(), b["mse"].mean(), 25)]
        place_labels(ax, [(p[x], p["mse"].mean(), short(p["label"])) for p in fr],
                     [(p[x], p["mse"].mean()) for p in ps] + path)
    save(fig, "fig_mse_vs_latency", out)
    return plotted


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else "results"
    setup()
    df = load(results)
    fig_pxl_heatmap(df)
    fig_rq2_forest(df)
    fig_cal_ablation(df)
    fig_mse_vs_latency(df, results)


if __name__ == "__main__":
    main()
