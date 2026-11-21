"""CPU tests for the figures (patch 0006).

    python tests/test_figures.py

1. Two builds of every figure (fig_ablation and the four of plot_final.py) give byte-identical PDFs,
   and no PDF carries a creation date (the build time made ./reproduce.sh tables leave a dirty tree).
2. With the pinned table-build versions (requirements-tables.txt), the rebuilt PDFs equal the committed
   figures/*.pdf byte for byte (skipped, and said so, under another matplotlib version).
3. What each figure plots equals the numbers in results/report_numbers.txt (written by the table code and
   checked independently by tests/verify_report_numbers.py): heatmap cells, RQ2 shift effects and D,
   calendar-ablation differences, and the Pareto sets of MSE vs latency.
4. The PDFs are vector (no embedded raster image).
"""
import importlib.metadata as md
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
FIGS = ("fig_ablation", "fig_pxl_heatmap", "fig_rq2_forest", "fig_cal_ablation", "fig_mse_vs_latency")
OK = []


def check(name, ok, detail=""):
    OK.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def render(out):
    env = dict(os.environ, P24_FIGURES=str(out))
    for script in ("plot_ablation.py", "plot_final.py"):
        p = subprocess.run([sys.executable, str(KIT / script), str(KIT / "results")], capture_output=True, text=True, env=env)
        assert p.returncode == 0, p.stdout + p.stderr
    return {f: (Path(out) / f"{f}.pdf").read_bytes() for f in FIGS}


with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
    a, b = render(d1), render(d2)
same = [f for f in FIGS if a[f] == b[f]]
dated = [f for f in FIGS if b"/CreationDate" in a[f]]
check("1. two builds byte-identical, no creation date in any PDF", len(same) == len(FIGS) and not dated,
      f"identical: {len(same)}/{len(FIGS)}; dated: {dated}")

pins = dict(l.split("==") for l in (KIT / "requirements-tables.txt").read_text().split() if "==" in l)
have = {p: md.version(p) for p in pins}
if have == pins:
    diff = [f for f in FIGS if a[f] != (KIT / "figures" / f"{f}.pdf").read_bytes()]
    check("2. rebuilt figures equal the committed figures/*.pdf (pinned versions)", not diff, f"differ: {diff}" if diff else "")
else:
    print(f"[SKIP] 2. versions {have} differ from requirements-tables.txt {pins}: committed PDFs not compared")

sys.path.insert(0, str(KIT))
import final_tables as FT  # noqa: E402
import plot_final as PF  # noqa: E402
from report_tables import load  # noqa: E402

NUM = {}
for line in (KIT / "results" / "report_numbers.txt").read_text().splitlines():
    if line and not line.startswith("#"):
        k, v, ci, *_ = line.split(" | ")
        NUM[k] = (v, ci)


def fmt(m, lo, hi):
    return f"{m:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]"


PF.setup()
df = load(str(KIT / "results"))
bad = []
with tempfile.TemporaryDirectory() as d:
    hm = PF.fig_pxl_heatmap(df, d)
    for i, P in enumerate(FT.GRID_P):
        for j, L in enumerate(FT.GRID_L):
            if NUM[f"grid_PxL_H336.all.P{P}.L{L}.mse"][0] != f"{hm['mean'][i, j]:.4f} ± {hm['sd'][i, j]:.4f}":
                bad.append(f"heatmap P{P} L{L}")
    key = {("iTransformer", ""): "iTransformer", ("iTransformer", "nots"): "iTransformer_no_tokens", ("PatchTST", ""): "PatchTST",
           ("DLinear", ""): "DLinear", ("iTransformer", "ci"): "Control2"}
    for (H, kind, k), (m, lo, hi, n) in PF.fig_rq2_forest(df, d).items():
        nk = f"rq2.H{H}.{key[k]}.Delta_shift_minus_real" if kind == "delta" else f"rq2.H{H}.D.{key[k[0]]}_vs_{key[k[1]]}"
        if NUM[nk] != fmt(m, lo, hi):
            bad.append(nk)
    names = {FT.CAL_FULL: "full", "cal-dow-hol": "minus_hour", "cal-hour-hol": "minus_dow", "cal-hour-dow": "minus_holiday"}
    for (kind, v), (m, lo, hi, _) in PF.fig_cal_ablation(df, d).items():
        nk = f"ablation_cal_ONZ.{names[v]}.{'minus_PatchTST' if kind == 'vs_pt' else 'minus_full'}.mse"
        if NUM[nk] != fmt(m, lo, hi):
            bad.append(nk)
    lat = PF.fig_mse_vs_latency(df, str(KIT / "results"), d)
    for ds in ("ETTh1", "ONZ"):
        front = {lbl for (dd, lbl), (_, _, _, on) in lat.items() if dd == ds and on}
        if front != set(NUM[f"fig_mse_vs_latency.{ds}.pareto_b32"][0].split(", ")):
            bad.append(f"pareto {ds}")
check("3. plotted values = report_numbers.txt (heatmap, forest, calendar ablation, Pareto sets)", not bad,
      ", ".join(bad) or f"{len(hm['mean'].ravel())} cells, forest, ablation, 2 Pareto sets")

raster = [f for f in FIGS if re.search(rb"/Subtype\s*/Image", a[f])]
check("4. vector PDFs (no embedded raster image)", not raster, f"raster: {raster}" if raster else "")

print("\nOVERALL:", "PASS" if all(OK) else "FAIL", f"({sum(OK)}/{len(OK)} checks)")
sys.exit(0 if all(OK) else 1)
