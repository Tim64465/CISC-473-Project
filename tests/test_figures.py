"""CPU tests for the figures (patch 0006; checks 2 and 5 from patch 0010).

    python tests/test_figures.py

1. Two builds of every figure (fig_ablation and the four of plot_final.py) give byte-identical PDFs,
   and no PDF carries a creation date (the build time made ./reproduce.sh tables leave a dirty tree).
2. With the pinned table-build versions (requirements-tables.txt), the rebuilt PDFs equal the committed
   figures/*.pdf byte for byte (skipped, and said so, under another matplotlib version). "Committed" means
   the files in git HEAD, read with `git show`, not the working tree: ./reproduce.sh tables overwrites
   figures/ first, so a working-tree comparison would compare the rebuild with itself.
3. What each figure plots equals the numbers in results/report_numbers.txt (written by the table code and
   checked independently by tests/verify_report_numbers.py): heatmap cells, RQ2 shift effects and D,
   calendar-ablation differences, and the Pareto sets of MSE vs latency.
4. The PDFs are vector (no embedded raster image).
5. The PDFs are uncompressed (no /Filter on any stream), so their bytes do not depend on the zlib inside
   the Python build (Python 3.14.6 ships zlib 1.3.2 and compressed the same drawing differently than 3.13),
   and git treats them as binary (.gitattributes), so it never converts their line endings or whitespace.
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

FIG_PKGS = ("numpy", "pandas", "scipy", "matplotlib")  # the versions the figure bytes depend on
pins = {p: v for p, v in (l.split("==") for l in (KIT / "requirements-tables.txt").read_text().split() if "==" in l)
        if p in FIG_PKGS}
have = {p: md.version(p) for p in pins}


def committed(f):
    """figures/<f>.pdf as committed in git HEAD; the working-tree file only if the kit is not a git checkout."""
    p = subprocess.run(["git", "-C", str(KIT), "show", f"HEAD:figures/{f}.pdf"], capture_output=True)
    return (p.stdout, "git HEAD") if p.returncode == 0 else ((KIT / "figures" / f"{f}.pdf").read_bytes(), "working tree, not a git checkout")


if have == pins:
    ref = {f: committed(f) for f in FIGS}
    diff = [f for f in FIGS if a[f] != ref[f][0]]
    check("2. rebuilt figures equal the committed figures/*.pdf (pinned versions)", not diff,
          (f"differ: {diff}; " if diff else "") + f"compared with {ref[FIGS[0]][1]}")
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

filtered = [f for f in FIGS if re.search(rb"/Filter", a[f])]
attr = subprocess.run(["git", "-C", str(KIT), "check-attr", "text", "diff", "--"] + [f"figures/{f}.pdf" for f in FIGS],
                      capture_output=True, text=True)
textlike = [] if attr.returncode else sorted({ln.split(":")[0] for ln in attr.stdout.splitlines() if not ln.endswith(": unset")})
check("5. uncompressed PDFs (no /Filter), so the bytes do not depend on Python's zlib build; git treats them as binary",
      not filtered and not textlike, f"compressed: {filtered}; not binary in git: {textlike}" if filtered or textlike
      else ("git attributes: text and diff unset" if not attr.returncode else "not a git checkout"))

print("\nOVERALL:", "PASS" if all(OK) else "FAIL", f"({sum(OK)}/{len(OK)} checks)")
sys.exit(0 if all(OK) else 1)
