"""C6 tests for report_tables.py (rule 4 and the new columns).

    python tests/test_report_tables.py

1. Regression: the committed results rebuild tables/*.tex byte for byte.
2. A dual-scored re-run with the same run_id (new schema) whose official score equals the
   logged one merges silently: all tables unchanged.
3. The same re-run with a different official score stops the build (determinism check).
4. Rows of variants (nots, ci, cal, lr) never enter the baseline tables.
5. More seeds (2024-25) in some cells: each table uses the seeds present in all its cells;
   with 5 seeds in every RQ2 cell the RQ2 table changes and the caption counts are computed.
Rows marked FIXTURE below are invented test inputs, not results.
"""
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import p24  # noqa: E402

OK = []


def check(name, ok, detail=""):
    OK.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def build(extra_rows):
    """Copy the committed results, add one new-schema file, rebuild; return (rc, stdout, {name: tex})."""
    with tempfile.TemporaryDirectory() as d:
        res, tab = Path(d) / "results", Path(d) / "tables"
        shutil.copytree(KIT / "results", res)
        if extra_rows:
            with open(res / "runs_T.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=p24.FIELDS)
                w.writeheader()
                for r in extra_rows:
                    w.writerow({k: r.get(k, "") for k in p24.FIELDS})
        p = subprocess.run([sys.executable, str(KIT / "report_tables.py"), str(res)], capture_output=True, text=True,
                           env={"P24_TABLES": str(tab), "PATH": "/usr/bin:/bin"})
        texs = {f.name: f.read_text() for f in sorted(tab.glob("*.tex"))} if tab.exists() else {}
        return p.returncode, p.stdout + p.stderr, texs


def logged(run_id):
    for f in sorted((KIT / "results").glob("runs_*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["run_id"] == run_id:
                    return dict(r)
    raise KeyError(run_id)


committed = {f.name: f.read_text() for f in sorted((KIT / "tables").glob("*.tex"))}
rc, out, tex = build([])
check("1. committed results -> tables/*.tex byte-identical", rc == 0 and tex == committed, f"{sorted(tex)}")

v1 = logged("PatchTST_ETTh1_L336_H96_s2021")
bs1 = logged("PatchTST_ETTh1_L336_H96_s2021_bs1")
dual = dict(v1, plan="V", mse=bs1["mse"], mae=bs1["mae"], n_scored=bs1["n_scored"], test_drop_last="False",
            mse_official=v1["mse"], mae_official=v1["mae"], n_scored_official=v1["n_scored"],
            scoring="all+official", variant="")  # FIXTURE: a perfect V1 re-run
rc, out, tex = build([dual])
check("2. dual-scored re-run of runs_A.csv:2 with the same official score -> merged, tables unchanged",
      rc == 0 and tex == committed, out.splitlines()[0])
bad = dict(dual, mse_official=f"{float(v1['mse']) + 1e-6:.6f}")  # FIXTURE: a non-reproducing re-run
rc, out, tex = build([bad])
check("3. re-run with a different official score -> build stops", rc != 0 and "disagree on mse_off" in out,
      out.strip().splitlines()[-1])

variants = []
for rid, var in (("iTransformer_ONZ_real_L96_H96_s2021", "nots"), ("iTransformer_ONZ_shift_L96_H96_s2021", "ci"),
                 ("PatchTST_ETTh1_L336_H96_s2022", "cal-hour-dow"), ("iTransformer_ETTh1_L96_H96_s2023", "lr0.0003")):
    r = logged(rid)  # FIXTURE: same config with a variant tag and an obviously different score
    variants.append(dict(r, run_id=rid + "_" + var, variant=var, mse="9.999999", mae="9.999999",
                         scoring="all+official" if r["model"] == "PatchTST" else "official",
                         mse_official="9.999999" if r["model"] == "PatchTST" else "",
                         mae_official="9.999999" if r["model"] == "PatchTST" else "",
                         n_scored_official=r["n_scored"] if r["model"] == "PatchTST" else ""))
rc, out, tex = build(variants)
check("4. variant rows (nots, ci, cal, lr) do not enter the baseline tables", rc == 0 and tex == committed)

more = []
for m in ("iTransformer", "PatchTST", "DLinear"):
    for ds in ("ONZ_real", "ONZ_shift"):
        base = logged(f"{m}_{ds}_L96_H96_s2021")
        for k, seed in enumerate((2024, 2025)):
            v = float(base["mse"]) + 0.001 * (k + 1)  # FIXTURE values
            r = dict(base, run_id=f"{m}_{ds}_L96_H96_s{seed}", seed=seed, plan="onzonal_seeds", mse=f"{v:.6f}",
                     scoring="all+official" if m == "PatchTST" else "official", variant="")
            if m == "PatchTST":
                r.update(n_scored=base["n_test_windows"], mse_official=f"{v + 0.0002:.6f}", mae_official=base["mae"],
                         n_scored_official=base["n_scored"], test_drop_last="False")
            more.append(r)
rc, out, tex = build(more[:2])  # seeds 2024-25 for one cell only
check("5a. extra seeds in one cell only -> every table uses the seeds common to all its cells (unchanged)",
      rc == 0 and tex == committed, next((l for l in out.splitlines() if "not used" in l), ""))
rc, out, tex = build(more)
changed = sorted(k for k in tex if tex[k] != committed.get(k))
check("5b. seeds 2024-25 in every RQ2 cell -> only onzonal_rq2.tex changes (5 seeds, PatchTST all-window from "
      "the dual rows)", rc == 0 and changed == ["onzonal_rq2.tex"], f"changed: {changed}")
print("\nOVERALL:", "PASS" if all(OK) else "FAIL", f"({sum(OK)}/{len(OK)} checks)")
sys.exit(0 if all(OK) else 1)
