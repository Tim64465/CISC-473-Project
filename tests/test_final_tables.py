"""CPU tests for the final-report tables and results/report_numbers.txt (patch 0006).

    python tests/test_final_tables.py

1. Each new table and report_numbers.txt rebuilt from the committed results equals the committed file.
2. Every number in report_numbers.txt and in the new tables agrees with an independent second
   computation (tests/verify_report_numbers.py: csv module, its own merging and statistics).
3. The build does not depend on Python's hash seed (two builds, PYTHONHASHSEED 1 and 2, identical).
4. report_numbers.txt: five fields per line, unique keys.
5. Helpers on FIXTURE inputs: lowest-at-shown-precision with ties, seed ranges, Pareto set,
   paired/Welch direction (b - a), one-sample interval.
6. F3: the committed f3_selection.json is what `p24.py f3select` writes from the logged runs (only the
   default patch P=16 enters the PatchTST default-LR loss); a selection file carrying the P=64 loss
   of plan grid336 (the bug fixed in patch 0006) stops the build.
7. results/env_D_session1.txt is job D's first-session env.txt from git history (commit 6269e2c).
8. Each new table compiles with pdflatex and fits 6.5 in at the size its header states (skipped if
   pdflatex is not installed).
Rows and files marked FIXTURE are invented test inputs, not results.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import final_tables as FT  # noqa: E402

NEW = ("main_ETTh1", "main_ONZ", "repro_ETTm1", "rq1_decomposition", "f3_selection", "rq2_final", "ablation_cal_ONZ",
       "grid_PxL_H336", "env_per_job", "timing_main")
OK = []


def check(name, ok, detail=""):
    OK.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def build(results_dir, extra_env=None):
    """final_tables.py on a copy of results_dir; returns (rc, output, {table: tex}, report_numbers text)."""
    with tempfile.TemporaryDirectory() as d:
        res, tab = Path(d) / "results", Path(d) / "tables"
        shutil.copytree(results_dir, res)
        env = {"P24_TABLES": str(tab), "PATH": "/usr/bin:/bin", **(extra_env or {})}
        p = subprocess.run([sys.executable, str(KIT / "final_tables.py"), str(res)], capture_output=True, text=True, env=env)
        texs = {f.stem: f.read_text() for f in tab.glob("*.tex")} if tab.exists() else {}
        nums = (res / "report_numbers.txt").read_text() if p.returncode == 0 else ""
        return p.returncode, p.stdout + p.stderr, texs, nums


# 1 -----------------------------------------------------------------------------------------
rc, out, tex, nums = build(KIT / "results")
same = {n: tex.get(n) == (KIT / "tables" / f"{n}.tex").read_text() for n in NEW}
check("1. new tables and report_numbers.txt rebuild byte-identical",
      rc == 0 and all(same.values()) and nums == (KIT / "results" / "report_numbers.txt").read_text(),
      ", ".join(n for n, v in same.items() if not v) or f"{len(NEW)} tables + report_numbers.txt")

# 2 -----------------------------------------------------------------------------------------
p = subprocess.run([sys.executable, str(KIT / "tests" / "verify_report_numbers.py")], capture_output=True, text=True)
check("2. independent second computation agrees with every number", p.returncode == 0,
      " / ".join(l for l in p.stdout.splitlines() if l.startswith(("tables:", "report_numbers.txt:", "OVERALL"))))

# 3 -----------------------------------------------------------------------------------------
r1 = build(KIT / "results", {"PYTHONHASHSEED": "1"})
r2 = build(KIT / "results", {"PYTHONHASHSEED": "2"})
check("3. output independent of the hash seed", r1[0] == r2[0] == 0 and r1[2] == r2[2] and r1[3] == r2[3])

# 4 -----------------------------------------------------------------------------------------
lines = [l for l in (KIT / "results" / "report_numbers.txt").read_text().splitlines() if l and not l.startswith("#")]
keys = [l.split(" | ")[0] for l in lines]
check("4. report_numbers.txt: 5 fields per line, unique keys",
      all(len(l.split(" | ")) == 5 for l in lines) and len(keys) == len(set(keys)), f"{len(lines)} lines")

# 5 -----------------------------------------------------------------------------------------
fx = {"a": 0.3754, "b": 0.3746, "c": 0.376}  # FIXTURE values: a and b both print as 0.375
t1 = FT.best(fx, 3) == {"a", "b"} and FT.best(fx, 4) == {"b"}
t2 = FT.seeds_txt([2023, 2021, 2022]) == "2021-2023" and FT.seeds_txt([2021, 2023]) == "2021,2023" \
    and FT.seeds_txt([2021, 2022, 2023, 2024, 2025], tex=True) == "2021--2025"
pts = [dict(label=k, b32=x, mse=np.array([y, y])) for k, x, y in  # FIXTURE points (latency, MSE)
       (("fast-bad", 1.0, 0.50), ("mid", 2.0, 0.45), ("slow-good", 5.0, 0.40), ("dominated", 3.0, 0.46), ("tie", 2.0, 0.45))]
t3 = FT.pareto(pts) == ["fast-bad", "mid", "slow-good", "tie"]
a, b = np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.4, 3.7])  # FIXTURE
m, lo, hi, pv = FT.paired_ci(a, b)
w = FT.welch_ci(a, b)
o = FT.onesample_ci(b - a)
t4 = np.isclose(m, 0.5333333333) and lo < m < hi and np.isclose(w[0], b.mean() - a.mean()) and np.allclose(o, (m, lo, hi, pv))
check("5. helpers: ties at shown precision, seed ranges, Pareto set, difference direction", t1 and t2 and t3 and t4,
      f"best {t1}, seeds {t2}, pareto {t3}, b-a {t4}")

# 6 -----------------------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    sel = Path(d) / "f3.json"
    p = subprocess.run([sys.executable, str(KIT / "p24.py"), "f3select", str(KIT / "results")], capture_output=True, text=True,
                       env={"P24_F3_SELECTION": str(sel), "PATH": "/usr/bin:/bin"})
    regen = sel.read_text() if sel.exists() else ""
committed_sel = (KIT / "results" / "f3_selection.json").read_text()
with tempfile.TemporaryDirectory() as d:
    res = Path(d) / "results"
    shutil.copytree(KIT / "results", res)
    j = json.loads(committed_sel)
    j["PatchTST"]["336"]["vali_loss_best"]["0.0001"] = 1.3259569  # FIXTURE: the P=64 run's loss, as before the fix
    (res / "f3_selection.json").write_text(json.dumps(j, indent=1))
    rc6, out6, _, _ = build(res)
check("6. f3_selection.json = p24.py f3select on the logged runs (P=16 only); the old P=64 value stops the build",
      p.returncode == 0 and regen == committed_sel and rc6 != 0 and "AssertionError" in out6,
      out6.strip().splitlines()[-1][:120] if out6 else "")

# 7 -----------------------------------------------------------------------------------------
g = subprocess.run(["git", "-C", str(KIT), "show", "6269e2c:results/env_D.txt"], capture_output=True)
if g.returncode == 0:
    check("7. env_D_session1.txt = git show 6269e2c:results/env_D.txt", g.stdout == (KIT / "results" / "env_D_session1.txt").read_bytes())
else:
    print("[SKIP] 7. not a git clone with history (env_D_session1.txt not compared)")

# 8 -----------------------------------------------------------------------------------------
if shutil.which("pdflatex"):
    bad = []
    for n in NEW:
        t = (KIT / "tables" / f"{n}.tex").read_text()
        size = re.search(r"at \\(\w+)\.", t.splitlines()[0]).group(1)
        doc = ("\\documentclass[11pt,letterpaper]{article}\\usepackage{amsmath,amssymb,booktabs,multirow,pifont}"
               "\\usepackage[margin=1in]{geometry}\\newcommand{\\cmark}{\\ding{51}}\\newcommand{\\xmark}{\\ding{55}}"
               f"\\begin{{document}}\\begin{{table}}\\centering\\{size}\\input{{{KIT / 'tables' / n}}}\\end{{table}}\\end{{document}}")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "t.tex").write_text(doc)
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"], cwd=d, capture_output=True)
            log = (Path(d) / "t.log").read_text(errors="replace") if (Path(d) / "t.log").exists() else "no log"
            if not (Path(d) / "t.pdf").exists() or re.search(r"^! ", log, re.M) or "Overfull \\hbox" in log:
                bad.append(n)
    check("8. every new table compiles and fits the text width at its stated size", not bad, ", ".join(bad))
else:
    print("[SKIP] 8. pdflatex not installed")

print("\nOVERALL:", "PASS" if all(OK) else "FAIL", f"({sum(OK)}/{len(OK)} checks)")
sys.exit(0 if all(OK) else 1)
