"""Writes notebooks/P24_run.ipynb (run: python notebooks/make_notebook.py). Edit the cells here."""
import json
from pathlib import Path

MD = r"""# P24 run notebook (Kaggle or Colab, GPU on: Tesla T4 only)

Pick **one JOB** per notebook session, set it in the first code cell, run all cells. Every run is logged as soon
as it finishes. A session starts no new run after `STOP_AFTER_H` hours, so it ends before Kaggle's 12-hour limit
with its output saved; run the same JOB again to continue (finished runs are skipped).
**Resume:** Colab keeps `results_<JOB>` on Drive, so it resumes by itself. Kaggle starts every version empty: commit
the session's `runs.csv` as `results/runs_<JOB>.csv` on branch `final-runs` first, and the next session starts from it.
After every session: download `results_<JOB>.zip` (and `preds_<JOB>.zip` if there is one), then commit
`results_<JOB>/runs.csv` as `results/runs_<JOB>.csv`, `results_<JOB>/env.txt` as `results/env_<JOB>.txt` and
`logs_<JOB>/` as `logs/logs_<JOB>/` on branch `final-runs`. Never commit `preds_<JOB>.zip` (ON-Zonal values are IESO data).
Plan IDs refer to `claude/P24_final_run_plan_and_P1.md` (section 4).

| JOB | Plan ID | What (p24 plan) | Account, in start order | Seeds | Runs |
|---|---|---|---|---|---|
| `V` | V1-V3 | validation gate (`V`): **PASSED 2026-11-11** (`results/runs_V.csv`) | Kaggle-1 | 2021 | 3 |
| `D` | M6 | ETTm1: PatchTST L=336, Autoformer L=96, iTransformer L=96, dual-scored (`ETTm1 --only ...`); several sessions | Kaggle-1 #1 | 2021-23 | 36 |
| `Q` | S4 (phase 1) | F3 LR grid at L=96, seed 2021 (`f3`), then `f3select` -> `f3_selection.json`. Needs job G's rows committed | Kaggle-1 #2 | 2021 | 20 |
| `Q2` | S4 (phase 2) | F3 seeds 2022-23 where a non-default LR won (reads the committed `results/f3_selection.json`) | Kaggle-1 #3 | 2022-23 | <=16 |
| `G` | M3 | ETTh1 same L: PatchTST L=96, iTransformer L=336 (`sameL`); 6 runs must reproduce `runs_C.csv` | Kaggle-2 #1 | 2021-23 | 24 |
| `K` | M7 | ON-Zonal prediction export, seed 2021 real (`onzonal_export`); must reproduce `runs_B.csv` | Kaggle-2 #2 | 2021 | 3 |
| `F` | M2 | RQ2 seeds 2024-25: iTransformer, PatchTST, DLinear, iTransformer-nots (`onzonal_seeds`) | Kaggle-2 #3 | 2024-25 | 16 |
| `H` | M4 | PatchTST+Cal ON-Zonal: full, -hour, -dow, -holiday (`onzonal_cal`) | Kaggle-2 #4 | 2021-23 | 12 |
| `N` | S2 | re-train the 24 official-only ablation runs, dual-scored (`ablation_redo`); must reproduce `runs_C.csv` | Kaggle-2 #5 | 2021-23 | 24 |
| `P` | S3 | RQ2 at H=720 (`onzonal_h720`) | Kaggle-2 #6 | 2021-23 | 18 |
| `W` | S5 | Weather: PatchTST L=336 + iTransformer L=96 (`weather`); only if the section-2 trigger fires | Kaggle-2 #7 | 2021-23 | 24 |
| `E` | M1 | iTransformer without calendar tokens, real + shift (`onzonal_nots`) | Colab-1 #1 | 2021-23 | 6 |
| `L` | M8 | Control 2: channel-independent iTransformer, real + shift (`onzonal_ci`) | Colab-1 #2 | 2021-23 | 6 |
| `J` | M5 | PatchTST+Cal ETTh1 L=336 hour+dow (`etth1_cal`) | Colab-1 #3 | 2021-23 | 12 |
| `T` | M9 | inference timing, batch 1 and 32, every planned config (`timing.py`, no training) | Colab-2 #1 | - | 0 |
| `M` | S1 | P x L grid at H=336 (`grid336`) | Colab-2 #2 | 2021-23 | 30 |
| `S` | - | optional 1-epoch GPU smoke of every new plan, real data (`tests/smoke_all.sh`) | spare | first seed | 1 epoch each |
| `X` | I1 | Weather iTransformer L=336 (`weather_l336`; run `W` first if S5 was not run) | spare | 2021-23 | 12 |
| `Y` | I2 | RQ2 at L=336 (`onzonal_l336`) | spare | 2021-23 | 18 |
| `Z` | I3 | ETTh1 DLinear L=96 + Autoformer L=336 (`sameL_extra`) | spare | 2021-23 | 24 |
| - | I4 | PatchTST+Cal design ablations (injection point, time anchor): **not implemented** | - | - | - |
| - | I5 | DLinear with the 2022 loader: **not implemented** | - | - | - |

Jobs that must reproduce logged rows exactly (G, K, N, Q) run on Kaggle: plan V showed that Kaggle's current
environment reproduces Job A bit for bit, and Jobs A and C agree exactly on the runs they share.
Jobs A, B, C are finished (`results/runs_A/B/C.csv`).

**Kaggle:** Settings -> Accelerator: GPU T4; Internet: On (phone-verified account). Then *Save Version -> Save & Run All*.
Download the outputs from the Output tab. **Colab:** Runtime -> Change runtime type -> T4 GPU. Results go to Google Drive.
"""

CELLS = [
    ("markdown", MD),
    ("code", r'''# ---- parameters ----
JOB = "V"            # one letter (or Q2) from the table above
SEEDS = None         # None = the plan's own seeds (seed 2021 of every config runs first)
BASE = ""            # output folder; "" = /kaggle/working on Kaggle, <Drive>/MyDrive/P24 on Colab
WORK = ""            # repos, data and checkpoints (fast local disk); "" = <system temp>/p24_work
KIT_REPO = "https://github.com/Tim64465/CISC-473-Project"
KIT_BRANCH = "final-runs"
STOP_AFTER_H = 9     # no new run starts after this many hours (Kaggle ends a session at 12 h); 0 = no limit
'''),
    ("code", r'''import os, sys, subprocess, tempfile
ON_KAGGLE = os.path.exists("/kaggle/working")
if not BASE:
    if ON_KAGGLE:
        BASE = "/kaggle/working"
    else:
        from google.colab import drive
        drive.mount("/content/drive")
        BASE = os.path.join("/content/drive/MyDrive", "P24")
WORK = WORK or os.path.join(tempfile.gettempdir(), "p24_work")
KIT = os.path.join(os.path.dirname(WORK), "p24_kit")
os.makedirs(BASE, exist_ok=True)
os.environ["P24_RESULTS"] = os.path.join(BASE, f"results_{JOB}")
os.environ["P24_LOGS"] = os.path.join(BASE, f"logs_{JOB}")
os.environ["P24_WORK"] = WORK
os.environ["P24_TABLES"] = os.path.join(BASE, f"tables_{JOB}")
os.environ["P24_STOP_AFTER_H"] = str(STOP_AFTER_H)
os.makedirs(os.environ["P24_RESULTS"], exist_ok=True)
print("results ->", os.environ["P24_RESULTS"], "| work ->", WORK)
'''),
    ("code", r'''!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install reformer_pytorch==1.4.4 einops==0.8.2 holidays==0.107
import torch, numpy, pandas; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| numpy", numpy.__version__, "| pandas", pandas.__version__)
assert torch.cuda.is_available(), "No GPU: turn the GPU accelerator on, then restart"
if "T4" not in torch.cuda.get_device_name(0):
    print("WARNING: not a Tesla T4. Determinism has only been shown on T4; use T4 for final runs.")
env_txt = os.path.join(os.environ["P24_RESULTS"], "env.txt")
!python -m pip freeze > {env_txt}
!nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> {env_txt}
print("pip freeze ->", env_txt)
'''),
    ("code", r'''import shutil
shutil.rmtree(KIT, ignore_errors=True)
!git clone -q -b {KIT_BRANCH} {KIT_REPO} {KIT}
os.chdir(KIT)
!git rev-parse HEAD | sed 's/^/kit commit /' | tee -a {env_txt}
!python p24.py setup
# Resume: a Kaggle version starts with an empty /kaggle/working. If this JOB's earlier rows are committed
# on the branch (results/runs_<JOB>.csv), start from them, so finished runs are skipped.
prev, cur = os.path.join(KIT, "results", f"runs_{JOB}.csv"), os.path.join(os.environ["P24_RESULTS"], "runs.csv")
if os.path.exists(prev) and not os.path.exists(cur):
    shutil.copy(prev, cur)
    print("resuming from", prev)
'''),
    ("code", r'''seeds = f"--seeds {' '.join(map(str, SEEDS))}" if SEEDS else ""
RUN = "python p24.py run --plan"
ONZ = ["python onzonal.py"]   # builds ON-Zonal from IESO (needs internet)
WTH = ["python p24.py weather-data"]
F3SEL = os.path.join(os.environ["P24_RESULTS"], "f3_selection.json")
JOBS = {
    "V":  [f"{RUN} V"],
    "S":  ONZ + ["P24_SMOKE_SYNTHETIC=0 P24_SMOKE_FULL=1 bash tests/smoke_all.sh"],
    "D":  [f"{RUN} ETTm1 {seeds} --only PatchTST Autoformer iTransformer"],
    "Q":  [f"{RUN} f3 --seeds 2021", f"P24_F3_SELECTION={F3SEL} python p24.py f3select results {os.environ['P24_RESULTS']}"],
    "Q2": [f"{RUN} f3 --seeds {' '.join(map(str, SEEDS or [2022, 2023]))}"],
    "E":  ONZ + [f"{RUN} onzonal_nots {seeds}"],
    "F":  ONZ + [f"{RUN} onzonal_seeds {seeds}"],
    "K":  ONZ + [f"{RUN} onzonal_export {seeds}"],
    "H":  ONZ + [f"{RUN} onzonal_cal {seeds}"],
    "L":  ONZ + [f"{RUN} onzonal_ci {seeds}"],
    "P":  ONZ + [f"{RUN} onzonal_h720 {seeds}"],
    "W":  WTH + [f"{RUN} weather {seeds}"],
    "G":  [f"{RUN} sameL {seeds}"],
    "J":  [f"{RUN} etth1_cal {seeds}"],
    "N":  [f"{RUN} ablation_redo {seeds}"],
    "T":  [f"python timing.py --out {os.path.join(os.environ['P24_RESULTS'], 'timing.csv')}"],
    "M":  [f"{RUN} grid336 {seeds}"],
    "X":  WTH + [f"{RUN} weather_l336 {seeds}"],
    "Y":  ONZ + [f"{RUN} onzonal_l336 {seeds}"],
    "Z":  [f"{RUN} sameL_extra {seeds}"],
}
for cmd in JOBS[JOB]:
    print(">>>", cmd, flush=True)
    !{cmd}
'''),
    ("code", r'''# Quick per-job tables and the GPU time used (merge all jobs later on one machine: ./reproduce.sh tables)
R = os.environ["P24_RESULTS"]
if os.path.exists(os.path.join(R, "runs.csv")):
    !python tables.py {R}
    !python budget.py {R} results | head -12
os.chdir(BASE)
!zip -qr results_{JOB}.zip results_{JOB} logs_{JOB} -x "results_{JOB}/preds/*"
if os.path.isdir(os.path.join(R, "preds")):
    !zip -qr preds_{JOB}.zip results_{JOB}/preds
!ls -la {BASE}
'''),
]


def main():
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    for kind, src in CELLS:
        lines = src.strip("\n").splitlines(True)
        c = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb["cells"].append(c)
    out = Path(__file__).resolve().parent / "P24_run.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
