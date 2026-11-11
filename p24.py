"""P24 run kit: set up the official repos, run experiments, log every run.

Commands
  python p24.py setup                      clone pinned repos, patch, add ETT data
  python p24.py run --plan A --seeds 2021 2022 2023   ETTh1 reproduction (+F1 re-scoring)
  python p24.py run --plan ETTm1 --seeds 2021         ETTm1 reproduction (dual-scored)
  python p24.py run --plan ablation --seeds 2021      PatchTST patch length / look-back
  python p24.py run --plan onzonal --seeds 2021 2022 2023   RQ2 preliminary (needs onzonal.py first)
  python p24.py run --plan V                         validation gate V1-V3 (prints PASS/FAIL)
  python p24.py run --plan <name>                    any plan in PLANS below (default seeds per plan)
  python p24.py weather-data                         download + checksum the Weather benchmark
  python p24.py f3select [results_dir ...]           F3: pick the LR with the lowest validation loss (seed 2021)
  python p24.py check-v [--smoke]                    re-print the V1-V3 checks from the logged rows
  python p24.py tables                     build result tables from results/runs.csv

Every run is appended to results/runs.csv as soon as it finishes, so a
disconnect loses at most one run. Re-running a plan skips runs already logged.
Each run's full console output is kept in logs/<run_id>.log.
Smoke tests (--smoke) train 1 epoch and log to results/smoke_runs.csv only.
P24_STOP_AFTER_H=<hours>: start no new run after that many hours (for platforms that end
sessions at a fixed time); run the plan again to continue.

Scoring (column `scoring`, rows logged from branch final-runs on; blank in older rows):
  official      the model's own test loader, unchanged
  all           every test window, batch-size-1 re-score of a checkpoint (mode rescore_bs1)
  all+official  PatchTST, one test pass (P24_ALL_WINDOWS=1): `mse`/`mae` over every window,
                `mse_official`/`mae_official` over the first n_scored_official windows, which
                are exactly the windows and predictions of the official loader.
In every row, old and new, `mse` and `mae` are computed over `n_scored` of the
`n_test_windows` test windows; every window counts if and only if n_scored == n_test_windows.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

KIT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("P24_WORK", KIT / "work"))
REPOS = WORK / "repos"
RESULTS = Path(os.environ.get("P24_RESULTS", KIT / "results"))
LOGS = Path(os.environ.get("P24_LOGS", KIT / "logs"))

# Official repositories, pinned to the commits we checked on 2026-10-17.
COMMITS = {
    "PatchTST": ("https://github.com/yuqinie98/PatchTST", "204c21efe0b39603ad6e2ca640ef5896646ab1a9"),
    "LTSF-Linear": ("https://github.com/cure-lab/LTSF-Linear", "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"),
    "Autoformer": ("https://github.com/thuml/Autoformer", "51c7d416ae120b805fd5beef2f4ccf7de496a6ff"),
    "iTransformer": ("https://github.com/thuml/iTransformer", "c2426e68ca13f74aaec08045c5c724d8ad328124"),
}
REPO_DIR = {  # working directory each repo's scripts expect
    "PatchTST": REPOS / "PatchTST" / "PatchTST_supervised",
    "LTSF-Linear": REPOS / "LTSF-Linear",
    "Autoformer": REPOS / "Autoformer",
    "iTransformer": REPOS / "iTransformer",
}
DATA_SUBDIR = {"PatchTST": "dataset", "LTSF-Linear": "dataset",
               "Autoformer": "dataset/ETT-small", "iTransformer": "dataset/ETT-small"}
MODEL_REPO = {"PatchTST": "PatchTST", "DLinear": "LTSF-Linear",
              "Autoformer": "Autoformer", "iTransformer": "iTransformer"}

ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{}.csv"
ETT_SHA256 = {  # files as downloaded on 2026-10-17
    "ETTh1": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    "ETTm1": "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
}
# Weather: the official download in the pinned Autoformer repo (utils/download_data.py:4),
# a zip of the six benchmarks; we keep only weather/weather.csv.
WEATHER_URL = "https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/files/?p=%2Fall_six_datasets.zip&dl=1"
WEATHER_SHA256 = None  # \TODO{record the sha256 printed by the first verified download, then commit}
WEATHER_DIR = {"PatchTST": "dataset", "iTransformer": "dataset/weather"}  # paths used by the official scripts
WEATHER_NVAR = 21  # --enc_in 21 in scripts/PatchTST/weather.sh and iTransformer's Weather script

HORIZONS = [96, 192, 336, 720]
FIELDS = ["run_id", "plan", "mode", "model", "dataset", "L", "H", "seed", "patch_len", "stride",
          "test_batch_size", "test_drop_last", "n_test_windows", "n_scored", "epochs_run",
          "early_stopped", "mse", "mae", "seconds", "device", "repo_commit", "timestamp_utc",
          # appended on branch final-runs (blank in rows logged before it)
          "mse_official", "mae_official", "n_scored_official", "scoring", "variant", "vali_loss_best"]
N_OLD_FIELDS = 22

CAL_SHORT = {"hour": "hour", "dow": "dow", "holiday": "hol"}
CAL_ORDER = ("hour", "dow", "holiday")
P24_ENV = ("P24_NO_TIMESTAMPS", "P24_ITR_CI", "P24_ALL_WINDOWS", "P24_EXPORT", "P24_CAL_FREEZE")


# ----------------------------------------------------------------- setup
def sh(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def setup():
    REPOS.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in COMMITS.items():
        d = REPOS / name
        if not (d / ".git").exists():
            sh(["git", "init", "-q", d])
            sh(["git", "-C", d, "fetch", "-q", "--depth", "1", url, commit])
            sh(["git", "-C", d, "checkout", "-q", "FETCH_HEAD"])
        head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        assert head == commit, f"{name}: HEAD {head} != pinned {commit}"
    sh([sys.executable, KIT / "patch_repos.py", REPOS])
    data = WORK / "data"
    data.mkdir(exist_ok=True)
    for ds, digest in ETT_SHA256.items():
        f = data / f"{ds}.csv"
        if not f.exists():
            urllib.request.urlretrieve(ETT_URL.format(ds), f)
        got = sha256(f)
        assert got == digest, f"{ds}.csv checksum {got} differs from the expected file"
        for repo, sub in DATA_SUBDIR.items():
            dst = REPO_DIR[repo] / sub
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst / f.name)
    print("Setup complete. Repos pinned, patched, data verified.")


WEATHER_MANUAL = """Weather could not be downloaded by script. Manual step:
  1. Download all_six_datasets.zip from the link in the pinned Autoformer README
     (Google Drive folder 1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy, or the Tsinghua Cloud URL in
     Autoformer/utils/download_data.py:4) and take all_six_datasets/weather/weather.csv.
  2. Either put it at {dst}, or set P24_WEATHER_CSV=/path/to/weather.csv (on Kaggle: add it
     as a Kaggle dataset and point P24_WEATHER_CSV at /kaggle/input/<name>/weather.csv).
  3. Run: python p24.py weather-data   (prints and checks the sha256)."""


def weather_data():
    """Fetch weather.csv, check its sha256 and shape, copy it where the official scripts read it."""
    data = WORK / "data"
    data.mkdir(parents=True, exist_ok=True)
    f = data / "weather.csv"
    src = os.environ.get("P24_WEATHER_CSV")
    if not f.exists() and src:
        shutil.copy(src, f)
    if not f.exists():
        z = data / "all_six_datasets.zip"
        try:
            if not z.exists():
                req = urllib.request.Request(WEATHER_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=600) as r, open(z, "wb") as out:
                    shutil.copyfileobj(r, out)
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.endswith("weather/weather.csv") and "__MACOSX" not in n]
                assert len(names) == 1, f"weather/weather.csv not found in {z.name}: {names}"
                with zf.open(names[0]) as r, open(f, "wb") as out:
                    shutil.copyfileobj(r, out)
        except Exception as e:  # network refused, not a zip, ...
            sys.exit(f"{type(e).__name__}: {e}\n" + WEATHER_MANUAL.format(dst=f))
    got = sha256(f)
    expected = os.environ.get("P24_WEATHER_SHA256") or WEATHER_SHA256
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "weather_sha256.txt").write_text(f"{got}  weather.csv\n")
    if expected is None:
        print(f"weather.csv sha256 = {got}\nNo expected checksum recorded yet: this is the first download. "
              "Record this value as WEATHER_SHA256 in p24.py and commit it, so every later download is checked.")
    elif got != expected:
        sys.exit(f"weather.csv checksum {got} differs from the expected {expected}")
    else:
        print(f"weather.csv sha256 verified ({got[:12]}...)")
    with open(f) as fh:
        header = next(csv.reader(fh))
    assert header[0] == "date" and len(header) == 1 + WEATHER_NVAR and header[-1] == "OT", \
        f"weather.csv: expected date + {WEATHER_NVAR} columns ending in OT, got {len(header)} columns"
    for repo, sub in WEATHER_DIR.items():
        dst = REPO_DIR[repo] / sub
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst / "weather.csv")
    print("Weather ready.")


# ----------------------------------------------------------- run configs
def model_args(model, ds, L, H, seed, P=16, S=None, epochs=None, patience=None, des=None):
    """Arguments copied from each official script; only seed, des and model_id differ."""
    S = S if S is not None else P // 2
    onz = ds.startswith("ONZ")
    weather = ds == "weather"
    custom = onz or weather
    nvar = 10 if onz else (WEATHER_NVAR if weather else 7)
    data_flag = "custom" if custom else ds
    csv_name = f"{ds}.csv"
    des = des or f"Exp_s{seed}"
    big = ds in ("ETTm1",) or onz  # ON-Zonal uses each model's ETTm1 configuration
    if weather and model not in ("PatchTST", "iTransformer"):
        raise ValueError(f"no official Weather configuration wired up for {model}")
    if model == "PatchTST":
        a = ["run_longExp.py", "--random_seed", seed, "--is_training", 1, "--root_path", "./dataset/",
             "--data_path", csv_name, "--model_id", f"{ds}_{L}_{H}_P{P}S{S}", "--model", "PatchTST",
             "--data", data_flag, "--features", "M", "--seq_len", L, "--pred_len", H, "--enc_in", nvar,
             "--e_layers", 3, "--patch_len", P, "--stride", S, "--des", des, "--itr", 1,
             "--batch_size", 128, "--learning_rate", 0.0001]
        if weather:  # scripts/PatchTST/weather.sh
            a += ["--n_heads", 16, "--d_model", 128, "--d_ff", 256, "--dropout", 0.2, "--fc_dropout", 0.2,
                  "--head_dropout", 0, "--train_epochs", 100, "--patience", 20]
        elif big:  # scripts/PatchTST/ettm1.sh
            a += ["--n_heads", 16, "--d_model", 128, "--d_ff", 256, "--dropout", 0.2, "--fc_dropout", 0.2,
                  "--head_dropout", 0, "--train_epochs", 100, "--patience", 20, "--lradj", "TST",
                  "--pct_start", 0.4]
        else:    # scripts/PatchTST/etth1.sh
            a += ["--n_heads", 4, "--d_model", 16, "--d_ff", 128, "--dropout", 0.3, "--fc_dropout", 0.3,
                  "--head_dropout", 0, "--train_epochs", 100]
    elif model == "DLinear":  # scripts/EXP-LongForecasting/Linear/etth1.sh, ettm1.sh
        bs, lr = (8, 0.0001) if big else (32, 0.005)
        a = ["run_longExp.py", "--is_training", 1, "--root_path", "./dataset/", "--data_path", csv_name,
             "--model_id", f"{ds}_{L}_{H}", "--model", "DLinear", "--data", data_flag, "--features", "M",
             "--seq_len", L, "--pred_len", H, "--enc_in", nvar, "--des", des, "--itr", 1,
             "--batch_size", bs, "--learning_rate", lr]
    elif model == "Autoformer":  # scripts/ETT_script/Autoformer_ETTh1.sh, _ETTm1.sh
        a = ["run.py", "--is_training", 1, "--root_path", "./dataset/ETT-small/", "--data_path", csv_name,
             "--model_id", f"{ds}_{L}_{H}", "--model", "Autoformer", "--data", data_flag, "--features", "M",
             "--seq_len", L, "--label_len", 48, "--pred_len", H, "--e_layers", 2, "--d_layers", 1,
             "--factor", 3, "--enc_in", nvar, "--dec_in", nvar, "--c_out", nvar, "--des", des, "--itr", 1]
        if ds == "ETTm1":
            a += ["--freq", "t"]
    elif model == "iTransformer":
        if weather:  # scripts/multivariate_forecasting/Weather/iTransformer.sh
            a = ["run.py", "--is_training", 1, "--root_path", "./dataset/weather/", "--data_path", "weather.csv",
                 "--model_id", f"{ds}_{L}_{H}", "--model", "iTransformer", "--data", "custom", "--features", "M",
                 "--seq_len", L, "--pred_len", H, "--e_layers", 3, "--enc_in", nvar, "--dec_in", nvar,
                 "--c_out", nvar, "--des", des, "--d_model", 512, "--d_ff", 512, "--itr", 1]
        else:  # scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh, _ETTm1.sh
            dm = 128 if big else (256 if H <= 192 else 512)
            a = ["run.py", "--is_training", 1, "--root_path", "./dataset/ETT-small/", "--data_path", csv_name,
                 "--model_id", f"{ds}_{L}_{H}", "--model", "iTransformer", "--data", data_flag, "--features", "M",
                 "--seq_len", L, "--pred_len", H, "--e_layers", 2, "--enc_in", nvar, "--dec_in", nvar,
                 "--c_out", nvar, "--des", des, "--d_model", dm, "--d_ff", dm, "--itr", 1]
    else:
        raise ValueError(model)
    if onz:
        a += ["--target", "West", "--freq", "h"]
    if epochs is not None:
        a = _set(a, "--train_epochs", epochs)
    if patience is not None:
        a = _set(a, "--patience", patience)
    return [str(x) for x in a]


def _set(a, flag, val):
    a = [str(x) for x in a]
    if flag in a:
        a[a.index(flag) + 1] = str(val)
    else:
        a += [flag, str(val)]
    return a


DEFAULT_BATCH = {"PatchTST": 128, "DLinear": 32, "Autoformer": 32, "iTransformer": 32}  # argparse defaults


def test_batch_size(model, args):
    if model == "iTransformer":
        return 1  # iTransformer's data_factory forces batch size 1 for testing
    if "--batch_size" in args:
        return int(args[args.index("--batch_size") + 1])
    return DEFAULT_BATCH[model]  # e.g. the Autoformer scripts do not pass --batch_size


def test_drop_last(model):
    # Verified in each repo's data_provider/data_factory.py at the pinned commit.
    return {"PatchTST": True, "DLinear": False, "Autoformer": False, "iTransformer": True}[model]


def variant_parts(spec):
    """Variant tags, in a fixed order. They extend run_id and --des (own checkpoint folder)."""
    parts = []
    if spec.get("no_timestamps"):
        parts.append("nots")  # iTransformer without calendar tokens (RQ2 timestamp check)
    if "cal" in spec:  # PatchTST+Cal; cal=() is the V2 run (calendar kit, every component off)
        names = [n for n in CAL_ORDER if n in spec["cal"]]
        assert len(names) == len(spec["cal"]), spec["cal"]
        parts.append("cal-" + ("-".join(CAL_SHORT[n] for n in names) if names else "none")
                     + ("-frozen" if spec.get("cal_freeze") else ""))
    if spec.get("ci"):
        parts.append("ci")  # channel-independent iTransformer (Control 2)
    if spec.get("lr") is not None:
        parts.append(f"lr{spec['lr']:g}")
    return parts


def run_id_of(spec):
    model, ds, L, H, seed = spec["model"], spec["dataset"], spec["L"], spec["H"], spec["seed"]
    P, S = spec.get("P", 16), spec.get("S")
    S = S if S is not None else P // 2
    tag = f"_P{P}S{S}" if model == "PatchTST" and (P != 16 or S != 8) else ""
    tag += "".join("_" + p for p in variant_parts(spec))
    return f"{model}_{ds}_L{L}_H{H}_s{seed}{tag}" + ("_bs1" if spec.get("mode", "train") == "rescore_bs1" else "")


# ------------------------------------------------------------- running
def load_done(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {r["run_id"] for r in csv.DictReader(f) if r.get("mse")}


def check_schema(csv_path):
    """Refuse to append to a results file written with other columns (never rewrite old files)."""
    if csv_path.exists() and csv_path.stat().st_size:
        with open(csv_path) as f:
            header = next(csv.reader(f), [])
        if header != FIELDS:
            sys.exit(f"{csv_path} has {len(header)} columns, this p24.py writes {len(FIELDS)}. "
                     "It was written by an older kit: use a new P24_RESULTS folder (a new JOB letter).")


def append_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    check_schema(csv_path)
    new = not csv_path.exists() or not csv_path.stat().st_size
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


MSE_RE = re.compile(r"mse:\s*([0-9.eE+-]+),\s*mae:\s*([0-9.eE+-]+)")
TEST_RE = re.compile(r"^test\s+(\d+)\s*$", re.M)
EPOCH_RE = re.compile(r"Epoch:\s*(\d+)")
VALI_RE = re.compile(r"Vali Loss:\s*([0-9.eE+-]+|nan)")
OFFICIAL_RE = re.compile(r"P24_OFFICIAL_SUBSET n_scored_official=(\d+) n_test=(\d+) batch_size=(\d+) "
                         r"mse_official=([0-9.eE+-]+) mae_official=([0-9.eE+-]+)")
SETTING_RE = re.compile(r">>>>>>>testing : (.+?)<{5,}")


def export_preds(run_id, out, repo, t0):
    """Copy pred.npy (+ true.npy) as one compressed npz, and the test_results PDFs."""
    import numpy as np
    setting = SETTING_RE.findall(out)[-1]
    res, figs = REPO_DIR[repo] / "results" / setting, REPO_DIR[repo] / "test_results" / setting
    assert (res / "pred.npy").stat().st_mtime >= t0 - 1, f"{res}/pred.npy is older than this run"
    dst = RESULTS / "preds" / run_id
    (dst / "test_results").mkdir(parents=True, exist_ok=True)
    arrays = {k: np.load(res / f"{k}.npy") for k in ("pred", "true") if (res / f"{k}.npy").exists()}
    np.savez_compressed(dst / "pred_true.npz", **arrays)
    pdfs = sorted(figs.glob("*.pdf"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    for p in pdfs:
        shutil.copy2(p, dst / "test_results" / p.name)
    (dst / "setting.txt").write_text(setting + "\n")
    print(f"exported {run_id}: {', '.join(f'{k}{tuple(v.shape)}' for k, v in arrays.items())}, {len(pdfs)} PDFs")


def command(spec, smoke=False):
    """(run_id, repo, args, env switches) for one spec: the exact command run_one executes."""
    model, ds, L, H, seed = spec["model"], spec["dataset"], spec["L"], spec["H"], spec["seed"]
    P, S, mode = spec.get("P", 16), spec.get("S"), spec.get("mode", "train")
    S = S if S is not None else P // 2
    parts = variant_parts(spec)
    repo = MODEL_REPO[model]
    des = f"Exp_s{seed}" + "".join("_" + p for p in parts)  # separate checkpoint folder per variant
    args = model_args(model, ds, L, H, seed, P, S, epochs=spec.get("epochs"), patience=spec.get("patience"),
                      des=des)
    if smoke:
        args = _set(args, "--train_epochs", 1)
    if mode == "rescore_bs1":
        args = _set(args, "--is_training", 0)
        args = _set(args, "--batch_size", 1)
    if spec.get("lr") is not None:
        args = _set(args, "--learning_rate", spec["lr"])
    env = {}
    if spec.get("no_timestamps"):
        assert model == "iTransformer", "the timestamp switch exists only for iTransformer"
        env["P24_NO_TIMESTAMPS"] = "1"
    if spec.get("ci"):
        assert model == "iTransformer", "Control 2 (channel independence) exists only for iTransformer"
        env["P24_ITR_CI"] = "1"
    if "cal" in spec:
        assert model == "PatchTST", "PatchTST+Cal exists only for PatchTST"
        assert "holiday" not in spec["cal"] or ds.startswith("ONZ"), "the Ontario holiday flag is for ON-Zonal only"
        args += [f"--cal_{n}" for n in CAL_ORDER if n in spec["cal"]]
        if spec.get("cal_freeze"):
            env["P24_CAL_FREEZE"] = "1"
    if model == "PatchTST" and mode == "train":  # every new PatchTST training run is dual-scored
        env["P24_ALL_WINDOWS"] = "1"
    if spec.get("export"):
        env["P24_EXPORT"] = "1"
    return run_id_of(spec), repo, args, env


def run_one(spec, csv_path, done, smoke=False):
    model, ds, L, H, seed = spec["model"], spec["dataset"], spec["L"], spec["H"], spec["seed"]
    P, S, mode = spec.get("P", 16), spec.get("S"), spec.get("mode", "train")
    S = S if S is not None else P // 2
    parts = variant_parts(spec)
    run_id, repo, args, switches = command(spec, smoke)
    if run_id in done:
        print(f"skip {run_id} (already logged)")
        return
    env = dict(os.environ, SEED=str(seed), PYTHONUNBUFFERED="1")
    for k in P24_ENV:
        env.pop(k, None)
    env.update(switches)
    dual = switches.get("P24_ALL_WINDOWS") == "1"
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{run_id}{'_SMOKE' if smoke else ''}.log"
    print(f"\n=== {run_id} ===", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write("CMD: python " + " ".join(args) + f"\nSEED={seed}\n")
        flags = {k: env[k] for k in P24_ENV if k in env}
        if flags:
            log.write("P24_ENV: " + " ".join(f"{k}={v}" for k, v in flags.items()) + "\n")
        log.flush()
        p = subprocess.run([sys.executable, "-u"] + args, cwd=REPO_DIR[repo], env=env,
                           stdout=log, stderr=subprocess.STDOUT)
    secs = time.time() - t0
    out = log_path.read_text()
    m = MSE_RE.findall(out)
    if p.returncode != 0 or not m:
        print(f"FAILED {run_id} (exit {p.returncode}). Last lines of {log_path}:")
        print("\n".join(out.splitlines()[-25:]))
        return
    mse, mae = map(float, m[-1])
    tests = TEST_RE.findall(out)
    n_test = int(tests[-1]) if tests else None
    bs = 1 if mode == "rescore_bs1" else test_batch_size(model, args)
    dl = test_drop_last(model) and bs > 1 and not dual
    n_scored = (n_test // bs) * bs if (n_test and dl) else n_test
    off = {}
    if dual:
        o = OFFICIAL_RE.findall(out)
        assert o, f"{run_id}: dual scoring requested but no P24_OFFICIAL_SUBSET line in {log_path}"
        k, n, b, o_mse, o_mae = o[-1]
        assert int(n) == n_test and int(b) == bs and int(k) == (n_test // bs) * bs, o[-1]
        off = dict(mse_official=f"{float(o_mse):.6f}", mae_official=f"{float(o_mae):.6f}", n_scored_official=int(k))
    scoring = "all+official" if dual else ("all" if mode == "rescore_bs1" else "official")
    epochs = EPOCH_RE.findall(out)
    vali = [float(v) for v in VALI_RE.findall(out)]
    vali = [v for v in vali if not math.isnan(v)]
    try:
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "unknown"
    row = dict(run_id=run_id, plan=spec.get("plan", ""), mode=mode, model=model, dataset=ds, L=L, H=H,
               seed=seed, patch_len=P if model == "PatchTST" else "", stride=S if model == "PatchTST" else "",
               test_batch_size=bs, test_drop_last=dl, n_test_windows=n_test, n_scored=n_scored,
               epochs_run=max(map(int, epochs)) if epochs else "", early_stopped="Early stopping" in out,
               mse=f"{mse:.6f}", mae=f"{mae:.6f}", seconds=f"{secs:.0f}", device=device,
               repo_commit=COMMITS[repo][1][:10],
               timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
               scoring=scoring, variant="+".join(parts),
               vali_loss_best=f"{min(vali):.7f}" if (vali and mode == "train") else "", **off)
    append_row(csv_path, row)
    done.add(run_id)
    extra = f", official subset {off['n_scored_official']}: mse={float(off['mse_official']):.4f}" if dual else ""
    print(f"done {run_id}: mse={mse:.4f} mae={mae:.4f} ({secs/60:.1f} min, scored {n_scored}/{n_test}{extra})")
    if spec.get("export"):
        try:
            export_preds(run_id, out, repo, t0)
        except Exception as e:  # the run is logged; a failed export must not stop the plan
            print(f"EXPORT FAILED for {run_id}: {type(e).__name__}: {e}")


# ------------------------------------------------------------- plans
ONZ = dict(epochs=10, patience=3)  # matched ON-Zonal protocol: same budget for every model
F3_LRS = (3e-5, 1e-4, 3e-4)        # 1e-4 = both defaults (PatchTST etth1.sh; iTransformer run.py:65)
F3_DEFAULT = 1e-4
CAL_ABLATION = (("hour", "dow", "holiday"), ("dow", "holiday"), ("hour", "holiday"), ("hour", "dow"))  # full, -hour, -dow, -holiday
PLANS = {  # name: (plan ID in claude/P24_final_run_plan_and_P1.md, default seeds, what)
    "A": ("done (Job A)", (2021, 2022, 2023), "ETTh1 reproduction, 4 models x 4 H, + F1 re-scores"),
    "ablation": ("done (Job C)", (2021, 2022, 2023), "PatchTST P and L sweeps, H=96/720"),
    "onzonal": ("done (Job B)", (2021, 2022, 2023), "RQ2 preliminary, ON-Zonal L=96 H=96"),
    "onzonal_nots": ("M1", (2021, 2022, 2023), "iTransformer without calendar tokens, real + shift"),
    "onzonal_seeds": ("M2", (2024, 2025), "RQ2 seeds 2024-25: iTransformer, PatchTST, DLinear, iTransformer-nots"),
    "sameL": ("M3", (2021, 2022, 2023), "ETTh1 same L: PatchTST L=96 and iTransformer L=336, 4 H"),
    "onzonal_cal": ("M4", (2021, 2022, 2023), "PatchTST+Cal ON-Zonal: full, -hour, -dow, -holiday"),
    "etth1_cal": ("M5", (2021, 2022, 2023), "PatchTST+Cal ETTh1 L=336 (hour + dow), 4 H"),
    "ETTm1": ("M6", (2021, 2022, 2023), "ETTm1 reproduction (use --only PatchTST Autoformer iTransformer)"),
    "onzonal_export": ("M7", (2021,), "ON-Zonal prediction export, seed 2021 real"),
    "onzonal_ci": ("M8", (2021, 2022, 2023), "Control 2: channel-independent iTransformer, real + shift"),
    "grid336": ("S1", (2021, 2022, 2023), "P x L grid at H=336 (minus (16,336) and (16,96))"),
    "ablation_redo": ("S2", (2021, 2022, 2023), "the 24 official-only ablation runs, re-trained dual-scored"),
    "onzonal_h720": ("S3", (2021, 2022, 2023), "RQ2 at H=720 (L=96)"),
    "f3": ("S4", (2021,), "F3 LR grid at L=96 (seed 2021); seeds 2022-23 only for non-default winners"),
    "weather": ("S5 / I1", (2021, 2022, 2023), "Weather: PatchTST L=336 + iTransformer L=96, 4 H"),
    "weather_l336": ("I1", (2021, 2022, 2023), "Weather: iTransformer L=336 (same-L pair)"),
    "onzonal_l336": ("I2", (2021, 2022, 2023), "RQ2 at L=336 (H=96)"),
    "sameL_extra": ("I3", (2021, 2022, 2023), "ETTh1: DLinear L=96 and Autoformer L=336, 4 H"),
    "V": ("V1-V3", (2021,), "validation gate (prints PASS/FAIL)"),
}


def f3_selection_path():
    return Path(os.environ.get("P24_F3_SELECTION", KIT / "results" / "f3_selection.json"))


def plan_specs(plan, seeds):
    s = []

    def add(**kw):
        s.append(dict(plan=plan, **kw))

    if plan in ("A", "ETTm1"):
        ds = "ETTh1" if plan == "A" else "ETTm1"
        for seed in seeds:  # one complete seed first, so a full table exists early
            for model, L in (("PatchTST", 336), ("DLinear", 336), ("iTransformer", 96), ("Autoformer", 96)):
                for H in HORIZONS:
                    add(model=model, dataset=ds, L=L, H=H, seed=seed)
                    if model == "PatchTST" and plan == "A":  # F1: re-score the same checkpoint with nothing dropped
                        add(model=model, dataset=ds, L=L, H=H, seed=seed, mode="rescore_bs1")
                    # ETTm1: no re-score run; dual scoring logs both scores in the training run
    elif plan in ("ablation", "ablation_redo"):  # ablation_redo = the 24 runs of 'ablation' without P=16, L=336
        for seed in seeds:
            for H in (96, 720):
                for P in (8, 16, 32, 64):  # stride = P/2 keeps PatchTST's default 50% overlap
                    if plan == "ablation" or P != 16:
                        add(model="PatchTST", dataset="ETTh1", L=336, H=H, seed=seed, P=P)
                for L in (96, 720):        # L = 336 is the run above
                    if plan == "ablation" or L == 720:  # L=96 is re-trained in sameL (M3)
                        add(model="PatchTST", dataset="ETTh1", L=L, H=H, seed=seed, P=16)
    elif plan == "onzonal":
        # Matched protocol for the new dataset: same look-back (96), same budget
        # (10 epochs, patience 3) for every model; ETTm1 configurations otherwise.
        for seed in seeds:
            for ds in ("ONZ_real", "ONZ_shift"):
                for model in ("iTransformer", "PatchTST", "DLinear"):
                    add(model=model, dataset=ds, L=96, H=96, seed=seed, **ONZ)
                    if model == "PatchTST":
                        add(model=model, dataset=ds, L=96, H=96, seed=seed, mode="rescore_bs1", **ONZ)
    elif plan == "onzonal_nots":
        # RQ2 timestamp check: iTransformer without its calendar tokens, same protocol as "onzonal".
        for seed in seeds:
            for ds in ("ONZ_real", "ONZ_shift"):
                add(model="iTransformer", dataset=ds, L=96, H=96, seed=seed, no_timestamps=True, **ONZ)
    elif plan == "onzonal_seeds":
        for seed in seeds:
            for ds in ("ONZ_real", "ONZ_shift"):
                for model, extra in (("iTransformer", {}), ("PatchTST", {}), ("DLinear", {}),
                                     ("iTransformer", {"no_timestamps": True})):
                    add(model=model, dataset=ds, L=96, H=96, seed=seed, **ONZ, **extra)
    elif plan == "sameL":
        for seed in seeds:
            for H in HORIZONS:
                add(model="PatchTST", dataset="ETTh1", L=96, H=H, seed=seed)
            for H in HORIZONS:
                add(model="iTransformer", dataset="ETTh1", L=336, H=H, seed=seed)
    elif plan == "onzonal_cal":
        for seed in seeds:
            for cal in CAL_ABLATION:
                add(model="PatchTST", dataset="ONZ_real", L=96, H=96, seed=seed, cal=cal, **ONZ)
    elif plan == "etth1_cal":
        for seed in seeds:
            for H in HORIZONS:
                add(model="PatchTST", dataset="ETTh1", L=336, H=H, seed=seed, cal=("hour", "dow"))
    elif plan == "onzonal_export":  # same run_ids as runs_B.csv: each must reproduce its logged row
        for seed in seeds:
            for model in ("iTransformer", "PatchTST", "DLinear"):
                add(model=model, dataset="ONZ_real", L=96, H=96, seed=seed, export=True, **ONZ)
    elif plan == "onzonal_ci":
        for seed in seeds:
            for ds in ("ONZ_real", "ONZ_shift"):
                add(model="iTransformer", dataset=ds, L=96, H=96, seed=seed, ci=True, **ONZ)
    elif plan in ("onzonal_h720", "onzonal_l336"):
        L, H = (96, 720) if plan == "onzonal_h720" else (336, 96)
        for seed in seeds:
            for ds in ("ONZ_real", "ONZ_shift"):
                for model in ("iTransformer", "PatchTST", "DLinear"):
                    add(model=model, dataset=ds, L=L, H=H, seed=seed, **ONZ)
    elif plan == "grid336":
        for seed in seeds:
            for L in (96, 336, 720):
                for P in (8, 16, 32, 64):
                    if (P, L) in ((16, 336), (16, 96)):  # (16,336): runs_A; (16,96): sameL (M3)
                        continue
                    add(model="PatchTST", dataset="ETTh1", L=L, H=336, seed=seed, P=P)
    elif plan == "f3":
        for seed in seeds:
            if seed == 2021:  # selection round: every LR, select on validation loss
                for model in ("PatchTST", "iTransformer"):
                    for H in HORIZONS:
                        for lr in F3_LRS:
                            if lr == F3_DEFAULT and model == "PatchTST":
                                continue  # = PatchTST_ETTh1_L96_H*_s2021 of plan sameL (same run_id)
                            # iTransformer default LR = iTransformer_ETTh1_L96_H*_s2021 of runs_A.csv, re-run
                            # because its validation loss was never logged (Job A logs were not kept)
                            add(model=model, dataset="ETTh1", L=96, H=H, seed=seed,
                                lr=None if lr == F3_DEFAULT else lr)
            else:  # seeds 2022-23 only where a non-default LR won
                sel_path = f3_selection_path()
                if not sel_path.exists():
                    sys.exit(f"{sel_path} not found: run seed 2021 first, then `python p24.py f3select results`")
                sel = json.loads(sel_path.read_text())
                for model in ("PatchTST", "iTransformer"):
                    for H in HORIZONS:
                        w = sel.get(model, {}).get(str(H), {})
                        if w.get("winner_lr") is None:
                            sys.exit(f"f3 selection for {model} H={H} is missing in {sel_path}")
                        if w["winner_lr"] != F3_DEFAULT:
                            add(model=model, dataset="ETTh1", L=96, H=H, seed=seed, lr=w["winner_lr"])
    elif plan in ("weather", "weather_l336"):
        for seed in seeds:
            if plan == "weather":
                for H in HORIZONS:
                    add(model="PatchTST", dataset="weather", L=336, H=H, seed=seed)
            for H in HORIZONS:
                add(model="iTransformer", dataset="weather", L=96 if plan == "weather" else 336, H=H, seed=seed)
    elif plan == "sameL_extra":
        for seed in seeds:
            for H in HORIZONS:
                add(model="DLinear", dataset="ETTh1", L=96, H=H, seed=seed)
            for H in HORIZONS:
                add(model="Autoformer", dataset="ETTh1", L=336, H=H, seed=seed)
    elif plan == "V":
        for seed in seeds:
            base = dict(model="PatchTST", dataset="ETTh1", L=336, H=96, seed=seed, export=True)
            add(**base)                                                # V1
            add(**base, cal=())                                        # V2: PatchTST+Cal, every component off
            add(**base, cal=("hour", "dow"), cal_freeze=True)          # V3: hour+dow on, embeddings frozen at 0
    else:
        raise ValueError(plan)
    return s


# ------------------------------------------------------------- validation (plan V)
V_TOL = 1e-5  # V1b: batch 128 (all windows) vs batch 1 re-score differ only by float32 rounding


def _rows(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def check_v(smoke=False):
    """V1: official subset == runs_A.csv:2 exactly; every-window ~ runs_A.csv:3.
    V2, V3: identical to V1 bit for bit (metrics and exported predictions)."""
    import numpy as np
    new = {r["run_id"]: r for r in _rows(RESULTS / ("smoke_runs.csv" if smoke else "runs.csv"))}
    with open(KIT / "results" / "runs_A.csv", newline="") as f:
        lines = list(csv.reader(f))
    ref_off, ref_all = (dict(zip(lines[0], lines[i])) for i in (1, 2))  # runs_A.csv:2 and :3
    assert ref_off["run_id"] == "PatchTST_ETTh1_L336_H96_s2021" and ref_all["run_id"] == ref_off["run_id"] + "_bs1"
    rid = ref_off["run_id"]
    v1, v2, v3 = new.get(rid), new.get(rid + "_cal-none"), new.get(rid + "_cal-hour-dow-frozen")
    res = []

    def say(name, ok, msg):
        res.append(ok)
        print(f"{name}: {'PASS' if ok else 'FAIL'}  {msg}")

    print("\n--- plan V checks" + (" (SMOKE: 1 epoch on CPU, V1a/V1b are expected to FAIL)" if smoke else "") + " ---")
    if v1 is None:
        say("V1", False, f"{rid} not logged")
    else:
        say("V1a", v1["mse_official"] == ref_off["mse"] and v1["mae_official"] == ref_off["mae"],
            f"official subset ({v1['n_scored_official']} windows): mse {v1['mse_official']} mae {v1['mae_official']}"
            f" | runs_A.csv:2 mse {ref_off['mse']} mae {ref_off['mae']} ({ref_off['n_scored']} windows)")
        d = float(v1["mse"]) - float(ref_all["mse"])
        say("V1b", abs(d) <= V_TOL,
            f"every window ({v1['n_scored']}): mse {v1['mse']} | runs_A.csv:3 re-score mse {ref_all['mse']}"
            f" | difference {d:+.2e} (tolerance {V_TOL:g})")
    for name, v in (("V2", v2), ("V3", v3)):
        if v1 is None or v is None:
            say(name, False, "run not logged")
            continue
        same = all(v[k] == v1[k] for k in ("mse", "mae", "mse_official", "mae_official", "n_scored"))
        a, b = (RESULTS / "preds" / r / "pred_true.npz" for r in (rid, v["run_id"]))
        if a.exists() and b.exists():
            pa, pb = np.load(a)["pred"], np.load(b)["pred"]
            bit = pa.shape == pb.shape and pa.tobytes() == pb.tobytes()
            maxdiff = float(np.abs(pa.astype(np.float64) - pb).max()) if pa.shape == pb.shape else float("nan")
            msg = f"metrics identical: {same}; predictions bitwise identical: {bit} (max |diff| {maxdiff:.1e})"
        else:
            bit, msg = False, "exported predictions missing"
        say(name, same and bit, f"{v['run_id']} vs V1: {msg}")
    print("--- V overall: " + ("PASS" if all(res) else "FAIL") + " ---")
    return all(res)


# ------------------------------------------------------------- F3 selection
def f3select(dirs, smoke=False):
    import pandas as pd
    files = sorted({f for d in dirs for f in Path(d).rglob("smoke_runs.csv" if smoke else "runs*.csv")})
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if "vali_loss_best" not in df:
        sys.exit("no vali_loss_best column in any runs*.csv: run plan f3 (seed 2021) and plan sameL first")
    df = df[(df.dataset == "ETTh1") & (df.L == 96) & (df.seed == 2021) & (df["mode"] == "train")]
    sel = {}
    for model in ("PatchTST", "iTransformer"):
        for H in HORIZONS:
            cand = {}
            for lr in F3_LRS:
                var = "" if lr == F3_DEFAULT else f"lr{lr:g}"
                r = df[(df.model == model) & (df.H == H) & (df["variant"].fillna("") == var)
                       & df["vali_loss_best"].notna()]
                if len(r):
                    cand[lr] = float(r["vali_loss_best"].iloc[-1])
            missing = [lr for lr in F3_LRS if lr not in cand]
            win = min(cand, key=cand.get) if not missing else None
            sel.setdefault(model, {})[str(H)] = {"vali_loss_best": {f"{k:g}": v for k, v in cand.items()},
                                                 "missing": [f"{m:g}" for m in missing], "winner_lr": win}
            print(f"{model} H={H}: " + ", ".join(f"lr {k:g}: {v:.7f}" for k, v in sorted(cand.items()))
                  + (f" -> winner {win:g}" if win is not None else f" -> not measured: {missing}"))
    out = f3_selection_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sel, indent=1))
    print(f"wrote {out} (commit it; plan f3 with seeds 2022 2023 reads it)")


# ------------------------------------------------------------- driver
def run_plan(plan, seeds, smoke=False, only=None, horizons=None):
    csv_path = RESULTS / ("smoke_runs.csv" if smoke else "runs.csv")
    check_schema(csv_path)  # before any GPU time is spent
    done = load_done(csv_path)
    seeds = sorted(seeds if seeds else PLANS[plan][1])  # 2021 (the lowest seed) of every config first
    specs = plan_specs(plan, seeds)
    if only:
        specs = [x for x in specs if x["model"] in only]
    if horizons:
        specs = [x for x in specs if x["H"] in horizons]
    need = sorted({(MODEL_REPO[x["model"]], x["dataset"]) for x in specs if x["dataset"].startswith("ONZ")})
    missing = [str(REPO_DIR[r] / DATA_SUBDIR[r] / f"{d}.csv") for r, d in need
               if not (REPO_DIR[r] / DATA_SUBDIR[r] / f"{d}.csv").exists()]
    if missing:  # stop early, clearly, if onzonal.py did not build the data
        sys.exit("ON-Zonal data not found (did onzonal.py fail above?). Missing:\n  " + "\n  ".join(missing))
    if any(x["dataset"] == "weather" for x in specs):
        weather_data()
    print(f"Plan {plan} ({PLANS[plan][0]}): {len(specs)} runs, "
          f"{sum(1 for x in specs if x.get('mode', 'train') == 'train')} trainings, seeds {seeds}")
    # P24_STOP_AFTER_H: start no new run after this many hours, so a session that the platform
    # ends at a fixed limit (Kaggle: 12 h) stops cleanly and keeps its output. Unset = no limit.
    stop_h, t_plan = float(os.environ.get("P24_STOP_AFTER_H") or 0), time.time()
    for i, spec in enumerate(specs):
        if stop_h and time.time() - t_plan > stop_h * 3600 and run_id_of(spec) not in done:
            left = sum(1 for x in specs[i:] if run_id_of(x) not in done)
            print(f"\nSTOP: this session has used its {stop_h:g} h (P24_STOP_AFTER_H); {left} run(s) not started. "
                  "Run this JOB again to continue: finished runs are skipped.", flush=True)
            return
        try:
            run_one(spec, csv_path, done, smoke)
        except Exception as e:  # one bad run must never stop the plan
            print(f"ERROR in {spec}: {type(e).__name__}: {e} -- continuing with the next run", flush=True)
    if plan == "V":
        check_v(smoke)


def tables():
    import tables as T
    T.main(RESULTS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("weather-data")
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True, choices=list(PLANS))
    r.add_argument("--seeds", nargs="+", type=int, default=None, help="default: the plan's own seeds (PLANS)")
    r.add_argument("--only", nargs="*", help="restrict to these models")
    r.add_argument("--horizons", nargs="*", type=int, help="restrict to these horizons (smoke tests, debugging)")
    r.add_argument("--smoke", action="store_true", help="1 epoch, logged to smoke_runs.csv only")
    r.add_argument("--list", action="store_true", help="print the run_ids of the plan and exit")
    f3 = sub.add_parser("f3select")
    f3.add_argument("dirs", nargs="*", default=[str(RESULTS)])
    f3.add_argument("--smoke", action="store_true", help="read smoke_runs.csv (smoke tests)")
    cv = sub.add_parser("check-v")
    cv.add_argument("--smoke", action="store_true")
    sub.add_parser("plans")
    sub.add_parser("tables")
    a = ap.parse_args()
    if a.cmd == "setup":
        setup()
    elif a.cmd == "weather-data":
        weather_data()
    elif a.cmd == "run" and a.list:
        for x in plan_specs(a.plan, sorted(a.seeds or PLANS[a.plan][1])):
            if not a.only or x["model"] in a.only:
                print(run_id_of(x))
    elif a.cmd == "run":
        run_plan(a.plan, a.seeds, a.smoke, a.only, a.horizons)
    elif a.cmd == "f3select":
        f3select(a.dirs, a.smoke)
    elif a.cmd == "check-v":
        sys.exit(0 if check_v(a.smoke) else 1)
    elif a.cmd == "plans":
        for k, (pid, seeds, what) in PLANS.items():
            print(f"{k:15s} {pid:13s} seeds {','.join(map(str, seeds)):15s} {len(plan_specs(k, seeds)):>4} runs  {what}")
    else:
        sys.path.insert(0, str(KIT))
        tables()
