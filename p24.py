"""P24 run kit: set up the official repos, run experiments, log every run.

Commands
  python p24.py setup                      clone pinned repos, patch, add ETT data
  python p24.py run --plan A --seeds 2021 2022 2023   ETTh1 reproduction (+F1 re-scoring)
  python p24.py run --plan ETTm1 --seeds 2021         ETTm1 reproduction
  python p24.py run --plan ablation --seeds 2021      PatchTST patch length / look-back
  python p24.py run --plan onzonal --seeds 2021 2022 2023   RQ2 preliminary (needs onzonal.py first)
  python p24.py tables                     build result tables from results/runs.csv

Every run is appended to results/runs.csv as soon as it finishes, so a
disconnect loses at most one run. Re-running a plan skips runs already logged.
Each run's full console output is kept in logs/<run_id>.log.
Smoke tests (--smoke) train 1 epoch and log to results/smoke_runs.csv only.
"""
import argparse
import csv
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
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
HORIZONS = [96, 192, 336, 720]
FIELDS = ["run_id", "plan", "mode", "model", "dataset", "L", "H", "seed", "patch_len", "stride",
          "test_batch_size", "test_drop_last", "n_test_windows", "n_scored", "epochs_run",
          "early_stopped", "mse", "mae", "seconds", "device", "repo_commit", "timestamp_utc"]


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


# ----------------------------------------------------------- run configs
def model_args(model, ds, L, H, seed, P=16, S=None, epochs=None, patience=None):
    """Arguments copied from each official script; only seed, des and model_id differ."""
    S = S if S is not None else P // 2
    custom = ds.startswith("ONZ")
    nvar = 10 if custom else 7
    data_flag = "custom" if custom else ds
    csv_name = f"{ds}.csv"
    des = f"Exp_s{seed}"
    big = ds in ("ETTm1",) or custom  # ON-Zonal uses each model's ETTm1 configuration
    if model == "PatchTST":
        a = ["run_longExp.py", "--random_seed", seed, "--is_training", 1, "--root_path", "./dataset/",
             "--data_path", csv_name, "--model_id", f"{ds}_{L}_{H}_P{P}S{S}", "--model", "PatchTST",
             "--data", data_flag, "--features", "M", "--seq_len", L, "--pred_len", H, "--enc_in", nvar,
             "--e_layers", 3, "--patch_len", P, "--stride", S, "--des", des, "--itr", 1,
             "--batch_size", 128, "--learning_rate", 0.0001]
        if big:  # scripts/PatchTST/ettm1.sh
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
    elif model == "iTransformer":  # scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh, _ETTm1.sh
        dm = 128 if big else (256 if H <= 192 else 512)
        a = ["run.py", "--is_training", 1, "--root_path", "./dataset/ETT-small/", "--data_path", csv_name,
             "--model_id", f"{ds}_{L}_{H}", "--model", "iTransformer", "--data", data_flag, "--features", "M",
             "--seq_len", L, "--pred_len", H, "--e_layers", 2, "--enc_in", nvar, "--dec_in", nvar,
             "--c_out", nvar, "--des", des, "--d_model", dm, "--d_ff", dm, "--itr", 1]
    else:
        raise ValueError(model)
    if custom:
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


# ------------------------------------------------------------- running
def load_done(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {r["run_id"] for r in csv.DictReader(f) if r.get("mse")}


def append_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


MSE_RE = re.compile(r"mse:\s*([0-9.eE+-]+),\s*mae:\s*([0-9.eE+-]+)")
TEST_RE = re.compile(r"^test\s+(\d+)\s*$", re.M)
EPOCH_RE = re.compile(r"Epoch:\s*(\d+)")


def run_one(spec, csv_path, done, smoke=False):
    model, ds, L, H, seed = spec["model"], spec["dataset"], spec["L"], spec["H"], spec["seed"]
    P, S, mode = spec.get("P", 16), spec.get("S"), spec.get("mode", "train")
    S = S if S is not None else P // 2
    tag = f"_P{P}S{S}" if model == "PatchTST" and (P != 16 or S != 8) else ""
    run_id = f"{model}_{ds}_L{L}_H{H}_s{seed}{tag}" + ("_bs1" if mode == "rescore_bs1" else "")
    if run_id in done:
        print(f"skip {run_id} (already logged)")
        return
    repo = MODEL_REPO[model]
    args = model_args(model, ds, L, H, seed, P, S, epochs=spec.get("epochs"), patience=spec.get("patience"))
    if smoke:
        args = _set(args, "--train_epochs", 1)
    if mode == "rescore_bs1":
        args = _set(args, "--is_training", 0)
        args = _set(args, "--batch_size", 1)
    env = dict(os.environ, SEED=str(seed), PYTHONUNBUFFERED="1")
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{run_id}{'_SMOKE' if smoke else ''}.log"
    print(f"\n=== {run_id} ===", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write("CMD: python " + " ".join(args) + f"\nSEED={seed}\n")
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
    dl = test_drop_last(model) and bs > 1
    n_scored = (n_test // bs) * bs if (n_test and dl) else n_test
    epochs = EPOCH_RE.findall(out)
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
               timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    append_row(csv_path, row)
    done.add(run_id)
    print(f"done {run_id}: mse={mse:.4f} mae={mae:.4f} ({secs/60:.1f} min, scored {n_scored}/{n_test})")


def plan_specs(plan, seeds):
    s = []
    if plan in ("A", "ETTm1"):
        ds = "ETTh1" if plan == "A" else "ETTm1"
        for seed in seeds:  # one complete seed first, so a full table exists early
            for model, L in (("PatchTST", 336), ("DLinear", 336), ("iTransformer", 96), ("Autoformer", 96)):
                for H in HORIZONS:
                    s.append(dict(plan=plan, model=model, dataset=ds, L=L, H=H, seed=seed))
                    if model == "PatchTST":  # F1: re-score the same checkpoint with nothing dropped
                        s.append(dict(plan=plan, model=model, dataset=ds, L=L, H=H, seed=seed,
                                      mode="rescore_bs1"))
    elif plan == "ablation":
        for seed in seeds:
            for H in (96, 720):
                for P in (8, 16, 32, 64):  # stride = P/2 keeps PatchTST's default 50% overlap
                    s.append(dict(plan=plan, model="PatchTST", dataset="ETTh1", L=336, H=H, seed=seed, P=P))
                for L in (96, 720):        # L = 336 is the run above
                    s.append(dict(plan=plan, model="PatchTST", dataset="ETTh1", L=L, H=H, seed=seed, P=16))
    elif plan == "onzonal":
        # Matched protocol for the new dataset: same look-back (96), same budget
        # (10 epochs, patience 3) for every model; ETTm1 configurations otherwise.
        for seed in seeds:
            for ds in ("ONZ_real", "ONZ_shift"):
                for model in ("iTransformer", "PatchTST", "DLinear"):
                    s.append(dict(plan=plan, model=model, dataset=ds, L=96, H=96, seed=seed,
                                  epochs=10, patience=3))
                    if model == "PatchTST":
                        s.append(dict(plan=plan, model=model, dataset=ds, L=96, H=96, seed=seed,
                                      epochs=10, patience=3, mode="rescore_bs1"))
    else:
        raise ValueError(plan)
    return s


def run_plan(plan, seeds, smoke=False, only=None):
    csv_path = RESULTS / ("smoke_runs.csv" if smoke else "runs.csv")
    done = load_done(csv_path)
    if plan == "onzonal":  # stop early, clearly, if onzonal.py did not build the data
        need = [REPO_DIR[r] / DATA_SUBDIR[r] / f"{n}.csv"
                for r in ("PatchTST", "LTSF-Linear", "iTransformer") for n in ("ONZ_real", "ONZ_shift")]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            sys.exit("ON-Zonal data not found (did onzonal.py fail above?). Missing:\n  " + "\n  ".join(missing))
    specs = plan_specs(plan, seeds)
    if only:
        specs = [x for x in specs if x["model"] in only]
    print(f"Plan {plan}: {len(specs)} runs, {sum(1 for x in specs if x.get('mode','train')=='train')} trainings")
    for spec in specs:
        try:
            run_one(spec, csv_path, done, smoke)
        except Exception as e:  # one bad run must never stop the plan
            print(f"ERROR in {spec}: {type(e).__name__}: {e} -- continuing with the next run", flush=True)


# --------------------------------------------------------------- tables
def tables():
    import tables as T
    T.main(RESULTS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True, choices=["A", "ETTm1", "ablation", "onzonal"])
    r.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023])
    r.add_argument("--only", nargs="*", help="restrict to these models")
    r.add_argument("--smoke", action="store_true", help="1 epoch, logged to smoke_runs.csv only")
    sub.add_parser("tables")
    a = ap.parse_args()
    if a.cmd == "setup":
        setup()
    elif a.cmd == "run":
        run_plan(a.plan, a.seeds, a.smoke, a.only)
    else:
        sys.path.insert(0, str(KIT))
        tables()
