"""Forward-pass latency and parameter count per model configuration (M9; no training).

    python timing.py                          every distinct config of the plans below -> results/timing.csv
    python timing.py --plans A sameL --reps 50 --warmup 10 --batches 1 32
    python timing.py --smoke                  CPU-friendly: plan A configs, 3 reps, 1 warm-up

Each configuration is built exactly as p24.py runs it: the same command-line arguments
(p24.model_args), parsed by the official run script's own argparse block, then the official
Model class. Inputs are random tensors of the shapes the official test loop feeds the model.
Per batch size: `warmup` untimed forward passes, then `reps` timed ones, each bracketed by
torch.cuda.synchronize(); we report the median (and quartiles) in milliseconds.

Untrained weights are fine: a forward pass runs the same operations on tensors of the same
shapes whatever the weight values (dense matrix products, convolutions, FFTs; no
data-dependent early exit or sparsity), so its cost does not depend on training. The one
data-dependent step, Autoformer's top-k delay selection, picks a fixed number k of lags, so
its cost is fixed too. Timing excludes data loading and the host-to-device copy.
Only compare latencies measured on the same GPU type (the `device` column).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p24  # noqa: E402

DEFAULT_PLANS = ("A", "ETTm1", "sameL", "sameL_extra", "ablation", "grid336", "onzonal", "onzonal_nots",
                 "onzonal_cal", "etth1_cal", "onzonal_ci", "onzonal_h720", "onzonal_l336", "weather",
                 "weather_l336")
FIELDS = ["config_id", "model", "dataset", "L", "H", "patch_len", "stride", "variant", "batch", "params",
          "params_trainable", "median_ms", "p25_ms", "p75_ms", "reps", "warmup", "device", "torch",
          "timestamp_utc"]

WORKER = r'''
import argparse, json, os, sys, textwrap, time
import numpy as np, torch
cfgs = json.loads(sys.stdin.read())
src = open(cfgs[0]["script"]).read()
a, b = src.index("parser = argparse.ArgumentParser"), src.index("args = parser.parse_args()")
a = src.rfind("\n", 0, a) + 1  # from the start of that line, so dedent sees one indentation
ns = {"argparse": argparse}
exec(textwrap.dedent("\n".join(l for l in src[a:b].splitlines())), ns)  # the official parser, unchanged
from utils.timefeatures import time_features_from_frequency_str
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
for c in cfgs:
    for k in ("P24_ITR_CI", "P24_NO_TIMESTAMPS"):
        os.environ.pop(k, None)
    os.environ.update(c["env"])
    args = ns["parser"].parse_args(c["argv"])
    args.use_gpu = dev.type == "cuda"
    if c["model"] == "iTransformer":
        from model import iTransformer as M
    else:
        M = __import__("models." + c["model"], fromlist=["Model"])
    torch.manual_seed(0)
    model = M.Model(args).float().to(dev).eval()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_mark = len(time_features_from_frequency_str(args.freq)) + int(bool(getattr(args, "cal_holiday", False)))
    g = torch.Generator().manual_seed(0)
    out = []
    for B in c["batches"]:
        x = torch.randn(B, args.seq_len, args.enc_in, generator=g).to(dev)
        xm = torch.rand(B, args.seq_len, n_mark, generator=g).sub(0.5).to(dev)
        dec = torch.zeros(B, args.label_len + args.pred_len, args.enc_in).to(dev)
        ym = torch.rand(B, args.label_len + args.pred_len, n_mark, generator=g).sub(0.5).to(dev)
        if c["model"] == "PatchTST":
            cal = any(getattr(args, k, False) for k in ("cal_hour", "cal_dow", "cal_holiday"))
            if cal:  # integer calendar values, as the data loader produces them
                xm[..., 0] = torch.randint(0, 24, xm.shape[:2], generator=g).float().div(23).sub(0.5).to(dev)
                xm[..., 1] = torch.randint(0, 7, xm.shape[:2], generator=g).float().div(6).sub(0.5).to(dev)
                if getattr(args, "cal_holiday", False):
                    xm[..., -1] = torch.randint(0, 2, xm.shape[:2], generator=g).float().to(dev)
            f = (lambda: model(x, xm)) if cal else (lambda: model(x))
        elif c["model"] == "DLinear":
            f = lambda: model(x)
        else:  # Autoformer, iTransformer: model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            xm_in = None if (c["model"] == "iTransformer" and os.environ.get("P24_NO_TIMESTAMPS") == "1") else xm
            f = lambda: model(x, xm_in, dec, ym)
        ts = []
        with torch.no_grad():
            for i in range(c["warmup"] + c["reps"]):
                sync(); t0 = time.perf_counter(); f(); sync()
                if i >= c["warmup"]:
                    ts.append(1000 * (time.perf_counter() - t0))
        q = np.percentile(ts, [25, 50, 75])
        out.append(dict(batch=B, median_ms=float(q[1]), p25_ms=float(q[0]), p75_ms=float(q[2])))
    print("P24TIMING " + json.dumps(dict(config_id=c["config_id"], params=params, params_trainable=trainable, res=out,
          device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", torch=torch.__version__)), flush=True)
'''


def configs(plans, batches, reps, warmup):
    seen, out = set(), {}
    for plan in plans:
        for spec in p24.plan_specs(plan, (2021,)):
            if spec.get("mode", "train") != "train":
                continue
            cid = p24.run_id_of(spec).replace("_s2021", "")
            if cid in seen:
                continue
            seen.add(cid)
            m = spec["model"]
            P = spec.get("P", 16)
            argv = p24.model_args(m, spec["dataset"], spec["L"], spec["H"], 2021, P, spec.get("S"))[1:]
            argv += [f"--cal_{n}" for n in p24.CAL_ORDER if n in spec.get("cal", ())]
            env = {}
            if spec.get("no_timestamps"):
                env["P24_NO_TIMESTAMPS"] = "1"
            if spec.get("ci"):
                env["P24_ITR_CI"] = "1"
            repo = p24.MODEL_REPO[m]
            out.setdefault(repo, []).append(dict(
                config_id=cid, model=m, argv=argv, env=env, batches=batches, reps=reps, warmup=warmup,
                script=p24.model_args(m, spec["dataset"], spec["L"], spec["H"], 2021, P, spec.get("S"))[0],
                meta=dict(dataset=spec["dataset"], L=spec["L"], H=spec["H"],
                          patch_len=P if m == "PatchTST" else "", stride=(spec.get("S") or P // 2) if m == "PatchTST" else "",
                          variant="+".join(p24.variant_parts(spec)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="+", default=list(DEFAULT_PLANS))
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 32])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=str(p24.RESULTS / "timing.csv"))
    ap.add_argument("--smoke", action="store_true", help="plan A configs, 3 reps, 1 warm-up (CPU check)")
    a = ap.parse_args()
    if a.smoke:
        a.plans, a.reps, a.warmup = ["A"], 3, 1
    jobs = configs(a.plans, a.batches, a.reps, a.warmup)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for repo, cfgs in jobs.items():
        print(f"{repo}: timing {len(cfgs)} configuration(s)", flush=True)
        p = subprocess.run([sys.executable, "-c", WORKER], cwd=p24.REPO_DIR[repo], input=json.dumps(cfgs),
                           capture_output=True, text=True)
        got = [json.loads(l.split(" ", 1)[1]) for l in p.stdout.splitlines() if l.startswith("P24TIMING ")]
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-3000:])
            print(f"{repo}: worker failed after {len(got)} of {len(cfgs)} configurations")
        meta = {c["config_id"]: c["meta"] for c in cfgs}
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for g in got:
            for r in g["res"]:
                rows.append(dict(config_id=g["config_id"], model=g["config_id"].split("_")[0], **meta[g["config_id"]],
                                 batch=r["batch"], params=g["params"], params_trainable=g["params_trainable"],
                                 median_ms=f"{r['median_ms']:.3f}", p25_ms=f"{r['p25_ms']:.3f}",
                                 p75_ms=f"{r['p75_ms']:.3f}", reps=a.reps, warmup=a.warmup, device=g["device"],
                                 torch=g["torch"], timestamp_utc=ts))
                print(f"  {g['config_id']:45s} batch {r['batch']:>3}: {r['median_ms']:9.3f} ms  params {g['params']}")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
