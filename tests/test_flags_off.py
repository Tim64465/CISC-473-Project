"""Rule 2 / C1 acceptance test: with every new switch OFF the patched code behaves exactly
like the official code plus the patches of commit 82a177c; with P24_ALL_WINDOWS=1 the
official subset is exactly the official loader's result.

    python tests/test_flags_off.py OLD_REPOS NEW_REPOS [--epochs 1]

OLD_REPOS: the pinned repos patched with patch_repos.py from 82a177c (git show 82a177c:patch_repos.py).
NEW_REPOS: the same repos patched with this branch's patch_repos.py. Both need ETTh1.csv.
Each command is p24.model_args(...) for seed 2021 with --train_epochs set to `epochs`.
Compared: the whole console log (minus wall-clock lines), the metrics, and pred.npy byte for byte.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p24  # noqa: E402

TIMING = re.compile(r"cost time|speed:|left time")
SETTING_RE = re.compile(r">>>>>>>testing : (.+?)<{5,}")
SUB = {"PatchTST": "PatchTST/PatchTST_supervised", "LTSF-Linear": "LTSF-Linear",
       "Autoformer": "Autoformer", "iTransformer": "iTransformer"}


def run(root, repo, args, env_extra):
    cwd = Path(root) / SUB[repo]
    env = dict(os.environ, SEED="2021", PYTHONUNBUFFERED="1")
    for k in p24.P24_ENV:
        env.pop(k, None)
    env.update(env_extra)
    p = subprocess.run([sys.executable, "-u"] + args, cwd=cwd, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    setting = SETTING_RE.findall(out)[-1]
    res = cwd / "results" / setting
    arrays = {k: np.load(res / f"{k}.npy") for k in ("pred", "true") if (res / f"{k}.npy").exists()}
    return out, arrays


def clean(out):
    return [l for l in out.splitlines() if not TIMING.search(l)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    ok_all = True
    cases = [("PatchTST", "ETTh1", 336, 96), ("iTransformer", "ETTh1", 96, 96), ("DLinear", "ETTh1", 336, 96)]
    for model, ds, L, H in cases:
        repo = p24.MODEL_REPO[model]
        args = p24._set(p24.model_args(model, ds, L, H, 2021), "--train_epochs", a.epochs)
        print(f"\n### {model} {ds} L={L} H={H}, {a.epochs} epoch(s): python {' '.join(args)}", flush=True)
        o_out, o_arr = run(a.old, repo, args, {})
        n_out, n_arr = run(a.new, repo, args, {})
        same_log = clean(o_out) == clean(n_out)
        same_pred = o_arr["pred"].tobytes() == n_arr["pred"].tobytes()
        mse_line = [l for l in clean(o_out) if l.startswith("mse:")][-1]
        print(f"  flags off: console log identical (minus wall-clock lines): {same_log}; pred.npy identical: "
              f"{same_pred} {o_arr['pred'].shape}; {mse_line}")
        if not same_log:
            import difflib
            print("\n".join(list(difflib.unified_diff(clean(o_out), clean(n_out), lineterm=""))[:40]))
        ok_all &= same_log and same_pred
        if model == "DLinear":  # P24_EXPORT=1 adds true.npy and changes nothing else
            e_out, e_arr = run(a.new, repo, args, {"P24_EXPORT": "1"})
            ok = clean(e_out) == clean(o_out) and e_arr["pred"].tobytes() == o_arr["pred"].tobytes() and "true" in e_arr
            print(f"  P24_EXPORT=1: log and pred identical, true.npy saved {e_arr['true'].shape if 'true' in e_arr else None}: {ok}")
            ok_all &= ok
        if model == "PatchTST":  # C1: dual scoring
            d_out, d_arr = run(a.new, repo, args, {"P24_ALL_WINDOWS": "1", "P24_EXPORT": "1"})
            o_lines, d_lines = clean(o_out), clean(d_out)
            cut = lambda ls: ls[:next(i for i, l in enumerate(ls) if l.startswith(">>>>>>>testing"))]
            train_same = cut(o_lines) == cut(d_lines)
            m = p24.OFFICIAL_RE.findall(d_out)[-1]
            k, n = int(m[0]), int(m[1])
            o_mse, o_mae = map(float, p24.MSE_RE.findall(o_out)[-1])
            d_mse, d_mae = map(float, p24.MSE_RE.findall(d_out)[-1])
            sub_same = float(m[3]) == o_mse and float(m[4]) == o_mae
            pred_prefix = d_arr["pred"][:k].tobytes() == o_arr["pred"].tobytes()
            print(f"  P24_ALL_WINDOWS=1: training + per-epoch Test Loss lines identical to the official run: {train_same}")
            print(f"    official loader: {o_arr['pred'].shape[0]} windows scored, mse {o_mse!r} mae {o_mae!r}")
            print(f"    dual: n_test {n}, official subset k = floor({n}/{args[args.index('--batch_size')+1]})*"
                  f"{args[args.index('--batch_size')+1]} = {k}: mse_official {float(m[3])!r} mae_official {float(m[4])!r}"
                  f" -> identical: {sub_same}")
            print(f"    dual pred.npy {d_arr['pred'].shape}; first {k} predictions byte-identical to the official "
                  f"pred.npy: {pred_prefix}; every-window mse {d_mse!r} (diff vs official {d_mse - o_mse:+.3e});"
                  f" true.npy saved {d_arr['true'].shape if 'true' in d_arr else None}")
            ok_all &= train_same and sub_same and pred_prefix and k == o_arr["pred"].shape[0]
    print("\nOVERALL:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
