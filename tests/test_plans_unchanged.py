"""The plans that produced the logged results build the same commands as at commit 82a177c.

    python tests/test_plans_unchanged.py OLD_P24_PY

OLD_P24_PY: `git show 82a177c:p24.py > /tmp/p24_old.py`. For plans A, ablation, onzonal,
onzonal_nots and ETTm1 we compare, run by run, the run_id, the full command line and the
environment switches. Expected, and reported: (1) PatchTST training runs now also get
P24_ALL_WINDOWS=1 (dual scoring; tests/test_flags_off.py shows it leaves training and the
official score unchanged); (2) plan ETTm1 no longer has re-score runs (dual scoring replaces them).
Also checks that the run_ids of the re-run plans exist in the logged CSVs, so they merge.
"""
import csv
import importlib.util
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import p24  # noqa: E402

spec = importlib.util.spec_from_file_location("p24_old", sys.argv[1])
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
OK = []


def check(name, ok, detail=""):
    OK.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def old_command(s):  # the argument logic of the old run_one (82a177c p24.py:203-225)
    model, ds, L, H, seed = s["model"], s["dataset"], s["L"], s["H"], s["seed"]
    P, S, mode = s.get("P", 16), s.get("S"), s.get("mode", "train")
    S = S if S is not None else P // 2
    tag = f"_P{P}S{S}" if model == "PatchTST" and (P != 16 or S != 8) else ""
    if s.get("no_timestamps"):
        tag += "_nots"
    run_id = f"{model}_{ds}_L{L}_H{H}_s{seed}{tag}" + ("_bs1" if mode == "rescore_bs1" else "")
    args = old.model_args(model, ds, L, H, seed, P, S, epochs=s.get("epochs"), patience=s.get("patience"))
    if mode == "rescore_bs1":
        args = old._set(args, "--is_training", 0)
        args = old._set(args, "--batch_size", 1)
    env = {}
    if s.get("no_timestamps"):
        env["P24_NO_TIMESTAMPS"] = "1"
        args = old._set(args, "--des", f"Exp_s{seed}_nots")
    return run_id, args, env


seeds = (2021, 2022, 2023)
for plan in ("A", "ablation", "onzonal", "onzonal_nots", "ETTm1"):
    o = [old_command(s) for s in old.plan_specs(plan, list(seeds))]
    n = [p24.command(s) for s in p24.plan_specs(plan, seeds)]
    n = [(rid, args, env) for rid, _, args, env in n]
    if plan == "ETTm1":
        o = [x for x in o if not x[0].endswith("_bs1")]
    same_ids = [x[0] for x in o] == [x[0] for x in n]
    same_args = all(a[1] == b[1] for a, b in zip(o, n))
    env_diff = {tuple(sorted(set(b[2].items()) - set(a[2].items()))) for a, b in zip(o, n)} - {()}
    env_ok = env_diff <= {(("P24_ALL_WINDOWS", "1"),)} and all(
        (b[2].get("P24_ALL_WINDOWS") == "1") == (b[0].startswith("PatchTST") and not b[0].endswith("_bs1")) for b in n)
    check(f"plan {plan}: {len(n)} runs, same run_ids in the same order, identical command lines",
          same_ids and same_args)
    check(f"plan {plan}: environment identical except P24_ALL_WINDOWS=1 on PatchTST training runs", env_ok,
          f"added switches: {sorted(env_diff)}")
r_old = sum(1 for s in old.plan_specs("ETTm1", list(seeds)) if s.get("mode") == "rescore_bs1")
check(f"plan ETTm1: the {r_old} re-score runs are gone (dual scoring logs both scores in the training run)",
      not any(s.get("mode") == "rescore_bs1" for s in p24.plan_specs("ETTm1", seeds)))

logged = {}
for f in sorted((KIT / "results").glob("runs_*.csv")):
    with open(f) as fh:
        for i, r in enumerate(csv.DictReader(fh), start=2):
            logged.setdefault(r["run_id"], f"{f.name}:{i}")
for plan, expect in (("ablation_redo", 24), ("onzonal_export", 3), ("V", 1)):
    ids = [p24.run_id_of(s) for s in p24.plan_specs(plan, p24.PLANS[plan][1])]
    hit = [i for i in ids if i in logged]
    check(f"plan {plan}: {len(hit)} run_ids are logged runs (expected {expect}); they must reproduce them",
          len(hit) == expect, ", ".join(f"{i} = {logged[i]}" for i in hit[:3]) + (" ..." if len(hit) > 3 else ""))
same = [p24.run_id_of(s) for s in p24.plan_specs("sameL", seeds) if p24.run_id_of(s) in logged]
check(f"plan sameL: {len(same)} run_ids are logged runs (the 6 PatchTST L=96 runs of runs_C.csv)",
      len(same) == 6 and all(logged[i].startswith("runs_C") for i in same))
f3d = [p24.run_id_of(s) for s in p24.plan_specs("f3", (2021,)) if p24.run_id_of(s) in logged]
check(f"plan f3: {len(f3d)} default-LR runs re-run logged iTransformer rows (validation loss was not logged)",
      len(f3d) == 4 and all(i.startswith("iTransformer") and logged[i].startswith("runs_A") for i in f3d))
print("\nOVERALL:", "PASS" if all(OK) else "FAIL", f"({sum(OK)}/{len(OK)} checks)")
sys.exit(0 if all(OK) else 1)
