"""GPU time used so far, and a projection for the runs still to do (no estimates).

    python budget.py results                     every plan except those already finished
    python budget.py results --plans sameL onzonal_cal
    python budget.py results --csv out/          also write budget_used.csv and budget_projection.csv

Used: wall-clock `seconds` of every logged run (results/**/runs*.csv; smoke files are
ignored), summed per job (runs_<JOB>.csv or results_<JOB>/) and per configuration
(run_id without its seed).
Projection: a run that is not logged yet costs the logged time of the SAME configuration
with seed 2021 (the plans run seed 2021 of every config first; if that run was logged twice,
the longer of the two times). With no such row the run is
counted as "not measured"; nothing is extrapolated from other configurations.
A run counts as done only if this kit logged it (column `scoring` filled): the new plans
deliberately re-train some logged configs (dual scoring, export, validation loss), and those
re-runs are projected from their own logged seed-2021 time.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p24  # noqa: E402

FINISHED = ("A", "ablation", "onzonal")  # logged in runs_A/B/C.csv
SEED_RE = re.compile(r"_s(\d+)(?=_|$)")


def config_key(run_id):
    return SEED_RE.sub("", run_id, count=1)


def job_of(path: Path):
    m = re.match(r"runs_(\w+)\.csv$", path.name)
    if m:
        return m.group(1)
    m = re.match(r"results_(\w+)$", path.parent.name)
    return m.group(1) if m else str(path)


def load(dirs):
    rows = []
    for d in dirs:
        for f in sorted(Path(d).rglob("runs*.csv")):
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    if r.get("mse") and r.get("seconds"):
                        r["_job"] = job_of(f)
                        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=["results"])
    ap.add_argument("--plans", nargs="+", default=[p for p in p24.PLANS if p not in FINISHED + ("V",)])
    ap.add_argument("--csv", help="folder for budget_used.csv and budget_projection.csv")
    a = ap.parse_args()
    rows = load(a.dirs)
    if not rows:
        sys.exit(f"no logged runs under {a.dirs}")

    print("== GPU time used, per job ==")
    by_job = defaultdict(list)
    for r in rows:
        by_job[r["_job"]].append(r)
    for job, rs in sorted(by_job.items()):
        devs = sorted({r["device"] for r in rs})
        print(f"job {job:6s} {len(rs):4d} runs  {sum(float(r['seconds']) for r in rs)/3600:7.2f} GPU-h  {devs}")
    print(f"total      {len(rows):4d} runs  {sum(float(r['seconds']) for r in rows)/3600:7.2f} GPU-h "
          "(a run logged by two jobs is counted twice: it ran twice)")

    used = defaultdict(lambda: dict(n=0, sec=0.0, s2021=None, device=set()))
    for r in rows:
        u = used[config_key(r["run_id"])]
        u["n"] += 1
        u["sec"] += float(r["seconds"])
        u["device"].add(r["device"])
        if r["seed"] == "2021" and (u["s2021"] is None or float(r["seconds"]) > u["s2021"][0]):
            u["s2021"] = (float(r["seconds"]), r["device"])  # logged twice: the longer time (conservative)
    print("\n== GPU time used, per configuration (run_id without seed) ==")
    for k, u in sorted(used.items(), key=lambda kv: -kv[1]["sec"]):
        s21 = f"seed 2021: {u['s2021'][0]:.0f} s" if u["s2021"] else "seed 2021: not logged"
        print(f"{k:55s} {u['n']:2d} runs {u['sec']/3600:6.2f} GPU-h  {s21}  {sorted(u['device'])}")
    logged = {r["run_id"] for r in rows if r.get("scoring")}  # logged by the final-runs kit
    any_logged = {r["run_id"] for r in rows}

    print("\n== Projection for runs not logged yet (logged seed-2021 time of the same config) ==")
    proj_rows, grand, grand_nm = [], 0.0, 0
    for plan in a.plans:
        try:
            specs = p24.plan_specs(plan, p24.PLANS[plan][1])
        except SystemExit as e:  # e.g. f3 phase 2 before the selection exists
            print(f"{plan}: {e}")
            continue
        todo = [s for s in specs if p24.run_id_of(s) not in logged]
        rerun = sum(1 for s in todo if p24.run_id_of(s) in any_logged)
        sec, nm, per = 0.0, 0, defaultdict(lambda: [0, None])
        for s in todo:
            rid = p24.run_id_of(s)
            k = config_key(rid)
            t = used[k]["s2021"] if k in used else None
            per[k][0] += 1
            per[k][1] = t
            if t is None:
                nm += 1
            else:
                sec += t[0]
            proj_rows.append(dict(plan=plan, plan_id=p24.PLANS[plan][0], run_id=rid, config=k,
                                  seconds_seed2021=f"{t[0]:.0f}" if t else "not measured",
                                  device=t[1] if t else ""))
        grand += sec
        grand_nm += nm
        print(f"{plan:15s} {p24.PLANS[plan][0]:8s} {len(todo):3d} of {len(specs):3d} runs to do: "
              f"{sec/3600:6.2f} GPU-h for {len(todo)-nm} measured" + (f" + {nm} not measured" if nm else "")
              + (f"  ({rerun} re-run(s) of logged run_ids)" if rerun else ""))
        for k, (n, t) in per.items():
            if t is None:
                continue
            print(f"    {k:50s} x{n}: {t[0]:.0f} s each (seed 2021, {t[1]})")
    print(f"\nall listed plans: {grand/3600:.2f} GPU-h for the measured runs; {grand_nm} run(s) not measured")
    if a.csv:
        out = Path(a.csv)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "budget_used.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config", "runs", "gpu_hours", "seed2021_seconds", "devices"])
            for k, u in sorted(used.items()):
                w.writerow([k, u["n"], f"{u['sec']/3600:.3f}", f"{u['s2021'][0]:.0f}" if u["s2021"] else "",
                            ";".join(sorted(u["device"]))])
        with open(out / "budget_projection.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(proj_rows[0]) if proj_rows else ["plan"])
            w.writeheader()
            w.writerows(proj_rows)
        print(f"wrote {out}/budget_used.csv and budget_projection.csv")


if __name__ == "__main__":
    main()
