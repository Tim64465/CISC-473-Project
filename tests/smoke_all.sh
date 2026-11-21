#!/usr/bin/env bash
# Smoke test of every new plan: 1 epoch (--smoke), first seed of each plan; rows go to
# $P24_RESULTS/smoke_runs.csv and every console log to $P24_LOGS/*_SMOKE.log.
#   CPU (default): synthetic ON-Zonal/Weather (tests/make_smoke_data.py) and the real ETTh1/ETTm1
#     (their loaders use fixed split borders, so they cannot be shortened). Autoformer is too slow
#     to train on 2 CPU cores: its new configs are checked by timing.py forward passes (end), and
#     this branch does not change its code.
#   GPU (notebook JOB S): P24_SMOKE_SYNTHETIC=0 P24_SMOKE_FULL=1, after onzonal.py has built the real data.
#   P24_WORK=... P24_RESULTS=... P24_LOGS=... bash tests/smoke_all.sh
set -u
cd "$(dirname "$0")/.."
SMOKE="python p24.py run --smoke"
step() { echo; echo "################ $* ($(date -u +%H:%M:%S))"; }
if [ "${P24_SMOKE_SYNTHETIC:-1}" = 1 ]; then
  python tests/make_smoke_data.py "$P24_WORK"
  export P24_WEATHER_SHA256=$(sha256sum "$P24_WORK/data/weather.csv" | cut -d' ' -f1)   # the synthetic file
fi
M6="--only PatchTST iTransformer --horizons 96 720"; I3="--only DLinear"
if [ "${P24_SMOKE_FULL:-0}" = 1 ]; then M6="--only PatchTST Autoformer iTransformer"; I3=""; fi
step "V (V1-V3)";                 $SMOKE --plan V
step "M1 onzonal_nots";           $SMOKE --plan onzonal_nots --seeds 2021
step "M2 onzonal_seeds";          $SMOKE --plan onzonal_seeds --seeds 2024
step "M3 sameL";                  $SMOKE --plan sameL --seeds 2021
step "M4 onzonal_cal";            $SMOKE --plan onzonal_cal --seeds 2021
step "M5 etth1_cal";              $SMOKE --plan etth1_cal --seeds 2021
step "M6 ETTm1 ($M6)";            $SMOKE --plan ETTm1 --seeds 2021 $M6
step "M7 onzonal_export";         $SMOKE --plan onzonal_export
step "M8 onzonal_ci";             $SMOKE --plan onzonal_ci --seeds 2021
step "S1 grid336";                $SMOKE --plan grid336 --seeds 2021
step "S2 ablation_redo";          $SMOKE --plan ablation_redo --seeds 2021
step "S3 onzonal_h720";           $SMOKE --plan onzonal_h720 --seeds 2021
step "S4 f3 (seed 2021)";         $SMOKE --plan f3 --seeds 2021
step "S4 f3select";               P24_F3_SELECTION=$P24_RESULTS/f3_selection_smoke.json python p24.py f3select "$P24_RESULTS" --smoke
step "S4 f3 (seed 2022: non-default winners only)"; P24_F3_SELECTION=$P24_RESULTS/f3_selection_smoke.json $SMOKE --plan f3 --seeds 2022
step "S5 weather";                $SMOKE --plan weather --seeds 2021
step "I1 weather_l336";           $SMOKE --plan weather_l336 --seeds 2021
step "I2 onzonal_l336";           $SMOKE --plan onzonal_l336 --seeds 2021
step "I3 sameL_extra ($I3)";      $SMOKE --plan sameL_extra --seeds 2021 $I3
step "M9 timing: every planned config incl. Autoformer (forward passes, 1 warm-up, 2 reps)"
python timing.py --reps 2 --warmup 1 --out "$P24_RESULTS/timing_smoke.csv"
step "summary"
python - "$P24_RESULTS/smoke_runs.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(f"{len(rows)} smoke runs logged; by plan:", {p: sum(r['plan'] == p for r in rows) for p in dict.fromkeys(r['plan'] for r in rows)})
print("scoring:", {s: sum(r['scoring'] == s for r in rows) for s in sorted({r['scoring'] for r in rows})})
PY
step "done"
