#!/usr/bin/env bash
# ./reproduce.sh tables   rebuild every table, the ablation figure and the GPU-time summary from results/ (no GPU)
# ./reproduce.sh all      setup + ON-Zonal + Weather + every plan (default seeds) + timing + tables + figures.
#                         Needs a Tesla T4 GPU and days of compute; new rows go to results/runs.csv, and
#                         re-runs of logged run_ids must reproduce them exactly (report_tables.py checks).
set -euo pipefail
cd "$(dirname "$0")"
tables() {
  python report_tables.py results
  python plot_ablation.py results
  python budget.py results
}
case "${1:-}" in
  tables)
    tables
    ;;
  all)
    python -m pip install -r requirements.txt
    python p24.py setup
    python onzonal.py
    python p24.py weather-data          # stops with the manual step if the download is refused
    python p24.py run --plan V          # validation gate: prints PASS/FAIL
    for plan in A ablation onzonal onzonal_nots onzonal_seeds sameL onzonal_cal etth1_cal onzonal_export \
                onzonal_ci grid336 ablation_redo onzonal_h720 f3 weather weather_l336 onzonal_l336 sameL_extra; do
      python p24.py run --plan "$plan"
    done
    python p24.py run --plan ETTm1 --only PatchTST Autoformer iTransformer
    python p24.py f3select results
    python p24.py run --plan f3 --seeds 2022 2023
    python timing.py
    tables
    ;;
  *)
    echo "usage: ./reproduce.sh tables | all" >&2
    exit 2
    ;;
esac
