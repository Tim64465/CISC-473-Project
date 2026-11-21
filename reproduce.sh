#!/usr/bin/env bash
# ./reproduce.sh tables   rebuild every table, figure and results/report_numbers.txt from results/ (no GPU),
#                         then check that git status is clean for tables/, figures/ and report_numbers.txt.
#                         Needs requirements-tables.txt (the figures are byte-identical only with those versions).
# ./reproduce.sh all      setup + ON-Zonal + Weather + every plan (default seeds) + timing + tables + figures.
#                         Needs a Tesla T4 GPU and days of compute; new rows go to results/runs.csv, and
#                         re-runs of logged run_ids must reproduce them exactly (report_tables.py checks).
set -euo pipefail
cd "$(dirname "$0")"
tables() {
  python report_tables.py results   # tables/*.tex (midterm + final) and results/report_numbers.txt
  python plot_ablation.py results   # figures/fig_ablation.pdf
  python plot_final.py results      # figures/fig_pxl_heatmap, fig_rq2_forest, fig_cal_ablation, fig_mse_vs_latency
  python budget.py results          # GPU time per job (printed only)
}
versions() {  # the figures' bytes depend on matplotlib (and numpy): say if this environment differs from the pins
  python - <<'PY'
import importlib.metadata as md
def version(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return "not installed"
pins = dict(l.split("==") for l in open("requirements-tables.txt").read().split() if "==" in l)
diff = {p: (version(p), v) for p, v in pins.items() if version(p) != v}
print("table-build versions:", ", ".join(f"{p} {version(p)}" for p in pins))
if diff:
    print("WARNING: not the pinned versions (" + "; ".join(f"{p} {a} != {b}" for p, (a, b) in diff.items())
          + "): tables are unaffected, figure PDFs may differ in bytes. pip install -r requirements-tables.txt")
PY
}
check_clean() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "not a git checkout: clean-tree check skipped"
    return 0
  fi
  local changed
  changed=$(git status --porcelain -- tables figures results/report_numbers.txt)
  if [ -n "$changed" ]; then
    echo "NOT CLEAN: the rebuild differs from the committed files:"
    echo "$changed"
    git --no-pager diff --stat -- tables figures results/report_numbers.txt
    exit 1
  fi
  echo "CLEAN: tables/, figures/ and results/report_numbers.txt rebuilt byte-identical to the committed files"
}
case "${1:-}" in
  tables)
    versions
    tables
    check_clean
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
