# CHANGELOG — branch `final-runs` (Task P1)

Base: `82a177c`. Line numbers refer to this branch; lines inside the official repos refer to
the repos after `python p24.py setup` (pinned commit + `patch_repos.py`). Rules 1–6 of the P1
prompt apply throughout: no number in any file is typed in by hand except the pinned checksums
and URLs, values quoted from the plan, and test thresholds stated as such.

## C1 — Dual scoring for PatchTST (`P24_ALL_WINDOWS=1`)
- `patch_repos.py:154-197`: three asserted edits of PatchTST `exp/exp_main.py`:
  - `PatchTST/PatchTST_supervised/exp/exp_main.py:222`: `test()` replaces the official test loader by the same `DataLoader` call with `drop_last=False` (same dataset, batch size, `shuffle=False`, workers). The per-epoch "Test Loss" pass in `train()` is not touched.
  - `PatchTST/PatchTST_supervised/exp/exp_main.py:292`: batches are stacked with `np.concatenate` (same order and values) because the last one is shorter; the official `np.array(...).reshape` path is kept for the flag-off case.
  - `PatchTST/PatchTST_supervised/exp/exp_main.py:310`: prints `P24_OFFICIAL_SUBSET n_scored_official=k ... mse_official mae_official` for the first k = floor(n/b)*b windows, then the official every-window `mse:` line.
- Why the subset is exactly the official result: README section "Why the official subset ..." (eval mode, sequential sampler, identical full batches, per-window model).
- `p24.py:83-87`: new columns appended after the 22 old ones: `mse_official, mae_official, n_scored_official, scoring` (as specified), then `variant` and `vali_loss_best` (needed to keep variants out of baseline tables and for F3 selection). `command()` sets `P24_ALL_WINDOWS=1` for every PatchTST training run (`p24.py:386`); `p24.py:334` parses the subset line; `p24.py:439` sets `scoring`. Old rows are never rewritten; a results file with the old header is refused before any run (`p24.py:309`).
- Semantics of `mse`: module docstring `p24.py:21` and README "The results CSV".
- Test: `tests/test_flags_off.py` -> `tests/output/test_flags_off.txt`.

## C2 — Prediction export (`export=True`, `P24_EXPORT=1`)
- `patch_repos.py:198`: PatchTST `PatchTST/PatchTST_supervised/exp/exp_main.py:327` and LTSF-Linear `LTSF-Linear/exp/exp_main.py:302` save `true.npy` when `P24_EXPORT=1` (iTransformer and Autoformer already save it).
- `p24.py:339`: reads the setting from the log, checks `pred.npy` is from this run, writes `$P24_RESULTS/preds/<run_id>/pred_true.npz` (compressed) + `test_results/*.pdf` + `setting.txt`. Off by default; plans `V` and `onzonal_export` use it.
- `.gitignore`: `preds/` (ON-Zonal `true.npy` is z-scored IESO data, not to be redistributed).

## C3 — PatchTST+Cal (`--cal_hour`, `--cal_dow`, `--cal_holiday`)
- `p24_calendar.py` (new; copied into PatchTST as `layers/P24_calendar.py` by `patch_repos.py:209`): `P24CalendarPE`. Zero embeddings without RNG draw `p24_calendar.py:31`; feature columns from the official feature list `p24_calendar.py:44`; hour formula `p24_calendar.py:64` (from `utils/timefeatures.py:38`), day-of-week formula `p24_calendar.py:69` (from `utils/timefeatures.py:45`); first step of patch j = j*stride `p24_calendar.py:74`.
- Flags with `argparse.SUPPRESS` defaults: `PatchTST/PatchTST_supervised/run_longExp.py:95-98`.
- Model: embeddings built last `PatchTST/PatchTST_supervised/models/PatchTST.py:83-86`; `forward(x, x_mark=None)` `PatchTST/PatchTST_supervised/models/PatchTST.py:88`; backbone `PatchTST/PatchTST_supervised/layers/PatchTST_backbone.py:74`; added next to `W_pos` `PatchTST/PatchTST_supervised/layers/PatchTST_backbone.py:165-166` (official line `PatchTST_backbone.py:165`), shared across channels via `repeat_interleave(n_vars)`.
- `batch_x_mark` passed only when a flag is on: `PatchTST/PatchTST_supervised/exp/exp_main.py:25` and the 8 model calls at `PatchTST/PatchTST_supervised/exp/exp_main.py:73,81,150,164,254,262,357,365` (vali x2, train x2, test x2, predict x2; the official lines 72, 80, 149, 163 from the plan plus the 4 in `test()`/`predict()`, which also call `self.model(batch_x)`).
- Holiday flag: `Dataset_Custom` only `PatchTST/PatchTST_supervised/data_provider/data_loader.py:195`, `PatchTST/PatchTST_supervised/data_provider/data_loader.py:272-276`; passed by `PatchTST/PatchTST_supervised/data_provider/data_factory.py:35` only when `--cal_holiday` is given (other datasets raise `TypeError`).
- V3 hook: `P24_CAL_FREEZE=1` freezes the embeddings at 0 (`p24_calendar.py:58`); used only by plan V.
- Tests (a)-(c): `tests/test_cal.py` -> `tests/output/test_cal.txt`; end to end: plan V smoke (V2, V3) in `tests/output/smoke_all.txt`.

## C4 — Control 2: channel-independent iTransformer (`P24_ITR_CI=1`)
- `patch_repos.py:291`: switch `iTransformer/model/iTransformer.py:22`, reshape `iTransformer/model/iTransformer.py:53-57`, output `iTransformer/model/iTransformer.py:71-74`.
- `p24.py:379` sets the switch; plan `onzonal_ci`.
- Test: `tests/test_itr_ci.py` -> `tests/output/test_itr_ci.txt` (shapes, no parameters added, equals the original model run per variate).

## C5 — Plans (`p24.py`)
- Plan table `p24.py:473`; plan definitions `p24.py:501`; seeds sorted so seed 2021 (the lowest) of every config runs first `p24.py:720`.
- Variant tags in `run_id` and `--des` `p24.py:275`; command builder `p24.py:356`.
- Plans: `V` (+ `check_v` `p24.py:639`, PASS/FAIL against `runs_A.csv:2-3`), `onzonal_seeds`, `sameL`, `onzonal_cal`, `etth1_cal`, `onzonal_export`, `onzonal_ci`, `onzonal_h720`, `grid336`, `ablation_redo` (the 24 runs of plan section 3), `f3` (+ `f3select` `p24.py:686`), `weather` (+ `weather-data` `p24.py:140`: official Autoformer URL, sha256 check, manual step), and Tier 3 `weather_l336`, `onzonal_l336`, `sameL_extra`. ETTm1 keeps its plan; its re-score runs are dropped because dual scoring replaces them (`p24.py:515`).
- Weather configs from the official scripts `p24.py:204`, `p24.py:228`.
- CLI: `run --list`, `--horizons`, `plans`, `check-v`, `f3select`, `weather-data` (`p24.py:749`).
- Tests: `tests/test_plans_unchanged.py` (old plans build identical commands), `tests/smoke_all.sh` -> `tests/output/smoke_all.txt`.

## C6 — `report_tables.py`
- No hard-coded seeds: `SEEDS` and every `len == 3` / seed-list assert removed (old lines 28, 107, 132, 189, 211); each table uses the seeds present in all its cells `report_tables.py:121`; the repro caption counts them `report_tables.py:204`.
- New columns read and normalised `report_tables.py:65`; `score(kind='official'|'all')` `report_tables.py:99`; variants excluded by default `report_tables.py:92`; same-run_id copies must agree on both scores and are merged `report_tables.py:52`.
- `ablation_ETTh1_all.tex` is written only once every cell has every-window scores (`report_tables.py:295`).
- `plot_ablation.py:41`: seed jitter for any seed count (identical for 3 seeds). `tables.py` (notebook quick tables) reads the new columns (`tables.py:43`).
- Tests: `tests/test_report_tables.py` -> `tests/output/test_report_tables.txt` (regression byte-identical; merge; disagreement stops; variants; 5 seeds).

## C7 — `timing.py` (new)
- Builds each planned config with p24's arguments and the official parser/Model, times forward passes at batch 1 and 32 (warm-up, `cuda.synchronize`, median of N) and counts parameters -> `results/timing.csv`. Why untrained weights are fine: module docstring `timing.py:13`.
- Smoke: `tests/output/timing_smoke.txt` (CPU, plan A configs) and `tests/output/timing_all_configs_cpu.txt` (all 108 planned configs, CPU; these CPU latencies are not results).

## C8 — `budget.py` (new)
- GPU-h used per job and per config; projection from the logged seed-2021 time of the same config only, else "not measured" `budget.py:56`. A run counts as done only if this kit logged it, so deliberate re-runs are projected (`budget.py:88`).
- Output on the committed results: `tests/output/budget_results.txt`.

## Compatibility fix found by the smoke test
- `patch_repos.py:91`: `utils/timefeatures.py` of all four repos (e.g. Autoformer `Autoformer/utils/timefeatures.py:110`): `to_offset('t')` -> `to_offset('min')`. pandas >= 3 removed the `'t'` alias, so Autoformer with `--freq t` (its official ETTm1 script, plan M6) crashed at data loading; on pandas 2 the calendar features are byte-identical. Test: `tests/test_freq_alias.py` -> `tests/output/test_freq_alias.txt`.

## C9 — Reproducibility
- `requirements.txt`: `einops` (iTransformer `layers/SelfAttention_Family.py:7`) and `holidays`.
- `notebooks/P24_run.ipynb` (generated by `notebooks/make_notebook.py`): `BASE`, `WORK`, `KIT_BRANCH` parameters (no fixed `/tmp`), `pip freeze` + GPU driver + kit commit -> `results_<JOB>/env.txt`, a JOB letter per plan (table in cell 1), predictions zipped separately as `preds_<JOB>.zip`.
- `reproduce.sh` (new): `tables` and `all`.
- `README.md`: one-command section, plan table, flags, CSV columns and `mse` semantics, why dual scoring is exact, changes 4–7, timing, Weather manual step.

## Not done / open
- I4 (PatchTST+Cal injection point / time anchor) and I5 (DLinear with the 2022 loader) need code that is not in P1.
- `WEATHER_SHA256` and the published Weather targets: \TODO (recorded after the first verified download / from the papers).

## Patch 2 (2026-11-11, after plan V passed on Kaggle)
- `results/runs_V.csv`, `results/env_V.txt`, `logs/logs_V/`: the V gate run (V1a, V1b, V2, V3 PASS on a Tesla T4).
  With it, `report_tables.py results` still writes byte-identical `tables/*.tex` (the V1 row merges with
  its copies in `runs_A.csv`/`runs_C.csv`; the variants stay out).
- `p24.py:737`: `P24_STOP_AFTER_H` (off unless set): no new run starts after that many hours, so a Kaggle
  session ends cleanly before the 12-hour limit; running the plan again resumes. No computation changes.
- Notebook (`notebooks/make_notebook.py`): `STOP_AFTER_H = 9` (line 59); resume from the committed
  `results/runs_<JOB>.csv` when a Kaggle session starts empty (line 98); pinned `einops==0.8.2`,
  `holidays==0.107` (line 82, the versions plan V ran with); JOB table re-allocated so that the jobs that
  must reproduce logged rows (G, K, N, Q) run on Kaggle; JOB S marked optional.
- `requirements.txt`: pinned to `results/env_V.txt`.
- Test: `tests/output/test_stop_resume.txt` (CPU: session 1 stops after the first run, session 2 skips it and
  finishes; V2/V3 bitwise identical again).

## Patch 3 (2026-11-16, after job K)
- Finding: job K (Kaggle T4, torch 2.11.0) re-ran three Job B configs. PatchTST and DLinear are identical
  to `runs_B.csv` in every digit. iTransformer: MSE identical in every printed digit
  (0.28572067618370056); MAE 0.387782484 against 0.387782544 in October (6e-8), which rounds to 0.387782
  against 0.387783. Its training losses differ in the 7th decimal from epoch 1 on (`logs/logs_B` vs `logs/logs_K`).
- `report_tables.py:37`: `DUP_TOL = 1.5e-6`. Copies of one run_id that differ by at most one unit of the
  6th logged decimal are accepted and listed; the first-logged value is used, so existing tables do not
  change. Larger differences still stop the build.
- `tests/test_report_tables.py`: test 3a (one-unit difference accepted, tables unchanged), 3b (1e-5 stops);
  the variant fixture no longer reuses run_ids that are now real runs (`runs_E.csv`).

## Patch 4 (2026-11-17, after jobs F and L)
- `tables/onzonal_rq2.tex`: rebuilt by `python report_tables.py results` now that `runs_F.csv` adds seeds
  2024–25 to every RQ2 cell (5 seeds). No other table changes; with jobs H and J added the tables are the same.
  The 3-seed midterm version stays in git history.
- `tests/test_report_tables.py`: test 5 used invented rows for seeds 2024–25, which are real runs since job F;
  its FIXTURE seeds are now 2026–27 (5a: one cell, tables unchanged; 5b: every RQ2 cell, 7 seeds, only
  `onzonal_rq2.tex` changes). Before this patch, tests 1, 2, 3a, 4, 5a and 5b failed on the committed
  results because the committed RQ2 table was the 3-seed one.
- `tests/output/test_report_tables.txt`: refreshed (PASS 7/7 with the 14 results files of jobs A, B, C, D, E,
  F, G, H, J, K, L, M, V and Vcolab: 252 distinct runs; test 2 reads 15 files because it adds its fixture file).
- `README.md`: notes the 5-seed RQ2 table and what `runs_Vcolab.csv` is.

## Patch 5 (2026-11-18, after job N)
- `tables/ablation_ETTh1_all.tex`: new, written by `python report_tables.py results` once `runs_N.csv` gives
  every-window scores for every ablation cell (the PatchTST patch-length and look-back ablation, scored on every
  test window). No existing table changes. Without it, tests 1, 2, 3a, 4, 5a and 5b failed once N was committed.
- `tests/output/test_report_tables.txt`: refreshed (PASS 7/7 with the results files of jobs A, B, C, D, E, F, G,
  H, J, K, L, M, N, P, Q, V and Vcolab).
- `README.md`: names the new table.

## Patch 6 (2026-11-18, task P3: final tables and figures from the committed results)
- `final_tables.py` (new; called at the end of `report_tables.main`, so `python report_tables.py results` builds
  everything): ten tables, each from `results/` and `published.csv` only, with a suggested caption (LaTeX comment)
  naming scoring, seeds and tests, and the size at which it fits the report's 6.5 in text width:
  `main_ETTh1` (published PatchTST/42, DLinear, iTransformer vs our PatchTST, job A, and PatchTST+Cal, job J;
  official loader; paired PatchTST+Cal - PatchTST), `main_ONZ` (seasonal naive, DLinear, PatchTST, iTransformer,
  PatchTST+Cal at H=96, jobs B/H; the baselines at H=720, job P; every window, seeds 2021-23), `repro_ETTm1` (job D vs
  published, F1 from the dual-scored rows), `rq1_decomposition` (G under pi_0 to pi_3, C_k, Shapley values on the
  F1 x F2 design at common L=96 and L=336; jobs A, C, G, Q, Q2), `f3_selection` (appendix), `rq2_final` (H=96: B+F with
  5 seeds, no tokens E+F, Control 2 L with 3 seeds; D per control, controls compared, Control 2 - iTransformer on
  real and shifted zones, calendar tokens; then H=720, job P), `ablation_cal_ONZ` (job H), `grid_PxL_H336` (jobs M, A,
  G; every window and official loader), `env_per_job` (all results/env_*.txt) and `timing_main` (job T).
  Statistics are report_tables.py's own helpers: mean +- sample std over the seeds present in every cell of a row or
  comparison; paired t-interval within a model; Welch between models and for differences of within-model effects.
- `results/report_numbers.txt` (new, written by the build): 446 labelled lines, `key | value | 95% CI | test, p | seeds`.
- `tests/verify_report_numbers.py` (new): a second, independent computation (csv module, its own run merging, run_id
  lookup and t-intervals) of every line of report_numbers.txt and, number by number in reading order, of every number
  in the ten new tables (762). PASS on the committed results; a changed digit or two swapped rows fail it.
- Figures (`plot_final.py`, new; style in `p24_plotstyle.py`): `fig_pxl_heatmap`, `fig_rq2_forest`, `fig_cal_ablation`,
  `fig_mse_vs_latency` (batch-32 latency, interquartile ranges, Pareto set marked and labelled). Vector PDFs (the heatmap
  and its colour bar are drawn as vector cells), DejaVu Sans, at most three hues, every series with its own marker
  (readable in greyscale). Each plotted number comes from the functions that write the tables.
- Reproducible figures (handoff section 8): PDFs are saved without `CreationDate`, so rebuilds are byte-identical;
  `figures/fig_ablation.pdf` is re-rendered by the unchanged plotting code (same data; the October file was made with
  matplotlib 3.11.2 and carried its build date). PNG copies are written for slides and git-ignored (`figures/*.png`).
  `requirements-tables.txt` (new) pins the table/figure build (numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, matplotlib
  3.10.9; wheels for Python 3.11-3.14); the committed PDFs were rendered with exactly these.
- `reproduce.sh tables`: also runs `plot_final.py`, prints the installed versions against `requirements-tables.txt`,
  and ends with `git status --porcelain -- tables figures results/report_numbers.txt`, failing if anything changed.
- Bug fix, `p24.py f3select`: the PatchTST default-LR candidate is now restricted to the default patch (P=16, S=8).
  Plan grid336 logs PatchTST at L=96, H=336 with P=8/32/64 and no variant tag; the old code took the last such row, so
  the committed `results/f3_selection.json` listed 1.3259569 (the P=64 run) as PatchTST's H=336 validation loss at LR
  1e-4 instead of 1.2891653 (`PatchTST_ETTh1_L96_H336_s2021`, job G). The file is regenerated: that one value changes;
  all eight winners are unchanged (3e-4 still has the lowest loss, 1.2764710), so job Q2 ran the right configurations.
  `final_tables.py` checks the committed selection against the logged validation losses and stops if they disagree.
- `results/env_D_session1.txt` (new): byte-identical to `git show 6269e2c:results/env_D.txt` (job D's first session,
  kit `2a50dc1`; 22 of its 36 runs); session 2 is `env_D.txt`. Both list the same GPU, driver, torch and numpy.
- `tests/test_report_tables.py`: test 1 also compares `report_numbers.txt`; the fixture file is `runs_TEST.csv`
  (T is now a real job, timing); fixture tests compare everything except `env_per_job.tex` and the `env.*` lines, which
  list the results files themselves; test 4's variant fixtures use seed 2026 (the final tables use the real variant runs,
  and a second row for one of their seeds rightly stops the build); 5b expects `onzonal_rq2.tex`, `rq2_final.tex` and
  `report_numbers.txt` to change.
- New tests: `tests/test_final_tables.py` (rebuild, independent second computation, hash-seed independence, format,
  helper fixtures, F3 selection file and the old P=64 value, env_D_session1.txt vs git history, every table compiles and
  fits at its stated size), `tests/test_figures.py` (byte-identical rebuilds, no creation date, committed PDFs with the
  pinned versions, plotted values = report_numbers.txt, vector PDFs). Outputs in `tests/output/`.

## Patch 7 (2026-11-21, task P3: job K prediction analysis)
- `analyze_preds.py` (new): reads `preds_K.zip` in memory (iTransformer, PatchTST, DLinear; ON-Zonal real, L=96, H=96,
  seed 2021), rebuilds ON-Zonal with `onzonal.py`, and writes `preds_analysis.zip` with aggregates and figures only:
  `overall`, `zone_errors`, `error_by_step`, `holiday_vs_rest`, `peak_vs_rest`, `worst_windows`, `examples`,
  `holiday_dates` (CSV), `checks.json`, `MANIFEST.txt` (versions, sha256), `fig_forecast_examples`, `fig_error_by_step`,
  `fig_zone_errors` (PDF + PNG; the example figure's data lines are rasterized). It stops unless every model's exported
  targets equal the rebuilt z-scored data at the window positions (window i targets test rows i..i+95), the three target
  arrays agree, the arrays reproduce `results/runs_K.csv` (MSE and MAE within 5e-6), the rebuilt split equals
  `results/onzonal_stats.json`, and (with `--rebuilt`) the rebuilt seasonal-naive scores equal the committed ones.
  MW errors use the demand as the loaders read it (0.1 MW CSV) and the training mean and population std per zone;
  holidays are `holidays.Canada(subdiv="ON")` as in PatchTST+Cal; a peak is a zone's top 1% of test-split demand;
  worst windows are non-overlapping (origins at least 96 hours apart). The seasonal naive is included for reference.
- `tests/test_analyze_preds.py` (new): synthetic 10-zone data and three synthetic models with known errors; checks the
  outputs (files, columns, raster lines), the closed-form errors, MW and MAPE, holiday and peak counts from a separate
  loop, distinct worst windows, the three stop conditions, and byte-identical reruns. PASS 5/5.
- `requirements-tables.txt`: `holidays==0.107` (for the analysis and its test); `reproduce.sh` and `tests/test_figures.py`
  tolerate a missing package in the version report and compare only the figure-relevant versions.
- README: "Prediction analysis (job K)" with the Colab cells.

## Patch 8 (2026-11-21, task P3: job K aggregates)
- `results/preds_analysis/` (new): the `preds_analysis.zip` that `analyze_preds.py` wrote on Colab on 21 Nov, committed
  as downloaded (16 files; `MANIFEST.txt` sha256 all match). Colab: Python 3.13.18, numpy 2.4.6, pandas 2.2.3,
  matplotlib 3.10.0, holidays 0.107. Its checks: exported targets equal the rebuilt data (max difference 2.3e-7 z-units,
  float32), the arrays reproduce `runs_K.csv` (MSE and MAE within 1e-6), the rebuild equals the committed split and
  seasonal-naive scores. Verified here before committing: every table recombines to the overall rows (zones, steps,
  holiday and peak groups), holiday counts and dates recomputed from the calendar, worst windows non-overlapping and
  consistent with their dates, aggregate columns only, data lines of the example figure rasterized. The `holidays`
  package named 1 Sep 2024 and 2025 "Labor Day" on Colab (the report should write Labour Day); the dates are right.
- `figures/fig_forecast_examples.pdf`, `fig_error_by_step.pdf`, `fig_zone_errors.pdf`: byte-identical copies.
- `final_tables.py`: tables `predsK_failure` (failure categories: all windows, steps 1/24/96, holiday target hours and
  windows, peak zone-hours and zone-windows; MSE and MAPE per model and seasonal naive), `predsK_zones`, `predsK_worst`,
  and 273 `predsK.*` lines in `results/report_numbers.txt` (719 lines in all), each marked "seed 2021 (n=1); one run per
  model, descriptive".
- `tests/verify_report_numbers.py`: recomputes every `predsK.*` line and every number of the three new tables from the
  CSVs with the csv module (719 lines, 986 table numbers: PASS). `tests/test_final_tables.py` check 9: MANIFEST hashes,
  figure copies, the recorded checks, aggregate columns only, raster example lines, and no `preds_*.zip`, `preds/`,
  `.npy`, `.npz` or `onzonal_anomalies.csv` tracked.

## Patch 9 (2026-11-21, task P3: final patch)
- README: "Reproduce" replaces "One command": the clone-to-`CLEAN` steps for every table and figure (CPU), the test
  commands after a fresh clone, the notebook JOB table for retraining (jobs A to T of the final report; W, X, Y and Z
  not run; S is the smoke test), and the branch note (keep `final-runs`, the notebook's `KIT_BRANCH`). The PatchTST+Cal
  parameter sentence now says ETTh1 adds (24+7) x 16 = 496 parameters (hour and weekday only) and ON-Zonal adds
  (24+7+2) x 128 = 4,224. The four plans that were not run are marked in the plan table, and the Weather `\TODO`s are
  replaced by the fact that Weather was not run (also the `WEATHER_SHA256` comment in `p24.py:80`; no code change).
  The `final_tables.py` row lists the job K tables (amended into patch 8).
- `reproduce.sh all`: only the plans the report uses (no `weather-data`, `weather`, `weather_l336`, `onzonal_l336`,
  `sameL_extra`), and F3 in the order of jobs Q and Q2 (`f3 --seeds 2021`, `f3select`, `f3 --seeds 2022 2023`).
- `tests/test_plans_unchanged.py`: the merge check now reads only the October logs (`runs_A/B/C.csv`); with every
  results file it failed once jobs V, G and Q had logged their own runs. New checks: every run of plans V, sameL,
  f3 (seed 2021), ablation_redo and onzonal_export is logged (21/21 PASS, `tests/output/test_plans_unchanged.txt`).
- Closes "Not done / open": the Weather items are moot (Weather was not run); I4 and I5 were not implemented.
