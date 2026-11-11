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
