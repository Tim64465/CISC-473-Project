# P24 — Protocol or Architecture?

CISC 473 Deep Learning, Fall 2026. Sneh Mangroliya, Tirthkumar Patel.

A controlled re-evaluation of PatchTST, iTransformer and DLinear (with Autoformer as the
older baseline) on ETTh1/ETTm1, and a new ten-zone Ontario demand dataset (ON-Zonal)
built from IESO's public reports.

## Rebuild the midterm report's tables and figure

```bash
pip install -r requirements.txt
python report_tables.py results     # writes tables/*.tex (Tables 4-7, 9) and prints every number quoted in the text
python plot_ablation.py results     # writes figures/fig_ablation.pdf (Figure 1)
```

`results/` holds every logged run: `runs_A.csv` (ETTh1 reproduction and F1, Job A),
`runs_B.csv` (ON-Zonal RQ2, Job B), `runs_C.csv` (PatchTST ablation, Job C), plus the
ON-Zonal build outputs (`onzonal_checks.json`, `onzonal_stats.json`,
`onzonal_seasonal_naive.csv`, `fig_onzonal_corr.png` = Figure 2). Console logs for Jobs B
and C are in `logs/`. All runs used NVIDIA Tesla T4 GPUs on Kaggle and Colab. Jobs A and C
share six PatchTST runs; `report_tables.py` checks that they agree exactly (they do).

## What is here

| File | Purpose |
|---|---|
| `p24.py` | Clones the four **official** repositories at pinned commits, applies `patch_repos.py`, downloads and checksums the ETT data, runs experiment plans, and logs every run to `results/runs.csv`. |
| `patch_repos.py` | The only changes made to the official code (see below). |
| `onzonal.py` | Downloads IESO Hourly Zonal Demand 2015–2025, validates it, builds `ONZ_real.csv` and the shifted control `ONZ_shift.csv`, and computes the similarity statistics and the seasonal-naive baseline. |
| `report_tables.py`, `plot_ablation.py` | Build the report's tables and Figure 1 from `results/`. |
| `tables.py` | Quick per-job tables used inside the notebook. |
| `published.csv` | Published target numbers, each tagged with its paper and table. |
| `notebooks/P24_run.ipynb` | One notebook for Kaggle or Colab; choose a JOB (A–E). |

## Re-run the experiments

```bash
python p24.py setup                                   # repos + data, verified
python p24.py run --plan A --seeds 2021 2022 2023     # ETTh1 reproduction + F1 re-scoring
python onzonal.py                                     # build ON-Zonal (needs internet)
python p24.py run --plan onzonal --seeds 2021 2022 2023
python p24.py run --plan ablation --seeds 2021 2022 2023
python p24.py run --plan onzonal_nots --seeds 2021 2022 2023   # Week 7: iTransformer without calendar tokens
```

A GPU is strongly recommended. Every run appends one row to the results CSV (model, data,
look-back, horizon, seed, test batch size, windows scored, epochs, MSE, MAE, time, device,
repo commit). Re-running a plan skips runs that are already logged.

## Official code used (pinned)

| Model | Repository | Commit |
|---|---|---|
| PatchTST | yuqinie98/PatchTST (`PatchTST_supervised`) | `204c21e` |
| DLinear | cure-lab/LTSF-Linear | `0c11366` |
| Autoformer | thuml/Autoformer | `51c7d41` |
| iTransformer | thuml/iTransformer | `c2426e6` |

Hyper-parameters are copied from each repository's own ETTh1/ETTm1 script. For Autoformer,
whose ETTh1 script uses horizons 24/48/168/336/720, we use the same arguments with
horizons 96/192/336/720.

## Changes to the official code

`patch_repos.py` makes only these changes, and asserts that each one applied:

1. `np.Inf` → `np.inf`, `df.drop([...], 1)` → `df.drop(columns=[...])`,
   `s.apply(f, 1)` → `s.apply(f)`, so the 2021–2023 code runs on numpy ≥ 2 and pandas ≥ 2.
   The models, training loops and metrics are unchanged.
2. LTSF-Linear, Autoformer and iTransformer hard-code their seed (2021, 2021, 2023). They now
   read it from the `SEED` environment variable, with the original value as the default.
3. An optional switch, off by default: with `P24_NO_TIMESTAMPS=1`, iTransformer does not
   receive its four calendar tokens (hour of day, day of week, day of month, day of year),
   exactly as its own code already does for the PEMS and Solar datasets. Only the
   `onzonal_nots` plan sets it.

## Test-loader behaviour at the pinned commits (checked in `data_provider/data_factory.py`)

| Repo | Test batch size | `drop_last` on test |
|---|---|---|
| PatchTST | `--batch_size` (128) | **True**: the last partial batch is never scored |
| LTSF-Linear | `--batch_size` | False (changed from True on 2023-03-15, commit `6fe4c28`) |
| Autoformer | `--batch_size` | False (changed from True on 2022-12-22, commit `d910070`) |
| iTransformer | forced to 1 | True, but with batch size 1 nothing is dropped |

Factor F1 re-scores each PatchTST checkpoint with batch size 1, so every test window counts.

## Data licence

ETT: from github.com/zhouhaoyi/ETDataset (checksums in `p24.py`).

ON-Zonal is built from IESO's Hourly Zonal Demand reports
(https://reports-public.ieso.ca/public/DemandZonal/). Copyright © Independent Electricity
System Operator; subject to the Terms of Use at www.ieso.ca. The data files are not
redistributed in this repository; `onzonal.py` downloads them. `results/` contains only
aggregate statistics computed from them.
