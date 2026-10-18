"""Build ON-Zonal from IESO's public Hourly Zonal Demand reports.

python onzonal.py            download 2015-2025, validate, build, analyse
python onzonal.py --local DIR   use already-downloaded PUB_DemandZonal_YYYY.csv files

Outputs (data files are NOT committed to git: IESO's terms allow reproduction
with their notice but not sublicensing, so the repo ships this script instead):
  work/data/ONZ_real.csv     date + 10 zone columns (MW), hourly
  work/data/ONZ_shift.csv    same, but zone k is lagged by k whole weeks (Control 1)
  results/onzonal_checks.json      data-quality checks and the anomaly count
  results/onzonal_anomalies.csv    hour-to-hour jumps flagged for the anomaly log
  results/onzonal_stats.json       similarity statistics (train split only)
  results/onzonal_seasonal_naive.csv   seasonal-naive baseline on the test split
  results/fig_onzonal_corr.png     zone correlation, real vs shifted

Conventions
- Timestamp = Date + (Hour - 1) hours, i.e. the start of IESO's "hour ending" interval.
- Provincial columns (Ontario Demand, Zone Total, Diff) are dropped so no input is a
  sum of the others.
- Both datasets drop the first 9 weeks so they have identical rows and splits.
- Splits follow Dataset_Custom in the official repos: first 70% train, last 20% test,
  10% validation in between; z-scores use training-split statistics only.
"""
import argparse
import json
import math
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("P24_WORK", KIT / "work"))
RESULTS = Path(os.environ.get("P24_RESULTS", KIT / "results"))
DATA = WORK / "data"
YEARS = list(range(2015, 2026))
URL = "https://reports-public.ieso.ca/public/DemandZonal/PUB_DemandZonal_{}.csv"
ZONES = ["Northwest", "Northeast", "Ottawa", "East", "Toronto", "Essa", "Bruce",
         "Southwest", "Niagara", "West"]
HEADER = ["Date", "Hour", "Ontario Demand"] + ZONES + ["Zone Total", "Diff"]
WEEK = 168
MAX_SHIFT_WEEKS = len(ZONES) - 1  # zone k is lagged k weeks, k = 0..9


def download(raw: Path):
    raw.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        f = raw / f"PUB_DemandZonal_{y}.csv"
        if not f.exists():
            req = urllib.request.Request(URL.format(y), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(f, "wb") as out:
                out.write(r.read())
            print("downloaded", f.name)


def parse_year(f: Path) -> pd.DataFrame:
    lines = f.read_text().splitlines()
    idx = [i for i, l in enumerate(lines) if l.startswith("Date,Hour")]
    assert len(idx) == 1, f"{f.name}: header line not found"
    df = pd.read_csv(f, skiprows=idx[0])
    assert list(df.columns) == HEADER, f"{f.name}: unexpected columns {list(df.columns)}"
    return df


def build(raw: Path):
    frames = [parse_year(raw / f"PUB_DemandZonal_{y}.csv") for y in YEARS]
    df = pd.concat(frames, ignore_index=True)
    checks = {}
    hours_per_day = df.groupby("Date")["Hour"].agg(["count", "min", "max"])
    bad_days = hours_per_day[(hours_per_day["count"] != 24) | (hours_per_day["min"] != 1)
                             | (hours_per_day["max"] != 24)]
    checks["days"] = int(len(hours_per_day))
    checks["days_without_exactly_hours_1_to_24"] = bad_days.reset_index().astype(str).to_dict("records")
    df["date"] = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hour"] - 1, unit="h")
    checks["duplicate_timestamps"] = int(df["date"].duplicated().sum())
    df = df.drop_duplicates("date").set_index("date").sort_index()
    full = pd.date_range(df.index[0], df.index[-1], freq="h")
    missing = full.difference(df.index)
    checks["missing_hours"] = [str(t) for t in missing[:50]]
    checks["n_missing_hours"] = int(len(missing))
    z = df[ZONES].astype(float).reindex(full)
    checks["nan_cells_before_fill"] = int(z.isna().sum().sum())
    checks["non_positive_cells"] = int((z <= 0).sum().sum())
    diff = df["Zone Total"] - df["Ontario Demand"]
    checks["zone_total_minus_ontario_MW"] = {"mean": float(diff.mean()), "p1": float(diff.quantile(.01)),
                                             "p99": float(diff.quantile(.99))}
    checks["diff_column_matches_zone_total_minus_ontario"] = bool((diff == df["Diff"]).all())
    # Linear interpolation only for missing hours (counted above); none expected.
    z = z.interpolate(limit_direction="both")
    checks["first_timestamp"], checks["last_timestamp"] = str(z.index[0]), str(z.index[-1])
    checks["rows"] = int(len(z))

    # Anomaly log: hour-to-hour jumps larger than 8 robust SDs for that zone.
    d = z.diff()
    mad = (d - d.median()).abs().median() * 1.4826
    flag = (d - d.median()).abs() > 8 * mad
    an = []
    for zone in ZONES:
        for t in d.index[flag[zone].fillna(False).values]:
            an.append(dict(timestamp=str(t), zone=zone, value_MW=float(z.at[t, zone]),
                           change_MW=float(d.at[t, zone])))
    checks["n_anomaly_flags"] = len(an)
    return z, checks, pd.DataFrame(an)


def make_shifted(z: pd.DataFrame) -> pd.DataFrame:
    s = z.copy()
    for k, zone in enumerate(ZONES):
        s[zone] = z[zone].shift(k * WEEK)  # value at row t is zone k's demand k weeks earlier
    return s


def split_bounds(n):
    num_train, num_test = int(n * 0.7), int(n * 0.2)
    return num_train, n - num_test  # train = [0, num_train), test = [n - num_test, n)


def zscore_train(x: np.ndarray, num_train: int):
    mu, sd = x[:num_train].mean(0), x[:num_train].std(0)
    return (x - mu) / sd, mu, sd


def pc1_share(x: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(np.corrcoef(x, rowvar=False))
    return float(ev.max() / ev.sum())


def hour_of_week_residual(df: pd.DataFrame, num_train: int) -> np.ndarray:
    """Remove each zone's mean hour-of-week profile (estimated on train rows)."""
    how = (df.index.dayofweek * 24 + df.index.hour).values
    x = df[ZONES].values.astype(float)
    prof = np.vstack([x[:num_train][how[:num_train] == h].mean(0) for h in range(168)])
    return x - prof[how]


def seasonal_naive(x_raw: np.ndarray, mu, sd, test_start: int, H: int):
    """Forecast row o+h with the value 168*ceil(h/168) hours earlier (same hour of week)."""
    n = len(x_raw)
    xn = (x_raw - mu) / sd
    origins = np.arange(test_start - 1, n - H)  # last observed row of each window
    lags = WEEK * np.ceil(np.arange(1, H + 1) / WEEK).astype(int)
    se = ae = ae_mw = ape = 0.0
    cnt = 0
    for o in origins:
        tgt = o + np.arange(1, H + 1)
        src = tgt - lags
        e = xn[tgt] - xn[src]
        e_mw = x_raw[tgt] - x_raw[src]
        se += (e ** 2).sum(); ae += np.abs(e).sum(); ae_mw += np.abs(e_mw).sum()
        ape += (np.abs(e_mw) / x_raw[tgt]).sum()
        cnt += e.size
    return dict(H=H, n_windows=len(origins), mse=se / cnt, mae=ae / cnt, mae_MW=ae_mw / cnt,
                mape_pct=100 * ape / cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="folder with PUB_DemandZonal_YYYY.csv files")
    a = ap.parse_args()
    raw = Path(a.local) if a.local else DATA / "ieso_raw"
    if not a.local:
        download(raw)
    z, checks, anomalies = build(raw)
    trim = MAX_SHIFT_WEEKS * WEEK
    real = z.iloc[trim:]
    shift = make_shifted(z).iloc[trim:]
    assert not shift.isna().any().any() and len(real) == len(shift)
    checks["rows_after_trim"] = int(len(real))
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, df in (("ONZ_real", real), ("ONZ_shift", shift)):
        out = df.copy()
        out.index.name = "date"
        out.reset_index().to_csv(DATA / f"{name}.csv", index=False, float_format="%.1f")
    anomalies.to_csv(RESULTS / "onzonal_anomalies.csv", index=False)
    (RESULTS / "onzonal_checks.json").write_text(json.dumps(checks, indent=2))

    n = len(real)
    num_train, test_start = split_bounds(n)
    stats = dict(rows=n, num_train=num_train, test_start=test_start,
                 train_period=[str(real.index[0]), str(real.index[num_train - 1])],
                 test_period=[str(real.index[test_start]), str(real.index[-1])])
    corr = {}
    for name, df in (("real", real), ("shift", shift)):
        xz, _, _ = zscore_train(df[ZONES].values.astype(float), num_train)
        res = hour_of_week_residual(df, num_train)
        stats[f"pc1_share_{name}"] = pc1_share(xz[:num_train])
        stats[f"pc1_share_residual_{name}"] = pc1_share(res[:num_train])
        c = np.corrcoef(xz[:num_train], rowvar=False)
        cr = np.corrcoef(res[:num_train], rowvar=False)
        off = ~np.eye(len(ZONES), dtype=bool)
        stats[f"mean_offdiag_corr_{name}"] = float(c[off].mean())
        stats[f"mean_offdiag_corr_residual_{name}"] = float(cr[off].mean())
        corr[name] = cr
    (RESULTS / "onzonal_stats.json").write_text(json.dumps(stats, indent=2))

    x = real[ZONES].values.astype(float)
    _, mu, sd = zscore_train(x, num_train)
    rows = [seasonal_naive(x, mu, sd, test_start, H) for H in (96, 192, 336, 720)]
    pd.DataFrame(rows).to_csv(RESULTS / "onzonal_seasonal_naive.csv", index=False, float_format="%.6f")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for i, name in enumerate(("real", "shift")):
            im = ax[i].imshow(corr[name], vmin=-1, vmax=1, cmap="RdBu_r")
            ax[i].set_xticks(range(10), ZONES, rotation=90, fontsize=7)
            ax[i].set_yticks(range(10), ZONES, fontsize=7)
            ax[i].set_title(f"{'Real' if name == 'real' else 'Shifted (Control 1)'}: residual correlation",
                            fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(RESULTS / "fig_onzonal_corr.png", dpi=200)
    except Exception as e:  # the figure is optional
        print("figure skipped:", e)

    # Make the two CSVs visible to the official repos' 'custom' loaders.
    for sub in ("PatchTST/PatchTST_supervised/dataset", "LTSF-Linear/dataset", "iTransformer/dataset/ETT-small"):
        d = WORK / "repos" / sub
        if d.parent.exists():
            d.mkdir(parents=True, exist_ok=True)
            for name in ("ONZ_real", "ONZ_shift"):
                shutil.copy(DATA / f"{name}.csv", d / f"{name}.csv")
    print(json.dumps({k: checks[k] for k in ("rows", "rows_after_trim", "n_missing_hours",
                                             "duplicate_timestamps", "n_anomaly_flags")}, indent=1))
    print(json.dumps(stats, indent=1))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
