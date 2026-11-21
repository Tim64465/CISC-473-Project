"""CPU test of analyze_preds.py on synthetic data and synthetic predictions (no IESO data).

    python tests/test_analyze_preds.py

Everything here is a FIXTURE: an hourly 10-zone series (sines + noise, 3000 rows from 2023-09-01, so the
test split covers Christmas, Boxing Day and New Year's Day), and three "models" whose errors are known:
  iTransformer  +0.1 (z) in Northwest only        -> MSE 0.001 overall, 0.01 in Northwest, 0 elsewhere
  PatchTST      +0.01 h (z) at step h, every zone  -> MSE (0.01 h)^2 at step h
  DLinear       +1 (z) on statutory-holiday hours  -> MSE 1 on holiday hours, 0 on the others
1. The script runs end to end and writes preds_analysis.zip with the listed files and nothing else (no
   arrays); every CSV column is an error, a count, a label or a date; the forecast-example lines are raster.
2. The numbers equal the closed forms above, MW errors and MAPE equal hand computations, and the holiday
   and peak counts equal counts made here by a separate loop over dates and rows.
3. Worst windows are distinct episodes (forecast origins at least 96 hours apart), and every DLinear window
   with an error touches a holiday (the FIXTURE is short, so fewer than 10 episodes exist per ranking).
4. The script stops when: the exported targets are shifted by one window; a logged score differs; the split
   differs from the committed statistics.
5. Two runs give byte-identical output.
"""
import io
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import analyze_preds as AP  # noqa: E402
from onzonal import ZONES, split_bounds  # noqa: E402

OK = []


def check(name, ok, detail=""):
    OK.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# ------------------------------------------------------------------ FIXTURE data and predictions
N_ROWS, H = 3000, 96
START = datetime(2023, 9, 1)
rng = np.random.default_rng(473)
t = np.arange(N_ROWS)
x = np.stack([1000 + 150 * k + (80 + 10 * k) * np.sin(2 * np.pi * (t + k) / 24) + 40 * np.sin(2 * np.pi * t / 168)
              + rng.normal(0, 15, N_ROWS) for k in range(len(ZONES))], axis=1)
stamps = [START + timedelta(hours=int(i)) for i in t]
csv_df = pd.DataFrame(x, columns=ZONES)
csv_df.insert(0, "date", [s.strftime("%Y-%m-%d %H:%M:%S") for s in stamps])
num_train, test_start = split_bounds(N_ROWS)
n_win = N_ROWS - test_start - H + 1
HOLIDAYS = {date(2023, 12, 25), date(2023, 12, 26), date(2024, 1, 1)}  # Ontario statutory days in the test targets


def fixture_inputs(d, shift_true=False, logged_delta=0.0, stats_delta=0):
    """Write the FIXTURE csv, preds zip, runs csv and stats json into d; return the CLI arguments."""
    d = Path(d)
    csv_df.to_csv(d / "ONZ_real.csv", index=False, float_format="%.1f")
    xr = pd.read_csv(d / "ONZ_real.csv")[ZONES].to_numpy()            # what the loaders read (0.1 MW)
    mu, sd = xr[:num_train].mean(0), xr[:num_train].std(0)
    tgt = test_start + np.arange(n_win)[:, None] + np.arange(H)[None, :]
    true = ((xr - mu) / sd)[tgt].astype(np.float32)
    hol = np.array([[stamps[r].date() in HOLIDAYS for r in row] for row in tgt])
    preds = {"iTransformer": true + np.float32(0.1) * (np.arange(len(ZONES)) == 0),
             "PatchTST": true + (0.01 * np.arange(1, H + 1, dtype=np.float32))[None, :, None],
             "DLinear": true + hol[:, :, None].astype(np.float32)}
    runs = []
    with zipfile.ZipFile(d / "preds_K.zip", "w") as z:
        for m, p in preds.items():
            rid = f"{m}_ONZ_real_L96_H96_s2021"
            tt = np.roll(true, 1, axis=0) if shift_true else true
            buf = io.BytesIO()
            np.savez_compressed(buf, pred=p.astype(np.float32), true=tt)
            z.writestr(f"results_K/preds/{rid}/pred_true.npz", buf.getvalue())
            z.writestr(f"results_K/preds/{rid}/setting.txt", "FIXTURE\n")
            runs.append(dict(run_id=rid, mse=round(float(np.mean((p - true) ** 2)), 6) + logged_delta,
                             mae=round(float(np.mean(np.abs(p - true))), 6)))
    pd.DataFrame(runs).to_csv(d / "runs_K.csv", index=False)
    stats = dict(rows=N_ROWS, num_train=num_train, test_start=test_start + stats_delta,
                 test_period=[str(pd.Timestamp(stamps[test_start])), str(pd.Timestamp(stamps[-1]))])
    (d / "stats.json").write_text(json.dumps(stats))
    return ["--preds", str(d / "preds_K.zip"), "--data", str(d / "ONZ_real.csv"), "--runs", str(d / "runs_K.csv"),
            "--stats", str(d / "stats.json"), "--out", str(d / "out" / "preds_analysis.zip")], xr, mu, sd, hol


def run(args):
    Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
    AP.main(args)
    with zipfile.ZipFile(args[-1]) as z:
        return {n.split("/", 1)[1]: z.read(n) for n in z.namelist()}


# 1 ---------------------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as d1:
    args, xr, mu, sd, hol = fixture_inputs(d1)
    files = run(args)
    tab = {n[:-4]: pd.read_csv(io.BytesIO(b)) for n, b in files.items() if n.endswith(".csv")}
    checks = json.loads(files["checks.json"])
want = {"checks.json", "MANIFEST.txt", "overall.csv", "zone_errors.csv", "error_by_step.csv", "holiday_vs_rest.csv",
        "peak_vs_rest.csv", "worst_windows.csv", "examples.csv", "holiday_dates.csv"} | {
    f"{f}.{e}" for f in ("fig_forecast_examples", "fig_error_by_step", "fig_zone_errors") for e in ("pdf", "png")}
allowed_cols = {"model", "zone", "h", "group", "unit", "n", "n_cells", "mse_z", "mae_z", "mae_MW", "mape_pct",
                "n_cells_without_mape", "ranking", "rank", "forecast_origin", "first_target", "last_target",
                "window_mse_z", "worst_zone", "touches_holiday", "contains_peak_hour", "panel", "rule", "window",
                "date", "name"} | {f"mse_z_{m}" for m in AP.MODELS} | {
    f"mse_z_{m}_{s}" for m in AP.MODELS for s in (AP.EXAMPLE_ZONE, "all_zones")}
cols = set().union(*(t.columns for t in tab.values()))
raster = b"/Subtype /Image" in files["fig_forecast_examples.pdf"] or b"/Subtype/Image" in files["fig_forecast_examples.pdf"]
check("1. runs end to end: the listed files only, aggregate columns only, example lines rasterized",
      set(files) == want and cols <= allowed_cols and raster and max(len(t) for t in tab.values()) <= 4 * H,
      f"extra files {sorted(set(files) - want)}, missing {sorted(want - set(files))}, other columns {sorted(cols - allowed_cols)}")

# 2 ---------------------------------------------------------------------------------------
ov = tab["overall"].set_index("model")
zt = tab["zone_errors"].set_index(["model", "zone"])
st = tab["error_by_step"].set_index(["model", "h"])
hv = tab["holiday_vs_rest"].set_index(["model", "group"])
tgt = test_start + np.arange(n_win)[:, None] + np.arange(H)[None, :]
demand = xr[tgt]
c1 = (abs(ov.loc["iTransformer", "mse_z"] - 0.001) < 1e-6 and abs(zt.loc[("iTransformer", "Northwest"), "mse_z"] - 0.01) < 1e-6
      and zt.loc[("iTransformer", "Ottawa"), "mse_z"] < 1e-12)
mw = 0.1 * sd[0]
c2 = (abs(zt.loc[("iTransformer", "Northwest"), "mae_MW"] - mw) < 1e-3
      and abs(zt.loc[("iTransformer", "Northwest"), "mape_pct"] - 100 * np.mean(mw / demand[:, :, 0])) < 1e-4)
c3 = all(abs(st.loc[("PatchTST", h), "mse_z"] - (0.01 * h) ** 2) < 1e-6 for h in range(1, H + 1))
n_hol_hours = int(hol.sum())
n_hol_windows = sum(any(stamps[r].date() in HOLIDAYS for r in range(test_start + i, test_start + i + H)) for i in range(n_win))
c4 = (abs(hv.loc[("DLinear", "holiday target hours"), "mse_z"] - 1) < 1e-6 and hv.loc[("DLinear", "other target hours"), "mse_z"] < 1e-12
      and hv.loc[("DLinear", "holiday target hours"), "n"] == n_hol_hours
      and hv.loc[("DLinear", "windows touching a holiday"), "n"] == n_hol_windows
      and sorted(tab["holiday_dates"]["date"]) == sorted(str(h_) for h_ in HOLIDAYS))
pk = tab["peak_vs_rest"].set_index(["model", "zone", "group"])
c5 = True
for j, zname in enumerate(ZONES):
    thr = np.quantile(xr[test_start:, j], 0.99)
    rows = [r for r in range(test_start, N_ROWS) if xr[r, j] >= thr]
    pairs = sum(1 for r in rows for i in range(n_win) if test_start + i <= r < test_start + i + H)  # (window, step) cells
    wins = sum(1 for i in range(n_win) if any(test_start + i <= r < test_start + i + H for r in rows))
    c5 &= pk.loc[("PatchTST", zname, "peak target hours"), "n"] == pairs
    c5 &= pk.loc[("PatchTST", zname, "windows containing a peak hour"), "n"] == wins
check("2. closed forms: iTransformer zone error, MW and MAPE, PatchTST step error, DLinear holiday error; counts",
      c1 and c2 and c3 and c4 and c5, f"zone {c1}, MW/MAPE {c2}, step {c3}, holiday {c4}, peak counts {c5}")

# 3 ---------------------------------------------------------------------------------------
ww = tab["worst_windows"]
gaps_ok = all((pd.to_datetime(g["forecast_origin"]).sort_values().diff().dropna() >= pd.Timedelta(hours=H)).all()
              for _, g in ww.groupby("ranking"))
dl = ww[ww.ranking == "DLinear"]
dl_err = dl[dl["window_mse_z"] > 1e-9]  # windows with any DLinear error: exactly those with holiday hours
check("3. worst windows: distinct episodes; DLinear's with an error all touch a holiday",
      gaps_ok and len(dl_err) >= 2 and dl_err["touches_holiday"].all(), f"gaps {gaps_ok}, DLinear error windows {len(dl_err)}")

# 4 ---------------------------------------------------------------------------------------
def stops(**kw):
    with tempfile.TemporaryDirectory() as d:
        a = fixture_inputs(d, **kw)[0]
        Path(a[-1]).parent.mkdir(parents=True, exist_ok=True)
        try:
            AP.main(a)
        except AssertionError as e:
            return str(e)
    return ""


s1, s2, s3 = stops(shift_true=True), stops(logged_delta=1e-4), stops(stats_delta=1)
check("4. stops on shifted targets, a logged-score mismatch and a split mismatch",
      "alignment" in s1 and "logged" in s2 and "committed" in s3, f"{s1[:60]!r} / {s2[:60]!r} / {s3[:60]!r}")

# 5 ---------------------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as d2:
    args2 = fixture_inputs(d2)[0]
    files2 = run(args2)
diff = sorted(n for n in files if files[n] != files2.get(n) and n != "MANIFEST.txt")
check("5. two runs give byte-identical files", not diff and json.loads(files2["checks.json"]) == checks, f"differ: {diff}")

print("\nOVERALL:", "PASS" if all(OK) else "FAIL", f"({sum(OK)}/{len(OK)} checks)")
sys.exit(0 if all(OK) else 1)
