"""Job K prediction analysis: errors of iTransformer, PatchTST and DLinear on ON-Zonal (real zones,
L=96, H=96, seed 2021), from preds_K.zip. Runs on Colab; only aggregates and figures leave it.

    python onzonal.py                                 # rebuild ON-Zonal from IESO (work/data/ONZ_real.csv)
    python analyze_preds.py --preds preds_K.zip       # writes preds_analysis.zip

preds_K.zip holds IESO-derived values (forecasts and z-scored demand), so it never goes to GitHub or
to a chat. This script reads it in memory (nothing is extracted) and writes preds_analysis.zip with:

  checks.json                 alignment and reproduction checks (the script stops if one fails)
  overall.csv                 per model: MSE, MAE (z-scored), MAE in MW, MAPE; seasonal naive for reference
  zone_errors.csv             the same per zone
  error_by_step.csv           the same per forecast step h = 1..96 (all zones)
  holiday_vs_rest.csv         target hours on an Ontario statutory holiday vs the other target hours, and
                              windows whose 96 target hours touch a holiday vs the other windows
  peak_vs_rest.csv            per zone and pooled: target hours in that zone's top 1% of test-split demand
                              vs the rest, and windows containing such an hour vs the rest
  worst_windows.csv           per model and for the three-model mean: the 10 worst windows that do not
                              overlap, by date (forecast origin), with their errors only
  examples.csv                the windows drawn in fig_forecast_examples (dates, zone, errors only)
  holiday_dates.csv           the statutory holidays inside the test targets (holidays package, as PatchTST+Cal)
  fig_forecast_examples.pdf   a median-error window, the worst holiday-start window and a demand-peak window
                              (Toronto, last 96 input hours + 96 forecast hours; data lines rasterized)
  fig_error_by_step.pdf       MSE and MAPE by forecast step
  fig_zone_errors.pdf         MSE and MAPE by zone
  (+ PNG copies of the figures, MANIFEST.txt with versions and sha256 of every file)

Definitions (one seed, 2021, so every number is descriptive: no intervals across seeds):
  * windows: the test split's every window, in loader order; window i forecasts rows
    test_start + i .. test_start + i + 95; its forecast origin (last input hour) is row test_start + i - 1.
  * z-scored errors are on the scale the models were trained and scored on (training mean and std of each
    zone, population std as in the loaders' StandardScaler); MW errors are pred * std + mean - demand,
    with the demand as the loaders read it (work/data/ONZ_real.csv, 0.1 MW); MAPE = 100 mean(|error| / demand).
  * holidays: holidays.Canada(subdiv="ON"), the definition PatchTST+Cal uses (holidays==0.107; the Civic
    Holiday is not a statutory holiday there).
  * peak: a zone's demand at or above the 99th percentile of that zone's demand over the test split.
Checks before anything is written: every model's `true` equals the rebuilt z-scored data at the window
positions above; the three `true` arrays agree; MSE and MAE from the arrays reproduce results/runs_K.csv;
the rebuilt dataset matches the committed results/onzonal_stats.json (rows and split) and, with
--rebuilt, the committed seasonal-naive scores.
"""
import argparse
import hashlib
import io
import json
import platform
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from onzonal import WEEK, ZONES, split_bounds  # noqa: E402

H = 96
MODELS = ("iTransformer", "PatchTST", "DLinear")
NAIVE = "Seasonal naive"
EXAMPLE_ZONE = "Toronto"
PEAK_Q = 0.99
N_WORST = 10
TOL_TRUE = 1e-4      # z-units: float32 storage of the loaders' float64 z-scores
TOL_LOGGED = 5e-6    # runs_K.csv logs 6 decimals; float32 means in the official metric


# ------------------------------------------------------------------ inputs
def read_preds(zip_path):
    """{model: (pred, true)} from every */preds/<run_id>/pred_true.npz inside the zip, read in memory."""
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            m = re.search(r"(?:^|/)preds/([^/]+)/pred_true\.npz$", name)
            if not m:
                continue
            run_id = m.group(1)
            model = run_id.split("_")[0]
            assert model in MODELS and run_id == f"{model}_ONZ_real_L96_H{H}_s2021", f"unexpected run {run_id}"
            with np.load(io.BytesIO(z.read(name))) as a:
                out[model] = (run_id, a["pred"].astype(np.float64), a["true"].astype(np.float64))
    assert sorted(out) == sorted(MODELS), f"expected one export per model {MODELS}, found {sorted(out)}"
    return out


def read_data(csv_path):
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["date"] + ZONES, f"unexpected columns {list(df.columns)}"
    idx = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    secs = idx.values.astype("datetime64[s]").astype(np.int64)  # unit-free (pandas 3 parses to microseconds)
    assert (np.diff(secs) == 3600).all(), "rows are not consecutive hours"
    return idx, df[ZONES].to_numpy(np.float64)


def ontario_holidays(years):
    import holidays
    return holidays.Canada(subdiv="ON", years=sorted(years)), holidays.__version__


# ------------------------------------------------------------------ metrics
def metrics(e_z, e_mw, demand, mask=None):
    """MSE, MAE (z), MAE (MW), MAPE over the cells in mask (all cells when None)."""
    if mask is not None:
        e_z, e_mw, demand = e_z[mask], e_mw[mask], demand[mask]
    pos = demand > 0
    return dict(n_cells=int(e_z.size), mse_z=float(np.mean(e_z ** 2)), mae_z=float(np.mean(np.abs(e_z))),
                mae_MW=float(np.mean(np.abs(e_mw))),
                mape_pct=float(100 * np.mean(np.abs(e_mw[pos]) / demand[pos])) if pos.any() else float("nan"),
                n_cells_without_mape=int((~pos).sum()))


def non_overlapping_worst(score, k, gap):
    """Indices of the k highest scores, no two closer than gap windows (distinct episodes)."""
    order, chosen = np.argsort(-score, kind="stable"), []
    for i in order:
        if all(abs(int(i) - j) >= gap for j in chosen):
            chosen.append(int(i))
            if len(chosen) == k:
                break
    return chosen


# ------------------------------------------------------------------ analysis
def analyze(preds, idx, x, logged=None, committed_stats=None):
    n = len(x)
    num_train, test_start = split_bounds(n)
    n_win = n - test_start - H + 1
    mu, sd = x[:num_train].mean(0), x[:num_train].std(0)  # StandardScaler: population std
    xz = (x - mu) / sd
    tgt = test_start + np.arange(n_win)[:, None] + np.arange(H)[None, :]          # (N, H) target rows
    true_rebuilt = xz[tgt]                                                          # (N, H, Z)
    demand = x[tgt]
    checks = dict(rows=n, num_train=num_train, test_start=test_start, n_windows=n_win,
                  test_period=[str(idx[test_start]), str(idx[-1])])
    if committed_stats is not None:
        for k in ("rows", "num_train", "test_start", "test_period"):
            assert checks[k] == committed_stats[k], f"rebuilt {k} {checks[k]} != committed {committed_stats[k]}"
        checks["matches_committed_onzonal_stats"] = True
    trues = []
    for m in MODELS:
        run_id, p, t = preds[m]
        assert p.shape == t.shape == (n_win, H, len(ZONES)), f"{m}: shape {p.shape}, expected {(n_win, H, len(ZONES))}"
        d = float(np.max(np.abs(t - true_rebuilt)))
        checks[f"{m}.max_abs_true_minus_rebuilt_z"] = d
        assert d < TOL_TRUE, f"{m}: exported targets differ from the rebuilt data by {d} (alignment or data mismatch)"
        trues.append(t)
        mse, mae = float(np.mean((p - t) ** 2)), float(np.mean(np.abs(p - t)))
        checks[f"{m}.mse_from_arrays"], checks[f"{m}.mae_from_arrays"] = mse, mae
        if logged is not None:
            lm, la = logged[run_id]
            checks[f"{m}.mse_logged"], checks[f"{m}.mae_logged"] = lm, la
            assert abs(mse - lm) < TOL_LOGGED and abs(mae - la) < TOL_LOGGED, \
                f"{m}: arrays give mse {mse:.7f} mae {mae:.7f}, runs_K.csv logged {lm} {la}"
    checks["max_abs_true_between_models_z"] = float(max(np.max(np.abs(a - trues[0])) for a in trues[1:]))

    # errors: z-scored and in MW; seasonal naive (same hour one week earlier) for reference
    err = {}
    for m in MODELS:
        p = preds[m][1]
        err[m] = (p - true_rebuilt, p * sd + mu - demand)
    src = tgt - WEEK * np.ceil(np.arange(1, H + 1) / WEEK).astype(int)[None, :]
    err[NAIVE] = (xz[src] - true_rebuilt, x[src] - demand)
    order = list(MODELS) + [NAIVE]

    # calendar masks
    stamps = idx.values[tgt]                                                        # (N, H) datetime64
    day = stamps.astype("datetime64[D]")
    hour = ((stamps - day) // np.timedelta64(1, "h")).astype(int)
    hol, hol_version = ontario_holidays(set(pd.DatetimeIndex(day.ravel()).year))
    hol_days = sorted({d for d in hol if idx[test_start].date() <= d <= idx[-1].date()})
    is_hol = np.isin(day, np.array(hol_days, dtype="datetime64[D]"))
    checks["holidays_version"] = hol_version
    checks["holiday_dates_in_test_targets"] = len(hol_days)
    thr = np.quantile(x[test_start:], PEAK_Q, axis=0)                               # per zone, test split
    peak = demand >= thr[None, None, :]                                             # (N, H, Z)

    rows = {k: [] for k in ("overall", "zone", "step", "holiday", "peak")}
    for m in order:
        ez, emw = err[m]
        rows["overall"].append(dict(model=m, **metrics(ez, emw, demand)))
        for j, zname in enumerate(ZONES):
            rows["zone"].append(dict(model=m, zone=zname, **metrics(ez[:, :, j], emw[:, :, j], demand[:, :, j])))
        for h in range(H):
            rows["step"].append(dict(model=m, h=h + 1, **metrics(ez[:, h], emw[:, h], demand[:, h])))
        hmask = np.broadcast_to(is_hol[:, :, None], ez.shape)
        wmask = np.broadcast_to(is_hol.any(1)[:, None, None], ez.shape)
        for grp, msk, unit, cnt in (("holiday target hours", hmask, "target hours", int(is_hol.sum())),
                                    ("other target hours", ~hmask, "target hours", int((~is_hol).sum())),
                                    ("windows touching a holiday", wmask, "windows", int(is_hol.any(1).sum())),
                                    ("other windows", ~wmask, "windows", int((~is_hol.any(1)).sum()))):
            rows["holiday"].append(dict(model=m, group=grp, unit=unit, n=cnt, **metrics(ez, emw, demand, msk)))
        for j, zname in list(enumerate(ZONES)) + [(None, "all zones")]:
            sl = slice(None) if j is None else j
            pk = peak[:, :, sl]
            pw = np.broadcast_to(pk.any(1, keepdims=True), pk.shape)
            e1, e2, dd = ez[:, :, sl], emw[:, :, sl], demand[:, :, sl]
            for grp, msk, unit, cnt in (("peak target hours", pk, "zone-hours", int(pk.sum())),
                                        ("other target hours", ~pk, "zone-hours", int((~pk).sum())),
                                        ("windows containing a peak hour", pw, "zone-windows", int(pk.any(1).sum())),
                                        ("other windows", ~pw, "zone-windows", int((~pk.any(1)).sum()))):
                rows["peak"].append(dict(model=m, zone=zname, group=grp, unit=unit, n=cnt, **metrics(e1, e2, dd, msk)))

    # worst windows (distinct episodes), per model and for the three-model mean
    win_mse = {m: np.mean(err[m][0] ** 2, axis=(1, 2)) for m in order}
    win_mse["three-model mean"] = np.mean([win_mse[m] for m in MODELS], axis=0)
    worst = []
    for m in list(MODELS) + ["three-model mean"]:
        for r, i in enumerate(non_overlapping_worst(win_mse[m], N_WORST, H), 1):
            per_zone = {m2: np.mean(err[m2][0][i] ** 2, axis=0) for m2 in MODELS}
            zmse = per_zone[m] if m in MODELS else np.mean([per_zone[m2] for m2 in MODELS], axis=0)
            worst.append(dict(ranking=m, rank=r, forecast_origin=str(idx[test_start + i - 1]),
                              first_target=str(idx[test_start + i]), last_target=str(idx[test_start + i + H - 1]),
                              window_mse_z=float(win_mse[m][i]),
                              **{f"mse_z_{m2}": float(win_mse[m2][i]) for m2 in MODELS},
                              worst_zone=ZONES[int(np.argmax(zmse))], touches_holiday=bool(is_hol[i].any()),
                              contains_peak_hour=bool(peak[i].any())))

    # example windows (Toronto): median error, worst holiday-start window, a demand peak at step 48
    zj = ZONES.index(EXAMPLE_ZONE)
    mean3 = win_mse["three-model mean"]
    i_med = int(np.argsort(mean3, kind="stable")[(n_win - 1) // 2])
    starts = np.where(is_hol[:, 0] & (hour[:, 0] == 0))[0]
    assert len(starts), "no window starts at 00:00 on a holiday"
    i_hol = int(starts[np.argmax(mean3[starts])])
    r_peak = test_start + int(np.argmax(x[test_start:, zj]))
    i_peak = int(np.clip(r_peak - test_start - 47, 0, n_win - 1))
    ex = []
    for panel, rule, i in (("typical", "median three-model mean MSE over all windows", i_med),
                           ("holiday", "largest three-model mean MSE among windows whose first target hour is 00:00 "
                                       "on a statutory holiday", i_hol),
                           ("peak", f"{EXAMPLE_ZONE}'s highest test-split demand hour at forecast step 48", i_peak)):
        ex.append(dict(panel=panel, rule=rule, window=i, zone=EXAMPLE_ZONE, forecast_origin=str(idx[test_start + i - 1]),
                       first_target=str(idx[test_start + i]),
                       **{f"mse_z_{m}_{EXAMPLE_ZONE}": float(np.mean(err[m][0][i, :, zj] ** 2)) for m in MODELS},
                       **{f"mse_z_{m}_all_zones": float(win_mse[m][i]) for m in MODELS}))

    tables = dict(overall=pd.DataFrame(rows["overall"]), zone_errors=pd.DataFrame(rows["zone"]),
                  error_by_step=pd.DataFrame(rows["step"]), holiday_vs_rest=pd.DataFrame(rows["holiday"]),
                  peak_vs_rest=pd.DataFrame(rows["peak"]), worst_windows=pd.DataFrame(worst),
                  examples=pd.DataFrame(ex),
                  holiday_dates=pd.DataFrame([dict(date=str(d), name=hol.get(d)) for d in hol_days]))
    plot = dict(idx=idx, x=x, mu=mu, sd=sd, preds={m: preds[m][1] for m in MODELS}, test_start=test_start,
                examples=ex, zone=zj)
    return tables, checks, plot


# ------------------------------------------------------------------ figures
def figures(tables, plot, out_dir):
    from p24_plotstyle import AQUA, BLUE, FULL_W, INK, INK2, MUTED, ORANGE, plt, save, setup, tidy
    setup()
    col = {"iTransformer": (ORANGE, "s"), "PatchTST": (BLUE, "o"), "DLinear": (AQUA, "^"), NAIVE: (MUTED, "x")}
    files = []
    # forecast examples: data lines rasterized, so the PDF does not carry the demand series as vector paths
    x, ts, zj = plot["x"], plot["test_start"], plot["zone"]
    fig, axes = plt.subplots(3, 1, figsize=(FULL_W, 6.6))
    titles = {"typical": "(a) median-error window", "holiday": "(b) worst window starting on a statutory holiday",
              "peak": f"(c) window with {EXAMPLE_ZONE}'s highest test demand at step 48"}
    for ax, e in zip(axes, plot["examples"]):
        i = e["window"]
        rows_ = np.arange(ts + i - H, ts + i + H)                     # last 96 input hours, then the 96 targets
        ax.plot(np.arange(-H + 1, H + 1), x[rows_, zj], color=INK, lw=1.3, rasterized=True)
        for m in MODELS:
            c, _ = col[m]
            ax.plot(np.arange(1, H + 1), plot["preds"][m][i, :, zj] * plot["sd"][zj] + plot["mu"][zj], color=c, lw=1.2,
                    rasterized=True)
        ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
        mse = ", ".join(f"{m} {e[f'mse_z_{m}_{EXAMPLE_ZONE}']:.3f}" for m in MODELS)
        ax.set_title(f"{titles[e['panel']]}: {EXAMPLE_ZONE}, forecast from {e['forecast_origin'][:16]}\n"
                     f"MSE in this window ({EXAMPLE_ZONE}, z-scored): {mse}", loc="left", fontsize=8, color=INK)
        ax.set_ylabel("demand (MW)")
        tidy(ax)
    axes[-1].set_xlabel("hours relative to the forecast origin (input: last 96 hours; forecast: next 96)")
    handles = [plt.Line2D([], [], color=INK, lw=1.3, label="actual")] + [
        plt.Line2D([], [], color=col[m][0], lw=1.2, label=m) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    files.append(save(fig, "fig_forecast_examples", out_dir))
    # error by step
    st = tables["error_by_step"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8))
    for ax, metric, lab in ((axes[0], "mse_z", "MSE (z-scored)"), (axes[1], "mape_pct", "MAPE (%)")):
        for m in list(MODELS) + [NAIVE]:
            c, mk = col[m]
            s = st[st.model == m]
            ax.plot(s["h"], s[metric], color=c, lw=1.4 if m != NAIVE else 1.0, ls="-" if m != NAIVE else (0, (3, 2)),
                    marker=mk, markevery=12, ms=4.5, mfc="white" if m != NAIVE else c, label=m)
        ax.set_xlabel("forecast step h (hours ahead)")
        ax.set_ylabel(lab)
        ax.set_xticks([1, 24, 48, 72, 96])
        tidy(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    files.append(save(fig, "fig_error_by_step", out_dir))
    # zone errors
    zt = tables["zone_errors"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.3), sharey=True)
    y = np.arange(len(ZONES))[::-1]
    for ax, metric, lab in ((axes[0], "mse_z", "MSE (z-scored)"), (axes[1], "mape_pct", "MAPE (%)")):
        for k, m in enumerate(list(MODELS) + [NAIVE]):
            c, mk = col[m]
            s = zt[zt.model == m].set_index("zone").loc[ZONES]
            ax.plot(s[metric], y + (k - 1.5) * 0.15, ls="", marker=mk, ms=5, color=c,
                    mfc="white" if m in MODELS else c, mew=1.1, label=m)
        ax.set_xlabel(lab)
        tidy(ax, "x")
    axes[0].set_yticks(y, ZONES)
    axes[0].tick_params(axis="y", length=0)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    files.append(save(fig, "fig_zone_errors", out_dir))
    return files


# ------------------------------------------------------------------ output
def write_zip(out_zip, tables, checks, fig_files, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6f")
    files = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != "MANIFEST.txt")
    import matplotlib
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    matplotlib=matplotlib.__version__, holidays=checks["holidays_version"])
    man = ["preds_analysis: aggregates and figures only (no forecasts, no hourly demand). Seed 2021, one run per model.",
           "versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()), ""]
    man += [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in files]
    (out_dir / "MANIFEST.txt").write_text("\n".join(man) + "\n")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.iterdir()):
            z.write(p, f"preds_analysis/{p.name}")
    print(f"wrote {out_zip}: {', '.join(p.name for p in sorted(out_dir.iterdir()))}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", required=True, help="preds_K.zip")
    ap.add_argument("--data", default=str(KIT / "work" / "data" / "ONZ_real.csv"), help="ONZ_real.csv from onzonal.py")
    ap.add_argument("--runs", default=str(KIT / "results" / "runs_K.csv"), help="logged scores to reproduce")
    ap.add_argument("--stats", default=str(KIT / "results" / "onzonal_stats.json"), help="committed split stats")
    ap.add_argument("--rebuilt", help="P24_RESULTS folder of the rebuild: its seasonal-naive file must match the committed one")
    ap.add_argument("--out", default="preds_analysis.zip")
    a = ap.parse_args(argv)
    runs = pd.read_csv(a.runs)
    logged = {r.run_id: (float(r.mse), float(r.mae)) for r in runs.itertuples()}
    stats = json.loads(Path(a.stats).read_text())
    idx, x = read_data(a.data)
    tables, checks, plot = analyze(read_preds(a.preds), idx, x, logged, stats)
    if a.rebuilt:
        mine = pd.read_csv(Path(a.rebuilt) / "onzonal_seasonal_naive.csv")
        ours = pd.read_csv(KIT / "results" / "onzonal_seasonal_naive.csv")
        assert mine.equals(ours), "the rebuilt seasonal-naive scores differ from the committed ones (IESO data changed?)"
        checks["rebuilt_seasonal_naive_equals_committed"] = True
    naive = tables["overall"].set_index("model").loc[NAIVE, "mse_z"]
    checks["seasonal_naive_mse_from_csv"] = float(naive)
    out_zip = Path(a.out).resolve()
    out_dir = out_zip.parent / "preds_analysis"
    if out_dir.exists():
        for p in out_dir.iterdir():
            p.unlink()
    figs = figures(tables, plot, out_dir)
    write_zip(out_zip, tables, checks, figs, out_dir)
    print(json.dumps(checks, indent=1))
    print(tables["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
