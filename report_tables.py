"""Rebuild every table in the midterm report from the run logs.

    python report_tables.py results            # writes tables/*.tex and prints checks

Inputs (all in the results folder):
  runs_*.csv                  one row per run, written by p24.py
  onzonal_seasonal_naive.csv  written by onzonal.py
and published.csv (next to this script).

Nothing is typed by hand: every number in tables/*.tex comes from these files.
Statistics: mean +- sample std over seeds; 95% t-intervals; paired where the design
pairs runs (same checkpoint re-scored; same model and seed on real vs shifted data),
Welch intervals between different architectures.

Seeds are not hard-coded: each table uses the seeds present in every one of its cells
(printed), and needs at least 2. Two scores are derived for every row (see load()):
  mse_off  the model's own (official) test loader
  mse_all  every test window
For PatchTST the every-window score comes from the batch-1 re-score row when one exists,
otherwise from the dual-scored training row (scoring = all+official). The two never mix in
one table: each table states which one it uses.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("P24_TABLES", HERE / "tables"))
H_ALL = (96, 192, 336, 720)
MIN_SEEDS = 2  # a sample std needs two values
SCORE_COLS = ("mse_off", "mae_off", "mse_all", "mae_all")


# ------------------------------------------------------------------ loading
def load(results):
    files = sorted(glob.glob(str(Path(results) / "**" / "runs*.csv"), recursive=True))
    assert files, f"no runs*.csv under {results}"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["mse"].notna()]
    df = normalise(df)
    # The same run_id can appear in several jobs (Jobs A and C share six PatchTST runs; plans V,
    # sameL, ablation_redo and onzonal_export re-train logged configs). Training is deterministic
    # on T4, so every copy must give the same official and every-window scores; then keep one
    # copy that carries both scores.
    dup = df[df.duplicated("run_id", keep=False)]
    for rid, g in dup.groupby("run_id"):
        for c in SCORE_COLS:
            v = g[c].dropna()
            assert v.nunique() <= 1, f"{rid}: duplicate runs disagree on {c}: {list(v)}"
    n_dup = dup["run_id"].nunique()
    for c in SCORE_COLS:
        df[c] = df.groupby("run_id")[c].transform("first")  # first non-missing value of each run_id
    df["_prio"] = (df["scoring"] != "all+official").astype(int)
    df = df.sort_values("_prio", kind="stable").drop_duplicates("run_id").sort_index().drop(columns="_prio")
    print(f"{len(df)} distinct runs from {len(files)} file(s); {n_dup} run_id(s) logged more than once, all copies agreeing")
    return df


def normalise(df):
    """Columns added on branch final-runs are blank in older rows; derive them.

    scoring (blank in old rows): rescore_bs1 rows scored every window ('all'); every other
    old row used the model's official loader ('official').
    mse_off/mae_off: official-loader score. mse_all/mae_all: every-window score. In every row
    `mse` is computed over `n_scored` of `n_test_windows` windows.
    """
    df = df.copy()
    for c in ("mse_official", "mae_official", "n_scored_official", "scoring", "variant"):
        if c not in df:
            df[c] = np.nan
    df["variant"] = df["variant"].fillna("").astype(str)
    old = df["scoring"].isna()
    df.loc[old & df["run_id"].str.contains("_nots"), "variant"] = "nots"  # tag used before the column existed
    df["scoring"] = df["scoring"].where(~old, np.where(df["mode"] == "rescore_bs1", "all", "official"))
    sc = df["scoring"]
    every = df["n_scored"] == df["n_test_windows"]
    for m in ("mse", "mae"):
        df[f"{m}_off"] = np.where(sc == "official", df[m], np.where(sc == "all+official", df[f"{m}_official"], np.nan))
        df[f"{m}_all"] = np.where((sc == "all") | (sc == "all+official") | ((sc == "official") & every), df[m], np.nan)
    # official-loader window count (n_scored is the every-window count in dual-scored rows)
    df["n_scored_off"] = np.where(sc == "all+official", df["n_scored_official"], np.where(sc == "official", df["n_scored"], np.nan))
    return df


def rows(df, **kw):
    kw.setdefault("variant", "")  # the unmodified models unless a variant is asked for
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m].sort_values("seed")


def score(df, kind, P=None, **kw):
    """One row per seed, with mse/mae (and n_scored) set to the chosen score.

    kind='official': training rows, the official loader's score.
    kind='all': every-window score; a batch-1 re-score row wins over a training row."""
    assert kind in ("official", "all")
    r = rows(df, **kw)
    if kw.get("model") == "PatchTST":  # one patch setting per cell (default P=16, S=8)
        P = 16 if P is None else P
        r = r[(r.patch_len == P) & (r.stride == P // 2)]
    if kind == "official":
        r = r[(r["mode"] == "train") & r["mse_off"].notna()].copy()
        r["mse"], r["mae"], r["n_scored"] = r["mse_off"], r["mae_off"], r["n_scored_off"]
    else:
        r = r[r["mse_all"].notna()].copy()
        r["_bs1"] = (r["mode"] != "rescore_bs1").astype(int)
        r = r.sort_values(["seed", "_bs1"]).drop_duplicates("seed").drop(columns="_bs1")
        r["mse"], r["mae"] = r["mse_all"], r["mae_all"]
    assert not r["seed"].duplicated().any(), (kind, kw, list(r["run_id"]))
    return r.sort_values("seed")


def common_seeds(name, cells):
    """Seeds present in every cell of one table (cells: dict label -> rows)."""
    seeds = set.intersection(*(set(r["seed"]) for r in cells.values()))
    assert len(seeds) >= MIN_SEEDS, f"{name}: only seeds {sorted(seeds)} are in every cell"
    extra = {k: sorted(set(r["seed"]) - seeds) for k, r in cells.items() if set(r["seed"]) - seeds}
    if extra:
        print(f"{name}: using seeds {sorted(seeds)} (in every cell); not used: {extra}")
    return sorted(seeds)


def at(r, seeds):
    return r[r["seed"].isin(seeds)].sort_values("seed")


def patchtst_default(r):
    return r[(r["patch_len"] == 16) & (r["stride"] == 8)]


# ------------------------------------------------------------------ formatting
def num(x, d):
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$")


def ms(x, d=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f}$\\pm${x.std(ddof=1):.{d}f}"


def ci_str(m, lo, hi, d=4):
    return f"{num(m, d)} [{num(lo, d)}, {num(hi, d)}]"


def paired_ci(a, b):
    """mean(b - a) over matched seeds, t-interval."""
    d = np.asarray(b, float) - np.asarray(a, float)
    hw = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - hw, d.mean() + hw, stats.ttest_1samp(d, 0).pvalue


def welch_ci(a, b):
    """mean(b) - mean(a), Welch interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    q = stats.t.ppf(0.975, dof)
    m = b.mean() - a.mean()
    return m, m - q * se, m + q * se, stats.ttest_ind(b, a, equal_var=False).pvalue


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)
    print(f"wrote {OUT.name}/{name}.tex")


# ------------------------------------------------------------------ tables
def table_repro(df, P):
    cfg = (("PatchTST", 336), ("DLinear", 336), ("iTransformer", 96), ("Autoformer", 96))
    cells = {}
    for model, L in cfg:
        for H in H_ALL:
            r = score(df, "official", model=model, dataset="ETTh1", L=L, H=H)
            cells[(model, H)] = patchtst_default(r) if model == "PatchTST" else r
    seeds = common_seeds("repro_ETTh1", cells)
    out = []
    for model, L in cfg:
        for H in H_ALL:
            r = at(cells[(model, H)], seeds)
            p = P[(P.model == model) & (P.dataset == "ETTh1") & (P.H == H)].iloc[0]
            m = r["mse"].mean()
            hw = stats.t.ppf(0.975, len(r) - 1) * r["mse"].std(ddof=1) / np.sqrt(len(r))
            rel = 100 * (m - p.mse) / p.mse
            ok2 = r"\cmark" if abs(rel) <= 2 else r"\xmark"
            inci = r"\cmark" if m - hw <= p.mse <= m + hw else r"\xmark"
            first = f"\\multirow{{4}}{{*}}{{{model} ($L{{=}}{L}$)}}" if H == 96 else ""
            out.append(f"{first} & {H} & {p.mse:.3f} & {p.mae:.3f} & {ms(r['mse'])} & {ms(r['mae'])} & "
                       f"{num(rel, 1)}\\% & {ok2} & {inci} \\\\")
        out.append(r"\midrule")
    out = out[:-1]
    write("repro_ETTh1", r"""\begin{tabular}{@{}llrrllrcc@{}}
\toprule
 & & \multicolumn{2}{c}{Published} & \multicolumn{2}{c}{Ours (official code, """ + f"{len(seeds)} seeds" + r""")} & & & Pub.\ in \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Model & $H$ & MSE & MAE & MSE & MAE & $\Delta$MSE & $\le$2\% & 95\% CI \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}""")


def f1_pairs(df, H, seeds=None):
    """Official loader vs every window, same PatchTST checkpoints (L=336, P=16)."""
    off = patchtst_default(score(df, "official", model="PatchTST", dataset="ETTh1", L=336, H=H))
    al = patchtst_default(score(df, "all", model="PatchTST", dataset="ETTh1", L=336, H=H))
    if seeds is None:
        seeds = common_seeds(f"F1 H={H}", {"official": off, "all": al})
    off, al = at(off, seeds), at(al, seeds)
    assert list(off.seed) == list(al.seed) == list(seeds), H
    return off, al


def f1_seeds(df, extra=None):
    cells = {}
    for H in H_ALL:
        off = patchtst_default(score(df, "official", model="PatchTST", dataset="ETTh1", L=336, H=H))
        al = patchtst_default(score(df, "all", model="PatchTST", dataset="ETTh1", L=336, H=H))
        cells[("official", H)], cells[("all", H)] = off, al
    cells.update(extra or {})
    return common_seeds("F1", cells)


def table_f1(df):
    out = []
    seeds = f1_seeds(df)
    for H in H_ALL:
        off, al = f1_pairs(df, H, seeds)
        n = int(al["n_test_windows"].iloc[0])
        k = int(off["n_scored"].iloc[0])
        skip = n - k
        # MSE of the skipped windows alone: n*MSE_all = k*MSE_official + (n-k)*MSE_skipped
        sk = (n * al["mse"].values - k * off["mse"].values) / skip
        m, lo, hi, _ = paired_ci(off["mse"], al["mse"])
        pct = 100 * m / off["mse"].mean()
        sk_s = ms(sk) if skip >= 10 else "--"  # one window: not estimated (rounding dominates)
        out.append(f"{H} & {skip} ({100*skip/n:.2f}\\%) & {ms(off['mse'])} & {ms(al['mse'])} & {sk_s} & "
                   f"{ci_str(m, lo, hi)} & {num(pct, 2)}\\% \\\\")
    write("f1_ETTh1", r"""\begin{tabular}{@{}lrlllll@{}}
\toprule
 & Windows & \multicolumn{3}{c}{MSE} & & \\
\cmidrule(lr){3-5}
$H$ & skipped & official loader & all windows & skipped windows & $\Delta$ [95\% CI] & $\Delta$\% \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}""")


def table_gap(df, P):
    out = []
    it_cells = {("iTransformer", H): score(df, "official", model="iTransformer", dataset="ETTh1", L=96, H=H)
                for H in H_ALL}
    seeds = f1_seeds(df, it_cells)
    for H in H_ALL:
        it = at(it_cells[("iTransformer", H)], seeds)["mse"]
        off, al = f1_pairs(df, H, seeds)
        gpub = (P[(P.model == "iTransformer") & (P.dataset == "ETTh1") & (P.H == H)].mse.iloc[0]
                - P[(P.model == "PatchTST") & (P.dataset == "ETTh1") & (P.H == H)].mse.iloc[0])
        g0 = welch_ci(off["mse"], it)
        g1 = welch_ci(al["mse"], it)
        c = paired_ci(off["mse"], al["mse"])
        out.append(f"{H} & {num(gpub, 3)} & {ci_str(*g0[:3])} & {ci_str(*g1[:3])} & {ci_str(*c[:3])} \\\\")
    write("gap_ETTh1", r"""\begin{tabular}{@{}lllll@{}}
\toprule
$H$ & $G_{\text{pub}}$ & $G_0$ (reproduced) & $G_1$ (+F1) & $C_{\text{F1}} = G_0 - G_1$ \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}""")


def ablation_cells(df, kind="official", required=True):
    """PatchTST P and L sweeps. kind='official' is the midterm table (official loader);
    kind='all' scores every window and exists only once the dual-scored re-runs are logged."""
    rr = {}
    for H in (96, 720):
        for P_ in (8, 16, 32, 64):
            rr[("P", P_, H)] = score(df, kind, P=P_, model="PatchTST", dataset="ETTh1", L=336, H=H)
        for L in (96, 336, 720):
            rr[("L", L, H)] = patchtst_default(score(df, kind, model="PatchTST", dataset="ETTh1", L=L, H=H))
    if not required and min(len(r) for r in rr.values()) < MIN_SEEDS:
        return None
    seeds = common_seeds(f"ablation ({kind})", rr)
    return {k: at(r, seeds)["mse"].values for k, r in rr.items()}


def table_ablation(df, kind="official"):
    c = ablation_cells(df, kind, required=(kind == "official"))
    if c is None:
        print(f"ablation_ETTh1_all: not written (not every cell has every-window scores yet)")
        return
    out = [f"$P{{=}}{p}$, $S{{=}}{p//2}$ & {ms(c[('P', p, 96)])} & {ms(c[('P', p, 720)])} \\\\" for p in (8, 16, 32, 64)]
    out.append(r"\midrule")
    out += [f"$L{{=}}{L}$ & {ms(c[('L', L, 96)])} & {ms(c[('L', L, 720)])} \\\\" for L in (96, 336, 720)]
    write("ablation_ETTh1" + ("" if kind == "official" else "_all"), r"""\begin{tabular}{@{}lll@{}}
\toprule
PatchTST setting & $H{=}96$ & $H{=}720$ \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}""")


def table_rq2(df, results):
    eff, out = {}, []
    models = ("iTransformer", "PatchTST", "DLinear")
    cells = {(m, ds): score(df, "all", model=m, dataset=ds, L=96, H=96) for m in models for ds in ("ONZ_real", "ONZ_shift")}
    seeds = common_seeds("onzonal_rq2", cells)
    for model in models:  # every window scored (official loader for iTransformer and DLinear)
        real, shift = at(cells[(model, "ONZ_real")], seeds), at(cells[(model, "ONZ_shift")], seeds)
        assert list(real.seed) == list(shift.seed) == list(seeds), model
        m, lo, hi, _ = paired_ci(real["mse"], shift["mse"])
        eff[model] = shift["mse"].values - real["mse"].values
        out.append(f"{model} & {ms(real['mse'], 4)} & {ms(shift['mse'], 4)} & {ci_str(m, lo, hi)} \\\\")
    sn = pd.read_csv(Path(results) / "onzonal_seasonal_naive.csv")
    out.append(f"Seasonal naive (1 week) & {float(sn[sn.H == 96].mse.iloc[0]):.4f} & -- & -- \\\\")
    d_pt = welch_ci(eff["PatchTST"], eff["iTransformer"])
    d_dl = welch_ci(eff["DLinear"], eff["iTransformer"])
    ctl = welch_ci(eff["DLinear"], eff["PatchTST"])
    write("onzonal_rq2", r"""\begin{tabular}{@{}llll@{}}
\toprule
Model & Real zones & Shifted zones & Shift $-$ real [95\% CI] \\
\midrule
""" + "\n".join(out) + "\n" + r"""\midrule
\multicolumn{3}{@{}l}{$D$ with PatchTST as control: $\Delta_{\text{iTrans}} - \Delta_{\text{PatchTST}}$ (Welch)} & """ + ci_str(*d_pt[:3]) + r""" \\
\multicolumn{3}{@{}l}{$D$ with DLinear as control: $\Delta_{\text{iTrans}} - \Delta_{\text{DLinear}}$ (Welch)} & """ + ci_str(*d_dl[:3]) + r""" \\
\multicolumn{3}{@{}l}{Controls compared: $\Delta_{\text{PatchTST}} - \Delta_{\text{DLinear}}$ (Welch)} & """ + ci_str(*ctl[:3]) + r""" \\
\bottomrule
\end{tabular}""")
    return dict(D_PatchTST=d_pt, D_DLinear=d_dl, controls=ctl)


# ------------------------------------------------------------------ text numbers
def text_numbers(df, P, results, rq2):
    """Every number quoted in the report's prose, printed so it can be checked."""
    print("\n--- numbers quoted in the text ---")
    it_l96 = {H: rows(df, model="iTransformer", dataset="ETTh1", L=96, H=H, mode="train")["mse"].mean() for H in H_ALL}
    c = ablation_cells(df)
    for H in (96, 720):
        print(f"ETTh1 H={H}: PatchTST L=96 {c[('L', 96, H)].mean():.4f} (official loader) vs iTransformer L=96 {it_l96[H]:.4f}; "
              f"PatchTST L=336 {c[('L', 336, H)].mean():.4f}")
    for H in (96, 336, 720):
        off, al = f1_pairs(df, H)
        print(f"F1 H={H}: delta {al['mse'].mean()-off['mse'].mean():+.4f} vs PatchTST seed std {off['mse'].std(ddof=1):.4f}")
    for k, v in rq2.items():
        print(f"{k}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] Welch p={v[3]:.4f}")
    real = {m: score(df, "all", model=m, dataset="ONZ_real", L=96, H=96)["mse"].mean()
            for m in ("iTransformer", "PatchTST", "DLinear")}
    sn = pd.read_csv(Path(results) / "onzonal_seasonal_naive.csv")
    sn96 = float(sn[sn.H == 96].mse.iloc[0])
    print("ON-Zonal gain over seasonal naive:", {m: f"{100*(1-v/sn96):.1f}%" for m, v in real.items()})
    print(f"D as % of iTransformer real MSE: {100*rq2['D_PatchTST'][0]/real['iTransformer']:.2f}% / "
          f"{100*rq2['D_DLinear'][0]/real['iTransformer']:.2f}%")
    onz = df[df.dataset.str.startswith("ONZ") & (df["variant"] == "")]
    for m in ("PatchTST", "iTransformer", "DLinear"):
        r = onz[(onz.model == m) & (onz["mode"] == "train")]
        print(f"ON-Zonal {m}: early-stopped in {int(r.early_stopped.sum())} of {len(r)} runs; epochs {sorted(r.epochs_run.unique())}")
    print(f"GPU-hours: {df.seconds.sum()/3600:.1f} over {len(df)} distinct runs; devices {sorted(df.device.unique())}")


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else "results"
    P = pd.read_csv(HERE / "published.csv")
    df = load(results)
    table_repro(df, P)
    table_f1(df)
    table_gap(df, P)
    table_ablation(df)
    rq2 = table_rq2(df, results)
    table_ablation(df, "all")  # written only when every cell has every-window scores
    text_numbers(df, P, results, rq2)


if __name__ == "__main__":
    main()
