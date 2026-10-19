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
SEEDS = (2021, 2022, 2023)


# ------------------------------------------------------------------ loading
def load(results):
    files = sorted(glob.glob(str(Path(results) / "**" / "runs*.csv"), recursive=True))
    assert files, f"no runs*.csv under {results}"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["mse"].notna()]
    # The same run_id can appear in two jobs (Jobs A and C share six PatchTST runs).
    # They must agree exactly; then keep one copy.
    dup = df[df.duplicated("run_id", keep=False)]
    for rid, g in dup.groupby("run_id"):
        assert g["mse"].nunique() == 1 and g["mae"].nunique() == 1, f"{rid}: duplicate runs disagree"
    n_dup = dup["run_id"].nunique()
    df = df.drop_duplicates("run_id")
    print(f"{len(df)} distinct runs from {len(files)} file(s); {n_dup} run(s) logged twice with identical MSE/MAE")
    return df


def rows(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m].sort_values("seed")


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
    out = []
    for model, L in cfg:
        for H in H_ALL:
            r = rows(df, model=model, dataset="ETTh1", L=L, H=H, mode="train")
            if model == "PatchTST":
                r = patchtst_default(r)
            assert len(r) == 3, (model, H, len(r))
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
 & & \multicolumn{2}{c}{Published} & \multicolumn{2}{c}{Ours (official code, 3 seeds)} & & & Pub.\ in \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Model & $H$ & MSE & MAE & MSE & MAE & $\Delta$MSE & $\le$2\% & 95\% CI \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}""")


def f1_pairs(df, H):
    off = patchtst_default(rows(df, model="PatchTST", dataset="ETTh1", L=336, H=H, mode="train"))
    al = rows(df, model="PatchTST", dataset="ETTh1", L=336, H=H, mode="rescore_bs1")
    assert list(off.seed) == list(al.seed) == list(SEEDS), H
    return off, al


def table_f1(df):
    out = []
    for H in H_ALL:
        off, al = f1_pairs(df, H)
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
    for H in H_ALL:
        it = rows(df, model="iTransformer", dataset="ETTh1", L=96, H=H, mode="train")["mse"]
        off, al = f1_pairs(df, H)
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


def ablation_cells(df):
    cells = {}
    for H in (96, 720):
        for P_ in (8, 16, 32, 64):
            r = rows(df, model="PatchTST", dataset="ETTh1", L=336, H=H, mode="train")
            cells[("P", P_, H)] = r[(r.patch_len == P_) & (r.stride == P_ // 2)]["mse"].values
        for L in (96, 336, 720):
            r = patchtst_default(rows(df, model="PatchTST", dataset="ETTh1", L=L, H=H, mode="train"))
            cells[("L", L, H)] = r["mse"].values
    for k, v in cells.items():
        assert len(v) == 3, (k, len(v))
    return cells


def table_ablation(df):
    c = ablation_cells(df)
    out = [f"$P{{=}}{p}$, $S{{=}}{p//2}$ & {ms(c[('P', p, 96)])} & {ms(c[('P', p, 720)])} \\\\" for p in (8, 16, 32, 64)]
    out.append(r"\midrule")
    out += [f"$L{{=}}{L}$ & {ms(c[('L', L, 96)])} & {ms(c[('L', L, 720)])} \\\\" for L in (96, 336, 720)]
    write("ablation_ETTh1", r"""\begin{tabular}{@{}lll@{}}
\toprule
PatchTST setting & $H{=}96$ & $H{=}720$ \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}""")


def table_rq2(df, results):
    eff, out = {}, []
    for model, mode in (("iTransformer", "train"), ("PatchTST", "rescore_bs1"), ("DLinear", "train")):
        real = rows(df, model=model, dataset="ONZ_real", L=96, H=96, mode=mode)
        shift = rows(df, model=model, dataset="ONZ_shift", L=96, H=96, mode=mode)
        assert list(real.seed) == list(shift.seed) == list(SEEDS), model
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
    real = {m: rows(df, model=m, dataset="ONZ_real", L=96, H=96, mode=mo)["mse"].mean()
            for m, mo in (("iTransformer", "train"), ("PatchTST", "rescore_bs1"), ("DLinear", "train"))}
    sn = pd.read_csv(Path(results) / "onzonal_seasonal_naive.csv")
    sn96 = float(sn[sn.H == 96].mse.iloc[0])
    print("ON-Zonal gain over seasonal naive:", {m: f"{100*(1-v/sn96):.1f}%" for m, v in real.items()})
    print(f"D as % of iTransformer real MSE: {100*rq2['D_PatchTST'][0]/real['iTransformer']:.2f}% / "
          f"{100*rq2['D_DLinear'][0]/real['iTransformer']:.2f}%")
    onz = df[df.dataset.str.startswith("ONZ")]
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
    text_numbers(df, P, results, rq2)


if __name__ == "__main__":
    main()
