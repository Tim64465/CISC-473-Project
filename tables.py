"""Build the report's result tables from logged runs.

python tables.py [results_dir ...]

Reads every runs*.csv under the given folders (default: ./results), plus
published.csv, and writes LaTeX tables to ./tables/ for the report to \\input.
Cells with no runs yet print as \\pend (defined in the report), so the report
compiles before, during and after the experiments. Nothing here is typed by hand:
every number comes from a run log or from published.csv.

Statistics follow the plan in the report: per-cell mean +- sample std over seeds;
paired differences over shared seeds with a t-based 95% interval.
"""
import glob
import math
import os
import sys
from pathlib import Path

import pandas as pd

KIT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("P24_TABLES", KIT / "tables"))
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
H_ALL = [96, 192, 336, 720]
PEND = r"\pend"


def load_runs(dirs):
    files = sorted({f for d in dirs for f in glob.glob(str(Path(d) / "**" / "runs*.csv"), recursive=True)})
    if not files:
        cols = ["run_id", "model", "dataset", "mode", "L", "H", "seed", "patch_len", "stride",
                "n_test_windows", "n_scored", "mse", "mae"]
        return pd.DataFrame({c: pd.Series(dtype="float64" if c in ("mse", "mae", "patch_len", "stride")
                                          else ("int64" if c in ("L", "H", "seed") else "object"))
                             for c in cols})
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["mse"].notna()].drop_duplicates("run_id", keep="first")
    for c in ("L", "H", "seed"):
        df[c] = df[c].astype(int)
    df["patch_len"] = pd.to_numeric(df["patch_len"], errors="coerce")
    df["stride"] = pd.to_numeric(df["stride"], errors="coerce")
    print(f"loaded {len(df)} runs from {len(files)} file(s)")
    return df


def sel(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m]


def by_seed(rows, col="mse"):
    return {int(s): float(v) for s, v in zip(rows["seed"], rows[col])}


def ms(vals):
    """'mean +- std (n)' string, or \\pend."""
    v = list(vals)
    if not v:
        return PEND
    m = sum(v) / len(v)
    if len(v) == 1:
        return f"{m:.3f}" + r"$^{\dagger}$"
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return f"{m:.3f}$\\pm${sd:.3f}"


def paired(a: dict, b: dict):
    """b - a over shared seeds: (mean, half-width of 95% CI or None, n)."""
    s = sorted(set(a) & set(b))
    if not s:
        return None
    d = [b[k] - a[k] for k in s]
    n = len(d)
    m = sum(d) / n
    if n < 2:
        return m, None, n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return m, T975.get(n, 1.96) * sd / math.sqrt(n), n


def welch(a, b):
    """mean(b) - mean(a) for two independent samples, Welch 95% interval: (mean, half-width or None, n)."""
    a, b = list(a), list(b)
    if not a or not b:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    if len(a) < 2 or len(b) < 2:
        return mb - ma, None, min(len(a), len(b))
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se2 = va / len(a) + vb / len(b)
    if se2 == 0:
        return mb - ma, 0.0, min(len(a), len(b))
    df = se2 ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    try:
        from scipy.stats import t
        q = float(t.ppf(0.975, df))
    except Exception:  # conservative fallback: smallest integer df
        q = T975.get(max(2, int(math.floor(df)) + 1), 1.96)
    return mb - ma, q * math.sqrt(se2), min(len(a), len(b))


def fmt_paired(p, scale=1.0, digits=3, pct_of=None):
    if p is None:
        return PEND
    m, hw, n = p
    s = f"{m*scale:+.{digits}f}"
    if hw is not None:
        s += f" [{(m-hw)*scale:+.{digits}f}, {(m+hw)*scale:+.{digits}f}]"
    else:
        s += r"$^{\dagger}$"
    return s


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)
    print(f"wrote tables/{name}.tex")


def pub(P, model, ds, H, col="mse"):
    r = P[(P.model == model) & (P.dataset == ds) & (P.H == H)]
    return float(r[col].iloc[0]) if len(r) else None


# ------------------------------------------------------------------ tables
def table_published_gap(P):
    """Motivation: the PatchTST-iTransformer gap in published numbers flips with protocol."""
    lines = []
    for ds in ("ETTh1", "ETTm1"):
        for H in H_ALL:
            pt, it = pub(P, "PatchTST", ds, H), pub(P, "iTransformer", ds, H)
            pt96 = pub(P, "PatchTST_L96", ds, H)
            g1, g2 = it - pt, it - pt96
            lines.append(f"{ds} & {H} & {pt:.3f} & {it:.3f} & {g1:+.3f} & {pt96:.3f} & {it:.3f} & {g2:+.3f} \\\\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    body = (r"""\begin{tabular}{@{}llrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{Each model's own paper} & \multicolumn{3}{c}{iTransformer paper, both $L{=}96$} \\
\cmidrule(lr){3-5}\cmidrule(l){6-8}
Data & $H$ & PatchTST & iTrans. & Gap & PatchTST & iTrans. & Gap \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}""")
    write("published_gap", body)


def table_repro(df, P, ds="ETTh1"):
    rows = []
    for model, L in (("PatchTST", 336), ("DLinear", 336), ("iTransformer", 96), ("Autoformer", 96)):
        for H in H_ALL:
            r = sel(df, model=model, dataset=ds, L=L, H=H, mode="train")
            if model == "PatchTST" and not r.empty:
                r = r[(r.patch_len == 16) & (r.stride == 8)]
            pm, pa = pub(P, model, ds, H, "mse"), pub(P, model, ds, H, "mae")
            if r.empty:
                ours_mse, ours_mae, rel, ok, n = PEND, PEND, PEND, PEND, 0
            else:
                m = r["mse"].mean()
                ours_mse, ours_mae = ms(r["mse"]), ms(r["mae"])
                rel = f"{100*(m-pm)/pm:+.1f}\\%"
                ok = r"\cmark" if abs(m - pm) / pm <= 0.05 else r"\xmark"
                n = len(r)
            first = f"\\multirow{{4}}{{*}}{{{model} ($L{{=}}{L}$)}}" if H == 96 else ""
            rows.append(f"{first} & {H} & {pm:.3f} & {pa:.3f} & {ours_mse} & {ours_mae} & {rel} & {ok} \\\\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    body = (r"""\begin{tabular}{@{}llrrllrc@{}}
\toprule
 & & \multicolumn{2}{c}{Published} & \multicolumn{2}{c}{Ours (official code)} & & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Model & $H$ & MSE & MAE & MSE & MAE & $\Delta$MSE & $\le$5\% \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")
    write(f"repro_{ds}", body)


def table_f1(df, ds="ETTh1"):
    rows = []
    for H in H_ALL:
        a = sel(df, model="PatchTST", dataset=ds, L=336, H=H, mode="train")
        b = sel(df, model="PatchTST", dataset=ds, L=336, H=H, mode="rescore_bs1")
        if not a.empty:
            a = a[(a.patch_len == 16) & (a.stride == 8)]
        if not b.empty:
            b = b[(b.patch_len == 16) & (b.stride == 8)]
        n_test = int(b["n_test_windows"].iloc[0]) if len(b) else (int(a["n_test_windows"].iloc[0]) if len(a) else None)
        n_sc = int(a["n_scored"].iloc[0]) if len(a) else None
        drop = f"{n_test - n_sc} ({100*(n_test-n_sc)/n_test:.1f}\\%)" if (n_test and n_sc is not None) else PEND
        p = paired(by_seed(a), by_seed(b)) if (len(a) and len(b)) else None
        pct = PEND
        if p is not None:
            pct = f"{100*p[0]/a['mse'].mean():+.2f}\\%"
        rows.append(f"{H} & {drop} & {ms(a['mse']) if len(a) else PEND} & {ms(b['mse']) if len(b) else PEND} & "
                    f"{fmt_paired(p)} & {pct} \\\\")
    body = (r"""\begin{tabular}{@{}lrllll@{}}
\toprule
$H$ & Windows dropped & MSE, official loader & MSE, all windows & $\Delta$ [95\% CI] & $\Delta$\% \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")
    write(f"f1_{ds}", body)


def table_gap(df, P, ds="ETTh1"):
    """Preliminary decomposition: gap = MSE(iTransformer) - MSE(PatchTST), positive = PatchTST better."""
    rows = []
    for H in H_ALL:
        g_pub = pub(P, "iTransformer", ds, H) - pub(P, "PatchTST", ds, H)
        it = by_seed(sel(df, model="iTransformer", dataset=ds, L=96, H=H, mode="train"))
        pt_rows = sel(df, model="PatchTST", dataset=ds, L=336, H=H, mode="train")
        pt1_rows = sel(df, model="PatchTST", dataset=ds, L=336, H=H, mode="rescore_bs1")
        pt = by_seed(pt_rows[(pt_rows.patch_len == 16)]) if len(pt_rows) else {}
        pt1 = by_seed(pt1_rows[(pt1_rows.patch_len == 16)]) if len(pt1_rows) else {}
        # Different architectures: unpaired Welch intervals. Same checkpoint re-scored: paired.
        g_rep = welch(pt.values(), it.values()) if (it and pt) else None    # it - pt
        g_f1 = welch(pt1.values(), it.values()) if (it and pt1) else None   # it - pt(bs1)
        c_f1 = paired(pt, pt1) if (pt and pt1) else None                    # pt(bs1) - pt = shrinkage of the gap
        rows.append(f"{H} & {g_pub:+.3f} & {fmt_paired(g_rep)} & {fmt_paired(g_f1)} & {fmt_paired(c_f1)} \\\\")
    body = (r"""\begin{tabular}{@{}lllll@{}}
\toprule
$H$ & $G_{\text{pub}}$ & $G_0$ (reproduced) & $G_1$ (+F1) & $C_{\text{F1}} = G_0 - G_1$ \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")
    write(f"gap_{ds}", body)


def table_ablation(df):
    def cell(**kw):
        r = sel(df, model="PatchTST", dataset="ETTh1", mode="train", **{k: v for k, v in kw.items() if k in ("L", "H")})
        if r.empty:
            return PEND
        r = r[(r.patch_len == kw["P"]) & (r.stride == kw["P"] // 2)]
        return ms(r["mse"]) if len(r) else PEND
    rows = [f"$P{{=}}{P}$, $S{{=}}{P//2}$ & {cell(P=P, L=336, H=96)} & {cell(P=P, L=336, H=720)} \\\\"
            for P in (8, 16, 32, 64)]
    rows.append(r"\midrule")
    rows += [f"$L{{=}}{L}$ & {cell(P=16, L=L, H=96)} & {cell(P=16, L=L, H=720)} \\\\" for L in (96, 336, 720)]
    body = (r"""\begin{tabular}{@{}lll@{}}
\toprule
PatchTST setting & $H{=}96$ & $H{=}720$ \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")
    write("ablation_ETTh1", body)


def table_onzonal(df, results_dirs):
    rows, d = [], {}
    for model, mode in (("iTransformer", "train"), ("PatchTST", "rescore_bs1"), ("DLinear", "train")):
        real = by_seed(sel(df, model=model, dataset="ONZ_real", L=96, H=96, mode=mode))
        shift = by_seed(sel(df, model=model, dataset="ONZ_shift", L=96, H=96, mode=mode))
        d[model] = (real, shift)
        p = paired(real, shift) if (real and shift) else None
        rows.append(f"{model} & {ms(real.values())} & {ms(shift.values())} & {fmt_paired(p)} \\\\")
    sn = PEND
    for rd in results_dirs:
        f = Path(rd) / "onzonal_seasonal_naive.csv"
        if f.exists():
            t = pd.read_csv(f)
            sn = f"{float(t[t.H == 96]['mse'].iloc[0]):.3f}"
    rows.append(f"Seasonal naive (1 week) & {sn} & -- & -- \\\\")
    # Difference-in-differences: extra harm of the shift to iTransformer beyond PatchTST.
    did = PEND
    (ir, ish), (pr, psh) = d["iTransformer"], d["PatchTST"]
    # Each model's shift effect is paired within the model (same seed => same init and batch order);
    # the two models' effects are then compared with an unpaired Welch interval.
    d_it = [ish[k] - ir[k] for k in sorted(set(ir) & set(ish))]
    d_pt = [psh[k] - pr[k] for k in sorted(set(pr) & set(psh))]
    if d_it and d_pt:
        did = fmt_paired(welch(d_pt, d_it))
    body = (r"""\begin{tabular}{@{}llll@{}}
\toprule
Model & Real zones & Shifted zones & Shift $-$ real [95\% CI] \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\midrule
\multicolumn{3}{@{}l}{Cross-zone effect $D = \Delta_{\text{iTrans}} - \Delta_{\text{PatchTST}}$} & """ + did + r""" \\
\bottomrule
\end{tabular}""")
    write("onzonal_rq2", body)


def summary(df):
    if df.empty:
        print("No runs logged yet.")
        return
    g = df.groupby(["dataset", "model", "mode", "L", "H"])["mse"].agg(["count", "mean", "std"])
    print(g.round(4).to_string())


def main(results_dirs=None):
    dirs = [str(x) for x in (results_dirs if isinstance(results_dirs, (list, tuple)) else
                             [results_dirs or (KIT / "results")])]
    P = pd.read_csv(KIT / "published.csv")
    df = load_runs(dirs)
    table_published_gap(P)
    for ds in ("ETTh1", "ETTm1"):
        table_repro(df, P, ds)
        table_f1(df, ds)
        table_gap(df, P, ds)
    table_ablation(df)
    table_onzonal(df, dirs)
    summary(df)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
