"""Final-report tables (patch 0006) and results/report_numbers.txt, from the run logs.

    python report_tables.py results     # the midterm tables, then everything here (report_tables.main)
    python final_tables.py results      # only the tables here

Writes tables/<name>.tex (booktabs tabulars, \\input by the report) and
results/report_numbers.txt (every number the report quotes, one labelled line each).

Statistics are those of report_tables.py, whose loader and helpers are used unchanged:
  * cells: mean +- sample std (ddof=1) over the seeds present in every cell of the row or
    comparison (common_seeds); the seeds are printed in each table and in report_numbers.txt;
  * paired 95% t-interval within one model (same seed; the runs differ in one setting:
    scoring, look-back, learning rate, calendar terms or tokens, real vs shifted data);
  * Welch 95% t-interval between models (and for differences of within-model effects,
    e.g. D = Delta_iTransformer - Delta_control, as in onzonal_rq2.tex).
Each table file starts with a suggested caption (LaTeX comment lines) that names its
scoring, seeds and tests; the report copies it.

Scoring: official loader for reproduction and published comparisons (main_ETTh1,
repro_ETTm1, pi_0 of RQ1); every test window everywhere else (stated per table).
Nothing is typed by hand: every number comes from results/ and published.csv.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_tables import (H_ALL, at, ci_str, common_seeds, load, num, paired_ci, patchtst_default,  # noqa: E402
                           score, welch_ci, write)

HERE = Path(__file__).resolve().parent
CAL_FULL = "cal-hour-dow-hol"
CAL_ETTH1 = "cal-hour-dow"
CAL_VARIANTS = (("full", CAL_FULL, (1, 1, 1)), ("$-$hour", "cal-dow-hol", (0, 1, 1)),
                ("$-$dow", "cal-hour-hol", (1, 0, 1)), ("$-$holiday", "cal-hour-dow", (1, 1, 0)))
F3_LRS = (3e-5, 1e-4, 3e-4)
F3_DEFAULT = 1e-4
ONZ_H96 = (("DLinear", ""), ("PatchTST", ""), ("iTransformer", ""), ("PatchTST", CAL_FULL))


# ------------------------------------------------------------------ formatting helpers
def seeds_txt(seeds, tex=False):
    s = sorted(int(x) for x in seeds)
    if len(s) > 1 and s == list(range(s[0], s[-1] + 1)):
        return f"{s[0]}{'--' if tex else '-'}{s[-1]}"
    return ",".join(map(str, s))


def seeds_field(*named):
    """report_numbers.txt seeds field: 'seeds 2021-2023 (n=3)' or one entry per named cell."""
    if len(named) == 1 and isinstance(named[0], (list, tuple, np.ndarray, pd.Series)):
        s = list(named[0])
        return f"seeds {seeds_txt(s)} (n={len(s)})"
    return "; ".join(f"{name}: seeds {seeds_txt(s)} (n={len(s)})" for name, s in named)


def sgn(x, d):
    return f"{x:+.{d}f}"


def ptxt(p):
    if not np.isfinite(p):
        return "p=nan"
    return "p<0.0001" if p < 0.0001 else f"p={p:.4f}"


def msd(x, d, bold=False):
    """mean$\\pm$std; the mean in bold if asked."""
    x = np.asarray(x, float)
    m = f"{x.mean():.{d}f}"
    return (f"\\textbf{{{m}}}" if bold else m) + f"$\\pm${x.std(ddof=1):.{d}f}"


def best(values, d):
    """Keys whose value, rounded to the d decimals shown, is the lowest (ties: all of them)."""
    r = {k: float(f"{v:.{d}f}") for k, v in values.items() if v is not None}
    lo = min(r.values())
    return {k for k, v in r.items() if v == lo}


def ci_tex(res, d=4):
    return ci_str(res[0], res[1], res[2], d)


def onesample_ci(d):
    """mean(d) with a t-interval: used for per-seed combinations of paired differences."""
    d = np.asarray(d, float)
    return paired_ci(np.zeros_like(d), d)


def caption(lines, size):
    head = [f"Fits the report's text width (6.5 in) at \\{size}. Suggested caption (written by final_tables.py):"]
    return "".join(f"% {l}\n" for l in head + lines)


def put(name, cap, body, size="small", sep=None):
    """Write tables/<name>.tex: caption comment, then the tabular (in a group with a narrower
    \\tabcolsep when sep is given, so the setting does not leak into the report)."""
    if sep:
        body = f"{{\\setlength{{\\tabcolsep}}{{{sep}}}%\n" + body.rstrip("\n") + "%\n}\n"
    write(name, caption(cap, size) + body)


# ------------------------------------------------------------------ report_numbers.txt
class Numbers:
    """Every number the report quotes: 'key | value | 95% CI | test or statistic | seeds'."""

    HEADER = ("# P24 report numbers. Written by report_tables.py (final_tables.py) from results/*.csv and\n"
              "# published.csv; do not edit by hand. Re-check: python tests/verify_report_numbers.py\n"
              "# Format: key | value | 95% CI | test or statistic | seeds\n"
              "# Statistics: mean +- sample std over seeds; paired t-interval within a model (same seeds);\n"
              "# Welch t-interval between models; MSE/MAE on z-scored data unless stated.\n")

    def __init__(self):
        self.lines, self.keys = [], set()

    def add(self, key, value, ci="--", what="--", seeds="--"):
        assert key not in self.keys, f"duplicate key {key}"
        assert "|" not in "".join((value, ci, what, seeds)), key
        self.keys.add(key)
        self.lines.append(" | ".join((key, value, ci, what, seeds)))

    def mean_sd(self, key, x, seeds, d=4):
        x = np.asarray(x, float)
        self.add(key, f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}", "--", "mean ± sample std", seeds_field(seeds))

    def stat(self, key, res, test, seeds, d=4):
        m, lo, hi, p = res
        self.add(key, sgn(m, d), f"[{sgn(lo, d)}, {sgn(hi, d)}]", f"{test}, {ptxt(p)}", seeds)

    def write(self, path):
        Path(path).write_text(self.HEADER + "\n".join(self.lines) + "\n")
        print(f"wrote {path} ({len(self.lines)} numbers)")


# ------------------------------------------------------------------ cells
def etth1_official(df, model, L, H, variant=""):
    r = score(df, "official", model=model, dataset="ETTh1", L=L, H=H, variant=variant)
    return patchtst_default(r) if model == "PatchTST" else r


def onz(df, model, ds, H, variant=""):
    return score(df, "all", model=model, dataset=ds, L=96, H=H, variant=variant)


def label(model, variant):
    return {"": model, CAL_FULL: "PatchTST+Cal", CAL_ETTH1: "PatchTST+Cal", "nots": "iTransformer, no tokens",
            "ci": "Control 2"}.get(variant, f"{model} {variant}")


# ------------------------------------------------------------------ (a) main_ETTh1
def table_main_etth1(df, P, N):
    pub = {m: P[(P.model == m) & (P.dataset == "ETTh1")].set_index("H") for m in ("PatchTST", "DLinear", "iTransformer")}
    cells = {}
    for H in H_ALL:
        cells[("PatchTST", H)] = etth1_official(df, "PatchTST", 336, H)
        cells[("Cal", H)] = etth1_official(df, "PatchTST", 336, H, CAL_ETTH1)
    seeds = common_seeds("main_ETTh1", cells)
    S = seeds_field(seeds)
    out = []
    for H in H_ALL:
        pt, cal = at(cells[("PatchTST", H)], seeds), at(cells[("Cal", H)], seeds)
        for k, metric in enumerate(("mse", "mae")):
            vals = {"pubPT": pub["PatchTST"].loc[H, metric], "pubDL": pub["DLinear"].loc[H, metric],
                    "pubIT": pub["iTransformer"].loc[H, metric], "PT": pt[metric].mean(), "Cal": cal[metric].mean()}
            b = best(vals, 3)
            pubs = [(f"\\textbf{{{vals[k_]:.3f}}}" if k_ in b else f"{vals[k_]:.3f}") for k_ in ("pubPT", "pubDL", "pubIT")]
            res = paired_ci(pt[metric], cal[metric])
            first = f"\\multirow{{2}}{{*}}{{{H}}}" if k == 0 else ""
            out.append(f"{first} & {metric.upper()} & {' & '.join(pubs)} & {msd(pt[metric], 3, 'PT' in b)} & "
                       f"{msd(cal[metric], 3, 'Cal' in b)} & {ci_tex(res)} \\\\")
            key = f"main_ETTh1.H{H}"
            N.mean_sd(f"{key}.PatchTST.{metric}", pt[metric], seeds)
            N.mean_sd(f"{key}.PatchTST+Cal.{metric}", cal[metric], seeds)
            N.stat(f"{key}.PatchTST+Cal_minus_PatchTST.{metric}", res, "paired t", S)
            N.add(f"{key}.PatchTST+Cal_minus_PatchTST.{metric}_pct", f"{100 * res[0] / pt[metric].mean():+.2f}%",
                  what="100 x paired mean difference / mean of our PatchTST", seeds=S)
            for name, x in (("PatchTST", pt), ("PatchTST+Cal", cal)):
                rel = 100 * (x[metric].mean() - vals["pubPT"]) / vals["pubPT"]
                N.add(f"{key}.{name}_vs_published_PatchTST42.{metric}_pct", f"{rel:+.1f}%",
                      what="100 x (our mean - published PatchTST/42) / published", seeds=S)
            names = {"pubPT": "published PatchTST/42", "pubDL": "published DLinear", "pubIT": "published iTransformer",
                     "PT": "our PatchTST", "Cal": "PatchTST+Cal"}
            N.add(f"{key}.best.{metric}", ", ".join(names[k_] for k_ in ("pubPT", "pubDL", "pubIT", "PT", "Cal") if k_ in b),
                  what="lowest mean at the 3 decimals shown (bold in the table)", seeds=S)
        if H != H_ALL[-1]:
            out.append(r"\midrule")
    cap = [f"ETTh1, official test loader (as published), mean $\\pm$ std over {len(seeds)} seeds "
           f"({seeds_txt(seeds, True)}); bold: lowest at the precision shown.",
           "Published: PatchTST/42 and DLinear from Nie et al. (2023, Table 3), iTransformer (iTrans.) from Liu et al. (2024, Table 10).",
           "Ours: the official code at pinned commits; PatchTST+Cal adds hour-of-day and day-of-week embeddings (Sec. 3).",
           "Last column: PatchTST+Cal $-$ our PatchTST, mean of the per-seed differences with a paired 95\\% t-interval."]
    put("main_ETTh1", cap, r"""\begin{tabular}{@{}llrrrlll@{}}
\toprule
 & & \multicolumn{3}{c}{Published} & \multicolumn{2}{c}{Ours (""" + f"{len(seeds)} seeds" + r""")} & Ours $-$ PatchTST \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(l){8-8}
$H$ & & PatchTST/42 & DLinear & iTrans. & PatchTST & PatchTST+Cal & $\Delta$ [95\% CI] \\
 & & $L{=}336$ & $L{=}336$ & $L{=}96$ & $L{=}336$ & (proposed) & \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}
""", sep="3.5pt")


# ------------------------------------------------------------------ (b) main_ONZ
def table_main_onz(df, results, N):
    sn = pd.read_csv(Path(results) / "onzonal_seasonal_naive.csv").set_index("H")
    cells = {(m, v, 96): onz(df, m, "ONZ_real", 96, v) for m, v in ONZ_H96}
    cells.update({(m, "", 720): onz(df, m, "ONZ_real", 720) for m in ("DLinear", "PatchTST", "iTransformer")})
    seeds = common_seeds("main_ONZ", cells)
    S = seeds_field(seeds)
    c = {k: at(r, seeds) for k, r in cells.items()}
    rows = [("Seasonal naive (1 week)", None, None)] + [(label(m, v), m, v) for m, v in ONZ_H96]
    bold = {}
    for H in (96, 720):
        for metric in ("mse", "mae"):
            vals = {"naive": float(sn.loc[H, metric])}
            vals.update({(m, v): c[(m, v, H)][metric].mean() for m, v in ONZ_H96 if (m, v, H) in c})
            bold[(H, metric)] = best(vals, 4)
    out = []
    for name, m, v in rows:
        cols = []
        for H in (96, 720):
            for metric in ("mse", "mae"):
                if m is None:
                    x = float(sn.loc[H, metric])
                    cols.append(f"\\textbf{{{x:.4f}}}" if "naive" in bold[(H, metric)] else f"{x:.4f}")
                elif (m, v, H) in c:
                    cols.append(msd(c[(m, v, H)][metric], 4, (m, v) in bold[(H, metric)]))
                else:
                    cols.append("--")
        out.append(f"{name}{' (proposed)' if v == CAL_FULL else ''} & " + " & ".join(cols) + r" \\")
    # numbers
    for H in (96, 720):
        for metric in ("mse", "mae"):
            N.add(f"main_ONZ.H{H}.seasonal_naive.{metric}", f"{float(sn.loc[H, metric]):.4f}",
                  what=f"seasonal naive (1 week), deterministic, {int(sn.loc[H, 'n_windows'])} windows")
            for m, v in ONZ_H96:
                if (m, v, H) in c:
                    N.mean_sd(f"main_ONZ.H{H}.{label(m, v)}.{metric}", c[(m, v, H)][metric], seeds)
        for m, v in ONZ_H96:
            if (m, v, H) in c:
                g = 100 * (1 - c[(m, v, H)]["mse"].mean() / float(sn.loc[H, "mse"]))
                N.add(f"main_ONZ.H{H}.{label(m, v)}.mse_gain_over_seasonal_naive_pct", f"{g:.1f}%",
                      what="100 x (1 - mean MSE / seasonal-naive MSE)", seeds=S)
        order = ["naive"] + [(m, v) for m, v in ONZ_H96]  # fixed order (a set's order is not)
        N.add(f"main_ONZ.H{H}.best.mse", ", ".join("seasonal naive" if k == "naive" else label(*k)
                                                   for k in order if k in bold[(H, "mse")]),
              what="lowest mean MSE (4 decimals)", seeds=S)
    diffs = []
    for name, a, b, test, Hs in (
            ("PatchTST+Cal $-$ PatchTST", ("PatchTST", ""), ("PatchTST", CAL_FULL), "paired", (96,)),
            ("PatchTST+Cal $-$ iTrans.", ("iTransformer", ""), ("PatchTST", CAL_FULL), "Welch", (96,)),
            ("PatchTST $-$ iTrans.", ("iTransformer", ""), ("PatchTST", ""), "Welch", (96, 720))):
        cols = []
        for H in (96, 720):
            if H not in Hs:
                cols.append(r"\multicolumn{2}{l}{--}")
                continue
            for metric in ("mse", "mae"):
                f = paired_ci if test == "paired" else welch_ci
                res = f(c[(*a, H)][metric], c[(*b, H)][metric])
                key = f"main_ONZ.H{H}.{label(*b)}_minus_{label(*a)}.{metric}"
                N.stat(key, res, f"{test} t", S)
                N.add(key + "_pct", f"{100 * res[0] / c[(*a, H)][metric].mean():+.1f}%",
                      what=f"100 x difference / mean of {label(*a)}", seeds=S)
                if metric == "mse":
                    cols.append(f"\\multicolumn{{2}}{{l}}{{{ci_tex(res)}}}")
        diffs.append(f"{name} ({test}) & " + " & ".join(cols) + r" \\")
    cap = [f"ON-Zonal (real zones), $L{{=}}96$, every test window, mean $\\pm$ std over {len(seeds)} seeds "
           f"({seeds_txt(seeds, True)}); bold: lowest at the precision shown. Seasonal naive repeats the last week (deterministic).",
           "Bottom rows: MSE differences with 95\\% t-intervals, paired within PatchTST (same seeds), Welch between models.",
           "PatchTST+Cal (hour, day of week, Ontario holiday) was run at $H{=}96$ only."]
    put("main_ONZ", cap, r"""\begin{tabular}{@{}lllll@{}}
\toprule
 & \multicolumn{2}{c}{$H{=}96$} & \multicolumn{2}{c}{$H{=}720$} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Model & MSE & MAE & MSE & MAE \\
\midrule
""" + "\n".join(out) + "\n" + r"""\midrule
\multicolumn{5}{@{}l}{MSE difference [95\% CI], seeds """ + seeds_txt(seeds, True) + r""":} \\
""" + "\n".join(diffs) + "\n" + r"""\bottomrule
\end{tabular}
""", sep="3pt")


# ------------------------------------------------------------------ (c) repro_ETTm1
def table_repro_ettm1(df, P, N):
    cfg = (("PatchTST", 336), ("iTransformer", 96), ("Autoformer", 96))
    cells = {}
    for model, L in cfg:
        for H in H_ALL:
            cells[(model, "off", H)] = score(df, "official", model=model, dataset="ETTm1", L=L, H=H)
            cells[(model, "all", H)] = score(df, "all", model=model, dataset="ETTm1", L=L, H=H)
    seeds = common_seeds("repro_ETTm1", cells)
    S = seeds_field(seeds)
    out, f1_runs = [], []
    for model, L in cfg:
        for H in H_ALL:
            r, a = at(cells[(model, "off", H)], seeds), at(cells[(model, "all", H)], seeds)
            assert list(r.seed) == list(a.seed) == list(seeds)
            p = P[(P.model == model) & (P.dataset == "ETTm1") & (P.H == H)].iloc[0]
            m = r["mse"].mean()
            ci = paired_ci(np.zeros(len(r)), r["mse"])  # t-interval of the mean (paired_ci against 0)
            rel = 100 * (m - p.mse) / p.mse
            ok2 = r"\cmark" if abs(rel) <= 2 else r"\xmark"
            inci = r"\cmark" if ci[1] <= p.mse <= ci[2] else r"\xmark"
            n, k = int(r["n_test_windows"].iloc[0]), int(r["n_scored"].iloc[0])
            if n > k:  # PatchTST: the official loader drops the last partial batch; F1 from the dual-scored rows
                f1 = paired_ci(r["mse"], a["mse"])
                f1_s = num(f1[0], 4)
                f1_ci = f"{{\\scriptsize[{num(f1[1], 4)}, {num(f1[2], 4)}]}}"
                f1_runs += list(a["mse"].values - r["mse"].values)
                N.stat(f"repro_ETTm1.{model}.H{H}.F1_all_minus_official.mse", f1, "paired t", S)
            else:
                assert np.allclose(a["mse"].values, r["mse"].values), (model, H)
                f1_s, f1_ci = "--", ""
            first = f"\\multirow{{{8 if n > k else 4}}}{{*}}{{{model}}}" if H == 96 else ""
            out.append(f"{first} & {H} & {p.mse:.3f} & {p.mae:.3f} & {msd(r['mse'], 3)} & {msd(r['mae'], 3)} & "
                       f"{num(rel, 1)}\\% & {ok2} & {inci} & {n - k} & {f1_s} \\\\" + ("\n & & & & & & & & & & " + f1_ci + " \\\\" if f1_ci else ""))
            N.mean_sd(f"repro_ETTm1.{model}.H{H}.mse_official", r["mse"], seeds)
            N.mean_sd(f"repro_ETTm1.{model}.H{H}.mae_official", r["mae"], seeds)
            N.add(f"repro_ETTm1.{model}.H{H}.mse_vs_published_pct", f"{rel:+.1f}%",
                  what=f"100 x (mean - published {p.mse:.3f}) / published ({p.source})", seeds=S)
            N.add(f"repro_ETTm1.{model}.H{H}.published_in_95ci", "yes" if inci == r"\cmark" else "no",
                  ci=f"[{ci[1]:.4f}, {ci[2]:.4f}]", what="t-interval of our mean MSE", seeds=S)
        out.append(r"\midrule")
    out = out[:-1]
    f1_runs = np.array(f1_runs)
    N.add("repro_ETTm1.PatchTST.F1_all_minus_official.mse_range_over_runs",
          f"{f1_runs.min():+.4f} to {f1_runs.max():+.4f}", what=f"min and max over the {len(f1_runs)} PatchTST runs", seeds=S)
    for model, _ in cfg:
        rels = [100 * (at(cells[(model, 'off', H)], seeds)['mse'].mean() - P[(P.model == model) & (P.dataset == 'ETTm1') & (P.H == H)].mse.iloc[0])
                / P[(P.model == model) & (P.dataset == 'ETTm1') & (P.H == H)].mse.iloc[0] for H in H_ALL]
        N.add(f"repro_ETTm1.{model}.max_abs_mse_vs_published_pct", f"{max(abs(x) for x in rels):.1f}%",
              what="largest absolute relative difference over H=96-720", seeds=S)
    cap = [f"ETTm1 reproduction: official test loader (as published), mean $\\pm$ std over {len(seeds)} seeds "
           f"({seeds_txt(seeds, True)}); PatchTST $L{{=}}336$, iTransformer and Autoformer $L{{=}}96$.",
           "Published: PatchTST/42 from Nie et al. (2023, Table 3), iTransformer and Autoformer from Liu et al. (2024, Table 10).",
           "$\\Delta$MSE: relative to published; checks: within 2\\%, and published value inside the 95\\% t-interval of our mean.",
           "Skipped: test windows the official loader does not score (iTransformer and Autoformer score every window, so F1 is 0).",
           "F1: every window $-$ official loader, same checkpoints (dual scoring), MSE with its paired 95\\% t-interval below."]
    put("repro_ETTm1", cap, r"""\begin{tabular}{@{}llrrllrccrl@{}}
\toprule
 & & \multicolumn{2}{c}{Published} & \multicolumn{2}{c}{Ours (""" + f"{len(seeds)} seeds" + r""")} & & & Pub.\ in & & F1, paired \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Model & $H$ & MSE & MAE & MSE & MAE & $\Delta$MSE & $\le$2\% & 95\% CI & Skipped & $\Delta$ [95\% CI] \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}
""", size="footnotesize", sep="3.5pt")


# ------------------------------------------------------------------ (d) RQ1 decomposition and F3
def f3_winners(results):
    """The committed selection (results/f3_selection.json), checked against the logged validation losses.

    Read from the raw files: a re-run copy (job Q) carries vali_loss_best where the first-logged
    copy of the same run_id (job A) does not, and load() keeps one copy per run_id."""
    sel = json.loads((Path(results) / "f3_selection.json").read_text())
    files = sorted(Path(results).glob("**/runs*.csv"))
    r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "vali_loss_best" not in r:
        r["vali_loss_best"] = np.nan
    r = r[(r.dataset == "ETTh1") & (r.L == 96) & (r.seed == 2021) & (r["mode"] == "train") & r["vali_loss_best"].notna()]
    r = r.assign(variant=r["variant"].fillna("").astype(str))
    win = {}
    for model in ("PatchTST", "iTransformer"):
        for H in H_ALL:
            logged = {}
            for lr in F3_LRS:
                var = "" if lr == F3_DEFAULT else f"lr{lr:g}"
                x = r[(r.model == model) & (r.H == H) & (r["variant"] == var)]
                if model == "PatchTST":  # default patch only (grid336 logs P=8/32/64 at L=96, H=336)
                    x = x[(x.patch_len == 16) & (x.stride == 8)]
                assert len(x) >= 1 and x["vali_loss_best"].nunique() == 1, (model, H, lr, list(x["vali_loss_best"]))
                logged[lr] = float(x["vali_loss_best"].iloc[0])
            j = sel[model][str(H)]
            assert not j["missing"], (model, H)
            assert {float(k): v for k, v in j["vali_loss_best"].items()} == logged, (model, H, j, logged)
            assert j["winner_lr"] == min(logged, key=logged.get), (model, H)
            win[(model, H)] = (j["winner_lr"], logged)
    return win


def rq1_cells(df, results):
    win = f3_winners(results)
    c = {}
    for H in H_ALL:
        c[("PT336", "off", H)] = etth1_official(df, "PatchTST", 336, H)
        c[("PT336", "all", H)] = patchtst_default(score(df, "all", model="PatchTST", dataset="ETTh1", L=336, H=H))
        c[("PT96", "off", H)] = etth1_official(df, "PatchTST", 96, H)
        c[("PT96", "all", H)] = patchtst_default(score(df, "all", model="PatchTST", dataset="ETTh1", L=96, H=H))
        c[("iT96", "off", H)] = etth1_official(df, "iTransformer", 96, H)
        c[("iT96", "all", H)] = score(df, "all", model="iTransformer", dataset="ETTh1", L=96, H=H)
        c[("iT336", "all", H)] = score(df, "all", model="iTransformer", dataset="ETTh1", L=336, H=H)
        for model, key in (("PatchTST", "PT96"), ("iTransformer", "iT96")):
            lr = win[(model, H)][0]
            var = "" if lr == F3_DEFAULT else f"lr{lr:g}"
            c[(key + "sel", "all", H)] = score(df, "all", model=model, dataset="ETTh1", L=96, H=H, variant=var)
    seeds = common_seeds("rq1_decomposition", c)
    return {k: at(r, seeds)["mse"].values for k, r in c.items()}, seeds, win


def rq1_stats(df, P, results):
    v, seeds, win = rq1_cells(df, results)
    st = {}
    for H in H_ALL:
        g = lambda k: v[(*k, H)] if len(k) == 2 else None  # noqa: E731
        pt336o, pt336a, pt96o, pt96a = g(("PT336", "off")), g(("PT336", "all")), g(("PT96", "off")), g(("PT96", "all"))
        it96, it336 = g(("iT96", "all")), g(("iT336", "all"))
        assert np.array_equal(it96, g(("iT96", "off")))  # iTransformer's official loader scores every window
        ptsel, itsel = g(("PT96sel", "all")), g(("iT96sel", "all"))
        gpub = (P[(P.model == "iTransformer") & (P.dataset == "ETTh1") & (P.H == H)].mse.iloc[0]
                - P[(P.model == "PatchTST") & (P.dataset == "ETTh1") & (P.H == H)].mse.iloc[0])
        s = dict(Gpub=gpub,
                 G0=welch_ci(pt336o, it96), G1=welch_ci(pt336a, it96), G2=welch_ci(pt96a, it96),
                 G3=welch_ci(ptsel, itsel), G2alt=welch_ci(pt336a, it336),
                 C1=paired_ci(pt336o, pt336a), C2=paired_ci(pt336a, pt96a),
                 C3=welch_ci(pt96a - ptsel, it96 - itsel), C2alt=paired_ci(it336, it96),
                 phi1=onesample_ci(0.5 * ((pt336a - pt336o) + (pt96a - pt96o))),
                 phi2=onesample_ci(0.5 * ((pt96o - pt336o) + (pt96a - pt336a))),
                 phi1alt=onesample_ci(pt336a - pt336o), phi2alt=onesample_ci(it96 - it336),
                 G11=welch_ci(pt96a, it96), G11alt=welch_ci(pt336a, it336),
                 lr=(win[("PatchTST", H)][0], win[("iTransformer", H)][0]))
        g0 = s["G0"][0]
        # identities (exact up to float rounding): chain and Shapley efficiency
        assert np.isclose(s["G0"][0] - s["G1"][0], s["C1"][0]) and np.isclose(s["G1"][0] - s["G2"][0], s["C2"][0])
        assert np.isclose(s["G2"][0] - s["G3"][0], s["C3"][0]) and np.isclose(s["G1"][0] - s["G2alt"][0], s["C2alt"][0])
        assert np.isclose(s["phi1"][0] + s["phi2"][0], g0 - s["G11"][0])
        assert np.isclose(s["phi1alt"][0] + s["phi2alt"][0], g0 - s["G11alt"][0])
        s["share"] = {k: 100 * s[k][0] / g0 for k in ("phi1", "phi2", "G11", "phi1alt", "phi2alt", "G11alt")}
        s["g0_ci_has_0"] = s["G0"][1] <= 0 <= s["G0"][2]
        st[H] = s
    return st, seeds


def table_rq1(df, P, results, N):
    st, seeds = rq1_stats(df, P, results)
    S = seeds_field(seeds)
    sp = r"\addlinespace[1pt]"

    dagger = "$^\\dagger$"

    def row(lbl, key, d=4, share=None):
        vals = []
        for H in H_ALL:
            v = num(st[H][key][0], d)
            if share:
                v += f" ({st[H]['share'][share]:.0f}".replace("-", "$-$") + "\\%" + (dagger if st[H]["g0_ci_has_0"] else "") + ")"
            vals.append(v)
        cis = [f"{{\\scriptsize[{num(st[H][key][1], d)}, {num(st[H][key][2], d)}]}}" for H in H_ALL]
        return f"{lbl} & " + " & ".join(vals) + " \\\\\n & " + " & ".join(cis) + " \\\\" + sp

    lines = [r"\multicolumn{5}{@{}l}{\textit{Gap $G$ [95\% CI, Welch]}} \\",
             "Published & " + " & ".join(num(st[H]["Gpub"], 3) for H in H_ALL) + r" \\" + sp,
             row(r"$\pi_0$: reproduction", "G0"),
             row(r"$\pi_1=\pi_0$ + F1", "G1"),
             row(r"$\pi_2=\pi_1$ + F2 ($L{=}96$)", "G2"),
             row(r"$\pi_3=\pi_2$ + F3", "G3"),
             row(r"$\pi_1$ + F2 ($L{=}336$; alt.)", "G2alt"),
             r"\midrule",
             r"\multicolumn{5}{@{}l}{\textit{$C_k=G(\pi_{k-1})-G(\pi_k)$ [95\% CI]}} \\",
             row(r"$C_1$: F1 (paired)", "C1"),
             row(r"$C_2$: F2, $L{=}96$ (paired)", "C2"),
             row(r"$C_3$: F3 (Welch)", "C3"),
             row(r"$C_2'$: F2, $L{=}336$ (paired)", "C2alt"),
             r"\midrule",
             r"\multicolumn{5}{@{}l}{\textit{Shapley, F1$\times$F2 (\% of $G_0$) [95\% CI]}} \\",
             row(r"$\phi_{\mathrm{F1}}$, $L{=}96$ (paired)", "phi1", share="phi1"),
             row(r"$\phi_{\mathrm{F2}}$, $L{=}96$ (paired)", "phi2", share="phi2"),
             row(r"$G$(F1,F2), $L{=}96$ (Welch)", "G11", share="G11"),
             row(r"$\phi_{\mathrm{F1}}$, $L{=}336$ (paired)", "phi1alt", share="phi1alt"),
             row(r"$\phi_{\mathrm{F2}}$, $L{=}336$ (paired)", "phi2alt", share="phi2alt"),
             row(r"$G$(F1,F2), $L{=}336$ (Welch)", "G11alt", share="G11alt"),
             r"\midrule",
             "F3 LR: PatchTST / iTrans. & " + " & ".join(
                 f"{st[H]['lr'][0]:.0e} / {st[H]['lr'][1]:.0e}".replace("e-0", "e-") for H in H_ALL) + r" \\",
             f"\\multicolumn{{5}}{{@{{}}l}}{{Seeds {seeds_txt(seeds, True)} in every cell ($n={len(seeds)}$).}} \\\\"]
    for H in H_ALL:
        s = st[H]
        N.add(f"rq1.H{H}.G_published", f"{s['Gpub']:+.3f}", what="published iTransformer (L=96) - PatchTST/42 (L=336)")
        for k, test, what in (("G0", "Welch t", "reproduction protocol: official loader, L=96 vs 336, default LR"),
                              ("G1", "Welch t", "+F1: every window"), ("G2", "Welch t", "+F1+F2: both L=96, every window"),
                              ("G3", "Welch t", "+F1+F2+F3: L=96, every window, LR selected on validation loss"),
                              ("G2alt", "Welch t", "+F1+F2 at L=336 (alternative): both L=336, every window"),
                              ("C1", "paired t (PatchTST, same checkpoints)", "C1 = G0 - G1"),
                              ("C2", "paired t (PatchTST L=96 vs 336)", "C2 = G1 - G2"),
                              ("C3", "Welch t on the per-seed LR effects of each model", "C3 = G2 - G3"),
                              ("C2alt", "paired t (iTransformer L=96 vs 336)", "C2' = G1 - G2alt"),
                              ("phi1", "paired t (per-seed Shapley value)", "Shapley value of F1, 2x2 at common L=96"),
                              ("phi2", "paired t (per-seed Shapley value)", "Shapley value of F2, 2x2 at common L=96"),
                              ("G11", "Welch t", "G with F1 and F2 at common L=96 (= G2)"),
                              ("phi1alt", "paired t (per-seed Shapley value)", "Shapley value of F1, 2x2 at common L=336 (= C1)"),
                              ("phi2alt", "paired t (per-seed Shapley value)", "Shapley value of F2, 2x2 at common L=336 (= C2')"),
                              ("G11alt", "Welch t", "G with F1 and F2 at common L=336 (= G2alt)")):
            N.stat(f"rq1.H{H}.{k}", s[k], f"{test}; {what}", S)
        for k in ("phi1", "phi2", "G11", "phi1alt", "phi2alt", "G11alt"):
            N.add(f"rq1.H{H}.{k}_share_of_G0", f"{s['share'][k]:.0f}%",
                  what="100 x value / G0" + ("; G0's 95% CI includes 0 (share unstable)" if s["g0_ci_has_0"] else ""), seeds=S)
        for k in ("C1", "C2", "C3"):
            N.add(f"rq1.H{H}.{k}_share_of_G0", f"{100 * s[k][0] / s['G0'][0]:.0f}%",
                  what="100 x C_k / G0" + ("; G0's 95% CI includes 0 (share unstable)" if s["g0_ci_has_0"] else ""), seeds=S)
    cap = [f"RQ1 on ETTh1: gap $G=\\overline{{\\mathrm{{MSE}}}}_{{\\mathrm{{iTransformer}}}}-\\overline{{\\mathrm{{MSE}}}}_{{\\mathrm{{PatchTST}}}}$ (test MSE) under each",
           f"protocol $\\pi_k$, seeds {seeds_txt(seeds, True)} in every cell; published: iTransformer ($L{{=}}96$) $-$ PatchTST/42 ($L{{=}}336$).",
           "$\\pi_0$: each model's official loader and look-back (iTransformer $L{=}96$, PatchTST $L{=}336$), default LR $10^{-4}$.",
           "F1: score every test window; F2: same look-back; F3: LR from $\\{3\\cdot10^{-5},10^{-4},3\\cdot10^{-4}\\}$ with the lowest",
           "validation loss (seed 2021, \\texttt{f3\\_selection.json}), then 3 seeds. $G$: Welch 95\\% t-interval; $C_1$, $C_2$, $C_2'$ and",
           "the Shapley values: paired t-intervals over seeds (one model changes); $C_3$: Welch interval on the per-seed LR effects.",
           "Shapley values on the 2$\\times$2 of F1 and F2 sum to $G_0-G(\\mathrm{F1,F2})$; shares are of $G_0$",
           "($^\\dagger$: $G_0$'s interval includes 0, so shares are unstable). At common $L{=}336$ F1 does not",
           "change iTransformer, so $\\phi_{\\mathrm{F1}}=C_1$ and $\\phi_{\\mathrm{F2}}=C_2'$."]
    put("rq1_decomposition", cap, r"""\begin{tabular}{@{}lllll@{}}
\toprule
 & $H{=}96$ & $H{=}192$ & $H{=}336$ & $H{=}720$ \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""", sep="4pt")
    return st, seeds


def table_f3(df, results, N):
    v, seeds, win = rq1_cells(df, results)
    S = seeds_field(seeds)
    out = []
    for model, key in (("PatchTST", "PT96"), ("iTransformer", "iT96")):
        for H in H_ALL:
            lr, logged = win[(model, H)]
            vl = [(f"\\textbf{{{logged[x]:.4f}}}" if x == lr else f"{logged[x]:.4f}") for x in F3_LRS]
            d, s = v[(key, "all", H)], v[(key + "sel", "all", H)]
            first = f"\\multirow{{4}}{{*}}{{{model}}}" if H == 96 else ""
            if lr == F3_DEFAULT:
                assert np.array_equal(d, s)
                tail = r"\multicolumn{2}{l}{= default}"
            else:
                res = paired_ci(d, s)
                tail = f"{msd(s, 4)} & {ci_tex(res)}"
                N.mean_sd(f"f3.{model}.H{H}.mse_selected_lr", s, seeds)
                N.stat(f"f3.{model}.H{H}.selected_minus_default.mse", res, f"paired t; LR {lr:g} vs {F3_DEFAULT:g}", S)
                N.add(f"f3.{model}.H{H}.selected_minus_default.mse_pct", f"{100 * res[0] / d.mean():+.1f}%",
                      what="100 x paired difference / mean at the default LR", seeds=S)
            out.append(f"{first} & {H} & {' & '.join(vl)} & {msd(d, 4)} & {tail} \\\\")
            N.add(f"f3.{model}.H{H}.selected_lr", f"{lr:g}", what="lowest validation loss over "
                  + ", ".join(f"{x:g}: {logged[x]:.7f}" for x in F3_LRS), seeds="seed 2021")
            N.mean_sd(f"f3.{model}.H{H}.mse_default_lr", d, seeds)
        if model == "PatchTST":
            out.append(r"\midrule")
    cap = ["F3 on ETTh1 ($L{=}96$, every test window): best validation loss of seed 2021 for each learning rate (bold: lowest,",
           "selected; \\texttt{results/f3\\_selection.json}), and test MSE at the default LR $10^{-4}$ and at the selected LR,",
           f"mean $\\pm$ std over seeds {seeds_txt(seeds, True)}; $\\Delta$: selected $-$ default, paired 95\\% t-interval."]
    put("f3_selection", cap, r"""\begin{tabular}{@{}llrrrlll@{}}
\toprule
 & & \multicolumn{3}{c}{Validation loss (seed 2021)} & \multicolumn{3}{c}{Test MSE (""" + f"{len(seeds)} seeds" + r""")} \\
\cmidrule(lr){3-5}\cmidrule(l){6-8}
Model & $H$ & $3{\cdot}10^{-5}$ & $10^{-4}$ & $3{\cdot}10^{-4}$ & LR $10^{-4}$ & selected LR & $\Delta$ [95\% CI] \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}
""", size="footnotesize", sep="4pt")


# ------------------------------------------------------------------ (e) RQ2 final
RQ2_MODELS = (("iTransformer", "", "iTransformer"), ("iTransformer", "nots", "iTransformer, no tokens"),
              ("PatchTST", "", "PatchTST"), ("DLinear", "", "DLinear"),
              ("iTransformer", "ci", "Control 2 (iTrans.-CI)"))


def rq2_stats(df, H):
    """Shift effects and D at one horizon; each comparison on the seeds common to its cells."""
    models = RQ2_MODELS if H == 96 else RQ2_MODELS[:1] + RQ2_MODELS[2:4]
    cell = {(m, v, ds): onz(df, m, ds, H, v) for m, v, _ in models for ds in ("ONZ_real", "ONZ_shift")}

    def seeds_of(*mv):
        return common_seeds(f"RQ2 H={H} {mv}", {(k, ds): cell[(*k, ds)] for k in mv for ds in ("ONZ_real", "ONZ_shift")})

    def eff(k, seeds):
        r, s = at(cell[(*k, "ONZ_real")], seeds), at(cell[(*k, "ONZ_shift")], seeds)
        assert list(r.seed) == list(s.seed) == list(seeds)
        return r["mse"].values, s["mse"].values

    st = {"rows": {}, "D": {}, "ctl": {}, "c2_minus_it": {}, "tokens": None}
    for m, v, name in models:
        sd = seeds_of((m, v))
        real, shift = eff((m, v), sd)
        st["rows"][(m, v)] = dict(name=name, seeds=sd, real=real, shift=shift, delta=paired_ci(real, shift))
    it, nots, pt, dl, ci = ("iTransformer", ""), ("iTransformer", "nots"), ("PatchTST", ""), ("DLinear", ""), ("iTransformer", "ci")
    pairs_D = [(it, pt), (it, dl)] + ([(it, ci), (nots, pt), (nots, dl)] if H == 96 else [])
    for a, b in pairs_D:  # D = Delta_a - Delta_b (Welch)
        sd = seeds_of(a, b)
        ea, eb = (lambda x: x[1] - x[0])(eff(a, sd)), (lambda x: x[1] - x[0])(eff(b, sd))
        st["D"][(a, b)] = (welch_ci(eb, ea), sd)
    pairs_ctl = [(pt, dl)] + ([(ci, pt), (ci, dl)] if H == 96 else [])
    for a, b in pairs_ctl:  # Delta_a - Delta_b (Welch)
        sd = seeds_of(a, b)
        ea, eb = (lambda x: x[1] - x[0])(eff(a, sd)), (lambda x: x[1] - x[0])(eff(b, sd))
        st["ctl"][(a, b)] = (welch_ci(eb, ea), sd)
    if H == 96:
        sd = seeds_of(ci, it)
        for j, ds in enumerate(("real", "shift")):
            x_ci, x_it = eff(ci, sd)[j], eff(it, sd)[j]
            st["c2_minus_it"][ds] = (welch_ci(x_it, x_ci), sd)
        sd = seeds_of(it, nots)
        st["tokens"] = (paired_ci(eff(nots, sd)[0], eff(it, sd)[0]), sd)  # with - without tokens, real zones
    return st


def table_rq2(df, N):
    nm = {("iTransformer", ""): "iTransformer", ("iTransformer", "nots"): "no tokens",
          ("PatchTST", ""): "PatchTST", ("DLinear", ""): "DLinear", ("iTransformer", "ci"): "C2"}
    key = {("iTransformer", ""): "iTransformer", ("iTransformer", "nots"): "iTransformer_no_tokens",
           ("PatchTST", ""): "PatchTST", ("DLinear", ""): "DLinear", ("iTransformer", "ci"): "Control2"}
    out = []
    stats = {}
    for H in (96, 720):
        st = rq2_stats(df, H)
        stats[H] = st
        out.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{$H{{=}}{H}$: MSE and $\\Delta=$ shifted $-$ real [95\\% CI, paired]}}}} \\\\")
        for k, r in st["rows"].items():
            out.append(f"{r['name']} & {seeds_txt(r['seeds'], True)} & {msd(r['real'], 4)} & {msd(r['shift'], 4)} & "
                       f"{ci_tex(r['delta'])} \\\\")
            p = f"rq2.H{H}.{key[k]}"
            N.mean_sd(p + ".real.mse", r["real"], r["seeds"])
            N.mean_sd(p + ".shift.mse", r["shift"], r["seeds"])
            N.stat(p + ".Delta_shift_minus_real", r["delta"], "paired t", seeds_field(r["seeds"]))
        out.append(r"\addlinespace")
        out.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{$H{{=}}{H}$: $D=\\Delta_{{\\mathrm{{iTrans}}}}-\\Delta_{{\\mathrm{{control}}}}$, "
                   f"controls compared [95\\% CI, Welch]}}}} \\\\")
        for (a, b), (res, sd) in st["D"].items():
            lbl = f"$D$ vs {'Control 2' if nm[b] == 'C2' else nm[b]}" + (" (no tokens)" if a == ("iTransformer", "nots") else "")
            out.append(f"{lbl} & {seeds_txt(sd, True)} & & & {ci_tex(res)} \\\\")
            N.stat(f"rq2.H{H}.D.{key[a]}_vs_{key[b]}", res, "Welch t on per-seed shift effects", seeds_field(sd))
        for (a, b), (res, sd) in st["ctl"].items():
            out.append(f"$\\Delta_{{\\mathrm{{{nm[a]}}}}}-\\Delta_{{\\mathrm{{{nm[b]}}}}}$ & {seeds_txt(sd, True)} & & & "
                       f"{ci_tex(res)} \\\\")
            N.stat(f"rq2.H{H}.controls.{key[a]}_minus_{key[b]}", res, "Welch t on per-seed shift effects", seeds_field(sd))
        if H == 96:
            for ds, lbl in (("real", "real"), ("shift", "shifted")):
                res, sd = st["c2_minus_it"][ds]
                out.append(f"Control 2 $-$ iTrans., {lbl} & {seeds_txt(sd, True)} & & & {ci_tex(res)} \\\\")
                N.stat(f"rq2.H96.Control2_minus_iTransformer.{ds}.mse", res, "Welch t", seeds_field(sd))
            res, sd = st["tokens"]
            out.append(f"Tokens, real zones (paired) & {seeds_txt(sd, True)} & & & {ci_tex(res)} \\\\")
            N.stat("rq2.H96.calendar_tokens.with_minus_without.real.mse", res, "paired t (iTransformer, same seeds)",
                   seeds_field(sd))
            out.append(r"\midrule")
    cap = ["RQ2 on ON-Zonal ($L{=}96$, every test window). Zone $k$ of the shifted copy is lagged by $k$ weeks. Each row and",
           "each comparison uses the seeds present in all its cells (column Seeds). $\\Delta$: shifted $-$ real, paired 95\\%",
           "t-interval over seeds; $D$ and the comparisons of controls: Welch 95\\% t-intervals on the per-seed shift effects;",
           "Control 2 $-$ iTransformer: Welch on MSE; calendar tokens (iTransformer with $-$ without): paired. Control 2 (C2) is",
           "iTransformer with each zone attended separately (channel-independent; calendar tokens kept)."]
    put("rq2_final", cap, r"""\begin{tabular}{@{}lllll@{}}
\toprule
Model & Seeds & Real zones & Shifted zones & $\Delta$ or $D$ [95\% CI] \\
\midrule
""" + "\n".join(out[:-1] if out[-1] == r"\midrule" else out) + "\n" + r"""\bottomrule
\end{tabular}
""", sep="3.5pt")
    return stats


# ------------------------------------------------------------------ (f) PatchTST+Cal ablation on ON-Zonal
def cal_stats(df):
    cells = {v: onz(df, "PatchTST", "ONZ_real", 96, v) for _, v, _ in CAL_VARIANTS}
    cells[""] = onz(df, "PatchTST", "ONZ_real", 96)
    seeds = common_seeds("ablation_cal_ONZ", cells)
    c = {k: at(r, seeds) for k, r in cells.items()}
    st = {}
    for name, v, comp in CAL_VARIANTS + (("PatchTST (no calendar)", "", (0, 0, 0)),):
        st[v] = dict(name=name, comp=comp, mse=c[v]["mse"].values, mae=c[v]["mae"].values,
                     vs_pt=paired_ci(c[""]["mse"], c[v]["mse"]) if v else None,
                     vs_full=paired_ci(c[CAL_FULL]["mse"], c[v]["mse"]) if v != CAL_FULL else None,
                     vs_pt_mae=paired_ci(c[""]["mae"], c[v]["mae"]) if v else None,
                     vs_full_mae=paired_ci(c[CAL_FULL]["mae"], c[v]["mae"]) if v != CAL_FULL else None)
    return st, seeds


def table_cal(df, N):
    st, seeds = cal_stats(df)
    S = seeds_field(seeds)
    out = []
    mk = {1: r"\cmark", 0: r"\xmark"}
    b = best({v: s["mse"].mean() for v, s in st.items()}, 4)
    kname = {CAL_FULL: "full", "cal-dow-hol": "minus_hour", "cal-hour-hol": "minus_dow", "cal-hour-dow": "minus_holiday",
             "": "PatchTST"}
    def two(res):  # difference, then its interval on a second (small) line
        if res is None:
            return "--", ""
        return num(res[0], 4), f"{{\\scriptsize[{num(res[1], 4)}, {num(res[2], 4)}]}}"

    for v, s in st.items():
        name = s["name"] if v else "none (PatchTST)"
        cols = [mk[x] for x in s["comp"]]
        (a1, a2), (b1, b2) = two(s["vs_pt"]), two(s["vs_full"])
        out.append(f"{name} & {' & '.join(cols)} & {msd(s['mse'], 4, v in b)} & {a1} & {b1} \\\\\n"
                   f" & & & & & {a2} & {b2} \\\\\\addlinespace[1pt]")
        p = f"ablation_cal_ONZ.{kname[v]}"
        N.mean_sd(p + ".mse", s["mse"], seeds)
        N.mean_sd(p + ".mae", s["mae"], seeds)
        if s["vs_pt"]:
            N.stat(p + ".minus_PatchTST.mse", s["vs_pt"], "paired t", S)
            N.add(p + ".minus_PatchTST.mse_pct", f"{100 * s['vs_pt'][0] / st['']['mse'].mean():+.1f}%",
                  what="100 x paired difference / PatchTST mean", seeds=S)
            N.stat(p + ".minus_PatchTST.mae", s["vs_pt_mae"], "paired t", S)
        if s["vs_full"]:
            N.stat(p + ".minus_full.mse", s["vs_full"], "paired t (removing components from the full model)", S)
            N.stat(p + ".minus_full.mae", s["vs_full_mae"], "paired t", S)
    if len(out) > 4:
        out.insert(4, r"\midrule")
    cap = [f"PatchTST+Cal ablation on ON-Zonal (real zones, $L{{=}}96$, $H{{=}}96$, every test window), mean $\\pm$ std over",
           f"seeds {seeds_txt(seeds, True)}; bold: lowest. Calendar terms: hour of day, day of week (dow), Ontario statutory holiday; MAE",
           "in \\texttt{results/report\\_numbers.txt}.",
           "Last columns: MSE difference with its paired 95\\% t-interval (second line), against plain PatchTST and against",
           "the full PatchTST+Cal (the cost of removing a component)."]
    put("ablation_cal_ONZ", cap, r"""\begin{tabular}{@{}lccclll@{}}
\toprule
 & \multicolumn{3}{c}{Calendar terms} & & \multicolumn{2}{c}{$\Delta$MSE, paired} \\
\cmidrule(lr){2-4}\cmidrule(l){6-7}
PatchTST+Cal & hour & dow & holiday & MSE & $-$ PatchTST & $-$ full \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}
""")


# ------------------------------------------------------------------ (g) P x L grid at H=336
GRID_P, GRID_L = (8, 16, 32, 64), (96, 336, 720)


def grid_cells(df, kind):
    cells = {}
    for P_ in GRID_P:
        for L in GRID_L:
            cells[(P_, L)] = score(df, kind, P=P_, model="PatchTST", dataset="ETTh1", L=L, H=336)
    seeds = common_seeds(f"grid_PxL_H336 ({kind})", cells)
    return {k: at(r, seeds)["mse"].values for k, r in cells.items()}, seeds


def table_grid(df, N):
    out = []
    for kind, title in (("all", "every test window"), ("official", "official loader")):
        g, seeds = grid_cells(df, kind)
        b = best({k: v.mean() for k, v in g.items()}, 4)
        out.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{title}, seeds {seeds_txt(seeds, True)}}}}} \\\\")
        for P_ in GRID_P:
            out.append(f"$P{{=}}{P_}$, $S{{=}}{P_ // 2}$ & " + " & ".join(msd(g[(P_, L)], 4, (P_, L) in b) for L in GRID_L) + r" \\")
            for L in GRID_L:
                N.mean_sd(f"grid_PxL_H336.{kind}.P{P_}.L{L}.mse", g[(P_, L)], seeds)
        order = sorted(g, key=lambda k: g[k].mean())
        N.add(f"grid_PxL_H336.{kind}.ranking", ", ".join(f"P{p}/L{l}" for p, l in order),
              what="cells from lowest to highest mean MSE", seeds=seeds_field(seeds))
        for L in GRID_L:
            col = sorted(GRID_P, key=lambda p: g[(p, L)].mean())
            N.add(f"grid_PxL_H336.{kind}.L{L}.best_P", f"P={col[0]}", what="lowest mean MSE in this look-back",
                  seeds=seeds_field(seeds))
        out.append(r"\midrule")
    cap = ["PatchTST on ETTh1 at $H{=}336$: test MSE for patch length $P$ (stride $S=P/2$) $\\times$ look-back $L$, mean $\\pm$ std",
           "over the seeds shown; top: every test window (used for comparisons), bottom: official loader; bold: lowest in each block."]
    put("grid_PxL_H336", cap, r"""\begin{tabular}{@{}llll@{}}
\toprule
 & $L{=}96$ & $L{=}336$ & $L{=}720$ \\
\midrule
""" + "\n".join(out[:-1]) + "\n" + r"""\bottomrule
\end{tabular}
""")


# ------------------------------------------------------------------ (h) environment per job, timing
def env_info(path):
    lines = Path(path).read_text().splitlines()
    pk = {}
    for line in lines:
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)$", line)
        if m:
            pk[m.group(1).lower()] = m.group(2)
    gpus = [line for line in lines if re.match(r"^[A-Za-z][^=@]*, \d+\.\d+(\.\d+)?$", line)]
    kit = [line.split()[-1] for line in lines if line.startswith("kit commit ")]
    assert len(kit) == 1 and gpus, path
    platform = "Kaggle" if "kaggle-environments" in pk else ("Colab" if any(l.startswith("google-colab") for l in lines) else "?")
    names = sorted({g.split(",")[0] for g in gpus})
    drivers = sorted({g.split(",")[1].strip() for g in gpus})
    assert len(names) == 1 and len(drivers) == 1, path
    return dict(platform=platform, gpu=names[0], n_gpu=len(gpus), driver=drivers[0], torch=pk["torch"],
                numpy=pk["numpy"], pandas=pk.get("pandas", "?"), kit=kit[0][:7])


def table_env(df, results, N):
    res = Path(results)
    jobs = sorted({re.match(r"runs_(\w+)\.csv$", f.name).group(1) for f in res.glob("runs_*.csv")}
                  | {re.match(r"env_(\w+)\.txt$", f.name).group(1) for f in res.glob("env_*.txt")} - {"D_session1"},
                  key=lambda j: (len(j) > 2, j))
    out, tot_runs, tot_s = [], 0, 0.0
    for j in jobs:
        f = res / f"runs_{j}.csv"
        r = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["mse", "seconds", "plan", "device"])
        r = r[r["mse"].notna()]
        plans = ", ".join(sorted(r["plan"].unique())) if len(r) else ("timing" if j == "T" else "--")
        gh = r["seconds"].sum() / 3600
        tot_runs += len(r)
        tot_s += r["seconds"].sum()
        e = res / f"env_{j}.txt"
        if e.exists():
            info = env_info(e)
            kit = info["kit"]
            if j == "D":  # session 1 environment from git history (6269e2c), committed as env_D_session1.txt
                s1 = env_info(res / "env_D_session1.txt")
                for k in ("platform", "gpu", "n_gpu", "driver", "torch", "numpy", "pandas"):
                    assert s1[k] == info[k], (k, s1[k], info[k])
                kit = f"{s1['kit']}, {info['kit']}"
            if len(r):
                assert set(r["device"]) == {info["gpu"]}, (j, set(r["device"]))
            gpu = info["gpu"] + (f" $\\times${info['n_gpu']}" if info["n_gpu"] > 1 else "")
            cols = [info["platform"], gpu, info["torch"], info["numpy"], kit]
            N.add(f"env.{j}", f"{info['platform']}; {info['gpu']} x{info['n_gpu']}; driver {info['driver']}; torch {info['torch']}; "
                  f"numpy {info['numpy']}; pandas {info['pandas']}; kit {kit}", what=f"{len(r)} runs, {gh:.2f} GPU-h")
        else:
            devs = sorted(r["device"].unique())
            cols = ["--", ", ".join(devs), r"\multicolumn{3}{l}{not logged (October)}"]
            N.add(f"env.{j}", f"not logged; device {', '.join(devs)}", what=f"{len(r)} runs, {gh:.2f} GPU-h")
        out.append(f"{j} & {plans.replace('_', chr(92) + '_')} & {len(r) if len(r) else '--'} & "
                   f"{f'{gh:.2f}' if len(r) else '--'} & " + " & ".join(cols) + r" \\")
    N.add("env.all_logged_runs", f"{tot_runs} runs, {tot_s / 3600:.1f} GPU-h",
          what="every logged row in results/runs_*.csv (re-runs counted each time they ran)")
    N.add("env.distinct_runs", f"{len(df)} runs, {df['seconds'].sum() / 3600:.1f} GPU-h",
          what="distinct run_ids (report_tables.load)")
    N.add("env.devices", ", ".join(sorted(df["device"].unique())), what="device column of every distinct run")
    cap = ["Environment of each job, from \\texttt{results/env\\_<JOB>.txt} (\\texttt{pip freeze}, \\texttt{nvidia-smi}, kit commit). Platform:",
           "Kaggle if \\texttt{kaggle-environments} is installed, else Colab. GPUs as listed by \\texttt{nvidia-smi}: Kaggle lists two T4s",
           "(no run passes \\texttt{--use\\_multi\\_gpu}); driver versions are in \\texttt{results/report\\_numbers.txt}. Job D ran in two sessions",
           "(kit commits in order; session 1 from git history, \\texttt{env\\_D\\_session1.txt}). GPU-h: wall-clock time of the logged runs.",
           "Jobs A--C (October) predate environment logging; V and Vcolab: the validation gate on Kaggle and Colab; T: timing only."]
    put("env_per_job", cap, r"""\begin{tabular}{@{}llrrlllll@{}}
\toprule
Job & Plan & Runs & GPU-h & Platform & GPU & torch & numpy & Kit commit \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}
""", size="footnotesize")


TIMING_MAIN = (  # (block, model label, config_id prefix, horizons)
    ("ETTh1", "PatchTST ($L{=}336$)", "PatchTST_ETTh1_L336_H{H}", H_ALL),
    ("ETTh1", "PatchTST+Cal ($L{=}336$)", "PatchTST_ETTh1_L336_H{H}_cal-hour-dow", H_ALL),
    ("ETTh1", "DLinear ($L{=}336$)", "DLinear_ETTh1_L336_H{H}", H_ALL),
    ("ETTh1", "iTransformer ($L{=}96$)", "iTransformer_ETTh1_L96_H{H}", H_ALL),
    ("ETTh1", "Autoformer ($L{=}96$)", "Autoformer_ETTh1_L96_H{H}", H_ALL),
    ("ON-Zonal", "DLinear", "DLinear_ONZ_real_L96_H{H}", (96, 720)),
    ("ON-Zonal", "PatchTST", "PatchTST_ONZ_real_L96_H{H}", (96, 720)),
    ("ON-Zonal", "PatchTST+Cal (full)", "PatchTST_ONZ_real_L96_H{H}_cal-hour-dow-hol", (96,)),
    ("ON-Zonal", "iTransformer", "iTransformer_ONZ_real_L96_H{H}", (96, 720)),
    ("ON-Zonal", "iTransformer, no tokens", "iTransformer_ONZ_real_L96_H{H}_nots", (96,)),
    ("ON-Zonal", "Control 2 (iTransformer-CI)", "iTransformer_ONZ_real_L96_H{H}_ci", (96,)),
)


def timing(results):
    t = pd.read_csv(Path(results) / "timing.csv")
    assert t["device"].nunique() == 1, "compare only latencies measured on the same GPU type"
    out = {}
    for cid, g in t.groupby("config_id"):
        assert g["params"].nunique() == 1 and sorted(g["batch"]) == [1, 32], cid
        b = {B: g[g.batch == B].iloc[0] for B in (1, 32)}
        out[cid] = dict(params=int(g["params"].iloc[0]), b1=float(b[1]["median_ms"]), b32=float(b[32]["median_ms"]),
                        q_b1=(float(b[1]["p25_ms"]), float(b[1]["p75_ms"])),
                        q_b32=(float(b[32]["p25_ms"]), float(b[32]["p75_ms"])))
    return out, t["device"].iloc[0], t["torch"].iloc[0]


def table_timing(results, N):
    tm, device, torch_v = timing(results)
    out, prev = [], None
    for block, name, pat, Hs in TIMING_MAIN:
        if prev and block != prev:
            out.append(r"\midrule")
        prev = block
        for i, H in enumerate(Hs):
            cid = pat.format(H=H)
            x = tm[cid]
            first = f"\\multirow{{{len(Hs)}}}{{*}}{{{name}}}" if i == 0 else ""
            blk = block if (i == 0 and name == [n for b, n, *_ in TIMING_MAIN if b == block][0]) else ""
            params = f"{x['params']:,}".replace(",", "{,}")
            out.append(f"{blk} & {first} & {H} & {params} & {x['b1']:.2f} & {x['b32']:.2f} \\\\")
            N.add(f"timing.{cid}", f"{x['params']} parameters; median {x['b1']:.3f} ms (batch 1), {x['b32']:.3f} ms (batch 32)",
                  what=f"timing.py forward pass, {device}, torch {torch_v}")
    for a, b in (("PatchTST_ETTh1_L336_H96_cal-hour-dow", "PatchTST_ETTh1_L336_H96"),
                 ("PatchTST_ONZ_real_L96_H96_cal-hour-dow-hol", "PatchTST_ONZ_real_L96_H96")):
        N.add(f"timing.params_added.{a}", f"{tm[a]['params'] - tm[b]['params']:+d}", what=f"parameters of {a} - {b}")
    cap = [f"Parameters and median forward-pass latency (ms) at batch 1 and 32 (\\texttt{{timing.py}}, job T: {device}, torch",
           f"{torch_v}; 10 warm-up + 50 timed passes; quartiles in \\texttt{{results/timing.csv}}). Data loading and host-to-device",
           "copies excluded. ON-Zonal: real-zone configurations ($L{=}96$)."]
    put("timing_main", cap, r"""\begin{tabular}{@{}lllrrr@{}}
\toprule
Data & Model & $H$ & Parameters & Batch 1 & Batch 32 \\
\midrule
""" + "\n".join(out) + "\n" + r"""\bottomrule
\end{tabular}
""")


# ------------------------------------------------------------------ figure data (plot_final.py uses these)
LAT_ETTH1_H = 336


def latency_points(df, results):
    """(label, family, config_id, mse array, seeds) for ETTh1 at H=336 and ON-Zonal real at H=96, every window."""
    tm, _, _ = timing(results)
    pts = {"ETTh1": [], "ONZ": []}
    cand = []
    for P_ in GRID_P:
        for L in GRID_L:
            tag = "" if P_ == 16 else f"_P{P_}S{P_ // 2}"
            cand.append(("ETTh1", f"PatchTST P{P_} L{L}", "PatchTST", f"PatchTST_ETTh1_L{L}_H{LAT_ETTH1_H}{tag}",
                         dict(model="PatchTST", dataset="ETTh1", L=L, H=LAT_ETTH1_H, P=P_)))
    cand += [("ETTh1", "PatchTST+Cal L336", "PatchTST+Cal", f"PatchTST_ETTh1_L336_H{LAT_ETTH1_H}_cal-hour-dow",
              dict(model="PatchTST", dataset="ETTh1", L=336, H=LAT_ETTH1_H, variant=CAL_ETTH1)),
             ("ETTh1", "iTransformer L96", "iTransformer", f"iTransformer_ETTh1_L96_H{LAT_ETTH1_H}",
              dict(model="iTransformer", dataset="ETTh1", L=96, H=LAT_ETTH1_H)),
             ("ETTh1", "iTransformer L336", "iTransformer", f"iTransformer_ETTh1_L336_H{LAT_ETTH1_H}",
              dict(model="iTransformer", dataset="ETTh1", L=336, H=LAT_ETTH1_H)),
             ("ETTh1", "DLinear L336", "DLinear", f"DLinear_ETTh1_L336_H{LAT_ETTH1_H}",
              dict(model="DLinear", dataset="ETTh1", L=336, H=LAT_ETTH1_H)),
             ("ETTh1", "Autoformer L96", "Autoformer", f"Autoformer_ETTh1_L96_H{LAT_ETTH1_H}",
              dict(model="Autoformer", dataset="ETTh1", L=96, H=LAT_ETTH1_H))]
    for m, v, fam, lbl in (("DLinear", "", "DLinear", "DLinear"), ("PatchTST", "", "PatchTST", "PatchTST"),
                           ("iTransformer", "", "iTransformer", "iTransformer"),
                           ("iTransformer", "nots", "iTransformer", "iTransformer, no tokens"),
                           ("iTransformer", "ci", "iTransformer", "Control 2")) + tuple(
            ("PatchTST", v, "PatchTST+Cal", "PatchTST+Cal " + n.replace("$-$", "-")) for n, v, _ in CAL_VARIANTS):
        cand.append(("ONZ", lbl, fam, f"{m}_ONZ_real_L96_H96" + (f"_{v}" if v else ""),
                     dict(model=m, dataset="ONZ_real", L=96, H=96, variant=v)))
    cells = {}
    for ds, lbl, fam, cid, kw in cand:
        kw = dict(kw)
        P_ = kw.pop("P", None)
        cells[(ds, lbl)] = (fam, cid, score(df, "all", P=P_, **kw))
    for ds in pts:
        seeds = common_seeds(f"fig_mse_vs_latency {ds}", {k: v[2] for k, v in cells.items() if k[0] == ds})
        for (d, lbl), (fam, cid, r) in cells.items():
            if d == ds:
                x = tm[cid]
                pts[ds].append(dict(label=lbl, family=fam, config_id=cid, mse=at(r, seeds)["mse"].values,
                                    seeds=seeds, b1=x["b1"], b32=x["b32"], q_b1=x["q_b1"], q_b32=x["q_b32"],
                                    params=x["params"]))
    return pts


def pareto(points, x="b32"):
    """Labels of points not dominated in (latency, mean MSE): no other point is at least as good in both and better in one."""
    keep = []
    for p in points:
        dom = any((q[x] <= p[x] and q["mse"].mean() <= p["mse"].mean()) and
                  (q[x] < p[x] or q["mse"].mean() < p["mse"].mean()) for q in points if q is not p)
        if not dom:
            keep.append(p["label"])
    return keep


def numbers_latency(df, results, N):
    pts = latency_points(df, results)
    for ds, ps in pts.items():
        for b in ("b1", "b32"):
            N.add(f"fig_mse_vs_latency.{ds}.pareto_{b}", ", ".join(pareto(ps, b)),
                  what=f"configs not dominated in (median latency at batch {b[1:]}, mean MSE); every window",
                  seeds=seeds_field(ps[0]["seeds"]))


# ------------------------------------------------------------------ job K: prediction analysis (patch 0008)
PK_MODELS = ("iTransformer", "PatchTST", "DLinear", "Seasonal naive")
PK_SEEDS = "seed 2021 (n=1); one run per model, descriptive"
PK_FILES = ("overall", "zone_errors", "error_by_step", "holiday_vs_rest", "peak_vs_rest", "worst_windows", "examples",
            "holiday_dates")


def preds_k(results):
    """results/preds_analysis/: written on Colab by analyze_preds.py from preds_K.zip (aggregates only)."""
    d = Path(results) / "preds_analysis"
    t = {n: pd.read_csv(d / f"{n}.csv") for n in PK_FILES}
    return t, json.loads((d / "checks.json").read_text())


def pk_fmt(metric, v):
    return {"mse_z": f"{v:.4f}", "mae_z": f"{v:.4f}", "mae_MW": f"{v:.1f}", "mape_pct": f"{v:.2f}"}[metric]


def pk_rows(t):
    """The categories of the failure table: (label, count text, {model: row})."""
    ov = t["overall"].set_index("model")
    st = t["error_by_step"].set_index(["model", "h"])
    hv = t["holiday_vs_rest"].set_index(["model", "group"])
    pk = t["peak_vs_rest"][t["peak_vs_rest"].zone == "all zones"].set_index(["model", "group"])
    n_win = int(ov.loc["iTransformer", "n_cells"] // (96 * 10))
    out = [("All test windows", f"{n_win:,}", {m: ov.loc[m] for m in PK_MODELS})]
    for h in (1, 24, 96):
        out.append((f"Step $h{{=}}{h}$ only", f"{n_win:,}", {m: st.loc[(m, h)] for m in PK_MODELS}))
    for grp, lbl in (("holiday target hours", "Holiday target hours"), ("other target hours", "Other target hours"),
                     ("windows touching a holiday", "Windows with a holiday"), ("other windows", "Other windows")):
        r = hv.loc[("iTransformer", grp)]
        out.append((lbl, f"{int(r['n']):,}", {m: hv.loc[(m, grp)] for m in PK_MODELS}))
    for grp, lbl in (("peak target hours", "Peak zone-hours"), ("other target hours", "Other zone-hours"),
                     ("windows containing a peak hour", "Zone-windows with a peak"), ("other windows", "Other zone-windows")):
        r = pk.loc[("iTransformer", grp)]
        out.append((lbl, f"{int(r['n']):,}", {m: pk.loc[(m, grp)] for m in PK_MODELS}))
    return out


def pk_cells(rowmap):
    """MSE and MAPE per model; the lowest of each (at the precision shown) in bold."""
    cells = []
    for metric in ("mse_z", "mape_pct"):
        d = 4 if metric == "mse_z" else 2
        b = best({m: float(rowmap[m][metric]) for m in PK_MODELS}, d)
        cells.append({m: (f"\\textbf{{{float(rowmap[m][metric]):.{d}f}}}" if m in b else f"{float(rowmap[m][metric]):.{d}f}")
                      for m in PK_MODELS})
    return " & ".join(f"{cells[0][m]} & {cells[1][m]}" for m in PK_MODELS)


PK_HEAD = r"""\toprule
 & & \multicolumn{2}{c}{iTransformer} & \multicolumn{2}{c}{PatchTST} & \multicolumn{2}{c}{DLinear} & \multicolumn{2}{c}{Seasonal naive} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}
"""


def table_predsk(results, N):
    t, ck = preds_k(results)
    # failure categories
    lines = []
    for k, (lbl, cnt, rowmap) in enumerate(pk_rows(t)):
        if k in (1, 4, 8):
            lines.append(r"\midrule")
        lines.append(f"{lbl} & {cnt.replace(',', '{,}')} & {pk_cells(rowmap)} \\\\")
    cap = ["Failure categories on ON-Zonal (real zones, $L{=}96$, $H{=}96$, every test window; job K: one run per model,",
           "seed 2021, so these are descriptive). MSE on z-scored data and MAPE (\\%) on demand in MW; bold: lowest at the",
           "precision shown. $n$ counts windows, (window, step) pairs for target hours (each over the 10 zones), (window,",
           "step, zone) cells for zone-hours and (window, zone) pairs for zone-windows. Holiday: an Ontario statutory holiday",
           "(\\texttt{holidays.Canada(subdiv=\"ON\")}, as in PatchTST+Cal); peak: a zone's top 1\\% of test-split demand.",
           "From \\texttt{results/preds\\_analysis/} (\\texttt{analyze\\_preds.py})."]
    put("predsK_failure", cap, r"""\begin{tabular}{@{}llrrrrrrrr@{}}
""" + PK_HEAD + r"""Category & $n$ & MSE & MAPE & MSE & MAPE & MSE & MAPE & MSE & MAPE \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""", size="small", sep="3pt")
    # zones
    zt = t["zone_errors"].set_index(["model", "zone"])
    ov = t["overall"].set_index("model")
    zones = list(dict.fromkeys(t["zone_errors"]["zone"]))
    zl = [f"{z} & {pk_cells({m: zt.loc[(m, z)] for m in PK_MODELS})} \\\\" for z in zones]
    zl += [r"\midrule", f"All zones & {pk_cells({m: ov.loc[m] for m in PK_MODELS})} \\\\"]
    cap = ["Errors per zone on ON-Zonal (real zones, $L{=}96$, $H{=}96$, every test window; job K, seed 2021, one run per",
           "model). MSE on z-scored data (each zone scaled by its training mean and std) and MAPE (\\%) on demand in MW;",
           "bold: lowest in the row. MAE in MW per zone in \\texttt{results/preds\\_analysis/zone\\_errors.csv}."]
    put("predsK_zones", cap, r"""\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{iTransformer} & \multicolumn{2}{c}{PatchTST} & \multicolumn{2}{c}{DLinear} & \multicolumn{2}{c}{Seasonal naive} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}
Zone & MSE & MAPE & MSE & MAPE & MSE & MAPE & MSE & MAPE \\
\midrule
""" + "\n".join(zl) + "\n" + r"""\bottomrule
\end{tabular}
""", size="small", sep="4pt")
    # worst windows (three-model mean ranking)
    ww = t["worst_windows"]
    w3 = ww[ww.ranking == "three-model mean"].sort_values("rank")
    mk = {True: r"\cmark", False: r"\xmark"}
    wl = [f"{int(r['rank'])} & {r['forecast_origin'][:16]} & {r['window_mse_z']:.4f} & {r['mse_z_iTransformer']:.4f} & "
          f"{r['mse_z_PatchTST']:.4f} & {r['mse_z_DLinear']:.4f} & {r['worst_zone']} & {mk[bool(r['touches_holiday'])]} & "
          f"{mk[bool(r['contains_peak_hour'])]} \\\\" for _, r in w3.iterrows()]
    cap = ["The 10 worst test windows on ON-Zonal by the mean MSE of the three models (z-scored, all zones; job K, seed 2021),",
           "chosen so that no two windows overlap (forecast origins at least 96 hours apart). Origin: the last input hour;",
           "worst zone: the zone with the largest three-model MSE in the window; holiday / peak: the 96 target hours include a",
           "statutory holiday / a zone's top-1\\% demand hour. Per-model rankings in \\texttt{results/preds\\_analysis/worst\\_windows.csv}."]
    put("predsK_worst", cap, r"""\begin{tabular}{@{}rlrrrrlcc@{}}
\toprule
 & & \multicolumn{4}{c}{Window MSE} & & & \\
\cmidrule(lr){3-6}
Rank & Forecast origin & Mean & iTrans. & PatchTST & DLinear & Worst zone & Holiday & Peak \\
\midrule
""" + "\n".join(wl) + "\n" + r"""\bottomrule
\end{tabular}
""", size="small", sep="4pt")
    numbers_predsk(t, ck, N)


def numbers_predsk(t, ck, N):
    S = PK_SEEDS
    models3 = PK_MODELS[:3]
    N.add("predsK.checks.max_abs_target_minus_rebuilt_z",
          f"{max(ck[f'{m}.max_abs_true_minus_rebuilt_z'] for m in models3):.1e}",
          what="exported targets vs ON-Zonal rebuilt on Colab, z-units, every model and window (stop above 1e-4)", seeds=S)
    for m in models3:
        N.add(f"predsK.checks.{m}.mse_reproduced", f"{ck[f'{m}.mse_from_arrays']:.6f} (logged {ck[f'{m}.mse_logged']:.6f})",
              what="MSE from the exported arrays vs results/runs_K.csv", seeds=S)
    N.add("predsK.checks.rebuild", f"rows {ck['rows']}, test {ck['test_period'][0]} to {ck['test_period'][1]}, "
          f"{ck['n_windows']} windows; split and seasonal naive equal the committed files", what="checks.json", seeds=S)
    ov = t["overall"].set_index("model")
    for m in PK_MODELS:
        for metric in ("mse_z", "mae_z", "mae_MW", "mape_pct"):
            N.add(f"predsK.overall.{m}.{metric}", pk_fmt(metric, ov.loc[m, metric]), what="every test window, all zones", seeds=S)
    zt = t["zone_errors"]
    for m in PK_MODELS:
        z = zt[zt.model == m].set_index("zone")
        for zone in z.index:
            for metric in ("mse_z", "mae_MW", "mape_pct"):
                N.add(f"predsK.zone.{m}.{zone}.{metric}", pk_fmt(metric, z.loc[zone, metric]), what="every test window", seeds=S)
        hi, lo = z["mape_pct"].idxmax(), z["mape_pct"].idxmin()
        N.add(f"predsK.zone.{m}.highest_mape", f"{hi} ({z.loc[hi, 'mape_pct']:.2f}%)", what="zone with the highest MAPE", seeds=S)
        N.add(f"predsK.zone.{m}.lowest_mape", f"{lo} ({z.loc[lo, 'mape_pct']:.2f}%)", what="zone with the lowest MAPE", seeds=S)
        him = z["mse_z"].idxmax()
        N.add(f"predsK.zone.{m}.highest_mse", f"{him} ({z.loc[him, 'mse_z']:.4f})", what="zone with the highest z-scored MSE",
              seeds=S)
    st = t["error_by_step"]
    for m in PK_MODELS:
        s = st[st.model == m].set_index("h")
        for h in (1, 24, 48, 96):
            for metric in ("mse_z", "mape_pct"):
                N.add(f"predsK.step.{m}.h{h}.{metric}", pk_fmt(metric, s.loc[h, metric]), what=f"forecast step {h}", seeds=S)
    piv = st[st.model.isin(models3)].pivot(index="h", columns="model", values="mse_z")
    low = piv.idxmin(axis=1)
    N.add("predsK.step.lowest_mse_model.h1", low.loc[1], what="lowest MSE of the three models at step 1", seeds=S)
    it_from = next((h for h in piv.index if (low.loc[h:] == "iTransformer").all()), None)
    N.add("predsK.step.iTransformer_lowest_from_h", f"{it_from}" if it_from else "never",
          what="first step from which iTransformer has the lowest MSE of the three models at every later step", seeds=S)
    hv = t["holiday_vs_rest"].set_index(["model", "group"])
    pk = t["peak_vs_rest"][t["peak_vs_rest"].zone == "all zones"].set_index(["model", "group"])
    for name, tab_, pairs in (("holiday", hv, (("holiday target hours", "other target hours", "hours"),
                                             ("windows touching a holiday", "other windows", "windows"))),
                              ("peak", pk, (("peak target hours", "other target hours", "hours"),
                                            ("windows containing a peak hour", "other windows", "windows")))):
        for g1, g2, kind in pairs:
            N.add(f"predsK.{name}.{kind}.n", f"{int(tab_.loc[('iTransformer', g1), 'n'])} vs {int(tab_.loc[('iTransformer', g2), 'n'])}",
                  what=f"{g1} vs {g2} ({tab_.loc[('iTransformer', g1), 'unit']})", seeds=S)
            for m in PK_MODELS:
                for metric in ("mse_z", "mae_MW", "mape_pct"):
                    N.add(f"predsK.{name}.{kind}.{m}.{metric}",
                          f"{pk_fmt(metric, tab_.loc[(m, g1), metric])} vs {pk_fmt(metric, tab_.loc[(m, g2), metric])}",
                          what=f"{g1} vs {g2}", seeds=S)
                N.add(f"predsK.{name}.{kind}.{m}.mse_ratio", f"{tab_.loc[(m, g1), 'mse_z'] / tab_.loc[(m, g2), 'mse_z']:.2f}",
                      what=f"MSE on {g1} / MSE on {g2}", seeds=S)
    ww = t["worst_windows"]
    w3 = ww[ww.ranking == "three-model mean"].sort_values("rank")
    for _, r in w3.iterrows():
        N.add(f"predsK.worst.three_model_mean.rank{int(r['rank'])}",
              f"origin {r['forecast_origin'][:16]}; MSE {r['window_mse_z']:.4f} (iTransformer {r['mse_z_iTransformer']:.4f}, "
              f"PatchTST {r['mse_z_PatchTST']:.4f}, DLinear {r['mse_z_DLinear']:.4f}); worst zone {r['worst_zone']}; "
              f"holiday {'yes' if r['touches_holiday'] else 'no'}; peak {'yes' if r['contains_peak_hour'] else 'no'}",
              what="non-overlapping windows ranked by the three-model mean MSE", seeds=S)
    months = pd.to_datetime(w3["forecast_origin"]).dt.month
    N.add("predsK.worst.three_model_mean.summary",
          f"{int(months.between(6, 9).sum())} of {len(w3)} start in June-September; {int(w3['contains_peak_hour'].sum())} contain a "
          f"peak hour; {int(w3['touches_holiday'].sum())} touch a holiday", what="the 10 worst windows above", seeds=S)
    for m in models3:
        r = ww[(ww.ranking == m) & (ww["rank"] == 1)].iloc[0]
        N.add(f"predsK.worst.{m}.rank1", f"origin {r['forecast_origin'][:16]}; MSE {r['window_mse_z']:.4f}; worst zone {r['worst_zone']}",
              what=f"worst window of {m}", seeds=S)
    for _, r in t["examples"].iterrows():
        N.add(f"predsK.example.{r['panel']}",
              f"origin {r['forecast_origin'][:16]}; Toronto MSE " + ", ".join(f"{m} {r[f'mse_z_{m}_Toronto']:.3f}" for m in models3),
              what=f"fig_forecast_examples: {r['rule']}", seeds=S)
    N.add("predsK.holiday_dates", f"{len(t['holiday_dates'])} dates, {t['holiday_dates']['date'].iloc[0]} to "
          f"{t['holiday_dates']['date'].iloc[-1]}", what="statutory holidays inside the test targets", seeds=S)


# ------------------------------------------------------------------ main
def build(df, P, results):
    N = Numbers()
    table_main_etth1(df, P, N)
    table_main_onz(df, results, N)
    table_repro_ettm1(df, P, N)
    table_rq1(df, P, results, N)
    table_f3(df, results, N)
    table_rq2(df, N)
    table_cal(df, N)
    table_grid(df, N)
    table_env(df, results, N)
    table_timing(results, N)
    numbers_latency(df, results, N)
    table_predsk(results, N)
    N.write(Path(results) / "report_numbers.txt")
    return N


if __name__ == "__main__":
    res = sys.argv[1] if len(sys.argv) > 1 else "results"
    build(load(res), pd.read_csv(HERE / "published.csv"), res)
