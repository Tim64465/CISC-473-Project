"""Second, independent computation of every line of results/report_numbers.txt.

    python tests/verify_report_numbers.py [results]      # exit 0 only if every line agrees

report_numbers.txt is written by final_tables.py (pandas, report_tables.load/score). This script
shares no code with them: it reads the raw runs_*.csv files with the csv module, merges copies of a
run_id itself, finds each cell by constructing its run_id, and computes means, sample standard
deviations and t-intervals by hand (scipy only for t quantiles and tail probabilities). Each line's
value, 95% CI, p-value and seeds are recomputed and compared as printed; lines whose value is a list
(Pareto sets, rankings) are compared as text. Any line not recomputed here counts as a failure.
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

from scipy import stats

KIT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else KIT / "results"
TOL = 1.5e-6
HS = (96, 192, 336, 720)
CAND_SEEDS = range(2015, 2035)


# ------------------------------------------------------------------ raw runs
def f(x):
    return float(x) if x not in ("", None) else None


RUNS, KEPT = {}, {}  # run_id -> merged scores; run_id -> the copy kept for seconds/device
for path in sorted(RES.glob("**/runs*.csv"), key=str):
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if not row.get("mse"):
                continue
            sc = row.get("scoring") or ("all" if row["mode"] == "rescore_bs1" else "official")
            full = int(row["n_scored"]) == int(row["n_test_windows"])
            off = (f(row["mse"]), f(row["mae"])) if sc == "official" else (
                (f(row["mse_official"]), f(row["mae_official"])) if sc == "all+official" else None)
            n_off = int(row["n_scored"]) if sc == "official" else (int(row["n_scored_official"]) if sc == "all+official" else None)
            every = (f(row["mse"]), f(row["mae"])) if (sc in ("all", "all+official") or (sc == "official" and full)) else None
            rid = row["run_id"]
            r = RUNS.setdefault(rid, dict(off=None, every=None, n_off=None, n_test=int(row["n_test_windows"]), row=row))
            for k, v in (("off", off), ("every", every)):
                if v is None:
                    continue
                if r[k] is None:
                    r[k] = v
                else:
                    assert max(abs(a - b) for a, b in zip(r[k], v)) <= TOL, (rid, k, r[k], v)
            if r["n_off"] is None and n_off is not None:
                r["n_off"] = n_off
            if rid not in KEPT or (sc == "all+official" and (KEPT[rid].get("scoring") or "") != "all+official"):
                KEPT[rid] = row


def rid(model, ds, L, H, seed, P=16, var=""):
    tag = f"_P{P}S{P // 2}" if model == "PatchTST" and P != 16 else ""
    return f"{model}_{ds}_L{L}_H{H}_s{seed}{tag}" + (f"_{var}" if var else "")


def off(model, ds, L, H, seed, P=16, var="", j=0):
    r = RUNS.get(rid(model, ds, L, H, seed, P, var))
    return None if r is None or r["off"] is None or r["row"]["mode"] != "train" else r["off"][j]


def every(model, ds, L, H, seed, P=16, var="", j=0):
    b = RUNS.get(rid(model, ds, L, H, seed, P, var) + "_bs1")  # a batch-1 re-score wins over the training row
    if b is not None and b["every"] is not None:
        return b["every"][j]
    r = RUNS.get(rid(model, ds, L, H, seed, P, var))
    return None if r is None or r["every"] is None else r["every"][j]


def seeds_where(*getters):
    return [s for s in CAND_SEEDS if all(g(s) is not None for g in getters)]


# ------------------------------------------------------------------ statistics by hand
def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def tint(d):
    """mean(d), 95% t-interval, two-sided p of the one-sample t-test."""
    n, m, s = len(d), mean(d), sd(d)
    if s == 0:
        return m, m, m, float("nan")
    h = stats.t.ppf(0.975, n - 1) * s / math.sqrt(n)
    return m, m - h, m + h, 2 * stats.t.sf(abs(m / (s / math.sqrt(n))), n - 1)


def paired(a, b):
    return tint([y - x for x, y in zip(a, b)])


def welch(a, b):
    """mean(b) - mean(a), Welch interval and p."""
    va, vb = sd(a) ** 2 / len(a), sd(b) ** 2 / len(b)
    se = math.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    m = mean(b) - mean(a)
    q = stats.t.ppf(0.975, dof)
    return m, m - q * se, m + q * se, 2 * stats.t.sf(abs(m / se), dof)


# ------------------------------------------------------------------ expected lines
EXP = {}  # key -> dict(value=..., ci=..., p=..., seeds=..., full=None)


def S(seeds):
    s = sorted(seeds)
    r = f"{s[0]}-{s[-1]}" if len(s) > 1 and s == list(range(s[0], s[-1] + 1)) else ",".join(map(str, s))
    return f"seeds {r} (n={len(s)})"


def P_(p):
    return "p=nan" if not math.isfinite(p) else ("p<0.0001" if p < 0.0001 else f"p={p:.4f}")


def put_ms(key, x, seeds, d=4):
    EXP[key] = dict(value=f"{mean(x):.{d}f} ± {sd(x):.{d}f}", ci="--", seeds=S(seeds))


def put_stat(key, res, seeds_text, d=4):
    m, lo, hi, p = res
    EXP[key] = dict(value=f"{m:+.{d}f}", ci=f"[{lo:+.{d}f}, {hi:+.{d}f}]", p=P_(p), seeds=seeds_text)


def put_val(key, value, seeds_text="--", ci="--"):
    EXP[key] = dict(value=value, ci=ci, seeds=seeds_text)


def r3(v, d):
    return float(f"{v:.{d}f}")


with open(KIT / "published.csv") as fh:
    PUB = {(r["model"], r["dataset"], int(r["H"]) if r["H"].isdigit() else r["H"]): r for r in csv.DictReader(fh)}

# (a) main_ETTh1 ---------------------------------------------------------------
cells = []
for H in HS:
    cells += [lambda s, H=H: off("PatchTST", "ETTh1", 336, H, s), lambda s, H=H: off("PatchTST", "ETTh1", 336, H, s, var="cal-hour-dow")]
sd_a = seeds_where(*cells)
for H in HS:
    for j, m in enumerate(("mse", "mae")):
        pt = [off("PatchTST", "ETTh1", 336, H, s, j=j) for s in sd_a]
        cal = [off("PatchTST", "ETTh1", 336, H, s, var="cal-hour-dow", j=j) for s in sd_a]
        k = f"main_ETTh1.H{H}"
        put_ms(f"{k}.PatchTST.{m}", pt, sd_a)
        put_ms(f"{k}.PatchTST+Cal.{m}", cal, sd_a)
        res = paired(pt, cal)
        put_stat(f"{k}.PatchTST+Cal_minus_PatchTST.{m}", res, S(sd_a))
        put_val(f"{k}.PatchTST+Cal_minus_PatchTST.{m}_pct", f"{100 * res[0] / mean(pt):+.2f}%", S(sd_a))
        pub = {n: float(PUB[(n, "ETTh1", H)][m]) for n in ("PatchTST", "DLinear", "iTransformer")}
        for name, x in (("PatchTST", pt), ("PatchTST+Cal", cal)):
            put_val(f"{k}.{name}_vs_published_PatchTST42.{m}_pct", f"{100 * (mean(x) - pub['PatchTST']) / pub['PatchTST']:+.1f}%", S(sd_a))
        vals = [("published PatchTST/42", pub["PatchTST"]), ("published DLinear", pub["DLinear"]),
                ("published iTransformer", pub["iTransformer"]), ("our PatchTST", mean(pt)), ("PatchTST+Cal", mean(cal))]
        lo = min(r3(v, 3) for _, v in vals)
        put_val(f"{k}.best.{m}", ", ".join(n for n, v in vals if r3(v, 3) == lo), S(sd_a))

# (b) main_ONZ -------------------------------------------------------------------
ONZ96 = (("DLinear", "", "DLinear"), ("PatchTST", "", "PatchTST"), ("iTransformer", "", "iTransformer"),
         ("PatchTST", "cal-hour-dow-hol", "PatchTST+Cal"))
getters = [lambda s, m=m, v=v: every(m, "ONZ_real", 96, 96, s, var=v) for m, v, _ in ONZ96]
getters += [lambda s, m=m: every(m, "ONZ_real", 96, 720, s) for m in ("DLinear", "PatchTST", "iTransformer")]
sd_b = seeds_where(*getters)
with open(RES / "onzonal_seasonal_naive.csv") as fh:
    SN = {int(r["H"]): r for r in csv.DictReader(fh)}
for H in (96, 720):
    models = ONZ96 if H == 96 else ONZ96[:3]
    for j, m in enumerate(("mse", "mae")):
        put_val(f"main_ONZ.H{H}.seasonal_naive.{m}", f"{float(SN[H][m]):.4f}")
        for mod, v, name in models:
            put_ms(f"main_ONZ.H{H}.{name}.{m}", [every(mod, "ONZ_real", 96, H, s, var=v, j=j) for s in sd_b], sd_b)
    for mod, v, name in models:
        x = [every(mod, "ONZ_real", 96, H, s, var=v) for s in sd_b]
        put_val(f"main_ONZ.H{H}.{name}.mse_gain_over_seasonal_naive_pct", f"{100 * (1 - mean(x) / float(SN[H]['mse'])):.1f}%", S(sd_b))
    vals = [("seasonal naive", float(SN[H]["mse"]))] + [(name, mean([every(mod, "ONZ_real", 96, H, s, var=v) for s in sd_b]))
                                                       for mod, v, name in models]
    lo = min(r3(v, 4) for _, v in vals)
    put_val(f"main_ONZ.H{H}.best.mse", ", ".join(n for n, v in vals if r3(v, 4) == lo), S(sd_b))
for name, a, b, test, Hs in (("PatchTST+Cal_minus_PatchTST", ("PatchTST", ""), ("PatchTST", "cal-hour-dow-hol"), paired, (96,)),
                             ("PatchTST+Cal_minus_iTransformer", ("iTransformer", ""), ("PatchTST", "cal-hour-dow-hol"), welch, (96,)),
                             ("PatchTST_minus_iTransformer", ("iTransformer", ""), ("PatchTST", ""), welch, (96, 720))):
    for H in Hs:
        for j, m in enumerate(("mse", "mae")):
            xa = [every(a[0], "ONZ_real", 96, H, s, var=a[1], j=j) for s in sd_b]
            xb = [every(b[0], "ONZ_real", 96, H, s, var=b[1], j=j) for s in sd_b]
            res = test(xa, xb)
            put_stat(f"main_ONZ.H{H}.{name}.{m}", res, S(sd_b))
            put_val(f"main_ONZ.H{H}.{name}.{m}_pct", f"{100 * res[0] / mean(xa):+.1f}%", S(sd_b))

# (c) repro_ETTm1 -----------------------------------------------------------------
CFG = (("PatchTST", 336), ("iTransformer", 96), ("Autoformer", 96))
g = []
for mod, L in CFG:
    for H in HS:
        g += [lambda s, mod=mod, L=L, H=H: off(mod, "ETTm1", L, H, s), lambda s, mod=mod, L=L, H=H: every(mod, "ETTm1", L, H, s)]
sd_c = seeds_where(*g)
f1_all = []
for mod, L in CFG:
    rels = []
    for H in HS:
        o = [off(mod, "ETTm1", L, H, s) for s in sd_c]
        a = [every(mod, "ETTm1", L, H, s) for s in sd_c]
        pub = PUB[(mod, "ETTm1", H)]
        put_ms(f"repro_ETTm1.{mod}.H{H}.mse_official", o, sd_c)
        put_ms(f"repro_ETTm1.{mod}.H{H}.mae_official", [off(mod, "ETTm1", L, H, s, j=1) for s in sd_c], sd_c)
        rel = 100 * (mean(o) - float(pub["mse"])) / float(pub["mse"])
        rels.append(rel)
        put_val(f"repro_ETTm1.{mod}.H{H}.mse_vs_published_pct", f"{rel:+.1f}%", S(sd_c))
        m_, lo_, hi_, _ = tint(o)
        put_val(f"repro_ETTm1.{mod}.H{H}.published_in_95ci", "yes" if lo_ <= float(pub["mse"]) <= hi_ else "no", S(sd_c),
                ci=f"[{lo_:.4f}, {hi_:.4f}]")
        r0 = RUNS[rid(mod, "ETTm1", L, H, sd_c[0])]
        if r0["n_off"] < r0["n_test"]:
            put_stat(f"repro_ETTm1.{mod}.H{H}.F1_all_minus_official.mse", paired(o, a), S(sd_c))
            f1_all += [y - x for x, y in zip(o, a)]
    put_val(f"repro_ETTm1.{mod}.max_abs_mse_vs_published_pct", f"{max(abs(x) for x in rels):.1f}%", S(sd_c))
put_val("repro_ETTm1.PatchTST.F1_all_minus_official.mse_range_over_runs", f"{min(f1_all):+.4f} to {max(f1_all):+.4f}", S(sd_c))

# (d) RQ1 and F3 ------------------------------------------------------------------
SEL = json.loads((RES / "f3_selection.json").read_text())
VALI = {}
for path in sorted(RES.glob("**/runs*.csv"), key=str):
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row.get("vali_loss_best") and row["mode"] == "train" and row["dataset"] == "ETTh1" and row["L"] == "96" \
                    and row["seed"] == "2021" and row["patch_len"] in ("", "16"):
                VALI.setdefault((row["model"], int(row["H"]), row.get("variant") or ""), set()).add(float(row["vali_loss_best"]))
WIN = {}
for mod in ("PatchTST", "iTransformer"):
    for H in HS:
        losses = {}
        for lr, var in ((3e-5, "lr3e-05"), (1e-4, ""), (3e-4, "lr0.0003")):
            v = VALI[(mod, H, var)]
            assert len(v) == 1, (mod, H, var, v)
            losses[lr] = v.pop()
        win = min(losses, key=losses.get)
        js = SEL[mod][str(H)]
        assert js["winner_lr"] == win and {float(k): x for k, x in js["vali_loss_best"].items()} == losses, (mod, H)
        WIN[(mod, H)] = (win, losses)


def lrvar(lr):
    return "" if lr == 1e-4 else {3e-5: "lr3e-05", 3e-4: "lr0.0003"}[lr]


g = []
for H in HS:
    g += [lambda s, H=H: off("PatchTST", "ETTh1", 336, H, s), lambda s, H=H: every("PatchTST", "ETTh1", 336, H, s),
          lambda s, H=H: off("PatchTST", "ETTh1", 96, H, s), lambda s, H=H: every("PatchTST", "ETTh1", 96, H, s),
          lambda s, H=H: off("iTransformer", "ETTh1", 96, H, s), lambda s, H=H: every("iTransformer", "ETTh1", 96, H, s),
          lambda s, H=H: every("iTransformer", "ETTh1", 336, H, s),
          lambda s, H=H: every("PatchTST", "ETTh1", 96, H, s, var=lrvar(WIN[("PatchTST", H)][0])),
          lambda s, H=H: every("iTransformer", "ETTh1", 96, H, s, var=lrvar(WIN[("iTransformer", H)][0]))]
sd_d = seeds_where(*g)
for H in HS:
    c = lambda fn, *a, **k: [fn(*a, s, **k) for s in sd_d]  # noqa: E731
    p3o, p3a = c(off, "PatchTST", "ETTh1", 336, H), c(every, "PatchTST", "ETTh1", 336, H)
    p9o, p9a = c(off, "PatchTST", "ETTh1", 96, H), c(every, "PatchTST", "ETTh1", 96, H)
    i9, i3 = c(every, "iTransformer", "ETTh1", 96, H), c(every, "iTransformer", "ETTh1", 336, H)
    psel = c(every, "PatchTST", "ETTh1", 96, H, var=lrvar(WIN[("PatchTST", H)][0]))
    isel = c(every, "iTransformer", "ETTh1", 96, H, var=lrvar(WIN[("iTransformer", H)][0]))
    gpub = float(PUB[("iTransformer", "ETTh1", H)]["mse"]) - float(PUB[("PatchTST", "ETTh1", H)]["mse"])
    put_val(f"rq1.H{H}.G_published", f"{gpub:+.3f}")
    res = dict(G0=welch(p3o, i9), G1=welch(p3a, i9), G2=welch(p9a, i9), G3=welch(psel, isel), G2alt=welch(p3a, i3),
               C1=paired(p3o, p3a), C2=paired(p3a, p9a),
               C3=welch([x - y for x, y in zip(p9a, psel)], [x - y for x, y in zip(i9, isel)]),
               C2alt=paired(i3, i9),
               phi1=tint([0.5 * ((a - b) + (cc - dd)) for a, b, cc, dd in zip(p3a, p3o, p9a, p9o)]),
               phi2=tint([0.5 * ((a - b) + (cc - dd)) for a, b, cc, dd in zip(p9o, p3o, p9a, p3a)]),
               G11=welch(p9a, i9), phi1alt=tint([a - b for a, b in zip(p3a, p3o)]),
               phi2alt=tint([a - b for a, b in zip(i9, i3)]), G11alt=welch(p3a, i3))
    for k, v in res.items():
        put_stat(f"rq1.H{H}.{k}", v, S(sd_d))
    g0 = res["G0"][0]
    for k in ("phi1", "phi2", "G11", "phi1alt", "phi2alt", "G11alt", "C1", "C2", "C3"):
        put_val(f"rq1.H{H}.{k}_share_of_G0", f"{100 * res[k][0] / g0:.0f}%", S(sd_d))
for mod in ("PatchTST", "iTransformer"):
    for H in HS:
        lr, losses = WIN[(mod, H)]
        d = [every(mod, "ETTh1", 96, H, s) for s in sd_d]
        put_val(f"f3.{mod}.H{H}.selected_lr", f"{lr:g}", "seed 2021")
        put_ms(f"f3.{mod}.H{H}.mse_default_lr", d, sd_d)
        if lr != 1e-4:
            x = [every(mod, "ETTh1", 96, H, s, var=lrvar(lr)) for s in sd_d]
            put_ms(f"f3.{mod}.H{H}.mse_selected_lr", x, sd_d)
            res = paired(d, x)
            put_stat(f"f3.{mod}.H{H}.selected_minus_default.mse", res, S(sd_d))
            put_val(f"f3.{mod}.H{H}.selected_minus_default.mse_pct", f"{100 * res[0] / mean(d):+.1f}%", S(sd_d))

# (e) RQ2 -------------------------------------------------------------------------
KEYN = {("iTransformer", ""): "iTransformer", ("iTransformer", "nots"): "iTransformer_no_tokens",
        ("PatchTST", ""): "PatchTST", ("DLinear", ""): "DLinear", ("iTransformer", "ci"): "Control2"}


def pair_seeds(H, *mv):
    return seeds_where(*[lambda s, m=m, v=v, ds=ds: every(m, ds, 96, H, s, var=v) for m, v in mv for ds in ("ONZ_real", "ONZ_shift")])


def eff(H, mv, seeds):
    return [every(*mv[:1], "ONZ_shift", 96, H, s, var=mv[1]) - every(*mv[:1], "ONZ_real", 96, H, s, var=mv[1]) for s in seeds]


IT, NOTS, PT, DL, CI = ("iTransformer", ""), ("iTransformer", "nots"), ("PatchTST", ""), ("DLinear", ""), ("iTransformer", "ci")
for H in (96, 720):
    models = (IT, NOTS, PT, DL, CI) if H == 96 else (IT, PT, DL)
    for mv in models:
        sd_ = pair_seeds(H, mv)
        real = [every(mv[0], "ONZ_real", 96, H, s, var=mv[1]) for s in sd_]
        shift = [every(mv[0], "ONZ_shift", 96, H, s, var=mv[1]) for s in sd_]
        put_ms(f"rq2.H{H}.{KEYN[mv]}.real.mse", real, sd_)
        put_ms(f"rq2.H{H}.{KEYN[mv]}.shift.mse", shift, sd_)
        put_stat(f"rq2.H{H}.{KEYN[mv]}.Delta_shift_minus_real", paired(real, shift), S(sd_))
    for a, b in [(IT, PT), (IT, DL)] + ([(IT, CI), (NOTS, PT), (NOTS, DL)] if H == 96 else []):
        sd_ = pair_seeds(H, a, b)
        put_stat(f"rq2.H{H}.D.{KEYN[a]}_vs_{KEYN[b]}", welch(eff(H, b, sd_), eff(H, a, sd_)), S(sd_))
    for a, b in [(PT, DL)] + ([(CI, PT), (CI, DL)] if H == 96 else []):
        sd_ = pair_seeds(H, a, b)
        put_stat(f"rq2.H{H}.controls.{KEYN[a]}_minus_{KEYN[b]}", welch(eff(H, b, sd_), eff(H, a, sd_)), S(sd_))
sd_ = pair_seeds(96, CI, IT)
for ds, name in (("ONZ_real", "real"), ("ONZ_shift", "shift")):
    res = welch([every("iTransformer", ds, 96, 96, s) for s in sd_], [every("iTransformer", ds, 96, 96, s, var="ci") for s in sd_])
    put_stat(f"rq2.H96.Control2_minus_iTransformer.{name}.mse", res, S(sd_))
sd_ = pair_seeds(96, IT, NOTS)
res = paired([every("iTransformer", "ONZ_real", 96, 96, s, var="nots") for s in sd_], [every("iTransformer", "ONZ_real", 96, 96, s) for s in sd_])
put_stat("rq2.H96.calendar_tokens.with_minus_without.real.mse", res, S(sd_))

# (f) PatchTST+Cal ablation ---------------------------------------------------------
CALV = (("full", "cal-hour-dow-hol"), ("minus_hour", "cal-dow-hol"), ("minus_dow", "cal-hour-hol"),
        ("minus_holiday", "cal-hour-dow"), ("PatchTST", ""))
sd_f = seeds_where(*[lambda s, v=v: every("PatchTST", "ONZ_real", 96, 96, s, var=v) for _, v in CALV])
for name, v in CALV:
    for j, m in enumerate(("mse", "mae")):
        x = [every("PatchTST", "ONZ_real", 96, 96, s, var=v, j=j) for s in sd_f]
        put_ms(f"ablation_cal_ONZ.{name}.{m}", x, sd_f)
        pt = [every("PatchTST", "ONZ_real", 96, 96, s, j=j) for s in sd_f]
        full = [every("PatchTST", "ONZ_real", 96, 96, s, var="cal-hour-dow-hol", j=j) for s in sd_f]
        if v:
            res = paired(pt, x)
            put_stat(f"ablation_cal_ONZ.{name}.minus_PatchTST.{m}", res, S(sd_f))
            if m == "mse":
                put_val(f"ablation_cal_ONZ.{name}.minus_PatchTST.mse_pct", f"{100 * res[0] / mean(pt):+.1f}%", S(sd_f))
        if v != "cal-hour-dow-hol":
            put_stat(f"ablation_cal_ONZ.{name}.minus_full.{m}", paired(full, x), S(sd_f))

# (g) P x L grid ----------------------------------------------------------------------
for kind, fn in (("all", every), ("official", off)):
    sd_g = seeds_where(*[lambda s, P=P, L=L: fn("PatchTST", "ETTh1", L, 336, s, P=P) for P in (8, 16, 32, 64) for L in (96, 336, 720)])
    means = {}
    for P in (8, 16, 32, 64):
        for L in (96, 336, 720):
            x = [fn("PatchTST", "ETTh1", L, 336, s, P=P) for s in sd_g]
            put_ms(f"grid_PxL_H336.{kind}.P{P}.L{L}.mse", x, sd_g)
            means[(P, L)] = mean(x)
    put_val(f"grid_PxL_H336.{kind}.ranking", ", ".join(f"P{p}/L{l}" for p, l in sorted(means, key=means.get)), S(sd_g))
    for L in (96, 336, 720):
        put_val(f"grid_PxL_H336.{kind}.L{L}.best_P", f"P={min((8, 16, 32, 64), key=lambda p: means[(p, L)])}", S(sd_g))

# (h) environment and timing (whole lines) -----------------------------------------------
FULL = {}


def envinfo(path):
    pk, gpus, kit, colab = {}, [], None, False
    for line in path.read_text().splitlines():
        if "==" in line and " " not in line:
            n, v = line.split("==", 1)
            pk[n.lower()] = v
        elif line.startswith("kit commit "):
            kit = line.split()[2][:7]
        elif re.fullmatch(r"[A-Za-z][^=@]*, [0-9.]+", line):
            gpus.append(line)
        if line.startswith("google-colab"):
            colab = True
    plat = "Kaggle" if "kaggle-environments" in pk else ("Colab" if colab else "?")
    name, drv = gpus[0].split(", ")
    return plat, name, len(gpus), drv, pk["torch"], pk["numpy"], pk.get("pandas", "?"), kit


jobs = sorted({p.name[5:-4] for p in RES.glob("runs_*.csv")} | {p.name[4:-4] for p in RES.glob("env_*.txt")} - {"D_session1"},
              key=lambda j: (len(j) > 2, j))
tot_n, tot_s = 0, 0.0
for j in jobs:
    rows = []
    if (RES / f"runs_{j}.csv").exists():
        with open(RES / f"runs_{j}.csv") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("mse")]
    n, sec = len(rows), sum(float(r["seconds"]) for r in rows)
    tot_n, tot_s = tot_n + n, tot_s + sec
    e = RES / f"env_{j}.txt"
    if e.exists():
        plat, name, ng, drv, tv, nv, pv, kit = envinfo(e)
        if j == "D":
            kit = envinfo(RES / "env_D_session1.txt")[7] + ", " + kit
        FULL[f"env.{j}"] = f"{plat}; {name} x{ng}; driver {drv}; torch {tv}; numpy {nv}; pandas {pv}; kit {kit} | -- | " \
                           f"{n} runs, {sec / 3600:.2f} GPU-h | --"
    else:
        FULL[f"env.{j}"] = f"not logged; device {', '.join(sorted({r['device'] for r in rows}))} | -- | {n} runs, {sec / 3600:.2f} GPU-h | --"
FULL["env.all_logged_runs"] = f"{tot_n} runs, {tot_s / 3600:.1f} GPU-h | -- | every logged row in results/runs_*.csv " \
                              "(re-runs counted each time they ran) | --"
FULL["env.distinct_runs"] = f"{len(KEPT)} runs, {sum(float(r['seconds']) for r in KEPT.values()) / 3600:.1f} GPU-h | -- | " \
                            "distinct run_ids (report_tables.load) | --"
FULL["env.devices"] = f"{', '.join(sorted({r['device'] for r in KEPT.values()}))} | -- | device column of every distinct run | --"

TM = {}
with open(RES / "timing.csv") as fh:
    for r in csv.DictReader(fh):
        TM.setdefault(r["config_id"], {})[int(r["batch"])] = r
dev = {r["device"] for d in TM.values() for r in d.values()}
tv = {r["torch"] for d in TM.values() for r in d.values()}
assert len(dev) == 1 and len(tv) == 1
dev, tv = dev.pop(), tv.pop()
for pat, Hs in (("PatchTST_ETTh1_L336_H{H}", HS), ("PatchTST_ETTh1_L336_H{H}_cal-hour-dow", HS), ("DLinear_ETTh1_L336_H{H}", HS),
                ("iTransformer_ETTh1_L96_H{H}", HS), ("Autoformer_ETTh1_L96_H{H}", HS), ("DLinear_ONZ_real_L96_H{H}", (96, 720)),
                ("PatchTST_ONZ_real_L96_H{H}", (96, 720)), ("PatchTST_ONZ_real_L96_H{H}_cal-hour-dow-hol", (96,)),
                ("iTransformer_ONZ_real_L96_H{H}", (96, 720)), ("iTransformer_ONZ_real_L96_H{H}_nots", (96,)),
                ("iTransformer_ONZ_real_L96_H{H}_ci", (96,))):
    for H in Hs:
        c = pat.format(H=H)
        a, b = TM[c][1], TM[c][32]
        assert a["params"] == b["params"]
        FULL[f"timing.{c}"] = (f"{a['params']} parameters; median {float(a['median_ms']):.3f} ms (batch 1), "
                               f"{float(b['median_ms']):.3f} ms (batch 32) | -- | timing.py forward pass, {dev}, torch {tv} | --")
for a, b in (("PatchTST_ETTh1_L336_H96_cal-hour-dow", "PatchTST_ETTh1_L336_H96"),
             ("PatchTST_ONZ_real_L96_H96_cal-hour-dow-hol", "PatchTST_ONZ_real_L96_H96")):
    FULL[f"timing.params_added.{a}"] = f"{int(TM[a][1]['params']) - int(TM[b][1]['params']):+d} | -- | parameters of {a} - {b} | --"

# Pareto sets of fig_mse_vs_latency (compared as sets of labels)
SETS = {}
pts = {"ETTh1": [], "ONZ": []}
for P in (8, 16, 32, 64):
    for L in (96, 336, 720):
        pts["ETTh1"].append((f"PatchTST P{P} L{L}", f"PatchTST_ETTh1_L{L}_H336" + ("" if P == 16 else f"_P{P}S{P // 2}"),
                             lambda s, P=P, L=L: every("PatchTST", "ETTh1", L, 336, s, P=P)))
pts["ETTh1"] += [("PatchTST+Cal L336", "PatchTST_ETTh1_L336_H336_cal-hour-dow", lambda s: every("PatchTST", "ETTh1", 336, 336, s, var="cal-hour-dow")),
                 ("iTransformer L96", "iTransformer_ETTh1_L96_H336", lambda s: every("iTransformer", "ETTh1", 96, 336, s)),
                 ("iTransformer L336", "iTransformer_ETTh1_L336_H336", lambda s: every("iTransformer", "ETTh1", 336, 336, s)),
                 ("DLinear L336", "DLinear_ETTh1_L336_H336", lambda s: every("DLinear", "ETTh1", 336, 336, s)),
                 ("Autoformer L96", "Autoformer_ETTh1_L96_H336", lambda s: every("Autoformer", "ETTh1", 96, 336, s))]
for m, v, lbl in (("DLinear", "", "DLinear"), ("PatchTST", "", "PatchTST"), ("iTransformer", "", "iTransformer"),
                  ("iTransformer", "nots", "iTransformer, no tokens"), ("iTransformer", "ci", "Control 2"),
                  ("PatchTST", "cal-hour-dow-hol", "PatchTST+Cal full"), ("PatchTST", "cal-dow-hol", "PatchTST+Cal -hour"),
                  ("PatchTST", "cal-hour-hol", "PatchTST+Cal -dow"), ("PatchTST", "cal-hour-dow", "PatchTST+Cal -holiday")):
    pts["ONZ"].append((lbl, f"{m}_ONZ_real_L96_H96" + (f"_{v}" if v else ""), lambda s, m=m, v=v: every(m, "ONZ_real", 96, 96, s, var=v)))
for ds, ps in pts.items():
    sd_p = seeds_where(*[g for _, _, g in ps])
    for b in (1, 32):
        q = [(lbl, float(TM[c][b]["median_ms"]), mean([g(s) for s in sd_p])) for lbl, c, g in ps]
        front = {l for l, x, y in q if not any(x2 <= x and y2 <= y and (x2 < x or y2 < y) for l2, x2, y2 in q if l2 != l)}
        SETS[f"fig_mse_vs_latency.{ds}.pareto_b{b}"] = (front, S(sd_p))

# ------------------------------------------------------------------ job K aggregates (results/preds_analysis/)
# Read with the csv module; every predsK line is rebuilt here from the CSV values (value field compared).
PKD = RES / "preds_analysis"
PKM = ["iTransformer", "PatchTST", "DLinear", "Seasonal naive"]
PKS = "seed 2021 (n=1); one run per model, descriptive"
PKV = {}


def pkcsv(name):
    with open(PKD / f"{name}.csv") as fh:
        return list(csv.DictReader(fh))


def pf(metric, v):
    v = float(v)
    return f"{v:.1f}" if metric == "mae_MW" else (f"{v:.2f}" if metric == "mape_pct" else f"{v:.4f}")


PK_OV = {r["model"]: r for r in pkcsv("overall")}
PK_Z = pkcsv("zone_errors")
PK_ST = pkcsv("error_by_step")
PK_HV = {(r["model"], r["group"]): r for r in pkcsv("holiday_vs_rest")}
PK_PK = {(r["model"], r["group"]): r for r in pkcsv("peak_vs_rest") if r["zone"] == "all zones"}
PK_WW = pkcsv("worst_windows")
PK_EX = pkcsv("examples")
PK_HD = pkcsv("holiday_dates")
CK = json.loads((PKD / "checks.json").read_text())
PKV["predsK.checks.max_abs_target_minus_rebuilt_z"] = f"{max(CK[m + '.max_abs_true_minus_rebuilt_z'] for m in PKM[:3]):.1e}"
for m in PKM[:3]:
    PKV[f"predsK.checks.{m}.mse_reproduced"] = f"{CK[m + '.mse_from_arrays']:.6f} (logged {CK[m + '.mse_logged']:.6f})"
PKV["predsK.checks.rebuild"] = (f"rows {CK['rows']}, test {CK['test_period'][0]} to {CK['test_period'][1]}, {CK['n_windows']} windows; "
                                "split and seasonal naive equal the committed files")
for m in PKM:
    for k in ("mse_z", "mae_z", "mae_MW", "mape_pct"):
        PKV[f"predsK.overall.{m}.{k}"] = pf(k, PK_OV[m][k])
    zr = [r for r in PK_Z if r["model"] == m]
    for r in zr:
        for k in ("mse_z", "mae_MW", "mape_pct"):
            PKV[f"predsK.zone.{m}.{r['zone']}.{k}"] = pf(k, r[k])
    hi = max(zr, key=lambda r: float(r["mape_pct"]))
    lo = min(zr, key=lambda r: float(r["mape_pct"]))
    hm = max(zr, key=lambda r: float(r["mse_z"]))
    PKV[f"predsK.zone.{m}.highest_mape"] = f"{hi['zone']} ({float(hi['mape_pct']):.2f}%)"
    PKV[f"predsK.zone.{m}.lowest_mape"] = f"{lo['zone']} ({float(lo['mape_pct']):.2f}%)"
    PKV[f"predsK.zone.{m}.highest_mse"] = f"{hm['zone']} ({float(hm['mse_z']):.4f})"
    for h in (1, 24, 48, 96):
        r = next(r for r in PK_ST if r["model"] == m and int(r["h"]) == h)
        for k in ("mse_z", "mape_pct"):
            PKV[f"predsK.step.{m}.h{h}.{k}"] = pf(k, r[k])
stp = {(r["model"], int(r["h"])): float(r["mse_z"]) for r in PK_ST}
low = {h: min(PKM[:3], key=lambda m: stp[(m, h)]) for h in range(1, 97)}
PKV["predsK.step.lowest_mse_model.h1"] = low[1]
PKV["predsK.step.iTransformer_lowest_from_h"] = next((str(h) for h in range(1, 97) if all(low[k] == "iTransformer" for k in range(h, 97))), "never")
for name, tb, pairs in (("holiday", PK_HV, (("holiday target hours", "other target hours", "hours"), ("windows touching a holiday", "other windows", "windows"))),
                        ("peak", PK_PK, (("peak target hours", "other target hours", "hours"), ("windows containing a peak hour", "other windows", "windows")))):
    for g1, g2, kind in pairs:
        PKV[f"predsK.{name}.{kind}.n"] = f"{tb[('iTransformer', g1)]['n']} vs {tb[('iTransformer', g2)]['n']}"
        for m in PKM:
            for k in ("mse_z", "mae_MW", "mape_pct"):
                PKV[f"predsK.{name}.{kind}.{m}.{k}"] = f"{pf(k, tb[(m, g1)][k])} vs {pf(k, tb[(m, g2)][k])}"
            PKV[f"predsK.{name}.{kind}.{m}.mse_ratio"] = f"{float(tb[(m, g1)]['mse_z']) / float(tb[(m, g2)]['mse_z']):.2f}"
w3 = sorted([r for r in PK_WW if r["ranking"] == "three-model mean"], key=lambda r: int(r["rank"]))
yn = {"True": "yes", "False": "no"}
for r in w3:
    PKV[f"predsK.worst.three_model_mean.rank{r['rank']}"] = (
        f"origin {r['forecast_origin'][:16]}; MSE {float(r['window_mse_z']):.4f} (iTransformer {float(r['mse_z_iTransformer']):.4f}, "
        f"PatchTST {float(r['mse_z_PatchTST']):.4f}, DLinear {float(r['mse_z_DLinear']):.4f}); worst zone {r['worst_zone']}; "
        f"holiday {yn[r['touches_holiday']]}; peak {yn[r['contains_peak_hour']]}")
summer = sum(1 for r in w3 if 6 <= int(r["forecast_origin"][5:7]) <= 9)
PKV["predsK.worst.three_model_mean.summary"] = (f"{summer} of {len(w3)} start in June-September; "
                                                f"{sum(r['contains_peak_hour'] == 'True' for r in w3)} contain a peak hour; "
                                                f"{sum(r['touches_holiday'] == 'True' for r in w3)} touch a holiday")
for m in PKM[:3]:
    r = next(r for r in PK_WW if r["ranking"] == m and r["rank"] == "1")
    PKV[f"predsK.worst.{m}.rank1"] = f"origin {r['forecast_origin'][:16]}; MSE {float(r['window_mse_z']):.4f}; worst zone {r['worst_zone']}"
for r in PK_EX:
    PKV[f"predsK.example.{r['panel']}"] = f"origin {r['forecast_origin'][:16]}; Toronto MSE " + ", ".join(
        f"{m} {float(r[f'mse_z_{m}_Toronto']):.3f}" for m in PKM[:3])
PKV["predsK.holiday_dates"] = f"{len(PK_HD)} dates, {PK_HD[0]['date']} to {PK_HD[-1]['date']}"

# ------------------------------------------------------------------ compare
lines = [l for l in (RES / "report_numbers.txt").read_text().splitlines() if l and not l.startswith("#")]
bad, seen = [], set()
for line in lines:
    parts = line.split(" | ")
    if len(parts) != 5:
        bad.append(f"malformed: {line}")
        continue
    key, value, ci, what, seeds = parts
    seen.add(key)
    if key in FULL:
        if " | ".join(parts[1:]) != FULL[key]:
            bad.append(f"{key}: file {' | '.join(parts[1:])!r} != recomputed {FULL[key]!r}")
    elif key in SETS:
        got = set(value.split(", "))
        if got != SETS[key][0] or seeds != SETS[key][1]:
            bad.append(f"{key}: file {sorted(got)} {seeds} != recomputed {sorted(SETS[key][0])} {SETS[key][1]}")
    elif key in PKV:
        if (value, ci, seeds) != (PKV[key], "--", PKS):
            bad.append(f"{key}: file {value!r} ({seeds}) != recomputed {PKV[key]!r} ({PKS})")
    elif key in EXP:
        e = EXP[key]
        p = re.search(r"p(?:=|<)[0-9.na]+$", what)
        got = dict(value=value, ci=ci, seeds=seeds)
        if "p" in e:
            got["p"] = p.group(0) if p else "(none)"
        diff = {k: (got[k], e[k]) for k in e if got[k] != e[k]}
        if diff:
            bad.append(f"{key}: " + "; ".join(f"{k}: file {a!r} != recomputed {b!r}" for k, (a, b) in diff.items()))
    else:
        bad.append(f"{key}: not recomputed by this script")
# ------------------------------------------------------------------ tables, number by number
# The numbers of each new table, in reading order (row by row, left to right, CI lines included),
# recomputed here and formatted as printed; compared with the numbers found in tables/<name>.tex.
def u(v, d):
    return f"{v:.{d}f}"


def sg(v, d):
    return f"{v:+.{d}f}"


def ms_(x, d):
    return [u(mean(x), d), u(sd(x), d)]


def res_(r, d=4):
    return [sg(r[0], d), sg(r[1], d), sg(r[2], d)]


SEQ = {}
q = SEQ.setdefault("main_ETTh1", [])
for H in HS:
    for j, m in enumerate(("mse", "mae")):
        pt = [off("PatchTST", "ETTh1", 336, H, s, j=j) for s in sd_a]
        cal = [off("PatchTST", "ETTh1", 336, H, s, var="cal-hour-dow", j=j) for s in sd_a]
        q += [u(float(PUB[(n, "ETTh1", H)][m]), 3) for n in ("PatchTST", "DLinear", "iTransformer")]
        q += ms_(pt, 3) + ms_(cal, 3) + res_(paired(pt, cal))
q = SEQ.setdefault("main_ONZ", [])
q += [u(float(SN[H][m]), 4) for H in (96, 720) for m in ("mse", "mae")]
for mod, v, _ in ONZ96:
    for H in ((96, 720) if v == "" else (96,)):
        for j in (0, 1):
            q += ms_([every(mod, "ONZ_real", 96, H, s, var=v, j=j) for s in sd_b], 4)
for a, b, test, Hs in ((("PatchTST", ""), ("PatchTST", "cal-hour-dow-hol"), paired, (96,)),
                       (("iTransformer", ""), ("PatchTST", "cal-hour-dow-hol"), welch, (96,)),
                       (("iTransformer", ""), ("PatchTST", ""), welch, (96, 720))):
    for H in Hs:
        q += res_(test([every(a[0], "ONZ_real", 96, H, s, var=a[1]) for s in sd_b],
                       [every(b[0], "ONZ_real", 96, H, s, var=b[1]) for s in sd_b]))
q = SEQ.setdefault("repro_ETTm1", [])
for mod, L in CFG:
    for H in HS:
        o = [off(mod, "ETTm1", L, H, s) for s in sd_c]
        pub = PUB[(mod, "ETTm1", H)]
        q += [u(float(pub["mse"]), 3), u(float(pub["mae"]), 3)] + ms_(o, 3) + ms_([off(mod, "ETTm1", L, H, s, j=1) for s in sd_c], 3)
        q += [sg(100 * (mean(o) - float(pub["mse"])) / float(pub["mse"]), 1)]
        r0 = RUNS[rid(mod, "ETTm1", L, H, sd_c[0])]
        if r0["n_off"] < r0["n_test"]:
            q += res_(paired(o, [every(mod, "ETTm1", L, H, s) for s in sd_c]))
q = SEQ.setdefault("rq1_decomposition", [])
RQ1 = {}
for H in HS:
    c = lambda fn, *a, **k: [fn(*a, s, **k) for s in sd_d]  # noqa: E731
    p3o, p3a = c(off, "PatchTST", "ETTh1", 336, H), c(every, "PatchTST", "ETTh1", 336, H)
    p9o, p9a = c(off, "PatchTST", "ETTh1", 96, H), c(every, "PatchTST", "ETTh1", 96, H)
    i9, i3 = c(every, "iTransformer", "ETTh1", 96, H), c(every, "iTransformer", "ETTh1", 336, H)
    psel = c(every, "PatchTST", "ETTh1", 96, H, var=lrvar(WIN[("PatchTST", H)][0]))
    isel = c(every, "iTransformer", "ETTh1", 96, H, var=lrvar(WIN[("iTransformer", H)][0]))
    RQ1[H] = dict(G0=welch(p3o, i9), G1=welch(p3a, i9), G2=welch(p9a, i9), G3=welch(psel, isel), G2alt=welch(p3a, i3),
                  C1=paired(p3o, p3a), C2=paired(p3a, p9a), C3=welch([x - y for x, y in zip(p9a, psel)], [x - y for x, y in zip(i9, isel)]),
                  C2alt=paired(i3, i9), phi1=tint([0.5 * ((a - b) + (cc - dd)) for a, b, cc, dd in zip(p3a, p3o, p9a, p9o)]),
                  phi2=tint([0.5 * ((a - b) + (cc - dd)) for a, b, cc, dd in zip(p9o, p3o, p9a, p3a)]), G11=welch(p9a, i9),
                  phi1alt=tint([a - b for a, b in zip(p3a, p3o)]), phi2alt=tint([a - b for a, b in zip(i9, i3)]), G11alt=welch(p3a, i3))
q += [sg(float(PUB[("iTransformer", "ETTh1", H)]["mse"]) - float(PUB[("PatchTST", "ETTh1", H)]["mse"]), 3) for H in HS]
for k in ("G0", "G1", "G2", "G3", "G2alt", "C1", "C2", "C3", "C2alt", "phi1", "phi2", "G11", "phi1alt", "phi2alt", "G11alt"):
    q += [sg(RQ1[H][k][0], 4) for H in HS]
    q += [x for H in HS for x in (sg(RQ1[H][k][1], 4), sg(RQ1[H][k][2], 4))]
q = SEQ.setdefault("f3_selection", [])
for mod in ("PatchTST", "iTransformer"):
    for H in HS:
        lr, losses = WIN[(mod, H)]
        d = [every(mod, "ETTh1", 96, H, s) for s in sd_d]
        q += [u(losses[x], 4) for x in (3e-5, 1e-4, 3e-4)] + ms_(d, 4)
        if lr != 1e-4:
            x = [every(mod, "ETTh1", 96, H, s, var=lrvar(lr)) for s in sd_d]
            q += ms_(x, 4) + res_(paired(d, x))
q = SEQ.setdefault("rq2_final", [])
for H in (96, 720):
    for mv in ((IT, NOTS, PT, DL, CI) if H == 96 else (IT, PT, DL)):
        sd_ = pair_seeds(H, mv)
        real = [every(mv[0], "ONZ_real", 96, H, s, var=mv[1]) for s in sd_]
        shift = [every(mv[0], "ONZ_shift", 96, H, s, var=mv[1]) for s in sd_]
        q += ms_(real, 4) + ms_(shift, 4) + res_(paired(real, shift))
    for a, b in [(IT, PT), (IT, DL)] + ([(IT, CI), (NOTS, PT), (NOTS, DL)] if H == 96 else []):
        sd_ = pair_seeds(H, a, b)
        q += res_(welch(eff(H, b, sd_), eff(H, a, sd_)))
    for a, b in [(PT, DL)] + ([(CI, PT), (CI, DL)] if H == 96 else []):
        sd_ = pair_seeds(H, a, b)
        q += res_(welch(eff(H, b, sd_), eff(H, a, sd_)))
    if H == 96:
        sd_ = pair_seeds(96, CI, IT)
        for ds in ("ONZ_real", "ONZ_shift"):
            q += res_(welch([every("iTransformer", ds, 96, 96, s) for s in sd_], [every("iTransformer", ds, 96, 96, s, var="ci") for s in sd_]))
        sd_ = pair_seeds(96, IT, NOTS)
        q += res_(paired([every("iTransformer", "ONZ_real", 96, 96, s, var="nots") for s in sd_],
                         [every("iTransformer", "ONZ_real", 96, 96, s) for s in sd_]))
q = SEQ.setdefault("ablation_cal_ONZ", [])
pt = [every("PatchTST", "ONZ_real", 96, 96, s) for s in sd_f]
full = [every("PatchTST", "ONZ_real", 96, 96, s, var="cal-hour-dow-hol") for s in sd_f]
for name, v in CALV:
    x = [every("PatchTST", "ONZ_real", 96, 96, s, var=v) for s in sd_f]
    a = paired(pt, x) if v else None
    b = paired(full, x) if v != "cal-hour-dow-hol" else None
    q += ms_(x, 4) + ([sg(a[0], 4)] if a else []) + ([sg(b[0], 4)] if b else [])
    q += ([sg(a[1], 4), sg(a[2], 4)] if a else []) + ([sg(b[1], 4), sg(b[2], 4)] if b else [])
q = SEQ.setdefault("grid_PxL_H336", [])
for kind, fn in (("all", every), ("official", off)):
    sd_g = seeds_where(*[lambda s, P=P, L=L: fn("PatchTST", "ETTh1", L, 336, s, P=P) for P in (8, 16, 32, 64) for L in (96, 336, 720)])
    for P in (8, 16, 32, 64):
        for L in (96, 336, 720):
            q += ms_([fn("PatchTST", "ETTh1", L, 336, s, P=P) for s in sd_g], 4)
q = SEQ.setdefault("env_per_job", [])
for j in jobs:
    rows = []
    if (RES / f"runs_{j}.csv").exists():
        with open(RES / f"runs_{j}.csv") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("mse")]
    if rows:
        q.append(u(sum(float(r["seconds"]) for r in rows) / 3600, 2))
    if (RES / f"env_{j}.txt").exists():
        e = envinfo(RES / f"env_{j}.txt")
        q += [e[4].split("+")[0], e[5]]
q = SEQ.setdefault("timing_main", [])
for pat, Hs in (("PatchTST_ETTh1_L336_H{H}", HS), ("PatchTST_ETTh1_L336_H{H}_cal-hour-dow", HS), ("DLinear_ETTh1_L336_H{H}", HS),
                ("iTransformer_ETTh1_L96_H{H}", HS), ("Autoformer_ETTh1_L96_H{H}", HS), ("DLinear_ONZ_real_L96_H{H}", (96, 720)),
                ("PatchTST_ONZ_real_L96_H{H}", (96, 720)), ("PatchTST_ONZ_real_L96_H{H}_cal-hour-dow-hol", (96,)),
                ("iTransformer_ONZ_real_L96_H{H}", (96, 720)), ("iTransformer_ONZ_real_L96_H{H}_nots", (96,)),
                ("iTransformer_ONZ_real_L96_H{H}_ci", (96,))):
    for H in Hs:
        q += [u(float(TM[pat.format(H=H)][b]["median_ms"]), 2) for b in (1, 32)]

# job K tables: MSE (4 decimals) and MAPE (2 decimals) per model, in reading order; worst windows: four MSEs per row
PKROWS = [("all", PK_OV)] + [(f"h{h}", {m: next(r for r in PK_ST if r["model"] == m and int(r["h"]) == h) for m in PKM}) for h in (1, 24, 96)]
PKROWS += [(g, {m: PK_HV[(m, g)] for m in PKM}) for g in ("holiday target hours", "other target hours", "windows touching a holiday", "other windows")]
PKROWS += [(g, {m: PK_PK[(m, g)] for m in PKM}) for g in ("peak target hours", "other target hours", "windows containing a peak hour", "other windows")]
SEQ["predsK_failure"] = [x for _, rm in PKROWS for m in PKM for x in (u(float(rm[m]["mse_z"]), 4), u(float(rm[m]["mape_pct"]), 2))]
zones_ = list(dict.fromkeys(r["zone"] for r in PK_Z))
SEQ["predsK_zones"] = [x for z in zones_ for m in PKM for r in [next(r for r in PK_Z if r["model"] == m and r["zone"] == z)]
                       for x in (u(float(r["mse_z"]), 4), u(float(r["mape_pct"]), 2))]
SEQ["predsK_zones"] += [x for m in PKM for x in (u(float(PK_OV[m]["mse_z"]), 4), u(float(PK_OV[m]["mape_pct"]), 2))]
SEQ["predsK_worst"] = [u(float(r[k]), 4) for r in w3 for k in ("window_mse_z", "mse_z_iTransformer", "mse_z_PatchTST", "mse_z_DLinear")]
ncol = [l.split("&")[1].strip().replace("{,}", "") for l in (KIT / "tables" / "predsK_failure.tex").read_text().split("\\midrule", 1)[1]
        .split("\\bottomrule")[0].splitlines() if "&" in l]
n_win_ = int(PK_OV["iTransformer"]["n_cells"]) // 960
want_n = [str(n_win_)] * 4 + [PK_HV[("iTransformer", g)]["n"] for g in ("holiday target hours", "other target hours", "windows touching a holiday", "other windows")]
want_n += [PK_PK[("iTransformer", g)]["n"] for g in ("peak target hours", "other target hours", "windows containing a peak hour", "other windows")]
if ncol != want_n:
    bad.append(f"tables/predsK_failure.tex: counts {ncol} != {want_n}")

NUM = re.compile(r"(\$-\$|\+)?(?<![\d.])(\d+\.\d+(?:\.\d+)?)(?![\d.])")  # decimals and versions (x.y.z)
n_tok = 0
for name, want in SEQ.items():
    tex = (KIT / "tables" / f"{name}.tex").read_text()
    body = tex[tex.index("\\midrule"):tex.index("\\bottomrule")]
    got = [{"$-$": "-"}.get(t.group(1), t.group(1) or "") + t.group(2) for t in NUM.finditer(body)]
    n_tok += len(got)
    if got != want:
        k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        bad.append(f"tables/{name}.tex: number {k + 1} of {len(got)} is {got[k] if k < len(got) else '(none)'!r}, "
                   f"recomputed {want[k] if k < len(want) else '(none)'!r} (expected {len(want)} numbers)")
print(f"tables: {n_tok} numbers in {len(SEQ)} new tables compared one by one with the recomputation")
missing = sorted((set(EXP) | set(FULL) | set(SETS) | set(PKV)) - seen)
for k in missing:
    bad.append(f"{k}: recomputed here but missing from report_numbers.txt")
print(f"report_numbers.txt: {len(lines)} lines; recomputed independently: {len(seen & (set(EXP) | set(FULL) | set(SETS) | set(PKV)))}")
for b in bad:
    print("MISMATCH", b)
print("OVERALL:", "PASS" if not bad else f"FAIL ({len(bad)} problems)")
sys.exit(0 if not bad else 1)
