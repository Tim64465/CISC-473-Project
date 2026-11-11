"""C3 acceptance tests for PatchTST+Cal (CPU).

    python tests/test_cal.py REPOS [ONZ_CSV]

REPOS: repos patched by this branch's patch_repos.py. ONZ_CSV: an ON-Zonal-format csv
(date + 10 zones, hourly) for the holiday test; tests/make_smoke_data.py writes a synthetic one.
"""
import argparse
import os
import sys
import textwrap
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import p24  # noqa: E402

a = argparse.ArgumentParser()
a.add_argument("repos")
a.add_argument("onz_csv", nargs="?")
A = a.parse_args()
REPO = Path(A.repos) / "PatchTST" / "PatchTST_supervised"
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from models import PatchTST  # noqa: E402
from utils.timefeatures import time_features  # noqa: E402
from layers.P24_calendar import P24CalendarPE  # noqa: E402

src = (REPO / "run_longExp.py").read_text()
i, j = src.index("parser = argparse.ArgumentParser"), src.index("args = parser.parse_args()")
ns = {"argparse": argparse}
exec(textwrap.dedent(src[src.rfind("\n", 0, i) + 1:j]), ns)
parser = ns["parser"]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def args_for(ds, L, H, flags=(), P=16):
    argv = p24.model_args("PatchTST", ds, L, H, 2021, P)[1:] + [f"--cal_{f}" for f in flags]
    return parser.parse_args(argv)


def build(args, seed=2021):
    torch.manual_seed(seed)
    m = PatchTST.Model(args).float()
    return m, torch.get_rng_state()


def realistic_mark(B, L, start="2024-12-20", freq="h", holiday=False):
    """x_mark exactly as the data loader builds it (timeF), random window starts."""
    idx = pd.date_range(start, periods=24 * 60, freq=freq)
    tf = time_features(idx, freq="h").T.astype(np.float32)
    if holiday:
        import holidays
        hol = holidays.Canada(subdiv="ON", years=sorted(set(idx.year)))
        tf = np.concatenate([tf, np.array([[d.date() in hol] for d in idx], dtype=np.float32)], 1)
    g = np.random.default_rng(0)
    st = g.integers(0, len(idx) - L, B)
    return torch.tensor(np.stack([tf[s:s + L] for s in st]))


print("## (a) flags off -> the namespace and the code path are unchanged")
args0 = args_for("ETTh1", 336, 96)
check("no cal_* attribute in args without flags (argparse.SUPPRESS)", not any(hasattr(args0, f"cal_{k}") for k in ("hour", "dow", "holiday")))
m0, rng0 = build(args0)
check("model.p24_cal is None without flags", m0.p24_cal is None)
check("no new parameter or buffer without flags",
      [k for k, _ in m0.state_dict().items() if "p24" in k] == [])

print("\n## RNG: building the calendar embeddings draws no random number")
for ds, L, flags in (("ETTh1", 336, ("hour", "dow")), ("ONZ_real", 96, ("hour", "dow", "holiday"))):
    a0, a1 = args_for(ds, L, 96), args_for(ds, L, 96, flags)
    b0, r0 = build(a0)
    b1, r1 = build(a1)
    sd0, sd1 = b0.state_dict(), b1.state_dict()
    base_same = all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    new = {k: v for k, v in sd1.items() if k not in sd0}
    check(f"{ds} {'+'.join(flags)}: baseline parameters identical, global RNG state identical after build",
          base_same and torch.equal(r0, r1), f"new tensors: {sorted(new)}")
    check(f"{ds} {'+'.join(flags)}: new embeddings are exactly zero", all(bool((v == 0).all()) for v in new.values()))
    added = sum(p.numel() for p in b1.parameters()) - sum(p.numel() for p in b0.parameters())
    d = a1.d_model
    expect = d * (24 * ("hour" in flags) + 7 * ("dow" in flags) + 2 * ("holiday" in flags))
    check(f"(c) {ds} d_model={d} {'+'.join(flags)}: added parameters = {added} "
          f"(= d_model x ({' + '.join(str(n) for n, f in ((24, 'hour'), (7, 'dow'), (2, 'holiday')) if f in flags)}))"
          f"; baseline {sum(p.numel() for p in b0.parameters())}", added == expect)

print("\n## (c) added parameters for every planned PatchTST+Cal config")
for ds, L, flags in [("ETTh1", 336, ("hour", "dow"))] + [("ONZ_real", 96, c) for c in p24.CAL_ABLATION]:
    a0, a1 = args_for(ds, L, 96), args_for(ds, L, 96, flags)
    n0 = sum(p.numel() for p in build(a0)[0].parameters())
    n1 = sum(p.numel() for p in build(a1)[0].parameters())
    print(f"    {ds:8s} L={L} H=96 cal={'+'.join(flags):18s} baseline {n0:>7d}  +Cal {n1:>7d}  added {n1 - n0:>5d}")

print("\n## (b) flags on, embeddings at zero -> outputs identical to the baseline")
for ds, L, flags, hol in (("ETTh1", 336, ("hour", "dow"), False), ("ONZ_real", 96, ("hour", "dow", "holiday"), True)):
    a0, a1 = args_for(ds, L, 96), args_for(ds, L, 96, flags)
    b0, _ = build(a0)
    b1, _ = build(a1)
    x = torch.randn(32, L, a0.enc_in, generator=torch.Generator().manual_seed(1))
    xm = realistic_mark(32, L, holiday=hol)
    for mode in ("eval", "train"):
        getattr(b0, mode)(), getattr(b1, mode)()
        torch.manual_seed(7)
        y0 = b0(x)
        torch.manual_seed(7)  # same dropout masks
        y1 = b1(x, xm)
        check(f"{ds} {'+'.join(flags)} {mode} mode: forward output bitwise identical", torch.equal(y0, y1))
    # five Adam steps with the embeddings frozen at zero: every baseline parameter stays bitwise equal
    os.environ["P24_CAL_FREEZE"] = "1"
    b2, _ = build(a1)
    os.environ.pop("P24_CAL_FREEZE")
    b0, _ = build(a0)
    o0 = torch.optim.Adam(b0.parameters(), lr=1e-3)
    o2 = torch.optim.Adam(b2.parameters(), lr=1e-3)
    y = torch.randn(32, 96, a0.enc_in, generator=torch.Generator().manual_seed(2))
    for step in range(5):
        for m, o, extra in ((b0, o0, ()), (b2, o2, (xm,))):
            m.train()
            o.zero_grad()
            torch.manual_seed(100 + step)
            torch.nn.functional.mse_loss(m(x, *extra), y).backward()
            o.step()
    sd0, sd2 = b0.state_dict(), b2.state_dict()
    check(f"{ds}: 5 training steps, embeddings frozen at 0 -> all baseline parameters bitwise identical",
          all(torch.equal(sd0[k], sd2[k]) for k in sd0) and all(bool((v == 0).all()) for k, v in sd2.items() if "p24" in k))
    # not frozen: the embeddings receive gradients and move away from zero
    b3, _ = build(a1)
    b3.train()
    torch.nn.functional.mse_loss(b3(x, xm), y).backward()
    g = {n: float(p.grad.abs().sum()) for n, p in b3.named_parameters() if "p24" in n}
    check(f"{ds}: trainable embeddings get non-zero gradients", all(v > 0 for v in g.values()), str(g))

print("\n## hour and day of week recovered from the timeF features")
for name, idx in (("hourly, 2 years", pd.date_range("2023-01-01", "2024-12-31 23:00", freq="h")),
                  ("15-min (ETTm1-style; PatchTST runs ETTm1 with --freq h)", pd.date_range("2016-07-01", periods=4 * 24 * 14, freq="15min"))):
    tf = torch.tensor(time_features(idx, freq="h").T.astype(np.float32))  # float32, as exp_main feeds the model
    h = P24CalendarPE.hour_index(tf[:, 0]).numpy()
    d = P24CalendarPE.dow_index(tf[:, 1]).numpy()
    check(f"{name}: hour = round((x0 + 0.5) * 23), dow = round((x1 + 0.5) * 6) exact for all {len(idx)} timestamps",
          (h == idx.hour.values).all() and (d == idx.dayofweek.values).all())

print("\n## each patch is indexed by its first time step")
for L, P in ((96, 16), (336, 16), (720, 16), (96, 64), (336, 8), (720, 64)):
    ar = args_for("ETTh1", L, 96, ("hour",), P=P)
    m, _ = build(ar)
    pe, bb = m.p24_cal, m.model
    z = torch.arange(L, dtype=torch.float32).view(1, 1, L)
    z = bb.padding_patch_layer(z) if bb.padding_patch == "end" else z
    first = z.unfold(-1, bb.patch_len, bb.stride)[0, 0, :, 0].long()  # first step of each patch, as PatchTST builds them
    ours = torch.arange(pe.patch_num) * pe.stride
    check(f"L={L} P={P} S={P//2}: {pe.patch_num} patches, first steps {ours[:3].tolist()}...{ours[-1].item()} match unfold",
          torch.equal(first, ours) and int(ours[-1]) < L)

if A.onz_csv:
    print("\n## holiday flag appended to data_stamp by Dataset_Custom (only with --cal_holiday)")
    import shutil
    import holidays
    from data_provider.data_factory import data_provider
    tmp = REPO / "dataset" / "_p24_test_ONZ.csv"
    shutil.copy(A.onz_csv, tmp)
    try:
        base = dict(root_path="./dataset/", data_path=tmp.name, target="West", freq="h")
        for flags in ((), ("holiday",), ("hour", "dow", "holiday")):
            ar = parser.parse_args(p24.model_args("PatchTST", "ONZ_real", 96, 96, 2021)[1:] + [f"--cal_{f}" for f in flags]
                                   + ["--data_path", tmp.name, "--num_workers", "0"])
            ds_, _ = data_provider(ar, "train")
            print(f"    flags {flags or '(none)'}: data_stamp shape {ds_.data_stamp.shape}")
            if "holiday" in flags:
                df = pd.read_csv(tmp)
                n_tr = int(len(df) * 0.7)
                dates = pd.to_datetime(df["date"][:n_tr])
                hol = holidays.Canada(subdiv="ON", years=sorted(set(dates.dt.year)))
                exp = np.array([d.date() in hol for d in dates], dtype=float)
                check(f"holiday column == holidays.Canada(subdiv='ON') for all {n_tr} training rows "
                      f"({int(exp.sum())} holiday hours = {int(exp.sum()) // 24} days)",
                      ds_.data_stamp.shape[1] == 5 and np.array_equal(ds_.data_stamp[:, -1], exp))
            else:
                check("no holiday column without --cal_holiday", ds_.data_stamp.shape[1] == 4)
        yr = sorted(set(pd.to_datetime(pd.read_csv(tmp)["date"]).dt.year))
        for y in yr[:1]:
            print(f"    Ontario holidays in {y} (holidays {holidays.__version__}):",
                  "; ".join(f"{d} {n}" for d, n in sorted(holidays.Canada(subdiv="ON", years=y).items())))
    finally:
        tmp.unlink()

print("\nOVERALL:", "PASS" if all(RESULTS) else "FAIL", f"({sum(RESULTS)}/{len(RESULTS)} checks)")
sys.exit(0 if all(RESULTS) else 1)
