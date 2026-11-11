"""C4 acceptance test: channel-independent iTransformer (P24_ITR_CI=1), CPU.

    python tests/test_itr_ci.py REPOS

Checks the shapes ([B,L,N] -> [B*N,L,1] -> [B,S,N]), that no parameter is added and no random
number is drawn, that calendar tokens are kept, and that the CI model equals the unmodified
model run on one variate at a time (so no information crosses variates).
"""
import argparse
import os
import sys
import textwrap
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import p24  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("repos")
A = ap.parse_args()
REPO = Path(A.repos) / "iTransformer"
os.chdir(REPO)
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
from model import iTransformer  # noqa: E402

src = (REPO / "run.py").read_text()
i, j = src.index("parser = argparse.ArgumentParser"), src.index("args = parser.parse_args()")
ns = {"argparse": argparse}
exec(textwrap.dedent(src[src.rfind("\n", 0, i) + 1:j]), ns)
OK = []


def check(name, ok, detail=""):
    OK.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def build(args, ci):
    os.environ.pop("P24_ITR_CI", None)
    if ci:
        os.environ["P24_ITR_CI"] = "1"
    torch.manual_seed(2021)
    m = iTransformer.Model(args).float().eval()
    return m, torch.get_rng_state()


for ds, L, H in (("ONZ_real", 96, 96), ("ONZ_real", 96, 720), ("ETTh1", 96, 96)):
    args = ns["parser"].parse_args(p24.model_args("iTransformer", ds, L, H, 2021)[1:])
    B, N, M = 4, args.enc_in, 4
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, L, N, generator=g)
    xm = torch.rand(B, L, M, generator=g) - 0.5
    dec = torch.zeros(B, 48 + H, N)
    ym = torch.rand(B, 48 + H, M, generator=g) - 0.5
    m0, r0 = build(args, False)
    m1, r1 = build(args, True)
    print(f"\n## {ds} L={L} H={H}: B={B}, N={N} variates, {M} calendar tokens, d_model={args.d_model}")
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    check("same parameters (none added), same values, same RNG state after build",
          sd0.keys() == sd1.keys() and all(torch.equal(sd0[k], sd1[k]) for k in sd0) and torch.equal(r0, r1),
          f"{sum(p.numel() for p in m1.parameters())} parameters")
    seen = {}
    h = m1.encoder.attn_layers[0].attention.register_forward_hook(lambda mod, i, o: seen.update(q=i[0].shape))  # returns None
    with torch.no_grad():
        y0 = m0(x, xm, dec, ym)
        y1 = m1(x, xm, dec, ym)
    h.remove()
    check(f"output shape {tuple(y1.shape)} == [B, H, N] (same as the original {tuple(y0.shape)})",
          tuple(y1.shape) == (B, H, N) == tuple(y0.shape))
    check(f"attention input {tuple(seen['q'])} == [B*N, 1 + calendar tokens, d_model]",
          tuple(seen["q"]) == (B * N, 1 + M, args.d_model))
    # reference: the unmodified model applied to one variate at a time
    with torch.no_grad():
        ref = torch.cat([m0(x[:, :, n:n + 1], xm, dec[:, :, n:n + 1], ym) for n in range(N)], dim=2)
    diff = float((ref - y1).abs().max())
    check("CI output == original model run on each variate separately (no cross-variate information)",
          torch.allclose(ref, y1, atol=1e-6, rtol=0), f"max |diff| {diff:.1e}")
    # cross-variate leakage: perturb variate 0 only; with CI, other variates' forecasts must not move
    x2 = x.clone()
    x2[:, :, 0] += 5.0
    with torch.no_grad():
        d_ci = float((m1(x2, xm, dec, ym) - y1)[:, :, 1:].abs().max())
        d_full = float((m0(x2, xm, dec, ym) - y0)[:, :, 1:].abs().max())
    check("perturbing variate 0 leaves variates 1..N-1 unchanged with CI (it does change them without CI)",
          d_ci == 0.0 and d_full > 0, f"max change of other variates: CI {d_ci:.1e}, original {d_full:.1e}")
    with torch.no_grad():
        y_nots = m1(x, None, dec, ym)
    check("CI also runs without calendar tokens (x_mark None)", tuple(y_nots.shape) == (B, H, N))

print("\nOVERALL:", "PASS" if all(OK) else "FAIL", f"({sum(OK)}/{len(OK)} checks)")
sys.exit(0 if all(OK) else 1)
