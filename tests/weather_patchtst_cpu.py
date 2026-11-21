"""CPU check of the PatchTST Weather path (plan `weather`), 1 epoch. The official configuration
(batch 128 x 21 channels) needs more RAM than this 8 GB CPU machine has (killed, exit -9, in
tests/output/smoke_all.txt); it is the paper's own configuration and fits a 16 GB T4. Here only
--batch_size is lowered to 16, so the data loader, training loop and dual-scored test run.
    P24_WORK=... python tests/weather_patchtst_cpu.py
"""
import os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p24
spec = dict(plan="weather", model="PatchTST", dataset="weather", L=336, H=96, seed=2021)
rid, repo, args, env = p24.command(spec, smoke=True)
args = p24._set(args, "--batch_size", 16)
print("python", " ".join(args), "|", env)
e = dict(os.environ, SEED="2021", **env)
out = subprocess.run([sys.executable, "-u"] + args, cwd=p24.REPO_DIR[repo], env=e, capture_output=True, text=True)
lines = [l for l in (out.stdout + out.stderr).splitlines() if l.startswith(("train ", "val ", "test ", "Epoch: 1,", "P24_OFFICIAL", "mse:"))]
print("\n".join(lines))
ok = out.returncode == 0 and any(l.startswith("P24_OFFICIAL") for l in lines)
print("OVERALL:", "PASS" if ok else f"FAIL (exit {out.returncode})")
sys.exit(0 if ok else 1)
