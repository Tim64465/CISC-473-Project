"""Compat patch for pandas >= 3 ('t' frequency alias): identical calendar features where both work.

    python tests/test_freq_alias.py OLD_REPOS NEW_REPOS PYTHON_WITH_PANDAS2 [PYTHON_WITH_PANDAS3]

Builds Autoformer's ETTm1 training set exactly as M6 does (Dataset_ETT_minute, timeF, --freq t,
the only configuration that passes 't') with the old-kit and the new-kit repos, under pandas 2
and pandas 3, and compares data_stamp byte for byte. Needs ETTm1.csv in Autoformer/dataset/ETT-small.
"""
import hashlib
import subprocess
import sys

old, new, py2 = sys.argv[1:4]
py3 = sys.argv[4] if len(sys.argv) > 4 else sys.executable
CODE = r"""
import sys, hashlib, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, ".")
import pandas
from data_provider.data_loader import Dataset_ETT_minute
try:
    d = Dataset_ETT_minute("./dataset/ETT-small/", flag="train", size=[96, 48, 96], features="M",
                           data_path="ETTm1.csv", timeenc=1, freq="t")
    print("OK", pandas.__version__, d.data_stamp.shape, hashlib.sha256(d.data_stamp.tobytes()).hexdigest())
except Exception as e:
    print("ERR", pandas.__version__, type(e).__name__, str(e)[:80])
"""
res = {}
for kit, root in (("old kit", old), ("new kit", new)):
    for py in (py2, py3):
        out = subprocess.run([py, "-c", CODE], cwd=f"{root}/Autoformer", capture_output=True, text=True).stdout.strip()
        res[(kit, py)] = out
        print(f"{kit:8s} {out}")
h = {k: v.split()[-1] for k, v in res.items() if v.startswith("OK")}
ok = (res[("old kit", py2)].startswith("OK") and res[("new kit", py2)].startswith("OK")
      and h[("old kit", py2)] == h[("new kit", py2)])
ok3 = res[("new kit", py3)].startswith("OK") and h[("new kit", py3)] == h[("old kit", py2)]
print(f"[{'PASS' if ok else 'FAIL'}] pandas 2: old and new kit give byte-identical data_stamp")
print(f"[{'PASS' if ok3 else 'FAIL'}] pandas 3: new kit works and equals the pandas-2 data_stamp "
      f"(old kit: {res[('old kit', py3)].split()[0]})")
print("OVERALL:", "PASS" if ok and ok3 else "FAIL")
sys.exit(0 if ok and ok3 else 1)
