"""Synthetic stand-ins for ON-Zonal and Weather, for CPU smoke tests only.

    python tests/make_smoke_data.py WORK_DIR

Writes WORK_DIR/data/{ONZ_real,ONZ_shift,weather}.csv with the real files' columns and
frequencies (ON-Zonal: date + 10 zones, hourly; Weather: date + 20 series + OT, 10-minute)
and copies them where p24.py's plans read them, exactly as onzonal.py / p24.py weather-data
do. Values are random sinusoids: these files test the code path, never a result.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onzonal import ZONES, WEEK, make_shifted  # noqa: E402

work = Path(sys.argv[1])
data = work / "data"
data.mkdir(parents=True, exist_ok=True)
g = np.random.default_rng(473)
N = 9000  # hourly rows: enough validation windows for H=720 with batch 128

idx = pd.date_range("2023-11-01", periods=N + 9 * WEEK, freq="h")
t = np.arange(len(idx))
z = pd.DataFrame({k: 1000 + 300 * np.sin(2 * np.pi * (t / 24 + i / 10)) + 150 * np.sin(2 * np.pi * t / WEEK)
                  + 40 * g.standard_normal(len(t)) for i, k in enumerate(ZONES)}, index=idx)
real, shift = z.iloc[9 * WEEK:], make_shifted(z).iloc[9 * WEEK:]
for name, df in (("ONZ_real", real), ("ONZ_shift", shift)):
    out = df.copy()
    out.index.name = "date"
    out.reset_index().to_csv(data / f"{name}.csv", index=False, float_format="%.1f")
for sub in ("PatchTST/PatchTST_supervised/dataset", "LTSF-Linear/dataset", "iTransformer/dataset/ETT-small"):
    d = work / "repos" / sub
    d.mkdir(parents=True, exist_ok=True)
    for name in ("ONZ_real", "ONZ_shift"):
        shutil.copy(data / f"{name}.csv", d / f"{name}.csv")

widx = pd.date_range("2020-01-01", periods=N, freq="10min")
tw = np.arange(N)
cols = [f"s{i:02d}" for i in range(20)] + ["OT"]
w = pd.DataFrame({c: np.sin(2 * np.pi * (tw / 144 + i / 21)) + 0.1 * g.standard_normal(N) for i, c in enumerate(cols)})
w.insert(0, "date", widx.strftime("%Y-%m-%d %H:%M:%S"))
w.to_csv(data / "weather.csv", index=False, float_format="%.4f")
print(f"wrote synthetic ONZ_real/ONZ_shift ({len(real)} rows, {real.index[0]} .. {real.index[-1]}) "
      f"and weather ({N} rows, {len(cols)} series) to {data}")
