import numpy as np
import json
from pathlib import Path

datasets_dir = Path("../locomotif-profiling/datasets_profile")
files = sorted(datasets_dir.glob("*.csv"))

for f in files:
    ts = np.loadtxt(f, delimiter=",")[:9000]
    n = len(ts)
    if ts.ndim == 1:
        sm = np.exp(-np.power(ts[:, None] - ts[None, :], 2)).astype(np.float32)
    else:
        sm = np.exp(-np.sum(np.power(ts[:, None, :] - ts[None, :, :], 2), axis=2)).astype(np.float32)
    
    tau = np.quantile(sm[np.triu_indices(n)], 0.6)
    print(f"{f.name}: {tau:.10f}")
