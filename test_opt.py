import numpy as np
import time
from numba import njit, prange

@njit(parallel=True)
def prepare_mask(csm, mask):
    n, m = csm.shape
    for i in prange(n):
        for j in range(m):
            if csm[i, j] <= 0:
                mask[i, j] = True

n = 1000
csm = np.random.randn(n, n).astype(np.float32)
mask = np.tril(np.ones((n, n), dtype=bool), k=10)
prepare_mask(csm, mask) # warmup

n = 21600
csm = np.random.randn(n, n).astype(np.float32)
mask = np.tril(np.ones((n, n), dtype=bool), k=10)
t0 = time.time()
mask = mask | (csm <= 0)
print("numpy bitwise |:", time.time() - t0)

mask = np.tril(np.ones((n, n), dtype=bool), k=10)
t0 = time.time()
prepare_mask(csm, mask)
print("numba prange:", time.time() - t0)
