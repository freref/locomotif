import numpy as np
import time
from numba import njit, prange

@njit
def _best_predecessor(a, b, c):
    if a >= b:
        if a >= c: return a, np.int8(0)
        return c, np.int8(2)
    if b >= c: return b, np.int8(1)
    return c, np.int8(2)

@njit(parallel=True)
def csm_parallel(sm, tau, da, dm):
    n, m = sm.shape
    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
    for i in range(n):
        for j in prange(m):
            sim = sm[i, j]
            ii = i + 2
            jj = j + 2
            p, d = _best_predecessor(csm[ii-1, jj-1], csm[ii-2, jj-1], csm[ii-1, jj-2])
            if sim < tau:
                v = dm * p - da
            else:
                v = sim + p
            if v > 0:
                csm[ii, jj] = v
                bp_dir[ii, jj] = d
    return csm

@njit
def csm_serial(sm, tau, da, dm):
    n, m = sm.shape
    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            sim = sm[i, j]
            ii = i + 2
            jj = j + 2
            p, d = _best_predecessor(csm[ii-1, jj-1], csm[ii-2, jj-1], csm[ii-1, jj-2])
            if sim < tau:
                v = dm * p - da
            else:
                v = sim + p
            if v > 0:
                csm[ii, jj] = v
                bp_dir[ii, jj] = d
    return csm

def run():
    N = 10000
    sm = np.random.rand(N, N).astype(np.float32)
    csm_serial(sm[:10,:10], 0.8, 1.0, 0.5)
    csm_parallel(sm[:10,:10], 0.8, 1.0, 0.5)
    
    t0 = time.time()
    c1 = csm_serial(sm, 0.8, 1.0, 0.5)
    print(f"Serial: {time.time()-t0:.3f}s")
    
    t0 = time.time()
    c2 = csm_parallel(sm, 0.8, 1.0, 0.5)
    print(f"Parallel: {time.time()-t0:.3f}s")
    
    print("Match:", np.allclose(c1, c2))

if __name__ == "__main__":
    run()
