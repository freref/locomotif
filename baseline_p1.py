import numpy as np

ts = np.loadtxt("../locomotif-profiling/datasets_profile/fantasia_f1o01.csv", delimiter=",")[:9000]
n = len(ts)
sm = np.exp(-np.power(ts[:, None] - ts[None, :], 2))
flat = sm[np.triu_indices(n)]
rho = 0.6
h = rho * (len(flat)-1)
k_lo, k_hi = int(np.floor(h)), int(np.ceil(h))
p = np.partition(flat, (k_lo, k_hi))
v_lo, v_hi = p[k_lo], p[k_hi]
weight = h - k_lo
tau = (1-weight)*v_lo + weight*v_hi

csm = np.zeros((n+2, n+2))
bp = np.full((n+2, n+2), -1, dtype=np.int8)
for i in range(n):
    # Only triu with diag_offset=0
    for j in range(i, n):
        sim = sm[i, j]
        ii, jj = i+2, j+2
        # No warping
        ps = csm[ii-1, jj-1]
        val = sim + ps if sim >= tau else 0.5*ps-1.0
        if val > 0:
            csm[ii, jj] = val
            bp[ii, jj] = 0

# Greedy path finding
mask = np.zeros((n+2, n+2), dtype=bool)
# Mask diagonal and below
for i in range(n+2):
    mask[i, :i+21] = True # vwidth=20

paths = []
while True:
    masked_csm = csm.copy()
    masked_csm[mask] = 0
    best_idx = np.argmax(masked_csm)
    if masked_csm.flat[best_idx] <= 0:
        break
    i, j = np.unravel_index(best_idx, csm.shape)
    # Extract path
    path = []
    curr_i, curr_j = i, j
    while curr_i >= 2 and curr_j >= 2 and not mask[curr_i, curr_j] and csm[curr_i, curr_j] > 0:
        path.append((curr_i-2, curr_j-2))
        curr_i -= 1
        curr_j -= 1
    path = path[::-1]
    paths.append(path)
    # Mask path
    for pi, pj in path:
        mask[pi+2, pj+2] = True
    if len(paths) >= 1:
        break

p1 = paths[0]
print(f"P1 baseline: {p1[0]}...{p1[-1]}, len={len(p1)}")
