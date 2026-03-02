
import numpy as np
ts = np.loadtxt("../locomotif-profiling/datasets_profile/fantasia_f1o01.csv", delimiter=",")[:9000]
n = len(ts)
fs = 83.33333333333333
l_min = int(0.5 * fs); l_max = int(1.5 * fs); rho = 0.6
vwidth = max(10, l_min // 2)

gamma = 1 / np.std(ts)**2
sm = np.exp(-gamma * np.power(ts[:, None] - ts[None, :], 2))
flat = sm[np.triu_indices(n)]
tau = np.quantile(flat, rho)
delta_a, delta_m = 2*tau, 0.5
csm = np.zeros((n+2, n+2))
for i in range(n):
    for j in range(i, n):
        sim = sm[i, j]; ii, jj = i+2, j+2
        ps = csm[ii-1, jj-1]
        val = sim + ps if sim >= tau else delta_m*ps-delta_a
        if val > 0: csm[ii, jj] = val

mask = np.full((n+2, n+2), True, dtype=bool)
mask[np.triu_indices(n+2, k=vwidth+1)] = False
mask = mask | (csm <= 0)

paths = []
# Prepend diagonal
paths.append(np.tile(np.arange(n, dtype=np.int32), (2, 1)).T)

while len(paths) < 100:
    masked_csm = csm.copy(); masked_csm[mask] = 0
    best_idx = np.argmax(masked_csm)
    if masked_csm.flat[best_idx] <= 0: break
    ci, cj = np.unravel_index(best_idx, csm.shape)
    p = []
    curr_i, curr_j = ci, cj
    while curr_i >= 2 and curr_j >= 2:
        p.append((curr_i, curr_j))
        v1, v2, v3 = csm[curr_i-1, curr_j-1], csm[curr_i-2, curr_j-1], csm[curr_i-1, curr_j-2]
        mx = max(v1, v2, v3)
        if v1 == mx:
            if mask[curr_i-1, curr_j-1]: break
            curr_i, curr_j = curr_i-1, curr_j-1
        elif v2 == mx:
            if mask[curr_i-2, curr_j-1]: break
            curr_i, curr_j = curr_i-2, curr_j-1
        else:
            if mask[curr_i-1, curr_j-2]: break
            curr_i, curr_j = curr_i-1, curr_j-2
    p = np.array(p[::-1], dtype=np.int32)
    if (p[-1, 0] - p[0, 0] + 1) >= l_min or (p[-1, 1] - p[0, 1] + 1) >= l_min:
        paths.append(p - 2)
        for pi, pj in p:
            mask[max(0, pi-vwidth):min(n+2, pi+vwidth+1), pj] = True
            mask[pi, max(0, pj-vwidth):min(n+2, pj+vwidth+1)] = True
    else: mask[ci, cj] = True

b, e = 1509, 1632
motif_set = []
for p in paths:
    if b >= p[0, 1] and p[-1, 1] + 1 >= e:
        idx_b = -1; idx_e = -1
        for k in range(len(p)):
            if p[k, 1] == b: idx_b = k
            if p[k, 1] == e-1: idx_e = k
        if idx_b != -1 and idx_e != -1:
            motif_set.append((int(p[idx_b, 0]), int(p[idx_e, 0] + 1)))
print(sorted(motif_set))
