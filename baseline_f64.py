import numpy as np

ts = np.loadtxt("../locomotif-profiling/datasets_profile/fantasia_f1o01.csv", delimiter=",")[:9000]
n = len(ts)
fs = 83.33333333333333
l_min = int(0.5 * fs)
l_max = int(1.5 * fs)
rho = 0.6
vwidth = max(10, l_min // 2)

# Main branch logic with float64 SM
gamma = 1 / np.std(ts)**2
sm = np.exp(-gamma * np.power(ts[:, None] - ts[None, :], 2)) # float64
flat = sm[np.triu_indices(n)]
tau = np.quantile(flat, rho)
delta_a = 2 * tau
delta_m = 0.5

# CSM
csm = np.zeros((n+2, n+2))
for i in range(n):
    for j in range(i, n):
        sim = sm[i, j]
        ii, jj = i+2, j+2
        ps = csm[ii-1, jj-1]
        val = sim + ps if sim >= tau else delta_m*ps-delta_a
        if val > 0: csm[ii, jj] = val

# Diagonal path
diag_path_sims = sm[np.arange(n), np.arange(n)]
diag_cum = np.zeros(n + 1)
diag_cum[1:] = np.cumsum(diag_path_sims)

# Mask
mask = np.full((n+2, n+2), True, dtype=bool)
mask[np.triu_indices(n+2, k=vwidth+1)] = False
mask = mask | (csm <= 0)

# Paths discovery like main
paths = []
while True:
    masked_csm = csm.copy()
    masked_csm[mask] = 0
    best_idx = np.argmax(masked_csm)
    if masked_csm.flat[best_idx] <= 0: break
    i_best, j_best = np.unravel_index(best_idx, csm.shape)
    
    p = []
    ci, cj = i_best, j_best
    while ci >= 2 and cj >= 2:
        p.append((ci, cj))
        v1, v2, v3 = csm[ci-1, cj-1], csm[ci-2, cj-1], csm[ci-1, cj-2]
        maximum = max(v1, v2, v3)
        if v1 == maximum:
            if mask[ci-1, cj-1]: break
            ci, cj = ci-1, cj-1
        elif v2 == maximum:
            if mask[ci-2, cj-1]: break
            ci, cj = ci-2, cj-1
        else:
            if mask[ci-1, cj-2]: break
            ci, cj = ci-1, cj-2
    p = np.array(p[::-1], dtype=np.int32)
    if (p[-1, 0] - p[0, 0] + 1) >= l_min or (p[-1, 1] - p[0, 1] + 1) >= l_min:
        paths.append(p)
        for pi, pj in p:
            for dr in range(-vwidth, vwidth+1):
                if 0 <= pi+dr < n+2: mask[pi+dr, pj] = True
                if 0 <= pj+dr < n+2: mask[pi, pj+dr] = True
    else:
        mask[i_best, j_best] = True

    if len(paths) >= 1: break

p1 = paths[0]
print(f"Set 1 P1 end: {p1[-1]-2}")
# Check induced segments
induced = []
# Diagonal
# project_to_vertical_axis returns i-axis
# diagonal has i=j
# ...
print(f"Tau: {tau}")
