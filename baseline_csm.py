import numpy as np

ts = np.loadtxt("../locomotif-profiling/datasets_profile/fantasia_f1o01.csv", delimiter=",")[:9000]
n = len(ts)
gamma = 1 / np.std(ts)**2
sm = np.exp(-gamma * np.power(ts[:, None] - ts[None, :], 2)).astype(np.float64)
flat = sm[np.triu_indices(n)]
rho = 0.6
h = rho * (len(flat)-1)
k_lo, k_hi = int(np.floor(h)), int(np.ceil(h))
p = np.partition(flat, (k_lo, k_hi))
v_lo, v_hi = p[k_lo], p[k_hi]
weight = h - k_lo
tau = (1-weight)*v_lo + weight*v_hi
delta_a = 2 * tau
delta_m = 0.5

csm = np.zeros((n+2, n+2))
for i in range(n):
    for j in range(i, n):
        sim = sm[i, j]
        ii, jj = i+2, j+2
        ps = csm[ii-1, jj-1]
        val = sim + ps if sim >= tau else delta_m*ps-delta_a
        if val > 0: csm[ii, jj] = val

# Greedy path finding
mask = np.full((n+2, n+2), True, dtype=bool)
vwidth = int(0.5 * 83.33) // 2
# Allowed region: j >= i + vwidth + 1
for i in range(n+2):
    for j in range(i + vwidth + 1, n+2):
        mask[i, j] = False

best_idx = np.argmax(csm * (~mask))
i_best, j_best = np.unravel_index(best_idx, csm.shape)
print(f"P1 end: {i_best-2, j_best-2}, val={csm[i_best, j_best]}")
