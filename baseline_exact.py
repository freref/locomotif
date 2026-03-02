import numpy as np

ts = np.loadtxt("../locomotif-profiling/datasets_profile/fantasia_f1o01.csv", delimiter=",")[:9000]
n = len(ts)
# Use exactly the same parameters as the profiling script
fs = 83.33333333333333
l_min = int(0.5 * fs)
l_max = int(1.5 * fs)
rho = 0.6
vwidth = l_min // 2

# Baseline tau
gamma = 1 / np.var(ts)
sm = np.exp(-gamma * np.power(ts[:, None] - ts[None, :], 2))
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

# Diagonal is path 0
diag = [(i, i) for i in range(n)]

# Find P1
mask = np.full((n+2, n+2), True, dtype=bool)
# main branch: mask[np.triu_indices(len(mask), k=vwidth+1)] = False
for i in range(n+2):
    for j in range(i + vwidth + 1, n+2):
        mask[i, j] = False

masked_csm = csm * (~mask)
best_idx = np.argmax(masked_csm)
i_best, j_best = np.unravel_index(best_idx, csm.shape)
# Extract path
p1 = []
ci, cj = i_best, j_best
while ci >= 2 and cj >= 2 and not mask[ci, cj] and csm[ci, cj] > 0:
    p1.append((ci-2, cj-2))
    ci -= 1; cj -= 1
p1 = p1[::-1]

# Path 0: diag, Path 1: p1, Path 2: p1_mirrored
# ... (skip full motif discovery, just check first candidate)
print(f"P1 end: {i_best-2, j_best-2}, val={csm[i_best, j_best]}")
print(f"P1 start: {p1[0]}")
