import numpy as np

ts = np.loadtxt("../locomotif-profiling/datasets_profile/fantasia_f1o01.csv", delimiter=",")[:9000]
n = len(ts)
fs = 83.33333333333333
l_min = int(0.5 * fs)
l_max = int(1.5 * fs)
rho = 0.6
vwidth = max(10, l_min // 2)

# Main branch logic
gamma = 1 / np.std(ts)**2
sm = np.exp(-gamma * np.power(ts[:, None] - ts[None, :], 2)).astype(np.float32)
flat = sm[np.triu_indices(n)]
tau = np.quantile(flat, rho)
delta_a = 2 * tau
delta_m = 0.5

# CSM
csm = np.zeros((n+2, n+2), dtype=np.float32)
for i in range(n):
    for j in range(i, n):
        sim = sm[i, j]
        ii, jj = i+2, j+2
        ps = csm[ii-1, jj-1]
        val = sim + ps if sim >= tau else delta_m*ps-delta_a
        if val > 0: csm[ii, jj] = val

# Diagonal path
diag_path_sims = sm[np.arange(n), np.arange(n)]
diag_cum = np.zeros(n + 1, dtype=np.float32)
diag_cum[1:] = np.cumsum(diag_path_sims)

# Mask
mask = np.full((n+2, n+2), True, dtype=bool)
mask[np.triu_indices(n+2, k=vwidth+1)] = False
# Mask zeros
mask = mask | (csm <= 0)

# Path 1
masked_csm = csm.copy()
masked_csm[mask] = 0
best_idx = np.argmax(masked_csm)
i_best, j_best = np.unravel_index(best_idx, csm.shape)

# Extraction like main
p1 = []
ci, cj = i_best, j_best
while ci >= 2 and cj >= 2:
    p1.append((ci, cj))
    maximum = max(csm[ci-1, cj-1], csm[ci-2, cj-1], csm[ci-1, cj-2])
    if csm[ci-1, cj-1] == maximum:
        if mask[ci-1, cj-1]: break
        ci, cj = ci-1, cj-1
    elif csm[ci-2, cj-1] == maximum:
        if mask[ci-2, cj-1]: break
        ci, cj = ci-2, cj-1
    else:
        if mask[ci-1, cj-2]: break
        ci, cj = ci-1, cj-2
p1 = p1[::-1]
# project_to_vertical_axis
p1_v = (p1[0][0]-2, p1[-1][0]-2+1)

print(f"P1 end: {i_best-2, j_best-2}, val={csm[i_best, j_best]}")
print(f"P1 range: {p1_v}")
print(f"Tau: {tau}")
