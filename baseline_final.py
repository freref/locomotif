import numpy as np

ts = np.loadtxt("../locomotif-profiling/datasets_profile/fantasia_f1o01.csv", delimiter=",")[:9000]
n = len(ts)
# Generic motif length for these periodic signals (0.5s to 1.5s)
fs = 83.33333333333333
l_min = int(0.5 * fs)
l_max = int(1.5 * fs)
rho = 0.6
vwidth = l_min // 2

# Main logic
gamma = 1 / np.var(ts)
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
diag_path = np.tile(np.arange(n, dtype=np.int32), (2, 1)).T
diag_sims = sm[np.arange(n), np.arange(n)]
diag_cum = np.zeros(n + 1, dtype=np.float32)
diag_cum[1:] = np.cumsum(diag_sims)

# Find first non-diagonal path
mask = np.full((n+2, n+2), True, dtype=bool)
for i in range(n+2):
    for j in range(i + vwidth + 1, n+2):
        mask[i, j] = False

masked_csm = csm * (~mask)
best_idx = np.argmax(masked_csm)
i_best, j_best = np.unravel_index(best_idx, csm.shape)
# Path 1 extraction
p1 = []
ci, cj = i_best, j_best
while ci >= 2 and cj >= 2 and not mask[ci, cj] and csm[ci, cj] > 0:
    p1.append((ci-2, cj-2))
    ci -= 1; cj -= 1
p1 = np.array(p1[::-1], dtype=np.int32)
p1_sims = sm[p1[:, 0], p1[:, 1]]
p1_cum = np.zeros(len(p1) + 1, dtype=np.float32)
p1_cum[1:] = np.cumsum(p1_sims)

# Candidate 1: b_repr=1509
b_repr = 1509
# Diagonal induced:
kb_diag = b_repr
# ke_diag = path.find_j(e_repr-1)
for e_repr in range(b_repr + l_min, b_repr + l_max + 1):
    ke_diag = e_repr - 1
    # total_length = (ke_diag - kb_diag + 1) + (ke_p1 - kb_p1 + 1)
    # n_score = (score - l_repr) / total_path_length
    # n_coverage = (total_length - 0.0 - l_repr) / n
    pass

print(f"P1 endpoint: {i_best-2, j_best-2}, val={csm[i_best, j_best]}")
print(f"Tau: {tau}")
