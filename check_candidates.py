import numpy as np
from locomotif import locomotif

ts = np.loadtxt("../locomotif-profiling/datasets_profile/fantasia_f1o01.csv", delimiter=",")[:9000]
fs = 83.33333333333333
l_min = int(0.5 * fs)
l_max = int(1.5 * fs)
rho = 0.6

lcm = locomotif.get_locomotif_instance(ts, l_min, l_max, rho=rho, warping=True)
lcm.find_best_paths()

n = len(ts)
mask = np.zeros(n, dtype=bool)
start_mask = np.ones(n, dtype=bool)
end_mask = np.ones(n, dtype=bool)

# Find first 3 sets
for _ in range(3):
    (rep, mset, fits) = next(lcm.find_best_motif_sets(nb=1, overlap=0.0))
    # Update mask manually to match internal logic
    for (bm, em) in mset:
        mask[bm:em] = True
    start_mask[mask] = False
    end_mask[mask] = False

# Now check candidates for set 4
res, best_fit, all_fits = locomotif._find_best_candidate_graph(n, l_min, l_max, 0.0, mask, mask, start_mask, end_mask, *lcm._path_data, keep_fitnesses=True)

# Sort all_fits by fitness descending
all_fits = all_fits[np.argsort(all_fits[:, 2])[::-1]]

print(f"Set 4 candidates (top 10):")
for i in range(min(10, len(all_fits))):
    print(f"rep=({int(all_fits[i,0])}, {int(all_fits[i,1])}), fit={all_fits[i,2]:.6f}, cov={all_fits[i,3]:.6f}, score={all_fits[i,4]:.6f}")

baseline_rep = (749, 790)
found_baseline = False
for i in range(len(all_fits)):
    if int(all_fits[i,0]) == baseline_rep[0] and int(all_fits[i,1]) == baseline_rep[1]:
        print(f"BASELINE rep ({baseline_rep}) found at rank {i+1}, fit={all_fits[i,2]:.6f}")
        found_baseline = True
        break
if not found_baseline:
    print(f"BASELINE rep ({baseline_rep}) NOT FOUND in candidates")
