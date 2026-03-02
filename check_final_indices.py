import numpy as np
import locomotif.locomotif as locomotif
from numba.typed import List

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

# MATERIALIZE PATHS
all_paths = List()
for p in lcm._path_collection:
    all_paths.append(p)

# Ported _find_best_candidate (the one from main)
# I need to ensure SortedPathArray works with my Paths.
# Actually I already have it in locomotif.py? No I removed it.
# Let's use the graph finder first to see what it gives.

motif_sets = list(lcm.find_best_motif_sets(nb=1, overlap=0.0))
(rep, mset, fit) = motif_sets[0]
print(f"Rep: {rep}, size={len(mset)}")
print(f"Induced: {mset}")
