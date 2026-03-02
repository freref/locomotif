import numpy as np
from locomotif import locomotif

ts = np.loadtxt("../locomotif-profiling/datasets_profile/fantasia_f1o01.csv", delimiter=",")[:9000]
fs = 83.33333333333333
l_min = int(0.5 * fs)
l_max = int(1.5 * fs)
rho = 0.6

lcm = locomotif.get_locomotif_instance(ts, l_min, l_max, rho=rho, warping=True)
lcm.find_best_paths()

print(f"Number of paths: {len(lcm._path_collection)}")
p1 = lcm._path_collection[1]
print(f"Path 1 (index 1) i range: {p1.i1, p1.il}")
print(f"Path 1 (index 1) j range: {p1.j1, p1.jl}")

# Find first motif set
motif_sets = list(lcm.find_best_motif_sets(nb=1, overlap=0.0))
(rep, mset, fits) = motif_sets[0]
print(f"Set 1: rep={rep}, size={len(mset)}")
print(f"First induced: {mset[0]}")
