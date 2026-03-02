import numpy as np
from locomotif import locomotif

ts = np.loadtxt("../locomotif-profiling/datasets_profile/fantasia_f1o01.csv", delimiter=",")[:9000]
fs = 83.33333333333333
l_min = int(0.5 * fs)
l_max = int(1.5 * fs)
rho = 0.6

lcm = locomotif.get_locomotif_instance(ts, l_min, l_max, rho=rho, warping=True)
lcm.find_best_paths()

# First motif set
motif_sets = list(lcm.find_best_motif_sets(nb=1, overlap=0.0))
(rep, mset, fits) = motif_sets[0]
print(f"Set 1 size: {len(mset)}")
print(f"Sorted induced: {sorted(mset)}")
