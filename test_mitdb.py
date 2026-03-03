import numpy as np
from locomotif import locomotif
import hashlib

def get_motif_fingerprint(motif_sets):
    motif_data = []
    for representative, motif_set, fitness in motif_sets:
        rep_canon = (int(representative[0]), int(representative[1]))
        mset_canon = sorted([(int(s[0]), int(s[1])) for s in motif_set])
        motif_data.append((rep_canon, mset_canon))
    motif_data.sort()
    data_to_hash = str(motif_data)
    return hashlib.md5(data_to_hash.encode()).hexdigest()[:8]

ts = np.loadtxt("examples/datasets/mitdb_patient214.csv", delimiter=",")
print(f"TS shape: {ts.shape}")
fs = 360
l_min = int(0.5 * fs)
l_max = int(1.5 * fs)
rho = 0.6

lcm = locomotif.get_locomotif_instance(ts, l_min, l_max, rho=rho, warping=True)
lcm.find_best_paths(vwidth=l_min // 2)
motif_sets = list(lcm.find_best_motif_sets(nb=5, overlap=0.0))

for i, (rep, mset, fits) in enumerate(motif_sets):
    print(f"Motif Set {i+1}: rep={rep}, size={len(mset)}")

print(f"Fingerprint: {get_motif_fingerprint(motif_sets)}")
