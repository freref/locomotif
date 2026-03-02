import cProfile
import hashlib
import json
import pstats
import time
from pathlib import Path

import numpy as np
import locomotif.locomotif as locomotif


def get_motif_fingerprint(motif_sets):
    # Create a fingerprint based on the start and end indices of all discovered motifs
    motif_data = []
    for representative, motif_set, fitness in motif_sets:
        # Convert to canonical form (tuples of ints)
        rep_canon = (int(representative[0]), int(representative[1]))
        mset_canon = sorted([(int(s[0]), int(s[1])) for s in motif_set])
        motif_data.append((rep_canon, mset_canon))
    
    # Sort the motif sets by representative to be order-insensitive
    motif_data.sort()
    
    data_to_hash = str(motif_data)
    return hashlib.md5(data_to_hash.encode()).hexdigest()[:8]

def profile_dataset(file_path):
    print(f"\n--- Profiling {file_path.name} ---")
    ts = np.loadtxt(file_path, delimiter=",")
    
    # Cap at 9,000 samples (1.5 times larger than previous 6,000)
    ts = ts[:9000]
    
    meta_path = Path("../locomotif-profiling") / file_path.with_suffix(".json").name
    if not meta_path.exists():
        # Try finding it in the same dir as the script copy
        meta_path = Path("../locomotif-profiling/datasets_profile") / file_path.with_suffix(".json").name
        
    meta = json.loads(meta_path.read_text())
    fs = meta["fs"]
    
    # Generic motif length for these periodic signals (0.5s to 1.5s)
    l_min = int(0.5 * fs)
    l_max = int(1.5 * fs)
    rho = 0.6
    nb_motifs = 5
    
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.perf_counter()
    lcm = locomotif.get_locomotif_instance(ts, l_min, l_max, rho=rho, warping=True)
    lcm.find_best_paths(vwidth=l_min // 2)
    # find_best_motif_sets returns (representative, motif_set, fitness)
    motif_sets_raw = list(lcm.find_best_motif_sets(nb=nb_motifs, overlap=0))
    total_time = time.perf_counter() - start_time
    
    pr.disable()
    
    # Process for summary and fingerprint
    representatives = []
    motif_set_sizes = []
    for rep, mset, fitness in motif_sets_raw:
        representatives.append([int(rep[0]), int(rep[1])])
        motif_set_sizes.append(len(mset))
    
    fingerprint = get_motif_fingerprint(motif_sets_raw)
    
    # Stats
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
    
    print(f"motif_fingerprint={fingerprint}")
    print(f"total_seconds={total_time:.6f}")
    print(f"motif_sets_found={len(motif_set_sizes)}")
    print(f"motif_set_sizes={motif_set_sizes}")
    print(f"representatives={representatives}")
    
    print("\nTop 5 slowest functions:")
    ps.print_stats(5)
    
    return {
        "dataset": file_path.name,
        "fingerprint": fingerprint,
        "total_time": total_time,
    }

def main():
    datasets_dir = Path("../locomotif-profiling/datasets_profile")
    csv_files = sorted(datasets_dir.glob("*.csv"))
    
    results = []
    for csv_file in csv_files:
        try:
            res = profile_dataset(csv_file)
            results.append(res)
        except Exception as e:
            print(f"Failed to profile {csv_file.name}: {e}")
    
    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    for res in results:
        print(f"{res['dataset']:30} | {res['fingerprint']} | {res['total_time']:6.2f}s")

if __name__ == "__main__":
    main()
