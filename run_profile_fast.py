import sys
from pathlib import Path
sys.path.append(".")
sys.path.append("../locomotif-profiling")
import locomotif_profiling_ecg
import numpy as np

def main():
    root = Path("../locomotif-profiling")
    path_to_series = root / "data" / "ecg_mitdb_patient214.csv"
    ts, fs, full_len, applied_extra_factor = locomotif_profiling_ecg.load_local_ecg_segment(
        path_to_series=path_to_series,
        factor=5,
        max_samples=-1,
    )
    print(f"segment_len={len(ts)}")
    result, l_min, l_max, total_pr = locomotif_profiling_ecg.run_locomotif_profile(
        ts=ts, fs=fs, rho=0.5, nb_motifs=5, overlap=0,
    )
    print(f"total_seconds={result['total_seconds']:.6f}")
    summary = locomotif_profiling_ecg.summarize_motif_sets(result["motif_sets"])
    print(f"motif_fingerprint={summary['motif_fingerprint']}")

if __name__ == "__main__":
    main()
