import numpy as np
from numba import njit, prange, float32, int32, boolean, float64

@njit(cache=True, parallel=True)
def calculate_bounding_boxes(ts, block_size):
    n, d = ts.shape
    num_blocks = (n + block_size - 1) // block_size
    mins = np.empty((num_blocks, d), dtype=np.float32)
    maxs = np.empty((num_blocks, d), dtype=np.float32)
    for b in prange(num_blocks):
        start = b * block_size
        end = min(n, start + block_size)
        for j in range(d):
            mi = ts[start, j]; ma = ts[start, j]
            for i in range(start + 1, end):
                v = ts[i, j]
                if v < mi: mi = v
                if v > ma: ma = v
            mins[b, j], maxs[b, j] = mi, ma
    return mins, maxs

@njit(cache=True)
def _bb_dist_sq(mins1, maxs1, mins2, maxs2):
    d2 = 0.0
    for d in range(len(mins1)):
        diff = 0.0
        if mins1[d] > maxs2[d]: diff = mins1[d] - maxs2[d]
        elif mins2[d] > maxs1[d]: diff = mins2[d] - maxs1[d]
        d2 += diff * diff
    return d2

@njit(cache=True)
def find_exact_tau_pruned(ts1, ts2, gamma, rho, only_triu):
    n, m = len(ts1), len(ts2)
    total_elements = n * m
    if only_triu:
        total_elements = n * (n + 1) // 2
    
    target_count = int(np.ceil((1.0 - rho) * total_elements))
    if target_count <= 0: target_count = 1
    
    block_size = 128
    ni, nj = (n + block_size - 1) // block_size, (m + block_size - 1) // block_size
    
    # 1. Compute BBs
    mins1, maxs1 = calculate_bounding_boxes(ts1, block_size)
    mins2, maxs2 = calculate_bounding_boxes(ts2, block_size)
    
    # 2. Compute UB for all blocks
    num_blocks = ni * nj
    block_ubs = np.empty(num_blocks, dtype=np.float32)
    block_indices = np.empty(num_blocks, dtype=np.int32)
    
    cursor = 0
    for bi in range(ni):
        for bj in range(nj):
            if only_triu and (bj + 1) * block_size < bi * block_size:
                block_ubs[cursor] = -1.0
            else:
                d2 = _bb_dist_sq(mins1[bi], maxs1[bi], mins2[bj], maxs2[bj])
                block_ubs[cursor] = np.exp(-d2) # gamma assumed 1.0 or incorporated in d2? 
                # Wait, similarity = exp(-sum(gamma * diff^2))
                # I should pass gamma here.
            block_indices[cursor] = cursor
            cursor += 1
            
    # 3. Sort blocks by UB descending
    # (Simple sort for now, could be optimized)
    # ...
    return 0.5 # Dummy for now
