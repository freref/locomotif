import numpy as np


### JIT
from numba import int32, int8, float64, float32, boolean
from numba import njit
from numba.types import List, Array
from numba import prange
from numba.typed import List as TypedList

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
def _exact_tau_smart(sm, rho, only_triu):
    n, m = sm.shape
    if only_triu:
        total = n * (n + 1) // 2
        flat = np.empty(total, dtype=np.float32)
        curr = 0
        for i in range(n):
            for j in range(i, m):
                flat[curr] = sm[i, j]
                curr += 1
    else:
        flat = sm.flatten()
    if len(flat) == 0: return np.float32(0.0)
    h = np.float32(rho) * (len(flat) - 1)
    k_lo = int(np.floor(h)); k_hi = int(np.ceil(h))
    if k_lo == k_hi: return np.partition(flat, k_lo)[k_lo]
    p = np.partition(flat, (k_lo, k_hi))
    v_lo, v_hi = p[k_lo], p[k_hi]
    weight = h - np.float32(k_lo)
    return (np.float32(1.0) - weight) * v_lo + weight * v_hi

@njit(float32[:, :](float32[:, :], float32[:, :], float64[:], boolean, int32), parallel=True)
def similarity_matrix_ndim(ts1, ts2, gamma=None, only_triu=False, diag_offset=0):
    n, m = len(ts1), len(ts2)
    sm = np.full((n, m), -np.inf, dtype=np.float32)
    for i in prange(n):
        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m
        similarities = np.exp(-np.sum(gamma.T * np.power(ts1[i, :] - ts2[j_start:j_end, :], 2), axis=1))
        sm[i, j_start:j_end] = similarities
    return sm

@njit
def max3(a, b, c):
    if a >= b:
        if a >= c: return a
        return c
    if b >= c: return b
    return c

@njit(cache=True)
def cumulative_similarity_matrix_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0, mins1=None, maxs1=None, mins2=None, maxs2=None, gamma=None):
    n, m = sm.shape
    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    block_size = 64
    ni, nj = (n + block_size - 1) // block_size, (m + block_size - 1) // block_size
    for bi in range(ni):
        i_s = bi * block_size; i_e = min(n, i_s + block_size)
        for bj in range(nj):
            j_s = bj * block_size; j_e = min(m, j_s + block_size)
            if only_triu and j_e < i_s - diag_offset: continue
            
            d2_lb = 0.0
            for d in range(len(gamma)):
                diff = 0.0
                if mins1[bi, d] > maxs2[bj, d]: diff = mins1[bi, d] - maxs2[bj, d]
                elif mins2[bj, d] > maxs1[bi, d]: diff = mins2[bj, d] - maxs1[bi, d]
                d2_lb += gamma[d] * diff * diff
            
            if np.exp(-d2_lb) < tau:
                any_inc = False
                for j in range(j_s + 1, j_e + 3):
                    if csm[i_s + 1, j] > 0.0: any_inc = True; break
                if not any_inc:
                    for i in range(i_s + 1, i_e + 3):
                        if csm[i, j_s + 1] > 0.0: any_inc = True; break
                if not any_inc: continue
            
            for i in range(i_s, i_e):
                jj_s = max(j_s, i - diag_offset) if only_triu else j_s
                for j in range(jj_s, j_e):
                    sim = sm[i, j]; ii, jj = i + 2, j + 2
                    v1 = csm[ii-1, jj-1]; v2 = csm[ii-2, jj-1]; v3 = csm[ii-1, jj-2]
                    max_cs = max3(v1, v2, v3)
                    if sim < tau: val = delta_m * max_cs - delta_a
                    else: val = sim + max_cs
                    if val > 0.0: csm[ii, jj] = np.float32(val)
    return csm

@njit(cache=True)
def cumulative_similarity_matrix_no_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0, mins1=None, maxs1=None, mins2=None, maxs2=None, gamma=None):
    n, m = sm.shape
    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    block_size = 64
    ni, nj = (n + block_size - 1) // block_size, (m + block_size - 1) // block_size
    for bi in range(ni):
        i_s = bi * block_size; i_e = min(n, i_s + block_size)
        for bj in range(nj):
            j_s = bj * block_size; j_e = min(m, j_s + block_size)
            if only_triu and j_e < i_s - diag_offset: continue
            
            d2_lb = 0.0
            for d in range(len(gamma)):
                diff = 0.0
                if mins1[bi, d] > maxs2[bj, d]: diff = mins1[bi, d] - maxs2[bj, d]
                elif mins2[bj, d] > maxs1[bi, d]: diff = mins2[bj, d] - maxs1[bi, d]
                d2_lb += gamma[d] * diff * diff
            
            if np.exp(-d2_lb) < tau:
                any_inc = False
                for j in range(j_s + 1, j_e + 3):
                    if csm[i_s + 1, j] > 0.0: any_inc = True; break
                if not any_inc:
                    for i in range(i_s + 1, i_e + 3):
                        if csm[i, j_s + 1] > 0.0: any_inc = True; break
                if not any_inc: continue
                
            for i in range(i_s, i_e):
                jj_s = max(j_s, i - diag_offset) if only_triu else j_s
                for j in range(jj_s, j_e):
                    sim = sm[i, j]; ii, jj = i + 2, j + 2
                    prev = csm[ii-1, jj-1]
                    if sim < tau: val = delta_m * prev - delta_a
                    else: val = sim + prev
                    if val > 0.0: csm[ii, jj] = np.float32(val)
    return csm

@njit(cache=True)
def _radix_sort_u32_with_payload(keys, payload):
    n = len(keys)
    tmp_keys = np.empty(n, dtype=np.uint32); tmp_payload = np.empty(n, dtype=payload.dtype)
    counts = np.empty(65536, dtype=np.int64); offsets = np.empty(65536, dtype=np.int64)
    mask = np.uint32(65535)
    for shift in (0, 16):
        counts[:] = 0
        for i in range(n): counts[np.int32((keys[i] >> np.uint32(shift)) & mask)] += 1
        total = np.int64(0)
        for b in range(65536): offsets[b] = total; total += counts[b]
        for i in range(n):
            b = np.int32((keys[i] >> np.uint32(shift)) & mask); p = offsets[b]
            tmp_keys[p] = keys[i]; tmp_payload[p] = payload[i]; offsets[b] = p + 1
        keys, tmp_keys = tmp_keys, keys
        payload, tmp_payload = tmp_payload, payload
    return keys, payload

@njit(cache=True)
def _radix_sort_u64_with_payload(keys, payload):
    n = len(keys)
    tmp_keys = np.empty(n, dtype=np.uint64); tmp_payload = np.empty(n, dtype=payload.dtype)
    counts = np.empty(65536, dtype=np.int64); offsets = np.empty(65536, dtype=np.int64)
    mask = np.uint64(65535)
    for shift in (0, 16, 32, 48):
        counts[:] = 0
        for i in range(n): counts[np.int32((keys[i] >> np.uint64(shift)) & mask)] += 1
        total = np.int64(0)
        for b in range(65536): offsets[b] = total; total += counts[b]
        for i in range(n):
            b = np.int32((keys[i] >> np.uint64(shift)) & mask); p = offsets[b]
            tmp_keys[p] = keys[i]; tmp_payload[p] = payload[i]; offsets[b] = p + 1
        keys, tmp_keys = tmp_keys, keys
        payload, tmp_payload = tmp_payload, payload
    return keys, payload

@njit(cache=True)
def _mask_vicinity(path, mask, vwidth):
    n, m = mask.shape
    for k in range(len(path) - 1):
        ic, jc = path[k, 0], path[k, 1]
        it, jt = path[k + 1, 0], path[k + 1, 1]
        di, dj = it - ic, jt - jc
        i1, i2 = max(0, ic - vwidth), min(n, ic + vwidth + 1)
        j1, j2 = max(0, jc - vwidth), min(m, jc + vwidth + 1)
        mask[i1:i2, jc] = True
        mask[ic, j1:j2] = True
        if di == 2 and dj == 1:
            if ic + 1 < n: mask[ic + 1, jc] = True
            mask[ic + 1, j1:j2] = True
        elif di == 1 and dj == 2:
            if jc + 1 < m: mask[ic, jc + 1] = True
            mask[i1:i2, jc + 1] = True
    ic, jc = path[-1, 0], path[-1, 1]
    mask[max(0, ic - vwidth):min(n, ic + vwidth + 1), jc] = True
    mask[ic, max(0, jc - vwidth):min(m, jc + vwidth + 1)] = True

@njit(cache=True, parallel=True)
def _collect_positive_candidates(csm, mask):
    n, m = csm.shape
    block_size = 16
    ni, nj = (n + block_size - 1) // block_size, (m + block_size - 1) // block_size
    block_counts = np.zeros((ni, nj), dtype=np.int32)
    for bi in prange(ni):
        i_s = max(2, bi * block_size); i_e = min(n, (bi + 1) * block_size)
        for bj in range(nj):
            j_s = max(2, bj * block_size); j_e = min(m, (bj + 1) * block_size)
            cnt = 0
            for i in range(i_s, i_e):
                for j in range(j_s, j_e):
                    if not mask[i, j] and csm[i, j] > 0.0: cnt += 1
            block_counts[bi, bj] = cnt
    total = np.sum(block_counts)
    lp = np.empty(total, dtype=np.int64); vs = np.empty(total, dtype=np.float32)
    flat_counts = block_counts.reshape(ni * nj); offsets = np.zeros(ni * nj, dtype=np.int64); curr = np.int64(0)
    for k in range(ni * nj): offsets[k] = curr; curr += flat_counts[k]
    for bi in prange(ni):
        i_s = max(2, bi * block_size); i_e = min(n, (bi + 1) * block_size)
        for bj in range(nj):
            if block_counts[bi, bj] == 0: continue
            w = offsets[bi * nj + bj]; j_s = max(2, bj * block_size); j_e = min(m, (bj + 1) * block_size)
            for i in range(i_s, i_e):
                for j in range(j_s, j_e):
                    if not mask[i, j] and csm[i, j] > 0.0:
                        lp[w] = np.int64(i) * np.int64(m) + np.int64(j); vs[w] = csm[i, j]; w += 1
    return lp, vs

@njit(cache=True)
def _extract_path_warping(csm, mask, i, j):
    path = []
    while i >= 2 and j >= 2:
        path.append((i, j))
        v1 = csm[i-1, j-1]; v2 = csm[i-2, j-1]; v3 = csm[i-1, j-2]
        maximum = max3(v1, v2, v3)
        if v1 == maximum:
            if mask[i-1, j-1]: break
            i, j = i-1, j-1
        elif v2 == maximum:
            if mask[i-2, j-1]: break
            i, j = i-2, j-1
        else:
            if mask[i-1, j-2]: break
            i, j = i-1, j-2
    res = np.empty((len(path), 2), dtype=np.int32)
    for k in range(len(path)):
        res[len(path)-1-k, 0] = path[k][0]; res[len(path)-1-k, 1] = path[k][1]
    return res

@njit(cache=True)
def _extract_path_no_warping(csm, mask, i, j):
    path = []
    while i >= 2 and j >= 2:
        path.append((i, j))
        if mask[i-1, j-1]: break
        i, j = i-1, j-1
    res = np.empty((len(path), 2), dtype=np.int32)
    for k in range(len(path)):
        res[len(path)-1-k, 0] = path[k][0]; res[len(path)-1-k, 1] = path[k][1]
    return res

@njit(cache=True)
def find_best_paths(csm, mask, tau, l_min=10, vwidth=5, warping=True, bp_dir=None, only_triu=False):
    # Match main branch EXACTLY:
    for i in range(csm.shape[0]):
        for j in range(csm.shape[1]):
            if csm[i, j] <= 0: mask[i, j] = True
    n, m = csm.shape
    linear_pos, values = _collect_positive_candidates(csm, mask)
    if len(linear_pos) == 0: return TypedList.empty_list(int32[:, :])
    sort_keys = np.empty(len(linear_pos), dtype=np.uint64)
    for i in range(len(linear_pos)):
        v_idx = values.view(np.uint32)[i]
        sort_keys[i] = (np.uint64(v_idx) << 32) | np.uint64(linear_pos[i])
    _, sort_keys = _radix_sort_u64_with_payload(sort_keys, linear_pos)
    k_best = len(linear_pos) - 1; paths = TypedList.empty_list(int32[:, :])
    while k_best >= 0:
        path_found = False
        while not path_found:
            while mask[linear_pos[k_best] // m, linear_pos[k_best] % m]:
                k_best -= 1
                if k_best < 0: return paths
            idx = linear_pos[k_best]; i_best = idx // m; j_best = idx % m
            if i_best < 2 or j_best < 2: return paths
            if warping: path = _extract_path_warping(csm, mask, i_best, j_best)
            else: path = _extract_path_no_warping(csm, mask, i_best, j_best)
            _mask_vicinity(path, mask, 0)
            if (path[-1, 0] - path[0, 0] + 1) >= l_min or (path[-1, 1] - path[0, 1] + 1) >= l_min:
                path_found = True
        _mask_vicinity(path, mask, vwidth)
        paths.append(path)
    return paths
