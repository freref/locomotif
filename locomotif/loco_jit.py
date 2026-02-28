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

@njit(cache=True, parallel=True)
def _collect_above_threshold(sm, threshold, only_triu):
    n, m = sm.shape
    local_counts = np.zeros(n, dtype=np.int32)
    for i in prange(n):
        js = i if only_triu else 0
        cnt = 0
        for j in range(js, m):
            if sm[i, j] >= threshold: cnt += 1
        local_counts[i] = cnt
    offsets = np.zeros(n, dtype=np.int64)
    curr = 0
    for i in range(n):
        offsets[i] = curr
        curr += local_counts[i]
    out = np.empty(curr, dtype=np.float32)
    for i in prange(n):
        js = i if only_triu else 0
        w = offsets[i]
        for j in range(js, m):
            if sm[i, j] >= threshold:
                out[w] = sm[i, j]
                w += 1
    return out

@njit(cache=True)
def _exact_tau_smart(sm, rho, only_triu):
    n, m = sm.shape
    total_elements = np.int64(n) * np.int64(n + 1) // 2 if only_triu else np.int64(n) * np.int64(m)
    h = (total_elements - 1) * rho
    step = max(1, n // 100)
    sample_size = 0
    for i in range(0, n, step):
        js = i if only_triu else 0
        sample_size += (m - js)
    sample = np.empty(sample_size, dtype=np.float32); curr = 0
    for i in range(0, n, step):
        js = i if only_triu else 0
        for j in range(js, m): sample[curr] = sm[i, j]; curr += 1
    s_idx = int(np.floor(rho * (sample_size - 1)))
    s_thresh = np.partition(sample, s_idx)[s_idx]
    thresh = s_thresh * 0.99
    collected = _collect_above_threshold(sm, thresh, only_triu)
    while len(collected) < (total_elements - h + 5):
        thresh *= 0.9
        collected = _collect_above_threshold(sm, thresh, only_triu)
    count_below = total_elements - len(collected)
    k_lo = int(np.floor(h - count_below)); k_hi = int(np.ceil(h - count_below))
    if k_lo == k_hi: return np.partition(collected, k_lo)[k_lo]
    else:
        p = np.partition(collected, (k_lo, k_hi))
        v_lo, v_hi = p[k_lo], p[k_hi]
        weight = h - np.floor(h)
        return (1.0 - weight) * v_lo + weight * v_hi

@njit(float32[:, :](float32[:, :], float32[:, :], float64[:], boolean, int32), parallel=True)
def similarity_matrix_ndim(ts1, ts2, gamma=None, only_triu=False, diag_offset=0):
    n, m = len(ts1), len(ts2)
    sm = np.full((n, m), -np.inf, dtype=np.float32)
    for i in prange(n):
        j_start = max(0, i-diag_offset) if only_triu else 0
        similarities = np.exp(-np.sum(gamma.T * np.power(ts1[i, :] - ts2[j_start:m, :], 2), axis=1))
        sm[i, j_start:m] = similarities
    return sm

@njit
def _best_predecessor(a, b, c):
    if a >= b:
        if a >= c: return a, np.int8(0)
        return c, np.int8(2)
    if b >= c: return b, np.int8(1)
    return c, np.int8(2)

@njit(cache=True)
def cumulative_similarity_matrix_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0, mins1=None, maxs1=None, mins2=None, maxs2=None, gamma=None):
    n, m = sm.shape
    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
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
                    ps, dr = _best_predecessor(csm[ii - 1, jj - 1], csm[ii - 2, jj - 1], csm[ii - 1, jj - 2])
                    val = sim + ps if sim >= tau else delta_m * ps - delta_a
                    if val > 0.0: csm[ii, jj], bp_dir[ii, jj] = val, dr
    return csm, bp_dir

@njit(cache=True)
def cumulative_similarity_matrix_no_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0, mins1=None, maxs1=None, mins2=None, maxs2=None, gamma=None):
    n, m = sm.shape
    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
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
                    ps = csm[ii - 1, jj - 1]
                    val = sim + ps if sim >= tau else delta_m * ps - delta_a
                    if val > 0.0: csm[ii, jj], bp_dir[ii, jj] = val, np.int8(0)
    return csm, bp_dir

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
def _mask_vicinity(path, mask, vwidth):
    n, m = mask.shape
    for k in range(len(path)):
        ic, jc = path[k, 0], path[k, 1]
        i1, i2 = max(0, ic - vwidth), min(n, ic + vwidth + 1); j1, j2 = max(0, jc - vwidth), min(m, jc + vwidth + 1)
        mask[i1:i2, jc] = True; mask[ic, j1:j2] = True
        if k < len(path) - 1:
            it, jt = path[k+1, 0], path[k+1, 1]; di, dj = it - ic, jt - jc
            if di == 2 and dj == 1:
                ii = ic + 1
                if ii < n: mask[ii, jc] = True; mask[ii, j1:j2] = True
            elif di == 1 and dj == 2:
                jj = jc + 1
                if jj < m: mask[ic, jj] = True; mask[i1:i2, jj] = True

@njit(cache=True, parallel=True)
def _collect_positive_candidates(csm, mask, bp_dir):
    n, m = csm.shape
    block_size = 16
    ni, nj = (n + block_size - 1) // block_size, (m + block_size - 1) // block_size
    
    # 1. Coarse scan to find active blocks
    active_blocks = np.zeros((ni, nj), dtype=np.bool_)
    for bi in prange(ni):
        i_s = max(2, bi * block_size)
        i_e = min(n, i_s + block_size)
        for bj in range(nj):
            j_s = max(2, bj * block_size)
            j_e = min(m, j_s + block_size)
            
            found = False
            for i in range(i_s, i_e):
                for j in range(j_s, j_e):
                    if not mask[i, j] and csm[i, j] > 0.0:
                        found = True; break
                if found: break
            active_blocks[bi, bj] = found

    # 2. Fine scan in active blocks
    row_counts = np.zeros(n, dtype=np.int32)
    for bi in prange(ni):
        i_s = max(2, bi * block_size)
        i_e = min(n, i_s + block_size)
        for bj in range(nj):
            if not active_blocks[bi, bj]: continue
            j_s = max(2, bj * block_size)
            j_e = min(m, j_s + block_size)
            
            for i in range(i_s, i_e):
                cnt = 0
                for j in range(j_s, j_e):
                    if not mask[i, j] and csm[i, j] > 0.0:
                        ok = True
                        if i+1 < n and j+1 < m and bp_dir[i+1, j+1] == 0 and csm[i+1, j+1] >= csm[i, j]: ok = False
                        if ok and i+2 < n and j+1 < m and bp_dir[i+2, j+1] == 1 and csm[i+2, j+1] >= csm[i, j]: ok = False
                        if ok and i+1 < n and j+2 < m and bp_dir[i+1, j+2] == 2 and csm[i+1, j+2] >= csm[i, j]: ok = False
                        if ok: cnt += 1
                # Numba doesn't like atomic add on array element in nested loop easily, 
                # but we can use a local variable and then add to row_counts[i]
                # Wait, row_counts[i] is indexed by i, which is unique to each bi thread if i_s/i_e don't overlap.
                # Since bi is in prange, i is unique.
                if cnt > 0:
                    # row_counts[i] += cnt # This is safe because i is unique to this bi
                    # Wait, multiple bj could cover the same row i!
                    # So we need to accumulate bj results for each row i.
                    pass
    
    # Let's rewrite the fine scan to be more robust.
    # We'll just do one pass over rows, and for each row check only active bj.
    for i in prange(2, n):
        bi = i // block_size
        cnt = 0
        for bj in range(nj):
            if active_blocks[bi, bj]:
                j_s = max(2, bj * block_size)
                j_e = min(m, j_s + block_size)
                for j in range(j_s, j_e):
                    if not mask[i, j] and csm[i, j] > 0.0:
                        ok = True
                        if i+1 < n and j+1 < m and bp_dir[i+1, j+1] == 0 and csm[i+1, j+1] >= csm[i, j]: ok = False
                        if ok and i+2 < n and j+1 < m and bp_dir[i+2, j+1] == 1 and csm[i+2, j+1] >= csm[i, j]: ok = False
                        if ok and i+1 < n and j+2 < m and bp_dir[i+1, j+2] == 2 and csm[i+1, j+2] >= csm[i, j]: ok = False
                        if ok: cnt += 1
        row_counts[i] = cnt

    total = np.sum(row_counts)
    row_offsets = np.zeros(n, dtype=np.int64); curr = np.int64(0)
    for i in range(n): row_offsets[i] = curr; curr += row_counts[i]
    lp = np.empty(total, dtype=np.int64); vs = np.empty(total, dtype=np.float32)
    for i in prange(2, n):
        bi = i // block_size
        w = row_offsets[i]
        for bj in range(nj):
            if active_blocks[bi, bj]:
                j_s = max(2, bj * block_size)
                j_e = min(m, j_s + block_size)
                for j in range(j_s, j_e):
                    if not mask[i, j] and csm[i, j] > 0.0:
                        ok = True
                        if i+1 < n and j+1 < m and bp_dir[i+1, j+1] == 0 and csm[i+1, j+1] >= csm[i, j]: ok = False
                        if ok and i+2 < n and j+1 < m and bp_dir[i+2, j+1] == 1 and csm[i+2, j+1] >= csm[i, j]: ok = False
                        if ok and i+1 < n and j+2 < m and bp_dir[i+1, j+2] == 2 and csm[i+1, j+2] >= csm[i, j]: ok = False
                        if ok: lp[w] = np.int64(i) * np.int64(m) + np.int64(j); vs[w] = csm[i, j]; w += 1
    return lp, vs

@njit(cache=True)
def _extract_path_to_buffer(mask_flat, bp_flat, m, i, j, buf):
    length = 0; lin = i * m + j; buf_idx = len(buf) - 1
    while i >= 2 and j >= 2:
        buf[buf_idx, 0], buf[buf_idx, 1] = i, j
        length += 1; buf_idx -= 1
        d = bp_flat[lin]
        if d == 0: i -= 1; j -= 1; lin -= m + 1
        elif d == 1: i -= 2; j -= 1; lin -= 2 * m + 1
        elif d == 2: i -= 1; j -= 2; lin -= m + 2
        else: break
        if mask_flat[lin]: break
    return length

@njit(cache=True, parallel=True)
def _apply_csm_mask(csm, mask):
    n, m = csm.shape
    for i in prange(n):
        for j in range(m):
            if csm[i, j] <= 0: mask[i, j] = True

@njit(cache=True)
def _mask_buffer_path_zero(mask_flat, m, trace_buf, buf_start):
    for k in range(buf_start, len(trace_buf)):
        i, j = trace_buf[k, 0], trace_buf[k, 1]; mask_flat[i * m + j] = True
        if k > buf_start:
            pi, pj = trace_buf[k-1, 0], trace_buf[k-1, 1]
            if (i - pi == 2 and j - pj == 1) or (i - pi == 1 and j - pj == 2): mask_flat[(i - 1) * m + (j - 1)] = True

@njit(cache=True)
def find_best_paths(csm, mask, tau, l_min=10, vwidth=5, warping=True, bp_dir=None):
    _apply_csm_mask(csm, mask)
    n, m = csm.shape
    mask_flat, bp_flat = mask.reshape(n * m), bp_dir.reshape(n * m)
    linear_pos, values = _collect_positive_candidates(csm, mask, bp_dir)
    if len(linear_pos) == 0: return TypedList.empty_list(int32[:, :])
    _, linear_pos = _radix_sort_u32_with_payload(values.view(np.uint32), linear_pos)
    k_best = len(linear_pos) - 1; paths = TypedList.empty_list(int32[:, :]); trace_buf = np.empty((n + m, 2), dtype=np.int32)
    while k_best >= 0:
        linear_idx = linear_pos[k_best]; k_best -= 1
        while mask_flat[linear_idx]:
            if k_best < 0: return paths
            linear_idx = linear_pos[k_best]; k_best -= 1
        i_best = linear_idx // np.int64(m); j_best = linear_idx - i_best * np.int64(m)
        if i_best < 2 or j_best < 2: return paths
        path_len = _extract_path_to_buffer(mask_flat, bp_flat, m, i_best, j_best, trace_buf)
        buf_start = len(trace_buf) - path_len
        if (i_best - trace_buf[buf_start, 0] + 1) >= l_min or (j_best - trace_buf[buf_start, 1] + 1) >= l_min:
            path = trace_buf[buf_start:].copy(); _mask_vicinity(path, mask, vwidth); paths.append(path)
        else: _mask_buffer_path_zero(mask_flat, m, trace_buf, buf_start)
    return paths
