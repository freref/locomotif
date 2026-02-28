import numpy as np


### JIT
from numba import int32, int8, float64, float32, boolean
from numba import njit
from numba.types import List, Array
from numba import prange
from numba.typed import List as TypedList

@njit(cache=True, parallel=True)
def _extract_triu_elements(sm):
    n = sm.shape[0]
    total = n * (n + 1) // 2
    out = np.empty(total, dtype=np.float32)
    
    # Calculate row start offsets mathematically
    row_starts = np.empty(n, dtype=np.int64)
    for i in prange(n):
        # elements before row i: n + (n-1) + ... + (n - i + 1)
        # = i * n - i * (i - 1) // 2
        row_starts[i] = i * n - (i * (i - 1)) // 2

    for i in prange(n):
        idx = row_starts[i]
        for j in range(i, n):
            out[idx] = sm[i, j]
            idx += 1
            
    return out

@njit(float32[:, :](float32[:, :], float32[:, :], float64[:], boolean, int32), parallel=True)
def similarity_matrix_ndim(ts1, ts2, gamma=None, only_triu=False, diag_offset=0):
    n, m = len(ts1), len(ts2)

    sm = np.full((n, m), -np.inf, dtype=np.float32)
    for i in prange(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end   = m
        
        similarities = np.exp(-np.sum(gamma.T * np.power(ts1[i, :] - ts2[j_start:j_end, :], 2), axis=1))
        
        sm[i, j_start:j_end] = similarities

    return sm

@njit
def _best_predecessor(a, b, c):
    if a >= b:
        if a >= c:
            return a, np.int8(0)
        return c, np.int8(2)
    if b >= c:
        return b, np.int8(1)
    return c, np.int8(2)


@njit(cache=True)
def cumulative_similarity_matrix_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)

    for i in range(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m

        for j in range(j_start, j_end):

            sim = sm[i, j]
            ii = i + 2
            jj = j + 2

            pred_score, direction = _best_predecessor(csm[ii - 1, jj - 1], csm[ii - 2, jj - 1], csm[ii - 1, jj - 2])
            if sim < tau:
                val = delta_m * pred_score - delta_a
            else:
                val = sim + pred_score

            if val > 0.0:
                csm[ii, jj] = val
                bp_dir[ii, jj] = direction

    return csm, bp_dir


@njit(cache=True)
def cumulative_similarity_matrix_no_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)

    for i in range(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m

        for j in range(j_start, j_end):

            sim = sm[i, j]
            ii = i + 2
            jj = j + 2
            pred_score = csm[ii - 1, jj - 1]

            if sim < tau:
                val = delta_m * pred_score - delta_a
            else:
                val = sim + pred_score

            if val > 0.0:
                csm[ii, jj] = val
                bp_dir[ii, jj] = np.int8(0)

    return csm, bp_dir


@njit(Array(int32, 2, 'C')(boolean[:, :], int8[:, :], int32, int32), cache=True)
def best_path_from_backpointers(mask, bp_dir, i, j):
    path = []
    while i >= 2 and j >= 2:
        path.append((i, j))

        direction = bp_dir[i, j]
        if direction == 0:
            pi, pj = i - 1, j - 1
        elif direction == 1:
            pi, pj = i - 2, j - 1
        elif direction == 2:
            pi, pj = i - 1, j - 2
        else:
            break

        if mask[pi, pj]:
            break
        i, j = pi, pj

    path.reverse()
    return np.array(path, dtype=np.int32)


@njit(cache=True)
def _trace_path_len_and_start(mask, bp_dir, i, j):
    length = 0
    start_i = i
    start_j = j

    while i >= 2 and j >= 2:
        length += 1
        start_i = i
        start_j = j

        direction = bp_dir[i, j]
        if direction == 0:
            pi, pj = i - 1, j - 1
        elif direction == 1:
            pi, pj = i - 2, j - 1
        elif direction == 2:
            pi, pj = i - 1, j - 2
        else:
            break

        if mask[pi, pj]:
            break
        i, j = pi, pj

    return length, start_i, start_j


@njit(cache=True)
def _trace_path_len_and_start_flat(mask_flat, bp_flat, m, i, j):
    length = 0
    start_i = i
    start_j = j
    lin = i * m + j

    while i >= 2 and j >= 2:
        length += 1
        start_i = i
        start_j = j

        direction = bp_flat[lin]
        if direction == 0:
            i -= 1
            j -= 1
            lin -= m + 1
        elif direction == 1:
            i -= 2
            j -= 1
            lin -= 2 * m + 1
        elif direction == 2:
            i -= 1
            j -= 2
            lin -= m + 2
        else:
            break

        if mask_flat[lin]:
            break

    return length, start_i, start_j


@njit(cache=True)
def _materialize_path_from_backpointers(mask, bp_dir, i, j, length):
    path = np.empty((length, 2), dtype=np.int32)
    k = length - 1
    while i >= 2 and j >= 2 and k >= 0:
        path[k, 0] = i
        path[k, 1] = j
        k -= 1

        direction = bp_dir[i, j]
        if direction == 0:
            pi, pj = i - 1, j - 1
        elif direction == 1:
            pi, pj = i - 2, j - 1
        elif direction == 2:
            pi, pj = i - 1, j - 2
        else:
            break

        if mask[pi, pj]:
            break
        i, j = pi, pj

    return path


@njit(cache=True)
def _materialize_path_from_backpointers_flat(mask_flat, bp_flat, m, i, j, length):
    path = np.empty((length, 2), dtype=np.int32)
    k = length - 1
    lin = i * m + j
    while i >= 2 and j >= 2 and k >= 0:
        path[k, 0] = i
        path[k, 1] = j
        k -= 1

        direction = bp_flat[lin]
        if direction == 0:
            i -= 1
            j -= 1
            lin -= m + 1
        elif direction == 1:
            i -= 2
            j -= 1
            lin -= 2 * m + 1
        elif direction == 2:
            i -= 1
            j -= 2
            lin -= m + 2
        else:
            break

        if mask_flat[lin]:
            break

    return path


@njit(cache=True)
def _mask_buffer_path_zero(mask_flat, m, trace_buf, buf_start):
    for k in range(buf_start, len(trace_buf)):
        i = trace_buf[k, 0]
        j = trace_buf[k, 1]
        lin = i * m + j
        mask_flat[lin] = True
        
        if k > buf_start:
            pi = trace_buf[k-1, 0]
            pj = trace_buf[k-1, 1]
            # Direction 1: i - pi == 2, j - pj == 1 -> intermediate is i-1, j-1
            # Direction 2: i - pi == 1, j - pj == 2 -> intermediate is i-1, j-1
            if (i - pi == 2 and j - pj == 1) or (i - pi == 1 and j - pj == 2):
                mask_flat[(i - 1) * m + (j - 1)] = True

@njit(cache=True)
def _radix_sort_u32_with_payload(keys, payload):
    n = len(keys)
    tmp_keys = np.empty(n, dtype=np.uint32)
    tmp_payload = np.empty(n, dtype=payload.dtype)

    counts = np.empty(65536, dtype=np.int64)
    offsets = np.empty(65536, dtype=np.int64)
    mask = np.uint32(65535)

    for shift in (0, 16):
        counts[:] = 0
        for i in range(n):
            b = np.int32((keys[i] >> np.uint32(shift)) & mask)
            counts[b] += 1

        total = np.int64(0)
        for b in range(65536):
            offsets[b] = total
            total += counts[b]

        for i in range(n):
            key = keys[i]
            b = np.int32((key >> np.uint32(shift)) & mask)
            p = offsets[b]
            tmp_keys[p] = key
            tmp_payload[p] = payload[i]
            offsets[b] = p + 1

        keys, tmp_keys = tmp_keys, keys
        payload, tmp_payload = tmp_payload, payload

    return keys, payload


@njit(boolean[:, :](int32[:, :], boolean[:, :], int32))
def mask_vicinity(path, mask, vwidth=10):
    n, m = mask.shape
    
    for k in range(len(path)-1):
        ic, jc = path[k]
        it, jt = path[k + 1]
        
        di, dj = (it - ic, jt - jc)
        
        i1, i2 = max(0, ic - vwidth), min(n, ic + vwidth + 1)
        j1, j2 = max(0, jc - vwidth), min(m, jc + vwidth + 1)
        
        mask[i1 : i2, jc] = True
        mask[ic, j1 : j2] = True
                
        if di == 2 and dj == 1:
            if i2 + 1 < n:
                mask[ic + 1, jc] = True
            mask[ic + 1, j1 : j2] = True
            
        elif di == 1 and dj == 2:
            if j2 + 1 < m:
                mask[ic, jc + 1] = True
            mask[i1 : i2, jc + 1] = True
            
        else:
            if not (di == 1 and dj == 1):
                raise Exception("Path does not comply to the allowed step sizes")

    (ic, jc) = path[-1]
    mask[max(0, ic - vwidth) : min(n, ic + vwidth + 1), jc] = True
    mask[ic, max(0, jc - vwidth) : min(m, jc + vwidth + 1)] = True
    return mask


@njit(cache=True)
def _mask_vicinity(path, mask, vwidth):
    n, m = mask.shape
    for k in range(len(path) - 1):
        ic = path[k, 0]
        jc = path[k, 1]
        it = path[k + 1, 0]
        jt = path[k + 1, 1]

        di = it - ic
        dj = jt - jc

        i1 = max(0, ic - vwidth)
        i2 = min(n, ic + vwidth + 1)
        j1 = max(0, jc - vwidth)
        j2 = min(m, jc + vwidth + 1)

        mask[i1:i2, jc] = True
        mask[ic, j1:j2] = True

        if di == 2 and dj == 1:
            ii = ic + 1
            if i2 + 1 < n:
                mask[ii, jc] = True
            mask[ii, j1:j2] = True
        elif di == 1 and dj == 2:
            jjc = jc + 1
            if j2 + 1 < m:
                mask[ic, jjc] = True
            mask[i1:i2, jjc] = True

    ic = path[-1, 0]
    jc = path[-1, 1]
    i1 = max(0, ic - vwidth)
    i2 = min(n, ic + vwidth + 1)
    j1 = max(0, jc - vwidth)
    j2 = min(m, jc + vwidth + 1)

    mask[i1:i2, jc] = True
    mask[ic, j1:j2] = True


@njit(cache=True, parallel=True)
def _collect_positive_candidates(csm, mask, bp_dir):
    n, m = csm.shape
    row_counts = np.zeros(n, dtype=np.int32)

    for i in prange(2, n):
        cnt = np.int32(0)
        for j in range(2, m):
            if not mask[i, j] and csm[i, j] > 0.0:
                has_larger_successor = False
                
                # Check (i+1, j+1) direction 0
                if i + 1 < n and j + 1 < m:
                    if bp_dir[i + 1, j + 1] == 0 and csm[i + 1, j + 1] >= csm[i, j]:
                        has_larger_successor = True
                
                # Check (i+2, j+1) direction 1
                if not has_larger_successor and i + 2 < n and j + 1 < m:
                    if bp_dir[i + 2, j + 1] == 1 and csm[i + 2, j + 1] >= csm[i, j]:
                        has_larger_successor = True
                        
                # Check (i+1, j+2) direction 2
                if not has_larger_successor and i + 1 < n and j + 2 < m:
                    if bp_dir[i + 1, j + 2] == 2 and csm[i + 1, j + 2] >= csm[i, j]:
                        has_larger_successor = True

                if not has_larger_successor:
                    cnt += 1
        row_counts[i] = cnt

    total = np.int64(0)
    row_offsets = np.zeros(n, dtype=np.int64)
    for i in range(2, n):
        row_offsets[i] = total
        total += row_counts[i]

    linear_pos = np.empty(total, dtype=np.int64)
    values = np.empty(total, dtype=np.float32)
    for i in prange(2, n):
        cursor = row_offsets[i]
        for j in range(2, m):
            if not mask[i, j] and csm[i, j] > 0.0:
                has_larger_successor = False
                
                if i + 1 < n and j + 1 < m:
                    if bp_dir[i + 1, j + 1] == 0 and csm[i + 1, j + 1] >= csm[i, j]:
                        has_larger_successor = True
                
                if not has_larger_successor and i + 2 < n and j + 1 < m:
                    if bp_dir[i + 2, j + 1] == 1 and csm[i + 2, j + 1] >= csm[i, j]:
                        has_larger_successor = True
                        
                if not has_larger_successor and i + 1 < n and j + 2 < m:
                    if bp_dir[i + 1, j + 2] == 2 and csm[i + 1, j + 2] >= csm[i, j]:
                        has_larger_successor = True

                if not has_larger_successor:
                    linear_pos[cursor] = np.int64(i) * np.int64(m) + np.int64(j)
                    values[cursor] = csm[i, j]
                    cursor += 1

    return linear_pos, values


@njit(cache=True, parallel=True)
def _collect_positive_candidates_pruned(csm, mask, bp_dir, dist, min_dist):
    n, m = csm.shape
    row_counts = np.zeros(n, dtype=np.int32)

    for i in prange(2, n):
        cnt = np.int32(0)
        for j in range(2, m):
            if not mask[i, j] and csm[i, j] > 0.0 and dist[i, j] >= min_dist:
                has_larger_successor = False
                
                if i + 1 < n and j + 1 < m:
                    if bp_dir[i + 1, j + 1] == 0 and csm[i + 1, j + 1] >= csm[i, j]:
                        has_larger_successor = True
                
                if not has_larger_successor and i + 2 < n and j + 1 < m:
                    if bp_dir[i + 2, j + 1] == 1 and csm[i + 2, j + 1] >= csm[i, j]:
                        has_larger_successor = True
                        
                if not has_larger_successor and i + 1 < n and j + 2 < m:
                    if bp_dir[i + 1, j + 2] == 2 and csm[i + 1, j + 2] >= csm[i, j]:
                        has_larger_successor = True

                if not has_larger_successor:
                    cnt += 1
        row_counts[i] = cnt

    total = np.int64(0)
    row_offsets = np.zeros(n, dtype=np.int64)
    for i in range(2, n):
        row_offsets[i] = total
        total += row_counts[i]

    linear_pos = np.empty(total, dtype=np.int64)
    values = np.empty(total, dtype=np.float32)
    for i in prange(2, n):
        cursor = row_offsets[i]
        for j in range(2, m):
            if not mask[i, j] and csm[i, j] > 0.0 and dist[i, j] >= min_dist:
                has_larger_successor = False
                
                if i + 1 < n and j + 1 < m:
                    if bp_dir[i + 1, j + 1] == 0 and csm[i + 1, j + 1] >= csm[i, j]:
                        has_larger_successor = True
                
                if not has_larger_successor and i + 2 < n and j + 1 < m:
                    if bp_dir[i + 2, j + 1] == 1 and csm[i + 2, j + 1] >= csm[i, j]:
                        has_larger_successor = True
                        
                if not has_larger_successor and i + 1 < n and j + 2 < m:
                    if bp_dir[i + 1, j + 2] == 2 and csm[i + 1, j + 2] >= csm[i, j]:
                        has_larger_successor = True

                if not has_larger_successor:
                    linear_pos[cursor] = np.int64(i) * np.int64(m) + np.int64(j)
                    values[cursor] = csm[i, j]
                    cursor += 1

    return linear_pos, values


@njit(cache=True)
def _extract_path_to_buffer(mask_flat, bp_flat, m, i, j, buf):
    length = 0
    lin = i * m + j
    buf_idx = len(buf) - 1

    while i >= 2 and j >= 2:
        buf[buf_idx, 0] = i
        buf[buf_idx, 1] = j
        length += 1
        buf_idx -= 1

        direction = bp_flat[lin]
        if direction == 0:
            i -= 1
            j -= 1
            lin -= m + 1
        elif direction == 1:
            i -= 2
            j -= 1
            lin -= 2 * m + 1
        elif direction == 2:
            i -= 1
            j -= 2
            lin -= m + 2
        else:
            break

        if mask_flat[lin]:
            break

    return length

@njit(cache=True, parallel=True)
def _apply_csm_mask(csm, mask):
    n, m = csm.shape
    for i in prange(n):
        for j in range(m):
            if csm[i, j] <= 0:
                mask[i, j] = True

@njit(cache=True)
def find_best_paths(csm, mask, tau, l_min=10, vwidth=5, warping=True, bp_dir=None):
    # Mask all zeros
    _apply_csm_mask(csm, mask)
    n, m = csm.shape
    mask_flat = mask.reshape(n * m)
    bp_flat = bp_dir.reshape(n * m)
    
    linear_pos, values = _collect_positive_candidates(csm, mask, bp_dir)
    
    candidate_count = len(linear_pos)
    if candidate_count == 0:
        return TypedList.empty_list(int32[:, :])

    paths = TypedList.empty_list(int32[:, :])
    _, linear_pos = _radix_sort_u32_with_payload(values.view(np.uint32), linear_pos)
    k_best = len(linear_pos) - 1

    trace_buf = np.empty((n + m, 2), dtype=np.int32)

    while k_best >= 0:
        path = np.empty((0, 0), dtype=np.int32)
        path_found = False

        while not path_found:
            linear_idx = linear_pos[k_best]
            while mask_flat[linear_idx]:
                k_best -= 1
                if k_best < 0:
                    return paths
                linear_idx = linear_pos[k_best]

            i_best = linear_idx // np.int64(m)
            j_best = linear_idx - i_best * np.int64(m)

            if i_best < 2 or j_best < 2:
                return paths

            path_len = _extract_path_to_buffer(mask_flat, bp_flat, m, i_best, j_best, trace_buf)
            buf_start = len(trace_buf) - path_len
            start_i = trace_buf[buf_start, 0]
            start_j = trace_buf[buf_start, 1]

            if (i_best - start_i + 1) >= l_min or (j_best - start_j + 1) >= l_min:
                path = trace_buf[buf_start : len(trace_buf)].copy()
                path_found = True
            else:
                _mask_buffer_path_zero(mask_flat, m, trace_buf, buf_start)

        _mask_vicinity(path, mask, vwidth)
        paths.append(path)

    return paths
