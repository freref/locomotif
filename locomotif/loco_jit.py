import numpy as np


### JIT
from numba import int32, int8, float64, float32, boolean
from numba import njit
from numba.types import List, Array
from numba import prange
from numba.typed import List as TypedList

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
    src_id = np.full((n + 2, m + 2), np.int32(-1), dtype=np.int32)
    dist = np.zeros((n + 2, m + 2), dtype=np.int32)

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

                if direction == 0:
                    pi, pj = ii - 1, jj - 1
                elif direction == 1:
                    pi, pj = ii - 2, jj - 1
                else:
                    pi, pj = ii - 1, jj - 2

                if csm[pi, pj] > 0.0:
                    src_id[ii, jj] = src_id[pi, pj]
                    dist[ii, jj] = dist[pi, pj] + 1
                else:
                    src_id[ii, jj] = ii * (m + 2) + jj

    return csm, bp_dir, src_id, dist


@njit(cache=True)
def cumulative_similarity_matrix_no_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
    src_id = np.full((n + 2, m + 2), np.int32(-1), dtype=np.int32)
    dist = np.zeros((n + 2, m + 2), dtype=np.int32)

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
                if pred_score > 0.0:
                    src_id[ii, jj] = src_id[ii - 1, jj - 1]
                    dist[ii, jj] = dist[ii - 1, jj - 1] + 1
                else:
                    src_id[ii, jj] = ii * (m + 2) + jj

    return csm, bp_dir, src_id, dist


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
            pi = i - 1
            pj = j - 1
            plin = lin - m - 1
        elif direction == 1:
            pi = i - 2
            pj = j - 1
            plin = lin - 2 * m - 1
        elif direction == 2:
            pi = i - 1
            pj = j - 2
            plin = lin - m - 2
        else:
            break

        if mask_flat[plin]:
            break

        i = pi
        j = pj
        lin = plin

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
            pi = i - 1
            pj = j - 1
            plin = lin - m - 1
        elif direction == 1:
            pi = i - 2
            pj = j - 1
            plin = lin - 2 * m - 1
        elif direction == 2:
            pi = i - 1
            pj = j - 2
            plin = lin - m - 2
        else:
            break

        if mask_flat[plin]:
            break
        i = pi
        j = pj
        lin = plin

    return path


@njit(cache=True)
def _mask_backpointer_path_zero(mask, bp_dir, i, j):
    while i >= 2 and j >= 2:
        mask[i, j] = True

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

        if direction == 1 or direction == 2:
            mask[i - 1, j - 1] = True

        i, j = pi, pj

    return mask


@njit(cache=True)
def _mask_backpointer_path_zero_flat(mask_flat, bp_flat, m, i, j):
    lin = i * m + j
    while i >= 2 and j >= 2:
        mask_flat[lin] = True

        direction = bp_flat[lin]
        if direction == 0:
            pi = i - 1
            pj = j - 1
            plin = lin - m - 1
        elif direction == 1:
            pi = i - 2
            pj = j - 1
            plin = lin - 2 * m - 1
        elif direction == 2:
            pi = i - 1
            pj = j - 2
            plin = lin - m - 2
        else:
            break

        if mask_flat[plin]:
            break

        if direction == 1 or direction == 2:
            mask_flat[lin - m - 1] = True

        i = pi
        j = pj
        lin = plin


@njit(cache=True)
def _radix_sort_u32_with_payload(keys, payload):
    n = len(keys)
    tmp_keys = np.empty(n, dtype=np.uint32)
    tmp_payload = np.empty(n, dtype=np.int32)

    counts = np.empty(65536, dtype=np.int32)
    offsets = np.empty(65536, dtype=np.int32)
    mask = np.uint32(65535)

    for shift in (0, 16):
        counts[:] = 0
        for i in range(n):
            b = np.int32((keys[i] >> np.uint32(shift)) & mask)
            counts[b] += 1

        total = 0
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
def _mask_vicinity_flat(path, mask_flat, n, m, vwidth):
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

        for ii in range(i1, i2):
            mask_flat[ii * m + jc] = True
        row_base = ic * m
        for jj in range(j1, j2):
            mask_flat[row_base + jj] = True

        if di == 2 and dj == 1:
            ii = ic + 1
            if i2 + 1 < n:
                mask_flat[ii * m + jc] = True
            row_base = ii * m
            for jj in range(j1, j2):
                mask_flat[row_base + jj] = True
        elif di == 1 and dj == 2:
            jjc = jc + 1
            if j2 + 1 < m:
                mask_flat[ic * m + jjc] = True
            for ii in range(i1, i2):
                mask_flat[ii * m + jjc] = True
        else:
            if not (di == 1 and dj == 1):
                raise Exception("Path does not comply to the allowed step sizes")

    ic = path[-1, 0]
    jc = path[-1, 1]
    i1 = max(0, ic - vwidth)
    i2 = min(n, ic + vwidth + 1)
    j1 = max(0, jc - vwidth)
    j2 = min(m, jc + vwidth + 1)

    for ii in range(i1, i2):
        mask_flat[ii * m + jc] = True
    row_base = ic * m
    for jj in range(j1, j2):
        mask_flat[row_base + jj] = True


@njit(cache=True, parallel=True)
def _collect_positive_candidates(csm, mask):
    n, m = csm.shape
    row_counts = np.zeros(n, dtype=np.int32)

    for i in prange(2, n):
        cnt = np.int32(0)
        for j in range(2, m):
            if not mask[i, j] and csm[i, j] > 0.0:
                cnt += 1
        row_counts[i] = cnt

    total = np.int32(0)
    row_offsets = np.zeros(n, dtype=np.int32)
    for i in range(2, n):
        row_offsets[i] = total
        total += row_counts[i]

    linear_pos = np.empty(total, dtype=np.int32)
    values = np.empty(total, dtype=np.float32)
    for i in prange(2, n):
        cursor = row_offsets[i]
        for j in range(2, m):
            if not mask[i, j] and csm[i, j] > 0.0:
                linear_pos[cursor] = i * m + j
                values[cursor] = csm[i, j]
                cursor += 1

    return linear_pos, values


@njit(cache=True, parallel=True)
def _collect_positive_candidates_pruned(csm, mask, dist, min_dist):
    n, m = csm.shape
    row_counts = np.zeros(n, dtype=np.int32)

    for i in prange(2, n):
        cnt = np.int32(0)
        for j in range(2, m):
            if not mask[i, j] and csm[i, j] > 0.0 and dist[i, j] >= min_dist:
                cnt += 1
        row_counts[i] = cnt

    total = np.int32(0)
    row_offsets = np.zeros(n, dtype=np.int32)
    for i in range(2, n):
        row_offsets[i] = total
        total += row_counts[i]

    linear_pos = np.empty(total, dtype=np.int32)
    values = np.empty(total, dtype=np.float32)
    for i in prange(2, n):
        cursor = row_offsets[i]
        for j in range(2, m):
            if not mask[i, j] and csm[i, j] > 0.0 and dist[i, j] >= min_dist:
                linear_pos[cursor] = i * m + j
                values[cursor] = csm[i, j]
                cursor += 1

    return linear_pos, values


@njit(cache=True)
def find_best_paths(csm, mask, tau, l_min=10, vwidth=5, warping=True, bp_dir=None, src_id=None, dist=None):
    # Mask all zeros
    mask = mask | (csm <= 0)
    n, m = csm.shape
    mask_flat = mask.reshape(n * m)
    bp_flat = bp_dir.reshape(n * m)
    if dist is None:
        linear_pos, values = _collect_positive_candidates(csm, mask)
    else:
        linear_pos, values = _collect_positive_candidates_pruned(csm, mask, dist, np.int32(l_min // 2))
    candidate_count = len(linear_pos)
    if candidate_count == 0:
        return TypedList.empty_list(int32[:, :])

    paths = TypedList.empty_list(int32[:, :])
    _, linear_pos = _radix_sort_u32_with_payload(values.view(np.uint32), linear_pos)
    k_best = len(linear_pos) - 1

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

            i_best = linear_idx // m
            j_best = linear_idx - i_best * m

            if i_best < 2 or j_best < 2:
                return paths

            src = src_id[i_best, j_best]
            if src >= 0:
                start_i = src // m
                start_j = src - start_i * m
                if (i_best - start_i + 1) < l_min and (j_best - start_j + 1) < l_min:
                    _mask_backpointer_path_zero_flat(mask_flat, bp_flat, m, i_best, j_best)
                    continue

            path_len, start_i, start_j = _trace_path_len_and_start_flat(mask_flat, bp_flat, m, i_best, j_best)
            if (i_best - start_i + 1) >= l_min or (j_best - start_j + 1) >= l_min:
                path = _materialize_path_from_backpointers_flat(mask_flat, bp_flat, m, i_best, j_best, path_len)
                path_found = True
            else:
                _mask_backpointer_path_zero_flat(mask_flat, bp_flat, m, i_best, j_best)

        _mask_vicinity_flat(path, mask_flat, n, m, vwidth)
        paths.append(path)

    return paths
