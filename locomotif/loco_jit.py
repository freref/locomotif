import numpy as np


### JIT
from numba import int32, int8, float64, float32, boolean
from numba import njit
from numba.types import List, Array
from numba import prange
from numba.typed import List as TypedList

@njit(float32[:, :](float32[:, :], float32[:, :], float64[:], boolean, int32))
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
                else:
                    src_id[ii, jj] = ii * (m + 2) + jj

    return csm, bp_dir, src_id


@njit(cache=True)
def cumulative_similarity_matrix_no_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
    src_id = np.full((n + 2, m + 2), np.int32(-1), dtype=np.int32)

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
                else:
                    src_id[ii, jj] = ii * (m + 2) + jj

    return csm, bp_dir, src_id


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
def _radix_argsort_u32(keys):
    n = len(keys)
    idx = np.empty(n, dtype=np.int32)
    tmp = np.empty(n, dtype=np.int32)
    for i in range(n):
        idx[i] = i

    counts = np.empty(256, dtype=np.int32)
    offsets = np.empty(256, dtype=np.int32)

    for shift in (0, 8, 16, 24):
        counts[:] = 0
        for i in range(n):
            b = np.int32((keys[idx[i]] >> np.uint32(shift)) & np.uint32(255))
            counts[b] += 1

        total = 0
        for b in range(256):
            offsets[b] = total
            total += counts[b]

        for i in range(n):
            ii = idx[i]
            b = np.int32((keys[ii] >> np.uint32(shift)) & np.uint32(255))
            p = offsets[b]
            tmp[p] = ii
            offsets[b] = p + 1

        idx, tmp = tmp, idx

    return idx


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
def find_best_paths(csm, mask, tau, l_min=10, vwidth=5, warping=True, bp_dir=None, src_id=None):
    # Mask all zeros
    mask = mask | (csm <= 0)
    n, m = csm.shape

    candidate_count = 0
    for i in range(2, n):
        for j in range(2, m):
            if mask[i, j] or csm[i, j] <= 0.0:
                continue
            source = src_id[i, j]
            if source < 0:
                continue
            source_i = source // m
            source_j = source - source_i * m
            if (i - source_i + 1) >= l_min or (j - source_j + 1) >= l_min:
                candidate_count += 1

    if candidate_count == 0:
        return TypedList.empty_list(int32[:, :])

    linear_pos = np.empty(candidate_count, dtype=np.int32)
    values = np.empty(candidate_count, dtype=np.float32)
    cursor = 0
    for i in range(2, n):
        for j in range(2, m):
            if mask[i, j] or csm[i, j] <= 0.0:
                continue
            source = src_id[i, j]
            if source < 0:
                continue
            source_i = source // m
            source_j = source - source_i * m
            if (i - source_i + 1) >= l_min or (j - source_j + 1) >= l_min:
                linear_pos[cursor] = i * m + j
                values[cursor] = csm[i, j]
                cursor += 1

    perm = _radix_argsort_u32(values.view(np.uint32))

    k_best = len(perm) - 1
    paths = TypedList.empty_list(int32[:, :])

    while k_best >= 0:
        path = np.empty((0, 0), dtype=np.int32)
        path_found = False

        while not path_found:
            linear_idx = linear_pos[perm[k_best]]
            i_best = linear_idx // m
            j_best = linear_idx - i_best * m
            while mask[i_best, j_best]:
                k_best -= 1
                if k_best < 0:
                    return paths
                linear_idx = linear_pos[perm[k_best]]
                i_best = linear_idx // m
                j_best = linear_idx - i_best * m

            if i_best < 2 or j_best < 2:
                return paths

            path_len, start_i, start_j = _trace_path_len_and_start(mask, bp_dir, i_best, j_best)
            if (i_best - start_i + 1) >= l_min or (j_best - start_j + 1) >= l_min:
                path = _materialize_path_from_backpointers(mask, bp_dir, i_best, j_best, path_len)
                path_found = True
            else:
                mask = _mask_backpointer_path_zero(mask, bp_dir, i_best, j_best)

        mask = mask_vicinity(path, mask, vwidth)
        paths.append(path)
    return paths
