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
def max3(a, b, c):
    if a >= b:
        if a >= c:
            return a
        else:
            return c
    else:
        if b >= c:
            return b
        else:
            return c
        
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


@njit(Array(int32, 2, 'C')(float32[:, :], boolean[:, :], int32, int32))
def best_path_warping(csm, mask, i, j):
    
    path = []
    while i >= 2 and j >= 2:

        path.append((i, j))

        maximum = max3(csm[i - 1, j - 1], csm[i - 2, j - 1], csm[i - 1, j - 2])

        if csm[i - 1, j - 1] == maximum:
            if mask[i - 1, j - 1]:
                break
            i, j = i - 1, j - 1
        elif csm[i - 2, j - 1] == maximum:
            if mask[i - 2, j - 1]:
                break
            i, j = i - 2, j - 1
        else:
            if mask[i - 1, j - 2]:
                break
            i, j = i - 1, j - 2

    path.reverse()
    return np.array(path, dtype=np.int32)

@njit(Array(int32, 2, 'C')(float32[:, :], boolean[:, :], int32, int32))
def best_path_no_warping(csm, mask, i, j):
    
    path = []
    while i >= 2 and j >= 2:

        path.append((i, j))

        if mask[i - 1, j - 1]:
            break

        i, j = i - 1, j - 1

    path.reverse()
    return np.array(path, dtype=np.int32)


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
    start_mask = np.zeros((n, m), dtype=np.bool_)
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
                start_mask[i, j] = True

    pos_i, pos_j = np.nonzero(start_mask)
    if len(pos_i) == 0:
        return TypedList.empty_list(int32[:, :])

    values = np.array([csm[pos_i[k], pos_j[k]] for k in range(len(pos_i))], dtype=np.float32)
    perm = np.argsort(values)
    sorted_pos_i = pos_i[perm]
    sorted_pos_j = pos_j[perm]

    k_best = len(sorted_pos_i) - 1
    paths = TypedList.empty_list(int32[:, :])

    while k_best >= 0:
        path = np.empty((0, 0), dtype=np.int32)
        path_found = False

        while not path_found:
            while mask[sorted_pos_i[k_best], sorted_pos_j[k_best]]:
                k_best -= 1
                if k_best < 0:
                    return paths

            i_best = sorted_pos_i[k_best]
            j_best = sorted_pos_j[k_best]

            if i_best < 2 or j_best < 2:
                return paths

            path = best_path_from_backpointers(mask, bp_dir, i_best, j_best)
            mask = mask_vicinity(path, mask, 0)

            if (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min:
                path_found = True

        mask = mask_vicinity(path, mask, vwidth)
        paths.append(path)
    return paths
