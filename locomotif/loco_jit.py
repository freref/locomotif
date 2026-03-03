import numpy as np


### JIT
from numba import int32, int8, float64, float32, boolean
from numba import njit
from numba.types import Array
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

@njit(cache=True, parallel=True)
def _collect_above_threshold(sm, threshold, only_triu):
    n, m = sm.shape
    local_counts = np.zeros(n, dtype=np.int32)
    for i in prange(n):
        js = i if only_triu else 0
        cnt = 0
        for j in range(js, m):
            if sm[i, j] >= threshold:
                cnt += 1
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
        sample_size += m - js
    sample = np.empty(sample_size, dtype=np.float32)
    curr = 0
    for i in range(0, n, step):
        js = i if only_triu else 0
        for j in range(js, m):
            sample[curr] = sm[i, j]
            curr += 1
    s_idx = int(np.floor(rho * (sample_size - 1)))
    s_thresh = np.partition(sample, s_idx)[s_idx]
    thresh = s_thresh * 0.99
    collected = _collect_above_threshold(sm, thresh, only_triu)
    while len(collected) < (total_elements - h + 5):
        thresh *= 0.9
        collected = _collect_above_threshold(sm, thresh, only_triu)
    count_below = total_elements - len(collected)
    k_lo = int(np.floor(h - count_below))
    k_hi = int(np.ceil(h - count_below))
    if k_lo == k_hi:
        return np.partition(collected, k_lo)[k_lo]
    p = np.partition(collected, (k_lo, k_hi))
    v_lo, v_hi = p[k_lo], p[k_hi]
    weight = h - np.floor(h)
    return (1.0 - weight) * v_lo + weight * v_hi


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
        else:
            return c, np.int8(2)
    else:
        if b >= c:
            return b, np.int8(1)
        else:
            return c, np.int8(2)

@njit
def _build_path_warping(bp_dir, mask, i, j):
    length = 0
    ii = i
    jj = j
    while ii >= 2 and jj >= 2:
        length += 1
        direction = bp_dir[ii, jj]
        if direction == 0:
            if mask[ii - 1, jj - 1]:
                break
            ii, jj = ii - 1, jj - 1
        elif direction == 1:
            if mask[ii - 2, jj - 1]:
                break
            ii, jj = ii - 2, jj - 1
        elif direction == 2:
            if mask[ii - 1, jj - 2]:
                break
            ii, jj = ii - 1, jj - 2
        else:
            break

    path = np.empty((length, 2), dtype=np.int32)
    idx = length - 1
    while i >= 2 and j >= 2:
        path[idx, 0] = i
        path[idx, 1] = j
        idx -= 1
        direction = bp_dir[i, j]
        if direction == 0:
            if mask[i - 1, j - 1]:
                break
            i, j = i - 1, j - 1
        elif direction == 1:
            if mask[i - 2, j - 1]:
                break
            i, j = i - 2, j - 1
        elif direction == 2:
            if mask[i - 1, j - 2]:
                break
            i, j = i - 1, j - 2
        else:
            break
    return path

@njit
def _build_path_no_warping(mask, i, j):
    length = 0
    ii = i
    jj = j
    while ii >= 2 and jj >= 2:
        length += 1
        if mask[ii - 1, jj - 1]:
            break
        ii, jj = ii - 1, jj - 1

    path = np.empty((length, 2), dtype=np.int32)
    idx = length - 1
    while i >= 2 and j >= 2:
        path[idx, 0] = i
        path[idx, 1] = j
        idx -= 1
        if mask[i - 1, j - 1]:
            break
        i, j = i - 1, j - 1
    return path

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

            max_cs, direction = _best_predecessor(csm[i - 1 + 2, j - 1 + 2], csm[i - 2 + 2, j - 1 + 2], csm[i - 1 + 2, j - 2 + 2])

            if sim < tau:
                csm[i + 2, j + 2] = max(0, delta_m * max_cs - delta_a)
            else:
                csm[i + 2, j + 2] = max(0, sim + max_cs)
            if csm[i + 2, j + 2] > 0:
                bp_dir[i + 2, j + 2] = direction
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

            if sim < tau:
                csm[i + 2, j + 2] = max(0, delta_m * csm[i - 1 + 2, j - 1 + 2] - delta_a)
            else:
                csm[i + 2, j + 2] = max(0, sim + csm[i - 1 + 2, j - 1 + 2])
            if csm[i + 2, j + 2] > 0:
                bp_dir[i + 2, j + 2] = 0

    return csm, bp_dir


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
def find_best_paths(csm, mask, tau, l_min=10, vwidth=5, warping=True, bp_dir=None):
    mask = mask | (csm <= 0)
    start_mask = (~mask)
    pos_i, pos_j = np.nonzero(start_mask)
    values = np.empty(len(pos_i), dtype=np.float32)
    for k in range(len(pos_i)):
        values[k] = csm[pos_i[k], pos_j[k]]
    perm = np.argsort(values)
    sorted_pos_i, sorted_pos_j = pos_i[perm], pos_j[perm]
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

            i_best, j_best = sorted_pos_i[k_best], sorted_pos_j[k_best]
            k_best -= 1

            if i_best < 2 or j_best < 2:
                return paths

            if warping and bp_dir is not None:
                path = _build_path_warping(bp_dir, mask, i_best, j_best)
            elif warping:
                path = best_path_warping(csm, mask, i_best, j_best)
            else:
                path = _build_path_no_warping(mask, i_best, j_best)

            mask = mask_vicinity(path, mask, 0)
            if (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min:
                path_found = True

        mask = mask_vicinity(path, mask, vwidth)
        paths.append(path)

    return paths
