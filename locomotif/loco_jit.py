import numpy as np


### JIT
from numba import int32, float64, float32, boolean
from numba import njit, types, objmode
from numba.types import List, Array
from numba import prange

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
        
@njit(types.Tuple((
        types.float32[:, :], 
        types.int32[:, :],
        types.int32[:, :, :]
    ))(float32[:, :], int32, float64, float64, float64, boolean, int32))
def cumulative_similarity_matrix_warping(sm, l_min=10, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    dist = np.zeros((n + 2, m + 2), dtype=np.int32)
    bp = np.full((n + 2, m + 2, 2), -1, dtype=np.int32)

    for i in range(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m

        for j in range(j_start, j_end):

            sim = sm[i, j]

            pred_diag = csm[i + 1, j + 1]
            pred_left = csm[i + 1, j]
            pred_up = csm[i, j + 1]

            pred_max = max3(pred_diag, pred_left, pred_up)

            if pred_max == pred_diag:
                pred_coord = (i+1, j+1)
            elif pred_max == pred_left:
                pred_coord = (i+1, j)
            else:
                pred_coord = (i, j+1)
            
            if sim < tau:
                csm[i + 2, j + 2] = max(0, delta_m * pred_max - delta_a)
            else:
                csm[i + 2, j + 2] = max(0, sim + pred_max)
            
            cur = csm[i + 2, j + 2]
            if pred_max > 0 and cur > 0:
                pi, pj = pred_coord

                bp[i + 2, j + 2, 0] = pi
                bp[i + 2, j + 2, 1] = pj

                dist[i+2, j+2] = dist[pi, pj] + 1
    return csm, dist, bp

@njit(float32[:, :](float32[:, :], float64, float64, float64, boolean, int32))
def cumulative_similarity_matrix_no_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)

    for i in range(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m

        for j in range(j_start, j_end):

            sim = sm[i, j]

            if sim < tau:
                csm[i + 2, j + 2] = max(0, delta_m * csm[i - 1 + 2, j - 1 + 2] - delta_a)
            else:
                csm[i + 2, j + 2] = max(0, sim + csm[i - 1 + 2, j - 1 + 2])

    return csm


@njit(Array(int32, 2, 'C')(int32[:, :, :], boolean[:, :], int32, int32))
def best_path_warping(bp, mask, i, j):
    path = []
    while i >= 0 and j >= 0:
        path.append((i, j))
        pi = bp[i, j, 0]
        pj = bp[i, j, 1]
        if pi < 0 or pj < 0:
            break
        if mask[pi, pj]:
            break
        if pi == i and pj == j:
            break
        i, j = pi, pj
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

@njit
def update_dist(mask, dist, bp):
    n, m = dist.shape
    for i in range(n):
        for j in range(m):
            if mask[i, j]:
                dist[i, j] = 0
            else:
                pi = bp[i, j, 0]
                pj = bp[i, j, 1]
                if pi >= 0 and pj >= 0:
                    dist[i, j] = dist[pi, pj] + 1
                else:
                    dist[i, j] = 0
    
    return dist

@njit(List(Array(int32, 2, 'C'))(float32[:, :], int32[:, :], int32[:, :, :], boolean[:, :], float32, int32, int32, boolean))
def find_best_paths(csm, dist, bp, mask, tau, l_min=10, vwidth=5, warping=True):
    paths = []
    mask_v = np.copy(mask)      # vicinity mask
    mask_d = np.copy(mask)      # distance mask

    while True:
        start_mask = (~(mask_v | mask_d)) & (dist >= l_min)
        pos_i, pos_j = np.nonzero(start_mask)
        if len(pos_i) == 0:
            break

        values = np.array([csm[pos_i[k], pos_j[k]] for k in range(len(pos_i))])

        if len(values) == 0:
            break

        k_best = np.argmax(values)
        i_best, j_best = pos_i[k_best], pos_j[k_best]
        if csm[i_best, j_best] <= 0:
            break

        if warping:
            path = best_path_warping(bp, mask_v, i_best, j_best)
        else:
            path = best_path_no_warping(csm, mask_v, i_best, j_best)

        mask_v = mask_vicinity(path, mask_v, vwidth)
        dist = update_dist(mask_v, dist, bp)
        mask_d = (dist < l_min)

        paths.append(path)
    
    return paths
