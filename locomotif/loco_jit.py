import numpy as np


### JIT
from numba import int32, float64, float32, boolean
from numba import njit, types
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
    ))(float32[:, :], int32, float64, float64, float64, boolean, int32))
def cumulative_similarity_matrix_warping(sm, l_min=10, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    dist = np.zeros((n + 2, m + 2), dtype=np.int32)

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
                dist[i+2, j+2] = dist[pi, pj] + 1
            
    return csm, dist

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


from numba import njit, objmode
import numpy as np, time
from numba.types import List, Array, int32, float32, boolean

@njit(List(Array(int32, 2, 'C'))(float32[:, :], int32[:, :], boolean[:, :], float32, int32, int32, boolean))
def find_best_paths(csm, dist, mask, tau, l_min=10, vwidth=5, warping=True):
    # Timers
    t_start = 0.0
    t_nonzero = 0.0
    t_values = 0.0
    t_argmax = 0.0  # here "argmax" includes sorting/selection
    t_path = 0.0
    t_updates = 0.0
    t_append = 0.0

    # Mask all zeros
    with objmode(ts='f8'): ts = time.perf_counter()
    mask = mask | (csm <= 0)
    with objmode(te='f8'): te = time.perf_counter()
    t_updates += te - ts

    # start_mask
    with objmode(ts='f8'): ts = time.perf_counter()
    start_mask = (~mask) & (dist >= l_min)
    with objmode(te='f8'): te = time.perf_counter()
    t_start += te - ts

    # nonzero
    with objmode(ts='f8'): ts = time.perf_counter()
    pos_i, pos_j = np.nonzero(start_mask)
    with objmode(te='f8'): te = time.perf_counter()
    t_nonzero += te - ts

    # values
    with objmode(ts='f8'): ts = time.perf_counter()
    values = np.array([csm[pos_i[k], pos_j[k]] for k in range(len(pos_i))])
    with objmode(te='f8'): te = time.perf_counter()
    t_values += te - ts

    # selection / sorting
    with objmode(ts='f8'): ts = time.perf_counter()
    perm = np.argsort(values)
    sorted_pos_i, sorted_pos_j = pos_i[perm], pos_j[perm]
    k_best = len(sorted_pos_i) - 1
    with objmode(te='f8'): te = time.perf_counter()
    t_argmax += te - ts

    paths = []

    while k_best >= 0:
        path = np.empty((0, 0), dtype=np.int32)
        path_found = False

        while not path_found:

            # advance k_best while masked
            with objmode(ts='f8'): ts = time.perf_counter()
            while (mask[sorted_pos_i[k_best], sorted_pos_j[k_best]]):
                k_best -= 1
                if k_best < 0:
                    # print times before early return
                    with objmode():
                        total = t_start+t_nonzero+t_values+t_argmax+t_path+t_updates+t_append
                        print("Times (s): start_mask",round(t_start,6),
                              "nonzero",round(t_nonzero,6),
                              "values",round(t_values,6),
                              "select",round(t_argmax,6),
                              "path",round(t_path,6),
                              "updates",round(t_updates,6),
                              "append",round(t_append,6),
                              "total",round(total,6))
                    return paths
            with objmode(te='f8'): te = time.perf_counter()
            t_argmax += te - ts  # count the selection walk as part of "argmax/select"

            i_best, j_best = sorted_pos_i[k_best], sorted_pos_j[k_best]

            if i_best < 2 or j_best < 2:
                with objmode():
                    total = t_start+t_nonzero+t_values+t_argmax+t_path+t_updates+t_append
                    print("Times (s): start_mask",round(t_start,6),
                          "nonzero",round(t_nonzero,6),
                          "values",round(t_values,6),
                          "select",round(t_argmax,6),
                          "path",round(t_path,6),
                          "updates",round(t_updates,6),
                          "append",round(t_append,6),
                          "total",round(total,6))
                return paths

            # build path
            with objmode(ts='f8'): ts = time.perf_counter()
            if warping:
                path = best_path_warping(csm, mask, i_best, j_best)
            else:
                path = best_path_no_warping(csm, mask, i_best, j_best)
            with objmode(te='f8'): te = time.perf_counter()
            t_path += te - ts

            # immediate updates (mask vicinity with width 0)
            with objmode(ts='f8'): ts = time.perf_counter()
            mask = mask_vicinity(path, mask, 0)
            with objmode(te='f8'): te = time.perf_counter()
            t_updates += te - ts

            # check min length
            if (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min:
                path_found = True

        # broader updates and append
        with objmode(ts='f8'): ts = time.perf_counter()
        mask = mask_vicinity(path, mask, vwidth)
        with objmode(te='f8'): te = time.perf_counter()
        t_updates += te - ts

        with objmode(ts='f8'): ts = time.perf_counter()
        paths.append(path)
        with objmode(te='f8'): te = time.perf_counter()
        t_append += te - ts

    # final timing report
    with objmode():
        total = t_start+t_nonzero+t_values+t_argmax+t_path+t_updates+t_append
        print("Times (s): start_mask",round(t_start,6),
              "nonzero",round(t_nonzero,6),
              "values",round(t_values,6),
              "select",round(t_argmax,6),
              "path",round(t_path,6),
              "updates",round(t_updates,6),
              "append",round(t_append,6),
              "total",round(total,6))

    return paths
