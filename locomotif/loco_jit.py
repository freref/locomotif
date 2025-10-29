import numpy as np


### JIT
from numba import int32, float64, float32, boolean
from numba import njit
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
        
# @njit(float32[:, :](float32[:, :], float64, float64, float64, boolean, int32))
def cumulative_similarity_matrix_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    min_point_matrix = np.full((n + 2, m + 2, 2), -1, dtype=(np.int32))
    # TODO only keep max value after euclid distance between max point and min point > l_min
    min_point_to_max_point = {}

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

                # we check if the previous cel has a min point and if min_point_to_max_point 
                # has already been initialised, if not we set the current value to the min_point of
                # our predecessor, we do this so we init only start points that have a path
                pred_min = min_point_matrix[pred_coord[0], pred_coord[1]]
                if pred_min[0] != -1 and pred_min[1] != -1:
                    k = (int(pred_min[0]), int(pred_min[1]))
                    # TODO change this to l_min
                    dist = np.linalg.norm(np.array([i+2, j+2]) - np.array(pred_min))
                    if k not in min_point_to_max_point and dist > 100:
                        min_point_to_max_point[k] = (i+2, j+2)



            if pred_max == 0:
                min_point_matrix[i + 1, j + 1] = - 1
                min_point_matrix[i + 1, j] = - 1
                min_point_matrix[i, j + 1] = - 1
                min_point_matrix[i+2, j+2] = (i+2, j+2)
                # min_point_to_max_point[(i+2, j+2)] = (i+2, j+2)
            else:
                min_point_matrix[i+2, j+2] = min_point_matrix[pred_coord[0], pred_coord[1]]

            key = tuple(min_point_matrix[i+2, j+2])
            if key[0] != -1 and key[1] != -1:
                max_point = min_point_to_max_point.get(key)
                if max_point is not None:
                    max_value = csm[max_point[0], max_point[1]]
                    if max_value < csm[i + 2, j + 2]:
                        min_point_to_max_point[key] = (i+2, j+2)

    import matplotlib.pyplot as plt

    # draw on a blank matrix with same shape (or use csm instead of bg if you want the heatmap)
    bg = np.zeros_like(csm)

    plt.figure(figsize=(6, 6))
    plt.imshow(bg, cmap="gray", origin="upper")

    for min_pt, max_pt in min_point_to_max_point.items():
        y1, x1 = min_pt
        y2, x2 = max_pt
        plt.plot([x1, x2], [y1, y2], linewidth=1)  # color cycles by default

    plt.title("Lines from min→max mapping")
    plt.xlim(-0.5, bg.shape[1]-0.5)
    plt.ylim(bg.shape[0]-0.5, -0.5)  # keep origin='upper'
    plt.tight_layout()
    plt.show()
    
    return csm

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


@njit(List(Array(int32, 2, 'C'))(float32[:, :], boolean[:, :], float32, int32, int32, boolean))
def find_best_paths(csm, mask, tau, l_min=10, vwidth=5, warping=True):
    # Mask all zeros
    mask = mask | (csm <= 0)
    
    # min_path_length = l_min if not warping else np.ceil(l_min / 2)
    start_mask = (~mask) # & (csm >= tau * min_path_length)
    
    pos_i, pos_j = np.nonzero(start_mask)
    
    values = np.array([csm[pos_i[k], pos_j[k]] for k in range(len(pos_i))])
    perm = np.argsort(values)
    sorted_pos_i, sorted_pos_j = pos_i[perm], pos_j[perm]

    k_best = len(sorted_pos_i) - 1
    paths = []

    while k_best >= 0:

        path = np.empty((0, 0), dtype=np.int32)
        path_found = False

        while not path_found:

            while (mask[sorted_pos_i[k_best], sorted_pos_j[k_best]]):
                k_best -= 1
                if k_best < 0:
                    return paths
                
            i_best, j_best = sorted_pos_i[k_best], sorted_pos_j[k_best]

            if i_best < 2 or j_best < 2:
                return paths
            
            if warping:
                path = best_path_warping(csm, mask, i_best, j_best)
            else:
                path = best_path_no_warping(csm, mask, i_best, j_best)
                
            mask = mask_vicinity(path, mask, 0)
            # mask = mask_path(path, mask)
            
            if (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min:
                path_found = True


        mask = mask_vicinity(path, mask, vwidth)
        paths.append(path)

    return paths
