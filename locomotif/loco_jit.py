import numpy as np


### JIT
from numba import int32, float64, float32, boolean, uint8
from numba import njit, types
from numba.types import List, Array
from numba import prange

@njit(float32[:, :](float32[:, :], float32[:, :], float64[:], boolean, int32), parallel=True, fastmath=True)
def similarity_matrix_ndim(ts1, ts2, gamma=None, only_triu=False, diag_offset=0):
    n, m = len(ts1), len(ts2)
    d = ts1.shape[1]

    sm = np.full((n, m), -np.inf, dtype=np.float32)
    for i in prange(n):
        j_start = max(0, i - diag_offset) if only_triu else 0
        for j in range(j_start, m):
            acc = 0.0
            for k in range(d):
                diff = ts1[i, k] - ts2[j, k]
                acc += gamma[k] * diff * diff
            sm[i, j] = np.float32(np.exp(-acc))

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
    ))(float32[:, :], int32, float64, float64, float64, boolean, int32), fastmath=True)
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

            if pred_diag >= pred_left:
                if pred_diag >= pred_up:
                    pred_max = pred_diag
                    pi, pj = i + 1, j + 1
                else:
                    pred_max = pred_up
                    pi, pj = i, j + 1
            else:
                if pred_left >= pred_up:
                    pred_max = pred_left
                    pi, pj = i + 1, j
                else:
                    pred_max = pred_up
                    pi, pj = i, j + 1
            
            if sim < tau:
                csm[i + 2, j + 2] = max(0, delta_m * pred_max - delta_a)
            else:
                csm[i + 2, j + 2] = max(0, sim + pred_max)
            
            cur = csm[i + 2, j + 2]
            if pred_max > 0 and cur > 0:
                dist[i+2, j+2] = dist[pi, pj] + 1

    return csm, dist

@njit(types.Tuple((
        types.float32[:, :],
        types.int32[:, :],
        types.uint8[:, :],
    ))(float32[:, :], int32, float64, float64, float64, boolean, int32), fastmath=True)
def cumulative_similarity_matrix_warping_bp(sm, l_min=10, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    dist = np.zeros((n + 2, m + 2), dtype=np.int32)
    bp = np.zeros((n + 2, m + 2), dtype=np.uint8)

    for i in range(n):
        j_start = max(0, i - diag_offset) if only_triu else 0

        for j in range(j_start, m):
            sim = sm[i, j]

            pred_diag = csm[i + 1, j + 1]
            pred_left = csm[i + 1, j]
            pred_up = csm[i, j + 1]

            if pred_diag >= pred_left:
                if pred_diag >= pred_up:
                    pred_max = pred_diag
                    pi, pj = i + 1, j + 1
                else:
                    pred_max = pred_up
                    pi, pj = i, j + 1
            else:
                if pred_left >= pred_up:
                    pred_max = pred_left
                    pi, pj = i + 1, j
                else:
                    pred_max = pred_up
                    pi, pj = i, j + 1

            if pred_diag >= pred_up:
                if pred_diag >= pred_left:
                    bp[i + 2, j + 2] = np.uint8(1)
                else:
                    bp[i + 2, j + 2] = np.uint8(3)
            else:
                if pred_up >= pred_left:
                    bp[i + 2, j + 2] = np.uint8(2)
                else:
                    bp[i + 2, j + 2] = np.uint8(3)

            if sim < tau:
                cur = delta_m * pred_max - delta_a
                if cur <= 0:
                    cur = 0.0
            else:
                cur = sim + pred_max
                if cur <= 0:
                    cur = 0.0

            csm[i + 2, j + 2] = cur
            if pred_max > 0 and cur > 0:
                dist[i + 2, j + 2] = dist[pi, pj] + 1
            
    return csm, dist, bp

@njit(float32[:, :](float32[:, :], float64, float64, float64, boolean, int32), fastmath=True)
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

@njit(types.Tuple((
        types.float32[:, :],
        types.int32[:, :],
        types.uint8[:, :],
    ))(float32[:, :], float64, float64, float64, boolean, int32), fastmath=True)
def cumulative_similarity_matrix_no_warping_bp(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    dist = np.zeros((n + 2, m + 2), dtype=np.int32)
    bp = np.zeros((n + 2, m + 2), dtype=np.uint8)

    for i in range(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m

        for j in range(j_start, j_end):

            sim = sm[i, j]

            bp[i + 2, j + 2] = np.uint8(1)
            pred_diag = csm[i + 1, j + 1]

            if sim < tau:
                cur = max(0, delta_m * pred_diag - delta_a)
            else:
                cur = max(0, sim + pred_diag)

            csm[i + 2, j + 2] = cur
            if pred_diag > 0 and cur > 0:
                dist[i + 2, j + 2] = dist[i + 1, j + 1] + 1

    return csm, dist, bp

@njit(Array(int32, 2, 'C')(float32[:, :], boolean[:, :], int32, int32, int32))
def best_path_warping(csm, mask, i, j, max_len):
    path_rev = np.empty((max_len, 2), dtype=np.int32)
    path_len = 0

    while i >= 2 and j >= 2:
        path_rev[path_len, 0] = i
        path_rev[path_len, 1] = j
        path_len += 1

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

    path = np.empty((path_len, 2), dtype=np.int32)
    for k in range(path_len):
        path[k, 0] = path_rev[path_len - 1 - k, 0]
        path[k, 1] = path_rev[path_len - 1 - k, 1]
    return path

@njit(Array(int32, 2, 'C')(float32[:, :], boolean[:, :], int32, int32))
def best_path_no_warping(csm, mask, i, j):
    max_len = i if i < j else j
    path_rev = np.empty((max_len, 2), dtype=np.int32)
    path_len = 0

    while i >= 2 and j >= 2:
        path_rev[path_len, 0] = i
        path_rev[path_len, 1] = j
        path_len += 1

        if mask[i - 1, j - 1]:
            break

        i, j = i - 1, j - 1

    path = np.empty((path_len, 2), dtype=np.int32)
    for k in range(path_len):
        path[k, 0] = path_rev[path_len - 1 - k, 0]
        path[k, 1] = path_rev[path_len - 1 - k, 1]
    return path

@njit(Array(int32, 2, 'C')(uint8[:, :], boolean[:, :], int32, int32, int32))
def best_path_warping_bp(bp, mask, i, j, max_len):
    path_rev = np.empty((max_len, 2), dtype=np.int32)
    path_len = 0

    while i >= 2 and j >= 2:
        path_rev[path_len, 0] = i
        path_rev[path_len, 1] = j
        path_len += 1

        direction = bp[i, j]
        if direction == 1:
            ni, nj = i - 1, j - 1
        elif direction == 2:
            ni, nj = i - 2, j - 1
        elif direction == 3:
            ni, nj = i - 1, j - 2
        else:
            ni, nj = i - 1, j - 1

        if ni < 0 or nj < 0:
            break
        if mask[ni, nj]:
            break
        i, j = ni, nj

    path = np.empty((path_len, 2), dtype=np.int32)
    for k in range(path_len):
        path[k, 0] = path_rev[path_len - 1 - k, 0]
        path[k, 1] = path_rev[path_len - 1 - k, 1]
    return path

@njit(Array(int32, 2, 'C')(uint8[:, :], boolean[:, :], int32, int32))
def best_path_no_warping_bp(bp, mask, i, j):
    max_len = i if i < j else j
    path_rev = np.empty((max_len, 2), dtype=np.int32)
    path_len = 0

    while i >= 2 and j >= 2:
        path_rev[path_len, 0] = i
        path_rev[path_len, 1] = j
        path_len += 1

        direction = bp[i, j]
        if direction == 1:
            ni, nj = i - 1, j - 1
        else:
            ni, nj = i - 1, j - 1

        if mask[ni, nj]:
            break
        i, j = ni, nj

    path = np.empty((path_len, 2), dtype=np.int32)
    for k in range(path_len):
        path[k, 0] = path_rev[path_len - 1 - k, 0]
        path[k, 1] = path_rev[path_len - 1 - k, 1]
    return path

@njit(boolean[:, :](int32[:, :], boolean[:, :]))
def mask_path(path, mask):
    for k in range(len(path)-1):
        ic, jc = path[k]
        it, jt = path[k + 1]
        mask[ic, jc] = True

        di, dj = (it - ic, jt - jc)
        if di == 2 and dj == 1:
            mask[ic + 1, jc] = True
        elif di == 1 and dj == 2:
            mask[ic, jc + 1] = True
        elif not (di == 1 and dj == 1):
            raise Exception("Path does not comply to the allowed step sizes")

    ic, jc = path[-1]
    mask[ic, jc] = True
    return mask


@njit(int32[:](float32[:]))
def radix_argsort_float32(values):
    n = len(values)
    order = np.arange(n, dtype=np.int32)
    if n <= 1:
        return order

    bits = values.view(np.uint32)
    keys = np.empty(n, dtype=np.uint32)
    sign_bit = np.uint32(0x80000000)
    for i in range(n):
        b = bits[i]
        if b & sign_bit:
            keys[i] = ~b
        else:
            keys[i] = b ^ sign_bit

    scratch = np.empty(n, dtype=np.int32)
    counts = np.zeros(256, dtype=np.int32)
    offsets = np.empty(256, dtype=np.int32)

    for pass_id in range(4):
        shift = pass_id * 8
        counts[:] = 0

        for i in range(n):
            k = keys[order[i]]
            b = np.int32((k >> shift) & np.uint32(0xFF))
            counts[b] += 1

        total = 0
        for b in range(256):
            offsets[b] = total
            total += counts[b]

        for i in range(n):
            idx = order[i]
            k = keys[idx]
            b = np.int32((k >> shift) & np.uint32(0xFF))
            pos = offsets[b]
            scratch[pos] = idx
            offsets[b] = pos + 1

        tmp = order
        order = scratch
        scratch = tmp

    return order


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

@njit(boolean(float32, int32, float32, int32))
def _pair_is_greater(v1, c1, v2, c2):
    if v1 > v2:
        return True
    if v1 < v2:
        return False
    return c1 > c2

@njit
def _heap3_sift_down(values, cells, tiles, start, size):
    i = start
    while True:
        left = 2 * i + 1
        if left >= size:
            break
        right = left + 1
        largest = left
        if right < size and _pair_is_greater(values[right], cells[right], values[left], cells[left]):
            largest = right
        if _pair_is_greater(values[i], cells[i], values[largest], cells[largest]):
            break
        tmp_v = values[i]
        tmp_c = cells[i]
        tmp_t = tiles[i]
        values[i] = values[largest]
        cells[i] = cells[largest]
        tiles[i] = tiles[largest]
        values[largest] = tmp_v
        cells[largest] = tmp_c
        tiles[largest] = tmp_t
        i = largest

@njit
def _heap3_build(values, cells, tiles, size):
    for i in range(size // 2 - 1, -1, -1):
        _heap3_sift_down(values, cells, tiles, i, size)

@njit(types.Tuple((float32, int32, int32, int32))(float32[:], int32[:], int32[:], int32))
def _heap3_pop_max(values, cells, tiles, size):
    top_value = values[0]
    top_cell = cells[0]
    top_tile = tiles[0]
    size -= 1
    if size > 0:
        values[0] = values[size]
        cells[0] = cells[size]
        tiles[0] = tiles[size]
        _heap3_sift_down(values, cells, tiles, 0, size)
    return top_value, top_cell, top_tile, size

@njit(int32(float32[:], int32[:], int32[:], int32, float32, int32, int32))
def _heap3_push(values, cells, tiles, size, value, cell, tile):
    i = size
    size += 1
    values[i] = value
    cells[i] = cell
    tiles[i] = tile
    while i > 0:
        p = (i - 1) // 2
        if _pair_is_greater(values[p], cells[p], values[i], cells[i]):
            break
        tmp_v = values[p]
        tmp_c = cells[p]
        tmp_t = tiles[p]
        values[p] = values[i]
        cells[p] = cells[i]
        tiles[p] = tiles[i]
        values[i] = tmp_v
        cells[i] = tmp_c
        tiles[i] = tmp_t
        i = p
    return size

@njit(types.Tuple((float32, int32))(float32[:, :], boolean[:, :], boolean[:, :], int32))
def _row_best_candidate(csm, eligible, mask, row):
    n_cols = csm.shape[1]
    base = row * n_cols
    best_value = np.float32(-np.inf)
    best_cell = np.int32(-1)

    for j in range(n_cols):
        if mask[row, j]:
            continue
        if not eligible[row, j]:
            continue
        cell = np.int32(base + j)
        value = csm[row, j]
        if _pair_is_greater(value, cell, best_value, best_cell):
            best_value = value
            best_cell = cell
    return best_value, best_cell

@njit
def _mask_path_mark_rows(path, mask, dirty_rows):
    for k in range(len(path) - 1):
        ic, jc = path[k]
        it, jt = path[k + 1]
        mask[ic, jc] = True
        dirty_rows[ic] = True

        di, dj = (it - ic, jt - jc)
        if di == 2 and dj == 1:
            mask[ic + 1, jc] = True
            dirty_rows[ic + 1] = True
        elif di == 1 and dj == 2:
            mask[ic, jc + 1] = True
            dirty_rows[ic] = True
        elif not (di == 1 and dj == 1):
            raise Exception("Path does not comply to the allowed step sizes")

    ic, jc = path[-1]
    mask[ic, jc] = True
    dirty_rows[ic] = True

@njit
def _mask_vicinity_mark_rows(path, mask, dirty_rows, vwidth=10):
    n, m = mask.shape

    for k in range(len(path) - 1):
        ic, jc = path[k]
        it, jt = path[k + 1]
        di, dj = (it - ic, jt - jc)

        i1, i2 = max(0, ic - vwidth), min(n, ic + vwidth + 1)
        j1, j2 = max(0, jc - vwidth), min(m, jc + vwidth + 1)

        mask[i1:i2, jc] = True
        dirty_rows[i1:i2] = True
        mask[ic, j1:j2] = True
        dirty_rows[ic] = True

        if di == 2 and dj == 1:
            if i2 + 1 < n:
                mask[ic + 1, jc] = True
                dirty_rows[ic + 1] = True
            mask[ic + 1, j1:j2] = True
            dirty_rows[ic + 1] = True

        elif di == 1 and dj == 2:
            if j2 + 1 < m:
                mask[ic, jc + 1] = True
                dirty_rows[ic] = True
            mask[i1:i2, jc + 1] = True
            dirty_rows[i1:i2] = True

        else:
            if not (di == 1 and dj == 1):
                raise Exception("Path does not comply to the allowed step sizes")

    ic, jc = path[-1]
    i1, i2 = max(0, ic - vwidth), min(n, ic + vwidth + 1)
    j1, j2 = max(0, jc - vwidth), min(m, jc + vwidth + 1)
    mask[i1:i2, jc] = True
    dirty_rows[i1:i2] = True
    mask[ic, j1:j2] = True
    dirty_rows[ic] = True

@njit(List(Array(int32, 2, 'C'))(float32[:, :], int32[:, :], uint8[:, :], boolean[:, :], float32, int32, int32, boolean), fastmath=True)
def find_best_paths_row_frontier_exact(csm, dist, bp, mask, tau, l_min=10, vwidth=5, warping=True):
    mask = mask | (csm <= 0)
    eligible = (dist >= l_min) & (csm > 0)

    n_rows, n_cols = csm.shape
    dirty_rows = np.zeros(n_rows, dtype=np.bool_)

    heap_values = np.empty(n_rows, dtype=np.float32)
    heap_cells = np.empty(n_rows, dtype=np.int32)
    heap_rows = np.empty(n_rows, dtype=np.int32)
    size = np.int32(0)

    for row in range(n_rows):
        value, cell = _row_best_candidate(csm, eligible, mask, row)
        if cell >= 0:
            heap_values[size] = value
            heap_cells[size] = cell
            heap_rows[size] = row
            size += 1

    _heap3_build(heap_values, heap_cells, heap_rows, size)
    paths = []

    while size > 0:
        path = np.empty((0, 0), dtype=np.int32)
        path_found = False

        while not path_found:
            while True:
                if size <= 0:
                    return paths

                _, cell, row, size = _heap3_pop_max(heap_values, heap_cells, heap_rows, size)

                if dirty_rows[row]:
                    dirty_rows[row] = False
                    cur_value, cur_cell = _row_best_candidate(csm, eligible, mask, row)
                    if cur_cell >= 0:
                        size = _heap3_push(heap_values, heap_cells, heap_rows, size, cur_value, cur_cell, row)
                    continue

                i_best = np.int32(cell // n_cols)
                j_best = np.int32(cell - i_best * n_cols)
                if mask[i_best, j_best] or not eligible[i_best, j_best]:
                    cur_value, cur_cell = _row_best_candidate(csm, eligible, mask, row)
                    if cur_cell >= 0:
                        size = _heap3_push(heap_values, heap_cells, heap_rows, size, cur_value, cur_cell, row)
                    continue
                break

            if i_best < 2 or j_best < 2:
                return paths

            if warping:
                path = best_path_warping_bp(bp, mask, i_best, j_best, dist[i_best, j_best] + 1)
            else:
                path = best_path_no_warping_bp(bp, mask, i_best, j_best)

            _mask_path_mark_rows(path, mask, dirty_rows)

            if (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min:
                path_found = True

        _mask_vicinity_mark_rows(path, mask, dirty_rows, vwidth)
        paths.append(path)

    return paths

@njit(List(Array(int32, 2, 'C'))(float32[:, :], int32[:, :], boolean[:, :], float32, int32, int32, boolean))
def find_best_paths(csm, dist, mask, tau, l_min=10, vwidth=5, warping=True):
    mask = mask | (csm <= 0)

    start_mask = (~mask) & (dist >= l_min)
    pos = np.flatnonzero(start_mask).astype(np.int32)
    values = csm.ravel()[pos]
    order = radix_argsort_float32(values)
    k_best = len(order) - 1

    n_cols = csm.shape[1]
    paths = []

    while k_best >= 0:

        path = np.empty((0, 0), dtype=np.int32)
        path_found = False

        while not path_found:
            while True:
                if k_best < 0:
                    return paths
                local_idx = order[k_best]
                k_best -= 1
                cell = pos[local_idx]
                i_best = np.int32(cell // n_cols)
                j_best = np.int32(cell - i_best * n_cols)
                if not mask[i_best, j_best]:
                    break

            if i_best < 2 or j_best < 2:
                return paths
            
            if warping:
                path = best_path_warping(csm, mask, i_best, j_best, dist[i_best, j_best] + 1)
            else:
                path = best_path_no_warping(csm, mask, i_best, j_best)
                
            mask = mask_path(path, mask)
            
            if (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min:
                path_found = True


        mask = mask_vicinity(path, mask, vwidth)
        paths.append(path)

    return paths
