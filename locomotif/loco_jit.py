import numpy as np


### JIT
from numba import int32, int64, int8, float64, float32, boolean
from numba import njit
from numba.types import Array
from numba import prange
from numba.typed import List as TypedList

@njit(float32[:, :](float32[:, :], float32[:, :], float64[:], boolean, int32), parallel=True)
def similarity_matrix_ndim(ts1, ts2, gamma=None, only_triu=False, diag_offset=0):
    n, m = len(ts1), len(ts2)

    sm = np.full((n, m), -np.inf, dtype=np.float32)
    for i in prange(n):
        j_start = max(0, i-diag_offset) if only_triu else 0
        similarities = np.exp(-np.sum(gamma.T * np.power(ts1[i, :] - ts2[j_start:m, :], 2), axis=1))
        sm[i, j_start:m] = similarities

    return sm

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
            mi = ts[start, j]
            ma = ts[start, j]
            for i in range(start + 1, end):
                v = ts[i, j]
                if v < mi:
                    mi = v
                if v > ma:
                    ma = v
            mins[b, j] = mi
            maxs[b, j] = ma
    return mins, maxs

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
def _extract_path_to_buffer(mask_flat, bp_flat, m, i, j, buf):
    length = 0
    lin = i * m + j
    buf_idx = len(buf) - 1
    while i >= 2 and j >= 2:
        buf[buf_idx, 0] = i
        buf[buf_idx, 1] = j
        length += 1
        buf_idx -= 1
        d = bp_flat[lin]
        if d == 0:
            i -= 1
            j -= 1
            lin -= m + 1
        elif d == 1:
            i -= 2
            j -= 1
            lin -= 2 * m + 1
        elif d == 2:
            i -= 1
            j -= 2
            lin -= m + 2
        else:
            break
        if mask_flat[lin]:
            break
    return length

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
            counts[np.int32((keys[i] >> np.uint32(shift)) & mask)] += 1
        total = np.int64(0)
        for b in range(65536):
            offsets[b] = total
            total += counts[b]
        for i in range(n):
            b = np.int32((keys[i] >> np.uint32(shift)) & mask)
            p = offsets[b]
            tmp_keys[p] = keys[i]
            tmp_payload[p] = payload[i]
            offsets[b] = p + 1
        keys, tmp_keys = tmp_keys, keys
        payload, tmp_payload = tmp_payload, payload
    return keys, payload

@njit(boolean(float32, int64, float32, int64))
def _pair_is_greater(v1, c1, v2, c2):
    if v1 > v2:
        return True
    if v1 < v2:
        return False
    return c1 < c2

@njit(cache=True)
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

@njit(cache=True)
def _heap3_build(values, cells, tiles, size):
    for i in range(size // 2 - 1, -1, -1):
        _heap3_sift_down(values, cells, tiles, i, size)

@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
def _tile_best_candidate(csm, active, tile, n_tile_cols, tile_size):
    n_rows, n_cols = csm.shape
    tr = tile // n_tile_cols
    tc = tile - tr * n_tile_cols
    i_start = tr * tile_size
    j_start = tc * tile_size
    i_end = min(n_rows, i_start + tile_size)
    j_end = min(n_cols, j_start + tile_size)
    best_value = np.float32(-np.inf)
    best_cell = np.int64(-1)
    for i in range(i_start, i_end):
        base = np.int64(i) * np.int64(n_cols)
        for j in range(j_start, j_end):
            if not active[i, j]:
                continue
            value = csm[i, j]
            cell = base + np.int64(j)
            if _pair_is_greater(value, cell, best_value, best_cell):
                best_value = value
                best_cell = cell
    return best_value, best_cell

@njit(cache=True, parallel=True)
def _tile_best_candidates_parallel(csm, active, n_tiles, n_tile_cols, tile_size):
    n_rows, n_cols = csm.shape
    values = np.full(n_tiles, np.float32(-np.inf), dtype=np.float32)
    cells = np.full(n_tiles, np.int64(-1), dtype=np.int64)
    for tile in prange(n_tiles):
        tr = tile // n_tile_cols
        tc = tile - tr * n_tile_cols
        i_start = tr * tile_size
        j_start = tc * tile_size
        i_end = min(n_rows, i_start + tile_size)
        j_end = min(n_cols, j_start + tile_size)
        best_value = np.float32(-np.inf)
        best_cell = np.int64(-1)
        for i in range(i_start, i_end):
            base = np.int64(i) * np.int64(n_cols)
            for j in range(j_start, j_end):
                if not active[i, j]:
                    continue
                value = csm[i, j]
                cell = base + np.int64(j)
                if _pair_is_greater(value, cell, best_value, best_cell):
                    best_value = value
                    best_cell = cell
        values[tile] = best_value
        cells[tile] = best_cell
    return values, cells

_SMALL_QUICKSORT = 15
_MAX_QUICKSORT_STACK = 100
_MAX_PARALLEL_SORT_SEGMENTS = 64
_MIN_PARALLEL_SORT_SEGMENT = 1000000

@njit(cache=True)
def _insertion_sort_pairs(vals, payload, low, high):
    if high <= low:
        return

    for i in range(low + 1, high + 1):
        kv = vals[i]
        kp = payload[i]
        j = i
        while j > low and kv < vals[j - 1]:
            vals[j] = vals[j - 1]
            payload[j] = payload[j - 1]
            j -= 1
        vals[j] = kv
        payload[j] = kp

@njit(cache=True)
def _partition_pairs(vals, payload, low, high):
    mid = (low + high) >> 1

    if vals[mid] < vals[low]:
        vals[low], vals[mid] = vals[mid], vals[low]
        payload[low], payload[mid] = payload[mid], payload[low]
    if vals[high] < vals[mid]:
        vals[high], vals[mid] = vals[mid], vals[high]
        payload[high], payload[mid] = payload[mid], payload[high]
    if vals[mid] < vals[low]:
        vals[low], vals[mid] = vals[mid], vals[low]
        payload[low], payload[mid] = payload[mid], payload[low]

    pivot = vals[mid]
    vals[high], vals[mid] = vals[mid], vals[high]
    payload[high], payload[mid] = payload[mid], payload[high]

    i = low
    j = high - 1
    while True:
        while i < high and vals[i] < pivot:
            i += 1
        while j >= low and pivot < vals[j]:
            j -= 1
        if i >= j:
            break
        vals[i], vals[j] = vals[j], vals[i]
        payload[i], payload[j] = payload[j], payload[i]
        i += 1
        j -= 1

    vals[i], vals[high] = vals[high], vals[i]
    payload[i], payload[high] = payload[high], payload[i]
    return i

@njit(cache=True)
def _sort_pairs_by_value(vals, payload):
    n = len(vals)
    if n < 2:
        return

    stack_low = np.empty(_MAX_QUICKSORT_STACK, dtype=np.int64)
    stack_high = np.empty(_MAX_QUICKSORT_STACK, dtype=np.int64)
    stack_low[0] = 0
    stack_high[0] = n - 1
    top = 1

    while top > 0:
        top -= 1
        low = stack_low[top]
        high = stack_high[top]

        while high - low >= _SMALL_QUICKSORT:
            i = _partition_pairs(vals, payload, low, high)
            if high - i > i - low:
                if high > i:
                    stack_low[top] = i + 1
                    stack_high[top] = high
                    top += 1
                high = i - 1
            else:
                if i > low:
                    stack_low[top] = low
                    stack_high[top] = i - 1
                    top += 1
                low = i + 1

        _insertion_sort_pairs(vals, payload, low, high)

@njit(cache=True)
def _sort_pairs_segment(vals, payload, low0, high0):
    if high0 <= low0:
        return

    stack_low = np.empty(_MAX_QUICKSORT_STACK, dtype=np.int64)
    stack_high = np.empty(_MAX_QUICKSORT_STACK, dtype=np.int64)
    stack_low[0] = low0
    stack_high[0] = high0
    top = 1

    while top > 0:
        top -= 1
        low = stack_low[top]
        high = stack_high[top]

        while high - low >= _SMALL_QUICKSORT:
            i = _partition_pairs(vals, payload, low, high)
            if high - i > i - low:
                if high > i:
                    stack_low[top] = i + 1
                    stack_high[top] = high
                    top += 1
                high = i - 1
            else:
                if i > low:
                    stack_low[top] = low
                    stack_high[top] = i - 1
                    top += 1
                low = i + 1

        _insertion_sort_pairs(vals, payload, low, high)

@njit(cache=True, parallel=True)
def _sort_pairs_segments_parallel(vals, payload, lows, highs, seg_count):
    for s in prange(seg_count):
        _sort_pairs_segment(vals, payload, lows[s], highs[s])

@njit(cache=True)
def _sort_pairs_by_value_parallel(vals, payload):
    n = len(vals)
    if n < 2:
        return
    if n < _MIN_PARALLEL_SORT_SEGMENT:
        _sort_pairs_by_value(vals, payload)
        return

    lows = np.empty(_MAX_PARALLEL_SORT_SEGMENTS, dtype=np.int64)
    highs = np.empty(_MAX_PARALLEL_SORT_SEGMENTS, dtype=np.int64)
    seg_count = 1
    lows[0] = 0
    highs[0] = n - 1
    idx = 0

    while idx < seg_count and seg_count < _MAX_PARALLEL_SORT_SEGMENTS:
        low = lows[idx]
        high = highs[idx]
        if high - low < _MIN_PARALLEL_SORT_SEGMENT:
            idx += 1
            continue

        i = _partition_pairs(vals, payload, low, high)
        lows[idx] = low
        highs[idx] = i - 1

        if high > i:
            lows[seg_count] = i + 1
            highs[seg_count] = high
            seg_count += 1
        idx += 1

    _sort_pairs_segments_parallel(vals, payload, lows, highs, seg_count)

@njit(cache=True, parallel=True)
def _apply_csm_mask(csm, mask):
    n, m = csm.shape
    for i in prange(n):
        for j in range(m):
            if csm[i, j] <= 0:
                mask[i, j] = True

@njit(cache=True, parallel=True)
def _collect_positive_candidates_row_major(csm, mask):
    n, m = csm.shape
    row_counts = np.zeros(n, dtype=np.int32)
    for i in prange(n):
        if i < 2:
            continue
        cnt = 0
        for j in range(2, m):
            if not mask[i, j]:
                cnt += 1
        row_counts[i] = cnt

    offsets = np.zeros(n, dtype=np.int64)
    curr = np.int64(0)
    for i in range(n):
        offsets[i] = curr
        curr += row_counts[i]

    linear_pos = np.empty(curr, dtype=np.int64)
    values = np.empty(curr, dtype=np.float32)
    for i in prange(n):
        if row_counts[i] == 0:
            continue
        w = offsets[i]
        for j in range(2, m):
            if not mask[i, j]:
                linear_pos[w] = np.int64(i) * np.int64(m) + np.int64(j)
                values[w] = csm[i, j]
                w += 1
    return linear_pos, values

@njit(cache=True)
def _mask_buffer_path_zero(mask_flat, m, trace_buf, buf_start):
    for k in range(buf_start, len(trace_buf)):
        i = trace_buf[k, 0]
        j = trace_buf[k, 1]
        mask_flat[i * m + j] = True
        if k > buf_start:
            pi = trace_buf[k - 1, 0]
            pj = trace_buf[k - 1, 1]
            if (i - pi == 2 and j - pj == 1) or (i - pi == 1 and j - pj == 2):
                mask_flat[(i - 1) * m + (j - 1)] = True

@njit(cache=True)
def _mask_buffer_vicinity(trace_buf, buf_start, mask, vwidth):
    n, m = mask.shape
    for k in range(buf_start, len(trace_buf)):
        ic = trace_buf[k, 0]
        jc = trace_buf[k, 1]
        i1 = max(0, ic - vwidth)
        i2 = min(n, ic + vwidth + 1)
        j1 = max(0, jc - vwidth)
        j2 = min(m, jc + vwidth + 1)
        mask[i1:i2, jc] = True
        mask[ic, j1:j2] = True
        if k < len(trace_buf) - 1:
            it = trace_buf[k + 1, 0]
            jt = trace_buf[k + 1, 1]
            di = it - ic
            dj = jt - jc
            if di == 2 and dj == 1:
                ii = ic + 1
                if ii < n:
                    mask[ii, jc] = True
                    mask[ii, j1:j2] = True
            elif di == 1 and dj == 2:
                jj = jc + 1
                if jj < m:
                    mask[ic, jj] = True
                    mask[i1:i2, jj] = True

@njit(cache=True)
def _mask_buffer_path_zero_active(mask_flat, active_flat, m, trace_buf, buf_start):
    for k in range(buf_start, len(trace_buf)):
        i = trace_buf[k, 0]
        j = trace_buf[k, 1]
        lin = i * m + j
        mask_flat[lin] = True
        active_flat[lin] = False
        if k > buf_start:
            pi = trace_buf[k - 1, 0]
            pj = trace_buf[k - 1, 1]
            if (i - pi == 2 and j - pj == 1) or (i - pi == 1 and j - pj == 2):
                bi = i - 1
                bj = j - 1
                lin = bi * m + bj
                mask_flat[lin] = True
                active_flat[lin] = False

@njit(cache=True)
def _mask_vicinity(path, mask, vwidth):
    n, m = mask.shape
    for k in range(len(path)):
        ic = path[k, 0]
        jc = path[k, 1]
        i1 = max(0, ic - vwidth)
        i2 = min(n, ic + vwidth + 1)
        j1 = max(0, jc - vwidth)
        j2 = min(m, jc + vwidth + 1)
        mask[i1:i2, jc] = True
        mask[ic, j1:j2] = True
        if k < len(path) - 1:
            it = path[k + 1, 0]
            jt = path[k + 1, 1]
            di = it - ic
            dj = jt - jc
            if di == 2 and dj == 1:
                ii = ic + 1
                if ii < n:
                    mask[ii, jc] = True
                    mask[ii, j1:j2] = True
            elif di == 1 and dj == 2:
                jj = jc + 1
                if jj < m:
                    mask[ic, jj] = True
                    mask[i1:i2, jj] = True

@njit(cache=True)
def _mask_buffer_vicinity_active(trace_buf, buf_start, mask, active, vwidth):
    n, m = mask.shape
    for k in range(buf_start, len(trace_buf)):
        ic = trace_buf[k, 0]
        jc = trace_buf[k, 1]
        i1 = max(0, ic - vwidth)
        i2 = min(n, ic + vwidth + 1)
        j1 = max(0, jc - vwidth)
        j2 = min(m, jc + vwidth + 1)
        mask[i1:i2, jc] = True
        active[i1:i2, jc] = False
        mask[ic, j1:j2] = True
        active[ic, j1:j2] = False
        if k < len(trace_buf) - 1:
            it = trace_buf[k + 1, 0]
            jt = trace_buf[k + 1, 1]
            di = it - ic
            dj = jt - jc
            if di == 2 and dj == 1:
                ii = ic + 1
                if ii < n:
                    mask[ii, jc] = True
                    active[ii, jc] = False
                    mask[ii, j1:j2] = True
                    active[ii, j1:j2] = False
            elif di == 1 and dj == 2:
                jj = jc + 1
                if jj < m:
                    mask[ic, jj] = True
                    active[ic, jj] = False
                    mask[i1:i2, jj] = True
                    active[i1:i2, jj] = False

@njit(cache=True)
def _mask_vicinity_active(path, mask, active, vwidth):
    n, m = mask.shape
    for k in range(len(path)):
        ic = path[k, 0]
        jc = path[k, 1]
        i1 = max(0, ic - vwidth)
        i2 = min(n, ic + vwidth + 1)
        j1 = max(0, jc - vwidth)
        j2 = min(m, jc + vwidth + 1)
        mask[i1:i2, jc] = True
        active[i1:i2, jc] = False
        mask[ic, j1:j2] = True
        active[ic, j1:j2] = False
        if k < len(path) - 1:
            it = path[k + 1, 0]
            jt = path[k + 1, 1]
            di = it - ic
            dj = jt - jc
            if di == 2 and dj == 1:
                ii = ic + 1
                if ii < n:
                    mask[ii, jc] = True
                    active[ii, jc] = False
                    mask[ii, j1:j2] = True
                    active[ii, j1:j2] = False
            elif di == 1 and dj == 2:
                jj = jc + 1
                if jj < m:
                    mask[ic, jj] = True
                    active[ic, jj] = False
                    mask[i1:i2, jj] = True
                    active[i1:i2, jj] = False

@njit(cache=True)
def cumulative_similarity_matrix_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0, mins1=None, maxs1=None, mins2=None, maxs2=None, gamma=None):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
    block_size = 64
    ni = (n + block_size - 1) // block_size
    nj = (m + block_size - 1) // block_size

    for bi in range(ni):
        i_s = bi * block_size
        i_e = min(n, i_s + block_size)
        for bj in range(nj):
            j_s = bj * block_size
            j_e = min(m, j_s + block_size)

            if only_triu and j_e < i_s - diag_offset:
                continue

            d2_lb = 0.0
            for d in range(len(gamma)):
                diff = 0.0
                if mins1[bi, d] > maxs2[bj, d]:
                    diff = mins1[bi, d] - maxs2[bj, d]
                elif mins2[bj, d] > maxs1[bi, d]:
                    diff = mins2[bj, d] - maxs1[bi, d]
                d2_lb += gamma[d] * diff * diff

            if np.exp(-d2_lb) < tau:
                any_inc = False
                for j in range(j_s + 1, j_e + 3):
                    if csm[i_s + 1, j] > 0.0:
                        any_inc = True
                        break
                if not any_inc:
                    for i in range(i_s + 1, i_e + 3):
                        if csm[i, j_s + 1] > 0.0:
                            any_inc = True
                            break
                if not any_inc:
                    continue

            for i in range(i_s, i_e):
                j_start = max(j_s, i - diag_offset) if only_triu else j_s
                for j in range(j_start, j_e):
                    sim = sm[i, j]
                    ii = i + 2
                    jj = j + 2
                    max_cs, direction = _best_predecessor(csm[ii - 1, jj - 1], csm[ii - 2, jj - 1], csm[ii - 1, jj - 2])

                    if sim < tau:
                        csm[ii, jj] = max(0, delta_m * max_cs - delta_a)
                    else:
                        csm[ii, jj] = max(0, sim + max_cs)
                    if csm[ii, jj] > 0:
                        bp_dir[ii, jj] = direction
    return csm, bp_dir

@njit(cache=True)
def cumulative_similarity_matrix_no_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0, mins1=None, maxs1=None, mins2=None, maxs2=None, gamma=None):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
    block_size = 64
    ni = (n + block_size - 1) // block_size
    nj = (m + block_size - 1) // block_size

    for bi in range(ni):
        i_s = bi * block_size
        i_e = min(n, i_s + block_size)
        for bj in range(nj):
            j_s = bj * block_size
            j_e = min(m, j_s + block_size)

            if only_triu and j_e < i_s - diag_offset:
                continue

            d2_lb = 0.0
            for d in range(len(gamma)):
                diff = 0.0
                if mins1[bi, d] > maxs2[bj, d]:
                    diff = mins1[bi, d] - maxs2[bj, d]
                elif mins2[bj, d] > maxs1[bi, d]:
                    diff = mins2[bj, d] - maxs1[bi, d]
                d2_lb += gamma[d] * diff * diff

            if np.exp(-d2_lb) < tau:
                any_inc = False
                for j in range(j_s + 1, j_e + 3):
                    if csm[i_s + 1, j] > 0.0:
                        any_inc = True
                        break
                if not any_inc:
                    for i in range(i_s + 1, i_e + 3):
                        if csm[i, j_s + 1] > 0.0:
                            any_inc = True
                            break
                if not any_inc:
                    continue

            for i in range(i_s, i_e):
                j_start = max(j_s, i - diag_offset) if only_triu else j_s
                for j in range(j_start, j_e):
                    sim = sm[i, j]
                    ii = i + 2
                    jj = j + 2

                    if sim < tau:
                        csm[ii, jj] = max(0, delta_m * csm[ii - 1, jj - 1] - delta_a)
                    else:
                        csm[ii, jj] = max(0, sim + csm[ii - 1, jj - 1])
                    if csm[ii, jj] > 0:
                        bp_dir[ii, jj] = 0

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
    _apply_csm_mask(csm, mask)
    n, m = csm.shape
    linear_pos, values = _collect_positive_candidates_row_major(csm, mask)
    if len(linear_pos) == 0:
        return TypedList.empty_list(int32[:, :])
    _sort_pairs_by_value_parallel(values, linear_pos)
    k_best = len(linear_pos) - 1
    paths = TypedList.empty_list(int32[:, :])
    mask_flat = mask.reshape(n * m)
    bp_flat = bp_dir.reshape(n * m) if bp_dir is not None else np.empty(0, dtype=np.int8)
    trace_buf = np.empty((n + m, 2), dtype=np.int32)

    while k_best >= 0:
        path = np.empty((0, 0), dtype=np.int32)
        path_found = False
        use_buffer_path = False

        while not path_found:
            while mask_flat[linear_pos[k_best]]:
                k_best -= 1
                if k_best < 0:
                    return paths

            linear_idx = linear_pos[k_best]
            k_best -= 1
            i_best = linear_idx // np.int64(m)
            j_best = linear_idx - i_best * np.int64(m)

            if warping and bp_dir is not None:
                path_len = _extract_path_to_buffer(mask_flat, bp_flat, m, i_best, j_best, trace_buf)
                buf_start = len(trace_buf) - path_len
                use_buffer_path = True
            elif warping:
                path = best_path_warping(csm, mask, i_best, j_best)
            else:
                path = _build_path_no_warping(mask, i_best, j_best)

            if use_buffer_path:
                first_i = trace_buf[buf_start, 0]
                first_j = trace_buf[buf_start, 1]
                last_i = trace_buf[len(trace_buf) - 1, 0]
                last_j = trace_buf[len(trace_buf) - 1, 1]
                long_enough = (last_i - first_i + 1) >= l_min or (last_j - first_j + 1) >= l_min
            else:
                long_enough = (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min

            if long_enough:
                path_found = True
            elif warping and bp_dir is not None:
                _mask_buffer_path_zero(mask_flat, m, trace_buf, buf_start)
                use_buffer_path = False
            else:
                mask = mask_vicinity(path, mask, 0)

        if use_buffer_path:
            _mask_buffer_vicinity(trace_buf, buf_start, mask, vwidth)
            path = trace_buf[buf_start:].copy()
        else:
            _mask_vicinity(path, mask, vwidth)
        paths.append(path)

    return paths
