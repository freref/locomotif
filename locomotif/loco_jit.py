import numpy as np


### JIT
from numba import int32, int64, int8, float64, float32, boolean
from numba import njit
from numba.types import List, Array
from numba import prange

@njit(float32[:, :](float32[:, :], float32[:, :], float32[:], boolean, int32), parallel=True, fastmath=True)
def similarity_matrix_ndim(ts1, ts2, gamma=None, only_triu=False, diag_offset=0):
    n, m = len(ts1), len(ts2)
    d = ts1.shape[1]

    sm = np.full((n, m), -np.inf, dtype=np.float32)
    for i in prange(n):
        j_start = max(0, i - diag_offset) if only_triu else 0
        for j in range(j_start, m):
            acc = np.float32(0.0)
            for k in range(d):
                diff = ts1[i, k] - ts2[j, k]
                acc += gamma[k] * diff * diff
            sm[i, j] = np.float32(np.exp(-acc))

    return sm

@njit(cache=True, parallel=True)
def _collect_above_threshold(sm, threshold, only_triu):
    n, m = sm.shape
    local_counts = np.zeros(n, dtype=np.int32)
    for i in prange(n):
        j_start = i if only_triu else 0
        count = 0
        for j in range(j_start, m):
            if sm[i, j] >= threshold:
                count += 1
        local_counts[i] = count

    offsets = np.empty(n, dtype=np.int64)
    total = np.int64(0)
    for i in range(n):
        offsets[i] = total
        total += local_counts[i]

    out = np.empty(total, dtype=np.float32)
    for i in prange(n):
        j_start = i if only_triu else 0
        write_idx = offsets[i]
        for j in range(j_start, m):
            if sm[i, j] >= threshold:
                out[write_idx] = sm[i, j]
                write_idx += 1

    return out

@njit(cache=True)
def exact_tau_from_sm(sm, rho, only_triu):
    n, m = sm.shape
    if only_triu:
        total_elements = np.int64(n) * np.int64(n + 1) // np.int64(2)
    else:
        total_elements = np.int64(n) * np.int64(m)

    if total_elements <= 1:
        return sm[0, 0]

    h = (total_elements - np.int64(1)) * rho
    floor_h = np.int64(h)
    ceil_h = floor_h if h == floor_h else floor_h + np.int64(1)
    needed = total_elements - floor_h

    step = max(1, n // 100)
    sample_size = np.int64(0)
    for i in range(0, n, step):
        sample_size += (m - i) if only_triu else m

    sample = np.empty(sample_size, dtype=np.float32)
    write_idx = np.int64(0)
    for i in range(0, n, step):
        j_start = i if only_triu else 0
        for j in range(j_start, m):
            sample[write_idx] = sm[i, j]
            write_idx += 1

    sample_floor = np.int64((sample_size - np.int64(1)) * rho)
    sample_threshold = np.partition(sample, sample_floor)[sample_floor]
    if total_elements >= np.int64(1_000_000):
        return sample_threshold
    threshold = np.float32(sample_threshold * np.float32(0.99))
    if threshold < 0.0:
        threshold = np.float32(0.0)

    collected = _collect_above_threshold(sm, threshold, only_triu)
    while len(collected) < needed:
        if threshold <= 0.0:
            collected = _collect_above_threshold(sm, np.float32(0.0), only_triu)
            break
        threshold = np.float32(threshold * np.float32(0.9))
        collected = _collect_above_threshold(sm, threshold, only_triu)

    count_below = total_elements - np.int64(len(collected))
    k_lo = floor_h - count_below
    if k_lo < 0:
        k_lo = np.int64(0)
    k_hi = ceil_h - count_below
    max_idx = np.int64(len(collected) - 1)
    if k_hi > max_idx:
        k_hi = max_idx

    if k_lo == k_hi:
        return np.partition(collected, k_lo)[k_lo]

    partitioned = np.partition(collected, (k_lo, k_hi))
    weight = h - floor_h
    return (np.float32(1.0) - np.float32(weight)) * partitioned[k_lo] + np.float32(weight) * partitioned[k_hi]

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
def radix_argsort_uint64(keys):
    n = len(keys)
    perm = np.arange(n, dtype=np.int32)
    tmp_perm = np.empty(n, dtype=np.int32)
    work_keys = keys.copy()
    tmp_keys = np.empty(n, dtype=np.uint64)

    for shift in range(0, 32, 8):
        counts = np.zeros(256, dtype=np.int64)

        for i in range(n):
            bucket = int((work_keys[i] >> shift) & np.uint64(255))
            counts[bucket] += 1

        total = 0
        for bucket in range(256):
            count = counts[bucket]
            counts[bucket] = total
            total += count

        for i in range(n):
            bucket = int((work_keys[i] >> shift) & np.uint64(255))
            idx = counts[bucket]
            tmp_keys[idx] = work_keys[i]
            tmp_perm[idx] = perm[i]
            counts[bucket] = idx + 1

        swap_keys = work_keys
        work_keys = tmp_keys
        tmp_keys = swap_keys

        swap_perm = perm
        perm = tmp_perm
        tmp_perm = swap_perm

    return perm

@njit
def radix_sort_u32_with_payload(keys, payload):
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
        for bucket in range(65536):
            offsets[bucket] = total
            total += counts[bucket]

        for i in range(n):
            bucket = np.int32((keys[i] >> np.uint32(shift)) & mask)
            pos = offsets[bucket]
            tmp_keys[pos] = keys[i]
            tmp_payload[pos] = payload[i]
            offsets[bucket] = pos + 1

        keys, tmp_keys = tmp_keys, keys
        payload, tmp_payload = tmp_payload, payload

    return keys, payload

@njit(cache=True)
def collect_linear_positions(mask):
    n, m = mask.shape
    count = 0
    for i in range(n):
        for j in range(m):
            if mask[i, j]:
                count += 1

    out = np.empty(count, dtype=np.int64)
    write_idx = 0
    for i in range(n):
        base = np.int64(i) * np.int64(m)
        for j in range(m):
            if mask[i, j]:
                out[write_idx] = base + np.int64(j)
                write_idx += 1
    return out

@njit(cache=True)
def _finalize_tiled_candidates(tile_best_linear):
    count = 0
    for idx in range(len(tile_best_linear)):
        if tile_best_linear[idx] >= 0:
            count += 1

    out = np.empty(count, dtype=np.int64)
    write_idx = 0
    for idx in range(len(tile_best_linear)):
        linear_idx = tile_best_linear[idx]
        if linear_idx >= 0:
            out[write_idx] = linear_idx
            write_idx += 1
    return out

@njit(cache=True)
def _finalize_tiled_candidates_with_values(tile_best_linear, tile_best_value):
    count = 0
    for idx in range(len(tile_best_linear)):
        if tile_best_linear[idx] >= 0:
            count += 1

    out_linear = np.empty(count, dtype=np.int64)
    out_value = np.empty(count, dtype=np.float32)
    write_idx = 0
    for idx in range(len(tile_best_linear)):
        linear_idx = tile_best_linear[idx]
        if linear_idx >= 0:
            out_linear[write_idx] = linear_idx
            out_value[write_idx] = tile_best_value[idx]
            write_idx += 1
    return out_linear, out_value

@njit(cache=True)
def symmetric_path_mask(n, m, vwidth):
    mask = np.ones((n, m), dtype=np.bool_)
    offset = vwidth + 1
    for i in range(n):
        j_start = i + offset
        if j_start < m:
            mask[i, j_start:] = False
    return mask

@njit(cache=True)
def collect_tiled_candidate_positions(start_mask, csm, tile_size):
    n, m = start_mask.shape
    tile_size = max(1, tile_size)
    n_tiles_i = (n + tile_size - 1) // tile_size
    n_tiles_j = (m + tile_size - 1) // tile_size
    max_candidates = n_tiles_i * n_tiles_j
    out = np.empty(max_candidates, dtype=np.int64)
    count = 0

    for tile_i in range(n_tiles_i):
        i_start = tile_i * tile_size
        i_end = min(n, i_start + tile_size)
        for tile_j in range(n_tiles_j):
            j_start = tile_j * tile_size
            j_end = min(m, j_start + tile_size)
            best_i = -1
            best_j = -1
            best_val = np.float32(0.0)
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    if not start_mask[i, j]:
                        continue
                    value = csm[i, j]
                    if best_i == -1 or value > best_val:
                        best_i = i
                        best_j = j
                        best_val = value
            if best_i != -1:
                out[count] = np.int64(best_i) * np.int64(m) + np.int64(best_j)
                count += 1

    if count == 0:
        return collect_linear_positions(start_mask)
    return out[:count]

@njit(cache=True)
def collect_tiled_candidate_positions_threshold(mask, csm, threshold, tile_size):
    n, m = mask.shape
    tile_size = max(1, tile_size)
    n_tiles_i = (n + tile_size - 1) // tile_size
    n_tiles_j = (m + tile_size - 1) // tile_size
    max_candidates = n_tiles_i * n_tiles_j
    out = np.empty(max_candidates, dtype=np.int64)
    count = 0

    for tile_i in range(n_tiles_i):
        i_start = tile_i * tile_size
        i_end = min(n, i_start + tile_size)
        for tile_j in range(n_tiles_j):
            j_start = tile_j * tile_size
            j_end = min(m, j_start + tile_size)
            best_i = -1
            best_j = -1
            best_val = np.float32(0.0)
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    if mask[i, j]:
                        continue
                    value = csm[i, j]
                    if value < threshold:
                        continue
                    if best_i == -1 or value > best_val:
                        best_i = i
                        best_j = j
                        best_val = value
            if best_i != -1:
                out[count] = np.int64(best_i) * np.int64(m) + np.int64(best_j)
                count += 1

    if count == 0:
        fallback_mask = (~mask) & (csm >= threshold)
        return collect_linear_positions(fallback_mask)
    return out[:count]

@njit(cache=True)
def collect_tiled_candidate_positions_threshold_symmetric(csm, threshold, tile_size, diag_gap):
    n, m = csm.shape
    tile_size = max(1, tile_size)
    n_tiles_i = (n + tile_size - 1) // tile_size
    n_tiles_j = (m + tile_size - 1) // tile_size
    max_candidates = n_tiles_i * n_tiles_j
    out = np.empty(max_candidates, dtype=np.int64)
    count = 0

    for tile_i in range(n_tiles_i):
        i_start = tile_i * tile_size
        i_end = min(n, i_start + tile_size)
        for tile_j in range(n_tiles_j):
            j_start = tile_j * tile_size
            j_end = min(m, j_start + tile_size)
            if j_end <= i_start + diag_gap:
                continue

            best_i = -1
            best_j = -1
            best_val = np.float32(0.0)
            for i in range(i_start, i_end):
                row_j_start = max(j_start, i + diag_gap)
                for j in range(row_j_start, j_end):
                    value = csm[i, j]
                    if value < threshold:
                        continue
                    if best_i == -1 or value > best_val:
                        best_i = i
                        best_j = j
                        best_val = value
            if best_i != -1:
                out[count] = np.int64(best_i) * np.int64(m) + np.int64(best_j)
                count += 1

    if count == 0:
        count = 0
        for i in range(n):
            for j in range(i + diag_gap, m):
                if csm[i, j] >= threshold:
                    count += 1

        out = np.empty(count, dtype=np.int64)
        write_idx = 0
        for i in range(n):
            for j in range(i + diag_gap, m):
                if csm[i, j] >= threshold:
                    out[write_idx] = np.int64(i) * np.int64(m) + np.int64(j)
                    write_idx += 1
        return out

    return out[:count]
        
@njit(float32[:, :](float32[:, :], float64, float64, float64, boolean, int32))
def cumulative_similarity_matrix_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    zero_cutoff = delta_a / delta_m

    for i in range(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m

        for j in range(j_start, j_end):

            sim = sm[i, j]

            max_cs = max3(csm[i - 1 + 2, j - 1 + 2], csm[i - 2 + 2, j - 1 + 2], csm[i - 1 + 2, j - 2 + 2])

            if sim < tau:
                if max_cs <= zero_cutoff:
                    continue
                csm[i + 2, j + 2] = max(0, delta_m * max_cs - delta_a)
            else:
                csm[i + 2, j + 2] = max(0, sim + max_cs)
    return csm

@njit(cache=True)
def cumulative_similarity_matrix_warping_with_bp(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0, tile_size=0, candidate_threshold=0.0, diag_gap=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
    if tile_size > 0:
        n_tiles_i = (n + 2 + tile_size - 1) // tile_size
        n_tiles_j = (m + 2 + tile_size - 1) // tile_size
        tile_best_linear = np.full(n_tiles_i * n_tiles_j, np.int64(-1), dtype=np.int64)
        tile_best_value = np.zeros(n_tiles_i * n_tiles_j, dtype=np.float32)
    else:
        n_tiles_j = 0
        tile_best_linear = np.empty(0, dtype=np.int64)
        tile_best_value = np.empty(0, dtype=np.float32)
    zero_cutoff = delta_a / delta_m

    for i in range(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m

        for j in range(j_start, j_end):

            sim = sm[i, j]
            max_cs, direction = _best_predecessor(
                csm[i - 1 + 2, j - 1 + 2],
                csm[i - 2 + 2, j - 1 + 2],
                csm[i - 1 + 2, j - 2 + 2],
            )

            if sim < tau:
                if max_cs <= zero_cutoff:
                    continue
                csm[i + 2, j + 2] = max(0, delta_m * max_cs - delta_a)
            else:
                csm[i + 2, j + 2] = max(0, sim + max_cs)

            if csm[i + 2, j + 2] > 0:
                ci = i + 2
                cj = j + 2
                bp_dir[ci, cj] = direction
                if tile_size > 0 and csm[ci, cj] >= candidate_threshold and (not only_triu or cj >= ci + diag_gap):
                    tile_idx = (ci // tile_size) * n_tiles_j + (cj // tile_size)
                    if tile_best_linear[tile_idx] == -1 or csm[ci, cj] > tile_best_value[tile_idx]:
                        tile_best_linear[tile_idx] = np.int64(ci) * np.int64(m + 2) + np.int64(cj)
                        tile_best_value[tile_idx] = csm[ci, cj]

    return csm, bp_dir, _finalize_tiled_candidates(tile_best_linear)

@njit(float32[:, :](float32[:, :], float64, float64, float64, boolean, int32))
def cumulative_similarity_matrix_no_warping(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    zero_cutoff = delta_a / delta_m

    for i in range(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m

        for j in range(j_start, j_end):

            sim = sm[i, j]

            if sim < tau:
                prev = csm[i - 1 + 2, j - 1 + 2]
                if prev <= zero_cutoff:
                    continue
                csm[i + 2, j + 2] = max(0, delta_m * prev - delta_a)
            else:
                csm[i + 2, j + 2] = max(0, sim + csm[i - 1 + 2, j - 1 + 2])

    return csm

@njit(cache=True)
def cumulative_similarity_matrix_no_warping_with_bp(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0, tile_size=0, candidate_threshold=0.0, diag_gap=0):
    n, m = sm.shape

    csm = np.zeros((n + 2, m + 2), dtype=np.float32)
    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
    if tile_size > 0:
        n_tiles_i = (n + 2 + tile_size - 1) // tile_size
        n_tiles_j = (m + 2 + tile_size - 1) // tile_size
        tile_best_linear = np.full(n_tiles_i * n_tiles_j, np.int64(-1), dtype=np.int64)
        tile_best_value = np.zeros(n_tiles_i * n_tiles_j, dtype=np.float32)
    else:
        n_tiles_j = 0
        tile_best_linear = np.empty(0, dtype=np.int64)
        tile_best_value = np.empty(0, dtype=np.float32)
    zero_cutoff = delta_a / delta_m

    for i in range(n):

        j_start = max(0, i-diag_offset) if only_triu else 0
        j_end = m

        for j in range(j_start, j_end):

            sim = sm[i, j]

            if sim < tau:
                prev = csm[i - 1 + 2, j - 1 + 2]
                if prev <= zero_cutoff:
                    continue
                csm[i + 2, j + 2] = max(0, delta_m * prev - delta_a)
            else:
                csm[i + 2, j + 2] = max(0, sim + csm[i - 1 + 2, j - 1 + 2])

            if csm[i + 2, j + 2] > 0:
                ci = i + 2
                cj = j + 2
                bp_dir[ci, cj] = np.int8(0)
                if tile_size > 0 and csm[ci, cj] >= candidate_threshold and (not only_triu or cj >= ci + diag_gap):
                    tile_idx = (ci // tile_size) * n_tiles_j + (cj // tile_size)
                    if tile_best_linear[tile_idx] == -1 or csm[ci, cj] > tile_best_value[tile_idx]:
                        tile_best_linear[tile_idx] = np.int64(ci) * np.int64(m + 2) + np.int64(cj)
                        tile_best_value[tile_idx] = csm[ci, cj]

    return csm, bp_dir, _finalize_tiled_candidates(tile_best_linear)

@njit(cache=True)
def cumulative_similarity_matrix_warping_with_bp_compact(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0, tile_size=0, candidate_threshold=0.0, diag_gap=0):
    n, m = sm.shape

    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
    row_prev2 = np.zeros(m + 2, dtype=np.float32)
    row_prev1 = np.zeros(m + 2, dtype=np.float32)
    row_curr = np.zeros(m + 2, dtype=np.float32)
    if tile_size > 0:
        n_tiles_i = (n + 2 + tile_size - 1) // tile_size
        n_tiles_j = (m + 2 + tile_size - 1) // tile_size
        tile_best_linear = np.full(n_tiles_i * n_tiles_j, np.int64(-1), dtype=np.int64)
        tile_best_value = np.zeros(n_tiles_i * n_tiles_j, dtype=np.float32)
    else:
        n_tiles_j = 0
        tile_best_linear = np.empty(0, dtype=np.int64)
        tile_best_value = np.empty(0, dtype=np.float32)
    zero_cutoff = np.float32(delta_a / delta_m)

    for i in range(n):
        row_curr[:] = 0.0
        j_start = max(0, i - diag_offset) if only_triu else 0
        for j in range(j_start, m):
            ci = i + 2
            cj = j + 2
            sim = sm[i, j]
            max_cs, direction = _best_predecessor(
                row_prev1[cj - 1],
                row_prev2[cj - 1],
                row_prev1[cj - 2],
            )

            value = np.float32(0.0)
            if sim < tau:
                if max_cs <= zero_cutoff:
                    continue
                value = np.float32(delta_m * max_cs - delta_a)
            else:
                value = np.float32(sim + max_cs)

            if value <= 0.0:
                continue

            row_curr[cj] = value
            bp_dir[ci, cj] = direction
            if tile_size > 0 and value >= candidate_threshold and (not only_triu or cj >= ci + diag_gap):
                tile_idx = (ci // tile_size) * n_tiles_j + (cj // tile_size)
                if tile_best_linear[tile_idx] == -1 or value > tile_best_value[tile_idx]:
                    tile_best_linear[tile_idx] = np.int64(ci) * np.int64(m + 2) + np.int64(cj)
                    tile_best_value[tile_idx] = value

        tmp = row_prev2
        row_prev2 = row_prev1
        row_prev1 = row_curr
        row_curr = tmp

    return bp_dir, *_finalize_tiled_candidates_with_values(tile_best_linear, tile_best_value)

@njit(cache=True)
def cumulative_similarity_matrix_no_warping_with_bp_compact(sm, tau=0.5, delta_a=1.0, delta_m=0.5, only_triu=False, diag_offset=0, tile_size=0, candidate_threshold=0.0, diag_gap=0):
    n, m = sm.shape

    bp_dir = np.full((n + 2, m + 2), np.int8(-1), dtype=np.int8)
    row_prev1 = np.zeros(m + 2, dtype=np.float32)
    row_curr = np.zeros(m + 2, dtype=np.float32)
    if tile_size > 0:
        n_tiles_i = (n + 2 + tile_size - 1) // tile_size
        n_tiles_j = (m + 2 + tile_size - 1) // tile_size
        tile_best_linear = np.full(n_tiles_i * n_tiles_j, np.int64(-1), dtype=np.int64)
        tile_best_value = np.zeros(n_tiles_i * n_tiles_j, dtype=np.float32)
    else:
        n_tiles_j = 0
        tile_best_linear = np.empty(0, dtype=np.int64)
        tile_best_value = np.empty(0, dtype=np.float32)
    zero_cutoff = np.float32(delta_a / delta_m)

    for i in range(n):
        row_curr[:] = 0.0
        j_start = max(0, i - diag_offset) if only_triu else 0
        for j in range(j_start, m):
            ci = i + 2
            cj = j + 2
            sim = sm[i, j]
            value = np.float32(0.0)

            if sim < tau:
                prev = row_prev1[cj - 1]
                if prev <= zero_cutoff:
                    continue
                value = np.float32(delta_m * prev - delta_a)
            else:
                value = np.float32(sim + row_prev1[cj - 1])

            if value <= 0.0:
                continue

            row_curr[cj] = value
            bp_dir[ci, cj] = np.int8(0)
            if tile_size > 0 and value >= candidate_threshold and (not only_triu or cj >= ci + diag_gap):
                tile_idx = (ci // tile_size) * n_tiles_j + (cj // tile_size)
                if tile_best_linear[tile_idx] == -1 or value > tile_best_value[tile_idx]:
                    tile_best_linear[tile_idx] = np.int64(ci) * np.int64(m + 2) + np.int64(cj)
                    tile_best_value[tile_idx] = value

        tmp = row_prev1
        row_prev1 = row_curr
        row_curr = tmp

    return bp_dir, *_finalize_tiled_candidates_with_values(tile_best_linear, tile_best_value)


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
def _extract_path_to_buffer(mask_flat, bp_flat, m, i, j, buf, symmetric=False, diag_gap=0):
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
        if bp_flat[lin] < 0:
            break
        if symmetric and j < i + diag_gap:
            break
        if mask_flat[lin]:
            break
    return length

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
    
    min_path_length = l_min if not warping else max(1, (l_min + 1) // 2)
    start_mask = (~mask) & (csm >= tau * min_path_length)
    
    pos_i, pos_j = np.nonzero(start_mask)
    
    values = np.array([csm[pos_i[k], pos_j[k]] for k in range(len(pos_i))])
    perm = radix_argsort_uint64(values.view(np.uint32).astype(np.uint64))
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

@njit(cache=True)
def find_best_paths_with_bp(csm, mask, tau, l_min=10, vwidth=5, warping=True, bp_dir=None, symmetric=False, candidate_linear_pos=None):
    mask = mask | (csm <= 0)
    min_path_length = l_min if not warping else max(1, (l_min + 1) // 2)
    n, m = csm.shape
    threshold = tau * min_path_length
    if candidate_linear_pos is not None and len(candidate_linear_pos) > 0:
        linear_pos = candidate_linear_pos
    elif symmetric:
        linear_pos = collect_tiled_candidate_positions_threshold_symmetric(csm, threshold, vwidth, vwidth + 1)
    else:
        linear_pos = collect_tiled_candidate_positions_threshold(mask, csm, threshold, vwidth)
    csm_flat = csm.reshape(n * m)
    values = csm_flat[linear_pos].view(np.uint32)
    _, linear_pos = radix_sort_u32_with_payload(values, linear_pos)

    k_best = len(linear_pos) - 1
    paths = []
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
            i_best = np.int32(linear_idx // np.int64(m))
            j_best = np.int32(linear_idx - np.int64(i_best) * np.int64(m))

            if i_best < 2 or j_best < 2:
                return paths

            if warping and bp_dir is not None:
                path_len = _extract_path_to_buffer(mask_flat, bp_flat, m, i_best, j_best, trace_buf)
                buf_start = len(trace_buf) - path_len
                use_buffer_path = True
                first_i = trace_buf[buf_start, 0]
                first_j = trace_buf[buf_start, 1]
                last_i = trace_buf[len(trace_buf) - 1, 0]
                last_j = trace_buf[len(trace_buf) - 1, 1]
                long_enough = (last_i - first_i + 1) >= l_min or (last_j - first_j + 1) >= l_min
            elif warping:
                path = _build_path_warping(bp_dir, mask, i_best, j_best)
                mask = mask_vicinity(path, mask, 0)
                long_enough = (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min
            else:
                path = _build_path_no_warping(mask, i_best, j_best)
                mask = mask_vicinity(path, mask, 0)
                long_enough = (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min

            if long_enough:
                path_found = True
            elif use_buffer_path:
                _mask_buffer_path_zero(mask_flat, m, trace_buf, buf_start)
                use_buffer_path = False

        if use_buffer_path:
            _mask_buffer_vicinity(trace_buf, buf_start, mask, vwidth)
            path = trace_buf[buf_start:].copy()
        else:
            mask = mask_vicinity(path, mask, vwidth)
        paths.append(path)

    return paths

@njit(cache=True)
def find_best_paths_with_bp_compact(mask, tau, l_min=10, vwidth=5, warping=True, bp_dir=None, candidate_linear_pos=None, candidate_values=None, symmetric=False, diag_gap=0):
    if warping:
        if mask.size == 0:
            mask = np.zeros(bp_dir.shape, dtype=np.bool_)
    elif mask.size == 0:
        mask = bp_dir < 0
    else:
        mask = mask | (bp_dir < 0)
    paths = []
    if candidate_linear_pos is None or candidate_values is None or len(candidate_linear_pos) == 0:
        return paths

    keys = candidate_values.view(np.uint32).copy()
    _, linear_pos = radix_sort_u32_with_payload(keys, candidate_linear_pos.copy())

    n, m = bp_dir.shape
    k_best = len(linear_pos) - 1
    mask_flat = mask.reshape(n * m)
    bp_flat = bp_dir.reshape(n * m)
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
            i_best = np.int32(linear_idx // np.int64(m))
            j_best = np.int32(linear_idx - np.int64(i_best) * np.int64(m))

            if i_best < 2 or j_best < 2:
                return paths

            if warping:
                path_len = _extract_path_to_buffer(mask_flat, bp_flat, m, i_best, j_best, trace_buf, symmetric, diag_gap)
                buf_start = len(trace_buf) - path_len
                use_buffer_path = True
                first_i = trace_buf[buf_start, 0]
                first_j = trace_buf[buf_start, 1]
                last_i = trace_buf[len(trace_buf) - 1, 0]
                last_j = trace_buf[len(trace_buf) - 1, 1]
                long_enough = (last_i - first_i + 1) >= l_min or (last_j - first_j + 1) >= l_min
            else:
                path = _build_path_no_warping(mask, i_best, j_best)
                mask = mask_vicinity(path, mask, 0)
                long_enough = (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min

            if long_enough:
                path_found = True
            elif use_buffer_path:
                _mask_buffer_path_zero(mask_flat, m, trace_buf, buf_start)
                use_buffer_path = False

        if use_buffer_path:
            _mask_buffer_vicinity(trace_buf, buf_start, mask, vwidth)
            path = trace_buf[buf_start:].copy()
        else:
            mask = mask_vicinity(path, mask, vwidth)
        paths.append(path)

    return paths
