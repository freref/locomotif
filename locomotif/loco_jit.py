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


@njit
def deactivate_sparse_idx(sparse_idx, active_sparse, sparse_to_active, active_size):
    active_idx = sparse_to_active[sparse_idx]
    if active_idx < 0 or active_idx >= active_size:
        return active_size

    last_active_idx = np.int32(active_size - 1)
    last_sparse_idx = active_sparse[last_active_idx]

    active_sparse[active_idx] = last_sparse_idx
    sparse_to_active[last_sparse_idx] = active_idx
    sparse_to_active[sparse_idx] = -1

    return last_active_idx


@njit
def deactivate_dense_idx(flat_idx, dense_to_sparse, active_sparse, sparse_to_active, active_size):
    sparse_idx = dense_to_sparse[flat_idx]
    if sparse_idx < 0:
        return active_size
    return deactivate_sparse_idx(sparse_idx, active_sparse, sparse_to_active, active_size)


@njit
def deactivate_forbidden(forbidden, dense_to_sparse, active_sparse, sparse_to_active, active_size):
    for flat_idx64 in forbidden:
        flat_idx = np.int32(flat_idx64)
        active_size = deactivate_dense_idx(flat_idx, dense_to_sparse, active_sparse, sparse_to_active, active_size)
    return active_size


@njit
def argmax_active(values, active_sparse, active_size):
    if active_size <= 0:
        return -1

    best_sparse = active_sparse[0]
    best_value = values[best_sparse]
    for k in range(1, active_size):
        sparse_idx = active_sparse[k]
        value = values[sparse_idx]
        if value > best_value:
            best_value = value
            best_sparse = sparse_idx

    return best_sparse


@njit
def mask_and_record(mask, forbidden, i, j):
    if not mask[i, j]:
        mask[i, j] = True
        forbidden.add(i * mask.shape[1] + j)


@njit
def mask_vicinity_record(path, mask, vwidth, forbidden):
    n, m = mask.shape

    for k in range(len(path) - 1):
        ic, jc = path[k]
        it, jt = path[k + 1]

        di, dj = (it - ic, jt - jc)

        i1, i2 = max(0, ic - vwidth), min(n, ic + vwidth + 1)
        j1, j2 = max(0, jc - vwidth), min(m, jc + vwidth + 1)

        for i in range(i1, i2):
            mask_and_record(mask, forbidden, i, jc)
        for j in range(j1, j2):
            mask_and_record(mask, forbidden, ic, j)

        if di == 2 and dj == 1:
            if i2 + 1 < n:
                mask_and_record(mask, forbidden, ic + 1, jc)
            for j in range(j1, j2):
                mask_and_record(mask, forbidden, ic + 1, j)
        elif di == 1 and dj == 2:
            if j2 + 1 < m:
                mask_and_record(mask, forbidden, ic, jc + 1)
            for i in range(i1, i2):
                mask_and_record(mask, forbidden, i, jc + 1)
        else:
            if not (di == 1 and dj == 1):
                raise Exception("Path does not comply to the allowed step sizes")

    ic, jc = path[-1]
    i1, i2 = max(0, ic - vwidth), min(n, ic + vwidth + 1)
    j1, j2 = max(0, jc - vwidth), min(m, jc + vwidth + 1)
    for i in range(i1, i2):
        mask_and_record(mask, forbidden, i, jc)
    for j in range(j1, j2):
        mask_and_record(mask, forbidden, ic, j)


@njit
def update_dist_and_record(mask, dist, bp, l_min, forbidden):
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

    for i in range(n):
        for j in range(m):
            if (not mask[i, j]) and dist[i, j] <= l_min:
                mask[i, j] = True
                forbidden.add(i * m + j)

    return dist


@njit
def normalize_dist_clamped(mask, dist, l_min):
    n, m = dist.shape
    cap = l_min + 1
    for i in range(n):
        for j in range(m):
            if mask[i, j]:
                dist[i, j] = 0
            elif dist[i, j] > cap:
                dist[i, j] = cap
    return dist


@njit(types.Tuple((int32[:], int32[:]))(int32[:, :, :]))
def build_reverse_bp_graph(bp):
    n, m, _ = bp.shape
    n_cells = n * m

    counts = np.zeros(n_cells, dtype=np.int32)
    for i in range(n):
        for j in range(m):
            pi = bp[i, j, 0]
            pj = bp[i, j, 1]
            if pi >= 0 and pj >= 0:
                parent = pi * m + pj
                counts[parent] += 1

    rev_offsets = np.empty(n_cells + 1, dtype=np.int32)
    rev_offsets[0] = 0
    for k in range(n_cells):
        rev_offsets[k + 1] = rev_offsets[k] + counts[k]

    n_edges = rev_offsets[n_cells]
    rev_children = np.empty(n_edges, dtype=np.int32)
    write = np.empty(n_cells, dtype=np.int32)
    for k in range(n_cells):
        write[k] = rev_offsets[k]

    for i in range(n):
        for j in range(m):
            pi = bp[i, j, 0]
            pj = bp[i, j, 1]
            if pi >= 0 and pj >= 0:
                parent = pi * m + pj
                idx = write[parent]
                rev_children[idx] = i * m + j
                write[parent] = idx + 1

    return rev_offsets, rev_children


@njit
def propagate_dist_from_masked_seeds(mask, dist, seed_masked, rev_offsets, rev_children, l_min, queue, touched_nodes, enqueued_stamp, touched_stamp, stamp):
    n, m = mask.shape
    cap = l_min + 1
    q_read = 0
    q_write = 0
    touched_count = 0

    for flat_idx64 in seed_masked:
        flat_idx = np.int32(flat_idx64)
        i = flat_idx // m
        j = flat_idx - i * m
        dist[i, j] = 0

        if touched_stamp[flat_idx] != stamp:
            touched_stamp[flat_idx] = stamp
            touched_nodes[touched_count] = flat_idx
            touched_count += 1

        if enqueued_stamp[flat_idx] != stamp:
            enqueued_stamp[flat_idx] = stamp
            queue[q_write] = flat_idx
            q_write += 1

    while q_read < q_write:
        parent = queue[q_read]
        q_read += 1

        pi = parent // m
        pj = parent - pi * m
        parent_dist = dist[pi, pj]

        start = rev_offsets[parent]
        end = rev_offsets[parent + 1]
        for idx in range(start, end):
            child = rev_children[idx]
            ci = child // m
            cj = child - ci * m

            if mask[ci, cj]:
                dist[ci, cj] = 0
                continue

            cand = parent_dist + 1
            if cand > cap:
                cand = cap

            if cand < dist[ci, cj]:
                dist[ci, cj] = cand
                if touched_stamp[child] != stamp:
                    touched_stamp[child] = stamp
                    touched_nodes[touched_count] = child
                    touched_count += 1

                if cand < cap and enqueued_stamp[child] != stamp:
                    enqueued_stamp[child] = stamp
                    queue[q_write] = child
                    q_write += 1

    return touched_count


@njit
def mask_touched_below_threshold(mask, dist, touched_nodes, touched_count, l_min, forbidden, new_pending_dist_masked):
    _, m = mask.shape
    for t in range(touched_count):
        flat_idx = touched_nodes[t]
        i = flat_idx // m
        j = flat_idx - i * m
        if (not mask[i, j]) and dist[i, j] <= l_min:
            mask[i, j] = True
            dist[i, j] = 0
            forbidden.add(np.int64(flat_idx))
            new_pending_dist_masked.add(np.int64(flat_idx))


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

    ic, jc = path[-1]
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
    n, m = csm.shape
    n_cells = n * m
    count = np.count_nonzero((csm == 0) | mask)
    print("amount:", csm.size - count)
    print("percentage:", 100.0 * count / csm.size)

    mask = mask | (csm <= 0)
    dist = normalize_dist_clamped(mask, dist, l_min)
    rev_offsets, rev_children = build_reverse_bp_graph(bp)

    pending_dist_masked = set()
    pending_dist_masked.add(np.int64(-1))
    pending_dist_masked.remove(np.int64(-1))
    pending_threshold_only = set()
    pending_threshold_only.add(np.int64(-1))
    pending_threshold_only.remove(np.int64(-1))
    for i in range(n):
        for j in range(m):
            if mask[i, j]:
                pending_dist_masked.add(np.int64(i * m + j))
            elif dist[i, j] <= l_min:
                pending_threshold_only.add(np.int64(i * m + j))

    queue = np.empty(n_cells, dtype=np.int32)
    touched_nodes = np.empty(n_cells, dtype=np.int32)
    enqueued_stamp = np.zeros(n_cells, dtype=np.int32)
    touched_stamp = np.zeros(n_cells, dtype=np.int32)
    stamp = np.int32(1)

    start_mask = (~mask) & (dist > l_min)
    pos_i, pos_j = np.nonzero(start_mask)

    nnz = len(pos_i)
    paths = []
    if nnz == 0:
        return paths

    values = np.empty(nnz, dtype=np.float32)
    for k in range(nnz):
        values[k] = csm[pos_i[k], pos_j[k]]

    dense_to_sparse = np.full(csm.size, -1, dtype=np.int32)
    for sparse_idx in range(nnz):
        dense_to_sparse[pos_i[sparse_idx] * m + pos_j[sparse_idx]] = sparse_idx

    active_sparse = np.arange(nnz, dtype=np.int32)
    sparse_to_active = np.arange(nnz, dtype=np.int32)
    active_size = np.int32(nnz)

    while active_size > 0:
        best_sparse_idx = argmax_active(values, active_sparse, active_size)
        if best_sparse_idx < 0:
            return paths

        i_best = pos_i[best_sparse_idx]
        j_best = pos_j[best_sparse_idx]
        if i_best < 2 or j_best < 2:
            active_size = deactivate_sparse_idx(best_sparse_idx, active_sparse, sparse_to_active, active_size)
            continue

        if warping:
            path = best_path_warping(csm, mask, i_best, j_best)
        else:
            path = best_path_no_warping(csm, mask, i_best, j_best)

        forbidden = set()
        forbidden.add(np.int64(-1))
        forbidden.remove(np.int64(-1))
        mask_vicinity_record(path, mask, 0, forbidden)
        active_size = deactivate_forbidden(forbidden, dense_to_sparse, active_sparse, sparse_to_active, active_size)

        if (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min:
            forbidden_vicinity = set()
            forbidden_vicinity.add(np.int64(-1))
            forbidden_vicinity.remove(np.int64(-1))
            mask_vicinity_record(path, mask, vwidth, forbidden_vicinity)
            active_size = deactivate_forbidden(forbidden_vicinity, dense_to_sparse, active_sparse, sparse_to_active, active_size)

            seed_masked = set()
            seed_masked.add(np.int64(-1))
            seed_masked.remove(np.int64(-1))
            for flat_idx in pending_dist_masked:
                seed_masked.add(flat_idx)
            for flat_idx in forbidden:
                seed_masked.add(flat_idx)
            for flat_idx in forbidden_vicinity:
                seed_masked.add(flat_idx)

            if stamp == np.int32(2147483647):
                enqueued_stamp[:] = 0
                touched_stamp[:] = 0
                stamp = np.int32(1)

            touched_count = propagate_dist_from_masked_seeds(mask, dist, seed_masked, rev_offsets, rev_children, l_min, queue, touched_nodes, enqueued_stamp, touched_stamp, stamp)
            stamp = np.int32(stamp + 1)

            forbidden_dist = set()
            forbidden_dist.add(np.int64(-1))
            forbidden_dist.remove(np.int64(-1))
            new_pending_dist_masked = set()
            new_pending_dist_masked.add(np.int64(-1))
            new_pending_dist_masked.remove(np.int64(-1))
            mask_touched_below_threshold(mask, dist, touched_nodes, touched_count, l_min, forbidden_dist, new_pending_dist_masked)
            for flat_idx in pending_threshold_only:
                idx = np.int32(flat_idx)
                i = idx // m
                j = idx - i * m
                if not mask[i, j]:
                    mask[i, j] = True
                    dist[i, j] = 0
                    forbidden_dist.add(flat_idx)
                    new_pending_dist_masked.add(flat_idx)
            pending_threshold_only = set()
            pending_threshold_only.add(np.int64(-1))
            pending_threshold_only.remove(np.int64(-1))
            active_size = deactivate_forbidden(forbidden_dist, dense_to_sparse, active_sparse, sparse_to_active, active_size)
            pending_dist_masked = new_pending_dist_masked

            count = np.count_nonzero((csm == 0) | mask)
            print("amount:", csm.size - count)
            print("percentage:", 100.0 * count / csm.size)
            paths.append(path)

    return paths
