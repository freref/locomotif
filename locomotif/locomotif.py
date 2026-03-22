import numpy as np

from . import loco
from .path import Path

import numba
from numba import int32, float32, boolean
from numba import njit
from numba.typed import List

class _LazyPathCollection:

    def __init__(self, path_data):
        self._path_data = path_data

    def __len__(self):
        return len(self._path_data[7])

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        return _materialize_path_from_graph(self._path_data, idx)


def _materialize_path_from_graph(path_data, idx):
    path_starts, _, cum_offsets, node_rows, node_cols, _, cumulative, _, _, _ = path_data
    start = int(path_starts[idx])
    end = int(path_starts[idx + 1])
    path = np.empty((end - start, 2), dtype=np.int32)
    path[:, 0] = node_rows[start:end]
    path[:, 1] = node_cols[start:end]
    cum_start = int(cum_offsets[idx])
    similarities = cumulative[cum_start + 1 : cum_start + 1 + (end - start)] - cumulative[cum_start : cum_start + (end - start)]
    return Path(path, similarities.astype(np.float32))


@njit(cache=True)
def _build_compact_path_graph(raw_paths, sm):
    total_paths = 0
    total_nodes = np.int64(0)
    total_cols = np.int64(0)
    total_cum = np.int64(0)

    for raw_path in raw_paths:
        path_len = len(raw_path)
        total_paths += 1
        total_nodes += path_len
        total_cols += raw_path[path_len - 1, 1] - raw_path[0, 1] + 1
        total_cum += path_len + 1

        is_diagonal = True
        for k in range(path_len):
            if raw_path[k, 0] != raw_path[k, 1]:
                is_diagonal = False
                break
        if not is_diagonal:
            total_paths += 1
            total_nodes += path_len
            total_cols += raw_path[path_len - 1, 0] - raw_path[0, 0] + 1
            total_cum += path_len + 1

    path_starts = np.empty(total_paths + 1, dtype=np.int64)
    col_offsets = np.empty(total_paths + 1, dtype=np.int64)
    cum_offsets = np.empty(total_paths + 1, dtype=np.int64)
    node_rows = np.empty(total_nodes, dtype=np.int32)
    node_cols = np.empty(total_nodes, dtype=np.int32)
    index_j = np.empty(total_cols, dtype=np.int32)
    cumulative = np.empty(total_cum, dtype=np.float32)
    path_j1 = np.empty(total_paths, dtype=np.int32)
    path_jl = np.empty(total_paths, dtype=np.int32)

    path_idx = 0
    node_cursor = np.int64(0)
    col_cursor = np.int64(0)
    cum_cursor = np.int64(0)

    for raw_path in raw_paths:
        path_idx, node_cursor, col_cursor, cum_cursor = _append_compact_path(
            raw_path,
            sm,
            False,
            path_idx,
            node_cursor,
            col_cursor,
            cum_cursor,
            path_starts,
            col_offsets,
            cum_offsets,
            node_rows,
            node_cols,
            index_j,
            cumulative,
            path_j1,
            path_jl,
        )

        is_diagonal = True
        for k in range(len(raw_path)):
            if raw_path[k, 0] != raw_path[k, 1]:
                is_diagonal = False
                break
        if not is_diagonal:
            path_idx, node_cursor, col_cursor, cum_cursor = _append_compact_path(
                raw_path,
                sm,
                True,
                path_idx,
                node_cursor,
                col_cursor,
                cum_cursor,
                path_starts,
                col_offsets,
                cum_offsets,
                node_rows,
                node_cols,
                index_j,
                cumulative,
                path_j1,
                path_jl,
            )

    path_starts[path_idx] = node_cursor
    col_offsets[path_idx] = col_cursor
    cum_offsets[path_idx] = cum_cursor
    start_order = np.argsort(path_j1).astype(np.int32)
    return (
        path_starts,
        col_offsets,
        cum_offsets,
        node_rows,
        node_cols,
        index_j,
        cumulative,
        path_j1,
        path_jl,
        start_order,
    )


@njit(cache=True)
def _append_compact_path(
    raw_path,
    sm,
    mirrored,
    path_idx,
    node_cursor,
    col_cursor,
    cum_cursor,
    path_starts,
    col_offsets,
    cum_offsets,
    node_rows,
    node_cols,
    index_j,
    cumulative,
    path_j1,
    path_jl,
):
    path_len = len(raw_path)
    if mirrored:
        first_j = raw_path[0, 0]
        last_j = raw_path[path_len - 1, 0]
    else:
        first_j = raw_path[0, 1]
        last_j = raw_path[path_len - 1, 1]

    col_len = last_j - first_j + 1
    path_starts[path_idx] = node_cursor
    col_offsets[path_idx] = col_cursor
    cum_offsets[path_idx] = cum_cursor
    path_j1[path_idx] = first_j
    path_jl[path_idx] = last_j + 1

    for t in range(col_len):
        index_j[col_cursor + np.int64(t)] = 0

    cumulative[cum_cursor] = 0.0
    prev_j = first_j

    for k in range(path_len):
        sim_row = raw_path[k, 0]
        sim_col = raw_path[k, 1]
        if mirrored:
            row = sim_col
            col = sim_row
        else:
            row = sim_row
            col = sim_col

        node_rows[node_cursor + np.int64(k)] = row
        node_cols[node_cursor + np.int64(k)] = col
        cumulative[cum_cursor + np.int64(k + 1)] = cumulative[cum_cursor + np.int64(k)] + sm[sim_row, sim_col]

        if k > 0 and col != prev_j:
            start = col_cursor + np.int64(prev_j - first_j) + 1
            end = col_cursor + np.int64(col - first_j) + 1
            for t in range(start, end):
                index_j[t] = k
            prev_j = col

    return path_idx + 1, node_cursor + path_len, col_cursor + col_len, cum_cursor + path_len + 1


def apply_locomotif(ts, l_min, l_max, rho=None, nb=None, start_mask=None, end_mask=None, overlap=0.0, warping=True):
    """Apply the LoCoMotif algorithm to find motif sets in the given time ts.

    :param ts: Univariate or multivariate time series, with the time axis being the 0-th dimension.
    :param l_min: Minimum length of the representative motifs.
    :param l_max: Maximum length of the representative motifs.
    :param rho: The strictness parameter between 0 and 1. It is the quantile of the similarity matrix to use as the threshold for the LoCo algorithm.
    :param nb: Maximum number of motif sets to find.
    :param start_mask: Mask for the starting time points of representative motifs, where True means allowed. If None, all points are allowed.
    :param end_mask: Mask for the ending time points of representative motifs, where True means allowed. If None, all points are allowed.
    :param overlap: Maximum allowed overlap between motifs, between 0 and 0.5. A new motif β can be discovered only when |β ∩ β'|/|β'| is less than this value for all existing motifs β'.
    :param warping: Whether warping is allowed (True) or not (False).
    
    :return: motif_sets: a list of motif sets, where each motif set is a list of segments as tuples.
    """   
    # Get a locomotif instance
    lcm = get_locomotif_instance(ts, l_min, l_max, rho=rho, warping=warping)
    # Apply LoCo
    lcm.find_best_paths(vwidth=l_min // 2)
    # Find the `nb` best motif sets
    motif_sets = []
    for representative, motif_set, _ in lcm.find_best_motif_sets(nb=nb, overlap=overlap, start_mask=start_mask, end_mask=end_mask):
        motif_sets.append((representative, motif_set))
    return motif_sets

def get_locomotif_instance(ts, l_min, l_max, rho=None, warping=True):
    return LoCoMotif.instance_from_rho(ts, l_min=l_min, l_max=l_max, rho=rho, warping=warping)


class LoCoMotif:

    def __init__(self, ts, l_min, l_max, gamma=None, tau=0.5, delta_a=1.0, delta_m=0.5, warping=True):        
        self.ts = ts
        l_min = max(4, l_min)
        self.l_min = np.int32(l_min)
        self.l_max = np.int32(l_max)
        # LoCo instance
        self._loco = loco.LoCo(ts, gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m, warping=warping)
        self._paths = None
        self._path_data = None
        self._path_collection = None

    @classmethod
    def instance_from_rho(cls, ts, l_min, l_max, rho=None, warping=True):
        # Handle default rho value
        if rho is None:
            rho = 0.8 if warping else 0.5  
        lcm = cls(ts=ts, l_min=l_min, l_max=l_max)
        lcm._loco = loco.LoCo.instance_from_rho(ts, rho, gamma=None, warping=warping)
        return lcm

    def find_best_paths(self, vwidth=None):
        vwidth = np.maximum(10, self.l_min // 2)
        raw_paths = self._loco.find_best_paths(self.l_min, vwidth)
        raw_paths = [np.ascontiguousarray(path, dtype=np.int32) for path in raw_paths]
        self._path_data = _build_compact_path_graph(raw_paths, self.self_similarity_matrix)
        self._path_collection = _LazyPathCollection(self._path_data)
        self._paths = None
        return self._path_collection

    def induced_paths(self, b, e, mask=None):
        if mask is None:
            mask = np.full(len(self.ts), False)
        if self._path_data is not None:
            induced = _induced_paths_graph(b, e, mask, *self._path_data[:9])
            return [(int(segment[0]), int(segment[1])) for segment in induced]
        return _induced_paths(b, e, mask, self._paths)

    # iteratively finds the best motif set
    def find_best_motif_sets(self, nb=None, start_mask=None, end_mask=None, overlap=0.0, keep_fitnesses=False):
        if self._paths is None and self._path_data is None:
            self.find_best_paths()
            
        n = len(self.ts)
        # Handle masks
        if start_mask is None:
            start_mask = np.full(n, True)
        if end_mask is None:
            end_mask   = np.full(n, True)
    
        assert 0.0 <= overlap and overlap <= 0.5
        assert start_mask.shape == (n,)
        assert end_mask.shape   == (n,)

        # iteratively find best motif sets
        current_nb = 0
        mask       = np.full(n, False)
        while (nb is None or current_nb < nb):

            if np.all(mask) or not np.any(start_mask) or not np.any(end_mask):
                break

            start_mask[mask] = False
            end_mask[mask]   = False

            if self._path_data is not None:
                (b, e), best_fitness, fitnesses = _find_best_candidate_graph(
                    n,
                    self.l_min,
                    self.l_max,
                    overlap,
                    mask,
                    mask,
                    start_mask,
                    end_mask,
                    *self._path_data,
                    keep_fitnesses=keep_fitnesses,
                )
            else:
                (b, e), best_fitness, fitnesses = _find_best_candidate(
                    self._paths,
                    n,
                    self.l_min,
                    self.l_max,
                    overlap,
                    mask,
                    mask,
                    start_mask,
                    end_mask,
                    keep_fitnesses=keep_fitnesses,
                )

            if best_fitness == 0.0:
                break

            motif_set = self.induced_paths(b, e, mask)
            mask = _mask_motif_set(mask, motif_set, overlap)

            current_nb += 1
            yield (b, e), motif_set, fitnesses
            
    @property
    def local_warping_paths(self):
        if self._path_collection is not None:
            return self._path_collection
        return self._paths
    
    @property
    def self_similarity_matrix(self):
        return self._loco.similarity_matrix
    
    @property
    def cumulative_similarity_matrix(self):
        return self._loco.cumulative_similarity_matrix

def _mask_motif_set(mask, motif_set, overlap):
    for (b_m, e_m) in motif_set:
        l = e_m - b_m
        l_mask = max(1, int((1 - 2*overlap) * l)) # mask length must be lower bounded by 1 (otherwise, nothing is masked when overlap=0.5)
        mask[b_m + (l - l_mask)//2 : b_m + (l - l_mask)//2 + l_mask] = True
    return mask

def _induced_paths(b, e, mask, P):
    induced_paths = []
    for path in P:        
        if b < path.j1 or path.jl < e:
            continue
        kb = path.find_j(b)
        ke = path.find_j(e - 1)
        b_m = path[kb][0]
        e_m = path[ke][0] + 1
        if not np.any(mask[b_m:e_m]):
            induced_paths.append((b_m, e_m))
    return induced_paths


@njit(cache=True)
def _induced_paths_graph(b, e, mask, path_starts, col_offsets, cum_offsets, node_rows, node_cols, index_j, cumulative, path_j1, path_jl):
    count = 0
    for path_idx in range(len(path_j1)):
        if b < path_j1[path_idx] or path_jl[path_idx] < e:
            continue
        col_base = col_offsets[path_idx]
        node_base = path_starts[path_idx]
        kb = index_j[col_base + np.int64(b - path_j1[path_idx])]
        ke = index_j[col_base + np.int64(e - 1 - path_j1[path_idx])]
        b_m = node_rows[node_base + np.int64(kb)]
        e_m = node_rows[node_base + np.int64(ke)] + 1
        blocked = False
        for pos in range(b_m, e_m):
            if mask[pos]:
                blocked = True
                break
        if not blocked:
            count += 1

    out = np.empty((count, 2), dtype=np.int32)
    write_idx = 0
    for path_idx in range(len(path_j1)):
        if b < path_j1[path_idx] or path_jl[path_idx] < e:
            continue
        col_base = col_offsets[path_idx]
        node_base = path_starts[path_idx]
        kb = index_j[col_base + np.int64(b - path_j1[path_idx])]
        ke = index_j[col_base + np.int64(e - 1 - path_j1[path_idx])]
        b_m = node_rows[node_base + np.int64(kb)]
        e_m = node_rows[node_base + np.int64(ke)] + 1
        blocked = False
        for pos in range(b_m, e_m):
            if mask[pos]:
                blocked = True
                break
        if not blocked:
            out[write_idx, 0] = b_m
            out[write_idx, 1] = e_m
            write_idx += 1
    return out

from numba.experimental import jitclass
@jitclass([
    ('keys', int32[:]),    
    ('path_indices', int32[:]),
    ('size', int32),      
    ('capacity', int32),
    ('P', numba.types.ListType(Path.class_type.instance_type)),
    ('j1s', int32[:]),
    ('jls', int32[:]),
    ('Q', int32[:]),
    ('q', int32),
    ('j', int32),
])
class SortedPathArray:

    def __init__(self, P, j, capacity):
        """Initialize a sorted list where 'keys' are sorted, and 'values' (indices) follow."""
        self.P = P
        self.keys = np.empty(capacity, np.int32)
        self.path_indices = np.empty(capacity, np.int32)
        
        self.size = 0
        self.capacity = capacity

        self.j1s = np.array([path.j1 for path in P], np.int32)
        self.jls = np.array([path.jl for path in P], np.int32)

        # Sort the paths on j1. This is the order in which they become relevant.
        self.Q = np.argsort(self.j1s).astype(np.int32)
        self.q = 0

        # TODO: Can be implemented more efficiently
        self.j = -1
        for _ in range(j+1):
            self.increment_j()


    def increment_j(self):
        self.j += 1

        # Remove the paths for which jl == j
        k_remove = 0
        for _ in range(self.size):
            if self.jls[self.path_indices[k_remove]] == self.j:
                self._remove_at(k_remove)
            else:
                k_remove += 1

        # If a path will be inserted, update the keys
        # if self.j1s[self.Q[self.q]] == self.j:
        self._update_keys()

        # Insert all paths for which j1 == b
        for q in range(self.q, len(self.P)):
            path_index = self.Q[q]
            if self.j1s[path_index] == self.j:
                self._insert(path_index)
            else:
                break
        self.q = q


    def get_path_at(self, k):
        return self.P[self.path_indices[k]]

    def _update_keys(self):
        # Update the keys (as paths cannot cross, the ordering of keys does not change)
        for k in range(self.size):
            path_to_update = self.get_path_at(k)
            self.keys[k] = path_to_update[path_to_update.find_j(self.j)][0]
  
    def _insert(self, path_index):
        """Insert (key, value) while maintaining sorted order of keys."""
        assert self.size < self.capacity
        # Binary search to find the correct position of the key
        path_to_insert = self.P[path_index]
        key = path_to_insert[path_to_insert.find_j(self.j)][0]

        k = np.searchsorted(self.keys[:self.size], key)
        # Shift items to the right
        self.keys[k+1:self.size+1] = self.keys[k:self.size]
        self.path_indices[k+1:self.size+1] = self.path_indices[k:self.size]
        # Insert the new item
        self.keys[k] = key
        self.path_indices[k] = path_index 
        # Increase size
        self.size += 1 

    def _remove_at(self, k):
        """Removes an item at a specific index."""
        assert k >= 0 and k < self.size
        # Shift elements left to fill the gap
        self.keys[k:self.size-1] = self.keys[k+1:self.size]
        self.path_indices[k:self.size-1] = self.path_indices[k+1:self.size]
        # Last element need not be cleared
        self.size -= 1  # Decrease size


@njit(numba.types.Tuple((numba.types.UniTuple(int32, 2), float32, float32[:, :]))(numba.types.ListType(Path.class_type.instance_type), int32, int32, int32, float32, boolean[:], boolean[:], boolean[:], boolean[:], boolean), cache=True)
def _find_best_candidate(P, n, l_min, l_max, nu, row_mask, col_mask, start_mask, end_mask, keep_fitnesses=False): 
    fitnesses = []
    # n is used for coverage normalization 
    r = len(row_mask)
    c = len(col_mask)

    # Max number of relevant paths
    max_size = int(np.ceil(r / (l_min // 2 + 1))) 
    # Initialize Pb
    Pb  = SortedPathArray(P, -1, max_size)
    # Pe is implemented as a mask
    Pe  = np.zeros(max_size)
    # Initialize
    es_checked = np.zeros(max_size, dtype=np.int32)

    best_fitness   = 0.0
    best_candidate = (0, 0) 

    # b-loop
    for b_repr in range(c - l_min + 1):
        Pb.increment_j()

        nb_paths = Pb.size
        
        # If less than 2 paths in Pb, skip this b.
        if nb_paths < 2 or not start_mask[b_repr] or col_mask[b_repr]:
            continue

        ### Check initial coincidence with previously discovered motifs
        # For the representative
        if np.any(col_mask[b_repr:b_repr + l_min - 1]):
            continue

        # For each of the induced segments
        Pe[:nb_paths] = True
        es_checked[:nb_paths] = Pb.keys[:nb_paths] 
        nb_remaining_paths = nb_paths

        for e_repr in range(b_repr + l_min, min(c + 1, b_repr + l_max + 1)):

            # Check coincidence with previously found motifs
            # For the representative 
            if col_mask[e_repr-1]:
                break

            # Skip iteration if representative cannot end at this index
            if not end_mask[e_repr-1]:
                continue

            # Calculate the fitness
            score = 0.0
            total_length = 0.0
            total_path_length = 0.0
            total_overlap = 0.0
            l_prev = 0
            e_prev = 0
            too_much_overlap = False

            for k in range(nb_paths):

                if nb_remaining_paths < 2:
                    break

                if not Pe[k]:
                    continue

                path = Pb.get_path_at(k)
                if path.jl < e_repr:
                    Pe[k] = False
                    nb_remaining_paths -= 1
                    continue

                kb = path.find_j(b_repr)
                b = path[kb][0]
                ke = path.find_j(e_repr-1)
                e  = path[ke][0] + 1
                
                if np.any(row_mask[es_checked[k]:e]):
                    Pe[k] = False
                    nb_remaining_paths -= 1
                    continue
                es_checked[k] = e

                l = e - b
                # Handle overlap within motif set
                if k > 0:
                    overlap = max(0, e_prev - b)
                    if nu * min(l, l_prev) < overlap:
                        too_much_overlap = True
                        break
                    total_overlap += overlap
            
                total_length += l
                total_path_length += ke - kb + 1
                score += path.cumulative_similarities[ke+1] - path.cumulative_similarities[kb]

                l_prev = l
                e_prev = e

            if nb_remaining_paths < 2:
                break

            if too_much_overlap:
                continue

            # Calculate normalized score and coverage
            l_repr = e_repr - b_repr
            n_score = (score - l_repr) / total_path_length
            n_coverage = (total_length - total_overlap - l_repr) / float(n)

            # Calculate the fitness value
            fit = 0.0
            if n_coverage != 0 or n_score != 0:
                fit = 2 * (n_coverage * n_score) / (n_coverage + n_score)
    
            if fit == 0.0:
                continue

            # Update best fitness
            if fit > best_fitness:
                best_candidate = (b_repr, e_repr)
                best_fitness   = fit

            # Store fitness if necessary
            if keep_fitnesses:
                fitnesses.append((b_repr, e_repr, fit, n_coverage, n_score))
        


    fitnesses = np.array(fitnesses, dtype=np.float32) if keep_fitnesses and fitnesses else np.empty((0, 5), dtype=np.float32)

    return best_candidate, best_fitness, fitnesses


@njit(cache=True)
def _find_best_candidate_graph(
    n,
    l_min,
    l_max,
    nu,
    row_mask,
    col_mask,
    start_mask,
    end_mask,
    path_starts,
    col_offsets,
    cum_offsets,
    node_rows,
    node_cols,
    index_j,
    cumulative,
    path_j1,
    path_jl,
    start_order,
    keep_fitnesses=False,
):
    fitnesses = []
    r = len(row_mask)
    c = len(col_mask)
    row_prefix = np.zeros(r + 1, dtype=np.int32)
    col_prefix = np.zeros(c + 1, dtype=np.int32)
    for idx in range(r):
        row_prefix[idx + 1] = row_prefix[idx] + (1 if row_mask[idx] else 0)
    for idx in range(c):
        col_prefix[idx + 1] = col_prefix[idx] + (1 if col_mask[idx] else 0)

    max_size = max(2, int(np.ceil(r / (l_min // 2 + 1))))
    active_paths = np.empty(max_size, dtype=np.int32)
    active_keys = np.empty(max_size, dtype=np.int32)
    merge_paths = np.empty(max_size, dtype=np.int32)
    merge_keys = np.empty(max_size, dtype=np.int32)
    new_paths = np.empty(max_size, dtype=np.int32)
    new_keys = np.empty(max_size, dtype=np.int32)
    active_size = 0
    next_start_idx = 0

    best_fitness = 0.0
    best_candidate = (0, 0)

    for b_repr in range(c - l_min + 1):
        next_j = b_repr

        write_idx = 0
        for active_idx in range(active_size):
            path_idx = active_paths[active_idx]
            if path_jl[path_idx] != next_j:
                active_paths[write_idx] = path_idx
                write_idx += 1
        active_size = write_idx

        for active_idx in range(active_size):
            path_idx = active_paths[active_idx]
            col_base = col_offsets[path_idx]
            node_base = path_starts[path_idx]
            key_idx = index_j[col_base + np.int64(next_j - path_j1[path_idx])]
            active_keys[active_idx] = node_rows[node_base + np.int64(key_idx)]

        group_start = next_start_idx
        while next_start_idx < len(start_order):
            path_idx = start_order[next_start_idx]
            if path_j1[path_idx] != next_j:
                break
            next_start_idx += 1

        group_size = next_start_idx - group_start
        if group_size > 0:
            write_idx = 0
            for idx in range(next_start_idx - 1, group_start - 1, -1):
                path_idx = start_order[idx]
                col_base = col_offsets[path_idx]
                node_base = path_starts[path_idx]
                key_idx = index_j[col_base]
                new_paths[write_idx] = path_idx
                new_keys[write_idx] = node_rows[node_base + np.int64(key_idx)]
                write_idx += 1

            for idx in range(1, group_size):
                key = new_keys[idx]
                path_idx = new_paths[idx]
                j = idx
                while j > 0 and key < new_keys[j - 1]:
                    new_keys[j] = new_keys[j - 1]
                    new_paths[j] = new_paths[j - 1]
                    j -= 1
                new_keys[j] = key
                new_paths[j] = path_idx

            old_idx = 0
            new_idx = 0
            merged_size = active_size + group_size
            write_idx = 0
            while old_idx < active_size and new_idx < group_size:
                if new_keys[new_idx] <= active_keys[old_idx]:
                    merge_keys[write_idx] = new_keys[new_idx]
                    merge_paths[write_idx] = new_paths[new_idx]
                    new_idx += 1
                else:
                    merge_keys[write_idx] = active_keys[old_idx]
                    merge_paths[write_idx] = active_paths[old_idx]
                    old_idx += 1
                write_idx += 1
            while new_idx < group_size:
                merge_keys[write_idx] = new_keys[new_idx]
                merge_paths[write_idx] = new_paths[new_idx]
                new_idx += 1
                write_idx += 1
            while old_idx < active_size:
                merge_keys[write_idx] = active_keys[old_idx]
                merge_paths[write_idx] = active_paths[old_idx]
                old_idx += 1
                write_idx += 1

            active_size = merged_size
            active_keys[:active_size] = merge_keys[:active_size]
            active_paths[:active_size] = merge_paths[:active_size]

        nb_paths = active_size
        if nb_paths < 2 or not start_mask[b_repr] or col_mask[b_repr]:
            continue

        max_possible_cov = (nb_paths * l_max - l_min) / float(n)
        if 2.0 * max_possible_cov < best_fitness:
            continue

        if col_prefix[b_repr + l_min - 1] - col_prefix[b_repr] > 0:
            continue

        E_max = min(c + 1, b_repr + l_max + 1) - (b_repr + l_min)
        E = 0
        for e_idx in range(E_max):
            if col_mask[b_repr + l_min + e_idx - 1]:
                break
            E += 1

        if E == 0:
            continue

        vec_score = np.zeros(E, dtype=np.float32)
        vec_len = np.zeros(E, dtype=np.int32)
        vec_path_len = np.zeros(E, dtype=np.int32)
        vec_overlap = np.zeros(E, dtype=np.int32)
        vec_valid = np.ones(E, dtype=np.bool_)

        for e_idx in range(E):
            if not end_mask[b_repr + l_min + e_idx - 1]:
                vec_valid[e_idx] = False

        e_prev_array = np.zeros(E, dtype=np.int32)
        l_prev_array = np.zeros(E, dtype=np.int32)
        paths_added = np.zeros(E, dtype=np.int32)

        for active_idx in range(nb_paths):
            path_idx = active_paths[active_idx]
            col_base = col_offsets[path_idx]
            node_base = path_starts[path_idx]
            cum_base = cum_offsets[path_idx]

            kb = index_j[col_base + np.int64(b_repr - path_j1[path_idx])]
            b = node_rows[node_base + np.int64(kb)]
            cum_b = cumulative[cum_base + np.int64(kb)]
            path_end = path_jl[path_idx]

            es_checked = b
            base_idx_e = col_base + np.int64(b_repr + l_min - 1 - path_j1[path_idx])

            for e_idx in range(E):
                if not vec_valid[e_idx]:
                    continue

                e_repr = b_repr + l_min + e_idx
                if path_end < e_repr:
                    break

                ke = index_j[base_idx_e + e_idx]
                e = node_rows[node_base + np.int64(ke)] + 1

                if row_prefix[e] - row_prefix[es_checked] > 0:
                    break

                es_checked = e
                l = e - b

                if paths_added[e_idx] > 0:
                    overlap = max(0, e_prev_array[e_idx] - b)
                    if nu * min(l, l_prev_array[e_idx]) < overlap:
                        vec_valid[e_idx] = False
                        continue
                    vec_overlap[e_idx] += overlap

                vec_len[e_idx] += l
                vec_path_len[e_idx] += ke - kb + 1
                vec_score[e_idx] += cumulative[cum_base + np.int64(ke + 1)] - cum_b

                e_prev_array[e_idx] = e
                l_prev_array[e_idx] = l
                paths_added[e_idx] += 1

        for e_idx in range(E):
            if not vec_valid[e_idx] or paths_added[e_idx] < 2:
                continue

            e_repr = b_repr + l_min + e_idx
            l_repr = e_repr - b_repr

            n_score = (vec_score[e_idx] - l_repr) / vec_path_len[e_idx]
            n_coverage = (vec_len[e_idx] - vec_overlap[e_idx] - l_repr) / float(n)

            fit = 0.0
            if n_coverage != 0 or n_score != 0:
                fit = 2 * (n_coverage * n_score) / (n_coverage + n_score)

            if fit > 0.0:
                if fit > best_fitness:
                    best_candidate = (b_repr, e_repr)
                    best_fitness = fit
                if keep_fitnesses:
                    fitnesses.append((b_repr, e_repr, fit, n_coverage, n_score))

    fitnesses = np.array(fitnesses, dtype=np.float32) if keep_fitnesses and fitnesses else np.empty((0, 5), dtype=np.float32)
    return best_candidate, best_fitness, fitnesses
