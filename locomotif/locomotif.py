import numpy as np

from . import loco
from .path import Path, project_to_vertical_axis

import numba
from numba import int32, float32, boolean
from numba import njit
from numba.typed import List

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
        self._paths_soa = None

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
        paths = self._loco.find_best_paths(self.l_min, vwidth)
       
        # LoCo finds paths the part of the SSM above the diagonal. 
        # Here, the paths are converted to Path objects, and the mirrored versions of the found paths are added.
        self._paths = List()
        
        for path in paths:
            i, j = path[:, 0], path[:, 1]
            path_similarities = self.self_similarity_matrix[i, j]
            self._paths.append(Path(path, path_similarities))
            # Also add the mirrored path
            # Do not mirror the diagonal
            if np.all(i == j):
                continue
            path_mirrored = np.zeros(path.shape, dtype=np.int32)
            path_mirrored[:, 0] = j
            path_mirrored[:, 1] = i
            self._paths.append(Path(path_mirrored, path_similarities))

        self._paths_soa = None
        return self._paths

    def induced_paths(self, b, e, mask=None):
        if mask is None:
            mask = np.full(len(self.ts), False)
        return _induced_paths(b, e, mask, self._paths)

    # iteratively finds the best motif set
    def find_best_motif_sets(self, nb=None, start_mask=None, end_mask=None, overlap=0.0, keep_fitnesses=False):
        if self._paths is None:
            self.find_best_paths()
        if not keep_fitnesses and self._paths_soa is None:
            self._paths_soa = _build_paths_soa(self._paths)
            
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

            if keep_fitnesses:
                (b, e), best_fitness, fitnesses = _find_best_candidate(self._paths, n, self.l_min, self.l_max, overlap, mask, mask, start_mask, end_mask, keep_fitnesses=True)
            else:
                b, e, best_fitness = _find_best_candidate_soa(
                    np.int32(n),
                    self.l_min,
                    self.l_max,
                    np.float32(overlap),
                    mask,
                    mask,
                    start_mask,
                    end_mask,
                    self._paths_soa[0],
                    self._paths_soa[1],
                    self._paths_soa[2],
                    self._paths_soa[3],
                    self._paths_soa[4],
                    self._paths_soa[5],
                    self._paths_soa[6],
                    self._paths_soa[7],
                )
                fitnesses = np.empty((0, 5), dtype=np.float32)

            if best_fitness == 0.0:
                break

            if not keep_fitnesses and self._paths_soa is not None:
                segments = _induced_segments_soa(
                    b,
                    e,
                    mask,
                    self._paths_soa[0],
                    self._paths_soa[1],
                    self._paths_soa[2],
                    self._paths_soa[3],
                    self._paths_soa[4],
                    self._paths_soa[5],
                )
                motif_set = [(segments[i, 0], segments[i, 1]) for i in range(len(segments))]
            else:
                motif_set = [project_to_vertical_axis(induced_path) for induced_path in self.induced_paths(b, e, mask)]
            mask = _mask_motif_set(mask, motif_set, overlap)

            current_nb += 1
            yield (b, e), motif_set, fitnesses
            
    @property
    def local_warping_paths(self):
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
        induced_path = path.get_subpath_between_col_indices(b, e-1)
        b_m, e_m = project_to_vertical_axis(induced_path)
        if not np.any(mask[b_m:e_m]):
            induced_paths.append(induced_path)
    return induced_paths


def _build_paths_soa(P):
    return _build_paths_soa_numba(P)


@njit(cache=True)
def _build_paths_soa_numba(P):
    n_paths = len(P)
    j1s = np.empty(n_paths, dtype=np.int32)
    jls = np.empty(n_paths, dtype=np.int32)

    index_offsets = np.empty(n_paths + 1, dtype=np.int32)
    path_offsets = np.empty(n_paths + 1, dtype=np.int32)
    csum_offsets = np.empty(n_paths + 1, dtype=np.int32)

    index_offsets[0] = 0
    path_offsets[0] = 0
    csum_offsets[0] = 0

    for i in range(n_paths):
        path = P[i]
        j1s[i] = path.j1
        jls[i] = path.jl
        index_offsets[i + 1] = index_offsets[i] + len(path.index_j)
        path_offsets[i + 1] = path_offsets[i] + len(path.path)
        csum_offsets[i + 1] = csum_offsets[i] + len(path.cumulative_similarities)

    index_values = np.empty(index_offsets[-1], dtype=np.int32)
    path_rows = np.empty(path_offsets[-1], dtype=np.int32)
    csum_values = np.empty(csum_offsets[-1], dtype=np.float32)

    for i in range(n_paths):
        path = P[i]
        i0, i1 = index_offsets[i], index_offsets[i + 1]
        p0, p1 = path_offsets[i], path_offsets[i + 1]
        s0, s1 = csum_offsets[i], csum_offsets[i + 1]
        for t in range(i1 - i0):
            index_values[i0 + t] = path.index_j[t]
        for t in range(p1 - p0):
            path_rows[p0 + t] = path.path[t, 0]
        for t in range(s1 - s0):
            csum_values[s0 + t] = path.cumulative_similarities[t]

    return (
        np.ascontiguousarray(j1s),
        np.ascontiguousarray(jls),
        np.ascontiguousarray(index_offsets),
        np.ascontiguousarray(index_values),
        np.ascontiguousarray(path_offsets),
        np.ascontiguousarray(path_rows),
        np.ascontiguousarray(csum_offsets),
        np.ascontiguousarray(csum_values),
    )

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
def _find_best_candidate_soa(
    n,
    l_min,
    l_max,
    nu,
    row_mask,
    col_mask,
    start_mask,
    end_mask,
    j1s,
    jls,
    index_offsets,
    index_values,
    path_offsets,
    path_rows,
    csum_offsets,
    csum_values,
):
    c = len(col_mask)
    n_paths_total = len(j1s)
    if n_paths_total == 0:
        return np.int32(0), np.int32(0), np.float32(0.0)

    q_order = np.argsort(j1s).astype(np.int32)
    active_keys = np.empty(n_paths_total, dtype=np.int32)
    active_paths = np.empty(n_paths_total, dtype=np.int32)
    active_size = np.int32(0)
    q = np.int32(0)

    pe = np.zeros(n_paths_total, dtype=np.bool_)
    es_checked = np.zeros(n_paths_total, dtype=np.int32)

    best_b = np.int32(0)
    best_e = np.int32(0)
    best_fitness = 0.0

    for b_repr in range(c - l_min + 1):
        new_size = np.int32(0)
        for old_k in range(active_size):
            path_index = active_paths[old_k]
            if jls[path_index] != b_repr:
                active_keys[new_size] = active_keys[old_k]
                active_paths[new_size] = path_index
                new_size += 1
        active_size = new_size

        for k in range(active_size):
            path_index = active_paths[k]
            idx0 = index_offsets[path_index]
            kb = index_values[idx0 + (b_repr - j1s[path_index])]
            active_keys[k] = path_rows[path_offsets[path_index] + kb]

        while q < n_paths_total:
            path_index = q_order[q]
            if j1s[path_index] != b_repr:
                break

            idx0 = index_offsets[path_index]
            kb = index_values[idx0 + (b_repr - j1s[path_index])]
            key = path_rows[path_offsets[path_index] + kb]

            pos = np.searchsorted(active_keys[:active_size], key)
            if pos < active_size:
                active_keys[pos + 1 : active_size + 1] = active_keys[pos:active_size]
                active_paths[pos + 1 : active_size + 1] = active_paths[pos:active_size]
            active_keys[pos] = key
            active_paths[pos] = path_index
            active_size += 1
            q += 1

        nb_paths = active_size
        if nb_paths < 2 or not start_mask[b_repr] or col_mask[b_repr]:
            continue

        blocked = False
        for t in range(b_repr, b_repr + l_min - 1):
            if col_mask[t]:
                blocked = True
                break
        if blocked:
            continue

        pe[:nb_paths] = True
        es_checked[:nb_paths] = active_keys[:nb_paths]
        nb_remaining_paths = nb_paths

        for e_repr in range(b_repr + l_min, min(c + 1, b_repr + l_max + 1)):
            if col_mask[e_repr - 1]:
                break
            if not end_mask[e_repr - 1]:
                continue

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
                if not pe[k]:
                    continue

                path_index = active_paths[k]
                if jls[path_index] < e_repr:
                    pe[k] = False
                    nb_remaining_paths -= 1
                    continue

                j1 = j1s[path_index]
                idx0 = index_offsets[path_index]
                p0 = path_offsets[path_index]
                c0 = csum_offsets[path_index]

                kb = index_values[idx0 + (b_repr - j1)]
                b = path_rows[p0 + kb]
                ke = index_values[idx0 + (e_repr - 1 - j1)]
                e = path_rows[p0 + ke] + 1

                masked = False
                for t in range(es_checked[k], e):
                    if row_mask[t]:
                        masked = True
                        break
                if masked:
                    pe[k] = False
                    nb_remaining_paths -= 1
                    continue
                es_checked[k] = e

                l = e - b
                if k > 0:
                    overlap = max(0, e_prev - b)
                    if nu * min(l, l_prev) < overlap:
                        too_much_overlap = True
                        break
                    total_overlap += overlap

                total_length += l
                total_path_length += ke - kb + 1
                score += csum_values[c0 + ke + 1] - csum_values[c0 + kb]

                l_prev = l
                e_prev = e

            if nb_remaining_paths < 2:
                break
            if too_much_overlap:
                continue

            l_repr = e_repr - b_repr
            n_score = (score - l_repr) / total_path_length
            n_coverage = (total_length - total_overlap - l_repr) / float(n)

            fit = 0.0
            if n_coverage != 0 or n_score != 0:
                fit = 2 * (n_coverage * n_score) / (n_coverage + n_score)
            if fit == 0.0:
                continue

            if fit > best_fitness:
                best_b = np.int32(b_repr)
                best_e = np.int32(e_repr)
                best_fitness = fit

    return best_b, best_e, np.float32(best_fitness)


@njit(cache=True)
def _induced_segments_soa(b, e, mask, j1s, jls, index_offsets, index_values, path_offsets, path_rows):
    n_paths = len(j1s)
    segments = np.empty((n_paths, 2), dtype=np.int32)
    count = 0

    for path_index in range(n_paths):
        j1 = j1s[path_index]
        jl = jls[path_index]
        if b < j1 or jl < e:
            continue

        idx0 = index_offsets[path_index]
        p0 = path_offsets[path_index]
        kb = index_values[idx0 + (b - j1)]
        ke = index_values[idx0 + (e - 1 - j1)]

        b_m = path_rows[p0 + kb]
        e_m = path_rows[p0 + ke] + 1

        masked = False
        for t in range(b_m, e_m):
            if mask[t]:
                masked = True
                break
        if masked:
            continue

        segments[count, 0] = b_m
        segments[count, 1] = e_m
        count += 1

    out = np.empty((count, 2), dtype=np.int32)
    for i in range(count):
        out[i, 0] = segments[i, 0]
        out[i, 1] = segments[i, 1]
    return out
