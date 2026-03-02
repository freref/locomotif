import numpy as np

from . import loco
from .path import Path, project_to_vertical_axis

import numba
from numba import int32, float32, boolean
from numba import njit
from numba.typed import List as TypedList

class _LazyPathCollection:

    def __init__(self, path_data): self._path_data = path_data

    def __len__(self):
        return len(self._path_data[0]) - 1

    def __iter__(self):
        for idx in range(len(self)): yield self[idx]

    def __getitem__(self, idx):
        if isinstance(idx, slice): return [self[i] for i in range(*idx.indices(len(self)))]
        return _materialize_path_from_graph(self._path_data, idx)


def _materialize_path_from_graph(path_data, idx):
    path_starts, _, _, node_rows, node_cols, _, cumulative, _, _, _ = path_data
    start = int(path_starts[idx]); end = int(path_starts[idx + 1])
    path = np.empty((end - start, 2), dtype=np.int32)
    path[:, 0] = node_rows[start:end]; path[:, 1] = node_cols[start:end]
    similarities = np.empty(len(path), dtype=np.float32)
    return Path(path, similarities)


@njit(cache=True)
def _build_compact_path_graph(raw_paths, sm, symmetric):
    diagonal_len = sm.shape[0] if symmetric else 0
    total_paths = 1 if symmetric else 0
    total_nodes = np.int64(diagonal_len)
    total_cols = np.int64(diagonal_len)
    total_cum = np.int64(diagonal_len + 1) if symmetric else np.int64(0)

    for raw_path in raw_paths:
        path_len = len(raw_path); total_paths += 1; total_nodes += path_len
        total_cols += (raw_path[path_len - 1, 1]) - (raw_path[0, 1]) + 1; total_cum += path_len + 1
        is_diagonal = True
        for k in range(path_len):
            if raw_path[k, 0] != raw_path[k, 1]: is_diagonal = False; break
        if not is_diagonal:
            total_paths += 1; total_nodes += path_len
            total_cols += (raw_path[path_len - 1, 0]) - (raw_path[0, 0]) + 1; total_cum += path_len + 1

    path_starts = np.empty(total_paths + 1, dtype=np.int64); col_offsets = np.empty(total_paths + 1, dtype=np.int64); cum_offsets = np.empty(total_paths + 1, dtype=np.int64)
    node_rows = np.empty(total_nodes, dtype=np.int32); node_cols = np.empty(total_nodes, dtype=np.int32); index_j = np.empty(total_cols, dtype=np.int32)
    cumulative = np.empty(total_cum, dtype=np.float32); path_j1 = np.empty(total_paths, dtype=np.int32); path_jl = np.empty(total_paths, dtype=np.int32)

    path_idx = 0; node_cursor = np.int64(0); col_cursor = np.int64(0); cum_cursor = np.int64(0)

    if symmetric:
        path_starts[path_idx] = node_cursor; col_offsets[path_idx] = col_cursor; cum_offsets[path_idx] = cum_cursor
        path_j1[path_idx] = 0; path_jl[path_idx] = diagonal_len; cumulative[cum_cursor] = 0.0
        for k in range(diagonal_len):
            node_rows[node_cursor + np.int64(k)] = k; node_cols[node_cursor + np.int64(k)] = k
            index_j[col_cursor + np.int64(k)] = k; cumulative[cum_cursor + np.int64(k + 1)] = cumulative[cum_cursor + np.int64(k)] + sm[k, k]
        path_idx += 1; node_cursor += diagonal_len; col_cursor += diagonal_len; cum_cursor += diagonal_len + 1

    for raw_path in raw_paths:
        path_idx, node_cursor, col_cursor, cum_cursor = _append_compact_path(raw_path, sm, False, path_idx, node_cursor, col_cursor, cum_cursor, path_starts, col_offsets, cum_offsets, node_rows, node_cols, index_j, cumulative, path_j1, path_jl)
        is_diagonal = True
        for k in range(len(raw_path)):
            if raw_path[k, 0] != raw_path[k, 1]: is_diagonal = False; break
        if not is_diagonal:
            path_idx, node_cursor, col_cursor, cum_cursor = _append_compact_path(raw_path, sm, True, path_idx, node_cursor, col_cursor, cum_cursor, path_starts, col_offsets, cum_offsets, node_rows, node_cols, index_j, cumulative, path_j1, path_jl)

    path_starts[path_idx] = node_cursor
    sort_keys = np.empty(total_paths, dtype=np.int64)
    for i in range(total_paths): sort_keys[i] = (np.int64(path_j1[i]) << 32) | np.int64(i)
    start_order = np.argsort(sort_keys).astype(np.int32)
    return (path_starts, col_offsets, cum_offsets, node_rows, node_cols, index_j, cumulative, path_j1, path_jl, start_order)


@njit(cache=True)
def _append_compact_path(raw_path, sm, mirrored, path_idx, node_cursor, col_cursor, cum_cursor, path_starts, col_offsets, cum_offsets, node_rows, node_cols, index_j, cumulative, path_j1, path_jl):
    path_len = len(raw_path)
    if mirrored: first_j = raw_path[0, 0]; last_j = raw_path[path_len - 1, 0]
    else: first_j = raw_path[0, 1]; last_j = raw_path[path_len - 1, 1]
    col_len = last_j - first_j + 1
    path_starts[path_idx] = node_cursor; col_offsets[path_idx] = col_cursor; cum_offsets[path_idx] = cum_cursor
    path_j1[path_idx] = first_j; path_jl[path_idx] = last_j + 1
    for t in range(col_len): index_j[col_cursor + np.int64(t)] = 0
    cumulative[cum_cursor] = 0.0; prev_j = first_j
    for k in range(path_len):
        sim_row_idx = raw_path[k, 0]; sim_col_idx = raw_path[k, 1]
        if mirrored: row, col = sim_col_idx, sim_row_idx
        else: row, col = sim_row_idx, sim_col_idx
        node_rows[node_cursor + np.int64(k)] = row; node_cols[node_cursor + np.int64(k)] = col
        cumulative[cum_cursor + np.int64(k + 1)] = cumulative[cum_cursor + np.int64(k)] + sm[sim_row_idx, sim_col_idx]
        if k > 0 and col != prev_j:
            start = col_cursor + np.int64(prev_j - first_j) + 1; end = col_cursor + np.int64(col - first_j) + 1
            for t in range(start, end): index_j[t] = k
            prev_j = col
    return path_idx + 1, node_cursor + path_len, col_cursor + col_len, cum_cursor + path_len + 1

def apply_locomotif(ts, l_min, l_max, rho=None, nb=None, start_mask=None, end_mask=None, overlap=0.0, warping=True):
    lcm = get_locomotif_instance(ts, l_min, l_max, rho=rho, warping=warping)
    lcm.find_best_paths(vwidth=l_min // 2)
    motif_sets = []
    for representative, motif_set, _ in lcm.find_best_motif_sets(nb=nb, overlap=overlap, start_mask=start_mask, end_mask=end_mask):
        motif_sets.append((representative, motif_set))
    return motif_sets

def get_locomotif_instance(ts, l_min, l_max, rho=None, warping=True):
    return LoCoMotif.instance_from_rho(ts, l_min=l_min, l_max=l_max, rho=rho, warping=warping)


class LoCoMotif:

    def __init__(self, ts, l_min, l_max, gamma=None, tau=0.5, delta_a=1.0, delta_m=0.5, warping=True):        
        self.ts = ts; l_min = max(4, l_min)
        self.l_min = np.int32(l_min); self.l_max = np.int32(l_max)
        self._loco = loco.LoCo(ts, gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m, warping=warping)
        self._paths = None; self._path_data = None; self._path_collection = None

    @classmethod
    def instance_from_rho(cls, ts, l_min, l_max, rho=None, warping=True):
        if rho is None: rho = 0.8 if warping else 0.5  
        lcm = cls(ts=ts, l_min=l_min, l_max=l_max)
        lcm._loco = loco.LoCo.instance_from_rho(ts, rho, gamma=None, warping=warping)
        return lcm

    def find_best_paths(self, vwidth=None):
        from .graph_loco import find_best_paths_graph_for_instance
        self._paths = find_best_paths_graph_for_instance(self, vwidth=vwidth)
        return self._paths

    def induced_paths(self, b, e, mask=None):
        if mask is None: mask = np.full(len(self.ts), False)
        if self._path_data is not None:
            induced = _induced_paths_graph(b, e, mask, *self._path_data[:9])
            out = []
            for segment in induced:
                out.append((int(segment[0]), int(segment[1])))
            return out
        return []

    def find_best_motif_sets(self, nb=None, start_mask=None, end_mask=None, overlap=0.0, keep_fitnesses=False):
        if self._paths is None and self._path_data is None: self.find_best_paths()
        n = len(self.ts)
        if start_mask is None: start_mask = np.full(n, True)
        if end_mask is None: end_mask   = np.full(n, True)
        current_nb = 0; mask = np.full(n, False)
        while (nb is None or current_nb < nb):
            if np.all(mask) or not np.any(start_mask) or not np.any(end_mask): break
            start_mask[mask] = False; end_mask[mask]   = False
            if self._path_data is not None:
                best_candidate, best_fitness, fitnesses = _find_best_candidate_graph(n, self.l_min, self.l_max, overlap, mask, mask, start_mask, end_mask, *self._path_data, keep_fitnesses=keep_fitnesses)
            else:
                best_candidate, best_fitness, fitnesses = (np.int32(0), np.int32(0)), 0.0, np.empty((0, 5), dtype=np.float32)
            b, e = best_candidate
            if best_fitness == 0.0: break
            motif_set = self.induced_paths(b, e, mask)
            mask = _mask_motif_set(mask, motif_set, overlap)
            current_nb += 1; yield (b, e), motif_set, fitnesses
            
    @property
    def local_warping_paths(self):
        if self._path_data is not None:
            if self._path_collection is None: self._path_collection = _LazyPathCollection(self._path_data)
            return self._path_collection
        return None
    
    @property
    def self_similarity_matrix(self): return self._loco.similarity_matrix
    
    @property
    def cumulative_similarity_matrix(self): return self._loco.cumulative_similarity_matrix

def _mask_motif_set(mask, motif_set, overlap):
    for (b_m, e_m) in motif_set:
        l = e_m - b_m; l_mask = max(1, int((1 - 2*overlap) * l))
        mask[b_m + (l - l_mask)//2 : b_m + (l - l_mask)//2 + l_mask] = True
    return mask

@njit(cache=True)
def _induced_paths_graph(b, e, mask, path_starts, col_offsets, cum_offsets, node_rows, node_cols, index_j, cumulative, path_j1, path_jl):
    count = 0
    # In main branch, _induced_paths iterates through P in order.
    # Our path_j1 order matches P order (Diagonal, then P1, Mirror P1, P2, Mirror P2...).
    for path_idx in range(len(path_j1)):
        if b < path_j1[path_idx] or path_jl[path_idx] < e: continue
        col_base = col_offsets[path_idx]; node_base = path_starts[path_idx]; pj1 = path_j1[path_idx]
        kb = index_j[col_base + np.int64(b - pj1)]
        ke = index_j[col_base + np.int64(e - 1 - pj1)]
        b_m = node_rows[node_base + np.int64(kb)]; e_m = node_rows[node_base + np.int64(ke)] + 1
        blocked = False
        # Exact mask check from main branch: not np.any(mask[b_m:e_m])
        for pos in range(b_m, e_m):
            if mask[pos]: blocked = True; break
        if not blocked: count += 1
    out = np.empty((count, 2), dtype=np.int32); write_idx = 0
    for path_idx in range(len(path_j1)):
        if b < path_j1[path_idx] or path_jl[path_idx] < e: continue
        col_base = col_offsets[path_idx]; node_base = path_starts[path_idx]; pj1 = path_j1[path_idx]
        kb = index_j[col_base + np.int64(b - pj1)]
        ke = index_j[col_base + np.int64(e - 1 - pj1)]
        b_m = node_rows[node_base + np.int64(kb)]; e_m = node_rows[node_base + np.int64(ke)] + 1
        blocked = False
        for pos in range(b_m, e_m):
            if mask[pos]: blocked = True; break
        if not blocked: out[write_idx, 0] = b_m; out[write_idx, 1] = e_m; write_idx += 1
    return out

@njit(cache=True)
def _find_best_candidate_graph(n, l_min, l_max, nu, row_mask, col_mask, start_mask, end_mask, path_starts, col_offsets, cum_offsets, node_rows, node_cols, index_j, cumulative, path_j1, path_jl, start_order, keep_fitnesses=False):
    fitness_list = TypedList()
    r = len(row_mask); c = len(col_mask)
    max_size = max(2, int(np.ceil(r / (l_min // 2 + 1))))
    active_paths = np.empty(max_size, dtype=np.int32); active_keys = np.empty(max_size, dtype=np.int32)
    active_size = 0; next_start_idx = 0; best_fitness = 0.0; best_candidate = (0, 0)
    Pe = np.empty(max_size, dtype=np.bool_); es_checked = np.empty(max_size, dtype=np.int32)
    for b_repr in range(c - l_min + 1):
        next_j = b_repr; write_idx = 0
        for i in range(active_size):
            if path_jl[active_paths[i]] != next_j: active_paths[write_idx] = active_paths[i]; write_idx += 1
        active_size = write_idx
        for i in range(active_size):
            p_idx = active_paths[i]
            active_keys[i] = node_rows[path_starts[p_idx] + np.int64(index_j[col_offsets[p_idx] + np.int64(next_j - path_j1[p_idx])])]
        while next_start_idx < len(start_order):
            p_idx = start_order[next_start_idx]
            if path_j1[p_idx] != next_j: break
            key = node_rows[path_starts[p_idx] + np.int64(index_j[col_offsets[p_idx]])]
            idx = np.searchsorted(active_keys[:active_size], key)
            active_paths[idx+1:active_size+1] = active_paths[idx:active_size]; active_keys[idx+1:active_size+1] = active_keys[idx:active_size]
            active_paths[idx] = p_idx; active_keys[idx] = key; active_size += 1; next_start_idx += 1
        if active_size < 2 or not start_mask[b_repr] or col_mask[b_repr]: continue
        if np.any(col_mask[b_repr : b_repr + l_min - 1]): continue
        Pe[:active_size] = True; es_checked[:active_size] = active_keys[:active_size]; nb_rem = active_size
        for e_repr in range(b_repr + l_min, min(c + 1, b_repr + l_max + 1)):
            if col_mask[e_repr-1]: break
            if not end_mask[e_repr-1]: continue
            score = 0.0; total_l = 0.0; total_pl = 0.0; total_ov = 0.0; l_prev = 0; e_prev = 0; first = True; too_much_ov = False
            for i in range(active_size):
                if nb_rem < 2: break
                if not Pe[i]: continue
                p_idx = active_paths[i]
                if path_jl[p_idx] < e_repr: Pe[i] = False; nb_rem -= 1; continue
                col_off = col_offsets[p_idx]; node_off = path_starts[p_idx]; cum_off = cum_offsets[p_idx]; pj1 = path_j1[p_idx]
                kb = index_j[col_off + np.int64(b_repr - pj1)]; b = node_rows[node_off + np.int64(kb)]
                ke = index_j[col_off + np.int64(e_repr - 1 - pj1)]; e = node_rows[node_off + np.int64(ke)] + 1
                if np.any(row_mask[es_checked[i] : e]): Pe[i] = False; nb_rem -= 1; continue
                es_checked[i] = e; l = e - b
                if not first:
                    ov = max(0, e_prev - b)
                    if nu * min(l, l_prev) < ov: too_much_ov = True; break
                    total_ov += ov
                total_l += l; total_pl += (ke - kb + 1); score += cumulative[cum_off + np.int64(ke + 1)] - cumulative[cum_off + np.int64(kb)]; l_prev = l; e_prev = e; first = False
            if nb_rem < 2 or too_much_ov: continue
            l_repr = e_repr - b_repr; n_score = (score - l_repr) / total_pl; n_cov = (total_l - total_ov - l_repr) / float(n)
            fit = 0.0
            if n_cov != 0 or n_score != 0: fit = 2 * (n_cov * n_score) / (n_cov + n_score)
            if fit > best_fitness: best_candidate = (b_repr, e_repr); best_fitness = fit
            if keep_fitnesses and fit > 0:
                row = np.array([float32(b_repr), float32(e_repr), float32(fit), float32(n_cov), float32(n_score)], dtype=np.float32)
                fitness_list.append(row)
    if not keep_fitnesses or len(fitness_list) == 0: return best_candidate, best_fitness, np.empty((0, 5), dtype=np.float32)
    res = np.empty((len(fitness_list), 5), dtype=np.float32)
    for i in range(len(fitness_list)): res[i] = fitness_list[i]
    return best_candidate, best_fitness, res
