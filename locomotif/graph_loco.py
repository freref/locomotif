import numpy as np

from numba import int32, njit
from numba.typed import List as TypedList

from .locomotif import get_locomotif_instance, _materialize_paths_with_mirror
from .loco_jit import mask_vicinity, best_path_from_backpointers


@njit(cache=True)
def _source_reachable_mask(csm, src_id, mask, l_min):
    n, m = csm.shape
    out = np.zeros((n, m), dtype=np.bool_)
    for i in range(2, n):
        for j in range(2, m):
            if mask[i, j] or csm[i, j] <= 0.0:
                continue
            source = src_id[i, j]
            if source < 0:
                continue
            source_i = source // m
            source_j = source - source_i * m
            if (i - source_i + 1) >= l_min or (j - source_j + 1) >= l_min:
                out[i, j] = True
    return out


@njit(cache=True)
def _extract_paths_graph(csm, bp_dir, src_id, mask, l_min, vwidth):
    mask = mask | (csm <= 0.0)
    paths = TypedList.empty_list(int32[:, :])
    start_mask = _source_reachable_mask(csm, src_id, mask, l_min)

    pos_i, pos_j = np.nonzero(start_mask)
    if len(pos_i) == 0:
        return paths

    values = np.array([csm[pos_i[k], pos_j[k]] for k in range(len(pos_i))], dtype=np.float32)
    perm = np.argsort(values)
    sorted_pos_i = pos_i[perm]
    sorted_pos_j = pos_j[perm]

    k_best = len(sorted_pos_i) - 1
    while k_best >= 0:
        path = np.empty((0, 0), dtype=np.int32)
        path_found = False

        while not path_found:
            while mask[sorted_pos_i[k_best], sorted_pos_j[k_best]]:
                k_best -= 1
                if k_best < 0:
                    return paths

            i_best = sorted_pos_i[k_best]
            j_best = sorted_pos_j[k_best]
            if i_best < 2 or j_best < 2:
                return paths

            path = best_path_from_backpointers(mask, bp_dir, i_best, j_best)
            mask = mask_vicinity(path, mask, 0)

            if (path[-1][0] - path[0][0] + 1) >= l_min or (path[-1][1] - path[0][1] + 1) >= l_min:
                path_found = True

        mask = mask_vicinity(path, mask, vwidth)
        paths.append(path)
    return paths


def find_best_paths_graph_for_instance(lcm, vwidth=None):
    if vwidth is None:
        vwidth = lcm.l_min // 2
    vwidth = np.maximum(10, vwidth)

    loco_obj = lcm._loco
    if loco_obj._csm is None:
        loco_obj.calculate_cumulative_similarity_matrix()

    mask = np.full(loco_obj._csm.shape, loco_obj._symmetric)
    if loco_obj._symmetric:
        mask[np.triu_indices(len(mask), k=vwidth + 1)] = False

    raw_paths = _extract_paths_graph(
        loco_obj._csm,
        loco_obj._bp_dir,
        loco_obj._src_id,
        mask,
        np.int32(lcm.l_min),
        np.int32(vwidth),
    )

    paths = [np.ascontiguousarray(path - 2, dtype=np.int32) for path in raw_paths]
    if loco_obj._symmetric:
        diagonal = np.ascontiguousarray(np.tile(np.arange(len(loco_obj.ts), dtype=np.int32), (2, 1)).T, dtype=np.int32)
        paths.insert(0, diagonal)

    lcm._paths = _materialize_paths_with_mirror(paths, lcm.self_similarity_matrix)
    return lcm._paths


def apply_locomotif_graph(
    ts,
    l_min,
    l_max,
    rho=None,
    nb=None,
    start_mask=None,
    end_mask=None,
    overlap=0.0,
    warping=True,
):
    lcm = get_locomotif_instance(ts, l_min, l_max, rho=rho, warping=warping)
    find_best_paths_graph_for_instance(lcm, vwidth=l_min // 2)
    motif_sets = []
    for representative, motif_set, _ in lcm.find_best_motif_sets(
        nb=nb, overlap=overlap, start_mask=start_mask, end_mask=end_mask
    ):
        motif_sets.append((representative, motif_set))
    return motif_sets
