import numpy as np

from .locomotif import _LazyPathCollection, _build_compact_path_graph, get_locomotif_instance
from . import loco_jit


def find_best_paths_graph_for_instance(lcm, vwidth=None):
    if vwidth is None:
        vwidth = lcm.l_min // 2
    vwidth = np.maximum(10, vwidth)

    loco_obj = lcm._loco
    if loco_obj._csm is None:
        loco_obj.calculate_cumulative_similarity_matrix()

    mask = np.full(loco_obj._csm.shape, loco_obj._symmetric)
    if loco_obj._symmetric:
        for row_idx in range(mask.shape[0]):
            col_start = row_idx + vwidth + 1
            if col_start < mask.shape[1]:
                mask[row_idx, col_start:] = False

    raw_paths = loco_jit.find_best_paths(
        loco_obj._csm,
        mask,
        loco_obj.tau,
        np.int32(lcm.l_min),
        np.int32(vwidth),
        loco_obj.warping,
        loco_obj._bp_dir,
    )

    lcm._path_data = _build_compact_path_graph(raw_paths, lcm.self_similarity_matrix, loco_obj._symmetric)
    lcm._path_collection = _LazyPathCollection(lcm._path_data)
    lcm._paths = None
    return lcm._path_collection


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
