import numpy as np

from .locomotif import get_locomotif_instance, _materialize_paths_with_mirror
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
        mask[np.triu_indices(len(mask), k=vwidth + 1)] = False

    raw_paths = loco_jit.find_best_paths(
        loco_obj._csm,
        mask,
        loco_obj.tau,
        np.int32(lcm.l_min),
        np.int32(vwidth),
        loco_obj.warping,
        loco_obj._bp_dir,
        loco_obj._src_id,
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
