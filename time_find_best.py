import time
import numpy as np

def hook_find_best_paths_graph_for_instance(lcm, vwidth=None):
    if vwidth is None:
        vwidth = lcm.l_min // 2
    vwidth = np.maximum(10, vwidth)

    loco_obj = lcm._loco
    if loco_obj._csm is None:
        t0 = time.time()
        loco_obj.calculate_cumulative_similarity_matrix()
        print("calculate_csm:", time.time() - t0)

    t0 = time.time()
    if loco_obj._symmetric:
        mask = np.tril(np.ones(loco_obj._csm.shape, dtype=bool), k=vwidth)
    else:
        mask = np.zeros(loco_obj._csm.shape, dtype=bool)
    print("mask creation:", time.time() - t0)

    from locomotif import loco_jit
    t0 = time.time()
    raw_paths = loco_jit.find_best_paths(
        loco_obj._csm,
        mask,
        loco_obj.tau,
        np.int32(lcm.l_min),
        np.int32(vwidth),
        loco_obj.warping,
        loco_obj._bp_dir,
    )
    print("loco_jit.find_best_paths:", time.time() - t0)

    from locomotif.locomotif import _build_compact_path_graph, _LazyPathCollection
    t0 = time.time()
    lcm._path_data = _build_compact_path_graph(raw_paths, lcm.self_similarity_matrix, loco_obj._symmetric)
    print("_build_compact_path_graph:", time.time() - t0)

    lcm._path_collection = _LazyPathCollection(lcm._path_data)
    lcm._paths = None
    return lcm._path_collection

import locomotif.graph_loco
locomotif.graph_loco.find_best_paths_graph_for_instance = hook_find_best_paths_graph_for_instance
import sys
sys.path.append("../locomotif-profiling")
import locomotif_profiling_ecg
locomotif_profiling_ecg.main()
