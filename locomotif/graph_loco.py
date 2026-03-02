import numpy as np
from . import locomotif
from . import loco_jit

def find_best_paths_graph_for_instance(lcm, vwidth=None):
    if vwidth is None:
        vwidth = np.maximum(10, lcm.l_min // 2)
    
    # 1. Similarity Matrix (Exact same logic as main/loco.py)
    sm = lcm._loco.calculate_similarity_matrix()
    lcm._loco.calculate_cumulative_similarity_matrix() # ensures tau is estimated
    
    # 2. Path Finding (Exact same logic as main/loco.py)
    # LoCo.find_best_paths implementation in main:
    # mask = np.full(self._csm.shape, self._symmetric)
    # if self._symmetric:
    #     mask[np.triu_indices(len(mask), k=vwidth+1)] = False
    
    n, m = lcm._loco.cumulative_similarity_matrix.shape
    mask = np.full((n, m), True, dtype=bool)
    # LoCo uses csm.shape, which is n+2 x m+2.
    # main's mask is for the CSM.
    # np.triu_indices(len(mask), k=vwidth+1) makes it False for j >= i + vwidth + 1.
    mask[np.triu_indices(n, k=vwidth+1)] = False
    
    # find_best_paths in loco_jit (optimized)
    raw_paths = loco_jit.find_best_paths(
        lcm._loco.cumulative_similarity_matrix,
        mask,
        lcm._loco.tau,
        l_min=lcm.l_min,
        vwidth=vwidth,
        warping=lcm._loco.warping
    )
    
    # 3. Graph Building
    # LoCo.find_best_paths returns paths-2.
    # We do this inside _build_compact_path_graph or here.
    # Main subtracts 2 from each path.
    raw_paths_minus_2 = []
    for p in raw_paths:
        raw_paths_minus_2.append(p - 2)
    
    lcm._path_data = locomotif._build_compact_path_graph(
        raw_paths_minus_2, 
        lcm._loco.similarity_matrix, 
        lcm._loco._symmetric
    )
    # LoCoMotif.find_best_paths returns self._paths (materialized)
    lcm._paths = lcm.local_warping_paths
    return lcm._paths
