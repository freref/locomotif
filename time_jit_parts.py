import sys
import locomotif.loco_jit
import numba

original_mask = locomotif.loco_jit._mask_vicinity

@numba.njit(cache=True)
def dummy_mask(path, mask, vwidth):
    pass

locomotif.loco_jit._mask_vicinity = dummy_mask
import locomotif.graph_loco
sys.path.append("../locomotif-profiling")
import locomotif_profiling_ecg
locomotif_profiling_ecg.main()
