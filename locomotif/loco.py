import numpy as np

from . import loco_jit

class LoCo:

    def __init__(self, ts, gamma=None, tau=0.5, delta_a=1.0, delta_m=0.5, warping=True, ts2=None, equal_weight_dims=False):
        # If ts2 is specified, we assume it is different from ts. Alternative is self._symmetric = np.array_equal(self.ts, self.ts2) 
        self._symmetric = False
        if ts2 is None:
            self._symmetric = True
            ts2 = ts

        self.ts  = ensure_multivariate(np.array(ts, dtype=np.float32))
        self.ts2 = ensure_multivariate(np.array(ts2, dtype=np.float32))
        assert self.ts.shape[1] == self.ts2.shape[1], "Input time series must have the same number of dimensions."

        # Handle the gamma argument.
        self.gamma = handle_gamma(self.ts, gamma, self._symmetric, equal_weight_dims)
                     
        # LoCo args
        self.warping = warping
        self.tau = np.float32(tau)
        self.delta_a = np.float32(delta_a)
        self.delta_m = np.float32(delta_m)
        # Self-similarity matrix
        self._sm = None
        # Cumulative similiarity matrix
        self._csm = None
        self._bp_dir = None
        self._candidate_linear_pos = None
        self._candidate_values = None
        self._last_csm_diag_offset = np.int32(0)
        # Local warping paths
        self._paths = None


    @property
    def similarity_matrix(self):
        return self._sm

    @property
    def cumulative_similarity_matrix(self):
        if self._csm is None and self._sm is not None and self._bp_dir is not None:
            self.calculate_cumulative_similarity_matrix(diag_offset=self._last_csm_diag_offset)
        if self._csm is None:
            return None
        return self._csm[2:, 2:]
    
    @property
    def local_warping_paths(self):
        return self._paths
     
    def calculate_similarity_matrix(self):
        self._sm  = similarity_matrix_ndim(self.ts, self.ts2, gamma=self.gamma, only_triu=self._symmetric, diag_offset=0)
        return self._sm
          
    def calculate_cumulative_similarity_matrix(self, diag_offset=0, candidate_threshold=None, tile_size=0, diag_gap=0):
        if self._sm is None:
            self.calculate_similarity_matrix()
        self._last_csm_diag_offset = np.int32(diag_offset)
        threshold = np.float32(0.0 if candidate_threshold is None else candidate_threshold)
        tile_size = np.int32(tile_size)
        diag_gap = np.int32(diag_gap)
        diag_offset = np.int32(diag_offset)
        if self.warping:
            self._csm, self._bp_dir, self._candidate_linear_pos = loco_jit.cumulative_similarity_matrix_warping_with_bp(
                self._sm, self.tau, self.delta_a, self.delta_m, self._symmetric, diag_offset, tile_size, threshold, diag_gap
            )
        else:
            self._csm, self._bp_dir, self._candidate_linear_pos = loco_jit.cumulative_similarity_matrix_no_warping_with_bp(
                self._sm, self.tau, self.delta_a, self.delta_m, self._symmetric, diag_offset, tile_size, threshold, diag_gap
            )
        self._candidate_values = None
        return self._csm

    def prepare_cumulative_similarity_for_path_search(self, diag_offset=0, candidate_threshold=None, tile_size=0, diag_gap=0):
        if self._sm is None:
            self.calculate_similarity_matrix()
        self._last_csm_diag_offset = np.int32(diag_offset)
        threshold = np.float32(0.0 if candidate_threshold is None else candidate_threshold)
        tile_size = np.int32(tile_size)
        diag_gap = np.int32(diag_gap)
        diag_offset = np.int32(diag_offset)
        self._csm = None
        if self.warping:
            self._bp_dir, self._candidate_linear_pos, self._candidate_values = loco_jit.cumulative_similarity_matrix_warping_with_bp_compact(
                self._sm, self.tau, self.delta_a, self.delta_m, self._symmetric, diag_offset, tile_size, threshold, diag_gap
            )
        else:
            self._bp_dir, self._candidate_linear_pos, self._candidate_values = loco_jit.cumulative_similarity_matrix_no_warping_with_bp_compact(
                self._sm, self.tau, self.delta_a, self.delta_m, self._symmetric, diag_offset, tile_size, threshold, diag_gap
            )
        return self._bp_dir

    def find_best_paths(self, l_min=None, vwidth=None):
        if l_min is None:
            l_min = min(len(self.ts), len(self.ts2)) // 10
        if vwidth is None:
            vwidth = l_min // 2
        l_min = np.int32(l_min)
        vwidth = np.int32(vwidth)

        if self._csm is None:
            diag_offset = np.int32(0)
            if self._symmetric:
                diag_offset = np.int32(-max(1, (int(vwidth) + 1) // 2))
            min_path_length = l_min if not self.warping else np.int32(max(1, (int(l_min) + 1) // 2))
            self.prepare_cumulative_similarity_for_path_search(
                diag_offset=diag_offset,
                candidate_threshold=self.tau * min_path_length,
                tile_size=vwidth,
                diag_gap=np.int32(vwidth + 1) if self._symmetric else np.int32(0),
            )

        if self._symmetric:
            shape = self._csm.shape if self._csm is not None else self._bp_dir.shape
            if self._csm is None and self.warping:
                mask = np.empty((0, 0), dtype=np.bool_)
            else:
                mask = loco_jit.symmetric_path_mask(shape[0], shape[1], vwidth)
        else:
            shape = self._csm.shape if self._csm is not None else self._bp_dir.shape
            mask = np.zeros(shape, dtype=np.bool_)

        if self._csm is not None:
            paths = loco_jit.find_best_paths_with_bp(
                self._csm, mask, self.tau, l_min, vwidth, self.warping, self._bp_dir, self._symmetric, self._candidate_linear_pos
            )
        else:
            paths = loco_jit.find_best_paths_with_bp_compact(
                mask, self.tau, l_min, vwidth, self.warping, self._bp_dir, self._candidate_linear_pos, self._candidate_values, self._symmetric, np.int32(vwidth + 1)
            )
        paths = [path-2 for path in paths]

        if self._symmetric:
            # Prepend the diagonal to the result set.
            diagonal = np.tile(np.arange(len(self.ts), dtype=np.int32), (2, 1)).T
            paths.insert(0, diagonal)

        self._paths = paths
        return self._paths

    @classmethod
    def instance_from_rho(cls, ts, rho, gamma=None, warping=True, ts2=None, equal_weight_dims=False):
        # Make LoCo instance
        loco = cls(ts, gamma=gamma, warping=warping, ts2=ts2, equal_weight_dims=equal_weight_dims)
        # Get the SM, determine tau and delta's
        sm = loco.calculate_similarity_matrix()
        tau = estimate_tau_from_sm(sm, rho, only_triu=loco._symmetric)
        loco.tau = np.float32(tau)
        loco.delta_a = np.float32(2.0) * loco.tau
        loco.delta_m = np.float32(0.5)
        return loco
    
# Calculate the similarity threshold tau as the rho-quantile of the similarity matrix.
def estimate_tau_from_sm(sm, rho, only_triu=False):
    return loco_jit.exact_tau_from_sm(sm, rho, only_triu)

def similarity_matrix_ndim(ts1, ts2, gamma=None, only_triu=False, diag_offset=0):
    return loco_jit.similarity_matrix_ndim(ts1, ts2, gamma, only_triu, diag_offset)

def cumulative_similarity_matrix(sm, tau=0.5, delta_a=1.0, delta_m=0.5, warping=True, only_triu=False, diag_offset=0):
    if warping:
        return loco_jit.cumulative_similarity_matrix_warping(sm, tau, delta_a, delta_m, only_triu, diag_offset)
    else:
        return loco_jit.cumulative_similarity_matrix_no_warping(sm, tau, delta_a, delta_m, only_triu, diag_offset)

def find_best_paths(csm, mask, tau, l_min=10, vwidth=5, warping=True):
    paths = loco_jit.find_best_paths(csm, mask, tau, l_min, vwidth, warping)
    return paths

def ensure_multivariate(ts):
    ts = np.asarray(ts)
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
    elif ts.ndim == 2 and ts.shape[1] == 1:
        pass
    elif ts.ndim != 2:
        raise ValueError(f"Time series must be 1D or 2D, got shape {ts.shape}")
    return ts

def handle_gamma(ts, gamma, symmetric, equal_weight_dims):
    _, D = ts.shape
    # If no value is specified, determine the gamma value(s) based on the input TS.
    if gamma is None:
        if symmetric:
            if D == 1 or not equal_weight_dims:
                gamma = D * [1 / np.std(ts, axis=None)**2]
            else:
                gamma = [1 / np.std(ts[:, d])**2 for d in range(D)]
        else:
            gamma = D * [1.0]
    # If a single value is specified for gamma, that value is used for every dimension. 
    elif np.isscalar(gamma):
        gamma = D * [gamma]
    # Else, len(gamma) should be equal to the number of dimensions
    else:
        assert np.ndim(gamma) == 1 and len(gamma) == D
    gamma = np.array(gamma, dtype=np.float32)
    return gamma
