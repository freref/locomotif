import numpy as np
from . import loco_jit

def ensure_multivariate(ts):
    ts = np.asarray(ts)
    if ts.ndim == 1: ts = ts[:, np.newaxis]
    elif ts.ndim == 2 and ts.shape[1] == 1: pass
    elif ts.ndim != 2: raise ValueError(f"Time series must be 1D or 2D, got shape {ts.shape}")
    return ts

def handle_gamma(ts, gamma, symmetric, equal_weight_dims):
    _, D = ts.shape
    if gamma is None:
        if symmetric:
            if D == 1 or not equal_weight_dims:
                gamma = D * [1 / np.std(ts, axis=None)**2]
            else:
                gamma = [1 / np.std(ts[:, d])**2 for d in range(D)]
        else:
            gamma = D * [1.0]
    elif np.isscalar(gamma): gamma = D * [gamma]
    else: assert np.ndim(gamma) == 1 and len(gamma) == D
    return np.array(gamma, dtype=np.float64)

def estimate_tau_from_sm(sm, rho, only_triu=False):
    if only_triu:
        # Use np.quantile on the upper triangle including diagonal
        n = sm.shape[0]
        iu = np.triu_indices(n)
        tau = np.quantile(sm[iu], rho, axis=None)
    else:
        tau = np.quantile(sm, rho, axis=None)
    return tau

class LoCo:
    def __init__(self, ts, gamma=None, tau=0.5, delta_a=1.0, delta_m=0.5, warping=True, ts2=None, equal_weight_dims=False):
        self.ts = ensure_multivariate(np.array(ts, dtype=np.float32))
        self._symmetric = (ts2 is None)
        self.ts2 = self.ts if self._symmetric else ensure_multivariate(np.array(ts2, dtype=np.float32))
        self.gamma = handle_gamma(self.ts, gamma, self._symmetric, equal_weight_dims)
        self.tau = tau; self.delta_a = delta_a; self.delta_m = delta_m; self.warping = warping
        self._sm = None; self._csm = None; self._paths = None

    @classmethod
    def instance_from_rho(cls, ts, rho, gamma=None, warping=True, ts2=None, equal_weight_dims=False):
        loco = cls(ts, gamma=gamma, warping=warping, ts2=ts2, equal_weight_dims=equal_weight_dims)
        sm = loco.calculate_similarity_matrix()
        tau = estimate_tau_from_sm(sm, rho, only_triu=loco._symmetric)
        loco.tau = tau; loco.delta_a = 2 * tau; loco.delta_m = 0.5
        return loco

    def calculate_similarity_matrix(self):
        if self._sm is None:
            self._sm = loco_jit.similarity_matrix_ndim(self.ts, self.ts2, gamma=self.gamma, only_triu=self._symmetric, diag_offset=0)
        return self._sm

    def calculate_cumulative_similarity_matrix(self):
        if self._csm is None:
            sm = self.calculate_similarity_matrix()
            mins1, maxs1 = loco_jit.calculate_bounding_boxes(self.ts, 64)
            mins2, maxs2 = (mins1, maxs1) if self._symmetric else loco_jit.calculate_bounding_boxes(self.ts2, 64)
            func = loco_jit.cumulative_similarity_matrix_warping if self.warping else loco_jit.cumulative_similarity_matrix_no_warping
            self._csm = func(sm, tau=self.tau, delta_a=self.delta_a, delta_m=self.delta_m, only_triu=self._symmetric, diag_offset=0, mins1=mins1, maxs1=maxs1, mins2=mins2, maxs2=maxs2, gamma=self.gamma)
        return self._csm

    @property
    def similarity_matrix(self): return self.calculate_similarity_matrix()
    
    @property
    def cumulative_similarity_matrix(self): return self.calculate_cumulative_similarity_matrix()

    def find_best_paths(self, l_min=None, vwidth=None):
        if l_min is None: l_min = min(len(self.ts), len(self.ts2)) // 10
        if vwidth is None: vwidth = l_min // 2
        csm = self.calculate_cumulative_similarity_matrix()
        mask = np.full(csm.shape, self._symmetric, dtype=bool)
        if self._symmetric:
            n = len(mask)
            mask[np.triu_indices(n, k=vwidth+1)] = False
        paths = loco_jit.find_best_paths(csm, mask, self.tau, l_min=l_min, vwidth=vwidth, warping=self.warping)
        self._paths = [p - 2 for p in paths]
        if self._symmetric:
            diagonal = np.tile(np.arange(len(self.ts), dtype=np.int32), (2, 1)).T
            self._paths.insert(0, diagonal)
        return self._paths
