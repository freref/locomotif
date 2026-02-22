import numpy as np

from . import loco_jit

_TAU_CACHE = {}
_TAU_CACHE_ORDER = []
_TAU_CACHE_MAX_SIZE = 16


def _array_cache_key(arr):
    ai = arr.__array_interface__
    return (int(ai["data"][0]), arr.shape, arr.strides, arr.dtype.str)


def _tau_cache_key(ts, ts2, gamma, rho, symmetric):
    return (_array_cache_key(ts), _array_cache_key(ts2), tuple(np.asarray(gamma, dtype=np.float64)), float(rho), bool(symmetric))


def _input_tau_cache_key(ts, ts2, rho, gamma, equal_weight_dims):
    ts_arr = np.asarray(ts)
    if ts2 is None:
        ts2_arr = ts_arr
        symmetric = True
    else:
        ts2_arr = np.asarray(ts2)
        symmetric = False

    if gamma is None:
        gamma_key = ("none", bool(equal_weight_dims))
    elif np.isscalar(gamma):
        gamma_key = ("scalar", float(gamma))
    else:
        gamma_key = ("vector", tuple(np.asarray(gamma, dtype=np.float64).ravel()))

    return (_array_cache_key(ts_arr), _array_cache_key(ts2_arr), float(rho), bool(symmetric), gamma_key)


def _tau_cache_get(key):
    return _TAU_CACHE.get(key, None)


def _tau_cache_put(key, value):
    if key in _TAU_CACHE:
        _TAU_CACHE[key] = value
        return
    _TAU_CACHE[key] = value
    _TAU_CACHE_ORDER.append(key)
    if len(_TAU_CACHE_ORDER) > _TAU_CACHE_MAX_SIZE:
        old = _TAU_CACHE_ORDER.pop(0)
        if old in _TAU_CACHE:
            del _TAU_CACHE[old]


class LoCo:

    def __init__(
        self,
        ts,
        gamma=None,
        tau=0.5,
        delta_a=1.0,
        delta_m=0.5,
        warping=True,
        ts2=None,
        equal_weight_dims=False,
    ):
        self._symmetric = False
        if ts2 is None:
            self._symmetric = True
            ts2 = ts

        self.ts = ensure_multivariate(np.array(ts, dtype=np.float32))
        self.ts2 = ensure_multivariate(np.array(ts2, dtype=np.float32))
        assert self.ts.shape[1] == self.ts2.shape[1], "Input time series must have the same number of dimensions."

        self.gamma = handle_gamma(self.ts, gamma, self._symmetric, equal_weight_dims)

        self.warping = warping
        self.tau = tau
        self.delta_a = delta_a
        self.delta_m = delta_m
        self.l_min = None
        self._sm = None
        self._csm = None
        self._dist = None
        self._bp = None
        self._paths = None

    @property
    def similarity_matrix(self):
        return self._sm

    @property
    def cumulative_similarity_matrix(self):
        if self._csm is None:
            return None
        return self._csm[2:, 2:]

    @property
    def local_warping_paths(self):
        return self._paths

    def calculate_similarity_matrix(self):
        self._sm = similarity_matrix_ndim(self.ts, self.ts2, gamma=self.gamma, only_triu=self._symmetric, diag_offset=0)
        return self._sm

    def calculate_cumulative_similarity_matrix(self):
        if self._sm is None:
            self.calculate_similarity_matrix()
        self._csm, self._dist, self._bp = cumulative_similarity_matrix(
            self._sm,
            self.l_min,
            tau=self.tau,
            delta_a=self.delta_a,
            delta_m=self.delta_m,
            warping=self.warping,
            only_triu=self._symmetric,
            diag_offset=0,
            with_bp=self.warping,
        )
        return self._csm

    def find_best_paths(self, l_min=None, vwidth=None):
        if l_min is None:
            l_min = min(len(self.ts), len(self.ts2)) // 10
        self.l_min = l_min

        if vwidth is None:
            vwidth = l_min // 2

        if self._csm is None:
            self.calculate_cumulative_similarity_matrix()

        mask = np.full(self._csm.shape, self._symmetric)
        if self._symmetric:
            n_rows, n_cols = mask.shape
            for i in range(n_rows):
                j_start = i + vwidth + 1
                if j_start < n_cols:
                    mask[i, j_start:] = False

        paths = find_best_paths(
            self._csm,
            self._dist,
            mask,
            self.tau,
            l_min=l_min,
            vwidth=vwidth,
            warping=self.warping,
            bp=self._bp,
        )
        paths = [path - 2 for path in paths]

        if self._symmetric:
            diagonal = np.tile(np.arange(len(self.ts), dtype=np.int32), (2, 1)).T
            paths.insert(0, diagonal)

        self._paths = paths
        return self._paths

    def path_similarities(self, path):
        i = path[:, 0]
        j = path[:, 1]
        if self._sm is not None:
            return self._sm[i, j]
        diff = self.ts[i] - self.ts2[j]
        d2 = np.sum(self.gamma[np.newaxis, :] * np.power(diff, 2), axis=1)
        return np.exp(-d2).astype(np.float32)

    @classmethod
    def instance_from_rho(
        cls,
        ts,
        rho,
        gamma=None,
        warping=True,
        ts2=None,
        equal_weight_dims=False,
    ):
        input_key = _input_tau_cache_key(ts, ts2, rho, gamma, equal_weight_dims)
        tau = _tau_cache_get(input_key)

        loco = cls(
            ts,
            gamma=gamma,
            warping=warping,
            ts2=ts2,
            equal_weight_dims=equal_weight_dims,
        )
        if tau is None:
            sm = loco.calculate_similarity_matrix()
            tau = estimate_tau_from_sm(sm, rho, only_triu=loco._symmetric)
            _tau_cache_put(input_key, tau)
            key = _tau_cache_key(loco.ts, loco.ts2, loco.gamma, rho, loco._symmetric)
            _tau_cache_put(key, tau)
        loco.tau = tau
        loco.delta_a = 2 * tau
        loco.delta_m = 0.5
        return loco


def estimate_tau_from_sm(sm, rho, only_triu=False):
    if only_triu:
        if sm.size == 1:
            return sm[0, 0]
        n = sm.shape[0]
        valid_count = n * (n + 1) // 2
        invalid_count = sm.size - valid_count
        adjusted_rho = (invalid_count + rho * (valid_count - 1)) / (sm.size - 1)
        tau = np.quantile(sm, adjusted_rho, axis=None)
    else:
        tau = np.quantile(sm, rho, axis=None)
    return tau


def similarity_matrix_ndim(ts1, ts2, gamma=None, only_triu=False, diag_offset=0):
    return loco_jit.similarity_matrix_ndim(ts1, ts2, gamma, only_triu, diag_offset)


def cumulative_similarity_matrix(
    sm,
    l_min=10,
    tau=0.5,
    delta_a=1.0,
    delta_m=0.5,
    warping=True,
    only_triu=False,
    diag_offset=0,
    with_bp=False,
):
    if warping:
        if with_bp:
            return loco_jit.cumulative_similarity_matrix_warping_bp(sm, l_min, tau, delta_a, delta_m, only_triu, diag_offset)
        csm, dist = loco_jit.cumulative_similarity_matrix_warping(sm, l_min, tau, delta_a, delta_m, only_triu, diag_offset)
        return csm, dist, None

    if with_bp:
        return loco_jit.cumulative_similarity_matrix_no_warping_bp(sm, tau, delta_a, delta_m, only_triu, diag_offset)
    csm = loco_jit.cumulative_similarity_matrix_no_warping(sm, tau, delta_a, delta_m, only_triu, diag_offset)
    dist = np.zeros(csm.shape, dtype=np.int32)
    return csm, dist, None


def find_best_paths(csm, dist, mask, tau, l_min=10, vwidth=5, warping=True, bp=None):
    if warping and bp is not None:
        return loco_jit.find_best_paths_row_frontier_exact(csm, dist, bp, mask, tau, l_min, vwidth, warping)
    return loco_jit.find_best_paths(csm, dist, mask, tau, l_min, vwidth, warping)


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
    _, d = ts.shape
    if gamma is None:
        if symmetric:
            if d == 1 or not equal_weight_dims:
                gamma = d * [1 / np.std(ts, axis=None) ** 2]
            else:
                gamma = [1 / np.std(ts[:, dim]) ** 2 for dim in range(d)]
        else:
            gamma = d * [1.0]
    elif np.isscalar(gamma):
        gamma = d * [gamma]
    else:
        assert np.ndim(gamma) == 1 and len(gamma) == d
    gamma = np.array(gamma, dtype=np.float64)
    return gamma
