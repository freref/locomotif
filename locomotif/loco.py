import numpy as np

from . import loco_jit

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

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
        sparse_events=False,
        sparse_event_tau=None,
        sparse_max_gap=8,
        sparse_row_topk=64,
        backend="auto",
        event_density_fallback=0.20,
        event_index="auto",
        event_probe_rows=128,
        block_tile_size=16,
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
        self.sparse_events = sparse_events
        self.sparse_event_tau = sparse_event_tau
        self.sparse_max_gap = sparse_max_gap
        self.sparse_row_topk = sparse_row_topk
        self.backend = backend
        self.event_density_fallback = float(event_density_fallback)
        self.event_index = event_index
        self.event_probe_rows = int(event_probe_rows)
        self.block_tile_size = int(block_tile_size)
        self._resolved_backend = None
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

    def _choose_backend(self):
        if not self.warping:
            return "dense_legacy"
        if self.backend != "auto":
            return self.backend
        return "nosort_frontier_exact"

    def calculate_cumulative_similarity_matrix(self):
        backend = self._choose_backend()
        if backend == "event_exact" and self.tau <= 0.0:
            backend = "dense_block_exact"
        self._resolved_backend = backend

        if backend == "event_exact":
            row_ptr, col_idx, sim_vals = build_event_csr(
                self.ts,
                self.ts2,
                self.gamma,
                self.tau,
                only_triu=self._symmetric,
                diag_offset=0,
                event_index=self.event_index,
            )
            self._csm, self._dist, self._bp = loco_jit.cumulative_similarity_matrix_events_bp(
                np.int32(len(self.ts)),
                np.int32(len(self.ts2)),
                row_ptr,
                col_idx,
                sim_vals,
                self.tau,
                self.delta_a,
                self.delta_m,
                self._symmetric,
                0,
                self.warping,
            )
            return self._csm

        if self._sm is None:
            self.calculate_similarity_matrix()
        with_bp = backend != "dense_legacy" and self.warping
        self._csm, self._dist, self._bp = cumulative_similarity_matrix(
            self._sm,
            self.l_min,
            tau=self.tau,
            delta_a=self.delta_a,
            delta_m=self.delta_m,
            warping=self.warping,
            only_triu=self._symmetric,
            diag_offset=0,
            sparse_events=self.sparse_events,
            sparse_event_tau=self.sparse_event_tau,
            sparse_max_gap=self.sparse_max_gap,
            sparse_row_topk=self.sparse_row_topk,
            with_bp=with_bp,
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

        backend = self._resolved_backend if self._resolved_backend is not None else self._choose_backend()

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
            backend=backend,
            bp=self._bp,
            block_tile_size=self.block_tile_size,
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
        sparse_events=False,
        sparse_event_tau=None,
        sparse_max_gap=8,
        sparse_row_topk=64,
        backend="auto",
        event_density_fallback=0.20,
        event_index="auto",
        event_probe_rows=128,
        block_tile_size=16,
    ):
        input_key = _input_tau_cache_key(ts, ts2, rho, gamma, equal_weight_dims)
        tau = _tau_cache_get(input_key)

        loco = cls(
            ts,
            gamma=gamma,
            warping=warping,
            ts2=ts2,
            equal_weight_dims=equal_weight_dims,
            sparse_events=sparse_events,
            sparse_event_tau=sparse_event_tau,
            sparse_max_gap=sparse_max_gap,
            sparse_row_topk=sparse_row_topk,
            backend=backend,
            event_density_fallback=event_density_fallback,
            event_index=event_index,
            event_probe_rows=event_probe_rows,
            block_tile_size=block_tile_size,
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
    sparse_events=False,
    sparse_event_tau=None,
    sparse_max_gap=8,
    sparse_row_topk=64,
    with_bp=False,
):
    if warping:
        if sparse_events and not with_bp:
            event_tau = tau if sparse_event_tau is None else sparse_event_tau
            csm, dist = loco_jit.cumulative_similarity_matrix_warping_sparse(
                sm, l_min, tau, delta_a, delta_m, only_triu, diag_offset, event_tau, sparse_max_gap, sparse_row_topk
            )
            return csm, dist, None
        if with_bp:
            return loco_jit.cumulative_similarity_matrix_warping_bp(sm, l_min, tau, delta_a, delta_m, only_triu, diag_offset)
        csm, dist = loco_jit.cumulative_similarity_matrix_warping(sm, l_min, tau, delta_a, delta_m, only_triu, diag_offset)
        return csm, dist, None

    if with_bp:
        return loco_jit.cumulative_similarity_matrix_no_warping_bp(sm, tau, delta_a, delta_m, only_triu, diag_offset)
    csm = loco_jit.cumulative_similarity_matrix_no_warping(sm, tau, delta_a, delta_m, only_triu, diag_offset)
    dist = np.zeros(csm.shape, dtype=np.int32)
    return csm, dist, None


def find_best_paths(csm, dist, mask, tau, l_min=10, vwidth=5, warping=True, backend="dense_legacy", bp=None, block_tile_size=16):
    if backend == "dense_legacy" or bp is None or not warping:
        return loco_jit.find_best_paths(csm, dist, mask, tau, l_min, vwidth, warping)
    if backend == "dense_block_exact":
        return loco_jit.find_best_paths_block_exact(csm, dist, bp, mask, tau, l_min, vwidth, warping, np.int32(block_tile_size))
    if backend == "nosort_frontier_exact" or backend == "dense_frontier_exact":
        return loco_jit.find_best_paths_row_frontier_exact(csm, dist, bp, mask, tau, l_min, vwidth, warping)
    return loco_jit.find_best_paths_bp_sorted(csm, dist, bp, mask, tau, l_min, vwidth, warping)


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


def _scaled_ts(ts, gamma):
    scale = np.sqrt(gamma, dtype=np.float64).astype(np.float32)
    return ts * scale[np.newaxis, :]


def estimate_event_density(ts1, ts2, gamma, tau, only_triu=False, diag_offset=0, probe_rows=1024, event_index="auto"):
    if cKDTree is None:
        return 1.0
    n, m = len(ts1), len(ts2)
    if n == 0 or m == 0:
        return 0.0
    if tau <= 0.0:
        return 1.0

    probe = min(int(probe_rows), n)
    if probe <= 0:
        return 1.0

    x1 = _scaled_ts(ts1, gamma)
    x2 = _scaled_ts(ts2, gamma)
    radius = np.sqrt(max(0.0, -np.log(max(tau, 1e-12)))) + 1e-12
    tree = cKDTree(x2)
    rows = np.linspace(0, n - 1, probe, dtype=np.int32)

    total = 0
    events = 0
    for i in rows:
        j_start = max(0, int(i) - diag_offset) if only_triu else 0
        if j_start >= m:
            continue
        total += m - j_start
        neigh = tree.query_ball_point(x1[int(i)], r=radius)
        if not only_triu:
            events += len(neigh)
        else:
            neigh_arr = np.asarray(neigh, dtype=np.int32)
            events += int(np.sum(neigh_arr >= j_start))

    if total <= 0:
        return 0.0
    return float(events) / float(total)


def build_event_csr(ts1, ts2, gamma, tau, only_triu=False, diag_offset=0, event_index="auto"):
    n, m = len(ts1), len(ts2)
    if n == 0 or m == 0:
        return np.zeros(n + 1, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    if cKDTree is None:
        raise ImportError("event_exact backend requires scipy (cKDTree).")
    if tau <= 0.0:
        raise ValueError("event_exact backend requires tau > 0.")

    x1 = _scaled_ts(ts1, gamma)
    x2 = _scaled_ts(ts2, gamma)
    radius = np.sqrt(max(0.0, -np.log(max(tau, 1e-12)))) + 1e-12
    tree = cKDTree(x2)

    row_ptr = np.zeros(n + 1, dtype=np.int32)
    cols = []
    sims = []
    nnz = 0

    for i in range(n):
        j_start = max(0, i - diag_offset) if only_triu else 0
        neigh = tree.query_ball_point(x1[i], r=radius)
        if len(neigh) == 0:
            row_ptr[i + 1] = nnz
            continue

        neigh_arr = np.asarray(neigh, dtype=np.int32)
        if only_triu:
            neigh_arr = neigh_arr[neigh_arr >= j_start]
        if neigh_arr.size == 0:
            row_ptr[i + 1] = nnz
            continue

        neigh_arr = np.sort(neigh_arr)
        diff = x2[neigh_arr, :] - x1[i : i + 1, :]
        d2 = np.sum(diff * diff, axis=1)
        sim = np.exp(-d2).astype(np.float32)
        keep = sim >= tau

        neigh_arr = neigh_arr[keep]
        sim = sim[keep]
        if neigh_arr.size > 0:
            cols.append(neigh_arr.astype(np.int32))
            sims.append(sim.astype(np.float32))
            nnz += len(neigh_arr)
        row_ptr[i + 1] = nnz

    if nnz == 0:
        return row_ptr, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    col_idx = np.concatenate(cols).astype(np.int32)
    sim_vals = np.concatenate(sims).astype(np.float32)
    return row_ptr, col_idx, sim_vals
