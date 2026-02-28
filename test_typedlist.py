import numba
from numba.typed import List
from numba import int32
import numpy as np
import time

@numba.njit
def test_append(n):
    paths = List.empty_list(int32[:, :])
    buf = np.zeros((100, 2), dtype=np.int32)
    for _ in range(n):
        paths.append(buf.copy())
    return paths

# Warmup
test_append(1)

t0 = time.time()
test_append(150000)
print("Time:", time.time() - t0)
