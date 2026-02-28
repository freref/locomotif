import time
import numpy as np
from locomotif import loco_jit

def run():
    print("Testing find_best_paths timing breakdown")
    # Simulate a scenario
    np.random.seed(42)
    N = 5000
    csm = np.random.rand(N, N).astype(np.float32)
    mask = np.zeros((N, N), dtype=np.bool_)
    bp_dir = np.random.randint(0, 3, size=(N, N)).astype(np.int8)
    src_id = np.zeros((N, N), dtype=np.int64)
    dist = np.ones((N, N), dtype=np.int32) * 50
    
    t0 = time.time()
    loco_jit.find_best_paths(csm, mask, 0.8, 10, 5, True, bp_dir, src_id, dist)
    t1 = time.time()
    print(f"Time: {t1-t0:.3f}s")
    
if __name__ == "__main__":
    run()
