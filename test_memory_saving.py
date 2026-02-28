import time
import numpy as np
from locomotif import loco_jit

def test():
    N = 1000
    M = 1000
    csm = np.random.rand(N, M).astype(np.float32)
    bp_dir = np.random.randint(0, 3, size=(N, M)).astype(np.int8)
    
    mask = np.zeros((N, M), dtype=np.bool_)
    
    t0 = time.time()
    paths = loco_jit.find_best_paths(csm, mask, 0.8, 10, 5, True, bp_dir, None, None)
    print(f"Time: {time.time()-t0:.3f}s")
    
if __name__ == "__main__":
    test()
