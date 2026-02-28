import numpy as np

n = 5
vwidth = 1

mask1 = np.full((n, n), True)
for row_idx in range(n):
    col_start = row_idx + vwidth + 1
    if col_start < n:
        mask1[row_idx, col_start:] = False

mask2 = np.tril(np.ones((n, n), dtype=bool), k=vwidth)

print("mask1:\n", mask1.astype(int))
print("mask2:\n", mask2.astype(int))
print("Equal:", np.array_equal(mask1, mask2))
