
from numba.experimental import jitclass
from numba import int32, float32
import numpy as np

@jitclass([("path", int32[:, :]), ("similarities", float32[:]), ("cumulative_similarities", float32[:]), ("index_i", int32[:]), ("index_j", int32[:]), ("i1", int32), ("il", int32), ("j1", int32), ("jl", int32)])
class Path:

    def __init__(self, path, similarities):
        assert len(path) == len(similarities)
        self.path = path
        self.similarities = np.asarray(similarities, dtype=np.float32)
        self.cumulative_similarities = np.empty(len(self.similarities) + 1, dtype=np.float32)
        self.cumulative_similarities[0] = 0.0
        self.cumulative_similarities[1:] = np.cumsum(self.similarities)
        self.i1 = path[0][0]
        self.il = path[len(path) - 1][0] + 1
        self.j1 = path[0][1]
        self.jl = path[len(path) - 1][1] + 1
        self._construct_index()

    def __getitem__(self, i):
        return self.path[i, :]

    def __len__(self):
        return len(self.path)

    def _construct_index(self):
        path = self.path
        i_curr = path[0][0]
        j_curr = path[0][1]

        index_i = np.zeros(self.il - self.i1, dtype=np.int32)
        index_j = np.zeros(self.jl - self.j1, dtype=np.int32)

        for i in range(1, len(path)):
            i_next = path[i, 0]
            j_next = path[i, 1]

            if i_next != i_curr:
                index_i[i_curr - self.i1 + 1 : i_next - self.i1 + 1] = i
                i_curr = i_next

            if j_next != j_curr:
                index_j[j_curr - self.j1 + 1 : j_next - self.j1 + 1] = i
                j_curr = j_next
        
        self.index_i = index_i
        self.index_j = index_j

    # returns the index of the first occurrence of the given row
    def find_i(self, i):
        assert i - self.i1 >= 0 and i - self.i1 < len(self.index_i)
        return self.index_i[i - self.i1]

    # returns the index of the first occurrence of the given column
    def find_j(self, j):
        assert j >= self.j1 and j - self.j1 < len(self.index_j)
        return self.index_j[j - self.j1]
    
    def get_subpath_between_col_indices(self, j1, j2):
        kb, ke = self.find_j(j1), self.find_j(j2)
        return self.path[kb:ke+1]

    def get_subpath_between_row_indices(self, i1, i2):
        kb, ke = self.find_i(i1), self.find_i(i2)
        return self.path[kb:ke+1]
    
def project_to_horizontal_axis(path):
    return (path[0][1], path[-1][1]+1)

def project_to_vertical_axis(path):
    return (path[0][0], path[-1][0]+1)
