import linear_ann_cpp

import numpy as np
import h5py
import os

class LinearANN:
    def __init__(self):
        self.index = linear_ann_cpp.LinearIndex()
        self._data = None

    def fit(self, data):
        self._data = data
        self.index.fit(data)

    def query(self, q, k):
        q_np = np.asarray(q, dtype=np.float32)
        return self.index.query(q_np, k)

    def total_distances_count(self):
        return self.index.total_distances_count()