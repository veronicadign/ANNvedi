import numpy as np
import sys
import os

# Add build directory to path for loading the binary module in local development
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

from competitors.linear.algorithm import Algorithm

def test_fit_query():
    data = np.random.rand(50, 16).astype(np.float32)

    index = Algorithm()
    index.fit(data)

    res = index.query(data[0], 5)

    assert len(res) == 5


def test_self_neighbor():
    data = np.random.rand(50, 16).astype(np.float32)

    index = Algorithm()
    index.fit(data)

    res = index.query(data[10], 1)

    assert res[0] == 10


def test_distance_counter():
    data = np.random.rand(20, 8).astype(np.float32)

    index = Algorithm()
    index.fit(data)

    index.query(data[0], 3)

    assert index.get_n_distances() == 20