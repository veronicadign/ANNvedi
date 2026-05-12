import numpy as np
import pytest
from lsh_ann.wrapper import LSHANN
from algorithm import Algorithm


def test_lsh_fit_query():
    data = np.random.rand(50, 16).astype(np.float32)
    idx = LSHANN()
    idx.fit(data)
    res = idx.query(data[0], 5)
    assert len(res) == 5


def test_lsh_self_neighbor():
    rng = np.random.default_rng(0)
    data = rng.random((50, 16)).astype(np.float32)
    idx = LSHANN(n_tables=20, n_bits=12)
    idx.fit(data)
    res = idx.query(data[10], 1)
    assert res[0] == 10


def test_lsh_distance_counter():
    data = np.random.rand(20, 8).astype(np.float32)
    idx = LSHANN()
    idx.fit(data)
    idx.query(data[0], 3)
    idx.query(data[1], 3)
    assert idx.total_distances_count() > 0


def test_algorithm_lsh_backend():
    data = np.random.rand(50, 16).astype(np.float32)
    algo = Algorithm()
    algo.fit(data, backend='lsh', n_tables=10, n_bits=8)
    res = algo.query(data[0], 5)
    assert len(res) == 5


def test_algorithm_linear_unchanged():
    data = np.random.rand(50, 16).astype(np.float32)
    algo = Algorithm()
    algo.fit(data)  # no backend → linear
    from linear_ann.wrapper import LinearANN
    assert isinstance(algo._index, LinearANN)
