import numpy as np
import sys
import os

# Add build directory to path for loading the binary module in local development
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

from competitors.lsh_optimized.algorithm import Algorithm as OptimizedLSHAlgorithm
from competitors.lsh_simple.algorithm import Algorithm as SimpleLSHAlgorithm

def test_lsh_fit_query():
    data = np.random.rand(50, 16).astype(np.float32)
    idx = OptimizedLSHAlgorithm()
    idx.fit(data, n_tables=10, n_bits=8)
    res = idx.query(data[0], 5)
    assert len(res) == 5

def test_lsh_self_neighbor():
    rng = np.random.default_rng(0)
    data = rng.random((50, 16)).astype(np.float32)
    idx = OptimizedLSHAlgorithm()
    idx.fit(data, n_tables=20, n_bits=12)
    res = idx.query(data[10], 1)
    assert res[0] == 10

def test_lsh_distance_counter():
    data = np.random.rand(20, 8).astype(np.float32)
    idx = OptimizedLSHAlgorithm()
    idx.fit(data, n_tables=10, n_bits=8)
    idx.query(data[0], 3)
    idx.query(data[1], 3)
    assert idx.get_n_distances() > 0

def test_algorithm_lsh_backend():
    data = np.random.rand(50, 16).astype(np.float32)
    algo = OptimizedLSHAlgorithm()
    algo.fit(data, n_tables=10, n_bits=8)
    res = algo.query(data[0], 5)
    assert len(res) == 5

def test_lsh_simple_fit_query():
    data = np.random.rand(50, 16).astype(np.float32)
    idx = SimpleLSHAlgorithm()
    idx.fit(data, n_tables=10, n_bits=8)
    res = idx.query(data[0], 5)
    assert len(res) == 5
