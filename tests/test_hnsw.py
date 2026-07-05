import numpy as np
from HNSW.algorithm import Algorithm

def test_hnsw_float():
    data = np.random.rand(50, 16).astype(np.float32)
    algo = Algorithm()
    algo.fit(data, mode="float")
    res = algo.query(data[0], 5)
    assert len(res) == 5

def test_hnsw_sq8():
    data = np.random.rand(50, 16).astype(np.float32)
    algo = Algorithm()
    algo.fit(data, mode="sq8")
    res = algo.query(data[0], 5)
    assert len(res) == 5

def test_hnsw_lsh():
    data = np.random.rand(50, 16).astype(np.float32)
    algo = Algorithm()
    algo.fit(data, mode="lsh")
    res = algo.query(data[0], 5)
    assert len(res) == 5

def test_hnsw_lsh_sq8():
    data = np.random.rand(50, 16).astype(np.float32)
    algo = Algorithm()
    algo.fit(data, mode="lsh_sq8")
    res = algo.query(data[0], 5)
    assert len(res) == 5

def test_hnsw_lsh_float():
    data = np.random.rand(50, 16).astype(np.float32)
    algo = Algorithm()
    algo.fit(data, mode="lsh_float")
    res = algo.query(data[0], 5)
    assert len(res) == 5

def test_hnsw_self_neighbor():
    rng = np.random.default_rng(0)
    data = rng.random((50, 16)).astype(np.float32)
    for mode in ["float", "sq8", "lsh", "lsh_sq8", "lsh_float"]:
        algo = Algorithm()
        algo.fit(data, mode=mode)
        res = algo.query(data[10], 1)
        assert res[0] == 10

def test_hnsw_flexible_params():
    data = np.random.rand(100, 16).astype(np.float32)
    algo = Algorithm()
    # Fit with ds_size limiting the index to first 50 points
    algo.fit(data, M=16, ef_construction=100, mode="sq8", ds_size=50)
    # Query with refine_r limiting exact refinement to 3 candidates
    res = algo.query(data[0], k=5, ef=20, refine_r=3)
    assert len(res) == 3
