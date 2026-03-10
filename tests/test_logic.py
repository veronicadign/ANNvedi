import numpy as np
from linear_ann.wrapper import LinearANN

def test_fit_query():
    data = np.random.rand(50, 16).astype(np.float32)

    index = LinearANN()
    index.fit(data)

    res = index.query(data[0], 5)

    assert len(res) == 5


def test_self_neighbor():
    data = np.random.rand(50, 16).astype(np.float32)

    index = LinearANN()
    index.fit(data)

    res = index.query(data[10], 1)

    assert res[0] == 10


def test_distance_counter():
    data = np.random.rand(20, 8).astype(np.float32)

    index = LinearANN()
    index.fit(data)

    index.query(data[0], 3)

    assert index.total_distances_count() == 20