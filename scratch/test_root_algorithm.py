import numpy as np
from algorithm import Algorithm

print("Creating dummy training and test data...")
data = np.random.rand(1000, 128).astype(np.float32)
q = np.random.rand(128).astype(np.float32)

backends = [
    {"name": "Linear", "params": {"backend": "linear"}},
    {"name": "IVF-LSH", "params": {"backend": "lsh", "n_tables": 5, "n_bits": 8, "n_clusters": 16}},
    {"name": "HNSW", "params": {"backend": "hnsw", "M": 8, "ef_construction": 50, "mode": "sq8"}},
    {"name": "MultiProbe LSH", "params": {"backend": "multiprobe", "n_tables": 5, "n_bits": 8}}
]

for b in backends:
    print(f"\n--- Testing Backend: {b['name']} ---")
    algo = Algorithm()
    algo.fit(data, **b["params"])
    
    # query
    res = algo.query(q, 5)
    print(f"Query returned 5 neighbors: {res}")
    
    # distance count
    dists = algo.get_n_distances()
    print(f"Distances computed: {dists}")

print("\nAll backends initialized and queried successfully!")
