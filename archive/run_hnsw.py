import numpy as np
from HNSW.algorithm import Algorithm

print("=== Starting HNSW Demonstration ===")

# 1. Generate random dummy dataset
n_vectors = 1000
dimensions = 128
print(f"Generating dummy dataset: {n_vectors} vectors, {dimensions} dimensions...")
data = np.random.rand(n_vectors, dimensions).astype(np.float32)

# 2. Run HNSW in different modes
for mode in ["float", "sq8", "lsh"]:
    print(f"\n--- Running in '{mode}' mode ---")
    
    # Create and fit index
    index = Algorithm()
    index.fit(data, M=16, ef_construction=100, mode=mode)
    print("Index successfully fit.")
    
    # Query index
    q = data[0]  # query with the first vector (should retrieve itself as index 0)
    k = 5
    ef = 50
    results = index.query(q, k, ef=ef)
    dists_count = index.get_n_distances()
    
    print(f"Query vector: first dataset element")
    print(f"Top-{k} nearest neighbor indices: {results}")
    print(f"Distance computations performed: {dists_count}")
    print(f"Self-neighbor check (index 0 is first?): {'PASSED' if results[0] == 0 else 'FAILED'}")

print("\n=== HNSW Demonstration Finished ===")
