import h5py
import numpy as np
import sys
from pathlib import Path

def compute_ground_truth(data, queries, k=100):
    neighbors = np.empty((queries.shape[0], k), dtype=np.int32)
    distances = np.empty((queries.shape[0], k), dtype=np.float32)
    
    for i, q in enumerate(queries):
        dists = np.linalg.norm(data - q, axis=1)
        idx = np.argsort(dists)[:k]
        neighbors[i] = idx
        distances[i] = dists[idx]
        if i % 10 == 0:
            print(f"Computed ground truth for query {i}/{queries.shape[0]}")
            
    return neighbors, distances

def main():
    if len(sys.argv) < 3:
        print("Usage: python create_small_dataset.py <input_hdf5> <output_hdf5> [n_train] [n_test]")
        return

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    print(f"Creating small dataset from {input_path} to {output_path}")
    print(f"Target: {n_train} train points, {n_test} test points")

    with h5py.File(input_path, "r") as f_in:
        train = f_in["train"][:n_train]
        test = f_in["test"][:n_test]
        
    print("Computing ground truth...")
    neighbors, distances = compute_ground_truth(train, test, k=100)

    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("train", data=train)
        f_out.create_dataset("test", data=test)
        f_out.create_dataset("neighbors", data=neighbors)
        f_out.create_dataset("distances", data=distances)

    print(f"Done! Small dataset saved to {output_path}")

if __name__ == "__main__":
    main()
