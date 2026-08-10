import os
import time
import h5py
import numpy as np
import yaml
from linear_ann.wrapper import LinearANN
from lsh_ann.wrapper import LSHANN
from algorithm import Algorithm

dataset_path = "dataset/yahoo-minilm-public.hdf5"

if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}")
    exit(1)

print("Loading full dataset...")
with h5py.File(dataset_path, 'r') as f:
    train = f['train'][:]
    test = f['test'][:]

print(f"Full Train shape: {train.shape}, dtype: {train.dtype}")
print(f"Full Test shape: {test.shape}, dtype: {test.dtype}")

# Load scenarios
with open("scenarios.yaml", "r") as f:
    scenarios_data = yaml.safe_load(f)["scenarios"]

# First compute the true ground truth on all test queries using brute-force Linear
print("\nComputing linear search ground truth on full dataset...")
linear_algo = Algorithm()
t0 = time.time()
linear_algo.fit(train)
fit_time_linear = time.time() - t0

t0 = time.time()
ground_truth = []
for q in test:
    ground_truth.append(linear_algo.query(q, 5))
query_time_linear = (time.time() - t0) * 1000 / len(test)

print(f"Linear Fit Time: {fit_time_linear:.4f} s")
print(f"Linear Avg Query Time: {query_time_linear:.4f} ms")

print("\n--- Evaluating Configured Scenarios ---")
print(f"{'Scenario':<20} | {'Recall@5':<10} | {'Query Time (ms)':<15} | {'Fit Time (s)':<12} | {'Dist Computations':<18}")
print("-" * 85)

for scenario_name, scenario_cfg in scenarios_data.items():
    cfg = scenario_cfg["default"]
    index_params = cfg.get("index_params", {})
    query_params = cfg.get("query_params", {})
    
    # Initialize algorithm
    algo = Algorithm()
    
    # Measure fit time
    t0 = time.time()
    algo.fit(train, **index_params)
    fit_time = time.time() - t0
    
    # Measure query time
    t0 = time.time()
    results = []
    for q in test:
        res = algo.query(q, 5, **query_params)
        results.append(res)
    query_time = (time.time() - t0) * 1000 / len(test)
    
    # Calculate recall
    recalls = []
    for i in range(len(test)):
        true_set = set(ground_truth[i][:5])
        pred_set = set(results[i][:5])
        intersection = true_set.intersection(pred_set)
        recalls.append(len(intersection) / 5.0)
    avg_recall = np.mean(recalls)
    
    dist_count = algo.get_n_distances()
    
    print(f"{scenario_name:<20} | {avg_recall:<10.4f} | {query_time:<15.4f} | {fit_time:<12.4f} | {dist_count:<18}")
