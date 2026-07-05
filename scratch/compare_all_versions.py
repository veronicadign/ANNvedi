import os
import time
import h5py
import numpy as np
import yaml
import importlib.util

dataset_path = "dataset/yahoo-minilm-public.hdf5"

if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}")
    exit(1)

print("Loading full dataset...")
with h5py.File(dataset_path, 'r') as f:
    train = f['train'][:]
    test = f['test'][:]

print(f"Dataset shapes: Train={train.shape}, Test={test.shape}")

# Precompute linear brute force ground truth on all test queries
print("\nComputing linear ground truth on full dataset...")
# Load root algorithm
spec = importlib.util.spec_from_file_location("algorithm", "algorithm.py")
root_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_mod)

linear_algo = root_mod.Algorithm()
t0 = time.time()
linear_algo.fit(train, backend="linear")
fit_time_linear = time.time() - t0

t0 = time.time()
ground_truth = []
for q in test:
    ground_truth.append(linear_algo.query(q, 5))
query_time_linear = (time.time() - t0) * 1000 / len(test)

print(f"Linear Fit: {fit_time_linear:.4f} s | Linear Query: {query_time_linear:.4f} ms")

# Define variants to test
variants = [
    {
        "name": "IVF-LSH (Our Version)",
        "algo_path": "algorithm.py",
        "scenarios_path": "scenarios.yaml"
    },
    {
        "name": "HNSW",
        "algo_path": "HNSW/algorithm.py",
        "scenarios_path": "HNSW/scenarios.yaml"
    },
    {
        "name": "MultiProbe LSH",
        "algo_path": "MultiProbe_LSH/algorithm.py",
        "scenarios_path": "MultiProbe_LSH/scenarios.yaml"
    }
]

results = []

for var in variants:
    print(f"\nEvaluating {var['name']}...")
    
    # Load module dynamically
    spec = importlib.util.spec_from_file_location("algorithm_" + var['name'].replace(" ", "_"), var['algo_path'])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AlgoClass = mod.Algorithm
    
    # Load scenarios
    with open(var['scenarios_path'], "r") as f:
        scenarios_data = yaml.safe_load(f)["scenarios"]
        
    for scenario_name, scenario_cfg in scenarios_data.items():
        cfg = scenario_cfg["default"]
        index_params = cfg.get("index_params", {})
        query_params = cfg.get("query_params", {})
        
        print(f"  Running scenario: {scenario_name}...")
        
        # Instantiate
        algo = AlgoClass()
        
        # Fit index
        t0 = time.time()
        algo.fit(train, **index_params)
        fit_time = time.time() - t0
        
        # Query index
        t0 = time.time()
        preds = []
        for q in test:
            preds.append(algo.query(q, 5, **query_params))
        query_time = (time.time() - t0) * 1000 / len(test)
        
        # Calculate recall@5
        recalls = []
        for i in range(len(test)):
            true_set = set(ground_truth[i][:5])
            pred_set = set(preds[i][:5])
            intersection = true_set.intersection(pred_set)
            recalls.append(len(intersection) / 5.0)
        avg_recall = np.mean(recalls)
        
        dist_count = algo.get_n_distances()
        
        results.append({
            "variant": var['name'],
            "scenario": scenario_name,
            "recall": avg_recall,
            "query_time": query_time,
            "fit_time": fit_time,
            "dist_count": dist_count
        })

# Print results as Markdown table
print("\n### Unified Performance Comparison Table")
print(f"| {'Algorithm/Variant':<25} | {'Scenario':<16} | {'Recall@5':<10} | {'Query (ms)':<12} | {'Fit (s)':<10} | {'Dist Computations':<18} |")
print("| :" + "-"*23 + " | :" + "-"*14 + " | :" + "-"*8 + " | :" + "-"*10 + " | :" + "-"*8 + " | :" + "-"*16 + " |")
for r in results:
    print(f"| {r['variant']:<25} | {r['scenario']:<16} | {r['recall']:<10.4f} | {r['query_time']:<12.4f} | {r['fit_time']:<10.4f} | {r['dist_count']:<18} |")
