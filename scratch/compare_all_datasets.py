import os
import time
import h5py
import numpy as np
import yaml
import importlib.util

dataset_dir = "dataset"
hdf5_files = [f for f in os.listdir(dataset_dir) if f.endswith(".hdf5")]
hdf5_files.sort()

if not hdf5_files:
    print(f"No HDF5 datasets found in {dataset_dir}")
    exit(1)

print(f"Found {len(hdf5_files)} datasets: {hdf5_files}")

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

all_dataset_results = {}

for db_file in hdf5_files:
    dataset_path = os.path.join(dataset_dir, db_file)
    print(f"\n======================================================================")
    print(f"EVALUATING DATASET: {db_file}")
    print(f"======================================================================")
    
    try:
        with h5py.File(dataset_path, 'r') as f:
            train_dataset = f['train']
            test_dataset = f['test']
            subset_size = min(20000, train_dataset.shape[0])
            train_sub = train_dataset[:subset_size]
            test_sub = test_dataset[:100]
    except Exception as e:
        print(f"Error loading {db_file}: {e}. Skipping.")
        continue

    print(f"Shapes: Train={train_sub.shape}, Test={test_sub.shape}")

    # Ground truth
    print("Computing linear ground truth...")
    spec = importlib.util.spec_from_file_location("algorithm_linear", "algorithm.py")
    root_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_mod)
    linear_algo = root_mod.Algorithm()
    
    t0 = time.time()
    linear_algo.fit(train_sub, backend="linear")
    fit_time_linear = time.time() - t0
    
    t0 = time.time()
    ground_truth = []
    for q in test_sub:
        ground_truth.append(linear_algo.query(q, 5))
    query_time_linear = (time.time() - t0) * 1000 / len(test_sub)
    print(f"Linear Fit: {fit_time_linear:.4f} s | Linear Query: {query_time_linear:.4f} ms")

    results = []

    for var in variants:
        print(f"Evaluating {var['name']}...")
        spec = importlib.util.spec_from_file_location("algorithm_" + var['name'].replace(" ", "_"), var['algo_path'])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        AlgoClass = mod.Algorithm

        with open(var['scenarios_path'], "r") as f:
            scenarios_data = yaml.safe_load(f)["scenarios"]

        for scenario_name, scenario_cfg in scenarios_data.items():
            if var['name'] == "HNSW" and "hybrid" in scenario_name:
                print(f"  Skipping HNSW hybrid scenario: {scenario_name}")
                continue

            cfg = scenario_cfg["default"]
            index_params = cfg.get("index_params", {})
            query_params = cfg.get("query_params", {})

            # Instantiate & fit
            algo = AlgoClass()
            t0 = time.time()
            algo.fit(train_sub, **index_params)
            fit_time = time.time() - t0

            # Query
            algo.get_n_distances()
            t0 = time.time()
            preds = []
            for q in test_sub:
                preds.append(algo.query(q, 5, **query_params))
            query_time = (time.time() - t0) * 1000 / len(test_sub)

            # Recall
            recalls = []
            for i in range(len(test_sub)):
                true_set = set(ground_truth[i][:5])
                pred_set = set(preds[i][:5])
                intersection = true_set.intersection(pred_set)
                recalls.append(len(intersection) / 5.0)
            avg_recall = np.mean(recalls)
            dist_count = algo.get_n_distances() / len(test_sub)

            if query_time <= 5.0:
                results.append({
                    "variant": var['name'],
                    "scenario": scenario_name,
                    "recall": avg_recall,
                    "query_time": query_time,
                    "fit_time": fit_time,
                    "dist_count": dist_count
                })
            else:
                print(f"    -> Skipped {scenario_name} (Query Time {query_time:.2f} ms > 5 ms)")

    all_dataset_results[db_file] = results
    
    # Print results for this dataset
    print(f"\n### Results for {db_file}")
    print(f"| {'Algorithm/Variant':<25} | {'Scenario':<18} | {'Recall@5':<10} | {'Query (ms)':<12} | {'Fit (s)':<10} | {'Dist Computations':<18} |")
    print("| :" + "-"*23 + " | :" + "-"*16 + " | :" + "-"*8 + " | :" + "-"*10 + " | :" + "-"*8 + " | :" + "-"*16 + " |")
    for r in results:
        print(f"| {r['variant']:<25} | {r['scenario']:<18} | {r['recall']:<10.4f} | {r['query_time']:<12.4f} | {r['fit_time']:<10.4f} | {r['dist_count']:<18.1f} |")

# Print overall summary markdown formatting for copy pasting
print("\n======================================================================")
print("ALL DATASETS BENCHMARK COMPLETED")
print("======================================================================")
for db_file, results in all_dataset_results.items():
    print(f"\n#### Dataset: {db_file}")
    print(f"| {'Algorithm/Variant':<25} | {'Scenario':<18} | {'Recall@5':<10} | {'Query (ms)':<12} | {'Fit (s)':<10} | {'Dist Computations':<18} |")
    print("| :" + "-"*23 + " | :" + "-"*16 + " | :" + "-"*8 + " | :" + "-"*10 + " | :" + "-"*8 + " | :" + "-"*16 + " |")
    for r in results:
        print(f"| {r['variant']:<25} | {r['scenario']:<18} | {r['recall']:<10.4f} | {r['query_time']:<12.4f} | {r['fit_time']:<10.4f} | {r['dist_count']:<18.1f} |")
