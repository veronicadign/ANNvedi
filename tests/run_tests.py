import sys
import types
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Mock the broken pytest module before importing any test files
mock_pytest = types.ModuleType("pytest")
sys.modules["pytest"] = mock_pytest

# Add root and tests folders to sys.path
sys.path.insert(0, "./tests")
sys.path.insert(0, ".")

test_funcs_metadata = [
    ("test_logic.test_fit_query", "test_logic", "test_fit_query"),
    ("test_logic.test_self_neighbor", "test_logic", "test_self_neighbor"),
    ("test_logic.test_distance_counter", "test_logic", "test_distance_counter"),
    ("test_lsh.test_lsh_fit_query", "test_lsh", "test_lsh_fit_query"),
    ("test_lsh.test_lsh_self_neighbor", "test_lsh", "test_lsh_self_neighbor"),
    ("test_lsh.test_lsh_distance_counter", "test_lsh", "test_lsh_distance_counter"),
    ("test_lsh.test_algorithm_lsh_backend", "test_lsh", "test_algorithm_lsh_backend"),
    ("test_lsh.test_algorithm_linear_unchanged", "test_lsh", "test_algorithm_linear_unchanged"),
    ("test_hnsw.test_hnsw_float", "test_hnsw", "test_hnsw_float"),
    ("test_hnsw.test_hnsw_sq8", "test_hnsw", "test_hnsw_sq8"),
    ("test_hnsw.test_hnsw_lsh", "test_hnsw", "test_hnsw_lsh"),
    ("test_hnsw.test_hnsw_lsh_sq8", "test_hnsw", "test_hnsw_lsh_sq8"),
    ("test_hnsw.test_hnsw_lsh_float", "test_hnsw", "test_hnsw_lsh_float"),
    ("test_hnsw.test_hnsw_self_neighbor", "test_hnsw", "test_hnsw_self_neighbor"),
    ("test_hnsw.test_hnsw_flexible_params", "test_hnsw", "test_hnsw_flexible_params")
]

def run_single_test_process(name, module_name, func_name):
    # Imports must occur inside the child process to prevent inheritance conflicts
    import sys
    import types
    mock_pytest = types.ModuleType("pytest")
    sys.modules["pytest"] = mock_pytest
    sys.path.insert(0, "./tests")
    sys.path.insert(0, ".")
    
    try:
        import importlib
        mod = importlib.import_module(module_name)
        func = getattr(mod, func_name)
        func()
        return name, True, None
    except Exception as e:
        import traceback
        err_msg = "".join(traceback.format_exception(*sys.exc_info()))
        return name, False, err_msg

def main():
    passed = 0
    failed = 0

    print("\n--- Running Workspace Test Cases in Parallel Processes ---")

    # Execute test cases in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_single_test_process, name, mod_name, func_name): name 
            for name, mod_name, func_name in test_funcs_metadata
        }
        
        for fut in as_completed(futures):
            name, success, err_msg = fut.result()
            if success:
                print(f"{name}... PASSED")
                passed += 1
            else:
                print(f"{name}... FAILED")
                print(err_msg)
                failed += 1

    print(f"\nTest Summary: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
