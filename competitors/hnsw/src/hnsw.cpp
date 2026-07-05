#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include "simd.h"

namespace py = pybind11;

// (distance, node_index)
using Pair = std::pair<float, int32_t>;

// Max-heap: top = furthest
using MaxHeap = std::priority_queue<Pair>;
// Min-heap: top = nearest
using MinHeap = std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>>;

// Tracker for visited nodes during search to avoid thread_local library issues
struct SearchTracker {
    std::vector<uint32_t> visited_gen;
    uint32_t current_gen = 1;
    std::mt19937 rng{std::random_device{}()};
};

// ---------------------------------------------------------------------------
class HNSWIndex {
public:
    HNSWIndex() = default;

    // ------------------------------------------------------------------
    // Build the index
    // ------------------------------------------------------------------
    void fit(py::array_t<float> data, int M, int ef_construction, const std::string& mode = "float") {
        py::buffer_info buf = data.request();
        if (buf.ndim != 2)
            throw std::runtime_error("Input must be 2D");

        npts_            = (int)buf.shape[0];
        dim_             = (int)buf.shape[1];
        M_               = M;
        Mmax0_           = 2 * M;          // layer-0 gets 2×M edges
        ef_construction_ = ef_construction;
        ml_              = 1.0 / std::log((double)M);  // level multiplier

        if (mode == "sq8") {
            mode_ = 1;
        } else if (mode == "lsh") {
            mode_ = 2;
        } else if (mode == "lsh_sq8") {
            mode_ = 3;
        } else if (mode == "lsh_float") {
            mode_ = 4;
        } else {
            mode_ = 0;
        }

        float* ptr = static_cast<float*>(buf.ptr);
        data_.assign(ptr, ptr + npts_ * dim_);

        // Perform quantization if SQ8 (1) or Hybrid LSH-SQ8 (3) mode is active
        if (mode_ == 1 || mode_ == 3) {
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();
            for (float val : data_) {
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
            if (max_val - min_val > 1e-8f) {
                scale_ = (max_val - min_val) / 255.0f;
                offset_ = min_val;
            } else {
                scale_ = 1.0f;
                offset_ = 0.0f;
            }

            quantized_data_.resize(data_.size());
            for (size_t i = 0; i < data_.size(); ++i) {
                float val = data_[i];
                int qval = std::round((val - offset_) / scale_);
                quantized_data_[i] = static_cast<uint8_t>(std::max(0, std::min(255, qval)));
            }
        } 
        
        // Perform LSH projection and signature calculations if LSH (2), Hybrid LSH-SQ8 (3), or Hybrid LSH-Float (4) is active
        if (mode_ == 2 || mode_ == 3 || mode_ == 4) {
            projections_.resize(n_bits_, std::vector<float>(dim_));
            std::mt19937 gen(42);
            std::normal_distribution<float> dist(0.0, 1.0);
            for (int b = 0; b < n_bits_; ++b) {
                for (int d = 0; d < dim_; ++d) {
                    projections_[b][d] = dist(gen);
                }
            }

            // Perform Gram-Schmidt orthogonalization to make projections orthogonal
            if (n_bits_ <= dim_) {
                for (int b = 0; b < n_bits_; ++b) {
                    for (int b_prev = 0; b_prev < b; ++b_prev) {
                        float dot = 0.0f;
                        for (int d = 0; d < dim_; ++d) {
                            dot += projections_[b][d] * projections_[b_prev][d];
                        }
                        for (int d = 0; d < dim_; ++d) {
                            projections_[b][d] -= dot * projections_[b_prev][d];
                        }
                    }
                    float norm = 0.0f;
                    for (int d = 0; d < dim_; ++d) {
                        norm += projections_[b][d] * projections_[b][d];
                    }
                    norm = std::sqrt(norm);
                    if (norm > 1e-8f) {
                        for (int d = 0; d < dim_; ++d) {
                            projections_[b][d] /= norm;
                        }
                    }
                }
            }

            signatures_.resize(npts_, 0);
            for (int i = 0; i < npts_; ++i) {
                signatures_[i] = compute_signature(&data_[i * dim_]);
            }
        }

        graph_.resize(npts_);
        level_.resize(npts_, 0);

        entry_point_ = -1;
        max_level_   = -1;
        n_distances_.store(0);

        is_building_ = true;

        // Insert first node sequentially to establish the entry point
        if (npts_ > 0) {
            std::mt19937 rng(42);
            level_[0] = random_level(rng);
            graph_[0].assign(level_[0] + 1, {});
            entry_point_ = 0;
            max_level_   = level_[0];
        }

        // Spawn parallel threads for inserting remaining nodes
        std::atomic<int> next_idx{1};
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = 4;

        {
            py::gil_scoped_release release;
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([this, &next_idx]() {
                    SearchTracker tracker;
                    tracker.visited_gen.assign(npts_, 0);
                    while (true) {
                        int idx = next_idx.fetch_add(1);
                        if (idx >= npts_) break;
                        insert(idx, tracker);
                    }
                });
            }
            for (auto& th : threads) {
                th.join();
            }
        }

        is_building_ = false;
    }

    // ------------------------------------------------------------------
    // Answer one k-NN query
    // ------------------------------------------------------------------
    py::array_t<int64_t> query(py::array_t<float> query_arr, int k, int ef, int refine_r = -1) {
        py::buffer_info qbuf = query_arr.request();
        if (qbuf.ndim != 1 || (int)qbuf.shape[0] != dim_)
            throw std::runtime_error("Query dimension mismatch");

        const float* q = static_cast<const float*>(qbuf.ptr);
        ef = std::max(ef, k);

        uint64_t q_sig = (mode_ == 2 || mode_ == 3 || mode_ == 4) ? compute_signature(q) : 0;

        int ep;
        int max_l;
        {
            std::lock_guard<std::mutex> lock(global_lock_);
            ep = entry_point_;
            max_l = max_level_;
        }

        SearchTracker tracker;
        tracker.visited_gen.assign(npts_, 0);

        // Drain max-heap → nearest-first
        std::vector<Pair> results;

        {
            py::gil_scoped_release release;
            // Greedy descent: layers max_level → 1  (ef=1)
            for (int l = max_l; l > 0; --l) {
                MaxHeap W = search_layer(q, q_sig, ep, 1, l, /*count=*/true, tracker);
                ep = nearest_in(W);
            }

            // Full beam search at layer 0
            MaxHeap W = search_layer(q, q_sig, ep, ef, 0, /*count=*/true, tracker);

            results.reserve(W.size());

            if (mode_ == 0 || mode_ == 4) {
                // Float or LSH-Float mode: Heap already has exact L2 distances
                while (!W.empty()) {
                    results.push_back(W.top());
                    W.pop();
                }
                std::reverse(results.begin(), results.end());
            } else {
                // SQ8 or Hybrid LSH-SQ8 mode: Rerank candidate list using exact floats
                int actual_refine_r = (refine_r > 0) ? std::min(refine_r, (int)W.size()) : (int)W.size();
                while ((int)W.size() > actual_refine_r) {
                    W.pop();
                }
                while (!W.empty()) {
                    int node = W.top().second;
                    float d_exact = dist_to_float(node, q);
                    results.push_back({d_exact, node});
                    W.pop();
                }
                std::sort(results.begin(), results.end());
            }
        }

        int out_k = std::min(k, (int)results.size());
        py::array_t<int64_t> out(out_k);
        int64_t* out_ptr = static_cast<int64_t*>(out.request().ptr);
        for (int i = 0; i < out_k; ++i)
            out_ptr[i] = results[i].second;

        return out;
    }

    int64_t total_distances_count() const { return n_distances_.load(); }
    void    reset_distances_count()       { n_distances_.store(0); }

private:
    // ------------------------------------------------------------------
    // Data members
    // ------------------------------------------------------------------
    int npts_ = 0, dim_ = 0;
    int M_ = 16, Mmax0_ = 32;
    int ef_construction_ = 100;
    double ml_ = 0.0;
    int max_level_ = -1, entry_point_ = -1;
    
    // Thread safety synchronization primitives
    mutable std::atomic<int64_t> n_distances_{0};
    std::mutex global_lock_;
    
    // Striped lock pool to prevent memory bad_alloc overhead of 100k mutexes
    static constexpr int LOCK_POOL_SIZE = 4096;
    mutable std::mutex node_locks_[LOCK_POOL_SIZE];

    int mode_ = 0; // 0 = Float, 1 = SQ8, 2 = LSH, 3 = Hybrid LSH-SQ8, 4 = Hybrid LSH-Float
    bool is_building_ = false;

    std::vector<float>   data_;
    std::vector<int>     level_;
    // graph_[node][layer] = list of neighbor indices
    std::vector<std::vector<std::vector<int32_t>>> graph_;

    // SQ8 members
    std::vector<uint8_t> quantized_data_;
    float scale_ = 1.0f;
    float offset_ = 0.0f;

    // LSH members
    int n_bits_ = 64;
    std::vector<std::vector<float>> projections_;
    std::vector<uint64_t> signatures_;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    inline float dist_to(int node, const float* q, float threshold = std::numeric_limits<float>::max()) const {
        const float* p = &data_[node * dim_];
        return compute_l2_distance(p, q, dim_, threshold);
    }

    inline float dist_to_float(int node, const float* q) const {
        const float* p = &data_[node * dim_];
        return compute_l2_distance(p, q, dim_);
    }

    uint64_t compute_signature(const float* vec) const {
        uint64_t sig = 0;
        for (int b = 0; b < n_bits_; ++b) {
            float dot = 0.0f;
            for (int d = 0; d < dim_; ++d) {
                dot += vec[d] * projections_[b][d];
            }
            if (dot > 0.0f) {
                sig |= (1ULL << b);
            }
        }
        return sig;
    }

    int random_level(std::mt19937& rng) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        double r = -std::log(uniform(rng) + 1e-10) * ml_;
        return std::min((int)r, 32);   // cap to avoid unbounded levels
    }

    // Get nearest element from a max-heap (drain & rebuild — used sparingly)
    static int nearest_in(MaxHeap& W) {
        std::vector<Pair> tmp;
        while (!W.empty()) { tmp.push_back(W.top()); W.pop(); }
        for (auto& p : tmp) W.push(p);
        return tmp.back().second;   // last = smallest dist after draining max-heap
    }

    // ------------------------------------------------------------------
    // Core: beam search at one layer
    // Returns max-heap of (dist, idx) of size <= ef
    // ------------------------------------------------------------------
    MaxHeap search_layer(const float* q, uint64_t q_sig, int ep, int ef, int layer, bool count, SearchTracker& tracker) const {
        auto& visited_gen = tracker.visited_gen;
        auto& current_gen = tracker.current_gen;

        auto mark_visited = [&](int i) { visited_gen[i] = current_gen; };
        auto is_visited = [&](int i) { return visited_gen[i] == current_gen; };
        auto reset_visited = [&]() {
            ++current_gen;
            if (current_gen == 0) {
                std::fill(visited_gen.begin(), visited_gen.end(), 0);
                current_gen = 1;
            }
        };

        reset_visited();
        mark_visited(ep);

        int cur_mode = mode_;
        if (mode_ == 3) {
            cur_mode = is_building_ ? 2 : 1; // LSH during build, SQ8 during query
        } else if (mode_ == 4) {
            cur_mode = is_building_ ? 2 : 0; // LSH during build, Float during query
        }

        float d_ep;
        if (cur_mode == 1) {
            d_ep = compute_l2_distance_quantized(&quantized_data_[ep * dim_], q, dim_, scale_, offset_);
        } else if (cur_mode == 2) {
            d_ep = static_cast<float>(__builtin_popcountll(q_sig ^ signatures_[ep]));
        } else {
            d_ep = dist_to(ep, q);
        }

        if (count) ++n_distances_;

        MinHeap C;   // candidates to explore (min-heap: nearest first)
        MaxHeap W;   // result window (max-heap: furthest first → easy to prune)
        C.push({d_ep, ep});
        W.push({d_ep, ep});

        while (!C.empty()) {
            auto [d_c, c] = C.top(); C.pop();
            // If closest unexplored candidate is farther than current worst result, stop
            if (d_c > W.top().first) break;

            // Copy node's neighbors under striped lock to prevent concurrent modification data races during build
            std::vector<int32_t> neighbors;
            if (is_building_) {
                std::lock_guard<std::mutex> lock(node_locks_[c % LOCK_POOL_SIZE]);
                neighbors = graph_[c][layer];
            } else {
                neighbors = graph_[c][layer];
            }

            int n_neighbors = neighbors.size();
            for (int idx = 0; idx < n_neighbors; ++idx) {
                int32_t e = neighbors[idx];
                if (is_visited(e)) continue;
                mark_visited(e);

                if (idx + 2 < n_neighbors) {
                    int32_t prefetch_node = neighbors[idx + 2];
                    if (cur_mode == 1) {
                        __builtin_prefetch(&quantized_data_[prefetch_node * dim_], 0, 3);
                    } else {
                        __builtin_prefetch(&data_[prefetch_node * dim_], 0, 3);
                    }
                }

                float d_e = 0.0f;
                if (cur_mode == 1) {
                    float threshold = ((int)W.size() >= ef) ? W.top().first : std::numeric_limits<float>::max();
                    d_e = compute_l2_distance_quantized(&quantized_data_[e * dim_], q, dim_, scale_, offset_, threshold);
                } else if (cur_mode == 2) {
                    d_e = static_cast<float>(__builtin_popcountll(q_sig ^ signatures_[e]));
                } else {
                    float threshold = ((int)W.size() >= ef) ? W.top().first : std::numeric_limits<float>::max();
                    d_e = dist_to(e, q, threshold);
                }

                if (count) ++n_distances_;

                if ((int)W.size() < ef || d_e < W.top().first) {
                    C.push({d_e, e});
                    W.push({d_e, e});
                    if ((int)W.size() > ef) W.pop();   // drop furthest
                }
            }
        }
        return W;
    }

    // ------------------------------------------------------------------
    // Select M_max nearest neighbors from a max-heap (drains the heap)
    // Returns indices sorted nearest-first
    // ------------------------------------------------------------------
    std::vector<int32_t> select_neighbors(MaxHeap& W, int M_max) {
        std::vector<Pair> tmp;
        tmp.reserve(W.size());
        while (!W.empty()) { tmp.push_back(W.top()); W.pop(); }
        // tmp[0] = furthest, tmp[back] = nearest
        int take = std::min((int)tmp.size(), M_max);
        std::vector<int32_t> result(take);
        for (int i = 0; i < take; ++i)
            result[i] = tmp[tmp.size() - 1 - i].second;
        return result;   // result[0] = nearest
    }

    // ------------------------------------------------------------------
    // Prune connections of a node to at most M_max (keep nearest)
    // ------------------------------------------------------------------
    void prune(int node, int layer, int M_max) {
        auto& nbrs = graph_[node][layer];
        if ((int)nbrs.size() <= M_max) return;
        const float* q = &data_[node * dim_];
        uint64_t q_sig = (mode_ == 2 || mode_ == 3 || mode_ == 4) ? signatures_[node] : 0;

        int cur_mode = mode_;
        if (mode_ == 3) {
            cur_mode = is_building_ ? 2 : 1;
        } else if (mode_ == 4) {
            cur_mode = is_building_ ? 2 : 0;
        }

        std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b) {
            if (cur_mode == 1) {
                return compute_l2_distance_quantized(&quantized_data_[a * dim_], q, dim_, scale_, offset_) < 
                       compute_l2_distance_quantized(&quantized_data_[b * dim_], q, dim_, scale_, offset_);
            } else if (cur_mode == 2) {
                return __builtin_popcountll(q_sig ^ signatures_[a]) < __builtin_popcountll(q_sig ^ signatures_[b]);
            } else {
                return dist_to(a, q) < dist_to(b, q);
            }
        });
        nbrs.resize(M_max);
    }

    // ------------------------------------------------------------------
    // Insert one node into the graph
    // ------------------------------------------------------------------
    void insert(int idx, SearchTracker& tracker) {
        int l = random_level(tracker.rng);
        level_[idx] = l;
        graph_[idx].assign(l + 1, {});

        int ep;
        int max_l;
        {
            std::lock_guard<std::mutex> lock(global_lock_);
            ep = entry_point_;
            max_l = max_level_;
        }

        if (ep == -1) {
            std::lock_guard<std::mutex> lock(global_lock_);
            if (entry_point_ == -1) {
                entry_point_ = idx;
                max_level_   = l;
                return;
            }
            ep = entry_point_;
            max_l = max_level_;
        }

        const float* q = &data_[idx * dim_];
        uint64_t q_sig = (mode_ == 2 || mode_ == 3 || mode_ == 4) ? signatures_[idx] : 0;

        // ---- Phase 1: greedy descent from max_l → l+1 ----
        for (int lc = max_l; lc > l; --lc) {
            MaxHeap W = search_layer(q, q_sig, ep, 1, lc, false, tracker);
            ep = nearest_in(W);
        }

        // ---- Phase 2: search & link from min(l, max_l) → 0 ----
        for (int lc = std::min(l, max_l); lc >= 0; --lc) {
            int M_max = (lc == 0) ? Mmax0_ : M_;

            MaxHeap W = search_layer(q, q_sig, ep, ef_construction_, lc, false, tracker);
            ep = nearest_in(W);   // best found so far → entry for next layer

            auto neighbors = select_neighbors(W, M_max);
            graph_[idx][lc] = neighbors;

            for (int32_t nb : neighbors) {
                std::lock_guard<std::mutex> lock(node_locks_[nb % LOCK_POOL_SIZE]);
                graph_[nb][lc].push_back(idx);
                if ((int)graph_[nb][lc].size() > M_max)
                    prune(nb, lc, M_max);
            }
        }

        if (l > max_l) {
            std::lock_guard<std::mutex> lock(global_lock_);
            if (l > max_level_) {
                entry_point_ = idx;
                max_level_   = l;
            }
        }
    }
};

// ---------------------------------------------------------------------------
PYBIND11_MODULE(hnsw_cpp, m) {
    py::class_<HNSWIndex>(m, "HNSWIndex")
        .def(py::init<>())
        .def("fit",   &HNSWIndex::fit,
             py::arg("data"), py::arg("M") = 16, py::arg("ef_construction") = 100, py::arg("mode") = "float")
        .def("query", &HNSWIndex::query,
             py::arg("query"), py::arg("k"), py::arg("ef") = 50, py::arg("refine_r") = -1)
        .def("total_distances_count", &HNSWIndex::total_distances_count)
        .def("reset_distances_count", &HNSWIndex::reset_distances_count);
}
