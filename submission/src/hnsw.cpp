#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include "simd.h"

namespace py = pybind11;

// Distance representation used by build and search. Exhaustive: fit() rejects
// any other mode string, so no code path ever sees an out-of-enum value.
enum class Mode {
    Float,  // exact float L2
    SQ8,    // 8-bit codes, one global [min,max] scale
    SQ8PD,  // 8-bit codes, per-dimension scales (ported from the IVF-LSH fork)
};

// (distance, node_index)
using Pair = std::pair<float, int32_t>;

// Max-heap: top = furthest
using MaxHeap = std::priority_queue<Pair>;
// Min-heap: top = nearest
using MinHeap = std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>>;

// Once a beam search finishes, the heap ORDER of its result window is no
// longer needed (callers scan or re-sort), so the underlying vector is moved
// out instead of popped element by element (each pop re-heapifies at
// O(log n)). Pointer-to-member is the legal way to reach the protected
// container of std::priority_queue.
struct HeapAccess : MaxHeap {
    static std::vector<Pair> take(MaxHeap&& w) { return std::move(w.*&HeapAccess::c); }
};

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
    void fit(py::array_t<float> data, int M, int ef_construction, const std::string& mode = "float",
             bool heuristic = false) {
        py::buffer_info buf = data.request();
        if (buf.ndim != 2 || buf.shape[0] == 0)
            throw std::runtime_error("Input must be a non-empty 2D array");

        npts_            = (int)buf.shape[0];
        dim_             = (int)buf.shape[1];
        M_               = M;
        Mmax0_           = 2 * M;          // layer-0 gets 2×M edges
        ef_construction_ = ef_construction;
        ml_              = 1.0 / std::log((double)M);  // level multiplier
        heuristic_       = heuristic;

        if (mode == "sq8") {
            mode_ = Mode::SQ8;
        } else if (mode == "sq8pd") {
            mode_ = Mode::SQ8PD;
        } else if (mode == "float") {
            mode_ = Mode::Float;
        } else {
            // the experimental lsh/hybrid modes live only in the dev repo
            throw std::runtime_error("unknown mode '" + mode + "' (use float, sq8 or sq8pd)");
        }

        float* ptr = static_cast<float*>(buf.ptr);
        data_.assign(ptr, ptr + npts_ * dim_);

        // Global-scale SQ8 quantization
        if (mode_ == Mode::SQ8) {
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

        // Per-dimension SQ8 (mode "sq8pd", ported from the IVF-LSH fork): each
        // dimension gets its own [min,max] range, so quantization error follows
        // the data's per-axis spread instead of the single global extreme.
        if (mode_ == Mode::SQ8PD) {
            pd_scale_.resize(dim_);
            pd_offset_.resize(dim_);
            for (int d = 0; d < dim_; ++d) {
                float min_val = std::numeric_limits<float>::max();
                float max_val = std::numeric_limits<float>::lowest();
                for (int i = 0; i < npts_; ++i) {
                    float val = data_[(size_t)i * dim_ + d];
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
                if (max_val - min_val > 1e-8f) {
                    pd_scale_[d] = (max_val - min_val) / 255.0f;
                    pd_offset_[d] = min_val;
                } else {
                    pd_scale_[d] = 1.0f;
                    pd_offset_[d] = 0.0f;
                }
            }
            quantized_data_.resize(data_.size());
            for (int i = 0; i < npts_; ++i) {
                for (int d = 0; d < dim_; ++d) {
                    float val = data_[(size_t)i * dim_ + d];
                    int qval = (int)std::round((val - pd_offset_[d]) / pd_scale_[d]);
                    quantized_data_[(size_t)i * dim_ + d] = static_cast<uint8_t>(std::max(0, std::min(255, qval)));
                }
            }
        }

        graph_.resize(npts_);
        n_distances_.store(0);

        is_building_ = true;

        // Insert the first node sequentially: fit() guarantees npts_ >= 1, so
        // every worker thread starts with a valid entry point already set.
        {
            std::mt19937 rng(42);
            int l0 = random_level(rng);
            graph_[0].assign(l0 + 1, {});
            entry_point_ = 0;
            max_level_   = l0;
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
    py::array_t<int64_t> query(py::array_t<float> query_arr, int k, int ef) {
        py::buffer_info qbuf = query_arr.request();
        if (qbuf.ndim != 1 || (int)qbuf.shape[0] != dim_)
            throw std::runtime_error("Query dimension mismatch");

        const float* q = static_cast<const float*>(qbuf.ptr);
        ef = std::max(ef, k);

        int ep = entry_point_;
        int max_l=max_level_;

        SearchTracker tracker;
        tracker.visited_gen.assign(npts_, 0);

        // Layer-0 candidate window, exact-sorted nearest-first
        std::vector<Pair> results;

        {
            py::gil_scoped_release release;
            // Greedy descent: layers max_level → 1  (ef=1 → the window holds
            // exactly the single nearest node found at this layer)
            for (int l = max_l; l > 0; --l) {
                auto W = search_layer(q, ep, 1, l, /*count=*/true, tracker);
                ep = W.front().second;
            }

            // Full beam search at layer 0 (unordered window, size <= ef)
            std::vector<Pair> W = search_layer(q, ep, ef, 0, /*count=*/true, tracker);

            if (mode_ == Mode::Float) {
                results = std::move(W);   // distances already exact
            } else {
                // Quantized modes: rerank the whole candidate window with exact floats
                results.reserve(W.size());
                for (const auto& p : W)
                    results.push_back({dist_to(p.second, q), p.second});
            }
            std::sort(results.begin(), results.end());
        }

        int out_k = std::min(k, (int)results.size());
        py::array_t<int64_t> out(out_k);
        int64_t* out_ptr = static_cast<int64_t*>(out.request().ptr);
        for (int i = 0; i < out_k; ++i)
            out_ptr[i] = results[i].second;

        return out;
    }

    int64_t total_distances_count() const { return n_distances_.load(); }

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

    Mode mode_ = Mode::Float;
    bool is_building_ = false;
    bool heuristic_ = false;  // diversity-based neighbor selection (Alg. 4);
                              // every shipped scenario enables it — at full scale
                              // it is worth ~+0.05 recall at equal ef

    std::vector<float>   data_;
    // graph_[node][layer] = list of neighbor indices
    std::vector<std::vector<std::vector<int32_t>>> graph_;

    // SQ8 members
    std::vector<uint8_t> quantized_data_;
    float scale_ = 1.0f;
    float offset_ = 0.0f;
    // per-dimension SQ8 (mode 2)
    std::vector<float> pd_scale_;
    std::vector<float> pd_offset_;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    inline float dist_to(int node, const float* q, float threshold = std::numeric_limits<float>::max()) const {
        const float* p = &data_[node * dim_];
        return compute_l2_distance(p, q, dim_, threshold);
    }

    int random_level(std::mt19937& rng) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        double r = -std::log(uniform(rng) + 1e-10) * ml_;
        return std::min((int)r, 32);   // cap to avoid unbounded levels
    }

    // ------------------------------------------------------------------
    // Core: beam search at one layer
    // Returns the candidate window as an UNORDERED vector of (dist, idx),
    // size in [1, ef] — it always contains at least the entry point.
    // ------------------------------------------------------------------
    std::vector<Pair> search_layer(const float* q, int ep, int ef, int layer, bool count, SearchTracker& tracker) const {
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

        // Per-dim SQ8: shift the query once per search so the hot loop is a
        // single fmsub per element. thread_local → safe under the parallel build.
        const float* q_shifted = nullptr;
        if (mode_ == Mode::SQ8PD) {
            static thread_local std::vector<float> q_shift_buf;
            q_shift_buf.resize(dim_);
            for (int d = 0; d < dim_; ++d)
                q_shift_buf[d] = q[d] - pd_offset_[d];
            q_shifted = q_shift_buf.data();
        }

        float d_ep;
        if (mode_ == Mode::SQ8) {
            d_ep = compute_l2_distance_quantized(&quantized_data_[ep * dim_], q, dim_, scale_, offset_);
        } else if (mode_ == Mode::SQ8PD) {
            d_ep = compute_l2_distance_quantized_pd(&quantized_data_[ep * dim_], q_shifted, dim_, pd_scale_.data());
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

            // During build: copy the node's neighbors under the striped lock to
            // prevent data races with concurrent inserts. At query time the
            // graph is immutable — read it in place (the copy per hop was a
            // measurable allocation cost on the hot path).
            std::vector<int32_t> neighbors_copy;
            if (is_building_) {
                std::lock_guard<std::mutex> lock(node_locks_[c % LOCK_POOL_SIZE]);
                neighbors_copy = graph_[c][layer];
            }
            const std::vector<int32_t>& neighbors = is_building_ ? neighbors_copy : graph_[c][layer];

            int n_neighbors = neighbors.size();
            for (int idx = 0; idx < n_neighbors; ++idx) {
                int32_t e = neighbors[idx];
                if (is_visited(e)) continue;
                mark_visited(e);

                if (idx + 2 < n_neighbors) {
                    int32_t prefetch_node = neighbors[idx + 2];
                    if (mode_ != Mode::Float) {
                        __builtin_prefetch(&quantized_data_[prefetch_node * dim_], 0, 3);
                    } else {
                        __builtin_prefetch(&data_[prefetch_node * dim_], 0, 3);
                    }
                }

                float threshold = ((int)W.size() >= ef) ? W.top().first : std::numeric_limits<float>::max();
                float d_e;
                if (mode_ == Mode::SQ8) {
                    d_e = compute_l2_distance_quantized(&quantized_data_[e * dim_], q, dim_, scale_, offset_, threshold);
                } else if (mode_ == Mode::SQ8PD) {
                    d_e = compute_l2_distance_quantized_pd(&quantized_data_[e * dim_], q_shifted, dim_, pd_scale_.data(), threshold);
                } else {
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
        return HeapAccess::take(std::move(W));
    }

    // ------------------------------------------------------------------
    // Neighbor selection core.
    //
    // Input: candidates as (dist_to_base, idx), sorted nearest-first.
    //
    // heuristic_=false  -> keep the M_max nearest (original behaviour).
    // heuristic_=true   -> diversity pruning (Malkov & Yashunin, Alg. 4):
    //   keep a candidate only if it is closer to the base node than to every
    //   neighbor kept so far. Dominated edges (both endpoints on the same
    //   side of the base) are redundant for navigation, so dropping them
    //   keeps the graph SPARSE in dense regions while spending the edge
    //   budget on long/diverse links -> better navigability than picking
    //   the closest (near-random within a cluster) nodes. Discarded
    //   candidates back-fill leftover slots (keepPrunedConnections).
    //
    // All heuristic distances use exact float vectors (data_ is always
    // retained), so the rule stays valid in the quantized build modes too.
    // ------------------------------------------------------------------
    std::vector<int32_t> select_from_sorted(const std::vector<Pair>& cand, int M_max) const {
        std::vector<int32_t> selected;
        selected.reserve(M_max);
        if (!heuristic_ || (int)cand.size() <= M_max) {
            for (const auto& p : cand) {
                if ((int)selected.size() >= M_max) break;
                selected.push_back(p.second);
            }
            return selected;
        }

        std::vector<int32_t> discarded;
        for (const auto& [d_base, e] : cand) {
            if ((int)selected.size() >= M_max) break;
            const float* pe = &data_[e * dim_];
            bool diverse = true;
            for (int32_t s : selected) {
                // early-exit threshold: we only care whether d(e,s) < d(e,base)
                float d_es = compute_l2_distance(&data_[s * dim_], pe, dim_, d_base);
                if (d_es < d_base) { diverse = false; break; }
            }
            if (diverse) selected.push_back(e);
            else         discarded.push_back(e);
        }
        // keepPrunedConnections: top up with the nearest discarded candidates
        for (size_t i = 0; i < discarded.size() && (int)selected.size() < M_max; ++i)
            selected.push_back(discarded[i]);
        return selected;
    }

    // Sort a search_layer candidate window nearest-first and run neighbor
    // selection on it. Window distances are float or SQ8-approx L2 — both on
    // the L2 scale the diversity rule expects.
    std::vector<int32_t> select_neighbors(std::vector<Pair> cand, int M_max) {
        std::sort(cand.begin(), cand.end());   // nearest-first (build metric)
        return select_from_sorted(cand, M_max);
    }

    // ------------------------------------------------------------------
    // Prune connections of a node to at most M_max.
    // Uses the same selection core as insertion (diversity heuristic when
    // enabled). Distances are computed ONCE up front — the previous version
    // recomputed them inside the sort comparator (O(n log n) distance
    // evaluations per prune) and always kept the plain nearest.
    // ------------------------------------------------------------------
    void prune(int node, int layer, int M_max) {
        auto& nbrs = graph_[node][layer];
        if ((int)nbrs.size() <= M_max) return;
        const float* q = &data_[node * dim_];

        std::vector<Pair> cand;
        cand.reserve(nbrs.size());
        for (int32_t e : nbrs)
            cand.push_back({compute_l2_distance(&data_[e * dim_], q, dim_), e});
        std::sort(cand.begin(), cand.end());   // nearest-first

        nbrs = select_from_sorted(cand, M_max);
    }

    // ------------------------------------------------------------------
    // Insert one node into the graph
    // ------------------------------------------------------------------
    void insert(int idx, SearchTracker& tracker) {
        int l = random_level(tracker.rng);
        graph_[idx].assign(l + 1, {});

        // Entry point is always valid: node 0 is inserted before threads start.
        int ep;
        int max_l;
        {
            std::lock_guard<std::mutex> lock(global_lock_);
            ep = entry_point_;
            max_l = max_level_;
        }

        const float* q = &data_[idx * dim_];

        // ---- Phase 1: greedy descent from max_l → l+1 (ef=1 → single result) ----
        for (int lc = max_l; lc > l; --lc) {
            auto W = search_layer(q, ep, 1, lc, false, tracker);
            ep = W.front().second;
        }

        // ---- Phase 2: search & link from min(l, max_l) → 0 ----
        for (int lc = std::min(l, max_l); lc >= 0; --lc) {
            int M_max = (lc == 0) ? Mmax0_ : M_;

            auto W = search_layer(q, ep, ef_construction_, lc, false, tracker);

            // select_neighbors sorts the window nearest-first, and both
            // selection branches always keep the nearest candidate at
            // position 0 — so neighbors[0] is the best node of this layer.
            auto neighbors = select_neighbors(std::move(W), M_max);
            ep = neighbors[0];   // entry point for the next layer down
            {
                // idx is already discoverable at higher layers, so another
                // thread can be reading graph_[idx][lc] (it copies under this
                // same striped lock) while we assign it — an unlocked
                // move-assign here is a torn-vector data race (manifested as
                // "malloc(): unaligned tcache chunk" crashes at full scale).
                std::lock_guard<std::mutex> lock(node_locks_[idx % LOCK_POOL_SIZE]);
                graph_[idx][lc] = neighbors;
            }

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
             py::arg("data"), py::arg("M") = 16, py::arg("ef_construction") = 100, py::arg("mode") = "float",
             py::arg("heuristic") = false)
        .def("query", &HNSWIndex::query,
             py::arg("query"), py::arg("k"), py::arg("ef") = 100)
        .def("total_distances_count", &HNSWIndex::total_distances_count);
}
