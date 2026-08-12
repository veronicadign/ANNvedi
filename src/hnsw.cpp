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
    void fit(py::array_t<float> data, int M, int ef_construction, const std::string& mode = "float",
             bool heuristic = false) {
        py::buffer_info buf = data.request();
        if (buf.ndim != 2)
            throw std::runtime_error("Input must be 2D");

        npts_            = (int)buf.shape[0];
        dim_             = (int)buf.shape[1];
        M_               = M;
        Mmax0_           = 2 * M;          // layer-0 gets 2×M edges
        ef_construction_ = ef_construction;
        ml_              = 1.0 / std::log((double)M);  // level multiplier
        heuristic_       = heuristic;

        if (mode == "sq8") {
            mode_ = 1;
        } else if (mode == "sq8pd") {
            mode_ = 5;
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

        // Per-dimension SQ8 (mode "sq8pd", ported from the IVF-LSH fork): each
        // dimension gets its own [min,max] range, so quantization error follows
        // the data's per-axis spread instead of the single global extreme.
        if (mode_ == 5) {
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

    int mode_ = 0; // 0 = Float, 1 = SQ8, 2 = LSH, 3 = Hybrid LSH-SQ8, 4 = Hybrid LSH-Float, 5 = per-dim SQ8
    bool is_building_ = false;
    bool heuristic_ = false;  // diversity-based neighbor selection; default OFF —
                              // at the competition's k=100 recall saturates and the
                              // extra edges only cost QPS/build time (docs/TUNING.md)

    std::vector<float>   data_;
    std::vector<int>     level_;
    // graph_[node][layer] = list of neighbor indices
    std::vector<std::vector<std::vector<int32_t>>> graph_;

    // SQ8 members
    std::vector<uint8_t> quantized_data_;
    float scale_ = 1.0f;
    float offset_ = 0.0f;
    // per-dimension SQ8 (mode 5)
    std::vector<float> pd_scale_;
    std::vector<float> pd_offset_;

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

        // Per-dim SQ8: shift the query once per search so the hot loop is a
        // single fmsub per element. thread_local → safe under the parallel build.
        const float* q_shifted = nullptr;
        if (cur_mode == 5) {
            static thread_local std::vector<float> q_shift_buf;
            q_shift_buf.resize(dim_);
            for (int d = 0; d < dim_; ++d)
                q_shift_buf[d] = q[d] - pd_offset_[d];
            q_shifted = q_shift_buf.data();
        }

        float d_ep;
        if (cur_mode == 1) {
            d_ep = compute_l2_distance_quantized(&quantized_data_[ep * dim_], q, dim_, scale_, offset_);
        } else if (cur_mode == 5) {
            d_ep = compute_l2_distance_quantized_pd(&quantized_data_[ep * dim_], q_shifted, dim_, pd_scale_.data());
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
                    if (cur_mode == 1 || cur_mode == 5) {
                        __builtin_prefetch(&quantized_data_[prefetch_node * dim_], 0, 3);
                    } else {
                        __builtin_prefetch(&data_[prefetch_node * dim_], 0, 3);
                    }
                }

                float d_e = 0.0f;
                if (cur_mode == 1) {
                    float threshold = ((int)W.size() >= ef) ? W.top().first : std::numeric_limits<float>::max();
                    d_e = compute_l2_distance_quantized(&quantized_data_[e * dim_], q, dim_, scale_, offset_, threshold);
                } else if (cur_mode == 5) {
                    float threshold = ((int)W.size() >= ef) ? W.top().first : std::numeric_limits<float>::max();
                    d_e = compute_l2_distance_quantized_pd(&quantized_data_[e * dim_], q_shifted, dim_, pd_scale_.data(), threshold);
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
    // retained), so the rule stays valid in sq8/lsh build modes too.
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

    // Drain a search_layer result heap into candidates (nearest-first) and
    // run neighbor selection on them.
    std::vector<int32_t> select_neighbors(MaxHeap& W, int M_max, const float* base_q) {
        std::vector<Pair> cand;
        cand.reserve(W.size());
        while (!W.empty()) { cand.push_back(W.top()); W.pop(); }
        std::reverse(cand.begin(), cand.end());   // nearest-first (build metric)

        if (heuristic_) {
            int cur_mode = mode_;
            if (mode_ == 3 || mode_ == 4) cur_mode = 2;   // lsh-hybrid builds search on hamming
            if (cur_mode == 2) {
                // Hamming heap distances are not comparable with the float
                // pair-distances used by the diversity rule: recompute in
                // float and re-sort. (float and sq8 heap distances are
                // already on the L2 scale — no recompute needed.)
                for (auto& p : cand)
                    p.first = compute_l2_distance(&data_[p.second * dim_], base_q, dim_);
                std::sort(cand.begin(), cand.end());
            }
        }
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

            auto neighbors = select_neighbors(W, M_max, q);
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
             py::arg("data"), py::arg("M") = 16, py::arg("ef_construction") = 100, py::arg("mode") = "float",
             py::arg("heuristic") = false)
        .def("query", &HNSWIndex::query,
             py::arg("query"), py::arg("k"), py::arg("ef") = 50, py::arg("refine_r") = -1)
        .def("total_distances_count", &HNSWIndex::total_distances_count)
        .def("reset_distances_count", &HNSWIndex::reset_distances_count);
}
