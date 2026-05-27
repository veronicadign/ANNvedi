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

namespace py = pybind11;

// (distance, node_index)
using Pair = std::pair<float, int32_t>;

// Max-heap: top = furthest
using MaxHeap = std::priority_queue<Pair>;
// Min-heap: top = nearest
using MinHeap = std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>>;

// ---------------------------------------------------------------------------
class HNSWIndex {
public:
    HNSWIndex() = default;

    // ------------------------------------------------------------------
    // Build the index
    // ------------------------------------------------------------------
    void fit(py::array_t<float> data, int M, int ef_construction) {
        py::buffer_info buf = data.request();
        if (buf.ndim != 2)
            throw std::runtime_error("Input must be 2D");

        npts_            = (int)buf.shape[0];
        dim_             = (int)buf.shape[1];
        M_               = M;
        Mmax0_           = 2 * M;          // layer-0 gets 2×M edges
        ef_construction_ = ef_construction;
        ml_              = 1.0 / std::log((double)M);  // level multiplier

        float* ptr = static_cast<float*>(buf.ptr);
        data_.assign(ptr, ptr + npts_ * dim_);

        graph_.resize(npts_);
        level_.resize(npts_, 0);
        visited_gen_.assign(npts_, 0);     // O(1) visited reset trick
        current_gen_ = 1;

        entry_point_ = -1;
        max_level_   = -1;
        n_distances_ = 0;

        for (int i = 0; i < npts_; ++i)
            insert(i);
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

        // Greedy descent: layers max_level → 1  (ef=1)
        for (int l = max_level_; l > 0; --l) {
            MaxHeap W = search_layer(q, ep, 1, l, /*count=*/true);
            ep = nearest_in(W);
        }

        // Full beam search at layer 0
        MaxHeap W = search_layer(q, ep, ef, 0, /*count=*/true);

        // Drain max-heap → nearest-first
        std::vector<Pair> results;
        results.reserve(W.size());
        while (!W.empty()) { results.push_back(W.top()); W.pop(); }
        std::reverse(results.begin(), results.end());

        int out_k = std::min(k, (int)results.size());
        py::array_t<int64_t> out(out_k);
        int64_t* out_ptr = static_cast<int64_t*>(out.request().ptr);
        for (int i = 0; i < out_k; ++i)
            out_ptr[i] = results[i].second;

        return out;
    }

    int64_t total_distances_count() const { return n_distances_; }
    void    reset_distances_count()       { n_distances_ = 0; }

private:
    // ------------------------------------------------------------------
    // Data members
    // ------------------------------------------------------------------
    int npts_ = 0, dim_ = 0;
    int M_ = 16, Mmax0_ = 32;
    int ef_construction_ = 100;
    double ml_ = 0.0;
    int max_level_ = -1, entry_point_ = -1;
    mutable int64_t n_distances_ = 0;

    std::vector<float>   data_;
    std::vector<int>     level_;
    // graph_[node][layer] = list of neighbor indices
    std::vector<std::vector<std::vector<int32_t>>> graph_;

    // O(1) visited-set using generation counters (no clearing needed)
    mutable std::vector<uint32_t> visited_gen_;
    mutable uint32_t current_gen_ = 1;

    std::mt19937 rng_{42};
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    inline float dist_to(int node, const float* q) const {
        const float* p = &data_[node * dim_];
        float s = 0.0f;
        for (int d = 0; d < dim_; ++d) {
            float diff = p[d] - q[d];
            s += diff * diff;
        }
        return s;
    }

    inline bool is_visited(int i) const {
        return visited_gen_[i] == current_gen_;
    }
    inline void mark_visited(int i) const {
        visited_gen_[i] = current_gen_;
    }
    inline void reset_visited() const {
        ++current_gen_;
        // Full wrap-around guard (extremely rare)
        if (current_gen_ == 0) {
            std::fill(visited_gen_.begin(), visited_gen_.end(), 0);
            current_gen_ = 1;
        }
    }

    int random_level() {
        double r = -std::log(uniform_(rng_) + 1e-10) * ml_;
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
    MaxHeap search_layer(const float* q, int ep, int ef, int layer, bool count) const {
        reset_visited();
        mark_visited(ep);

        float d_ep = dist_to(ep, q);
        if (count) ++n_distances_;

        MinHeap C;   // candidates to explore (min-heap: nearest first)
        MaxHeap W;   // result window (max-heap: furthest first → easy to prune)
        C.push({d_ep, ep});
        W.push({d_ep, ep});

        while (!C.empty()) {
            auto [d_c, c] = C.top(); C.pop();
            // If closest unexplored candidate is farther than current worst result, stop
            if (d_c > W.top().first) break;

            for (int32_t e : graph_[c][layer]) {
                if (is_visited(e)) continue;
                mark_visited(e);

                float d_e = dist_to(e, q);
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
        std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b) {
            return dist_to(a, q) < dist_to(b, q);
        });
        nbrs.resize(M_max);
    }

    // ------------------------------------------------------------------
    // Insert one node into the graph
    // ------------------------------------------------------------------
    void insert(int idx) {
        int l = random_level();
        level_[idx] = l;
        graph_[idx].assign(l + 1, {});

        if (entry_point_ == -1) {
            entry_point_ = idx;
            max_level_   = l;
            return;
        }

        const float* q = &data_[idx * dim_];
        int ep = entry_point_;

        // ---- Phase 1: greedy descent from max_level → l+1 ----
        for (int lc = max_level_; lc > l; --lc) {
            MaxHeap W = search_layer(q, ep, 1, lc, false);
            ep = nearest_in(W);
        }

        // ---- Phase 2: search & link from min(l, max_level) → 0 ----
        for (int lc = std::min(l, max_level_); lc >= 0; --lc) {
            int M_max = (lc == 0) ? Mmax0_ : M_;

            MaxHeap W = search_layer(q, ep, ef_construction_, lc, false);
            ep = nearest_in(W);   // best found so far → entry for next layer

            auto neighbors = select_neighbors(W, M_max);
            graph_[idx][lc] = neighbors;

            for (int32_t nb : neighbors) {
                graph_[nb][lc].push_back(idx);
                if ((int)graph_[nb][lc].size() > M_max)
                    prune(nb, lc, M_max);
            }
        }

        if (l > max_level_) {
            entry_point_ = idx;
            max_level_   = l;
        }
    }
};

// ---------------------------------------------------------------------------
PYBIND11_MODULE(hnsw_cpp, m) {
    py::class_<HNSWIndex>(m, "HNSWIndex")
        .def(py::init<>())
        .def("fit",   &HNSWIndex::fit,
             py::arg("data"), py::arg("M") = 16, py::arg("ef_construction") = 100)
        .def("query", &HNSWIndex::query,
             py::arg("query"), py::arg("k"), py::arg("ef") = 50)
        .def("total_distances_count", &HNSWIndex::total_distances_count)
        .def("reset_distances_count", &HNSWIndex::reset_distances_count);
}
