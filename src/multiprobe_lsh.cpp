#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <atomic>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helpers: generate all signatures at Hamming distance exactly `d` from `sig`
// ---------------------------------------------------------------------------
static void hamming_neighbors(uint32_t sig, int n_bits, int depth,
                               std::vector<uint32_t>& out) {
    // depth 1: flip one bit
    for (int i = 0; i < n_bits; ++i) {
        out.push_back(sig ^ (1U << i));
    }
    if (depth < 2) return;
    // depth 2: flip two bits
    for (int i = 0; i < n_bits; ++i) {
        for (int j = i + 1; j < n_bits; ++j) {
            out.push_back(sig ^ (1U << i) ^ (1U << j));
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-probe LSH index
// ---------------------------------------------------------------------------
class MultiProbeLSH {
public:
    MultiProbeLSH() = default;

    void fit(py::array_t<float> data, int n_tables, int n_bits) {
        py::buffer_info buf = data.request();
        if (buf.ndim != 2)
            throw std::runtime_error("Input must be 2D");

        npts_    = buf.shape[0];
        dim_     = buf.shape[1];
        n_tables_ = n_tables;
        n_bits_   = n_bits;

        float* ptr = static_cast<float*>(buf.ptr);
        data_.assign(ptr, ptr + npts_ * dim_);

        {
            py::gil_scoped_release release;
            // Random projection matrix: (n_tables * n_bits) x dim
            projections_.assign(n_tables_ * n_bits_, std::vector<float>(dim_));
            std::mt19937 gen(42);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (auto& row : projections_)
                for (float& v : row)
                    v = dist(gen);

            // Build hash tables
            tables_.clear();
            tables_.resize(n_tables_);
            for (int i = 0; i < npts_; ++i) {
                const float* pt = &data_[i * dim_];
                for (int t = 0; t < n_tables_; ++t) {
                    uint32_t sig = compute_signature(pt, t);
                    tables_[t][sig].push_back(i);
                }
            }
        }
    }

    // probe_depth: 0 = exact bucket only
    //              1 = exact + Hamming-1  (n_bits extra probes/table)
    //              2 = exact + Hamming-1 + Hamming-2
    py::array_t<int64_t> query(py::array_t<float> query_vec, int k,
                                int probe_depth = 1) {
        py::buffer_info qbuf = query_vec.request();
        if (qbuf.ndim != 1 || (int)qbuf.shape[0] != dim_)
            throw std::runtime_error("Query dimension mismatch");
        const float* q = static_cast<const float*>(qbuf.ptr);

        std::vector<std::pair<float, int32_t>> dists;
        int sort_k = 0;

        {
            py::gil_scoped_release release;
            std::vector<bool> visited(npts_, false);
            std::vector<int32_t> candidates;
            candidates.reserve(npts_ / 8);

            for (int t = 0; t < n_tables_; ++t) {
                uint32_t sig = compute_signature(q, t);

                // Collect all signatures to probe for this table
                std::vector<uint32_t> probes;
                probes.reserve(1 + n_bits_ + n_bits_ * (n_bits_ - 1) / 2);
                probes.push_back(sig);
                if (probe_depth > 0)
                    hamming_neighbors(sig, n_bits_, probe_depth, probes);

                for (uint32_t ps : probes) {
                    auto it = tables_[t].find(ps);
                    if (it == tables_[t].end()) continue;
                    for (int32_t idx : it->second) {
                        if (!visited[idx]) {
                            visited[idx] = true;
                            candidates.push_back(idx);
                        }
                    }
                }
            }

            // Fallback: if not enough candidates, use all points
            if ((int)candidates.size() < k) {
                candidates.resize(npts_);
                std::iota(candidates.begin(), candidates.end(), 0);
            }

            int n_cands = candidates.size();
            n_distances_ += n_cands;

            // Compute distances to candidates
            dists.resize(n_cands);
            for (int i = 0; i < n_cands; ++i) {
                int32_t idx = candidates[i];
                const float* pt = &data_[idx * dim_];
                float s = 0.0f;
                for (int d = 0; d < dim_; ++d) {
                    float diff = pt[d] - q[d];
                    s += diff * diff;
                }
                dists[i] = {s, idx};
            }

            sort_k = std::min(k, n_cands);
            std::partial_sort(dists.begin(), dists.begin() + sort_k, dists.end(),
                [](const std::pair<float,int32_t>& a,
                   const std::pair<float,int32_t>& b){ return a.first < b.first; });
        }

        py::array_t<int64_t> result(sort_k);
        int64_t* res = static_cast<int64_t*>(result.request().ptr);
        for (int i = 0; i < sort_k; ++i)
            res[i] = dists[i].second;

        return result;
    }

    int64_t total_distances_count() const { return n_distances_; }
    void reset_distances_count() { n_distances_ = 0; }

private:
    int npts_ = 0, dim_ = 0, n_tables_ = 0, n_bits_ = 0;
    std::vector<float> data_;
    std::vector<std::vector<float>> projections_;
    std::vector<std::unordered_map<uint32_t, std::vector<int32_t>>> tables_;
    std::atomic<int64_t> n_distances_{0};

    inline uint32_t compute_signature(const float* vec, int t) const {
        uint32_t sig = 0;
        for (int b = 0; b < n_bits_; ++b) {
            int proj_idx = t * n_bits_ + b;
            float dot = 0.0f;
            for (int d = 0; d < dim_; ++d)
                dot += vec[d] * projections_[proj_idx][d];
            if (dot > 0.0f)
                sig |= (1U << b);
        }
        return sig;
    }
};

PYBIND11_MODULE(multiprobe_lsh_cpp, m) {
    py::class_<MultiProbeLSH>(m, "MultiProbeLSH")
        .def(py::init<>())
        .def("fit",   &MultiProbeLSH::fit,
             py::arg("data"), py::arg("n_tables"), py::arg("n_bits"))
        .def("query", &MultiProbeLSH::query,
             py::arg("query"), py::arg("k"), py::arg("probe_depth") = 1)
        .def("total_distances_count", &MultiProbeLSH::total_distances_count)
        .def("reset_distances_count", &MultiProbeLSH::reset_distances_count);
}
