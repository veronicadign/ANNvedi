#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace py = pybind11;

class LSHIndex {
public:
    LSHIndex() = default;

    void fit(py::array_t<float> data, int n_tables, int n_bits) {
        py::buffer_info buf = data.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Input data must be 2D");
        }
        npts_ = buf.shape[0];
        dim_ = buf.shape[1];
        n_tables_ = n_tables;
        n_bits_ = n_bits;

        // Copy dataset data to flat vector
        float* ptr = static_cast<float*>(buf.ptr);
        data_.assign(ptr, ptr + (npts_ * dim_));

        // Initialize random projections with fixed seed for reproducibility
        projections_.clear();
        projections_.resize(n_tables_ * n_bits_, std::vector<float>(dim_));
        
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0, 1.0);
        for (int i = 0; i < n_tables_ * n_bits_; ++i) {
            for (int d = 0; d < dim_; ++d) {
                projections_[i][d] = dist(gen);
            }
        }

        // Initialize hash tables
        tables_.clear();
        tables_.resize(n_tables_);

        // For each point, calculate its signatures and add to tables
        for (int i = 0; i < npts_; ++i) {
            const float* pt = &data_[i * dim_];
            for (int t = 0; t < n_tables_; ++t) {
                uint32_t signature = 0;
                for (int b = 0; b < n_bits_; ++b) {
                    int proj_idx = t * n_bits_ + b;
                    float dot = 0.0f;
                    for (int d = 0; d < dim_; ++d) {
                        dot += pt[d] * projections_[proj_idx][d];
                    }
                    if (dot > 0.0f) {
                        signature |= (1U << b);
                    }
                }
                tables_[t][signature].push_back(i);
            }
        }
    }

    py::array_t<int64_t> query(py::array_t<float> query_vec, int k, int n_probes = 0, int n_probe_clusters = 8, int refine_r = -1) {
        py::buffer_info qbuf = query_vec.request();
        if (qbuf.ndim != 1 || qbuf.shape[0] != dim_) {
            throw std::runtime_error("Query dimension mismatch");
        }
        const float* q = static_cast<const float*>(qbuf.ptr);

        // Find candidate set across all tables
        std::vector<int32_t> candidates;
        candidates.reserve(npts_ / 4);

        // Flat boolean mask for O(1) visited checks
        std::vector<bool> visited(npts_, false);

        for (int t = 0; t < n_tables_; ++t) {
            uint32_t signature = 0;
            for (int b = 0; b < n_bits_; ++b) {
                int proj_idx = t * n_bits_ + b;
                float dot = 0.0f;
                for (int d = 0; d < dim_; ++d) {
                    dot += q[d] * projections_[proj_idx][d];
                }
                if (dot > 0.0f) {
                    signature |= (1U << b);
                }
            }

            auto it = tables_[t].find(signature);
            if (it != tables_[t].end()) {
                for (int32_t idx : it->second) {
                    if (!visited[idx]) {
                        visited[idx] = true;
                        candidates.push_back(idx);
                    }
                }
            }
        }

        // Fall back to all points if not enough candidates
        if (candidates.size() < static_cast<size_t>(k)) {
            candidates.resize(npts_);
            std::iota(candidates.begin(), candidates.end(), 0);
        }

        int n_cands = candidates.size();
        n_distances_ += n_cands;

        // Compute Euclidean distance to all candidates
        std::vector<std::pair<float, int32_t>> dists(n_cands);
        for (int i = 0; i < n_cands; ++i) {
            int32_t idx = candidates[i];
            const float* pt = &data_[idx * dim_];
            float diff_sum = 0.0f;
            for (int d = 0; d < dim_; ++d) {
                float diff = pt[d] - q[d];
                diff_sum += diff * diff;
            }
            dists[i] = {diff_sum, idx};
        }

        // Sort the top k candidates by distance
        int sort_k = std::min(k, n_cands);
        std::partial_sort(dists.begin(), dists.begin() + sort_k, dists.end(),
                          [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                              return a.first < b.first;
                          });

        // Copy top k to returned py::array_t<int64_t>
        py::array_t<int64_t> result(sort_k);
        py::buffer_info res_buf = result.request();
        int64_t* res_ptr = static_cast<int64_t*>(res_buf.ptr);
        for (int i = 0; i < sort_k; ++i) {
            res_ptr[i] = dists[i].second;
        }

        return result;
    }

    int64_t total_distances_count() const {
        return n_distances_;
    }

    void reset_distances_count() {
        n_distances_ = 0;
    }

private:
    int npts_ = 0;
    int dim_ = 0;
    int n_tables_ = 0;
    int n_bits_ = 0;
    std::vector<float> data_;
    std::vector<std::vector<float>> projections_;
    std::vector<std::unordered_map<uint32_t, std::vector<int32_t>>> tables_;
    int64_t n_distances_ = 0;
};

PYBIND11_MODULE(lsh_cpp_module, m) {
    py::class_<LSHIndex>(m, "LSHIndex")
        .def(py::init<>())
        .def("fit", &LSHIndex::fit, py::arg("data"), py::arg("n_tables"), py::arg("n_bits"))
        .def("query", &LSHIndex::query, py::arg("query"), py::arg("k"), py::arg("n_probes") = 0, py::arg("n_probe_clusters") = 8, py::arg("refine_r") = -1)
        .def("total_distances_count", &LSHIndex::total_distances_count)
        .def("reset_distances_count", &LSHIndex::reset_distances_count);
}
