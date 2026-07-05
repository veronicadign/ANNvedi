#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <thread>
#include <memory>
#include <atomic>
#include "thread_pool.h"
#include "simd.h"

using namespace std;

namespace py = pybind11;

struct LinearUserData {
    const float* data;
    const float* q;
    size_t dim;
    int k;
    std::pair<float, int>* dist_id;
};

// Thread worker function for computing L2 distances in parallel with early pruning
static void linear_worker(int start, int end, int thread_id, void* arg) {
    LinearUserData* ud = static_cast<LinearUserData*>(arg);
    int k = ud->k;
    
    // Allocate a small stack-allocated array to avoid heap allocations inside thread workers
    std::pair<float, int> local_top_k[64];
    int limit_k = std::min(k, 64);
    for (int j = 0; j < limit_k; ++j) {
        local_top_k[j] = {std::numeric_limits<float>::max(), -1};
    }

    for (int i = start; i < end; ++i) {
        if (i + 2 < end) {
            __builtin_prefetch(ud->data + (i + 2) * ud->dim, 0, 3);
        }
        const float* pt = ud->data + i * ud->dim;
        
        // Threshold is the distance of the worst element in the local top-k
        float threshold = local_top_k[limit_k - 1].first;
        
        float sum = compute_l2_distance(pt, ud->q, ud->dim, threshold);
        ud->dist_id[i] = {sum, i};
        
        if (sum < threshold) {
            // Insert in sorted order
            int pos = limit_k - 1;
            while (pos > 0 && sum < local_top_k[pos - 1].first) {
                local_top_k[pos] = local_top_k[pos - 1];
                pos--;
            }
            local_top_k[pos] = {sum, i};
        }
    }
}

class LinearIndex {
private:
    vector<float> data_;    
    size_t npts_ = 0;
    size_t dim_ = 0;
    std::atomic<size_t> total_distances_{0};
    std::unique_ptr<ThreadPool> pool_;

public:
    LinearIndex() = default;

    void fit(py::array_t<float> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("Input deve essere un array 2D");
        npts_ = buf.shape[0];
        dim_ = buf.shape[1];
        float* ptr = static_cast<float*>(buf.ptr);
        
        // Copy to flat vector for optimal cache locality
        data_.assign(ptr, ptr + (npts_ * dim_));

        // Initialize persistent ThreadPool based on hardware capacity
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = 4;
        pool_ = std::make_unique<ThreadPool>(num_threads);
    }

    py::array_t<int64_t> query(py::array_t<float> q, int k) {
        if (dim_ == 0 || npts_ == 0)
            throw runtime_error("Indice non addestrato");

        py::buffer_info qbuf = q.request();
        if (qbuf.ndim != 1 || qbuf.shape[0] != dim_)
            throw std::runtime_error("Query deve essere un vettore 1D di dimensione corretta");

        float* q_ptr = static_cast<float*>(qbuf.ptr);

        vector<pair<float, int>> dist_id(npts_);
        {
            py::gil_scoped_release release;
            // Parallel computation of distances
            LinearUserData ud = { data_.data(), q_ptr, dim_, k, dist_id.data() };
            // Force sequential execution to measure single-threaded latency
            linear_worker(0, npts_, 0, &ud);

            total_distances_ += npts_;

            if (k > static_cast<int>(npts_)) k = npts_;
            partial_sort(dist_id.begin(), dist_id.begin() + k, dist_id.end());
        }

        py::array_t<int64_t> result_arr(k);
        py::buffer_info res_buf = result_arr.request();
        int64_t* res_ptr = static_cast<int64_t*>(res_buf.ptr);
        for (int i = 0; i < k; ++i)
            res_ptr[i] = dist_id[i].second;

        return result_arr;
    }

    int64_t total_distances_count() const { return total_distances_; }
};

PYBIND11_MODULE(linear_ann_cpp, m) {
    py::class_<LinearIndex>(m, "LinearIndex")
        .def(py::init<>())
        .def("fit", &LinearIndex::fit)
        .def("query", &LinearIndex::query)
        .def("total_distances_count", &LinearIndex::total_distances_count);
}