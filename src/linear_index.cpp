#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

using namespace std;

namespace py = pybind11;

class LinearIndex {
private:
    vector<vector<float>> data_;    
    size_t npts_ = 0;
    size_t dim_ = 0;
    size_t total_distances_ = 0;

public:
    LinearIndex() = default;

    void fit(py::array_t<float> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("Input deve essere un array 2D");
        npts_ = buf.shape[0];
        dim_ = buf.shape[1];
        float* ptr = static_cast<float*>(buf.ptr);
        data_.resize(npts_);
        for (size_t i = 0; i < npts_; ++i) {
            data_[i].assign(ptr + i * dim_, ptr + (i + 1) * dim_);
        }
    }

    py::array_t<int64_t> query(py::array_t<float> q, int k) {
        if (dim_ == 0 || npts_ == 0)
            throw runtime_error("Indice non addestrato");

        py::buffer_info qbuf = q.request();
        if (qbuf.ndim != 1 || qbuf.shape[0] != dim_)
            throw std::runtime_error("Query deve essere un vettore 1D di dimensione corretta");

        float* q_ptr = static_cast<float*>(qbuf.ptr);

        vector<pair<float, int>> dist_id(npts_);
        for (int i = 0; i < npts_; ++i) {
            float sum = 0.0f;
            for (int d = 0; d < dim_; ++d) {
                float diff = data_[i][d] - q_ptr[d];
                sum += diff * diff; 
            }
            dist_id[i] = {sum, i};
        }

        total_distances_ += npts_;

        if (k > npts_) k = npts_;
        partial_sort(dist_id.begin(), dist_id.begin() + k, dist_id.end());

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