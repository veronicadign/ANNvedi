// GPU quantized-filter / CPU exact-rerank contestant.
//
// fit:   per-dim SQ8 quantize on CPU, transpose to dim-major, one bulk H2D.
// query: H2D shifted query (KBs) -> coalesced SQ8 scan over ALL points on
//        GPU -> top-R (CUB radix sort) -> D2H R ids (KBs) -> exact float
//        rerank on the CPU from host RAM -> top-k.
//
// Distance accounting: only FULL Euclidean computations count per the rules,
// which here is exactly R per query (the GPU scan is quantized/sketch work).
//
// Exposed as a plain C ABI for ctypes. The train pointer passed to gf_create
// must stay valid for the handle's lifetime (algorithm.py keeps the numpy
// array alive) — no duplicate float copy, which keeps host RSS lean.
//
// Compiled with PTX forward-compat (compute_80) so the image can be built on
// a GPU-less machine and JIT on any newer GPU (incl. Blackwell) at first run.

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CUDA_TRY(x)                                                   \
    do {                                                              \
        cudaError_t err = (x);                                        \
        if (err != cudaSuccess) {                                     \
            snprintf(g_err, sizeof(g_err), "CUDA error: %s at %s:%d", \
                     cudaGetErrorString(err), __FILE__, __LINE__);    \
            return nullptr;                                           \
        }                                                             \
    } while (0)

#define CUDA_TRY_INT(x)                                               \
    do {                                                              \
        cudaError_t err = (x);                                        \
        if (err != cudaSuccess) {                                     \
            snprintf(g_err, sizeof(g_err), "CUDA error: %s at %s:%d", \
                     cudaGetErrorString(err), __FILE__, __LINE__);    \
            return -1;                                                \
        }                                                             \
    } while (0)

static char g_err[512] = "";

__constant__ float c_qshift[4096];
__constant__ float c_scale[4096];

__global__ void sq8pd_scan(const uint8_t* __restrict__ codes_t, int n, int dim,
                           float* __restrict__ dist, int* __restrict__ idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float v = fmaf((float)codes_t[(size_t)d * n + i], c_scale[d], -c_qshift[d]);
        acc = fmaf(v, v, acc);
    }
    dist[i] = acc;
    idx[i] = i;
}

struct Handle {
    long n = 0, dim = 0;
    const float* train_host = nullptr;   // borrowed; owner is algorithm.py
    std::vector<float> scale, offset;
    uint8_t* d_codes = nullptr;
    float *d_dist = nullptr, *d_dist_sorted = nullptr;
    int *d_idx = nullptr, *d_idx_sorted = nullptr;
    void* d_tmp = nullptr;
    size_t tmp_bytes = 0;
    long long n_distances = 0;
};

extern "C" const char* gf_last_error() { return g_err; }

extern "C" void* gf_create(const float* train, long n, long dim) {
    g_err[0] = 0;
    if (dim > 4096) {
        snprintf(g_err, sizeof(g_err), "dim %ld > 4096 unsupported", dim);
        return nullptr;
    }
    int ndev = 0;
    cudaError_t derr = cudaGetDeviceCount(&ndev);
    if (derr != cudaSuccess || ndev == 0) {
        snprintf(g_err, sizeof(g_err),
                 "no CUDA device visible (%s) — was the container started with --gpus?",
                 derr == cudaSuccess ? "0 devices" : cudaGetErrorString(derr));
        return nullptr;
    }

    Handle* h = new Handle();
    h->n = n;
    h->dim = dim;
    h->train_host = train;

    // per-dim SQ8 (identical to the validated sq8pd scheme)
    h->scale.assign(dim, 1.0f);
    h->offset.assign(dim, 0.0f);
    std::vector<float> mn(dim, 3.4e38f), mx(dim, -3.4e38f);
    for (long i = 0; i < n; ++i) {
        const float* p = train + i * dim;
        for (long d = 0; d < dim; ++d) {
            mn[d] = std::min(mn[d], p[d]);
            mx[d] = std::max(mx[d], p[d]);
        }
    }
    for (long d = 0; d < dim; ++d) {
        if (mx[d] - mn[d] > 1e-8f) {
            h->scale[d] = (mx[d] - mn[d]) / 255.0f;
            h->offset[d] = mn[d];
        } else {
            h->scale[d] = 1.0f;
            h->offset[d] = mn[d];
        }
    }
    // quantize straight into dim-major layout
    std::vector<uint8_t> codes_t((size_t)n * dim);
    for (long i = 0; i < n; ++i) {
        const float* p = train + i * dim;
        for (long d = 0; d < dim; ++d) {
            int q = (int)lrintf((p[d] - h->offset[d]) / h->scale[d]);
            codes_t[(size_t)d * n + i] = (uint8_t)std::max(0, std::min(255, q));
        }
    }

    CUDA_TRY(cudaMalloc(&h->d_codes, (size_t)n * dim));
    CUDA_TRY(cudaMemcpy(h->d_codes, codes_t.data(), (size_t)n * dim, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMalloc(&h->d_dist, n * sizeof(float)));
    CUDA_TRY(cudaMalloc(&h->d_dist_sorted, n * sizeof(float)));
    CUDA_TRY(cudaMalloc(&h->d_idx, n * sizeof(int)));
    CUDA_TRY(cudaMalloc(&h->d_idx_sorted, n * sizeof(int)));
    cub::DeviceRadixSort::SortPairs(h->d_tmp, h->tmp_bytes, h->d_dist, h->d_dist_sorted,
                                    h->d_idx, h->d_idx_sorted, (int)n);
    CUDA_TRY(cudaMalloc(&h->d_tmp, h->tmp_bytes));
    CUDA_TRY(cudaMemcpyToSymbol(c_scale, h->scale.data(), dim * sizeof(float)));
    return h;
}

extern "C" int gf_query(void* handle, const float* q, int k, int r, long long* out_ids) {
    Handle* h = (Handle*)handle;
    const long n = h->n, dim = h->dim;
    if (r < k) r = k;
    if (r > n) r = (int)n;

    std::vector<float> qshift(dim);
    for (long d = 0; d < dim; ++d) qshift[d] = q[d] - h->offset[d];
    CUDA_TRY_INT(cudaMemcpyToSymbol(c_qshift, qshift.data(), dim * sizeof(float)));

    int threads = 256, blocks = (int)((n + threads - 1) / threads);
    sq8pd_scan<<<blocks, threads>>>(h->d_codes, (int)n, (int)dim, h->d_dist, h->d_idx);
    cub::DeviceRadixSort::SortPairs(h->d_tmp, h->tmp_bytes, h->d_dist, h->d_dist_sorted,
                                    h->d_idx, h->d_idx_sorted, (int)n);
    std::vector<int> cand(r);
    CUDA_TRY_INT(cudaMemcpy(cand.data(), h->d_idx_sorted, r * sizeof(int), cudaMemcpyDeviceToHost));

    // exact float rerank on the CPU — these are the only full Euclidean
    // distance computations in the pipeline
    std::vector<std::pair<float, int>> exact(r);
    for (int j = 0; j < r; ++j) {
        const float* p = h->train_host + (size_t)cand[j] * dim;
        float acc = 0.0f;
        for (long d = 0; d < dim; ++d) {
            float v = p[d] - q[d];
            acc += v * v;
        }
        exact[j] = {acc, cand[j]};
    }
    h->n_distances += r;
    int out_k = std::min(k, r);
    std::partial_sort(exact.begin(), exact.begin() + out_k, exact.end());
    for (int j = 0; j < out_k; ++j) out_ids[j] = exact[j].second;
    return out_k;
}

extern "C" long long gf_ndist(void* handle) { return ((Handle*)handle)->n_distances; }

extern "C" void gf_free(void* handle) {
    Handle* h = (Handle*)handle;
    if (!h) return;
    cudaFree(h->d_codes);
    cudaFree(h->d_dist);
    cudaFree(h->d_dist_sorted);
    cudaFree(h->d_idx);
    cudaFree(h->d_idx_sorted);
    cudaFree(h->d_tmp);
    delete h;
}
