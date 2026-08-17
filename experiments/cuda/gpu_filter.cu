// GPU-filter / CPU-refine test program for the g7.2xlarge final round.
//
// Pipeline per query (one query at a time, per competition rules):
//   1. H2D: the query's shifted form (dim floats — a few KB)
//   2. GPU: brute-force per-dimension SQ8 distances over ALL points
//      (codes live on the GPU, dim-major for coalescing; 1/4 the bytes of float)
//   3. GPU: top-R selection (CUB radix sort on (dist, id) pairs — simple v1;
//      a fused block-local selector can cut this stage later)
//   4. D2H: R candidate ids (KBs)
//   5. CPU: exact float rerank of the R candidates from host RAM, return top-k
//
// Recall is evaluator-style against the dataset's true distances.
//
// Build (on the g7 / any NVIDIA box, CUDA >= 12.8 for Blackwell):
//   nvcc -O3 -arch=native -o gpu_filter gpu_filter.cu
// Run:
//   python3 prepare_data.py ../../dataset/yahoo-minilm-public.hdf5 /tmp/gf
//   ./gpu_filter /tmp/gf 1000    # data dir, R
//
// Files read from the data dir (written by prepare_data.py):
//   meta.txt  : N dim n_queries k
//   codes.u8  : N*dim uint8, per-dim SQ8 codes, point-major
//   scale.f32 / offset.f32 : dim floats each
//   train.f32 : N*dim float32 (host-side exact rerank)
//   queries.f32 : n_queries*dim float32
//   gt.f32    : n_queries*k true distances (evaluator recall)

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                                 \
    do {                                                                              \
        cudaError_t err = (x);                                                        \
        if (err != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err),      \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

// Query data broadcast to all threads: constant memory is ideal (all threads
// read the same element each iteration). 4096 dims covers every dataset.
__constant__ float c_qshift[4096];
__constant__ float c_scale[4096];

// codes are DIM-MAJOR: codes_t[d * N + i] = code of dim d, point i, so that
// consecutive threads (points) read consecutive bytes — fully coalesced.
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

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static std::vector<uint8_t> read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <data_dir> [R=1000]\n", argv[0]); return 1; }
    std::string dir = argv[1];
    int R = argc > 2 ? atoi(argv[2]) : 1000;

    int n, dim, n_queries, k;
    {
        FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
        if (!f || fscanf(f, "%d %d %d %d", &n, &dim, &n_queries, &k) != 4) { fprintf(stderr, "bad meta\n"); return 1; }
        fclose(f);
    }
    printf("N=%d dim=%d queries=%d k=%d R=%d\n", n, dim, n_queries, k, R);
    if (dim > 4096) { fprintf(stderr, "dim > 4096 unsupported\n"); return 1; }

    auto codes = read_file(dir + "/codes.u8");
    auto scale_b = read_file(dir + "/scale.f32");
    auto offset_b = read_file(dir + "/offset.f32");
    auto train_b = read_file(dir + "/train.f32");
    auto queries_b = read_file(dir + "/queries.f32");
    auto gt_b = read_file(dir + "/gt.f32");
    const float* scale = (const float*)scale_b.data();
    const float* offset = (const float*)offset_b.data();
    const float* train = (const float*)train_b.data();
    const float* queries = (const float*)queries_b.data();
    const float* gt = (const float*)gt_b.data();

    // Transpose codes to dim-major once (host), then one bulk H2D — this is
    // the entire "pass the data to the GPU" cost, charged to build time.
    auto t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> codes_t((size_t)n * dim);
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < dim; ++d)
            codes_t[(size_t)d * n + i] = codes[(size_t)i * dim + d];
    uint8_t* d_codes;
    CUDA_CHECK(cudaMalloc(&d_codes, (size_t)n * dim));
    CUDA_CHECK(cudaMemcpy(d_codes, codes_t.data(), (size_t)n * dim, cudaMemcpyHostToDevice));
    printf("transpose+upload (build-time cost): %.0f ms (%zu MB on GPU)\n",
           ms_since(t0), (size_t)n * dim >> 20);

    float *d_dist, *d_dist_sorted;
    int *d_idx, *d_idx_sorted;
    CUDA_CHECK(cudaMalloc(&d_dist, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist_sorted, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idx_sorted, n * sizeof(int)));
    void* d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_dist, d_dist_sorted, d_idx, d_idx_sorted, n);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));

    std::vector<float> qshift(dim);
    std::vector<int> cand(R);
    std::vector<double> total_ms(n_queries), scan_ms(n_queries), sort_ms(n_queries), rerank_ms(n_queries);
    double recall_sum = 0.0;

    CUDA_CHECK(cudaMemcpyToSymbol(c_scale, scale, dim * sizeof(float)));  // constant across queries

    int threads = 256, blocks = (n + threads - 1) / threads;
    cudaEvent_t ev0, ev1, ev2;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2);

    for (int qi = 0; qi < n_queries; ++qi) {
        auto tq = std::chrono::steady_clock::now();
        const float* q = queries + (size_t)qi * dim;
        for (int d = 0; d < dim; ++d) qshift[d] = q[d] - offset[d];
        CUDA_CHECK(cudaMemcpyToSymbol(c_qshift, qshift.data(), dim * sizeof(float)));

        cudaEventRecord(ev0);
        sq8pd_scan<<<blocks, threads>>>(d_codes, n, dim, d_dist, d_idx);
        cudaEventRecord(ev1);
        cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_dist, d_dist_sorted, d_idx, d_idx_sorted, n);
        cudaEventRecord(ev2);
        CUDA_CHECK(cudaMemcpy(cand.data(), d_idx_sorted, R * sizeof(int), cudaMemcpyDeviceToHost));

        float ms01, ms12;
        cudaEventElapsedTime(&ms01, ev0, ev1);
        cudaEventElapsedTime(&ms12, ev1, ev2);
        scan_ms[qi] = ms01;
        sort_ms[qi] = ms12;

        // CPU exact rerank of R candidates (float data never left the host)
        auto tr = std::chrono::steady_clock::now();
        std::vector<std::pair<float, int>> exact(R);
        for (int j = 0; j < R; ++j) {
            const float* p = train + (size_t)cand[j] * dim;
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float v = p[d] - q[d];
                acc += v * v;
            }
            exact[j] = {acc, cand[j]};
        }
        std::partial_sort(exact.begin(), exact.begin() + k, exact.end());
        rerank_ms[qi] = ms_since(tr);
        total_ms[qi] = ms_since(tq);

        // evaluator-style recall: sqrt(dist) <= gt[k-1]
        const float* g = gt + (size_t)qi * k;
        int hits = 0;
        for (int j = 0; j < k; ++j)
            if (sqrtf(exact[j].first) <= g[k - 1] + 1e-6f) ++hits;
        recall_sum += (double)hits / k;
    }

    auto stat = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        return std::pair<double, double>(v[v.size() / 2], v[v.size() * 99 / 100]);
    };
    auto [tot_med, tot_p99] = stat(total_ms);
    auto [scan_med, _s] = stat(scan_ms);
    auto [sort_med, _o] = stat(sort_ms);
    auto [rr_med, _r] = stat(rerank_ms);
    printf("\nrecall@%d = %.4f  (R=%d)\n", k, recall_sum / n_queries, R);
    printf("per query: total median=%.3f ms (p99 %.3f) -> %.0f qps\n", tot_med, tot_p99, 1000.0 / tot_med);
    printf("  gpu scan  %.3f ms | gpu top-R sort %.3f ms | cpu rerank %.3f ms | rest = copies/launch\n",
           scan_med, sort_med, rr_med);
    return 0;
}
