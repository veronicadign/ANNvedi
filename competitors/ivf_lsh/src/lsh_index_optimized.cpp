#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <memory>
#include <limits>
#include <chrono>
#include <atomic>
#include "thread_pool.h"
#include "simd.h"

namespace py = pybind11;

// Structs for parallel K-Means
struct KMeansAssignData {
    const float* data;
    const float* centroids;
    int npts;
    int dim;
    int n_clusters;
    int* point_to_cluster;
};

// Thread worker to assign points to their nearest cluster centroids
static void kmeans_assign_worker(int start, int end, int thread_id, void* arg) {
    KMeansAssignData* ud = static_cast<KMeansAssignData*>(arg);
    for (int i = start; i < end; ++i) {
        const float* pt = ud->data + i * ud->dim;
        float min_dist = std::numeric_limits<float>::max();
        int best_cluster = 0;
        for (int c = 0; c < ud->n_clusters; ++c) {
            const float* centroid = ud->centroids + c * ud->dim;
            float dist = compute_l2_distance(pt, centroid, ud->dim);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        ud->point_to_cluster[i] = best_cluster;
    }
}

struct KMeansUpdateData {
    const float* data;
    const int* point_to_cluster;
    int npts;
    int dim;
    int n_clusters;
    float* local_sums;  // shape: num_threads x n_clusters x dim
    int* local_counts;  // shape: num_threads x n_clusters
};

// Thread worker to compute thread-local sums of vector components for centroids update
static void kmeans_update_worker(int start, int end, int thread_id, void* arg) {
    KMeansUpdateData* ud = static_cast<KMeansUpdateData*>(arg);
    float* my_sum = ud->local_sums + thread_id * ud->n_clusters * ud->dim;
    int* my_count = ud->local_counts + thread_id * ud->n_clusters;

    for (int i = start; i < end; ++i) {
        int c = ud->point_to_cluster[i];
        my_count[c]++;
        for (int d = 0; d < ud->dim; ++d) {
            my_sum[c * ud->dim + d] += ud->data[i * ud->dim + d];
        }
    }
}

struct LSHUserData {
    const int32_t* candidates;
    const uint8_t* quantized_data;
    const float* q_shifted;
    int dim;
    int k;
    const float* scale;
    std::pair<float, int32_t>* dists;
};

// Thread worker for parallel L2 distance evaluation on candidate vectors with early pruning
static void lsh_worker(int start, int end, int thread_id, void* arg) {
    LSHUserData* ud = static_cast<LSHUserData*>(arg);
    int k = ud->k;
    const float* scale = ud->scale;
    
    // Allocate a small stack-allocated array to avoid heap allocations inside thread workers
    std::pair<float, int32_t> local_top_k[64];
    int limit_k = std::min(k, 64);
    for (int j = 0; j < limit_k; ++j) {
        local_top_k[j] = {std::numeric_limits<float>::max(), -1};
    }

    for (int i = start; i < end; ++i) {
        if (i + 2 < end) {
            int32_t next_idx = ud->candidates[i + 2];
            __builtin_prefetch(ud->quantized_data + next_idx * ud->dim, 0, 3);
        }
        int32_t idx = ud->candidates[i];
        const uint8_t* pt = ud->quantized_data + idx * ud->dim;
        
        // Threshold is the distance of the worst element in the local top-k
        float threshold = local_top_k[limit_k - 1].first;
        
        float diff_sum = compute_l2_distance_quantized(pt, ud->q_shifted, ud->dim, scale, threshold);
        ud->dists[i] = {diff_sum, idx};
        
        if (diff_sum < threshold) {
            // Insert in sorted order
            int pos = limit_k - 1;
            while (pos > 0 && diff_sum < local_top_k[pos - 1].first) {
                local_top_k[pos] = local_top_k[pos - 1];
                pos--;
            }
            local_top_k[pos] = {diff_sum, idx};
        }
    }
}

class IVFLSHIndex {
public:
    IVFLSHIndex() = default;

    void fit(py::array_t<float> data, int n_tables, int n_bits, int n_clusters = 256) {
        py::buffer_info buf = data.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Input data must be 2D");
        }
        npts_ = buf.shape[0];
        dim_ = buf.shape[1];
        n_tables_ = n_tables;
        n_bits_ = n_bits;
        n_clusters_ = std::min(n_clusters, npts_);

        // Initialize persistent ThreadPool based on hardware capacity
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = 4;
        pool_ = std::make_unique<ThreadPool>(num_threads);

        // Copy dataset data to flat vector for cache locality
        float* ptr = static_cast<float*>(buf.ptr);
        data_.assign(ptr, ptr + (npts_ * dim_));

        {
            py::gil_scoped_release release;
            // Perform parallel K-Means clustering
            std::vector<int> point_to_cluster(npts_, 0);
            run_kmeans(point_to_cluster, num_threads);

            // Group points by cluster to construct reordered layout
            std::vector<std::vector<int32_t>> cluster_to_points(n_clusters_);
            for (int i = 0; i < npts_; ++i) {
                cluster_to_points[point_to_cluster[i]].push_back(i);
            }

            // Reorder data contiguously based on cluster assignment
            original_data_.resize(npts_ * dim_);
            reordered_to_original_.resize(npts_);
            std::vector<int> point_to_cluster_reordered(npts_);

            int new_idx = 0;
            for (int c = 0; c < n_clusters_; ++c) {
                for (int32_t orig_idx : cluster_to_points[c]) {
                    std::copy(data_.begin() + orig_idx * dim_, data_.begin() + (orig_idx + 1) * dim_, original_data_.begin() + new_idx * dim_);
                    reordered_to_original_[new_idx] = orig_idx;
                    point_to_cluster_reordered[new_idx] = c;
                    new_idx++;
                }
            }

            // Calculate mean of training dataset (for centering projections)
            mean_.assign(dim_, 0.0f);
            for (int i = 0; i < npts_; ++i) {
                for (int d = 0; d < dim_; ++d) {
                    mean_[d] += original_data_[i * dim_ + d];
                }
            }
            for (int d = 0; d < dim_; ++d) {
                mean_[d] /= npts_;
            }

            // Calculate dimension-wise quantization scales and offsets (SQ8 compression)
            scale_.resize(dim_);
            offset_.resize(dim_);
            for (int d = 0; d < dim_; ++d) {
                float min_val = std::numeric_limits<float>::max();
                float max_val = std::numeric_limits<float>::lowest();
                for (int i = 0; i < npts_; ++i) {
                    float val = original_data_[i * dim_ + d];
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
                if (max_val - min_val > 1e-8f) {
                    scale_[d] = (max_val - min_val) / 255.0f;
                    offset_[d] = min_val;
                } else {
                    scale_[d] = 1.0f;
                    offset_[d] = 0.0f;
                }
            }

            // Compress reordered dataset to 8-bit bytes
            quantized_data_.resize(original_data_.size());
            for (int i = 0; i < npts_; ++i) {
                for (int d = 0; d < dim_; ++d) {
                    float val = original_data_[i * dim_ + d];
                    int qval = std::round((val - offset_[d]) / scale_[d]);
                    quantized_data_[i * dim_ + d] = static_cast<uint8_t>(std::max(0, std::min(255, qval)));
                }
            }

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

            // Perform Gram-Schmidt orthogonalization per table if n_bits_ <= dim_
            if (n_bits_ <= static_cast<int>(dim_)) {
                for (int t = 0; t < n_tables_; ++t) {
                    for (int b = 0; b < n_bits_; ++b) {
                        int idx = t * n_bits_ + b;
                        
                        // Subtract projections onto all previous orthogonalized vectors in this table
                        for (int b_prev = 0; b_prev < b; ++b_prev) {
                            int idx_prev = t * n_bits_ + b_prev;
                            float dot = 0.0f;
                            for (int d = 0; d < dim_; ++d) {
                                dot += projections_[idx][d] * projections_[idx_prev][d];
                            }
                            for (int d = 0; d < dim_; ++d) {
                                projections_[idx][d] -= dot * projections_[idx_prev][d];
                            }
                        }
                        
                        float norm = 0.0f;
                        for (int d = 0; d < dim_; ++d) {
                            norm += projections_[idx][d] * projections_[idx][d];
                        }
                        norm = std::sqrt(norm);
                        if (norm > 1e-8f) {
                            for (int d = 0; d < dim_; ++d) {
                                projections_[idx][d] /= norm;
                            }
                        }
                    }
                }
            }

            // Build array-backed direct lookup tables: tables_[table][cluster_id][signature]
            int n_buckets = 1 << n_bits_;
            tables_.clear();
            tables_.resize(n_tables_, std::vector<std::vector<std::vector<int32_t>>>(n_clusters_, std::vector<std::vector<int32_t>>(n_buckets)));

            // Add reordered points to the flat array layout
            for (int i = 0; i < npts_; ++i) {
                const float* pt = &original_data_[i * dim_];
                int cluster_id = point_to_cluster_reordered[i];
                for (int t = 0; t < n_tables_; ++t) {
                    uint32_t signature = 0;
                    for (int b = 0; b < n_bits_; ++b) {
                        int proj_idx = t * n_bits_ + b;
                        float dot = 0.0f;
                        for (int d = 0; d < dim_; ++d) {
                            dot += (pt[d] - mean_[d]) * projections_[proj_idx][d];
                        }
                        if (dot > 0.0f) {
                            signature |= (1U << b);
                        }
                    }
                    tables_[t][cluster_id][signature].push_back(i);
                }
            }

            // Initialize visited tracker
            visited_query_ids_.assign(npts_, 0);
            current_query_id_ = 0;

            // Reset profiling
            reset_profile_results();
        }
    }

    struct Perturbation {
        float score;
        int bit1;
        int bit2; // -1 if 1-bit flip

        bool operator<(const Perturbation& other) const {
            return score < other.score;
        }
    };

    py::array_t<int64_t> query(py::array_t<float> query_vec, int k, int n_probes = 0, int n_probe_clusters = 8, int refine_r = -1) {
        auto t_start = std::chrono::high_resolution_clock::now();

        py::buffer_info qbuf = query_vec.request();
        if (qbuf.ndim != 1 || qbuf.shape[0] != dim_) {
            throw std::runtime_error("Query dimension mismatch");
        }
        const float* q = static_cast<const float*>(qbuf.ptr);

        std::vector<std::pair<float, int32_t>> dists;
        int actual_refine_r = 0;
        int sort_k = 0;

        {
            py::gil_scoped_release release;

            std::vector<float> q_shifted(dim_);
            for (int d = 0; d < dim_; ++d) {
                q_shifted[d] = q[d] - offset_[d];
            }

            // 1. Centroid distance checks
            std::vector<std::pair<float, int>> centroid_dists;
            centroid_dists.resize(n_clusters_);
            for (int c = 0; c < n_clusters_; ++c) {
                float dist = compute_l2_distance(q, &centroids_[c * dim_], dim_);
                centroid_dists[c] = {dist, c};
            }
            int active_clusters = std::min(n_probe_clusters, n_clusters_);
            std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + active_clusters, centroid_dists.end());

            auto t_centroids_done = std::chrono::high_resolution_clock::now();
            t_centroids_ += std::chrono::duration<double, std::milli>(t_centroids_done - t_start).count();

            // Thread-local visited tracker for query thread-safety
            thread_local std::vector<uint32_t> local_visited_ids;
            thread_local uint32_t local_current_id = 0;
            if (local_visited_ids.size() != static_cast<size_t>(npts_)) {
                local_visited_ids.assign(npts_, 0);
                local_current_id = 1;
            } else {
                local_current_id++;
                if (local_current_id == 0) {
                    std::fill(local_visited_ids.begin(), local_visited_ids.end(), 0);
                    local_current_id = 1;
                }
            }

            // Collect candidates only from active clusters with a candidate cap
            std::vector<int32_t> candidates;
            candidates.clear();
            candidates.reserve(npts_ / 8);

            int target_refine = (refine_r > 0) ? refine_r : 4 * k;
            size_t max_candidates = std::max(2000, 4 * target_refine);

            std::vector<float> dot_vals;
            dot_vals.resize(n_bits_);

            std::vector<std::pair<float, int>> bit_dist;
            bit_dist.resize(n_bits_);

            std::vector<Perturbation> perturbations;
            perturbations.clear();
            perturbations.reserve(n_bits_ + (n_bits_ * (n_bits_ - 1)) / 2);

            double query_signatures = 0.0;

            for (int cluster_idx = 0; cluster_idx < active_clusters; ++cluster_idx) {
                if (candidates.size() >= max_candidates) break;
                int cluster_id = centroid_dists[cluster_idx].second;
                
                // Construct the base signature for this table and cluster
                for (int t = 0; t < n_tables_; ++t) {
                    if (candidates.size() >= max_candidates) break;
                    auto t_sig_start = std::chrono::high_resolution_clock::now();
                    
                    uint32_t base_signature = 0;
                    for (int b = 0; b < n_bits_; ++b) {
                        int proj_idx = t * n_bits_ + b;
                        float dot = 0.0f;
                        for (int d = 0; d < dim_; ++d) {
                            dot += (q[d] - mean_[d]) * projections_[proj_idx][d];
                        }
                        dot_vals[b] = dot;
                        bit_dist[b] = {std::abs(dot), b};
                        if (dot > 0.0f) {
                            base_signature |= (1U << b);
                        }
                    }
                    
                    auto t_sig_end = std::chrono::high_resolution_clock::now();
                    query_signatures += std::chrono::duration<double, std::milli>(t_sig_end - t_sig_start).count();
                    
                    // Add points in the exact match bucket
                    const auto& bucket = tables_[t][cluster_id][base_signature];
                    for (int32_t idx : bucket) {
                        if (local_visited_ids[idx] != local_current_id) {
                            local_visited_ids[idx] = local_current_id;
                            candidates.push_back(idx);
                        }
                    }
                    
                    // Multi-probe neighborhood expansion (look up near neighbor buckets)
                    if (n_probes > 0) {
                        std::sort(bit_dist.begin(), bit_dist.end());
                        perturbations.clear();
                        
                        // Generate 1-bit flip perturbations
                        for (int i = 0; i < std::min(n_bits_, n_probes); ++i) {
                            perturbations.push_back({bit_dist[i].first, bit_dist[i].second, -1});
                        }
                        
                        // Generate 2-bit flip perturbations
                        for (int i = 0; i < std::min(n_bits_, n_probes); ++i) {
                            for (int j = i + 1; j < std::min(n_bits_, n_probes); ++j) {
                                float combined_score = bit_dist[i].first + bit_dist[j].first;
                                perturbations.push_back({combined_score, bit_dist[i].second, bit_dist[j].second});
                            }
                        }
                        
                        std::sort(perturbations.begin(), perturbations.end());
                        int probe_count = std::min(n_probes, static_cast<int>(perturbations.size()));
                        
                        for (int p = 0; p < probe_count; ++p) {
                            if (candidates.size() >= max_candidates) break;
                            uint32_t probed_signature = base_signature;
                            probed_signature ^= (1U << perturbations[p].bit1);
                            if (perturbations[p].bit2 != -1) {
                                probed_signature ^= (1U << perturbations[p].bit2);
                            }
                            
                            const auto& probe_bucket = tables_[t][cluster_id][probed_signature];
                            for (int32_t idx : probe_bucket) {
                                if (local_visited_ids[idx] != local_current_id) {
                                    local_visited_ids[idx] = local_current_id;
                                    candidates.push_back(idx);
                                }
                            }
                        }
                    }
                }
            }

            t_signatures_ += query_signatures;
            auto t_lookups_done = std::chrono::high_resolution_clock::now();
            t_lookups_ += (std::chrono::duration<double, std::milli>(t_lookups_done - t_centroids_done).count() - query_signatures);

            // Fall back to all points if not enough candidates
            if (candidates.size() < static_cast<size_t>(k)) {
                candidates.clear();
                candidates.resize(npts_);
                std::iota(candidates.begin(), candidates.end(), 0);
            }

            int n_cands = candidates.size();
            n_distances_ += n_cands;

            // Pass 1: Compute coarse 8-bit quantized distances in parallel (enjoys 4x smaller memory bandwidth)
            dists.resize(n_cands);
            LSHUserData ud = { candidates.data(), quantized_data_.data(), q_shifted.data(), dim_, k, scale_.data(), dists.data() };
            if (pool_ && n_cands >= pool_->get_num_threads() * 2) {
                pool_->run_parallel(lsh_worker, &ud, n_cands);
            } else {
                lsh_worker(0, n_cands, 0, &ud);
            }

            // Pass 2: Refine the top R candidates using original 32-bit floats
            target_refine = (refine_r > 0) ? refine_r : 4 * k;
            target_refine = std::max(k, target_refine);
            actual_refine_r = std::min(target_refine, n_cands);

            std::partial_sort(dists.begin(), dists.begin() + actual_refine_r, dists.end(),
                              [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                                  return a.first < b.first;
                              });

            for (int i = 0; i < actual_refine_r; ++i) {
                int32_t idx = dists[i].second;
                const float* pt = &original_data_[idx * dim_];
                float exact_dist = compute_l2_distance(pt, q, dim_);
                dists[i] = {exact_dist, idx};
            }

            auto t_distances_done = std::chrono::high_resolution_clock::now();
            t_distances_ += std::chrono::duration<double, std::milli>(t_distances_done - t_lookups_done).count();

            // Sort the refined top k neighbors
            sort_k = std::min(k, actual_refine_r);
            std::partial_sort(dists.begin(), dists.begin() + sort_k, dists.begin() + actual_refine_r,
                              [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                                  return a.first < b.first;
                              });

            auto t_sort_done = std::chrono::high_resolution_clock::now();
            t_sorting_ += std::chrono::duration<double, std::milli>(t_sort_done - t_distances_done).count();
        }

        // Copy top k and map them back to original indices
        py::array_t<int64_t> result(k);
        py::buffer_info res_buf = result.request();
        int64_t* res_ptr = static_cast<int64_t*>(res_buf.ptr);
        for (int i = 0; i < k; ++i) {
            if (i < sort_k) {
                res_ptr[i] = reordered_to_original_[dists[i].second];
            } else {
                res_ptr[i] = -1;
            }
        }

        query_count_++;
        return result;
    }

    int64_t total_distances_count() const {
        return n_distances_;
    }

    void reset_distances_count() {
        n_distances_ = 0;
    }

    void reset_profile_results() {
        t_centroids_ = 0.0;
        t_signatures_ = 0.0;
        t_lookups_ = 0.0;
        t_distances_ = 0.0;
        t_sorting_ = 0.0;
        query_count_ = 0;
    }

    py::dict get_profile_results() const {
        py::dict d;
        if (query_count_ > 0) {
            d["avg_centroid_search_ms"] = t_centroids_ / query_count_;
            d["avg_signature_calc_ms"] = t_signatures_ / query_count_;
            d["avg_bucket_lookup_ms"] = t_lookups_ / query_count_;
            d["avg_distance_calc_ms"] = t_distances_ / query_count_;
            d["avg_candidate_sort_ms"] = t_sorting_ / query_count_;
            d["query_count"] = query_count_;
        } else {
            d["query_count"] = 0;
        }
        return d;
    }

private:
    void run_kmeans(std::vector<int>& point_to_cluster, int num_threads) {
        // Initialize centroids by choosing random distinct points
        std::vector<int> rand_indices(npts_);
        std::iota(rand_indices.begin(), rand_indices.end(), 0);
        std::shuffle(rand_indices.begin(), rand_indices.end(), std::mt19937(1337));

        centroids_.resize(n_clusters_ * dim_);
        for (int c = 0; c < n_clusters_; ++c) {
            int pt_idx = rand_indices[c % npts_];
            std::copy(data_.begin() + pt_idx * dim_, data_.begin() + (pt_idx + 1) * dim_, centroids_.begin() + c * dim_);
        }

        // Run for 15 iterations (sufficient for cluster partitioning)
        const int max_iterations = 15;
        std::vector<float> local_sums(num_threads * n_clusters_ * dim_);
        std::vector<int> local_counts(num_threads * n_clusters_);

        KMeansAssignData assign_data = { data_.data(), centroids_.data(), npts_, dim_, n_clusters_, point_to_cluster.data() };
        KMeansUpdateData update_data = { data_.data(), point_to_cluster.data(), npts_, dim_, n_clusters_, local_sums.data(), local_counts.data() };

        for (int iter = 0; iter < max_iterations; ++iter) {
            // Step 1: Assign points to nearest centroids in parallel
            if (pool_) {
                pool_->run_parallel(kmeans_assign_worker, &assign_data, npts_);
            } else {
                kmeans_assign_worker(0, npts_, 0, &assign_data);
            }

            // Step 2: Clear local thread variables
            std::fill(local_sums.begin(), local_sums.end(), 0.0f);
            std::fill(local_counts.begin(), local_counts.end(), 0);

            // Step 3: Compute thread-local sums of vector components
            if (pool_) {
                pool_->run_parallel(kmeans_update_worker, &update_data, npts_);
            } else {
                kmeans_update_worker(0, npts_, 0, &update_data);
            }

            // Step 4: Combine local thread statistics to update centroids
            std::vector<float> new_centroids(n_clusters_ * dim_, 0.0f);
            std::vector<int> global_counts(n_clusters_, 0);

            for (int t = 0; t < num_threads; ++t) {
                for (int c = 0; c < n_clusters_; ++c) {
                    global_counts[c] += local_counts[t * n_clusters_ + c];
                    for (int d = 0; d < dim_; ++d) {
                        new_centroids[c * dim_ + d] += local_sums[(t * n_clusters_ + c) * dim_ + d];
                    }
                }
            }

            // Update centroid coordinates
            for (int c = 0; c < n_clusters_; ++c) {
                if (global_counts[c] > 0) {
                    for (int d = 0; d < dim_; ++d) {
                        centroids_[c * dim_ + d] = new_centroids[c * dim_ + d] / global_counts[c];
                    }
                } else {
                    // Reinitialize empty clusters
                    int pt_idx = rand_indices[(c * 37) % npts_];
                    std::copy(data_.begin() + pt_idx * dim_, data_.begin() + (pt_idx + 1) * dim_, centroids_.begin() + c * dim_);
                }
            }
        }
    }

private:
    int npts_ = 0;
    int dim_ = 0;
    int n_tables_ = 0;
    int n_bits_ = 0;
    int n_clusters_ = 0;

    std::vector<float> data_;
    std::vector<std::vector<float>> projections_;
    std::vector<float> centroids_;
    
    // 4D array-backed lookup table representation
    std::vector<std::vector<std::vector<std::vector<int32_t>>>> tables_;
    
    // Original 32-bit floats and Quantized 8-bit bytes
    std::vector<float> original_data_;
    std::vector<uint8_t> quantized_data_;
    std::vector<float> scale_;
    std::vector<float> offset_;
    std::vector<float> mean_;
    
    // Mapping from reordered contiguous indices back to original indices
    std::vector<int32_t> reordered_to_original_;
    
    std::atomic<int64_t> n_distances_{0};

    std::vector<uint32_t> visited_query_ids_;
    uint32_t current_query_id_ = 0;

    std::unique_ptr<ThreadPool> pool_;

    // Timing profile variables (in milliseconds)
    double t_centroids_ = 0.0;
    double t_signatures_ = 0.0;
    double t_lookups_ = 0.0;
    double t_distances_ = 0.0;
    double t_sorting_ = 0.0;
    int query_count_ = 0;
};

PYBIND11_MODULE(ivf_lsh_cpp, m) {
    py::class_<IVFLSHIndex>(m, "LSHIndex")
        .def(py::init<>())
        .def("fit", &IVFLSHIndex::fit, py::arg("data"), py::arg("n_tables"), py::arg("n_bits"), py::arg("n_clusters") = 256)
        .def("query", &IVFLSHIndex::query, py::arg("query"), py::arg("k"), py::arg("n_probes") = 0, py::arg("n_probe_clusters") = 8, py::arg("refine_r") = -1)
        .def("total_distances_count", &IVFLSHIndex::total_distances_count)
        .def("reset_distances_count", &IVFLSHIndex::reset_distances_count)
        .def("reset_profile_results", &IVFLSHIndex::reset_profile_results)
        .def("get_profile_results", &IVFLSHIndex::get_profile_results);
}
