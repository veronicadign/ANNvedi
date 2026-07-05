#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__AVX512F__)
#include <immintrin.h>
#define HAS_AVX512 1
#elif defined(__AVX__)
#include <immintrin.h>
#define HAS_AVX 1
#endif
#endif

#include <cstddef>
#include <limits>
#include <cstdint>
#include <cstring>
#include <algorithm>

#if defined(HAS_AVX512)
// Custom robust reduction helper to avoid compiler horizontal reduction optimization bugs under -ffast-math
inline float reduce_add_ps_512(const __m512& v) {
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(low, high);
    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, sum256);
    return temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
}
#endif

// Inline L2 (Euclidean) distance computation with AVX-512, AVX, or fallback loops, and threshold-based early pruning.
inline float compute_l2_distance(const float* a, const float* b, size_t dim, float threshold = std::numeric_limits<float>::max()) {
#if defined(HAS_AVX512)
    __m512 sum_vec = _mm512_setzero_ps();
    size_t i = 0;
    
    // Process 16 floats at a time
    for (; i + 15 < dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum_vec = _mm512_add_ps(sum_vec, _mm512_mul_ps(diff, diff));
        
        // Amortize reduction latency by checking threshold pruning every 96 floats
        if (i % 96 == 80) {
            float sum = reduce_add_ps_512(sum_vec);
            if (sum >= threshold) {
                return sum; // Early terminate!
            }
        }
    }
    
    float sum = reduce_add_ps_512(sum_vec);
    if (sum >= threshold) {
        return sum;
    }
    
    // Process remainder elements
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return sum;
#elif defined(HAS_AVX)
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    
    // Process 8 floats at a time
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(diff, diff));
        
        // Check threshold pruning every 96 floats (12 iterations of 8 floats)
        if (i % 96 == 88) {
            alignas(32) float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            if (sum >= threshold) {
                return sum;
            }
        }
    }
    
    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    if (sum >= threshold) {
        return sum;
    }
    
    // Process remainder elements
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    return sum;
#else
    // Fallback standard loop with periodic threshold checks
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
        if (i % 16 == 15) {
            if (sum >= threshold) {
                return sum;
            }
        }
    }
    return sum;
#endif
}

// Compute L2 distance between 8-bit quantized candidate vector and float query vector on the fly.
// Original version for HNSW/other modules
inline float compute_l2_distance_quantized(const uint8_t* a, const float* b, size_t dim, float scale, float offset, float threshold = std::numeric_limits<float>::max()) {
#if defined(HAS_AVX512)
    __m512 sum_vec = _mm512_setzero_ps();
    __m512 v_scale = _mm512_set1_ps(scale);
    __m512 v_offset = _mm512_set1_ps(offset);
    size_t i = 0;
    
    // Process 16 bytes at a time
    for (; i + 15 < dim; i += 16) {
        __m128i v_bytes = _mm_loadu_si128((const __m128i*)(a + i));
        __m512i v_ints = _mm512_cvtepu8_epi32(v_bytes);
        __m512 v_floats = _mm512_cvtepi32_ps(v_ints);
        // Dequantize: v_floats * v_scale + v_offset
        v_floats = _mm512_fmadd_ps(v_floats, v_scale, v_offset);
        
        __m512 v_b = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(v_floats, v_b);
        sum_vec = _mm512_add_ps(sum_vec, _mm512_mul_ps(diff, diff));
        
        if (i % 96 == 80) {
            float sum = reduce_add_ps_512(sum_vec);
            if (sum >= threshold) {
                return sum;
            }
        }
    }
    
    float sum = reduce_add_ps_512(sum_vec);
    if (sum >= threshold) {
        return sum;
    }
    
    // Process remainder
    for (; i < dim; ++i) {
        float dequant = a[i] * scale + offset;
        float diff = dequant - b[i];
        sum += diff * diff;
    }
    
    return sum;
#elif defined(HAS_AVX)
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256 v_offset = _mm256_set1_ps(offset);
    size_t i = 0;
    
    // Process 8 bytes at a time
    for (; i + 7 < dim; i += 8) {
        uint64_t val;
        std::memcpy(&val, a + i, 8);
        __m128i v_bytes = _mm_cvtsi64_si128(val);
        __m256i v_ints = _mm256_cvtepu8_epi32(v_bytes);
        __m256 v_floats = _mm256_cvtepi32_ps(v_ints);
        v_floats = _mm256_add_ps(_mm256_mul_ps(v_floats, v_scale), v_offset);
        
        __m256 v_b = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(v_floats, v_b);
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(diff, diff));
        
        if (i % 96 == 88) {
            alignas(32) float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            if (sum >= threshold) {
                return sum;
            }
        }
    }
    
    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    if (sum >= threshold) {
        return sum;
    }
    
    // Process remainder
    for (; i < dim; ++i) {
        float dequant = a[i] * scale + offset;
        float diff = dequant - b[i];
        sum += diff * diff;
    }
    
    return sum;
#else
    // Fallback standard dequantizing distance loop
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float dequant = a[i] * scale + offset;
        float diff = dequant - b[i];
        sum += diff * diff;
        if (i % 16 == 15) {
            if (sum >= threshold) {
                return sum;
            }
        }
    }
    return sum;
#endif
}

// Shifted-Query optimized version for optimized LSH
inline float compute_l2_distance_quantized_shifted(const uint8_t* a, const float* q_shifted, size_t dim, float scale, float threshold = std::numeric_limits<float>::max()) {
#if defined(HAS_AVX512)
    __m512 sum_vec = _mm512_setzero_ps();
    __m512 v_scale = _mm512_set1_ps(scale);
    size_t i = 0;
    
    // Process 16 bytes at a time
    for (; i + 15 < dim; i += 16) {
        __m128i v_bytes = _mm_loadu_si128((const __m128i*)(a + i));
        __m512i v_ints = _mm512_cvtepu8_epi32(v_bytes);
        __m512 v_floats = _mm512_cvtepi32_ps(v_ints);
        
        __m512 v_q_shifted = _mm512_loadu_ps(q_shifted + i);
        // diff = v_floats * v_scale - v_q_shifted
        __m512 diff = _mm512_fmsub_ps(v_floats, v_scale, v_q_shifted);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
        
        if (i % 96 == 80) {
            float sum = reduce_add_ps_512(sum_vec);
            if (sum >= threshold) {
                return sum;
            }
        }
    }
    
    float sum = reduce_add_ps_512(sum_vec);
    if (sum >= threshold) {
        return sum;
    }
    
    // Process remainder
    for (; i < dim; ++i) {
        float diff = a[i] * scale - q_shifted[i];
        sum += diff * diff;
    }
    
    return sum;
#elif defined(HAS_AVX)
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 v_scale = _mm256_set1_ps(scale);
    size_t i = 0;
    
    // Process 8 bytes at a time
    for (; i + 7 < dim; i += 8) {
        uint64_t val;
        std::memcpy(&val, a + i, 8);
        __m128i v_bytes = _mm_cvtsi64_si128(val);
        __m256i v_ints = _mm256_cvtepu8_epi32(v_bytes);
        __m256 v_floats = _mm256_cvtepi32_ps(v_ints);
        
        __m256 v_q_shifted = _mm256_loadu_ps(q_shifted + i);
        // diff = v_floats * v_scale - v_q_shifted
        __m256 diff = _mm256_fmsub_ps(v_floats, v_scale, v_q_shifted);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
        
        if (i % 96 == 88) {
            alignas(32) float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            if (sum >= threshold) {
                return sum;
            }
        }
    }
    
    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    if (sum >= threshold) {
        return sum;
    }
    
    // Process remainder
    for (; i < dim; ++i) {
        float diff = a[i] * scale - q_shifted[i];
        sum += diff * diff;
    }
    
    return sum;
#else
    // Fallback standard dequantizing distance loop
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] * scale - q_shifted[i];
        sum += diff * diff;
        if (i % 16 == 15) {
            if (sum >= threshold) {
                return sum;
            }
        }
    }
    return sum;
#endif
}
