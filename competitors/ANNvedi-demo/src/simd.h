#pragma once

#include <immintrin.h>
#include <cstddef>

#include <limits>
#include <cstdint>
#include <cstring>

#include <algorithm>

inline float reduce_add_ps_512(const __m512& v) {
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(low, high);
    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, sum256);
    return temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
}

// Inline L2 (Euclidean) distance computation with AVX-512, AVX, or fallback loops, and threshold-based early pruning.
inline float compute_l2_distance(const float* a, const float* b, size_t dim, float threshold = std::numeric_limits<float>::max()) {
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
}

// Compute L2 distance between 8-bit quantized candidate vector and float query vector on the fly.
inline float compute_l2_distance_quantized(const uint8_t* a, const float* b, size_t dim, float scale, float offset, float threshold = std::numeric_limits<float>::max()) {
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
}
