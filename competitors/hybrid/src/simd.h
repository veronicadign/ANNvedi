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


// Per-dimension SQ8 L2 distance (ported from the IVF-LSH fork's quantization):
// each dimension has its own scale, and the query is pre-shifted once per
// search as q_shifted[d] = q[d] - offset[d], so the inner loop is a single
// fused multiply-subtract per element: diff = code[d]*scale[d] - q_shifted[d].
inline float compute_l2_distance_quantized_pd(const uint8_t* a, const float* q_shifted, size_t dim,
                                              const float* scale,
                                              float threshold = std::numeric_limits<float>::max()) {
#if defined(HAS_AVX512)
    __m512 sum_vec = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        __m128i v_bytes = _mm_loadu_si128((const __m128i*)(a + i));
        __m512i v_ints = _mm512_cvtepu8_epi32(v_bytes);
        __m512 v_floats = _mm512_cvtepi32_ps(v_ints);
        __m512 v_scale = _mm512_loadu_ps(scale + i);
        __m512 v_q_shifted = _mm512_loadu_ps(q_shifted + i);
        __m512 diff = _mm512_fmsub_ps(v_floats, v_scale, v_q_shifted);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
        if (i % 96 == 80) {
            float sum = reduce_add_ps_512(sum_vec);
            if (sum >= threshold) return sum;
        }
    }
    float sum = reduce_add_ps_512(sum_vec);
    if (sum >= threshold) return sum;
    for (; i < dim; ++i) {
        float diff = a[i] * scale[i] - q_shifted[i];
        sum += diff * diff;
    }
    return sum;
#elif defined(HAS_AVX)
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < dim; i += 8) {
        uint64_t val;
        std::memcpy(&val, a + i, 8);
        __m128i v_bytes = _mm_cvtsi64_si128(val);
        __m256i v_ints = _mm256_cvtepu8_epi32(v_bytes);
        __m256 v_floats = _mm256_cvtepi32_ps(v_ints);
        __m256 v_scale = _mm256_loadu_ps(scale + i);
        __m256 v_q_shifted = _mm256_loadu_ps(q_shifted + i);
        __m256 diff = _mm256_fmsub_ps(v_floats, v_scale, v_q_shifted);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
        if (i % 96 == 88) {
            alignas(32) float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            if (sum >= threshold) return sum;
        }
    }
    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    if (sum >= threshold) return sum;
    for (; i < dim; ++i) {
        float diff = a[i] * scale[i] - q_shifted[i];
        sum += diff * diff;
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] * scale[i] - q_shifted[i];
        sum += diff * diff;
        if (i % 16 == 15) {
            if (sum >= threshold) return sum;
        }
    }
    return sum;
#endif
}

// Per-dimension SQ4 L2 distance: two dims per byte, 16 levels per dim. Byte j
// of each 16-byte block holds dim (32*blk + j) in the LOW nibble and dim
// (32*blk + 16 + j) in the HIGH nibble, so one 128-bit load unpacks into two
// runs of 16 consecutive dims. dim_pad is padded to a multiple of 32; scale
// and q_shifted are zero-padded there, so the tail contributes exactly 0.
inline float compute_l2_distance_quantized_pd4(const uint8_t* a, const float* q_shifted, size_t dim_pad,
                                               const float* scale,
                                               float threshold = std::numeric_limits<float>::max()) {
#if defined(HAS_AVX512)
    __m512 sum_vec = _mm512_setzero_ps();
    const __m128i nib_mask = _mm_set1_epi8(0x0F);
    for (size_t d = 0; d < dim_pad; d += 32) {
        __m128i v  = _mm_loadu_si128((const __m128i*)(a + d / 2));
        __m128i lo = _mm_and_si128(v, nib_mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), nib_mask);
        __m512 flo = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo));
        __m512 fhi = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi));
        __m512 dlo = _mm512_fmsub_ps(flo, _mm512_loadu_ps(scale + d),      _mm512_loadu_ps(q_shifted + d));
        __m512 dhi = _mm512_fmsub_ps(fhi, _mm512_loadu_ps(scale + d + 16), _mm512_loadu_ps(q_shifted + d + 16));
        sum_vec = _mm512_fmadd_ps(dlo, dlo, sum_vec);
        sum_vec = _mm512_fmadd_ps(dhi, dhi, sum_vec);
        if (d % 96 == 64) {
            float s = reduce_add_ps_512(sum_vec);
            if (s >= threshold) return s;
        }
    }
    return reduce_add_ps_512(sum_vec);
#else
    float sum = 0.0f;
    for (size_t d = 0; d < dim_pad; ++d) {
        size_t blk = d / 32, r = d % 32;
        uint8_t byte = a[blk * 16 + (r < 16 ? r : r - 16)];
        int nib = (r < 16) ? (byte & 0x0F) : (byte >> 4);
        float diff = nib * scale[d] - q_shifted[d];
        sum += diff * diff;
        if (d % 16 == 15 && sum >= threshold) return sum;
    }
    return sum;
#endif
}
