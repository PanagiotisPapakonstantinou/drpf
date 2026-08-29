#pragma once

#include <vector>
#include <numeric>
#include <memory>
#include <cmath>
#include <random>
#include <iostream>
#include <Eigen/Dense>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

#if defined(__SSE3__) && !defined(__AVX__)
#include <pmmintrin.h>
#endif

namespace drpf
{

    // -----------------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------------

    /**
     * @brief Overload for printing std::vector to streams.
     * Useful for debugging and logging vector contents in format [a, b, c].
     */
    std::ostream &operator<<(std::ostream &os, const std::vector<int> &vec)
    {
        os << "[";
        if (!vec.empty())
        {
            os << vec[0];
            for (std::size_t i = 1; i < vec.size(); ++i)
            {
                os << ", " << vec[i];
            }
        }
        os << "]";
        return os;
    }

    // -----------------------------------------------------------------------------
    // SIMD Optimized Squared Euclidean Distance
    // -----------------------------------------------------------------------------
    /*
     * Computes sum((a[i] - b[i])^2).
     * Selects the highest available instruction set at compile time.
     */

#if defined(__AVX512F__)

    /**
     * @brief AVX-512 implementation of squared distance.
     * Uses 512-bit registers (16 floats per operation).
     * Features loop unrolling and mask loading for safe tail handling.
     */
    inline float squaredDistance(const float *__restrict a,
                                 const float *__restrict b,
                                 int size) noexcept
    {
        // Two accumulators to break instruction dependency chains for better pipeline throughput
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();

        int i = 0;

        // Main loop: Process 32 floats per iteration (2 x 16-float registers)
        for (; i + 32 <= size; i += 32)
        {
            // Block 1
            __m512 va0 = _mm512_loadu_ps(a + i);
            __m512 vb0 = _mm512_loadu_ps(b + i);
            __m512 diff0 = _mm512_sub_ps(va0, vb0);
            acc0 = _mm512_fmadd_ps(diff0, diff0, acc0); // Fused Multiply-Add: acc += diff * diff

            // Block 2
            __m512 va1 = _mm512_loadu_ps(a + i + 16);
            __m512 vb1 = _mm512_loadu_ps(b + i + 16);
            __m512 diff1 = _mm512_sub_ps(va1, vb1);
            acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);
        }

        // Cleanup loop: Process remaining multiples of 16
        for (; i + 16 <= size; i += 16)
        {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 diff = _mm512_sub_ps(va, vb);
            acc0 = _mm512_fmadd_ps(diff, diff, acc0);
        }

        // Tail handling: Use AVX-512 masks to handle remaining < 16 elements
        // This avoids the need for a scalar fallback loop or buffer padding.
        int remaining = size - i;
        if (remaining > 0)
        {
            const __mmask16 mask = (1u << remaining) - 1;   // Create mask for valid bits
            __m512 va = _mm512_maskz_loadu_ps(mask, a + i); // Zero-masked load
            __m512 vb = _mm512_maskz_loadu_ps(mask, b + i);
            __m512 diff = _mm512_sub_ps(va, vb);
            acc0 = _mm512_fmadd_ps(diff, diff, acc0);
        }

        // Horizontal reduction: Sum all lanes
        __m512 sum = _mm512_add_ps(acc0, acc1);
        return _mm512_reduce_add_ps(sum);
    }

#elif defined(__AVX__) && defined(__FMA__)

    /**
     * @brief AVX2 + FMA implementation.
     * Uses 256-bit registers (8 floats).
     */
    inline float squaredDistance(const float *__restrict a,
                                 const float *__restrict b,
                                 int size) noexcept
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        int i = 0;

        // Unrolled loop: Process 16 floats (2 x 8) per iteration
        for (; i + 16 <= size; i += 16)
        {
            __m256 va0 = _mm256_loadu_ps(a + i);
            __m256 vb0 = _mm256_loadu_ps(b + i);
            __m256 diff0 = _mm256_sub_ps(va0, vb0);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

            __m256 va1 = _mm256_loadu_ps(a + i + 8);
            __m256 vb1 = _mm256_loadu_ps(b + i + 8);
            __m256 diff1 = _mm256_sub_ps(va1, vb1);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
        }

        // Process remaining chunk of 8
        if (i + 8 <= size)
        {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            acc0 = _mm256_fmadd_ps(diff, diff, acc0);
            i += 8;
        }

        // Scalar fallback for the final < 8 elements
        // (AVX doesn't have the clean masking loads of AVX-512)
        float sum = 0.0f;
        for (; i < size; ++i)
        {
            float d = a[i] - b[i];
            sum += d * d;
        }

        // Reduce AVX registers to a single float
        __m256 acc = _mm256_add_ps(acc0, acc1);

        // Split 256-bit register into two 128-bit lanes and add
        __m128 low = _mm256_castps256_ps128(acc);
        __m128 high = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(low, high);

        // Horizontal adds to sum up the 4 floats in the XMM register
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum += _mm_cvtss_f32(sum128);

        return sum;
    }

#elif defined(__AVX__)

    /**
     * @brief Standard AVX implementation (No FMA).
     * Performs explicit multiplication and addition steps.
     */
    inline float squaredDistance(const float *__restrict a,
                                 const float *__restrict b,
                                 int size) noexcept
    {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;

        for (; i + 8 <= size; i += 8)
        {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            __m256 sq = _mm256_mul_ps(diff, diff); // Mul and Add separate
            acc = _mm256_add_ps(acc, sq);
        }

        // Horizontal Reduction logic
        __m128 low = _mm256_castps256_ps128(acc);
        __m128 high = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(low, high);

        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        float sum = _mm_cvtss_f32(sum128);

        // Scalar tail
        for (; i < size; ++i)
        {
            float d = a[i] - b[i];
            sum += d * d;
        }

        return sum;
    }

#elif defined(__SSE3__)

    /**
     * @brief Legacy SSE implementation (128-bit).
     * Processes 4 floats at a time.
     */
    inline float squaredDistance(const float *__restrict a,
                                 const float *__restrict b,
                                 int size) noexcept
    {
        __m128 acc = _mm_setzero_ps();
        int i = 0;

        for (; i + 4 <= size; i += 4)
        {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 diff = _mm_sub_ps(va, vb);
            acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
        }

        acc = _mm_hadd_ps(acc, acc);
        acc = _mm_hadd_ps(acc, acc);

        float sum = _mm_cvtss_f32(acc);

        for (; i < size; ++i)
        {
            float d = a[i] - b[i];
            sum += d * d;
        }

        return sum;
    }

#else
    /**
     * @brief Scalar fallback.
     * Compiler auto-vectorization might still optimize this depending on flags (-O3).
     */
    inline float squaredDistance(const float *__restrict a, const float *__restrict b, int size) noexcept
    {
        float sum = 0.0;
        for (size_t i = 0; i < size; ++i)
        {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }
#endif

    // -----------------------------------------------------------------------------
    // Random Number Generation
    // -----------------------------------------------------------------------------

    // Thread-local RNG: Ensures thread safety without locking overhead.
    inline std::mt19937 &global_rng()
    {
        static thread_local std::mt19937 gen(std::random_device{}());
        return gen;
    }

    /**
     * @brief Fills an Eigen matrix with Normally Distributed random values in Parallel.
     * @param matrix The matrix to fill.
     * @param seed The base seed for reproducibility.
     */
    inline void generateRandomMatrix(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> &matrix, int seed = 0)
    {
        const long rows = matrix.rows();
        const long cols = matrix.cols();

        int base_seed = seed;
        if (base_seed == 0)
        {
            std::random_device rd;
            base_seed = rd();
        }

#pragma omp parallel
        {

            int tid = omp_get_thread_num();
            std::mt19937 gen(base_seed + tid);
            std::normal_distribution<float> dist(0.0f, 1.0f);

#pragma omp for schedule(static)
            for (long j = 0; j < cols; ++j)
            {
                float *col_ptr = matrix.data() + (j * rows);
                for (long i = 0; i < rows; ++i)
                {
                    col_ptr[i] = dist(gen);
                }
            }
        }
    }
}