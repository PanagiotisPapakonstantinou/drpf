#pragma once

#include <complex>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <iostream>
#include <memory>
#include <optional>
#include <span>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

#if defined(__SSE2__) && !defined(__AVX__)
#include <emmintrin.h>
#endif

using std::complex;

/**
 * @brief Utility to find the next power of 2.
 * FFT algorithms typically require the input size to be a power of 2
 * for the recursive Cooley-Tukey algorithm to work efficiently.
 */
inline int next_pow2(int x)
{
    if (x <= 1)
        return 1;
    int p = 1;
    while (p < x)
        p <<= 1;
    return p;
}

/**
 * @brief Fast Fourier Transform (FFT) implementation.
 *
 * This class implements the Cooley-Tukey Radix-2 decimation-in-time algorithm.
 * It is highly optimized using precomputed bit-reversal tables, precomputed
 * twiddle factors, and SIMD (AVX2/SSE2) intrinsics for complex multiplication.
 *
 * @tparam Real The floating point type (float or double).
 */
template <typename Real>
class FFT
{

public:
    using Cx = std::complex<Real>;

    /**
     * @brief Constructs the FFT plan.
     * Precomputes bit-reversal indices and twiddle factors (roots of unity)
     * to avoid expensive trigonometric calculations during the transform.
     * @param n Size of the transform (must be power of 2).
     */
    explicit FFT(int n) : n(n), logn(0)
    {
        if (n <= 0 || (n & (n - 1)) != 0)
            throw std::invalid_argument("FFT: n must be a power of two > 0");

        int temp_n = n;
        while (temp_n > 1)
        {
            temp_n >>= 1;
            ++logn;
        }

        // Precompute Bit-Reversal
        // This reorders the input array so that the output of the butterfly
        // operations comes out in natural order.
        bitrev.resize(n);
        bitrev[0] = 0;
        for (int i = 0; i < n; ++i)
        {
            bitrev[i] = (bitrev[i >> 1] >> 1) | ((i & 1) << (logn - 1));
        }

        // Precompute Twiddle Factors
        // W_N^k = e^(-i * 2 * pi * k / N)
        // We only need N/2 twiddles for the butterfly operations.
        twiddles.resize(n / 2);
        const double PI = std::acos(-1.0);
        for (int i = 0; i < n / 2; ++i)
        {
            double angle = -2.0 * PI * i / n;
            twiddles[i] = Cx(static_cast<Real>(std::cos(angle)), static_cast<Real>(std::sin(angle)));
        }
    }

    void fft(Cx *a)
    {
        fft_internal<false>(a);
    }

    void ifft(Cx *a)
    {
        fft_internal<true>(a);

        Real inv_n = Real(1) / Real(n);

        for (int i = 0; i < n; ++i)
        {
            a[i] *= inv_n;
        }
    }

private:
    int n, logn;
    std::vector<int> bitrev;
    std::vector<Cx> twiddles;

    /**
     * @brief Performs an in-place Forward FFT.
     * @param a Pointer to the array of complex numbers.
     */
    template <bool Inverse>
    void fft_internal(Cx *a)
    {
        if (n <= 1)
            return;

        // Bit-Reversal Permutation
        // Swaps elements to prepare for the iterative butterfly stages.
        for (int i = 0; i < n; ++i)
        {
            int j = bitrev[i];
            if (i < j)
                std::swap(a[i], a[j]);
        }

        // Cooley-Tukey Iteration
        // Loops through stages: len = 2, 4, 8, ..., N
        for (int len = 2; len <= n; len <<= 1)
        {
            int half = len >> 1; // Distance between butterfly pair elements
            int step = n / len;  // Stride for twiddle factors

            // Iterate through each block of size 'len'
            for (int i = 0; i < n; i += len)
            {
                int j = 0;

                // =========================================================
                // AVX2 Optimization
                // =========================================================
#if defined(__AVX2__)
                // ---------------- DOUBLE PRECISION ----------------
                // Processes 2 complex numbers (256 bits = 4 doubles) at a time.
                if constexpr (std::is_same<Real, double>::value)
                {
                    for (; j + 1 < half; j += 2)
                    {
                        // Load U (2 complex) and V (2 complex)
                        // Memory layout: [u0_r, u0_i, u1_r, u1_i]
                        __m256d u_vec = _mm256_loadu_pd(reinterpret_cast<const double *>(&a[i + j]));
                        __m256d v_in_vec = _mm256_loadu_pd(reinterpret_cast<const double *>(&a[i + j + half]));

                        // Load Twiddles for 2 positions
                        Real wr0 = twiddles[j * step].real();
                        Real wi0 = twiddles[j * step].imag();
                        Real wr1 = twiddles[(j + 1) * step].real();
                        Real wi1 = twiddles[(j + 1) * step].imag();

                        // Complex Multiplication Logic (V * W):
                        // (vr + i*vi)(wr + i*wi) = (vr*wr - vi*wi) + i(vr*wi + vi*wr)
                        // This requires duplicating W values and swapping V components.

                        // Prepare W vectors: duplicates for simultaneous calc
                        __m256d w_dup_r = _mm256_set_pd(wr1, wr1, wr0, wr0); // [wr1, wr1, wr0, wr0]
                        __m256d w_dup_i = _mm256_set_pd(wi1, wi1, wi0, wi0); // [wi1, wi1, wi0, wi0]

                        if constexpr (Inverse)
                        {
                            w_dup_i = _mm256_sub_pd(_mm256_setzero_pd(), w_dup_i);
                        }

                        // term1 = [vr1*wr1, vi1*wr1, vr0*wr0, vi0*wr0]
                        __m256d term1 = _mm256_mul_pd(v_in_vec, w_dup_r);

                        // Swap V real/imag: [vr1, vi1, vr0, vi0] -> [vi1, vr1, vi0, vr0]
                        __m256d v_swap_vec = _mm256_shuffle_pd(v_in_vec, v_in_vec, 0b0101);

                        // term2 = [vi1*wi1, vr1*wi1, vi0*wi0, vr0*wi0]
                        __m256d term2 = _mm256_mul_pd(v_swap_vec, w_dup_i);

                        // v_out = term1 +/- term2
                        // Real part: vr*wr - vi*wi (Subtraction)
                        // Imag part: vi*wr + vr*wi (Addition)
                        // _mm256_addsub_pd does [-, +, -, +]
                        __m256d v_out = _mm256_addsub_pd(term1, term2);

                        // Butterfly operations:
                        // new_u = u + (v*w)
                        // new_v = u - (v*w)
                        __m256d new_u = _mm256_add_pd(u_vec, v_out);
                        __m256d new_v = _mm256_sub_pd(u_vec, v_out);

                        _mm256_storeu_pd(reinterpret_cast<double *>(&a[i + j]), new_u);
                        _mm256_storeu_pd(reinterpret_cast<double *>(&a[i + j + half]), new_v);
                    }
                }
                // ---------------- FLOAT PRECISION ----------------
                // Processes 4 complex numbers (256 bits = 8 floats) at a time.
                else if constexpr (std::is_same<Real, float>::value)
                {
                    for (; j + 3 < half; j += 4)
                    {
                        // Load U and V (4 complex numbers each)
                        __m256 u_vec = _mm256_loadu_ps(reinterpret_cast<const float *>(&a[i + j]));
                        __m256 v_in_vec = _mm256_loadu_ps(reinterpret_cast<const float *>(&a[i + j + half]));

                        // Load Twiddles for 4 positions
                        float wr0 = twiddles[j * step].real();
                        float wi0 = twiddles[j * step].imag();
                        float wr1 = twiddles[(j + 1) * step].real();
                        float wi1 = twiddles[(j + 1) * step].imag();
                        float wr2 = twiddles[(j + 2) * step].real();
                        float wi2 = twiddles[(j + 2) * step].imag();
                        float wr3 = twiddles[(j + 3) * step].real();
                        float wi3 = twiddles[(j + 3) * step].imag();

                        // Create vector of real/imag parts of W
                        __m256 w_dup_r = _mm256_set_ps(wr3, wr3, wr2, wr2, wr1, wr1, wr0, wr0);
                        __m256 w_dup_i = _mm256_set_ps(wi3, wi3, wi2, wi2, wi1, wi1, wi0, wi0);

                        if constexpr (Inverse)
                        {
                            w_dup_i = _mm256_sub_ps(_mm256_setzero_ps(), w_dup_i);
                        }

                        // Complex Multiply: V * W
                        // term1 = [vr*wr, vi*wr...]
                        __m256 term1 = _mm256_mul_ps(v_in_vec, w_dup_r);

                        // Swap Real/Imag in V: [vr, vi] -> [vi, vr]
                        // _mm256_permute_ps uses 8-bit control. 0xB1 = 10 11 00 01 (swaps adjacent pairs)
                        __m256 v_swap_vec = _mm256_permute_ps(v_in_vec, 0xB1);

                        // term2 = [vi*wi, vr*wi...]
                        __m256 term2 = _mm256_mul_ps(v_swap_vec, w_dup_i);

                        // v_out = [real_part, imag_part]
                        // real = vr*wr - vi*wi
                        // imag = vi*wr + vr*wi
                        // addsub_ps performs subtractions on even indices, additions on odd
                        __m256 v_out = _mm256_addsub_ps(term1, term2);

                        // Butterfly: U +/- V_out
                        __m256 new_u = _mm256_add_ps(u_vec, v_out);
                        __m256 new_v = _mm256_sub_ps(u_vec, v_out);

                        _mm256_storeu_ps(reinterpret_cast<float *>(&a[i + j]), new_u);
                        _mm256_storeu_ps(reinterpret_cast<float *>(&a[i + j + half]), new_v);
                    }
                }
#endif

                // =========================================================
                // SSE2 Optimization (Fallback if AVX not present but SSE is)
                // =========================================================
#if defined(__SSE2__) && !defined(__AVX2__)
                if constexpr (std::is_same<Real, double>::value)
                {
                    // Processes 1 complex number (2 doubles) per iteration
                    for (; j < half; ++j)
                    {
                        __m128d u_vec = _mm_loadu_pd(reinterpret_cast<const double *>(&a[i + j]));
                        __m128d v_in_vec = _mm_loadu_pd(reinterpret_cast<const double *>(&a[i + j + half]));
                        Real wr = twiddles[j * step].real();
                        Real wi = twiddles[j * step].imag();
                        __m128d w_dup_r = _mm_set1_pd(wr);
                        __m128d w_dup_i = _mm_set1_pd(wi);

                        if constexpr (Inverse)
                        {
                            w_dup_i = _mm_sub_pd(_mm_setzero_pd(), w_dup_i);
                        }

                        __m128d term1 = _mm_mul_pd(v_in_vec, w_dup_r);
                        __m128d v_swap = _mm_shuffle_pd(v_in_vec, v_in_vec, 0x1);
                        __m128d term2 = _mm_mul_pd(v_swap, w_dup_i);
                        __m128d v_out = _mm_addsub_pd(term1, term2);
                        __m128d new_u = _mm_add_pd(u_vec, v_out);
                        __m128d new_v = _mm_sub_pd(u_vec, v_out);
                        _mm_storeu_pd(reinterpret_cast<double *>(&a[i + j]), new_u);
                        _mm_storeu_pd(reinterpret_cast<double *>(&a[i + j + half]), new_v);
                    }
                }
                else if constexpr (std::is_same<Real, float>::value)
                {
                    // Processes 2 complex numbers (4 floats) per iteration
                    for (; j + 1 < half; j += 2)
                    {
                        __m128 u_vec = _mm_loadu_ps(reinterpret_cast<const float *>(&a[i + j]));
                        __m128 v_in_vec = _mm_loadu_ps(reinterpret_cast<const float *>(&a[i + j + half]));

                        float wr0 = twiddles[j * step].real();
                        float wi0 = twiddles[j * step].imag();
                        float wr1 = twiddles[(j + 1) * step].real();
                        float wi1 = twiddles[(j + 1) * step].imag();

                        __m128 w_dup_r = _mm_set_ps(wr1, wr1, wr0, wr0);
                        __m128 w_dup_i = _mm_set_ps(wi1, wi1, wi0, wi0);

                        if constexpr (Inverse)
                        {
                            w_dup_i = _mm_sub_ps(_mm_setzero_ps(), w_dup_i);
                        }

                        __m128 term1 = _mm_mul_ps(v_in_vec, w_dup_r);
                        __m128 v_swap = _mm_shuffle_ps(v_in_vec, v_in_vec, 0xB1);
                        __m128 term2 = _mm_mul_ps(v_swap, w_dup_i);
                        __m128 v_out = _mm_addsub_ps(term1, term2);

                        __m128 new_u = _mm_add_ps(u_vec, v_out);
                        __m128 new_v = _mm_sub_ps(u_vec, v_out);

                        _mm_storeu_ps(reinterpret_cast<float *>(&a[i + j]), new_u);
                        _mm_storeu_ps(reinterpret_cast<float *>(&a[i + j + half]), new_v);
                    }
                }
#endif

                // ---------------------------------------------------------
                // Scalar Fallback (Cleanup for remaining items)
                // ---------------------------------------------------------
                // This handles cases where vector width > remaining elements,
                // or if no SIMD instruction set is available.
                for (; j < half; ++j)
                {
                    Cx u = a[i + j];
                    Cx w = twiddles[j * step];

                    if constexpr (Inverse)
                        w = std::conj(w);

                    Cx v = a[i + j + half] * w;
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                }
            }
        }
    }
};

// ---------- templated KDE FFT class ----------

/**
 * @brief Kernel Density Estimation using FFT convolution.
 *
 * This class approximates the Probability Density Function (PDF) of a dataset.
 * It uses the Convolution Theorem: f * g = IFFT(FFT(f) . FFT(g)).
 * This transforms the O(N*M) naive KDE complexity into O(N log N).
 *
 * Algorithm steps:
 * 1. Discretize data onto a grid (Linear Binning).
 * 2. Create a discretized Gaussian kernel.
 * 3. Perform FFT on both.
 * 4. Multiply in frequency domain.
 * 5. Inverse FFT to get density.
 */
template <typename T>
class KdeFFT
{
public:
    /**
     * @brief Constructor
     * @param grid_size Number of points in the estimation grid (will be rounded to power of 2).
     * @param bandwidth_method "silverman" (classic) or other heuristic.
     * @param normalize If true, data is internally z-scored (mean 0, std 1) during fitting.
     * @param split_data_bandwidth Scalar applied to data before bandwidth initialization.
     */
    KdeFFT(int grid_size = 1024, const std::string &bandwidth_method = "silverman",
           bool normalize = true, T split_data_bandwidth = T(0.1))
        : N_(grid_size), bandwidth_method_(bandwidth_method), normalize_(normalize),
          split_data_bandwidth_(split_data_bandwidth), fft_(nullptr)
    {
        if (N_ <= 0)
            throw std::invalid_argument("Grid size must be > 0");
        if (split_data_bandwidth_ == T(0))
            throw std::invalid_argument("split_data_bandwidth cannot be zero");

        N_ = next_pow2(N_);

        int max_M = next_pow2(2 * 1024);
        counts_pad_.reserve(max_M);
        kernel_pad_.reserve(max_M);
        allocate_buffers();
    }

    /**
     * @brief Fit the KDE model to a dataset.
     * @param data Vector of scalar values.
     */
    void fit(const std::vector<T> &data)
    {
        update(data);
    }

    /**
     * @brief Heuristic to adjust grid size based on number of data points.
     * Derived from Silverman's rule optimal bin counts.
     */
    void autotune_grid(size_t n_data)
    {
        // 1. Calculate optimal grid size (Silverman's rule derivative)
        // Formula: N ~ 20 * n^0.2
        double optimal_N = 20.0 * std::pow(static_cast<double>(n_data), 0.2);

        // 2. Clamp to powers of two between 64 and 1024
        int target_N = static_cast<int>(optimal_N);
        target_N = std::max(64, std::min(target_N, 1024));
        target_N = next_pow2(target_N);

        // 3. Only re-allocate if the grid size MUST change.
        if (target_N != N_)
        {
            N_ = target_N;
            allocate_buffers();
        }
    }

    /**
     * @brief Updates the density estimation with new data.
     * Checks if normalization is enabled and routes accordingly.
     */
    void update(const std::vector<T> &data)
    {
        if (normalize_)
            normalize_update(data);
        else
            raw_update(data);
    }

    /**
     * @brief Update without normalization (uses raw data values).
     */
    void raw_update(const std::vector<T> &data)
    {
        data_ = data;

        autotune_grid(data_.size());

        if (data_.size() < 2)
            throw std::invalid_argument("Need at least 2 points");
        if (std::all_of(data_.begin(), data_.end(), [&](T x)
                        { return x == data_.front(); }))
            throw std::invalid_argument("All values equal, invalid dataset.");

        mean_ = T(0);
        stddev_ = T(1);

        init_bandwidth();
        build_grid_extended();
        compute_density();
    }

    void normalize_update(const std::vector<T> &data)
    {
        if (data.empty())
            throw std::invalid_argument("KDE update: data cannot be empty");
        autotune_grid(data.size());

        double sum = 0.0;
        for (const T &v : data)
            sum += static_cast<double>(v);
        T temp_mean = sum / static_cast<T>(data.size());

        long double acc = 0.0L;
        for (const T &x : data)
        {
            long double d = static_cast<long double>(x) - static_cast<long double>(temp_mean);
            acc += d * d;
        }
        T var = (data.size() > 1) ? static_cast<T>(acc / (data.size() - 1)) : T(0);
        T temp_std = std::sqrt(static_cast<long double>(var));

        if (temp_std <= T(1e-9))
            temp_std = T(1);

        mean_ = temp_mean;
        stddev_ = temp_std;

        data_.resize(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            data_[i] = (data[i] - mean_) / stddev_;
        }

        init_bandwidth();
        build_grid_extended();
        compute_density();
    }
    const std::vector<T> &get_density_ref() const { return density_; }

    const std::vector<T> &get_internal_data() const { return data_; }

    const T &get_mean() const { return mean_; }

    const T &get_stddev() const { return stddev_; }

    /**
     * @brief Returns the X-coordinate for a specific grid index.
     * De-normalizes the value if normalization was used.
     */
    T get_grid_value(int i) const
    {
        T raw_val = xmin_ + (T(i) + T(0.5)) * dx_;
        return raw_val * stddev_ + mean_;
    }

    /**
     * @brief Finds a significant local minimum ("valley") in the density function.
     *
     * Heuristics used:
     * 1. Ignore "tails": Only search the middle 33% of the probability mass
     * to avoid detecting noise in the low-density tails.
     * 2. Local Minimum: Checks strictly if val <= left AND val < right.
     *
     * @return std::optional<float> The data value (x-axis) at the valley,
     * or nullopt if no valid valley is found.
     */
    std::optional<float> findDensityMinimum(const float min_ratio = 0.33) const
    {
        const auto &density = get_density_ref();
        if (density.empty())
            return std::nullopt;

        // 1. Calculate Total Mass (Normalization check)
        float totalMass = 0.0f;
        for (float v : density)
            totalMass += v;

        if (totalMass <= 0.0f)
            return std::nullopt;

        // 2. Define Search Region (Middle 33% (Default argument) of mass)
        // We skip the first 33% and last 33% of the cumulative mass
        // to focus on the area *between* potential peaks.
        const float minMassLimit = totalMass * min_ratio;
        const float maxMassLimit = totalMass * (1.0f - min_ratio);

        float currentMass = density[0];
        int bestIdx = -1;
        float lowestDensity = std::numeric_limits<float>::max();
        const int n = static_cast<int>(density.size());

        // 3. Scan for Minima
        for (int i = 1; i < n - 1; ++i)
        {
            float val = density[i];
            currentMass += val;

            // Stop if we've passed the search region
            if (currentMass > maxMassLimit)
                break;
            // Skip if we haven't reached the search region
            if (currentMass < minMassLimit)
                continue;

            // Check for Valley topology:  \ _ /
            if (val <= density[i - 1] && val < density[i + 1])
            {
                // Keep the deepest valley found so far
                if (val < lowestDensity)
                {
                    lowestDensity = val;
                    bestIdx = i;
                }
            }
        }

        if (bestIdx == -1)
            return std::nullopt;

        // Convert grid index back to the original data domain value
        return get_grid_value(bestIdx);
    }

private:
    std::vector<T> data_;
    int N_;
    std::string bandwidth_method_;
    T h_;
    T mean_ = T(0);
    T stddev_ = T(1);
    T xmin_ = T(0), xmax_ = T(0), dx_ = T(0);

    bool normalize_ = false;
    T split_data_bandwidth_;

    std::vector<T> normed_data_;

    std::vector<T> grid_;
    std::vector<T> density_;

    // Caching mechanism for Kernel FFT
    std::vector<std::complex<T>> kernel_fft_;
    T last_h_ = T(-1);
    bool kernel_fft_valid_ = false;

    // Buffers padded to 2*N to avoid circular convolution artifacts
    std::vector<std::complex<T>> counts_pad_;
    std::vector<std::complex<T>> kernel_pad_;
    std::unique_ptr<FFT<T>> fft_;

    /**
     * @brief Allocates FFT buffers.
     * The FFT size M must be >= 2*N (padded) to perform linear convolution
     * using the circular convolution property of FFT.
     */
    void allocate_buffers()
    {
        int M = next_pow2(2 * N_);
        counts_pad_.assign(M, std::complex<T>(0, 0));
        kernel_pad_.assign(M, std::complex<T>(0, 0));
        fft_ = std::make_unique<FFT<T>>(M);

        kernel_fft_valid_ = false;
        last_h_ = T(-1);
    }

    /**
     * @brief Calculates bandwidth (smoothing parameter).
     * Implements Silverman's Rule of Thumb.
     */
    void init_bandwidth()
    {
        const size_t n = data_.size();
        if (n < 2)
            throw std::invalid_argument("Need at least two points");

        bool has_nan = false;
        T data_min = data_[0], data_max = data_[0];
        for (const T &v : data_)
        {
            if (!std::isfinite(static_cast<long double>(v)))
            {
                has_nan = true;
                break;
            }
            if (v < data_min)
                data_min = v;
            if (v > data_max)
                data_max = v;
        }
        if (has_nan)
            throw std::invalid_argument("Data contains NaN or Inf");

        T sd = std_dev(data_);

        // Robust IQR (Interquartile Range) to handle outliers
        std::vector<T> scratch = data_;
        size_t i1 = n / 4;
        size_t i3 = 3 * n / 4;
        std::nth_element(scratch.begin(), scratch.begin() + i1, scratch.end());
        T q1 = scratch[i1];

        std::nth_element(scratch.begin() + i1 + 1, scratch.begin() + i3, scratch.end());
        T q3 = scratch[i3];

        T iqr = q3 - q1;

        // Silverman-like base: min(std_dev, IQR/1.34)
        T A = std::min(sd, iqr / T(1.34));
        T raw_h;

        // h = 0.9 * A * n^(-1/5)
        if (bandwidth_method_ == "silverman")
            raw_h = T(0.9) * A * std::pow(static_cast<T>(n), T(-0.2));
        else
            raw_h = sd * std::pow(static_cast<T>(n), T(-0.2));

        raw_h *= split_data_bandwidth_;

        // Safety checks for extremely small variance
        T range = data_max - data_min;
        T eps = static_cast<T>(std::numeric_limits<typename std::conditional<std::is_floating_point<T>::value, T, double>::type>::epsilon());
        T min_scale = std::max<T>(eps, range / std::max<T>(T(1), static_cast<T>(N_ * 1000)));

        if (sd > T(0))
            min_scale = std::min<T>(min_scale, sd);

        if (!(raw_h > T(0)))
            raw_h = std::max<T>(min_scale, eps * T(1e3));
        else if (raw_h < min_scale)
            raw_h = min_scale;

        h_ = raw_h * raw_h; // Store as variance

        if (!(h_ > T(0)))
        {
            h_ = static_cast<T>(eps);
        }
    }

    /**
     * @brief Determines the spatial grid limits.
     * Adds a margin based on bandwidth to ensure the density tails don't get cut off.
     */
    void build_grid_extended()
    {
        T data_min = *std::min_element(data_.begin(), data_.end());
        T data_max = *std::max_element(data_.begin(), data_.end());
        T h_std = std::sqrt(h_);

        // Margin helps prevent wrap-around effects in FFT convolution
        T margin = std::max(T(4.0) * h_std, T(0.1) * (data_max - data_min));
        xmin_ = data_min - margin;
        xmax_ = data_max + margin;
        if (xmax_ <= xmin_)
        {
            xmin_ -= 1.0;
            xmax_ += 1.0;
        }

        dx_ = (xmax_ - xmin_) / T(N_);

        grid_.resize(N_);
        for (int i = 0; i < N_; ++i)
            grid_[i] = xmin_ + (T(i) + T(0.5)) * dx_;
    }

    /**
     * @brief Core KDE computation logic using FFT convolution.
     *
     * This method implements the Fast Gauss Transform approximation:
     * 1. Discretizes data onto a linear grid (Linear Binning).
     * 2. Convolves the grid with a Gaussian kernel using the Convolution Theorem.
     * (Convolution in Time Domain == Multiplication in Frequency Domain).
     *
     * Complexity: O(N log N) where N is the grid size.
     * (Standard KDE is O(M*N) where M is data size).
     */
    void compute_density()
    {
        const size_t ndata = data_.size();
        if (ndata == 0)
            throw std::invalid_argument("compute_density: no data provided");

        const T ndata_t = static_cast<T>(ndata);
        const int M = counts_pad_.size();

        // ---------------------------------------------------------
        // Step 1: Linear Binning (Discretization)
        // ---------------------------------------------------------
        // We project continuous data points onto a discrete grid.
        // Unlike a simple histogram (nearest bin), we distribute the mass
        // of each point linearly to the two nearest grid nodes.
        // This acts as a first-order anti-aliasing filter.

        // Clear previous grid counts
        std::fill(counts_pad_.begin(), counts_pad_.end(), std::complex<T>(0, 0));

        T inv_dx = T(1) / dx_;
        int max_idx = N_ - 2; // Safety limit for grid indices

        for (const T &v : data_)
        {
            // Map data value to grid index (continuous)
            T pos = (v - xmin_) * inv_dx;
            int idx = static_cast<int>(pos);

            // Branchless Clamping
            // Ensures valid access to counts_pad_[idx] and [idx+1].
            // Although data is theoretically within bounds, floating point
            // epsilon errors or outliers can sometimes push values slightly off.
            idx = std::clamp(idx, 0, max_idx);

            // 'r' is the fractional distance to the next grid point.
            // Mass distribution: (1-r) to 'idx', (r) to 'idx+1'
            T r = pos - idx;

            counts_pad_[idx].real(counts_pad_[idx].real() + (T(1) - r));
            counts_pad_[idx + 1].real(counts_pad_[idx + 1].real() + r);
        }

        // ---------------------------------------------------------
        // Step 2: Prepare Kernel (Gaussian)
        // ---------------------------------------------------------
        // Recompute the cached kernel only if the bandwidth (smoothing factor)
        // has changed significantly since the last run.
        if (!kernel_fft_valid_ || std::abs(h_ - last_h_) > static_cast<T>(1e-6))
        {
            build_kernel_fft();
        }

        // ---------------------------------------------------------
        // Step 3: FFT Convolution
        // ---------------------------------------------------------
        // Transform the binned data into the frequency domain.
        fft_->fft(counts_pad_.data());

        // Perform point-wise multiplication with the kernel in frequency domain.
        // This effectively convolves the raw data with the Gaussian kernel.
        // Note: The loop is typically auto-vectorized by the compiler.
        for (int m = 0; m < M; ++m)
            counts_pad_[m] *= kernel_fft_[m];

        // Inverse transform to get back to the time (density) domain.
        fft_->ifft(counts_pad_.data());

        // ---------------------------------------------------------
        // Step 4: Post-Processing
        // ---------------------------------------------------------
        // 1. Extract Real component (Imaginary part should be near zero).
        // 2. Scale by data size (convert counts to probability).
        // 3. Clamp negatives (artifacts from FFT numerical precision).

        density_.resize(N_);
        T scale_factor = T(1) / ndata_t;

        for (int i = 0; i < N_; ++i)
        {
            T val = counts_pad_[i].real() * scale_factor;
            density_[i] = std::max(val, T(0));
        }
    }
    /**
     * @brief Constructs the Gaussian kernel in the frequency domain.
     * The kernel is wrapped around to satisfy circular convolution requirements.
     */
    void build_kernel_fft()
    {
        int M = counts_pad_.size();
        kernel_pad_.assign(M, std::complex<T>(0, 0));

        T h_std = std::sqrt(h_);
        T s2 = 2.0 * h_;
        T prefac = 1.0 / (h_std * std::sqrt(2.0 * M_PI));

        // Build Gaussian kernel in time domain
        // We calculate for both positive and negative lags using wrap-around indices
        for (int m = 0; m < M; ++m)
        {
            int k = (m <= M / 2) ? m : m - M;
            T x = k * dx_;
            kernel_pad_[m] = prefac * std::exp(-(x * x) / s2);
        }

        // Normalize kernel area to 1.0
        T ksum = 0;
        for (int m = 0; m < M; ++m)
            ksum += kernel_pad_[m].real();
        for (int m = 0; m < M; ++m)
            kernel_pad_[m] /= (ksum * dx_);

        // Compute FFT of the kernel
        kernel_fft_.resize(M);
        std::copy(kernel_pad_.begin(), kernel_pad_.end(), kernel_fft_.begin());
        fft_->fft(kernel_fft_.data());

        kernel_fft_valid_ = true;
        last_h_ = h_;
    }

    static T std_dev(const std::vector<T> &data)
    {
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double acc = 0.0;
        for (const T &v : data)
        {
            double d = v - mean;
            acc += d * d;
        }
        return static_cast<T>(std::sqrt(acc / (data.size() - 1)));
    }
};