#pragma once

#include <vector>
#include <numeric>
#include <memory>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <Eigen/Dense>
#include <bit>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE3__)
#include <pmmintrin.h>
#endif

namespace drpf
{

    inline constexpr char DRPF_MAGIC[8] = {'D', 'R', 'P', 'F', 'I', 'D', 'X', '\0'};
    inline constexpr uint32_t DRPF_FORMAT_VERSION = 1;

    struct DrpfHeader
    {
        char magic[8];
        uint32_t version;
        uint32_t scalar_size;
        int64_t rows;
        int64_t cols;
    };

    static_assert(sizeof(DrpfHeader) == 32, "serialization format assumes 32-byte header");

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

#pragma omp parallel for schedule(static)
        for (long j = 0; j < cols; ++j)
        {
            std::seed_seq sq{static_cast<uint32_t>(base_seed),
                             static_cast<uint32_t>(j)};
            std::mt19937 gen(sq);
            std::normal_distribution<float> dist(0.0f, 1.0f);

            float *col_ptr = matrix.data() + (j * rows);
            for (long i = 0; i < rows; ++i)
            {
                col_ptr[i] = dist(gen);
            }
        }
    }

    struct SrhtProjection
    {
        int dim = 0;                // original dimension
        int P = 0;                  // padded power of two
        int nblocks = 0;            // independent transforms
        std::vector<float> signs;   // nblocks * P random +-1 (first dim used)
        std::vector<int> col_block; // per projection column: which block
        std::vector<int> col_sel;   // per projection column: which Hadamard row

        bool valid() const { return P > 0 && !col_sel.empty(); }
    };

    inline void fwht(float *__restrict a, int P) // Fast Walsh–Hadamard Transform
    {
#if defined(__AVX2__) && defined(__FMA__)
        if (P >= 8)
        {
            const __m256 s1 = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);
            const __m256 s2 = _mm256_setr_ps(1, 1, -1, -1, 1, 1, -1, -1);
            const __m256 s4 = _mm256_setr_ps(1, 1, 1, 1, -1, -1, -1, -1);

            for (int i = 0; i < P; i += 8)
            {
                __m256 v = _mm256_loadu_ps(a + i);
                v = _mm256_fmadd_ps(v, s1, _mm256_permute_ps(v, 0xB1));
                v = _mm256_fmadd_ps(v, s2, _mm256_permute_ps(v, 0x4E));
                v = _mm256_fmadd_ps(v, s4, _mm256_permute2f128_ps(v, v, 1));
                _mm256_storeu_ps(a + i, v);
            }

            int h = 8;
            for (; h * 4 <= P; h <<= 2)
                for (int i = 0; i < P; i += 4 * h)
                    for (int j = i; j < i + h; j += 8)
                    {
                        __m256 x0 = _mm256_loadu_ps(a + j);
                        __m256 x1 = _mm256_loadu_ps(a + j + h);
                        __m256 x2 = _mm256_loadu_ps(a + j + 2 * h);
                        __m256 x3 = _mm256_loadu_ps(a + j + 3 * h);
                        __m256 s01 = _mm256_add_ps(x0, x1), d01 = _mm256_sub_ps(x0, x1);
                        __m256 s23 = _mm256_add_ps(x2, x3), d23 = _mm256_sub_ps(x2, x3);
                        _mm256_storeu_ps(a + j, _mm256_add_ps(s01, s23));
                        _mm256_storeu_ps(a + j + h, _mm256_add_ps(d01, d23));
                        _mm256_storeu_ps(a + j + 2 * h, _mm256_sub_ps(s01, s23));
                        _mm256_storeu_ps(a + j + 3 * h, _mm256_sub_ps(d01, d23));
                    }
            if (h < P)
                for (int i = 0; i < P; i += 2 * h)
                    for (int j = i; j < i + h; j += 8)
                    {
                        __m256 x = _mm256_loadu_ps(a + j);
                        __m256 y = _mm256_loadu_ps(a + j + h);
                        _mm256_storeu_ps(a + j, _mm256_add_ps(x, y));
                        _mm256_storeu_ps(a + j + h, _mm256_sub_ps(x, y));
                    }
            return;
        }
#elif defined(__SSE3__)
        if (P >= 4)
        {
            const __m128 s1 = _mm_setr_ps(1, -1, 1, -1);
            const __m128 s2 = _mm_setr_ps(1, 1, -1, -1);

            // strides 1,2 inside one register
            for (int i = 0; i < P; i += 4)
            {
                __m128 v = _mm_loadu_ps(a + i);
                v = _mm_add_ps(_mm_mul_ps(v, s1), _mm_shuffle_ps(v, v, 0xB1));
                v = _mm_add_ps(_mm_mul_ps(v, s2), _mm_shuffle_ps(v, v, 0x4E));
                _mm_storeu_ps(a + i, v);
            }

            int h = 4;
            for (; h * 4 <= P; h <<= 2)
                for (int i = 0; i < P; i += 4 * h)
                    for (int j = i; j < i + h; j += 4)
                    {
                        __m128 x0 = _mm_loadu_ps(a + j);
                        __m128 x1 = _mm_loadu_ps(a + j + h);
                        __m128 x2 = _mm_loadu_ps(a + j + 2 * h);
                        __m128 x3 = _mm_loadu_ps(a + j + 3 * h);
                        __m128 s01 = _mm_add_ps(x0, x1), d01 = _mm_sub_ps(x0, x1);
                        __m128 s23 = _mm_add_ps(x2, x3), d23 = _mm_sub_ps(x2, x3);
                        _mm_storeu_ps(a + j, _mm_add_ps(s01, s23));
                        _mm_storeu_ps(a + j + h, _mm_add_ps(d01, d23));
                        _mm_storeu_ps(a + j + 2 * h, _mm_sub_ps(s01, s23));
                        _mm_storeu_ps(a + j + 3 * h, _mm_sub_ps(d01, d23));
                    }
            if (h < P)
                for (int i = 0; i < P; i += 2 * h)
                    for (int j = i; j < i + h; j += 4)
                    {
                        __m128 x = _mm_loadu_ps(a + j);
                        __m128 y = _mm_loadu_ps(a + j + h);
                        _mm_storeu_ps(a + j, _mm_add_ps(x, y));
                        _mm_storeu_ps(a + j + h, _mm_sub_ps(x, y));
                    }
            return;
        }
#endif
        for (int h = 1; h < P; h <<= 1)
            for (int i = 0; i < P; i += h << 1)
                for (int j = i; j < i + h; ++j)
                {
                    const float x = a[j], y = a[j + h];
                    a[j] = x + y;
                    a[j + h] = x - y;
                }
    }

    inline void srhtProject(const SrhtProjection &sp, const float *q,
                            float *work, float *out)
    {
        const int ncols = static_cast<int>(sp.col_sel.size());
        const int *__restrict sel = sp.col_sel.data();
        int cur = -1;
        int c = 0;
        while (c < ncols)
        {
            const int b = sp.col_block[c];
            if (b != cur)
            {
                const float *s = sp.signs.data() + static_cast<size_t>(b) * sp.P;
                for (int j = 0; j < sp.dim; ++j)
                    work[j] = s[j] * q[j];
                std::fill(work + sp.dim, work + sp.P, 0.0f);
                fwht(work, sp.P);
                cur = b;
            }

            for (; c < ncols && sp.col_block[c] == b; ++c)
                out[c] = work[sel[c]];
        }
    }

    inline void generateSrhtProjection(
        SrhtProjection &sp, int dim, int cols, int seed = 0)
    {
        if (dim <= 0 || cols <= 0)
            throw std::invalid_argument("SRHT dimensions must be positive");

        int base_seed = seed;
        if (base_seed == 0)
        {
            std::random_device rd;
            base_seed = static_cast<int>(rd());
        }

        int P = 1;
        while (P < dim)
            P <<= 1;

        const int nblocks = (cols + P - 1) / P;

        sp.dim = dim;
        sp.P = P;
        sp.nblocks = nblocks;
        sp.signs.assign(static_cast<size_t>(nblocks) * P, 0.0f);
        sp.col_block.resize(cols);
        sp.col_sel.resize(cols);

        std::vector<int> perm(P);
        for (int b = 0; b < nblocks; ++b)
        {
            std::seed_seq seq{
                static_cast<uint32_t>(base_seed),
                static_cast<uint32_t>(b)};
            std::mt19937 gen(seq);

            for (int j = 0; j < dim; ++j)
                sp.signs[static_cast<size_t>(b) * P + j] =
                    (gen() & 1u) ? 1.0f : -1.0f;

            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), gen);

            for (int t = 0; t < P; ++t)
            {
                const int c = b * P + t;
                if (c >= cols)
                    break;
                sp.col_block[c] = b;
                sp.col_sel[c] = perm[t];
            }
        }
    }

    template <typename OutputMatrix>
    inline void projectDataSrht(
        const SrhtProjection &sp,
        const float *vectors,
        long rows,
        int dim,
        OutputMatrix &out)
    {
        if (dim != sp.dim)
            throw std::invalid_argument("SRHT input dimension mismatch");

        const int ncols = static_cast<int>(sp.col_sel.size());
        out.resize(rows, ncols);

#pragma omp parallel
        {
            std::vector<float> work(sp.P);
            std::vector<float> projected(ncols);

#pragma omp for schedule(static)
            for (long i = 0; i < rows; ++i)
            {
                const float *input =
                    vectors + static_cast<size_t>(i) * dim;

                srhtProject(sp, input, work.data(), projected.data());

                for (int c = 0; c < ncols; ++c)
                {
                    out(i, c) =
                        static_cast<typename OutputMatrix::Scalar>(projected[c]);
                }
            }
        }
    }

    inline void materializeSrhtMatrix(
        const SrhtProjection &sp,
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::ColMajor> &M)
    {
        const int ncols = static_cast<int>(sp.col_sel.size());
        M.resize(sp.dim, ncols);

#pragma omp parallel for schedule(static)
        for (int c = 0; c < ncols; ++c)
        {
            const uint32_t sel = static_cast<uint32_t>(sp.col_sel[c]);
            const float *signs =
                sp.signs.data() +
                static_cast<size_t>(sp.col_block[c]) * sp.P;
            float *col = M.data() + static_cast<size_t>(c) * sp.dim;

            for (int j = 0; j < sp.dim; ++j)
            {
                const bool negative =
                    (std::popcount(sel & static_cast<uint32_t>(j)) & 1) != 0;
                col[j] = negative ? -signs[j] : signs[j];
            }
        }
    }

    template <typename T>
    void writePod(std::ostream &os, const T &value)
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "writePod: T must be trivially copyable — raw byte serialization "
                      "is invalid for types owning heap memory (e.g. std::string, std::vector)");

        os.write(reinterpret_cast<const char *>(&value), sizeof(T));
    }

    template <typename T>
    void readPod(std::istream &is, T &value)
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "readPod: T must be trivially copyable");

        is.read(reinterpret_cast<char *>(&value), sizeof(T));
        if (!is)
            throw std::runtime_error("readPod: unexpected end of stream");
    }

    template <typename T>
    void writePodArray(std::ostream &os, const T *data, uint64_t count)
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "writePodArray: T must be trivially copyable — raw byte serialization "
                      "is invalid for types owning heap memory (e.g. std::string, std::vector)");

        os.write(reinterpret_cast<const char *>(&count), sizeof(count));
        if (count)
            os.write(reinterpret_cast<const char *>(data),
                     static_cast<std::streamsize>(count * sizeof(T)));
    }

    template <typename T>
    void readPodArray(std::istream &is, std::vector<T> &vec, uint64_t max_expected)
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "readPodArray: T must be trivially copyable");

        uint64_t count = 0;
        readPod(is, count);

        if (count > max_expected)
            throw std::runtime_error("readPodArray: element count " + std::to_string(count) +
                                     " exceeds expected maximum " + std::to_string(max_expected) +
                                     " (corrupt or foreign index file)");

        vec.resize(static_cast<std::size_t>(count));
        if (count)
        {
            is.read(reinterpret_cast<char *>(vec.data()),
                    static_cast<std::streamsize>(count * sizeof(T)));
            if (!is)
                throw std::runtime_error("readPodArray: truncated stream");
        }
    }

    template <typename Scalar, int Rows, int Cols, int Options>
    void writeMatrix(std::ostream &os, const Eigen::Matrix<Scalar, Rows, Cols, Options> &mat)
    {
        static_assert((Options & Eigen::RowMajorBit) == 0,
                      "writeMatrix: format assumes column-major storage");

        const uint64_t rows = static_cast<uint64_t>(mat.rows());
        const uint64_t cols = static_cast<uint64_t>(mat.cols());
        os.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        os.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

        writePodArray(os, mat.data(), static_cast<uint64_t>(mat.size()));
    }

    template <typename Scalar, int Rows, int Cols, int Options>
    void readMatrix(std::istream &is, Eigen::Matrix<Scalar, Rows, Cols, Options> &mat,
                    uint64_t max_elements)
    {
        static_assert((Options & Eigen::RowMajorBit) == 0,
                      "readMatrix: format assumes column-major storage");

        uint64_t rows = 0, cols = 0, count = 0;
        readPod(is, rows);
        readPod(is, cols);
        readPod(is, count);

        if (rows * cols != count)
            throw std::runtime_error("readMatrix: shape/count mismatch (corrupt index file)");
        if (count > max_elements)
            throw std::runtime_error("readMatrix: matrix too large (corrupt index file)");

        mat.resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
        if (count)
        {
            is.read(reinterpret_cast<char *>(mat.data()),
                    static_cast<std::streamsize>(count * sizeof(Scalar)));
            if (!is)
                throw std::runtime_error("readMatrix: truncated stream");
        }
    }

    float computeSSE(const Eigen::Ref<
                         const Eigen::Matrix<
                             float,
                             Eigen::Dynamic,
                             Eigen::Dynamic,
                             Eigen::RowMajor>> &data,
                     std::vector<int> &indices)
    {
        Eigen::RowVectorXf mean = Eigen::RowVectorXf::Zero(data.cols());

        for (int idx : indices)
            mean += data.row(idx);

        mean /= static_cast<float>(indices.size());

        float sse = 0.0f;

        for (int idx : indices)
        {
            Eigen::RowVectorXf diff = data.row(idx) - mean;
            sse += diff.squaredNorm();
        }

        return sse;
    }
}
