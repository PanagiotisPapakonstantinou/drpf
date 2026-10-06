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

}