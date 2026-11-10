#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <cstdint>
#include <span>
#include <omp.h>

#ifndef PREFETCH
#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch(addr)
#elif defined(_MSC_VER)
#include <xmmintrin.h>
#define PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T0)
#else
#define PREFETCH(addr)
#endif
#endif

#include "binarySplitTree.h"
#include "fft_kde.h"
#include "utils.h"

namespace drpf
{

    /**
     * @brief KDE-based Implementation of the Binary Split Tree.
     */
    template <typename Scalar = Eigen::half>
    class KdeBinarySplitTree : public BinarySplitTreeBase<Scalar>
    {
    private:
        std::vector<float> projectionValues;
        std::unique_ptr<fftkde::KdeFFT<float>> kde;
        float split_data_bandwidth;
        float min_ratio;

    protected:
        void splitNode(int node_idx) override
        {
            uint32_t node_start = this->leafDataPool[node_idx].start;
            uint32_t node_end = this->leafDataPool[node_idx].end;
            uint16_t node_depth = this->leafDataPool[node_idx].depth;

            int size = node_end - node_start;
            if (size <= 1)
                return;

            projectionValues.resize(size);

            const int col = this->offset + node_depth;
            const Scalar *col_ptr = this->rndm_projections->data() + std::size_t(col) * this->rndm_projections->rows();

            for (int i = 0; i < size; ++i)
            {
                int idx = this->indices[node_start + i];
                projectionValues[i] = static_cast<float>(col_ptr[idx]);
            }

            kde->update(projectionValues);
            std::optional<float> kde_result = kde->findDensityMinimum(this->min_ratio);

            float split_value;
            if (!kde_result.has_value())
            {
                size_t m = projectionValues.size() / 2;
                std::nth_element(projectionValues.begin(),
                                 projectionValues.begin() + m,
                                 projectionValues.end());
                split_value = projectionValues[m];
            }
            else
            {
                split_value = kde_result.value();
            }

            this->routingPool[node_idx].splitValue = split_value;

            auto partitionPoint = std::partition(
                this->indices.begin() + node_start,
                this->indices.begin() + node_end,
                [&](uint32_t idx)
                {
                    return static_cast<float>(col_ptr[idx]) < split_value;
                });

            int mid = static_cast<int>(std::distance(this->indices.begin(), partitionPoint));

            if (static_cast<uint32_t>(mid) == node_start || static_cast<uint32_t>(mid) == node_end)
            {
                size_t m = node_start + (size / 2);

                std::nth_element(
                    this->indices.begin() + node_start,
                    this->indices.begin() + m,
                    this->indices.begin() + node_end,
                    [&](uint32_t a, uint32_t b)
                    {
                        return static_cast<float>(col_ptr[a]) < static_cast<float>(col_ptr[b]);
                    });

                mid = static_cast<int>(m);
                split_value = static_cast<float>(col_ptr[this->indices[mid]]);

                this->routingPool[node_idx].splitValue = split_value;
            }

            int leftKey = ++(this->keyCounter);
            int rightKey = ++(this->keyCounter);

            this->allocChildren(node_idx, leftKey, node_start, mid, rightKey, mid, node_end);
        }

    public:
        KdeBinarySplitTree(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> &rndm_ref,
                           int index, int tree_depth, int split_depth, float search_space, float bw_modifier, float min_ratio)
            : BinarySplitTreeBase<Scalar>(rndm_ref, index, tree_depth, split_depth, search_space),
              split_data_bandwidth(bw_modifier),
              min_ratio(min_ratio)

        {
        }

        KdeBinarySplitTree(std::istream &is, int n_points, float bw_modifier, float min_ratio)
            : BinarySplitTreeBase<Scalar>(is, n_points),
              split_data_bandwidth(bw_modifier),
              min_ratio(min_ratio)
        {
        }

        int splitTree(bool approximate_search_space_size) override
        {
            if (!this->rndm_projections)
                throw std::runtime_error("splitTree: tree has no projections (deserialized trees cannot be re-split)");

            kde = std::make_unique<fftkde::KdeFFT<float>>(1024, "silverman", true, split_data_bandwidth);
            projectionValues.reserve(this->rndm_projections->rows());

            int result_depth = BinarySplitTreeBase<Scalar>::splitTree(approximate_search_space_size);

            kde.reset();
            projectionValues.clear();
            projectionValues.shrink_to_fit();

            return result_depth;
        }
    };

}

#include "drpf_backend.h"

namespace drpf
{

    /**
     * @brief Dense Random Projection Forest.
     */
    class DRPF
    {
    public:
        using ANNResult = DrpfBackend::ANNResult;

        enum class Device
        {
            CPU,
            GPU
        };

    private:
        const float *_data_ptr = nullptr;
        long _rows = 0;
        long _cols = 0;

        SrhtProjection srht;

        Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> rndm_projections;

        Eigen::VectorXf norms;

        int numTrees;
        int split_depth;
        int internal_depth;
        int true_depth;
        int max_search_buffer_size = 0;
        std::vector<std::unique_ptr<KdeBinarySplitTree<Eigen::half>>> forest;

        float search_space;
        bool approximate_search_space_size;

        int seed;
        float split_data_bandwidth;
        float min_ratio;

        Device device_kind;

        std::unique_ptr<DrpfBackend> backend;

        std::vector<std::unique_ptr<KdeBinarySplitTree<Eigen::half>>> buildForest(
            const Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> &projections,
            int numTrees,
            int internalDepth,
            int splitDepth,
            float search_space,
            bool greedy)
        {
            std::vector<std::unique_ptr<KdeBinarySplitTree<Eigen::half>>> forest(numTrees);
            std::vector<int> depths(numTrees, 0);

            std::exception_ptr first_error;

#pragma omp parallel for schedule(dynamic)
            for (int idx = 0; idx < numTrees; ++idx)
            {
                if (first_error)
                    continue;

                try
                {
                    auto tree = std::make_unique<KdeBinarySplitTree<Eigen::half>>(
                        projections, idx, internalDepth, splitDepth, search_space,
                        split_data_bandwidth, min_ratio);

                    depths[idx] = tree->splitTree(greedy);
                    forest[idx] = std::move(tree);
                }
                catch (...)
                {
#pragma omp critical(drpf_build_error)
                    if (!first_error)
                        first_error = std::current_exception();
                }
            }

            if (first_error)
                std::rethrow_exception(first_error);

            for (auto d : depths)
                true_depth = d > true_depth ? d : true_depth;

            return forest;
        }

        void compact()
        {
            const int packed_depth = true_depth;
            if (packed_depth >= internal_depth)
                return;

            std::vector<int> packed_blocks;
            std::vector<int> packed_sel;

            const size_t packed_cols =
                static_cast<size_t>(numTrees) * (packed_depth + 1);
            packed_blocks.reserve(packed_cols);
            packed_sel.reserve(packed_cols);

            for (int tree = 0; tree < numTrees; ++tree)
            {
                const int old_offset = tree * (internal_depth + 1);

                for (int d = 0; d <= packed_depth; ++d)
                {
                    packed_blocks.push_back(srht.col_block[old_offset + d]);
                    packed_sel.push_back(srht.col_sel[old_offset + d]);
                }

                forest[tree]->updateOffset(packed_depth);
            }

            srht.col_block = std::move(packed_blocks);
            srht.col_sel = std::move(packed_sel);

            max_search_buffer_size = 0;
            for (const auto &tree : forest)
                max_search_buffer_size += tree->getMaxLeafSize();

            internal_depth = packed_depth;
        }

    public:
        DRPF(int num_trees = 5,
             int split_depth = 3,
             int no_of_ss = 0,
             bool approximate_search_space_size = false,
             float bw_modifier = 0.1f,
             int seed = 0,
             float min_ratio = 0.33333f,
             int num_threads = 0,
             Device device_kind = Device::CPU)
            : numTrees(num_trees), split_depth(split_depth), search_space(no_of_ss),
              approximate_search_space_size(approximate_search_space_size),
              seed(seed), split_data_bandwidth(bw_modifier), min_ratio(min_ratio), device_kind(device_kind)
        {
            if (num_trees <= 0)
                throw std::invalid_argument("num_trees must be greater than 0.");

            if (split_depth <= 0)
                throw std::invalid_argument("split_depth must be greater than 0.");

            if (no_of_ss < 0)
                throw std::invalid_argument("no_of_ss (search space) cannot be negative.");

            if (bw_modifier <= 0.0f)
                throw std::invalid_argument("bw_modifier must be strictly positive (> 0.0).");

            if (min_ratio <= 0.0f || min_ratio >= 0.5f)
                throw std::invalid_argument("min_ratio must be between 0.0 and 0.5 (exclusive).");

            if (num_threads > 0)
            {
                omp_set_num_threads(num_threads);
                Eigen::setNbThreads(num_threads);
            }
            else
                Eigen::setNbThreads(omp_get_max_threads());

            internal_depth = static_cast<int>(std::ceil(split_depth * 1.7095112913514551) + 1);
            true_depth = 0;
        }

        void index(const float *data_ptr, size_t length, int dimensions)
        {
            if (dimensions <= 0)
                throw std::invalid_argument("dimensions must be greater than 0.");
            if (length == 0)
                throw std::invalid_argument("data cannot be empty.");
            if (length % static_cast<size_t>(dimensions) != 0)
                throw std::invalid_argument("length is not a multiple of dimensions.");

            this->_data_ptr = data_ptr;
            this->_rows = static_cast<long>(length) / dimensions;
            this->_cols = dimensions;

            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                data(_data_ptr, _rows, _cols);

            this->search_space = this->search_space == 0
                                     ? static_cast<float>(data.rows()) / std::pow(2.0f, split_depth)
                                     : this->search_space;

            const int proj_cols = (internal_depth + 1) * numTrees;

            generateSrhtProjection(srht, _cols, proj_cols, seed);

            projectDataSrht(srht, data.data(), _rows, _cols, rndm_projections);
            norms = data.rowwise().squaredNorm();

            forest = buildForest(rndm_projections, numTrees, internal_depth,
                                 split_depth, search_space, approximate_search_space_size);

            compact();

            rndm_projections.resize(0, 0);

            if (device_kind == Device::GPU)
            {
#ifdef USE_CUDA
                backend = std::make_unique<DRPFBackendGPU>(
                    _data_ptr, _rows, _cols,
                    &srht,
                    &norms,
                    forest, max_search_buffer_size);
#else
                throw std::runtime_error("GPU backend requested but built without USE_CUDA");
#endif
            }
            else
            {
                backend = std::make_unique<DRPFBackendCPU>(
                    _data_ptr, _rows, _cols,
                    &srht, &norms, &forest,
                    numTrees, max_search_buffer_size);
            }
        }

        ~DRPF() = default;

        std::vector<int> getLeafNodeSizes(int index = -1) const
        {
            if (index < -1 || (index != -1 && static_cast<size_t>(index) >= forest.size()))
                throw std::invalid_argument("Illegal Index");

            std::vector<int> sizes;

            if (search_space > 1.0f)
            {
                int estimated_leaves_per_tree = _rows / static_cast<int>(search_space);
                int multiplier = (index == -1) ? numTrees : 1;
                sizes.reserve(estimated_leaves_per_tree * multiplier);
            }

            if (index == -1)
            {
                for (const auto &tree : forest)
                {
                    if (tree)
                        tree->collectLeafSizes(sizes);
                }
            }
            else
            {
                if (forest[index])
                    forest[index]->collectLeafSizes(sizes);
            }

            return sizes;
        }

        std::vector<std::pair<int, int>> getForestIndices(const float *query_ptr, std::size_t length, int index = -1) const
        {
            if (index < -1 || (index != -1 && static_cast<size_t>(index) >= forest.size()))
                throw std::invalid_argument("Illegal Index");

            const Eigen::Map<const Eigen::RowVectorXf> query(query_ptr, 1, length);

            Eigen::RowVectorXf projectedQuery(static_cast<Eigen::Index>(srht.col_sel.size()));
            std::vector<float> work(srht.P);

            srhtProject(srht, query_ptr, work.data(), projectedQuery.data());

            std::vector<std::pair<int, int>> indices;
            size_t reserve_size = (index == -1) ? this->max_search_buffer_size : (this->max_search_buffer_size / forest.size());
            indices.reserve(reserve_size);

            if (index == -1)
            {
                for (size_t idx = 0; idx < forest.size(); ++idx)
                {
                    if (forest[idx])
                    {
                        auto leaf_span = forest[idx]->getLeafIndices(projectedQuery.data());
                        for (size_t i = 0; i < leaf_span.size(); ++i)
                        {
                            indices.emplace_back(static_cast<int>(idx), leaf_span[i]);
                        }
                    }
                }
            }
            else
            {
                if (forest[index])
                {
                    auto leaf_span = forest[index]->getLeafIndices(projectedQuery.data());
                    for (size_t i = 0; i < leaf_span.size(); ++i)
                    {
                        indices.emplace_back(index, leaf_span[i]);
                    }
                }
            }

            return indices;
        }

        ANNResult ann_batch(const float *queries, int n_queries, int dim, int k, int votes)
        {
            if (votes <= 0)
                throw std::invalid_argument("votes must be a positive integer (>= 1).");

            if (k <= 0)
                throw std::invalid_argument("k must be a positive integer (>= 1).");

            return backend->ann_batch(queries, n_queries, dim, k, votes);
        }

        ANNResult ann(const float *query_ptr, std::size_t length, int k, int votes)
        {
            if (votes <= 0)
                throw std::invalid_argument("votes must be a positive integer (>= 1).");

            if (k <= 0)
                throw std::invalid_argument("k must be a positive integer (>= 1).");

            return backend->ann(query_ptr, static_cast<int>(length), k, votes);
        }

        void save(std::ostream &os) const
        {
            if (!backend)
                throw std::runtime_error("save: index has not been built");

            os.exceptions(std::ios::failbit | std::ios::badbit);

            DrpfHeader header{};
            std::memcpy(header.magic, DRPF_MAGIC, sizeof(DRPF_MAGIC));
            header.version = DRPF_FORMAT_VERSION;
            header.scalar_size = static_cast<uint32_t>(sizeof(Eigen::half));
            header.rows = static_cast<int64_t>(_rows);
            header.cols = static_cast<int64_t>(_cols);

            writePod(os, header);

            writePod(os, numTrees);
            writePod(os, split_depth);
            writePod(os, internal_depth);
            writePod(os, true_depth);
            writePod(os, max_search_buffer_size);
            writePod(os, search_space);
            writePod(os, approximate_search_space_size);
            writePod(os, seed);
            writePod(os, split_data_bandwidth);
            writePod(os, min_ratio);

            writePod(os, srht.dim);
            writePod(os, srht.P);
            writePod(os, srht.nblocks);

            writePodArray(os, srht.signs.data(), srht.signs.size());
            writePodArray(os, srht.col_block.data(), srht.col_block.size());
            writePodArray(os, srht.col_sel.data(), srht.col_sel.size());

            writePodArray(os, norms.data(), static_cast<uint64_t>(norms.size()));

            writePod(os, static_cast<uint32_t>(forest.size()));
            for (const auto &tree : forest)
            {
                if (!tree)
                    throw std::runtime_error("save: null tree in forest");
                tree->serialise(os);
            }
        }

        void save(const std::string &path) const
        {
            std::ofstream os(path, std::ios::binary);
            if (!os)
                throw std::runtime_error("save: cannot open '" + path + "' for writing");
            save(os);
        }

        void load(std::istream &is, const float *data_ptr, size_t length, int dimensions)
        {

            if (dimensions <= 0)
                throw std::invalid_argument("dimensions must be greater than 0.");
            if (length == 0)
                throw std::invalid_argument("data cannot be empty.");
            if (length % static_cast<size_t>(dimensions) != 0)
                throw std::invalid_argument("length is not a multiple of dimensions.");

            backend.reset();

            DrpfHeader header{};
            readPod(is, header);

            if (std::memcmp(header.magic, DRPF_MAGIC, sizeof(DRPF_MAGIC)) != 0)
                throw std::runtime_error("load: not a DRPF index file");

            if (header.version != DRPF_FORMAT_VERSION)
                throw std::runtime_error("load: unsupported format version " +
                                         std::to_string(header.version));

            if (header.scalar_size != sizeof(Eigen::half))
                throw std::runtime_error("load: index built with a different scalar type");

            const int64_t given_rows = static_cast<int64_t>(length) / dimensions;
            if (header.rows != given_rows || header.cols != dimensions)
                throw std::runtime_error("load: dataset shape does not match the index");

            _data_ptr = data_ptr;
            _rows = static_cast<long>(header.rows);
            _cols = static_cast<long>(header.cols);

            readPod(is, numTrees);
            readPod(is, split_depth);
            readPod(is, internal_depth);
            readPod(is, true_depth);
            readPod(is, max_search_buffer_size);
            readPod(is, search_space);
            readPod(is, approximate_search_space_size);
            readPod(is, seed);
            readPod(is, split_data_bandwidth);
            readPod(is, min_ratio);

            if (numTrees <= 0 || internal_depth < 0)
                throw std::runtime_error("load: invalid index parameters");

            readPod(is, srht.dim);
            readPod(is, srht.P);
            readPod(is, srht.nblocks);

            const int initial_depth =
                static_cast<int>(std::ceil(split_depth * 1.7095112913514551) + 1);
            const uint64_t initial_cols =
                static_cast<uint64_t>(numTrees) * (initial_depth + 1);
            const uint64_t packed_cols =
                static_cast<uint64_t>(numTrees) * (internal_depth + 1);

            int expected_P = 1;
            while (expected_P < _cols)
                expected_P <<= 1;

            if (srht.dim != _cols ||
                srht.P != expected_P ||
                srht.nblocks != (initial_cols + srht.P - 1) / srht.P)
            {
                throw std::runtime_error("load: invalid SRHT metadata");
            }

            const uint64_t sign_count =
                static_cast<uint64_t>(srht.nblocks) * srht.P;

            readPodArray(is, srht.signs, sign_count);
            readPodArray(is, srht.col_block, packed_cols);
            readPodArray(is, srht.col_sel, packed_cols);

            if (srht.signs.size() != sign_count ||
                srht.col_block.size() != packed_cols ||
                srht.col_sel.size() != packed_cols)
            {
                throw std::runtime_error("load: SRHT metadata size mismatch");
            }

            for (size_t i = 0; i < packed_cols; ++i)
            {
                if (srht.col_block[i] < 0 || srht.col_block[i] >= srht.nblocks ||
                    srht.col_sel[i] < 0 || srht.col_sel[i] >= srht.P)
                {
                    throw std::runtime_error("load: invalid SRHT column mapping");
                }
            }

            {
                std::vector<float> norms_buf;
                readPodArray(is, norms_buf, static_cast<uint64_t>(_rows));
                if (norms_buf.size() != static_cast<size_t>(_rows))
                    throw std::runtime_error("load: norms length mismatch");
                norms = Eigen::Map<const Eigen::VectorXf>(norms_buf.data(), _rows);
            }

            uint32_t tree_count = 0;
            readPod(is, tree_count);
            if (tree_count != static_cast<uint32_t>(numTrees))
                throw std::runtime_error("load: forest size does not match numTrees");

            forest.clear();
            forest.reserve(tree_count);
            for (uint32_t i = 0; i < tree_count; ++i)
            {
                forest.push_back(std::make_unique<KdeBinarySplitTree<Eigen::half>>(
                    is, static_cast<int>(_rows), split_data_bandwidth, min_ratio));
            }

            rndm_projections.resize(0, 0);

            if (device_kind == Device::GPU)
            {
#ifdef USE_CUDA
                backend = std::make_unique<DRPFBackendGPU>(
                    _data_ptr, _rows, _cols,
                    &srht, &norms,
                    forest, max_search_buffer_size);
#else
                throw std::runtime_error("GPU backend requested but built without USE_CUDA");
#endif
            }
            else
            {
                backend = std::make_unique<DRPFBackendCPU>(
                    _data_ptr, _rows, _cols,
                    &srht, &norms, &forest,
                    numTrees, max_search_buffer_size);
            }
        }

        void load(const std::string &path, const float *data_ptr, size_t length, int dimensions)
        {
            std::ifstream is(path, std::ios::binary);
            if (!is)
                throw std::runtime_error("load: cannot open '" + path + "'");
            load(is, data_ptr, length, dimensions);
        }
    };
}