#pragma once

// Disable Eigen debug assertions for maximum performance
#define EIGEN_NO_DEBUG
// Enable OpenMP parallelization within Eigen operations
#define EIGEN_USE_OPENMP

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

#include "binarySplitTree.h"
#include "fft_kde.h"
#include "utils.h"

#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch(addr)
#else
#define PREFETCH(addr)
#endif

#ifdef _MSC_VER
#define attribute(x)
#define restrict restrict
#endif

/**
 * @brief KDE-based Implementation of the Binary Split Tree.
 */
template <typename Scalar = Eigen::half>
class KdeBinarySplitTree : public BinarySplitTreeBase<Scalar>
{
private:
    std::vector<float> projectionValues;
    std::unique_ptr<KdeFFT<float>> kde;
    float split_data_bandwidth;
    float min_ratio;

protected:
    void splitNode(int node_idx) override
    {
        uint32_t node_start = this->nodePool[node_idx].start;
        uint32_t node_end = this->nodePool[node_idx].end;
        uint16_t node_depth = this->nodePool[node_idx].depth;

        int size = node_end - node_start;
        if (size <= 1)
            return;

        projectionValues.resize(size);

        int local_col = node_depth % (this->treeDepth + 1);
        const int col = this->offset + local_col;
        const Scalar *col_ptr = this->rndm_projections.data() + std::size_t(col) * this->rndm_projections.rows();

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

        this->nodePool[node_idx].splitValue = split_value;

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
            this->nodePool[node_idx].to_split = false;
            this->nodePool[node_idx].left = -1;
            this->nodePool[node_idx].right = -1;
            return;
        }

        int leftKey = ++(this->keyCounter);
        int rightKey = ++(this->keyCounter);

        int leftIdx = this->allocNode(leftKey, node_start, mid);
        int rightIdx = this->allocNode(rightKey, mid, node_end);

        this->insertNodeAt(node_idx, leftIdx);
        this->insertNodeAt(node_idx, rightIdx);
    }

public:
    KdeBinarySplitTree(const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> &rndm_ref,
                       int index, int tree_depth, int split_depth, float search_space, float bw_modifier, float min_ratio)
        : BinarySplitTreeBase<Scalar>(rndm_ref, index, tree_depth, split_depth, search_space),
          split_data_bandwidth(bw_modifier),
          min_ratio(min_ratio)

    {
    }

    int splitTree(bool approximate_search_space_size) override
    {
        kde = std::make_unique<KdeFFT<float>>(1024, "silverman", true, split_data_bandwidth);
        projectionValues.reserve(this->rndm_projections.rows());

        int result_depth = BinarySplitTreeBase<Scalar>::splitTree(approximate_search_space_size);

        kde.reset();
        projectionValues.clear();
        projectionValues.shrink_to_fit();

        return result_depth;
    }
};

/**
 * @brief Dense Random Projection Forest.
 * * Combines multiple BinarySplitTrees with Product Quantization (PQ) for
 * efficient Approximate Nearest Neighbor search.
 */
class DRPF
{
private:
    const float *_data_ptr = nullptr;
    long _rows = 0;
    long _cols = 0;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> projectionMatrix;

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

    struct Candidate
    {
        int idx;
        float score;
    };

    struct SearchContext
    {
        std::vector<int> candidates;
        std::vector<float> scores;
        std::vector<uint16_t> seen;
        uint16_t generation = 0;
        std::vector<int> indices_buffer;

        void resize(int rows, int max_candidates)
        {
            if (seen.size() != rows)
            {
                seen.assign(rows, 0);
                generation = 0;
            }

            if (candidates.capacity() < max_candidates)
                candidates.reserve(max_candidates);
        }
    };

    std::vector<std::unique_ptr<KdeBinarySplitTree<Eigen::half>>> buildForest(
        const Eigen::Ref<const Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> &projections,
        int numTrees,
        int internalDepth,
        int splitDepth,
        float search_space,
        bool greedy)
    {
        std::vector<std::unique_ptr<KdeBinarySplitTree<Eigen::half>>> forest(numTrees);
        std::vector<int> depths(numTrees, 0);

#pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < numTrees; ++idx)
        {
            auto tree = std::make_unique<KdeBinarySplitTree<Eigen::half>>(
                projections, idx, internalDepth, splitDepth, search_space, split_data_bandwidth, min_ratio);

            depths[idx] = tree->splitTree(greedy);
            forest[idx] = std::move(tree);
        }

        for (auto d : depths)
            true_depth = d > true_depth ? d : true_depth;

        return forest;
    }

    void compact()
    {
        int packed_depth = true_depth;
        if (packed_depth >= internal_depth)
            return;

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
            packedMatrix(projectionMatrix.rows(), numTrees * (packed_depth + 1));

        for (int t = 0; t < numTrees; ++t)
        {
            int old_offset = t * (internal_depth + 1);
            int new_offset = t * (packed_depth + 1);

            packedMatrix.block(0, new_offset, projectionMatrix.rows(), packed_depth + 1) =
                projectionMatrix.block(0, old_offset, projectionMatrix.rows(), packed_depth + 1);

            forest[t]->updateOffset(packed_depth);
        }

        max_search_buffer_size = 0;
        for (const auto &tree : forest)
            max_search_buffer_size += tree->getMaxLeafSize();

        projectionMatrix = std::move(packedMatrix);
        this->internal_depth = packed_depth;
    }

public:
    DRPF(int num_trees = 5,
         int split_depth = 3,
         int no_of_ss = 0,
         bool approximate_search_space_size = false,
         float bw_modifier = 0.1f,
         int seed = 0,
         float min_ratio = 0.33333f,
         int num_threads = 0)
        : numTrees(num_trees), split_depth(split_depth), search_space(no_of_ss),
          approximate_search_space_size(approximate_search_space_size),
          seed(seed), split_data_bandwidth(bw_modifier), min_ratio(min_ratio)
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

    void index(const float *data_ptr,
               size_t length,
               int dimensions)
    {
        this->_data_ptr = data_ptr;
        this->_rows = static_cast<long>(length) / dimensions;
        this->_cols = dimensions;

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> data(_data_ptr, _rows, _cols);

        this->search_space = this->search_space == 0 ? static_cast<float>(data.rows()) / std::pow(2.0f, split_depth) : this->search_space;
        projectionMatrix.resize(data.cols(), (internal_depth + 1) * numTrees);
        generateRandomMatrix(projectionMatrix, seed);

        rndm_projections.resize(data.rows(), (internal_depth + 1) * numTrees);
        rndm_projections.noalias() = (data * projectionMatrix).cast<Eigen::half>();
        norms = data.rowwise().squaredNorm();

        forest = buildForest(rndm_projections, numTrees, internal_depth, split_depth, search_space, approximate_search_space_size);
        compact();
    }

    ~DRPF() = default;

    std::vector<int> getLeafNodeSizes(int index = -1) const
    {
        if (index < -1 || (index != -1 && static_cast<size_t>(index) >= forest.size()))
            throw std::invalid_argument("Illegal Index");

        std::vector<int> sizes;

        if (search_space > 0)
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

        Eigen::RowVectorXf projectedQuery(projectionMatrix.cols());
        projectedQuery.noalias() = query * projectionMatrix;

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

    inline void exact_ann(const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &data,
                          const Eigen::Ref<const Eigen::RowVectorXf> q,
                          const float *__restrict proj_q,
                          int k,
                          int *results,
                          SearchContext &ctx) __attribute__((hot, always_inline))
    {
        int rows = static_cast<int>(data.rows());
        ctx.resize(rows, max_search_buffer_size);

        ctx.generation++;
        if (ctx.generation == 0)
        {
            std::fill(ctx.seen.begin(), ctx.seen.end(), 0);
            ctx.generation = 1;
        }

        ctx.candidates.clear();

        for (int i = 0; i < numTrees; i++)
        {
            auto leafSpan = forest[i]->getLeafIndices(proj_q);
            const unsigned int *leaf_ptr = leafSpan.data();
            size_t leaf_size = leafSpan.size();

            for (size_t j = 0; j < leaf_size; ++j)
            {
                int idx = leaf_ptr[j];
                if (ctx.seen[idx] != ctx.generation)
                {
                    ctx.seen[idx] = ctx.generation;
                    ctx.candidates.push_back(idx);
                }
            }
        }

        size_t n_cands = ctx.candidates.size();
        if (n_cands == 0)
        {
            std::fill(results, results + k, -1);
            return;
        }

        ctx.scores.resize(n_cands);

        float q_norm = q.squaredNorm();
        const float *data_raw = data.data();
        int dim = static_cast<int>(data.cols());
        const int prefetch_dist = 32;

        for (size_t i = 0; i < n_cands; ++i)
        {
            if (i + prefetch_dist < n_cands)
            {
                int next_idx = ctx.candidates[i + prefetch_dist];
                PREFETCH(data_raw + static_cast<size_t>(next_idx) * dim);
            }

            int idx = ctx.candidates[i];
            float dot = data.row(idx).dot(q);
            ctx.scores[i] = norms[idx] + q_norm - 2 * dot;
        }

        if (ctx.indices_buffer.size() < n_cands)
            ctx.indices_buffer.resize(n_cands);

        std::iota(ctx.indices_buffer.begin(), ctx.indices_buffer.begin() + n_cands, 0);

        int k_eff = std::min((int)n_cands, k);

        std::nth_element(ctx.indices_buffer.begin(),
                         ctx.indices_buffer.begin() + k_eff,
                         ctx.indices_buffer.begin() + n_cands,
                         [&](int a, int b)
                         { return ctx.scores[a] < ctx.scores[b]; });

        std::sort(ctx.indices_buffer.begin(),
                  ctx.indices_buffer.begin() + k_eff,
                  [&](int a, int b)
                  { return ctx.scores[a] < ctx.scores[b]; });

        for (int i = 0; i < k_eff; i++)
            results[i] = ctx.candidates[ctx.indices_buffer[i]];
        for (int i = k_eff; i < k; i++)
            results[i] = -1;
    }

    std::vector<int> ann_batch(const float *queries, int n_queries, int dim, int k)
    {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> data(_data_ptr, _rows, _cols);
        if (dim != data.cols())
            throw std::invalid_argument("Query dimensionality mismatch");

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Q(queries, n_queries, dim);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> projected = Q * projectionMatrix;

        const int projDim = projected.cols();
        const float *proj_data = projected.data();
        std::vector<int> results(n_queries * k);

#pragma omp parallel
        {
            SearchContext ctx;

#pragma omp for schedule(dynamic)
            for (int i = 0; i < n_queries; ++i)
            {
                const Eigen::Map<const Eigen::RowVectorXf> q(queries + i * dim, dim);
                const float *proj_q = proj_data + static_cast<size_t>(i) * projDim;
                this->exact_ann(data, q, proj_q, k, &results[i * k], ctx);
            }
        }

        return results;
    }

    std::vector<int> ann(const float *query_ptr, std::size_t length, int k)
    {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> data(_data_ptr, _rows, _cols);
        const Eigen::Map<const Eigen::RowVectorXf> query(query_ptr, 1, length);

        Eigen::RowVectorXf projectedQuery(projectionMatrix.cols());
        projectedQuery.noalias() = query * projectionMatrix;

        std::vector<int> results(k);

        SearchContext ctx;
        this->exact_ann(data, query, projectedQuery.data(), k, results.data(), ctx);

        return results;
    }
};