#pragma once

#define EIGEN_NO_DEBUG

#define EIGEN_USE_OPENMP

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch(addr)
#else
#define PREFETCH(addr)
#endif

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

template <typename Scalar>
class KdeBinarySplitTree;

class DrpfBackend
{

public:
    struct ANNResult
    {
        std::vector<int> indices;
        std::vector<float> distances_sq;
    };

    virtual ~DrpfBackend() = default;

    virtual ANNResult ann_batch(const float *queries, int n_queries, int dim, int k) = 0;

    virtual ANNResult ann(const float *query, int dim, int k) = 0;
};

class DRPFBackendCPU : public DrpfBackend
{
private:
    const float *_data_ptr;
    long _rows;
    long _cols;
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> *projectionMatrix;
    const Eigen::VectorXf *norms;
    const std::vector<std::unique_ptr<KdeBinarySplitTree<Eigen::half>>> *forest;
    int numTrees;
    int max_search_buffer_size;

    struct SearchContext
    {
        std::vector<int> candidates;
        std::vector<uint32_t> seen;
        uint32_t generation = 0;
        std::vector<std::pair<float, int>> heap;

        void resize(int rows, int max_candidates, int k)
        {
            if (seen.size() != rows)
            {
                seen.assign(rows, 0);
                generation = 0;
            }
            if (candidates.capacity() < max_candidates)
                candidates.reserve(max_candidates);
            if (heap.capacity() < k + 1)
                heap.reserve(k + 1);
        }
    };

    FORCE_INLINE void exact_ann(
        const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &data,
        const Eigen::Ref<const Eigen::RowVectorXf> q,
        const float *__restrict proj_q,
        int k,
        int *results,
        float *distances_sq,
        SearchContext &ctx)
    {
        int rows = static_cast<int>(data.rows());
        ctx.resize(rows, max_search_buffer_size, k);

        ctx.generation++;
        if (ctx.generation == 0)
        {
            std::fill(ctx.seen.begin(), ctx.seen.end(), 0);
            ctx.generation = 1;
        }

        ctx.candidates.clear();

        for (int i = 0; i < numTrees; i++)
        {
            auto leafSpan = (*forest)[i]->getLeafIndices(proj_q);
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
            std::fill(distances_sq, distances_sq + k, std::numeric_limits<float>::max());
            return;
        }

        float q_norm = q.squaredNorm();
        float q_norm_sqrt = std::sqrt(q_norm);
        int dim = static_cast<int>(data.cols());
        const float *data_raw = data.data();
        const int prefetch_dist = 32;

        ctx.heap.clear();
        float heap_worst = std::numeric_limits<float>::max();

        for (size_t i = 0; i < n_cands; ++i)
        {
            if (i + prefetch_dist < n_cands)
            {
                int next_idx = ctx.candidates[i + prefetch_dist];
                PREFETCH(data_raw + static_cast<size_t>(next_idx) * dim);
            }

            int idx = ctx.candidates[i];

            // Cauchy-Schwarz lower bound: dist >= (||x|| - ||q||)^2
            float x_norm_sqrt = std::sqrt((*norms)[idx]);
            float diff = x_norm_sqrt - q_norm_sqrt;
            float lower_bound = diff * diff;

            if (lower_bound >= heap_worst)
                continue;

            float dot = data.row(idx).dot(q);
            float dist = (*norms)[idx] + q_norm - 2.0f * dot;

            if (static_cast<int>(ctx.heap.size()) < k)
            {
                ctx.heap.push_back({dist, idx});
                std::push_heap(ctx.heap.begin(), ctx.heap.end());
                if (static_cast<int>(ctx.heap.size()) == k)
                    heap_worst = ctx.heap.front().first;
            }
            else if (dist < heap_worst)
            {
                std::pop_heap(ctx.heap.begin(), ctx.heap.end());
                ctx.heap.back() = {dist, idx};
                std::push_heap(ctx.heap.begin(), ctx.heap.end());
                heap_worst = ctx.heap.front().first;
            }
        }

        std::sort_heap(ctx.heap.begin(), ctx.heap.end());
        int k_eff = static_cast<int>(ctx.heap.size());

        for (int i = k_eff; i < k; ++i)
        {
            results[i] = -1;
            distances_sq[i] = std::numeric_limits<float>::max();
        }

        for (int i = 0; i < k_eff; ++i)
        {
            results[i] = ctx.heap[i].second;
            distances_sq[i] = ctx.heap[i].first;
        }
    }

public:
    DRPFBackendCPU(
        const float *data_ptr, long rows, long cols,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> *projMatrix,
        const Eigen::VectorXf *nrms,
        const std::vector<std::unique_ptr<KdeBinarySplitTree<Eigen::half>>> *f,
        int trees, int max_buffer)
        : _data_ptr(data_ptr), _rows(rows), _cols(cols),
          projectionMatrix(projMatrix), norms(nrms), forest(f),
          numTrees(trees), max_search_buffer_size(max_buffer) {}

    ANNResult ann(const float *query, int dim, int k) override
    {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> data(_data_ptr, _rows, _cols);
        if (dim != data.cols())
            throw std::invalid_argument("Query dimensionality mismatch");

        const Eigen::Map<const Eigen::RowVectorXf> q(query, dim);

        Eigen::RowVectorXf projectedQuery = q * (*projectionMatrix);

        ANNResult out;
        out.indices.resize(k);
        out.distances_sq.resize(k);

        SearchContext ctx;
        this->exact_ann(data, q, projectedQuery.data(), k,
                        out.indices.data(),
                        out.distances_sq.data(),
                        ctx);

        return out;
    }

    ANNResult ann_batch(const float *queries, int n_queries, int dim, int k) override
    {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> data(_data_ptr, _rows, _cols);
        if (dim != data.cols())
            throw std::invalid_argument("Query dimensionality mismatch");

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Q(queries, n_queries, dim);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> projected = Q * (*projectionMatrix);

        const int projDim = projected.cols();
        const float *proj_data = projected.data();

        ANNResult out;
        out.indices.resize(n_queries * k);
        out.distances_sq.resize(n_queries * k);

#pragma omp parallel
        {
            SearchContext ctx;

#pragma omp for schedule(dynamic)
            for (int i = 0; i < n_queries; ++i)
            {
                const Eigen::Map<const Eigen::RowVectorXf> q(queries + i * dim, dim);
                const float *proj_q = proj_data + static_cast<size_t>(i) * projDim;
                this->exact_ann(data, q, proj_q, k,
                                &out.indices[i * k],
                                &out.distances_sq[i * k],
                                ctx);
            }
        }

        return out;
    }
};

#ifdef USE_CUDA
#include "drpf_cuda.cuh"

class DRPFBackendGPU : public DrpfBackend
{
private:
    GPUDataHandle _h;
    std::unique_ptr<DRPFBackendCPU> _cpu_fallback;
    int _dim;

    using TreeType = KdeBinarySplitTree<Eigen::half>;

    int flatten(const TreeType *tree,
                int root_cpu_idx,
                int tree_offset,
                int proj_cols_per_tree,
                std::vector<GPUNode> &flat_nodes,
                std::vector<unsigned int> &flat_leaf_data)
    {
        if (root_cpu_idx == -1)
            return -1;

        const auto &pool = tree->getNodePool();
        const auto &global_indices = tree->getIndices();

        int root_flat_idx = static_cast<int>(flat_nodes.size());
        flat_nodes.emplace_back();

        std::queue<std::pair<int, int>> q;
        q.push({root_cpu_idx, root_flat_idx});

        while (!q.empty())
        {
            auto [curr_cpu_idx, curr_flat_idx] = q.front();
            q.pop();

            const auto &cpu_node = pool[curr_cpu_idx];
            GPUNode gpu_node;

            if (cpu_node.is_leaf)
            {
                gpu_node.left_child = -1;
                gpu_node.right_child = -1;

                gpu_node.leaf_start_idx = static_cast<int>(flat_leaf_data.size());
                gpu_node.leaf_size = static_cast<int>(cpu_node.getSize());

                for (uint32_t i = cpu_node.start; i < cpu_node.end; ++i)
                {
                    flat_leaf_data.push_back(global_indices[i]);
                }
            }
            else
            {
                int local_col = cpu_node.depth % proj_cols_per_tree;

                gpu_node.split_dim = tree_offset + local_col;
                gpu_node.split_val = cpu_node.splitValue;

                if (cpu_node.left != -1)
                {
                    int left_flat_idx = static_cast<int>(flat_nodes.size());
                    flat_nodes.emplace_back();
                    gpu_node.left_child = left_flat_idx;
                    q.push({cpu_node.left, left_flat_idx});
                }
                else
                {
                    gpu_node.left_child = -1;
                }

                if (cpu_node.right != -1)
                {
                    int right_flat_idx = static_cast<int>(flat_nodes.size());
                    flat_nodes.emplace_back();
                    gpu_node.right_child = right_flat_idx;
                    q.push({cpu_node.right, right_flat_idx});
                }
                else
                {
                    gpu_node.right_child = -1;
                }
            }

            flat_nodes[curr_flat_idx] = gpu_node;
        }

        return root_flat_idx;
    }

public:
    DRPFBackendGPU(
        const float *data, long rows, long cols,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> *proj_matrix,
        const Eigen::VectorXf *norms,
        const std::vector<std::unique_ptr<KdeBinarySplitTree<Eigen::half>>> &cpu_forest,
        int max_buffer)
        : _dim(cols)
    {
        std::vector<GPUNode> flat_nodes;
        std::vector<unsigned int> flat_leaf_data;
        std::vector<int> tree_roots;

        int proj_cols_per_tree = (int)proj_matrix->cols() / (int)cpu_forest.size();
        for (const auto &tree : cpu_forest)
        {
            int tree_offset = tree->getOffset();
            int root_idx = flatten(tree.get(), tree->getRootIndex(),
                                   tree_offset, proj_cols_per_tree,
                                   flat_nodes, flat_leaf_data);
            tree_roots.push_back(root_idx);
        }

        FlattenedForest ff;
        ff.nodes = flat_nodes.data();
        ff.leaf_data = flat_leaf_data.data();
        ff.tree_roots = tree_roots.data();
        ff.num_nodes = (int)flat_nodes.size();
        ff.num_leaf_data = (int)flat_leaf_data.size();
        ff.num_trees = (int)tree_roots.size();

        _h = setup_gpu_backend(
            data, rows, cols,
            proj_matrix->data(), (int)proj_matrix->cols(),
            norms->data(),
            ff, max_buffer);

        _cpu_fallback = std::make_unique<DRPFBackendCPU>(
            data, rows, cols,
            proj_matrix, norms, &cpu_forest,
            (int)cpu_forest.size(), max_buffer);
    }

    ~DRPFBackendGPU()
    {
        free_gpu_handle(_h);
    }

    ANNResult ann(const float *query, int dim, int k) override
    {
        return _cpu_fallback->ann(query, dim, k);
    }

    ANNResult ann_batch(const float *queries, int n_queries, int dim, int k) override
    {
        constexpr int GPU_CROSSOVER = 64;
        if (n_queries < GPU_CROSSOVER)
        {
            return _cpu_fallback->ann_batch(queries, n_queries, dim, k);
        }

        ANNResult out;
        out.indices.resize(n_queries * k);
        out.distances_sq.resize(n_queries * k);
        search_gpu_batch(_h, queries, n_queries, dim, k,
                         out.indices.data(), out.distances_sq.data());
        return out;
    }
};

#endif