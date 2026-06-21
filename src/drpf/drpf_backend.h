#pragma once

#define EIGEN_NO_DEBUG

#define EIGEN_USE_OPENMP

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
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
#include <queue>
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

    virtual ANNResult ann_batch(const float *queries, int n_queries, int dim, int votes, int k) = 0;

    virtual ANNResult ann(const float *query, int dim, int votes, int k) = 0;
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
        std::vector<uint32_t> state;
        std::vector<int> candidates;
        uint32_t generation = 0;
        std::vector<std::pair<float, int>> heap;

        void resize(int rows, int max_candidates, int k)
        {
            if (state.size() != static_cast<size_t>(rows))
            {
                state.assign(rows, 0);
                generation = 0;
            }

            if (candidates.capacity() < static_cast<size_t>(max_candidates))
                candidates.reserve(max_candidates);

            if (heap.capacity() < static_cast<size_t>(k + 1))
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
        SearchContext &ctx, int votes)
    {
        int rows = static_cast<int>(data.rows());
        ctx.resize(rows, max_search_buffer_size, k);

        ctx.generation++;
        if (ctx.generation >= (1 << 24))
        {
            std::fill(ctx.state.begin(), ctx.state.end(), 0);
            ctx.generation = 1;
        }

        ctx.candidates.clear();

        int *cands_ptr = ctx.candidates.data();
        int num_cands = 0;

        uint32_t gen_mask = ctx.generation << 8;
        uint32_t target_votes = static_cast<uint32_t>(std::max(1, votes));

        const int prefetch_dist = 32;

        for (int i = 0; i < numTrees; i++)
        {
            auto leafSpan = (*forest)[i]->getLeafIndices(proj_q);
            const unsigned int *leaf_ptr = leafSpan.data();
            size_t leaf_size = leafSpan.size();

            for (size_t j = 0; j < leaf_size; ++j)
            {

                if (j + prefetch_dist < leaf_size)
                {
                    PREFETCH(&ctx.state[leaf_ptr[j + prefetch_dist]]);
                }

                int idx = leaf_ptr[j];
                uint32_t current_state = ctx.state[idx];

                uint32_t is_same_gen = ((current_state >> 8) == ctx.generation);

                uint32_t new_votes = ((current_state & 0xFF) * is_same_gen) + 1;

                ctx.state[idx] = gen_mask | new_votes;

                if (new_votes == target_votes)
                {
                    cands_ptr[num_cands++] = idx;
                }
            }
        }

        if (num_cands == 0)
        {
            std::fill(results, results + k, -1);
            std::fill(distances_sq, distances_sq + k, std::numeric_limits<float>::max());
            return;
        }

        float q_norm = q.squaredNorm();
        float q_norm_sqrt = std::sqrt(q_norm);
        int dim = static_cast<int>(data.cols());
        const float *data_raw = data.data();

        ctx.heap.clear();
        float heap_worst = std::numeric_limits<float>::max();

        for (size_t i = 0; i < num_cands; ++i)
        {
            if (i + prefetch_dist < num_cands)
            {
                int next_idx = cands_ptr[i + prefetch_dist];
                PREFETCH(data_raw + static_cast<size_t>(next_idx) * dim);
            }

            int idx = cands_ptr[i];

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

    ANNResult ann(const float *query, int dim, int votes, int k) override
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
                        ctx, votes);

        return out;
    }

    ANNResult ann_batch(const float *queries, int n_queries, int dim, int votes, int k) override
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
                                ctx, votes);
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
                std::vector<GPUNode> &flat_nodes,
                std::vector<GPULeafInfo> &flat_leaf_info,
                std::vector<unsigned int> &flat_leaf_data)
    {
        if (root_cpu_idx == -1)
            return -1;

        const auto &routingPool = tree->getRoutingPool();
        const auto &leafDataPool = tree->getLeafDataPool();
        const auto &global_indices = tree->getIndices();

        int root_flat_idx = static_cast<int>(flat_nodes.size());
        flat_nodes.emplace_back();
        flat_leaf_info.emplace_back();

        std::queue<std::pair<int, int>> q;
        q.push({root_cpu_idx, root_flat_idx});

        while (!q.empty())
        {
            auto [curr_cpu_idx, curr_flat_idx] = q.front();
            q.pop();

            const RoutingNode &cpu_rnode = routingPool[curr_cpu_idx];
            GPUNode gpu_node{};
            GPULeafInfo leaf_info{-1, 0};

            if (cpu_rnode.left == -1)
            {
                const LeafData &cpu_ldata = leafDataPool[curr_cpu_idx];
                gpu_node.left_child = -1;

                leaf_info.leaf_start_idx = static_cast<int>(flat_leaf_data.size());
                leaf_info.leaf_size = static_cast<int>(cpu_ldata.getSize());

                for (uint32_t i = cpu_ldata.start; i < cpu_ldata.end; ++i)
                    flat_leaf_data.push_back(global_indices[i]);
            }
            else
            {
                gpu_node.split_val = cpu_rnode.splitValue;

                int left_flat_idx = static_cast<int>(flat_nodes.size());
                flat_nodes.emplace_back();
                flat_leaf_info.emplace_back();
                gpu_node.left_child = left_flat_idx;
                q.push({cpu_rnode.left, left_flat_idx});

                int right_flat_idx = static_cast<int>(flat_nodes.size());
                flat_nodes.emplace_back();
                flat_leaf_info.emplace_back();
                q.push({cpu_rnode.left + 1, right_flat_idx});
            }

            flat_nodes[curr_flat_idx] = gpu_node;
            flat_leaf_info[curr_flat_idx] = leaf_info;
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
        std::vector<GPULeafInfo> flat_leaf_info;
        std::vector<unsigned int> flat_leaf_data;
        std::vector<int> tree_roots;
        std::vector<int> tree_offsets;

        int proj_cols_per_tree = (int)proj_matrix->cols() / (int)cpu_forest.size();
        for (const auto &tree : cpu_forest)
        {
            int tree_offset = tree->getOffset();
            int root_idx = flatten(tree.get(), tree->getRootIndex(),
                                   flat_nodes, flat_leaf_info, flat_leaf_data);
            tree_roots.push_back(root_idx);
            tree_offsets.push_back(tree_offset);
        }

        FlattenedForest ff;
        ff.nodes = flat_nodes.data();
        ff.leaf_info = flat_leaf_info.data();
        ff.leaf_data = flat_leaf_data.data();
        ff.tree_roots = tree_roots.data();
        ff.tree_offsets = tree_offsets.data();
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

    ANNResult ann(const float *query, int dim, int votes, int k) override
    {
        return _cpu_fallback->ann(query, dim, votes, k);
    }

    ANNResult ann_batch(const float *queries, int n_queries, int dim, int votes, int k) override
    {
        constexpr int GPU_CROSSOVER = 64;
        if (n_queries < GPU_CROSSOVER)
        {
            return _cpu_fallback->ann_batch(queries, n_queries, dim, votes, k);
        }

        ANNResult out;
        out.indices.resize(n_queries * k);
        out.distances_sq.resize(n_queries * k);
        search_gpu_batch(_h, queries, n_queries, dim, votes, k,
                         out.indices.data(), out.distances_sq.data());
        return out;
    }
};

#endif