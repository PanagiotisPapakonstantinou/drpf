#pragma once

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

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
#include <atomic>

#ifdef USE_CUDA
#include "drpf_cuda.cuh"
#endif

namespace drpf
{

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

        virtual ANNResult ann_batch(const float *queries, int n_queries, int dim, int k, int votes) = 0;

        virtual ANNResult ann(const float *query, int dim, int k, int votes) = 0;
    };

    class DRPFBackendCPU : public DrpfBackend
    {
    private:
        const float *_data_ptr;
        long _rows;
        long _cols;
        const SrhtProjection *srht;
        const Eigen::VectorXf *norms;
        const std::vector<std::unique_ptr<KdeBinarySplitTree<Eigen::half>>> *forest;
        int numTrees;
        int max_search_buffer_size;

        struct alignas(64) SearchContext
        {

            std::atomic<bool> in_use{false};

            std::vector<uint8_t> counts; // rows, all zero between queries
            std::vector<uint32_t> touched;
            std::vector<uint64_t> seen_words;
            uint32_t generation = 0;
            std::vector<std::pair<float, int>> heap;
            std::vector<int> candidates;

            void resize(int rows, int max_candidates, int k, bool use_seen_words = false)
            {
                if (use_seen_words)
                {
                    const size_t word_count =
                        (static_cast<size_t>(rows) + 31u) / 32u;

                    if (seen_words.size() != word_count)
                    {
                        seen_words.assign(word_count, 0);
                        generation = 0;
                    }

                    if (!counts.empty())
                    {
                        std::vector<uint8_t>().swap(counts);
                        std::vector<uint32_t>().swap(touched);
                    }
                }
                else
                {
                    if (counts.size() != static_cast<size_t>(rows))
                        counts.assign(rows, 0);

                    touched.clear();

                    if (!seen_words.empty())
                        std::vector<uint64_t>().swap(seen_words);
                }

                if (candidates.capacity() < static_cast<size_t>(max_candidates))
                    candidates.reserve(max_candidates);

                if (heap.capacity() < static_cast<size_t>(k + 1))
                    heap.reserve(k + 1);
            }

            FORCE_INLINE bool mark_seen_once(uint32_t idx) noexcept
            {
                const size_t word_idx = idx >> 5;
                const uint32_t bit = 1u << (idx & 31u);

                uint64_t word = seen_words[word_idx];
                uint32_t word_generation = static_cast<uint32_t>(word >> 32);
                uint32_t bits = static_cast<uint32_t>(word);

                if (word_generation != generation)
                    bits = 0;

                if (bits & bit)
                    return false;

                seen_words[word_idx] =
                    (static_cast<uint64_t>(generation) << 32) | (bits | bit);

                return true;
            }
        };

        std::vector<SearchContext> _ctx_pool;

        SearchContext *try_acquire_ctx()
        {
            const int n = static_cast<int>(_ctx_pool.size());
            const int hint = omp_get_thread_num() % n;
            for (int i = 0; i < n; ++i)
            {
                SearchContext &c = _ctx_pool[(hint + i) % n];
                if (!c.in_use.exchange(true, std::memory_order_acquire))
                    return &c;
            }
            return nullptr;
        }

        FORCE_INLINE void gather_candidates(
            const float *__restrict proj_q,
            SearchContext &ctx,
            int votes)
        {
            const uint32_t target_votes = static_cast<uint32_t>(std::max(1, votes));
            ctx.candidates.clear();
            constexpr int prefetch_dist = 16;
            constexpr int B = 8;

            using Tree = KdeBinarySplitTree<Eigen::half>;

            const auto find_leaves = [&](int t0, int nb, std::span<const uint32_t> *leaves)
            {
                const Tree *tr[B];
                int idx[B];
                for (int b = 0; b < nb; ++b)
                {
                    tr[b] = (*forest)[t0 + b].get();
                    idx[b] = tr[b]->getRootIndex();
                }

                for (int depth = 0, any = 1; any; ++depth)
                {
                    any = 0;
                    for (int b = 0; b < nb; ++b)
                        any |= tr[b]->stepDown(idx[b], depth, proj_q);
                }

                for (int b = 0; b < nb; ++b)
                    tr[b]->prefetchLeafData(idx[b]);

                for (int b = 0; b < nb; ++b)
                {
                    leaves[b] = tr[b]->leafSpan(idx[b]);
                    PREFETCH(leaves[b].data());
                }
            };

            if (target_votes == 1)
            {
                ctx.generation++;
                if (ctx.generation >= (1u << 16))
                {
                    std::fill(ctx.seen_words.begin(), ctx.seen_words.end(), 0);
                    ctx.generation = 1;
                }

                for (int t0 = 0; t0 < numTrees; t0 += B)
                {
                    const int nb = std::min(B, numTrees - t0);
                    std::span<const uint32_t> leaves[B];
                    find_leaves(t0, nb, leaves);

                    for (int b = 0; b < nb; ++b)
                    {
                        const uint32_t *ids = leaves[b].data();
                        const size_t count = leaves[b].size();

                        for (size_t j = 0; j < count; ++j)
                        {
                            if (j + prefetch_dist < count)
                                PREFETCH(&ctx.seen_words[ids[j + prefetch_dist] >> 5]);

                            const uint32_t id = ids[j];
                            if (ctx.mark_seen_once(id))
                                ctx.candidates.push_back(static_cast<int>(id));
                        }
                    }
                }
                return;
            }

            if (static_cast<int>(target_votes) > numTrees)
                return;

            ctx.touched.clear();
            uint8_t *cnt = ctx.counts.data();
            const uint8_t tv = static_cast<uint8_t>(std::min<uint32_t>(target_votes, 255));
            const int last_new = numTrees - static_cast<int>(target_votes);

            for (int t0 = 0; t0 < numTrees; t0 += B)
            {
                const int nb = std::min(B, numTrees - t0);
                std::span<const uint32_t> leaves[B];
                find_leaves(t0, nb, leaves);

                for (int b = 0; b < nb; ++b)
                {
                    const bool can_add = (t0 + b) <= last_new;
                    const uint32_t *ids = leaves[b].data();
                    const size_t count = leaves[b].size();

                    for (size_t j = 0; j < count; ++j)
                    {
                        if (j + prefetch_dist < count)
                            PREFETCH(&cnt[ids[j + prefetch_dist]]);

                        const uint32_t id = ids[j];
                        uint8_t c = cnt[id];

                        if (c == 0)
                        {
                            if (!can_add)
                                continue;
                            ctx.touched.push_back(id);
                        }
                        else if (c >= tv)
                            continue;

                        cnt[id] = ++c;
                        if (c == tv)
                            ctx.candidates.push_back(static_cast<int>(id));
                    }
                }
            }

            for (uint32_t id : ctx.touched)
                cnt[id] = 0;
        }

        FORCE_INLINE void compute_distances(
            const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &data,
            const Eigen::Ref<const Eigen::RowVectorXf> q,
            int k,
            int num_cands,
            int *results,
            float *distances_sq,
            SearchContext &ctx)
        {

            float q_norm = q.squaredNorm();
            float q_norm_sqrt = std::sqrt(q_norm);
            int dim = static_cast<int>(data.cols());
            const float *data_raw = data.data();

            ctx.heap.clear();
            float heap_worst = std::numeric_limits<float>::max();

            constexpr int prefetch_dist = 16;

            for (size_t i = 0; i < num_cands; ++i)
            {
                if (i + prefetch_dist < num_cands)
                {
                    int next_idx = ctx.candidates[i + prefetch_dist];
                    PREFETCH(data_raw + static_cast<size_t>(next_idx) * dim);
                }

                int idx = ctx.candidates[i];

                // Cauchy-Schwarz lower bound: dist >= (||x|| - ||q||)^2
                float x_norm_sqrt = std::sqrt((*norms)[idx]);
                float diff = x_norm_sqrt - q_norm_sqrt;
                float lower_bound = diff * diff;

                if (lower_bound > heap_worst)
                    continue;

                float dist = (data.row(idx) - q).squaredNorm();

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

        FORCE_INLINE void exact_ann(
            const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &data,
            const Eigen::Ref<const Eigen::RowVectorXf> q,
            const float *__restrict proj_q,
            int k,
            int votes,
            int *results,
            float *distances_sq,
            SearchContext &ctx)
        {
            int rows = static_cast<int>(data.rows());
            ctx.resize(rows, max_search_buffer_size, k, votes <= 1);

            gather_candidates(proj_q, ctx, votes);

            size_t num_cands = ctx.candidates.size();

            if (num_cands == 0)
            {
                std::fill(results, results + k, -1);
                std::fill(distances_sq, distances_sq + k, std::numeric_limits<float>::max());
                return;
            }

            compute_distances(data, q, k, static_cast<int>(num_cands), results, distances_sq, ctx);
        }

    public:
        DRPFBackendCPU(
            const float *data_ptr,
            long rows,
            long cols,
            const SrhtProjection *srht_,
            const Eigen::VectorXf *nrms,
            const std::vector<std::unique_ptr<KdeBinarySplitTree<Eigen::half>>> *f,
            int trees,
            int max_buffer)
            : _data_ptr(data_ptr), _rows(rows), _cols(cols), srht(srht_), norms(nrms), forest(f), numTrees(trees), max_search_buffer_size(max_buffer)
        {
            _ctx_pool = std::vector<SearchContext>(std::max(8, omp_get_max_threads() * 2));
        }

        ANNResult ann(const float *query, int dim, int k, int votes) override
        {
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> data(_data_ptr, _rows, _cols);
            if (dim != data.cols())
                throw std::invalid_argument("Query dimensionality mismatch");

            const Eigen::Map<const Eigen::RowVectorXf> q(query, dim);

            const int projDim = static_cast<int>(srht->col_sel.size());
            Eigen::RowVectorXf projectedQuery(projDim);

            std::vector<float> work(srht->P);
            srhtProject(*srht, query, work.data(), projectedQuery.data());

            ANNResult out;
            out.indices.resize(k);
            out.distances_sq.resize(k);

            SearchContext *pooled = try_acquire_ctx();
            SearchContext local;
            SearchContext &ctx = pooled ? *pooled : local;

            this->exact_ann(data, q, projectedQuery.data(), k, votes,
                            out.indices.data(), out.distances_sq.data(), ctx);

            if (pooled)
                pooled->in_use.store(false, std::memory_order_release);

            return out;
        }

        ANNResult ann_batch(const float *queries, int n_queries, int dim, int k, int votes) override
        {
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> data(_data_ptr, _rows, _cols);
            if (dim != data.cols())
                throw std::invalid_argument("Query dimensionality mismatch");

            const int projDim = static_cast<int>(srht->col_sel.size());

            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> projected(n_queries, projDim);

            projectDataSrht(*srht, queries, n_queries, dim, projected);

            const float *proj_data = projected.data();

            ANNResult out;
            out.indices.resize(static_cast<size_t>(n_queries) * k);
            out.distances_sq.resize(static_cast<size_t>(n_queries) * k);

#pragma omp parallel
            {
                SearchContext *pooled = try_acquire_ctx();
                SearchContext local;
                SearchContext &ctx = pooled ? *pooled : local;

#pragma omp for schedule(dynamic)
                for (int i = 0; i < n_queries; ++i)
                {
                    const float *qp = queries + static_cast<size_t>(i) * dim;
                    const Eigen::Map<const Eigen::RowVectorXf> q(qp, dim);

                    const float *proj_q = proj_data + static_cast<size_t>(i) * projDim;

                    this->exact_ann(data, q, proj_q, k, votes,
                                    &out.indices[static_cast<size_t>(i) * k],
                                    &out.distances_sq[static_cast<size_t>(i) * k],
                                    ctx);
                }

                if (pooled)
                    pooled->in_use.store(false, std::memory_order_release);
            }

            return out;
        }
    };

#ifdef USE_CUDA

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
            const SrhtProjection *srht,
            const Eigen::VectorXf *norms,
            const std::vector<std::unique_ptr<KdeBinarySplitTree<Eigen::half>>> &cpu_forest,
            int max_buffer)
            : _dim(cols)
        {

            if (!srht || !srht->valid())
                throw std::invalid_argument("GPU backend requires valid SRHT metadata");

            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
                projection_matrix;
            materializeSrhtMatrix(*srht, projection_matrix);

            std::vector<GPUNode> flat_nodes;
            std::vector<GPULeafInfo> flat_leaf_info;
            std::vector<unsigned int> flat_leaf_data;
            std::vector<int> tree_roots;
            std::vector<int> tree_offsets;

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
                projection_matrix.data(),
                static_cast<int>(projection_matrix.cols()),
                norms->data(), ff, max_buffer);

            _cpu_fallback = std::make_unique<DRPFBackendCPU>(
                data, rows, cols, srht, norms, &cpu_forest,
                static_cast<int>(cpu_forest.size()), max_buffer);
        }

        ANNResult ann(const float *query, int dim, int k, int votes) override
        {
            return _cpu_fallback->ann(query, dim, k, votes);
        }

        ANNResult ann_batch(const float *queries, int n_queries, int dim, int k, int votes) override
        {
            constexpr int GPU_CROSSOVER = 64;
            if (n_queries < GPU_CROSSOVER)
            {
                return _cpu_fallback->ann_batch(queries, n_queries, dim, k, votes);
            }

            ANNResult out;
            out.indices.resize(n_queries * k);
            out.distances_sq.resize(n_queries * k);
            search_gpu_batch(_h, queries, n_queries, dim, k, votes,
                             out.indices.data(), out.distances_sq.data());
            return out;
        }
    };

#endif
}
