#include "drpf_cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace drpf
{

#define CUDA_CHECK(call)                                                            \
    do                                                                              \
    {                                                                               \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                     \
        {                                                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "    \
                      << "code=" << err << " \"" << cudaGetErrorString(err) << "\"" \
                      << std::endl;                                                 \
            throw std::runtime_error("CUDA error");                                 \
        }                                                                           \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do                                                                     \
    {                                                                      \
        cublasStatus_t err = call;                                         \
        if (err != CUBLAS_STATUS_SUCCESS)                                  \
        {                                                                  \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << std::endl;                     \
            throw std::runtime_error("cuBLAS error");                      \
        }                                                                  \
    } while (0)

    void CudaDeleter::operator()(void *ptr) const
    {
        if (ptr)
            cudaFree(ptr);
    }

    void CublasDeleter::operator()(cublasContext *handle) const
    {
        if (handle)
            cublasDestroy(handle);
    }

    template <typename T>
    cuda_ptr<T> allocate_device(size_t num_elements)
    {
        T *ptr = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), num_elements * sizeof(T)));
        return cuda_ptr<T>(ptr);
    }

    __global__ void float_to_half_kernel(const float *src, __half *dst, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            dst[i] = __float2half(src[i]);
    }

    __device__ __forceinline__ bool record_vote(uint32_t *state_ptr, uint32_t generation, uint32_t target_votes)
    {
        uint32_t old_val = *state_ptr;
        while (true)
        {
            uint32_t old_gen = old_val >> 8;
            uint32_t old_votes = old_val & 0xFFu;
            uint32_t votes = (old_gen == generation) ? (old_votes + 1) : 1u;
            uint32_t new_val = (generation << 8) | votes;

            uint32_t prev = atomicCAS(state_ptr, old_val, new_val);
            if (prev == old_val)
                return votes == target_votes;
            old_val = prev;
        }
    }

    __global__ void traverse_trees_kernel(
        const float *__restrict__ d_proj_query,
        const GPUNode *__restrict__ d_nodes,
        const GPULeafInfo *__restrict__ d_leaf_info,
        const unsigned int *__restrict__ d_leaf_data,
        const int *__restrict__ d_tree_roots,
        const int *__restrict__ d_tree_offsets,
        int num_trees, int proj_cols, int max_search_buffer_size,
        int *__restrict__ d_cand_buf,
        int *__restrict__ d_num_candidates,
        uint32_t *__restrict__ d_state, uint32_t generation, uint32_t target_votes,
        int rows_total)
    {
        const int tid = threadIdx.x;
        const int q_idx = blockIdx.x;

        const float *proj_q = d_proj_query + (size_t)q_idx * proj_cols;

        constexpr int MAX_TREES = 512;
        __shared__ int s_leaf_start[MAX_TREES];
        __shared__ int s_leaf_size[MAX_TREES];
        __shared__ int s_num_candidates;

        int *s_candidates = d_cand_buf + (size_t)q_idx * max_search_buffer_size;

        if (tid == 0)
            s_num_candidates = 0;
        __syncthreads();

        for (int t = tid; t < num_trees; t += blockDim.x)
        {
            int curr = d_tree_roots[t];
            const int tree_offset = d_tree_offsets[t];
            int depth = 0;

            while (d_nodes[curr].left_child != -1)
            {
                float qv = proj_q[tree_offset + depth];
                curr = d_nodes[curr].left_child + (qv >= d_nodes[curr].split_val);
                depth++;
            }
            s_leaf_start[t] = d_leaf_info[curr].leaf_start_idx;
            s_leaf_size[t] = d_leaf_info[curr].leaf_size;
        }
        __syncthreads();

        uint32_t *state = d_state + (size_t)q_idx * rows_total;

        for (int t = 0; t < num_trees; ++t)
        {
            const int start = s_leaf_start[t];
            const int size = s_leaf_size[t];

            for (int i = tid; i < size; i += blockDim.x)
            {
                int cid = (int)d_leaf_data[start + i];
                if (record_vote(&state[cid], generation, target_votes))
                {
                    int pos = atomicAdd(&s_num_candidates, 1);
                    if (pos < max_search_buffer_size)
                        s_candidates[pos] = cid;
                }
            }
        }
        __syncthreads();

        if (tid == 0)
            d_num_candidates[q_idx] = min(s_num_candidates, max_search_buffer_size);
    }

#define DRPF_BLOCK_SIZE 256

    __global__ void ann_search_kernel(
        const float *__restrict__ d_orig_queries,
        int n_queries, int k,
        int max_search_buffer_size,
        const __half *__restrict__ d_full_dataset,
        const float *__restrict__ d_norms,
        int dim,
        int *__restrict__ d_out_indices, float *__restrict__ d_out_distances,
        const int *__restrict__ d_cand_buf,
        const int *__restrict__ d_num_candidates)
    {
        const int q_idx = blockIdx.x;
        if (q_idx >= n_queries)
            return;

        const int tid = threadIdx.x;
        const int lane = tid & 31;
        const int warp_id = tid >> 5;
        const int n_warps = blockDim.x >> 5;
        const unsigned mask = 0xffffffff;

        // ---- Shared memory ----
        extern __shared__ unsigned char smem_raw[];
        float *s_query = reinterpret_cast<float *>(smem_raw);

        __shared__ float s_query_norm;
        constexpr int MAX_K = 128;
        __shared__ float topk_dist[MAX_K];
        __shared__ int topk_idx[MAX_K];
        __shared__ float s_warp_dsq[32];
        __shared__ int s_warp_cid[32];

        // ---- Init ----
        if (tid < MAX_K)
        {
            topk_dist[tid] = 1e38f;
            topk_idx[tid] = -1;
        }
        if (tid == 0)
            s_query_norm = 0.0f;
        __syncthreads();

        // ---- Cache query vector + compute squared norm ----
        const float *qrow = d_orig_queries + (size_t)q_idx * dim;
        float thread_q_norm = 0.0f;
        for (int i = tid; i < dim; i += blockDim.x)
        {
            float v = qrow[i];
            s_query[i] = v;
            thread_q_norm += v * v;
        }
        atomicAdd(&s_query_norm, thread_q_norm);
        __syncthreads();

        // ---- Distance phase: WARP-PER-CANDIDATE ----
        const int *s_candidates = d_cand_buf + (size_t)q_idx * max_search_buffer_size;
        const int num_candidates = d_num_candidates[q_idx];
        const int k_eff = (k < MAX_K) ? k : MAX_K;
        const float qn = s_query_norm;

        for (int base = 0; base < num_candidates; base += n_warps)
        {
            const int my_slot = base + warp_id;
            float my_dsq = 1e38f;
            int my_cid = -1;

            if (my_slot < num_candidates)
            {
                const int cid = s_candidates[my_slot];
                const __half *drow = d_full_dataset + (size_t)cid * dim;
                const __half2 *drow2 = reinterpret_cast<const __half2 *>(drow);
                const int dim_pairs = dim >> 1;

                float dot = 0.0f;
                for (int p = lane; p < dim_pairs; p += 32)
                {
                    __half2 h2 = drow2[p];
                    float2 f2 = __half22float2(h2);
                    int d = p << 1;
                    dot += s_query[d] * f2.x;
                    dot += s_query[d + 1] * f2.y;
                }

                // Handle odd dimension
                if ((dim & 1) && lane == 0)
                    dot += s_query[dim - 1] * __half2float(drow[dim - 1]);

                // Warp reduction
                for (int off = 16; off > 0; off >>= 1)
                    dot += __shfl_down_sync(mask, dot, off);

                if (lane == 0)
                {
                    float dsq = qn + d_norms[cid] - 2.0f * dot;
                    if (dsq < 0.0f)
                        dsq = 0.0f;
                    my_dsq = dsq;
                    my_cid = cid;
                }
            }

            if (lane == 0)
            {
                s_warp_dsq[warp_id] = my_dsq;
                s_warp_cid[warp_id] = my_cid;
            }
            __syncthreads();

            if (warp_id == 0 && lane == 0)
            {
                const int rem = num_candidates - base;
                const int this_round = (rem < n_warps) ? rem : n_warps;
                for (int w = 0; w < this_round; ++w)
                {
                    const float dsq = s_warp_dsq[w];
                    const int cid = s_warp_cid[w];
                    if (cid < 0)
                        continue;

                    if (dsq < topk_dist[k_eff - 1])
                    {
                        int j = k_eff - 1;
                        while (j > 0 && topk_dist[j - 1] > dsq)
                        {
                            topk_dist[j] = topk_dist[j - 1];
                            topk_idx[j] = topk_idx[j - 1];
                            --j;
                        }
                        topk_dist[j] = dsq;
                        topk_idx[j] = cid;
                    }
                }
            }
            __syncthreads();
        }

        for (int i = tid; i < k; i += blockDim.x)
        {
            if (i < k_eff)
            {
                d_out_indices[q_idx * k + i] = topk_idx[i];
                d_out_distances[q_idx * k + i] = (topk_idx[i] == -1) ? 1e38f : topk_dist[i];
            }
            else
            {
                d_out_indices[q_idx * k + i] = -1;
                d_out_distances[q_idx * k + i] = 1e38f;
            }
        }
    }

    // ============================================================================
    // Setup: allocate index data + persistent scratch, create cuBLAS handle.
    // ============================================================================
    GPUDataHandle setup_gpu_backend(
        const float *data, long rows, long cols,
        const float *projection_matrix, int proj_cols,
        const float *norms,
        const FlattenedForest &forest,
        int max_search_buffer_size,
        int max_batch,
        int max_k)
    {
        GPUDataHandle h;
        h.rows = rows;
        h.cols = cols;
        h.num_trees = forest.num_trees;
        h.proj_cols = proj_cols;
        h.max_k = max_k;
        h.max_buf = max_search_buffer_size;
        h.generation = 0;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        size_t smem_bytes = ((size_t)cols * sizeof(float)) + ((size_t)proj_cols * sizeof(float));
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, ann_search_kernel, DRPF_BLOCK_SIZE, smem_bytes));

        int target_inflight = prop.multiProcessorCount * std::max(1, blocks_per_sm);
        int auto_batch = std::min(max_batch, target_inflight * 4);
        auto_batch = std::max(auto_batch, prop.multiProcessorCount);
        h.max_batch = auto_batch;

        h.d_full_dataset = allocate_device<float>((size_t)rows * cols);
        CUDA_CHECK(cudaMemcpy(h.d_full_dataset.get(), data,
                              (size_t)rows * cols * sizeof(float),
                              cudaMemcpyHostToDevice));

        h.d_full_dataset_h16 = allocate_device<__half>((size_t)rows * cols);
        {
            size_t total = (size_t)rows * cols;
            int threads = 256;
            int blocks = (int)((total + threads - 1) / threads);
            float_to_half_kernel<<<blocks, threads>>>(
                h.d_full_dataset.get(), h.d_full_dataset_h16.get(), total);
            CUDA_CHECK(cudaGetLastError());
        }

        h.d_full_dataset.reset();

        h.d_norms = allocate_device<float>((size_t)rows);
        CUDA_CHECK(cudaMemcpy(h.d_norms.get(), norms,
                              (size_t)rows * sizeof(float),
                              cudaMemcpyHostToDevice));

        h.d_projection_matrix = allocate_device<float>((size_t)cols * proj_cols);
        CUDA_CHECK(cudaMemcpy(h.d_projection_matrix.get(), projection_matrix,
                              (size_t)cols * proj_cols * sizeof(float),
                              cudaMemcpyHostToDevice));

        if (forest.num_nodes > 0)
        {
            h.d_nodes = allocate_device<GPUNode>(forest.num_nodes);
            CUDA_CHECK(cudaMemcpy(h.d_nodes.get(), forest.nodes,
                                  forest.num_nodes * sizeof(GPUNode),
                                  cudaMemcpyHostToDevice));

            h.d_leaf_info = allocate_device<GPULeafInfo>(forest.num_nodes);
            CUDA_CHECK(cudaMemcpy(h.d_leaf_info.get(), forest.leaf_info,
                                  forest.num_nodes * sizeof(GPULeafInfo),
                                  cudaMemcpyHostToDevice));

            h.d_leaf_data = allocate_device<unsigned int>(forest.num_leaf_data);
            CUDA_CHECK(cudaMemcpy(h.d_leaf_data.get(), forest.leaf_data,
                                  forest.num_leaf_data * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

            h.d_tree_roots = allocate_device<int>(forest.num_trees);
            CUDA_CHECK(cudaMemcpy(h.d_tree_roots.get(), forest.tree_roots,
                                  forest.num_trees * sizeof(int),
                                  cudaMemcpyHostToDevice));

            h.d_tree_offsets = allocate_device<int>(forest.num_trees);
            CUDA_CHECK(cudaMemcpy(h.d_tree_offsets.get(), forest.tree_offsets,
                                  forest.num_trees * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }

        cublasHandle_t raw_cublas = nullptr;
        CUBLAS_CHECK(cublasCreate(&raw_cublas));
        h.cublas.reset(raw_cublas);

        h.d_queries = allocate_device<float>((size_t)h.max_batch * cols);
        h.d_projected_queries = allocate_device<float>((size_t)h.max_batch * proj_cols);
        h.d_out_idx = allocate_device<int>((size_t)h.max_batch * max_k);
        h.d_out_dist = allocate_device<float>((size_t)h.max_batch * max_k);
        h.d_cand_buf = allocate_device<int>((size_t)h.max_batch * max_search_buffer_size);
        h.d_seen = allocate_device<uint32_t>((size_t)h.max_batch * rows);

        CUDA_CHECK(cudaMemset(h.d_seen.get(), 0,
                              (size_t)h.max_batch * rows * sizeof(uint32_t)));

        h.d_num_candidates = allocate_device<int>((size_t)h.max_batch);

        return h;
    }

    static void search_chunk(GPUDataHandle &h,
                             const float *queries, int n_queries, int dim, int votes, int k,
                             int *out_indices, float *out_distances)
    {
        if (k > h.max_k)
            throw std::runtime_error("k exceeds GPU handle max_k");

        h.generation++;
        if (h.generation >= (1u << 24))
        {
            CUDA_CHECK(cudaMemset(h.d_seen.get(), 0,
                                  (size_t)h.max_batch * h.rows * sizeof(uint32_t)));
            h.generation = 1;
        }

        uint32_t target_votes = (uint32_t)max(1, votes);

        CUDA_CHECK(cudaMemcpy(h.d_queries.get(), queries,
                              (size_t)n_queries * dim * sizeof(float), cudaMemcpyHostToDevice));

        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(h.cublas.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                                 h.proj_cols, n_queries, (int)h.cols,
                                 &alpha, h.d_projection_matrix.get(), (int)h.cols,
                                 h.d_queries.get(), (int)h.cols,
                                 &beta, h.d_projected_queries.get(), h.proj_cols));

        traverse_trees_kernel<<<n_queries, DRPF_BLOCK_SIZE>>>(
            h.d_projected_queries.get(),
            h.d_nodes.get(), h.d_leaf_info.get(), h.d_leaf_data.get(), h.d_tree_roots.get(), h.d_tree_offsets.get(),
            h.num_trees, h.proj_cols, h.max_buf,
            h.d_cand_buf.get(), h.d_num_candidates.get(),
            h.d_seen.get(), h.generation, target_votes, (int)h.rows);
        CUDA_CHECK(cudaGetLastError());

        size_t smem = (size_t)h.cols * sizeof(float);
        ann_search_kernel<<<n_queries, DRPF_BLOCK_SIZE, smem>>>(
            h.d_queries.get(), n_queries, k, h.max_buf,
            h.d_full_dataset_h16.get(), h.d_norms.get(), (int)h.cols,
            h.d_out_idx.get(), h.d_out_dist.get(),
            h.d_cand_buf.get(), h.d_num_candidates.get());
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(out_indices, h.d_out_idx.get(),
                              (size_t)n_queries * k * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_distances, h.d_out_dist.get(),
                              (size_t)n_queries * k * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void search_gpu_batch(
        GPUDataHandle &h,
        const float *queries, int n_queries, int dim, int votes, int k,
        int *out_indices, float *out_distances)
    {
        int offset = 0;
        while (offset < n_queries)
        {
            int chunk = std::min(h.max_batch, n_queries - offset);
            search_chunk(
                h,
                queries + (size_t)offset * dim,
                chunk, dim, votes, k,
                out_indices + (size_t)offset * k,
                out_distances + (size_t)offset * k);
            offset += chunk;
        }
    }

    // ============================================================================
    // GPU Indexing Math (Projections & Norms)
    // ============================================================================

    __global__ void compute_squared_norms_kernel(const float *__restrict__ data, float *__restrict__ norms, int rows, int cols)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < rows)
        {
            float sum = 0.0f;
            for (int i = 0; i < cols; ++i)
            {
                float val = data[row * cols + i];
                sum += val * val;
            }
            norms[row] = sum;
        }
    }

    __global__ void cast_float_to_half_kernel(const float *__restrict__ in_f32, __half *__restrict__ out_f16, size_t total_elements)
    {
        size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_elements)
        {
            out_f16[idx] = __float2half(in_f32[idx]);
        }
    }

    void compute_gpu_projections_and_norms(
        const float *h_data,
        const float *h_proj,
        void *h_out_half,
        float *h_norms,
        int rows,
        int cols,
        int proj_cols)
    {
        if (!h_data || !h_proj || !h_out_half || !h_norms)
        {
            throw std::invalid_argument("DRPF CUDA Error: One of the host matrix pointers is NULL.");
        }

        cublasHandle_t raw_handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&raw_handle));
        cublas_ptr handle(raw_handle);

        auto d_data = allocate_device<float>((size_t)rows * cols);
        auto d_norms = allocate_device<float>((size_t)rows);

        CUDA_CHECK(cudaMemcpy(d_data.get(), h_data, (size_t)rows * cols * sizeof(float), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksNorms = (rows + threadsPerBlock - 1) / threadsPerBlock;
        if (blocksNorms < 1)
            blocksNorms = 1;
        compute_squared_norms_kernel<<<blocksNorms, threadsPerBlock>>>(d_data.get(), d_norms.get(), rows, cols);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_norms, d_norms.get(), (size_t)rows * sizeof(float), cudaMemcpyDeviceToHost));

        const int MAX_CHUNK_COLS = 512;

        auto d_proj_chunk = allocate_device<float>((size_t)cols * MAX_CHUNK_COLS);
        auto d_out_f32_chunk = allocate_device<float>((size_t)rows * MAX_CHUNK_COLS);
        auto d_out_f16_chunk = allocate_device<__half>((size_t)rows * MAX_CHUNK_COLS);

        __half *h_out_f16 = reinterpret_cast<__half *>(h_out_half);
        const float alpha = 1.0f;
        const float beta = 0.0f;

        for (int offset = 0; offset < proj_cols; offset += MAX_CHUNK_COLS)
        {
            int current_cols = std::min(MAX_CHUNK_COLS, proj_cols - offset);

            CUDA_CHECK(cudaMemcpy(d_proj_chunk.get(),
                                  h_proj + ((size_t)offset * cols),
                                  (size_t)cols * current_cols * sizeof(float),
                                  cudaMemcpyHostToDevice));

            CUBLAS_CHECK(cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                                     rows, current_cols, cols,
                                     &alpha,
                                     d_data.get(), cols,
                                     d_proj_chunk.get(), cols,
                                     &beta,
                                     d_out_f32_chunk.get(), rows));

            size_t total_elements = (size_t)rows * current_cols;
            int blocksCast = (int)((total_elements + threadsPerBlock - 1) / threadsPerBlock);
            if (blocksCast < 1)
                blocksCast = 1;

            cast_float_to_half_kernel<<<blocksCast, threadsPerBlock>>>(d_out_f32_chunk.get(), d_out_f16_chunk.get(), total_elements);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_out_f16 + ((size_t)offset * rows),
                                  d_out_f16_chunk.get(),
                                  (size_t)rows * current_cols * sizeof(__half),
                                  cudaMemcpyDeviceToHost));
        }
    }

}