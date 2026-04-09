#include "drpf_cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>

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

__global__ void float_to_half_kernel(const float *src, __half *dst, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __float2half(src[i]);
}

#define DRPF_BLOCK_SIZE 256

__global__ void ann_search_kernel(
    const float *__restrict__ d_projected_queries,
    const float *__restrict__ d_orig_queries,
    const GPUNode *__restrict__ d_nodes,
    const unsigned int *__restrict__ d_leaf_data,
    const int *__restrict__ d_tree_roots,
    int num_trees, int n_queries, int proj_cols, int k,
    int max_search_buffer_size,
    const __half *__restrict__ d_full_dataset,
    const float *__restrict__ d_norms,
    int dim,
    int *__restrict__ d_out_indices, float *__restrict__ d_out_distances,
    uint32_t *__restrict__ d_seen, int rows_total,
    int *__restrict__ d_cand_buf,
    uint32_t generation)
{
    int q_idx = blockIdx.x;
    if (q_idx >= n_queries)
        return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = blockDim.x >> 5;
    const unsigned mask = 0xffffffff;

    // ---- Shared memory layout ----
    extern __shared__ unsigned char smem_raw[];
    float *s_query = reinterpret_cast<float *>(smem_raw);
    float *s_proj_query = reinterpret_cast<float *>(s_query + dim); // High-speed cache

    __shared__ int s_num_candidates;
    __shared__ float s_query_norm;

    constexpr int MAX_K = 128;
    // REVERTED to the single, safe queue
    __shared__ float topk_dist[MAX_K];
    __shared__ int topk_idx[MAX_K];

    __shared__ float s_warp_dsq[32];
    __shared__ int s_warp_cid[32];

    int *s_candidates = d_cand_buf + ((size_t)q_idx * max_search_buffer_size);

    // ---- Init ----
    if (tid < MAX_K)
    {
        topk_dist[tid] = 1e38f;
        topk_idx[tid] = -1;
    }
    if (tid == 0)
    {
        s_num_candidates = 0;
        s_query_norm = 0.0f;
    }
    __syncthreads();

    // ---- Cache queries ----
    const float *qrow = d_orig_queries + (size_t)q_idx * dim;
    const float *p_qrow = d_projected_queries + (size_t)q_idx * proj_cols;

    float thread_q_norm = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x)
    {
        float v = qrow[i];
        s_query[i] = v;
        thread_q_norm += v * v;
    }

    // Cache the projected query for zero-latency tree traversal
    for (int i = tid; i < proj_cols; i += blockDim.x)
    {
        s_proj_query[i] = p_qrow[i];
    }

    atomicAdd(&s_query_norm, thread_q_norm);
    __syncthreads();

    // ---- Traverse trees ----
    if (num_trees > 0 && d_nodes != nullptr)
    {
        uint32_t *seen = d_seen + (size_t)q_idx * rows_total;

        for (int t = tid; t < num_trees; t += blockDim.x)
        {
            int curr = d_tree_roots[t];
            while (true)
            {
                GPUNode node = d_nodes[curr];
                if (node.left_child == -1)
                {
                    for (int c = 0; c < node.leaf_size; ++c)
                    {
                        int cid = d_leaf_data[node.leaf_start_idx + c];

                        // Safe deduplication
                        if (atomicExch(&seen[cid], generation) != generation)
                        {
                            int pos = atomicAdd(&s_num_candidates, 1);
                            if (pos < max_search_buffer_size)
                                s_candidates[pos] = cid;
                        }
                    }
                    break;
                }

                // Fetch perfectly from L1 Shared Memory
                float qv = s_proj_query[node.split_dim];
                curr = (qv < node.split_val) ? node.left_child : node.right_child;
            }
        }
    }
    __syncthreads();

    // ---- Distance phase: WARP-PER-CANDIDATE ----
    int num_candidates = (s_num_candidates < max_search_buffer_size)
                             ? s_num_candidates
                             : max_search_buffer_size;
    int k_eff = (k < MAX_K) ? k : MAX_K;
    float qn = s_query_norm;

    for (int base = 0; base < num_candidates; base += n_warps)
    {
        int my_slot = base + warp_id;
        float my_dsq = 1e38f;
        int my_cid = -1;

        if (my_slot < num_candidates)
        {
            int cid = s_candidates[my_slot];
            const __half *drow = d_full_dataset + (size_t)cid * dim;

            float dot = 0.0f;
            int dim_pairs = dim >> 1;
            const __half2 *drow2 = reinterpret_cast<const __half2 *>(drow);

            for (int p = lane; p < dim_pairs; p += 32)
            {
                __half2 h2 = drow2[p];
                float2 f2 = __half22float2(h2);
                int d = p << 1;
                dot += s_query[d] * f2.x;
                dot += s_query[d + 1] * f2.y;
            }

            if ((dim & 1) && lane == 0)
            {
                int d = dim - 1;
                dot += s_query[d] * __half2float(drow[d]);
            }

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
            int rem = num_candidates - base;
            int this_round = (rem < n_warps) ? rem : n_warps;
            for (int w = 0; w < this_round; ++w)
            {
                float dsq = s_warp_dsq[w];
                int cid = s_warp_cid[w];
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

    CUDA_CHECK(cudaMalloc(&h.d_full_dataset, (size_t)rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(h.d_full_dataset, data,
                          (size_t)rows * cols * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&h.d_full_dataset_h16,
                          (size_t)rows * cols * sizeof(__half)));
    {
        size_t total = (size_t)rows * cols;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        float_to_half_kernel<<<blocks, threads>>>(
            h.d_full_dataset, h.d_full_dataset_h16, total);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(h.d_full_dataset));
    h.d_full_dataset = nullptr;

    CUDA_CHECK(cudaMalloc(&h.d_norms, (size_t)rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(h.d_norms, norms,
                          (size_t)rows * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&h.d_projection_matrix,
                          (size_t)cols * proj_cols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(h.d_projection_matrix, projection_matrix,
                          (size_t)cols * proj_cols * sizeof(float),
                          cudaMemcpyHostToDevice));

    if (forest.num_nodes > 0)
    {
        CUDA_CHECK(cudaMalloc(&h.d_nodes, forest.num_nodes * sizeof(GPUNode)));
        CUDA_CHECK(cudaMemcpy(h.d_nodes, forest.nodes,
                              forest.num_nodes * sizeof(GPUNode),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&h.d_leaf_data,
                              forest.num_leaf_data * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(h.d_leaf_data, forest.leaf_data,
                              forest.num_leaf_data * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&h.d_tree_roots, forest.num_trees * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(h.d_tree_roots, forest.tree_roots,
                              forest.num_trees * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    CUBLAS_CHECK(cublasCreate(&h.cublas));

    CUDA_CHECK(cudaMalloc(&h.d_queries,
                          (size_t)h.max_batch * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&h.d_projected_queries,
                          (size_t)h.max_batch * proj_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&h.d_out_idx,
                          (size_t)h.max_batch * max_k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h.d_out_dist,
                          (size_t)h.max_batch * max_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&h.d_cand_buf,
                          (size_t)h.max_batch * max_search_buffer_size * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&h.d_seen,
                          (size_t)h.max_batch * rows * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h.d_seen, 0,
                          (size_t)h.max_batch * rows * sizeof(uint32_t)));

    return h;
}

// ============================================================================
// Search: one chunk (n_queries <= handle.max_batch).
// ============================================================================
static void search_chunk(
    GPUDataHandle &h,
    const float *queries, int n_queries, int dim, int k,
    int *out_indices, float *out_distances)
{
    if (k > h.max_k)
        throw std::runtime_error("k exceeds GPU handle max_k");

    h.generation++;
    if (h.generation == 0)
    {
        CUDA_CHECK(cudaMemset(h.d_seen, 0, (size_t)h.max_batch * h.rows * sizeof(uint32_t)));
        h.generation = 1;
    }

    CUDA_CHECK(cudaMemcpy(h.d_queries, queries,
                          (size_t)n_queries * dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        h.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        h.proj_cols, n_queries, (int)h.cols,
        &alpha, h.d_projection_matrix, (int)h.cols,
        h.d_queries, (int)h.cols,
        &beta, h.d_projected_queries, h.proj_cols));

    int tpb = DRPF_BLOCK_SIZE;
    int bpg = n_queries;

    // Crucial: Allocate enough memory for BOTH the query and the projected query
    size_t smem_bytes = ((size_t)h.cols * sizeof(float)) + ((size_t)h.proj_cols * sizeof(float));

    ann_search_kernel<<<bpg, tpb, smem_bytes>>>(
        h.d_projected_queries, h.d_queries, h.d_nodes, h.d_leaf_data, h.d_tree_roots,
        h.num_trees, n_queries, h.proj_cols, k, h.max_buf,
        h.d_full_dataset_h16, h.d_norms, (int)h.cols,
        h.d_out_idx, h.d_out_dist, h.d_seen, (int)h.rows, h.d_cand_buf, h.generation);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out_indices, h.d_out_idx, (size_t)n_queries * k * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_distances, h.d_out_dist, (size_t)n_queries * k * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Public batch entry: chunks if n_queries > max_batch.
// ============================================================================
void search_gpu_batch(
    GPUDataHandle &h,
    const float *queries, int n_queries, int dim, int k,
    int *out_indices, float *out_distances)
{
    int offset = 0;
    while (offset < n_queries)
    {
        int chunk = std::min(h.max_batch, n_queries - offset);
        search_chunk(
            h,
            queries + (size_t)offset * dim,
            chunk, dim, k,
            out_indices + (size_t)offset * k,
            out_distances + (size_t)offset * k);
        offset += chunk;
    }
}

// ============================================================================
// Teardown
// ============================================================================
void free_gpu_handle(GPUDataHandle &h)
{
    if (h.cublas)
        cublasDestroy(h.cublas);
    if (h.d_queries)
        cudaFree(h.d_queries);
    if (h.d_projected_queries)
        cudaFree(h.d_projected_queries);
    if (h.d_out_idx)
        cudaFree(h.d_out_idx);
    if (h.d_out_dist)
        cudaFree(h.d_out_dist);
    if (h.d_cand_buf)
        cudaFree(h.d_cand_buf);
    if (h.d_full_dataset)
        cudaFree(h.d_full_dataset);
    if (h.d_full_dataset_h16)
        cudaFree(h.d_full_dataset_h16);
    if (h.d_norms)
        cudaFree(h.d_norms);
    if (h.d_projection_matrix)
        cudaFree(h.d_projection_matrix);
    if (h.d_nodes)
        cudaFree(h.d_nodes);
    if (h.d_leaf_data)
        cudaFree(h.d_leaf_data);
    if (h.d_tree_roots)
        cudaFree(h.d_tree_roots);
    if (h.d_seen)
        cudaFree(h.d_seen);

    h = GPUDataHandle{};
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

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float *d_data = nullptr, *d_norms = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_data, (size_t)rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_norms, (size_t)rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, (size_t)rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksNorms = (rows + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksNorms < 1)
        blocksNorms = 1;
    compute_squared_norms_kernel<<<blocksNorms, threadsPerBlock>>>(d_data, d_norms, rows, cols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_norms, d_norms, (size_t)rows * sizeof(float), cudaMemcpyDeviceToHost));

    const int MAX_CHUNK_COLS = 512;

    float *d_proj_chunk = nullptr, *d_out_f32_chunk = nullptr;
    __half *d_out_f16_chunk = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_proj_chunk, (size_t)cols * MAX_CHUNK_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_out_f32_chunk, (size_t)rows * MAX_CHUNK_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_out_f16_chunk, (size_t)rows * MAX_CHUNK_COLS * sizeof(__half)));

    __half *h_out_f16 = reinterpret_cast<__half *>(h_out_half);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int offset = 0; offset < proj_cols; offset += MAX_CHUNK_COLS)
    {
        int current_cols = std::min(MAX_CHUNK_COLS, proj_cols - offset);

        CUDA_CHECK(cudaMemcpy(d_proj_chunk,
                              h_proj + ((size_t)offset * cols),
                              (size_t)cols * current_cols * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 rows, current_cols, cols,
                                 &alpha,
                                 d_data, cols,
                                 d_proj_chunk, cols,
                                 &beta,
                                 d_out_f32_chunk, rows));

        size_t total_elements = (size_t)rows * current_cols;
        int blocksCast = (int)((total_elements + threadsPerBlock - 1) / threadsPerBlock);
        if (blocksCast < 1)
            blocksCast = 1;

        cast_float_to_half_kernel<<<blocksCast, threadsPerBlock>>>(d_out_f32_chunk, d_out_f16_chunk, total_elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out_f16 + ((size_t)offset * rows),
                              d_out_f16_chunk,
                              (size_t)rows * current_cols * sizeof(__half),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_norms));
    CUDA_CHECK(cudaFree(d_proj_chunk));
    CUDA_CHECK(cudaFree(d_out_f32_chunk));
    CUDA_CHECK(cudaFree(d_out_f16_chunk));
    CUBLAS_CHECK(cublasDestroy(handle));
}