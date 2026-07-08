#pragma once
#include <vector>
#include <cstdint>
#include <memory>

namespace drpf
{

    struct cublasContext;
    typedef struct cublasContext *cublasHandle_t;

    struct __half;

    struct CudaDeleter
    {
        void operator()(void *ptr) const;
    };

    struct CublasDeleter
    {
        void operator()(cublasContext *handle) const;
    };

    template <typename T>
    using cuda_ptr = std::unique_ptr<T, CudaDeleter>;
    using cublas_ptr = std::unique_ptr<cublasContext, CublasDeleter>;

    struct GPUNode
    {
        float split_val;
        int left_child;
    };

    struct GPULeafInfo
    {
        int leaf_start_idx;
        int leaf_size;
    };

    struct FlattenedForest
    {
        const GPUNode *nodes;
        const GPULeafInfo *leaf_info;
        const unsigned int *leaf_data;
        const int *tree_roots;
        const int *tree_offsets;
        int num_nodes;
        int num_leaf_data;
        int num_trees;
    };

    // Lives on the CPU but owns GPU resources.
    struct GPUDataHandle
    {
        cuda_ptr<float> d_full_dataset = nullptr;
        cuda_ptr<__half> d_full_dataset_h16 = nullptr;
        cuda_ptr<float> d_norms = nullptr;
        cuda_ptr<GPUNode> d_nodes = nullptr;
        cuda_ptr<GPULeafInfo> d_leaf_info = nullptr;
        cuda_ptr<unsigned int> d_leaf_data = nullptr;
        cuda_ptr<int> d_tree_roots = nullptr;
        cuda_ptr<int> d_tree_offsets = nullptr;
        cuda_ptr<float> d_projection_matrix = nullptr;

        long rows = 0;
        long cols = 0;
        int num_trees = 0;
        int proj_cols = 0;

        cublas_ptr cublas = nullptr;
        cuda_ptr<float> d_queries = nullptr;
        cuda_ptr<float> d_projected_queries = nullptr;
        cuda_ptr<int> d_out_idx = nullptr;
        cuda_ptr<float> d_out_dist = nullptr;
        cuda_ptr<int> d_cand_buf = nullptr;
        cuda_ptr<uint32_t> d_seen = nullptr;
        uint32_t generation = 0;
        cuda_ptr<int> d_num_candidates = nullptr;

        int max_batch = 0;
        int max_k = 0;
        int max_buf = 0;
    };

    // Initialize VRAM and persistent scratch.
    //    max_search_buffer_size and max_batch determine scratch sizing.
    GPUDataHandle setup_gpu_backend(
        const float *data, long rows, long cols,
        const float *projection_matrix, int proj_cols,
        const float *norms,
        const FlattenedForest &forest,
        int max_search_buffer_size,
        int max_batch = 4096,
        int max_k = 128);

    // Run search. Reuses persistent scratch in handle.
    //    If n_queries > handle.max_batch, processes in chunks.
    void search_gpu_batch(
        GPUDataHandle &handle,
        const float *queries, int n_queries, int dim, int votes, int k,
        int *out_indices, float *out_distances);

    // Computes the random projections via cuBLAS and the squared L2 norms via custom kernel
    void compute_gpu_projections_and_norms(
        const float *h_data,
        const float *h_proj,
        void *h_out_half,
        float *h_norms,
        int rows,
        int cols,
        int proj_cols);

}