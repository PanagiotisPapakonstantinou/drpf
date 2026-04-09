#pragma once
#include <vector>
#include <cstdint> 

struct cublasContext;
typedef struct cublasContext* cublasHandle_t;

struct __half;


struct alignas(16) GPUNode {
    union {
        float split_val;       // Used if internal node
        int   leaf_start_idx;  // Used if leaf node
    };
    union {
        int   split_dim;       // Used if internal node
        int   leaf_size;       // Used if leaf node
    };
    
    int left_child;  
    int right_child; 
};

struct FlattenedForest {
    const GPUNode*      nodes;
    const unsigned int* leaf_data;
    const int*          tree_roots;
    int num_nodes;
    int num_leaf_data;
    int num_trees;
};

// Lives on the CPU but owns GPU resources.
struct GPUDataHandle {
    // Index data (allocated once in setup_gpu_backend)
    float*        d_full_dataset      = nullptr;
    __half* d_full_dataset_h16        = nullptr;
    float* d_norms                    = nullptr;
    GPUNode*      d_nodes             = nullptr;
    unsigned int* d_leaf_data         = nullptr;
    int*          d_tree_roots        = nullptr;
    float*        d_projection_matrix = nullptr;

    long rows      = 0;
    long cols      = 0;
    int  num_trees = 0;
    int  proj_cols = 0;

    // Persistent scratch for search (allocated once, reused per call)
    cublasHandle_t cublas              = nullptr;
    float*         d_queries           = nullptr;
    float*         d_projected_queries = nullptr;
    int*           d_out_idx           = nullptr;
    float*         d_out_dist          = nullptr;
    int*           d_cand_buf          = nullptr;
    uint32_t* d_seen               = nullptr;  
    uint32_t  generation = 0;  

    int max_batch = 0;   // largest n_queries supported by scratch
    int max_k     = 0;   // largest k supported by scratch
    int max_buf   = 0;   // candidates per query
};

// Initialize VRAM and persistent scratch.
//    max_search_buffer_size and max_batch determine scratch sizing.
GPUDataHandle setup_gpu_backend(
    const float* data, long rows, long cols,
    const float* projection_matrix, int proj_cols,
    const float* norms,
    const FlattenedForest& forest,
    int max_search_buffer_size,
    int max_batch = 4096,
    int max_k     = 128
);

// Run search. Reuses persistent scratch in handle.
//    If n_queries > handle.max_batch, processes in chunks.
void search_gpu_batch(
    GPUDataHandle& handle,
    const float* queries, int n_queries, int dim, int k,
    int* out_indices, float* out_distances
);

// Computes the random projections via cuBLAS and the squared L2 norms via custom kernel
void compute_gpu_projections_and_norms(
    const float* h_data, 
    const float* h_proj, 
    void* h_out_half, 
    float* h_norms,
    int rows, 
    int cols, 
    int proj_cols
);


// Free everything.
void free_gpu_handle(GPUDataHandle& handle);
