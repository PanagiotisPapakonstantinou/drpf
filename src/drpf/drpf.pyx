from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from cython.operator cimport dereference
from libc.stddef cimport size_t
from libcpp cimport nullptr
from libc.string cimport memcpy
from libcpp cimport bool

import numpy as np
cimport numpy as np

# C++ class declaration
cdef extern from "drpf.h":
    cdef cppclass CDRPF "DRPF":
        CDRPF(int num_trees, int split_depth, int no_of_ss, bool approximate_search_space_size, float bw_modifier, int seed, float min_ratio, int num_threads)
        void index(const float* data_ptr, size_t length, int dimensions) except + nogil
        vector[int] ann(const float* data, size_t length, int k) except + nogil
        vector[int] ann_batch(const float* queries, int n_queries, int dim, int k) except + nogil
        vector[int] getLeafNodeSizes(int index) except +
        vector[pair[int, int]] getForestIndices(const float* query_ptr, size_t length, int index) except +


cdef class DRPF:
    """
    This class provides a high-performance Approximate Nearest Neighbor (ANN) search index.
    Unlike standard Random Projection Forests that split data at the median, this
    implementation projects data onto random vectors and uses Fast Fourier Transform 
    Kernel Density Estimation (FFT-KDE) to find the natural density "valleys" (minima) 
    between data clusters, resulting in highly pure leaf nodes and superior recall.

    Lifecycle:
        1. Initialize: Create the object with `DRPF(...)`.
        2. Index: Build the forest structure with `.index(data)`.
        3. Query: Perform searches using `.ann()` or `.ann_batch()`.
    """

    cdef CDRPF* c_drpf
    cdef object data_pointer
    cdef bool _is_indexed

    def __cinit__(self,
                  int num_trees=5,
                  int depth=3,
                  int no_of_ss=0,
                  bool approximate_search_space_size=False,
                  float bw_modifier=0.1,
                  int seed=0,
                  float min_ratio=0.33333,
                  int num_threads=0): 
        """
        Initialize the DRPF index structure.

        Parameters
        ----------
        num_trees : int, default=5
            Number of random projection trees in the forest. More trees improve recall 
            but increase memory usage and build/search time.
        depth : int, default=3
            The target split depth for the trees. Higher depth creates smaller leaf nodes,
            speeding up search at the cost of potential recall loss.
        no_of_ss : int, default=0
            Search space size parameter. If 0, it is automatically calculated 
            based on data size and depth.
        approximate_search_space_size : bool, default=False
            Determines the tree splitting strategy. If True, splits are constrained 
            strictly by tree depth. If False, splits continue until a target leaf 
            size (bag) is reached.
        bw_modifier : float, default=0.1
            The smoothing hyperparameter for KDE. Modifies the bandwidth calculated 
            by Silverman's rule. 
            - < 1.0 (e.g., 0.1): Aggressive/spiky. Sensitive to small local gaps.
            - 1.0: Standard optimal smoothing.
            - > 1.0: Over-smoothed. Only splits on massive, global clusters.
        seed : int, default=0
            Base random seed used for generating the random projection matrices. 
            Ensures deterministic and reproducible index building.
        min_ratio : float, default=0.33333
            The probability mass fraction to ignore at the tails of the distribution 
            when searching for a valley. For example, 0.33 restricts the search 
            to the middle 34% of the data mass, preventing splits on outlier noise.
        num_threads : int, default=0
            Number of OpenMP threads for parallel operations. 0 uses all available cores.
        """
        self.c_drpf = new CDRPF(num_trees, depth, no_of_ss, 
                                  approximate_search_space_size, 
                                  bw_modifier, seed, min_ratio, num_threads)
        self._is_indexed = False

    def __dealloc__(self):
        if self.c_drpf != nullptr:
            del self.c_drpf

    def index(self, np.ndarray[np.float32_t, ndim=2] data):
        """
        Builds the Random Projection Forest index from the provided dataset.

        Parameters
        ----------
        data : np.ndarray (n_samples, n_features), dtype=float32
            The dataset to index. Must be a C-contiguous float32 array.
        """
        cdef np.ndarray[np.float32_t, ndim=2, mode="c"] data_float32 = \
            np.ascontiguousarray(data, dtype=np.float32)

        self.data_pointer = data_float32
        cdef const float* data_ptr = <const float*> data_float32.data
        cdef size_t rows = data_float32.shape[0]
        cdef int cols = data_float32.shape[1]
        cdef size_t total_length = rows * cols

        with nogil:
            self.c_drpf.index(data_ptr, total_length, cols)
        self._is_indexed = True

    def ann(self, np.float32_t[:] q, int k):
        """
        Single query nearest neighbor search.

        Parameters
        ----------
        q : ndarray, shape (n_features,)
            The query vector.
        k : int
            Number of nearest neighbors to return.

        Returns
        -------
        indices : ndarray, shape (k,)
            Indices of the exact nearest neighbors found within the candidate pool.
        """
        if not self._is_indexed:
             raise RuntimeError("Index is empty. Call index() first.")

        cdef Py_ssize_t n = q.shape[0]
        cdef const float* data_ptr = &q[0]

        cdef vector[int] result = self.c_drpf.ann(data_ptr, n, k)
        cdef Py_ssize_t size = result.size()
        cdef np.ndarray[np.int32_t, ndim=1] np_array = np.empty(size, dtype=np.int32)

        if size > 0:
            memcpy(<void*> np_array.data, <void*> &result[0], size * sizeof(int))

        return np_array
        
    def ann_batch(self, np.ndarray[np.float32_t, ndim=2] queries, int k):
        """
        Parallel batch nearest neighbor search.

        This method projects all queries into the forest space, gathers unique 
        candidates from intersected leaf nodes across all trees, and computes 
        exact Euclidean distances for all gathered candidates to return the top-k.

        Parameters
        ----------
        queries : ndarray, shape (n_queries, n_features), dtype=float32
            The query vectors.
        k : int
            Number of nearest neighbors to return per query.

        Returns
        -------
        indices : ndarray, shape (n_queries, k)
            Indices of the exact nearest neighbors found within the candidate pool.
        """
        if not self._is_indexed:
             raise RuntimeError("Index is empty. Call index() first.")
             
        cdef np.ndarray[np.float32_t, ndim=2, mode="c"] queries_c = \
            np.ascontiguousarray(queries, dtype=np.float32)
        cdef const float* q_ptr = <const float*> queries_c.data
        cdef int n_queries = queries_c.shape[0]
        cdef int dim = queries_c.shape[1]
    
        cdef vector[int] cpp_results

        with nogil:
            cpp_results = self.c_drpf.ann_batch(q_ptr, n_queries, dim, k)
    
        cdef np.ndarray[np.int32_t, ndim=2] np_results = np.empty((n_queries, k), dtype=np.int32)
        if not cpp_results.empty():
            memcpy(<void*> np_results.data, <void*> cpp_results.data(),
                   n_queries * k * sizeof(int))
        return np_results

    def get_leaf_sizes(self, int index=-1):
        """
        Returns the number of data points stored in each leaf node.
        
        Parameters
        ----------
        index : int, default=-1
            The specific tree index to query. -1 returns aggregated sizes for all trees.
            
        Returns
        -------
        sizes : ndarray, shape (n_leaves,)
            Array containing the size (number of points) of each leaf node.
        """
        if not self._is_indexed:
             raise RuntimeError("Index is empty. Call index() first.")

        cdef vector[int] sizes_vec = self.c_drpf.getLeafNodeSizes(index)
        cdef Py_ssize_t size = sizes_vec.size()

        if size == 0:
            return np.array([], dtype=np.int32)

        cdef np.ndarray[np.int32_t, ndim=1] arr = np.empty(size, dtype=np.int32)
        memcpy(<void*> arr.data, <void*> sizes_vec.data(), size * sizeof(int))
        return arr
    
    def print_leaf_stats(self, name="DRPF"):
        """
        Prints statistical information about the sizes of the leaf nodes in the forest.
        Highly useful for tuning `bw_modifier` and `min_ratio`.
        
        Parameters
        ----------
        name : str, default="DRPF"
            A custom name identifier to print in the stats header.
        """
        sizes = self.get_leaf_sizes()
        if len(sizes) == 0:
            print(f"[{name}] No leaves found (index might be empty).")
            return

        print(f"--- {name} Leaf Statistics ---")
        print(f"Total Leaves: {len(sizes)}")
        print(f"Min Size:     {np.min(sizes)}")
        print(f"Max Size:     {np.max(sizes)}")
        print(f"Mean Size:    {np.mean(sizes):.2f}")
        print(f"Std Dev:      {np.std(sizes):.2f}")
        print("-------------------------------")

    def get_forest_indices(self, np.ndarray[np.float32_t, ndim=1] query, int index=-1):
        """
        Retrieves the tree and leaf indices that a given query vector falls into.

        Parameters
        ----------
        query : ndarray, shape (n_features,), dtype=float32
            The query vector to trace through the forest.
        index : int, default=-1
            The specific tree index to query. -1 queries all trees.

        Returns
        -------
        indices : ndarray, shape (N, 2), dtype=int32
            An array where each row is [tree_index, leaf_index].
        """
        if not self._is_indexed:
             raise RuntimeError("Index is empty. Call index() first.")

        cdef np.ndarray[np.float32_t, ndim=1, mode="c"] query_c = \
            np.ascontiguousarray(query, dtype=np.float32)

        cdef size_t length = query_c.shape[0]
        cdef const float* query_ptr = <const float*> query_c.data

        cdef vector[pair[int, int]] cpp_results = self.c_drpf.getForestIndices(query_ptr, length, index)
        cdef Py_ssize_t size = cpp_results.size()

        cdef np.ndarray[np.int32_t, ndim=2] np_results = np.empty((size, 2), dtype=np.int32)
        cdef Py_ssize_t i

        if size > 0:
            for i in range(size):
                np_results[i, 0] = cpp_results[i].first
                np_results[i, 1] = cpp_results[i].second

        return np_results

    def plot_query_leaves(self, np.ndarray[np.float32_t, ndim=1] query, str method='pca', int bg_samples=2500):
        """
        Projects the dataset into 2D and visualizes the query alongside the exact 
        leaf candidate points found in the forest.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        if not self._is_indexed:
             raise RuntimeError("Index is empty. Call index() first.")
             
        candidates = self.get_forest_indices(query, index=-1)
        if len(candidates) == 0:
            print("No candidates found for this query.")
            return

        point_trees = {}
        for row in candidates:
            t_idx, d_idx = row[0], row[1]
            if d_idx not in point_trees:
                point_trees[d_idx] = []
            point_trees[d_idx].append(t_idx)

        cand_indices = np.array(list(point_trees.keys()), dtype=np.intp)
        data = self.data_pointer

        all_indices = np.arange(data.shape[0])
        bg_pool = np.setdiff1d(all_indices, cand_indices, assume_unique=True)
        
        bg_size = min(bg_samples, len(bg_pool))
        bg_indices = np.random.choice(bg_pool, bg_size, replace=False)
        # --------------------------------------

        points_to_project = np.vstack([
            data[bg_indices],
            data[cand_indices],
            query.reshape(1, -1)
        ])

        print(f"Projecting {len(points_to_project)} points via {method.upper()}...")
        if method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        elif method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        elif method.lower() == 'umap':
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be 'pca', 'tsne', or 'umap'")

        proj = reducer.fit_transform(points_to_project)
        
        proj_bg = proj[:bg_size]
        proj_cand = proj[bg_size:bg_size + len(cand_indices)]
        proj_query = proj[-1]

        base_cmap = plt.colormaps.get_cmap('tab10')
        cand_colors = []
        for d_idx in cand_indices:
            trees = point_trees[d_idx]
            mixed_color = np.mean([np.array(base_cmap(t % 10)[:3]) for t in trees], axis=0)
            cand_colors.append(mixed_color)

        plt.figure(figsize=(10, 8))
        
        plt.scatter(proj_bg[:, 0], proj_bg[:, 1], c='#444444', s=10, alpha=0.3, label='Data Distribution')
        
        plt.scatter(proj_cand[:, 0], proj_cand[:, 1], c=cand_colors, s=10, 
                    edgecolor='white', linewidth=0.2, zorder=3, label='Leaf Candidates')
        
        plt.scatter(proj_query[0], proj_query[1], c='red', marker='*', s=150, 
                    edgecolor='black', zorder=4, label='Query Point')

        plt.title(f"DRPF Candidate Space ({method.upper()})")
        plt.axis('off')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()