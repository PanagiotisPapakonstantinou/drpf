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


"""
drpf.pyx — Cython bindings for the DRPF (Dense Random Projection Forest) C++ engine.

Exposes the native `drpf::DRPF` C++ class as a Python extension type, handling
NumPy <-> C++ buffer transfer, GIL release during index build/query,
and Device (CPU/GPU) selection.
"""
cdef extern from "drpf.h" namespace "drpf":
    cdef cppclass ANNResult "drpf::DRPF::ANNResult":
        vector[int]   indices
        vector[float] distances_sq

    cdef enum Device "drpf::DRPF::Device":
        CPU "drpf::DRPF::Device::CPU"
        GPU "drpf::DRPF::Device::GPU"

    cdef cppclass CDRPF "drpf::DRPF":
        CDRPF(int num_trees, int split_depth, int no_of_ss,
              bool approximate_search_space_size, float bw_modifier,
              int seed, float min_ratio, int num_threads, Device device_kind) except +
        void index(const float* data_ptr, size_t length, int dimensions) except + nogil
        ANNResult ann(const float* data, size_t length, int k, int votes) except + nogil
        ANNResult ann_batch(const float* queries, int n_queries, int dim, int k, int votes) except + nogil
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
                  int num_threads=0,
                  str device="cpu"): 
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
            A value of 0 seeds from OS entropy (`std::random_device`)
            on every call to `.index()`, so builds are non-deterministic and will
            differ across repeated calls. Pass a nonzero
            integer to get deterministic, reproducible index building.
        min_ratio : float, default=0.33333
            The probability mass fraction to ignore at the tails of the distribution 
            when searching for a valley. For example, 0.33 restricts the search 
            to the middle 34% of the data mass, preventing splits on outlier noise.
        num_threads : int, default=0
            Number of OpenMP threads for parallel operations. 0 uses all available cores.
        device : {"cpu", "gpu"}, default="cpu"
            Execution backend used for both index construction and querying.
            "gpu" requires the package to have been built with CUDA support and
            dispatches batched query traversal to the CUDA backend; "cpu" uses
            OpenMP across `num_threads` threads.

        Raises
        ------
        ValueError
            If `device` is not one of "cpu" or "gpu".
        """

        cdef Device bk
        if device == "cpu":
            bk = CPU
        elif device == "gpu":
            bk = GPU
        else:
            raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")

        self.c_drpf = new CDRPF(num_trees, depth, no_of_ss, 
                                  approximate_search_space_size, 
                                  bw_modifier, seed, min_ratio, num_threads, bk)
        self._is_indexed = False

    def __dealloc__(self):
        if self.c_drpf != nullptr:
            del self.c_drpf

    def index(self, np.ndarray[np.float32_t, ndim=2] data):
        """
        Build the index.

        Parameters
        ----------
        data : (n_samples, n_features) float32 ndarray
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

    def ann(self, np.float32_t[:] q, int k, int votes=1, bool return_distances=False):
        """
        Single query approximate nearest neighbor search.

        Parameters
        ----------
        q : ndarray, shape (n_features,), dtype=float32
            The query vector.
        k : int
            Number of nearest neighbors to return.
        votes : int, default=1
            Minimum number of trees in which a candidate must appear (i.e., land
            in the same leaf as the query) to be promoted to the exact-distance
            refinement stage. Lower values (e.g., 1) search a larger candidate
            pool for higher recall at the cost of speed; higher values prune
            more aggressively for faster, lower-recall search. Can be tuned
            per-query without rebuilding the index.
        return_distances : bool, default=False
            If True, also return the squared L2 distances to each neighbor.
            If False, only indices are returned (default behavior).

        Returns
        -------
        indices : ndarray, shape (k,), dtype=int32
            Indices of the approximate nearest neighbors found within the candidate pool,
            sorted by ascending distance.
        distances_sq : ndarray, shape (k,), dtype=float32
            Squared L2 distances to each neighbor, sorted ascending.
            Only returned when return_distances=True.
            To obtain true L2 distances: np.sqrt(np.maximum(distances_sq, 0))

        Notes
        -----
        Padding: if fewer than k candidates are found in the leaf nodes,
        remaining index slots are filled with -1 and distance slots with
        np.finfo(np.float32).max.

        Examples
        --------
        >>> indices = index.ann(query, k=5)
        >>> indices, distances_sq = index.ann(query, k=5, return_distances=True)
        >>> distances_l2 = np.sqrt(np.maximum(distances_sq, 0))
        """
        if not self._is_indexed:
            raise RuntimeError("Index is empty. Call index() first.")

        cdef Py_ssize_t n = q.shape[0]
        cdef const float* data_ptr = &q[0]
        cdef ANNResult result = self.c_drpf.ann(data_ptr, n, k, votes)
        cdef Py_ssize_t size = result.indices.size()

        cdef np.ndarray[np.int32_t, ndim=1] np_indices
        cdef np.ndarray[np.float32_t, ndim=1] np_distances

        if return_distances:
            np_distances = np.empty(size, dtype=np.float32)
            if size > 0:
                memcpy(<void*> np_distances.data, <void*> result.distances_sq.data(),
                       size * sizeof(float))
            np_indices = np.empty(size, dtype=np.int32)
            if size > 0:
                memcpy(<void*> np_indices.data, <void*> result.indices.data(),
                       size * sizeof(int))
            return np_indices, np_distances

        np_indices = np.empty(size, dtype=np.int32)
        if size > 0:
            memcpy(<void*> np_indices.data, <void*> result.indices.data(),
                   size * sizeof(int))
        return np_indices
        
    def ann_batch(self, np.ndarray[np.float32_t, ndim=2] queries, int k, int votes=1, bool return_distances=False):
        """
        Parallel batch approximate nearest neighbor search.

        Projects all queries into the forest space, gathers unique candidates
        from intersected leaf nodes across all trees, and computes exact squared
        Euclidean distances for all gathered candidates to return the top-k per query.

        Parameters
        ----------
        queries : ndarray, shape (n_queries, n_features), dtype=float32
            The query vectors. Must be a C-contiguous float32 array.
        k : int
            Number of nearest neighbors to return per query.
        votes : int, default=1
            Minimum number of trees in which a candidate must appear across
            the forest to be promoted to the exact-distance refinement stage,
            applied independently per query. Lower values widen the candidate
            pool (higher recall, slower); higher values prune more aggressively
            (faster, lower recall). Adjustable per call without rebuilding the index.
        return_distances : bool, default=False
            If True, also return the squared L2 distances to each neighbor.
            If False, only indices are returned (default behavior).

        Returns
        -------
        indices : ndarray, shape (n_queries, k), dtype=int32
            Indices of the approximate nearest neighbors for each query,
            sorted by ascending distance per row.
        distances_sq : ndarray, shape (n_queries, k), dtype=float32
            Squared L2 distances to each neighbor per query, sorted ascending per row.
            Only returned when return_distances=True.
            To obtain true L2 distances: np.sqrt(np.maximum(distances_sq, 0))

        Notes
        -----
        Padding: if fewer than k candidates are found for a given query,
        remaining index slots are filled with -1 and distance slots with
        np.finfo(np.float32).max.

        Examples
        --------
        >>> indices = index.ann_batch(queries, k=5)
        >>> indices, distances_sq = index.ann_batch(queries, k=5, return_distances=True)
        >>> distances_l2 = np.sqrt(np.maximum(distances_sq, 0))
        """
        if not self._is_indexed:
            raise RuntimeError("Index is empty. Call index() first.")

        cdef np.ndarray[np.float32_t, ndim=2, mode="c"] queries_c = \
            np.ascontiguousarray(queries, dtype=np.float32)
        cdef const float* q_ptr = <const float*> queries_c.data
        cdef int n_queries = queries_c.shape[0]
        cdef int dim = queries_c.shape[1]

        cdef ANNResult cpp_result
        with nogil:
            cpp_result = self.c_drpf.ann_batch(q_ptr, n_queries, dim, k, votes)

        cdef np.ndarray[np.int32_t, ndim=2]   np_indices   = np.empty((n_queries, k), dtype=np.int32)
        cdef np.ndarray[np.float32_t, ndim=2] np_distances

        if not cpp_result.indices.empty():
            memcpy(<void*> np_indices.data, <void*> cpp_result.indices.data(),
                   n_queries * k * sizeof(int))

        if return_distances:
            np_distances = np.empty((n_queries, k), dtype=np.float32)
            if not cpp_result.distances_sq.empty():
                memcpy(<void*> np_distances.data, <void*> cpp_result.distances_sq.data(),
                       n_queries * k * sizeof(float))
            return np_indices, np_distances

        return np_indices

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

    def plot_query_leaves(self, np.ndarray[np.float32_t, ndim=1] query,
                          str method='pca', int bg_samples=2500,
                          base_color='tab:blue', grey=(0.8, 0.8, 0.8),
                          clip_percentiles=(5, 95), float power=1.0):
        """
        Projects the dataset into 2D and visualizes the query alongside the exact
        leaf candidate points found in the forest. Candidate points are shaded
        from ``base_color`` (near = intense) toward ``grey`` (far = greyed out).
    
        Parameters
        ----------
        query : ndarray, shape (n_features,), dtype=float32
            The query vector to visualize.
        method : {'pca', 'tsne', 'umap'}, default='pca'
            Dimensionality reduction method for the 2D projection.
        bg_samples : int, default=2500
            Number of background points to sample from the dataset.
        base_color : str or tuple, default='tab:blue'
            Matplotlib color name or RGB tuple for the base hue of near points.
        grey : tuple, default=(0.8, 0.8, 0.8)
            RGB tuple to blend toward for far points.
        clip_percentiles : tuple, default=(5, 95)
            ``(low_pct, high_pct)`` used for robust distance scaling.
        power : float, default=1.0
            Exponent applied to normalized distances to adjust contrast.
            ``power < 1`` makes more points appear intense; ``power > 1``
            produces a sharper falloff.
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
            t_idx, d_idx = int(row[0]), int(row[1])
            if d_idx not in point_trees:
                point_trees[d_idx] = []
            point_trees[d_idx].append(t_idx)

        cand_indices = np.array(list(point_trees.keys()), dtype=np.intp)
        data = self.data_pointer

        all_indices = np.arange(data.shape[0])
        bg_pool = np.setdiff1d(all_indices, cand_indices, assume_unique=True)
        bg_size = min(bg_samples, len(bg_pool))
        bg_indices = np.random.choice(bg_pool, bg_size, replace=False)

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

        cand_vecs = data[cand_indices]                     

        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)[0]
        diff = cand_vecs - q
        dists = np.sum(diff * diff, axis=1)    

        if len(dists) == 0:
            print("No candidate vectors to color.")
            return

        lo_pct, hi_pct = clip_percentiles
        lo = np.percentile(dists, lo_pct)
        hi = np.percentile(dists, hi_pct)
        
        if hi <= lo:

            lo, hi = dists.min(), dists.max()
            if hi == lo:
                hi = lo + 1e-6


        norm = np.clip((dists - lo) / (hi - lo), 0.0, 1.0)

        alpha = np.exp(-norm * 2)

        if isinstance(base_color, str):
            base_rgb = mcolors.to_rgb(base_color)
        else:
            base_rgb = tuple(base_color)

        grey_rgb = tuple(grey)

        cand_colors = np.outer(alpha, np.ones(3)) * np.array(base_rgb)[None, :] + \
                      np.outer(1.0 - alpha, np.ones(3)) * np.array(grey_rgb)[None, :]

        plt.figure(figsize=(10, 8))

        plt.scatter(proj_bg[:, 0], proj_bg[:, 1], c='#444444', s=10, alpha=0.3, label='Data Distribution')

        plt.scatter(proj_cand[:, 0], proj_cand[:, 1], c=cand_colors, s=20,
                    edgecolor='white', linewidth=0.2, zorder=3, label='Leaf Candidates')

        plt.scatter(proj_query[0], proj_query[1], c='red', marker='*', s=150,
                    edgecolor='black', zorder=4, label='Query Point')

        plt.title(f"DRPF Candidate Space ({method.upper()})")
        plt.axis('off')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()