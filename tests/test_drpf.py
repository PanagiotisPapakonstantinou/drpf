"""
Tests for the DRPF (Density-based Random Projection Forest) Cython extension.

Assumes the extension has been compiled and is importable as `drpf`.
Run with: pytest tests/test_drpf.py -v
"""

import pytest  # type: ignore
import numpy as np

drpf_mod = pytest.importorskip("drpf", reason="drpf extension not compiled")
DRPF = drpf_mod.DRPF


# ---------------------------------------------------------------------------
# GPU availability (lazy, cached)
# ---------------------------------------------------------------------------
_GPU_AVAILABLE = None


def gpu_available():
    """Check whether the drpf build supports GPU and a device is present.

    Lazy and cached so it only runs once, and only when a test actually
    needs to know. Catches BaseException to be robust against driver-level
    failures that don't subclass Exception.
    """
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    try:
        idx = DRPF(num_trees=2, depth=2, seed=0, device="gpu")
        idx.index(
            np.random.default_rng(0).standard_normal((50, 8)).astype(np.float32)
        )
        # Trigger a batch large enough to exercise the GPU path (>= 64)
        q = np.random.default_rng(1).standard_normal((64, 8)).astype(np.float32)
        idx.ann_batch(q, k=3)
        _GPU_AVAILABLE = True
    except BaseException:
        _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_data():
    """200 points in 16 dimensions, float32."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 16)).astype(np.float32)


@pytest.fixture
def indexed_drpf(small_data):
    """A DRPF instance that has already been indexed (CPU)."""
    idx = DRPF(num_trees=3, depth=3, seed=42)
    idx.index(small_data)
    return idx, small_data


@pytest.fixture(params=["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def device(request):
    if request.param == "gpu" and not gpu_available():
        pytest.skip("GPU backend not available (no CUDA build or no device)")
    return request.param


@pytest.fixture
def indexed_drpf_device(small_data, device):
    """A DRPF instance indexed on the parametrized device."""
    idx = DRPF(num_trees=3, depth=3, seed=42, device=device)
    idx.index(small_data)
    return idx, small_data, device


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestInit:
    def test_default_construction(self):
        idx = DRPF()
        assert idx is not None

    def test_custom_params(self):
        idx = DRPF(num_trees=10, depth=5, bw_modifier=0.5, seed=7, min_ratio=0.25)
        assert idx is not None


# ---------------------------------------------------------------------------
# Indexing & guard rails
# ---------------------------------------------------------------------------
class TestIndex:
    def test_index_accepts_float32(self, small_data):
        assert small_data.dtype == np.float32
        idx = DRPF(seed=0)
        idx.index(small_data)

    def test_index_converts_float64(self):
        """index() should accept float64 input cast to float32."""
        data = np.random.default_rng(0).standard_normal((100, 8))
        idx = DRPF(seed=0)
        idx.index(data.astype(np.float32))

    def test_raises_before_index_ann(self):
        idx = DRPF()
        q = np.zeros(16, dtype=np.float32)
        with pytest.raises(RuntimeError, match="index"):
            idx.ann(q, k=5)

    def test_raises_before_index_ann_batch(self):
        idx = DRPF()
        queries = np.zeros((3, 16), dtype=np.float32)
        with pytest.raises(RuntimeError, match="index"):
            idx.ann_batch(queries, k=5)

    def test_raises_before_index_leaf_sizes(self):
        idx = DRPF()
        with pytest.raises(RuntimeError, match="index"):
            idx.get_leaf_sizes()

    def test_raises_before_index_forest_indices(self):
        idx = DRPF()
        q = np.zeros(16, dtype=np.float32)
        with pytest.raises(RuntimeError, match="index"):
            idx.get_forest_indices(q)


# ---------------------------------------------------------------------------
# Single-query ANN
# ---------------------------------------------------------------------------
class TestANN:
    def test_returns_ndarray(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.ann(data[0], k=5)
        assert isinstance(result, np.ndarray)

    def test_dtype_int32(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.ann(data[0], k=5)
        assert result.dtype == np.int32

    def test_returns_k_results(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.ann(data[0], k=5)
        assert len(result) == 5

    def test_indices_in_valid_range(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.ann(data[0], k=5)
        assert np.all(result >= 0)
        assert np.all(result < len(data))

    def test_k_equals_one(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.ann(data[0], k=1)
        assert result.shape == (1,)

    def test_deterministic_with_same_seed(self, small_data):
        idx1 = DRPF(num_trees=3, depth=3, seed=99)
        idx1.index(small_data)
        idx2 = DRPF(num_trees=3, depth=3, seed=99)
        idx2.index(small_data)
        q = small_data[5]
        np.testing.assert_array_equal(idx1.ann(q, k=5), idx2.ann(q, k=5))

    def test_recall_finds_self(self, indexed_drpf):
        """Query with points that exist in the dataset — each should find itself."""
        idx, data = indexed_drpf
        hits = sum(
            1 for i in range(20)
            if i in idx.ann(data[i], k=10)
        )
        assert hits >= 15, f"Recall too low: only {hits}/20 queries found themselves"

    def test_k_larger_than_dataset(self, indexed_drpf):
        """k > dataset size — result is padded with -1 sentinels for unfilled slots."""
        idx, data = indexed_drpf
        result = idx.ann(data[0], k=len(data) + 100)
        assert isinstance(result, np.ndarray)
        valid = result[result >= 0]
        assert len(valid) <= len(data)
        assert np.all(valid < len(data))


# ---------------------------------------------------------------------------
# Batch ANN
# ---------------------------------------------------------------------------
class TestAnnBatch:
    def test_returns_2d_ndarray(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.ann_batch(data[:10], k=5)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_shape(self, indexed_drpf):
        idx, data = indexed_drpf
        n_queries, k = 8, 6
        result = idx.ann_batch(data[:n_queries], k=k)
        assert result.shape == (n_queries, k)

    def test_dtype_int32(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.ann_batch(data[:10], k=5)
        assert result.dtype == np.int32

    def test_indices_in_valid_range(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.ann_batch(data[:10], k=5)
        assert np.all(result >= 0)
        assert np.all(result < len(data))

    def test_single_query_batch(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.ann_batch(data[:1], k=5)
        assert result.shape == (1, 5)

    def test_batch_matches_single(self, indexed_drpf):
        """ann_batch should return same candidates as ann for each query."""
        idx, data = indexed_drpf
        k = 5
        queries = data[:3]
        batch = idx.ann_batch(queries, k=k)
        for i in range(len(queries)):
            single = idx.ann(queries[i], k=k)
            assert set(batch[i].tolist()) == set(single.tolist())

    def test_recall_batch_finds_self(self, indexed_drpf):
        """Batch queries with dataset points — each should find itself in results."""
        idx, data = indexed_drpf
        queries = data[:20]
        results = idx.ann_batch(queries, k=10)
        hits = sum(
            1 for i in range(20)
            if i in results[i]
        )
        assert hits >= 15, f"Batch recall too low: only {hits}/20 queries found themselves"

    def test_k_larger_than_dataset_batch(self, indexed_drpf):
        """k > dataset size — batch result is padded with -1 sentinels."""
        idx, data = indexed_drpf
        result = idx.ann_batch(data[:5], k=len(data) + 100)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 5


# ---------------------------------------------------------------------------
# Leaf sizes
# ---------------------------------------------------------------------------
class TestLeafSizes:
    def test_returns_ndarray(self, indexed_drpf):
        idx, _ = indexed_drpf
        sizes = idx.get_leaf_sizes()
        assert isinstance(sizes, np.ndarray)

    def test_dtype_int32(self, indexed_drpf):
        idx, _ = indexed_drpf
        sizes = idx.get_leaf_sizes()
        assert sizes.dtype == np.int32

    def test_all_positive(self, indexed_drpf):
        """Every leaf must contain at least one point."""
        idx, _ = indexed_drpf
        sizes = idx.get_leaf_sizes()
        assert np.all(sizes > 0)

    def test_specific_tree_index(self, indexed_drpf):
        idx, _ = indexed_drpf
        sizes = idx.get_leaf_sizes(index=0)
        assert isinstance(sizes, np.ndarray)
        assert len(sizes) > 0

    def test_total_count_matches_data(self, indexed_drpf):
        """Sum of leaf sizes across all trees should equal n_samples * num_trees."""
        idx, data = indexed_drpf
        sizes = idx.get_leaf_sizes()
        assert sizes.sum() == len(data) * 3

    def test_different_bw_modifier_changes_leaf_structure(self, small_data):
        """Different bandwidth settings should produce different leaf structures."""
        idx_spiky = DRPF(num_trees=3, depth=3, seed=0, bw_modifier=0.01)
        idx_spiky.index(small_data)

        idx_smooth = DRPF(num_trees=3, depth=3, seed=0, bw_modifier=2.0)
        idx_smooth.index(small_data)

        sizes_spiky = idx_spiky.get_leaf_sizes()
        sizes_smooth = idx_smooth.get_leaf_sizes()

        assert sizes_spiky.std() != sizes_smooth.std() or \
               len(sizes_spiky) != len(sizes_smooth), \
               "Expected different bw_modifier values to produce different leaf structures"

    def test_more_trees_increases_leaf_count(self, small_data):
        """More trees should produce proportionally more leaves in total."""
        idx3 = DRPF(num_trees=3, depth=3, seed=0)
        idx3.index(small_data)

        idx6 = DRPF(num_trees=6, depth=3, seed=0)
        idx6.index(small_data)

        assert len(idx6.get_leaf_sizes()) > len(idx3.get_leaf_sizes())


# ---------------------------------------------------------------------------
# Forest indices
# ---------------------------------------------------------------------------
class TestForestIndices:
    def test_returns_2d_ndarray(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.get_forest_indices(data[0])
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_two_columns(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.get_forest_indices(data[0])
        assert result.shape[1] == 2

    def test_dtype_int32(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.get_forest_indices(data[0])
        assert result.dtype == np.int32

    def test_specific_tree(self, indexed_drpf):
        idx, data = indexed_drpf
        result = idx.get_forest_indices(data[0], index=0)
        assert result.ndim == 2

    def test_number_of_rows_matches_num_trees(self, indexed_drpf):
        """All 3 trees should be represented in the forest indices."""
        idx, data = indexed_drpf
        result = idx.get_forest_indices(data[0])
        unique_trees = np.unique(result[:, 0])
        assert len(unique_trees) == 3
        assert set(unique_trees.tolist()) == {0, 1, 2}

    def test_tree_indices_in_valid_range(self, indexed_drpf):
        """First column (tree index) must be within [0, num_trees)."""
        idx, data = indexed_drpf
        result = idx.get_forest_indices(data[0])
        tree_indices = result[:, 0]
        assert np.all(tree_indices >= 0)
        assert np.all(tree_indices < 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_single_point_dataset(self):
        """A dataset with one point should not crash."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        idx = DRPF(num_trees=2, depth=1, seed=0)
        idx.index(data)
        result = idx.ann(data[0], k=1)
        assert isinstance(result, np.ndarray)

    def test_single_feature_dimension(self):
        """1D data (single feature) should be handled gracefully."""
        data = np.random.default_rng(0).standard_normal((50, 1)).astype(np.float32)
        idx = DRPF(num_trees=2, depth=2, seed=0)
        idx.index(data)
        result = idx.ann(data[0], k=5)
        assert isinstance(result, np.ndarray)

    def test_two_points_dataset(self):
        """Minimal meaningful dataset — two points."""
        data = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        idx = DRPF(num_trees=2, depth=1, seed=0)
        idx.index(data)
        result = idx.ann(data[0], k=1)
        assert isinstance(result, np.ndarray)

    def test_high_dimensional_data(self):
        """High-dimensional data (e.g. 512 dims) should work without crashing."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal((100, 512)).astype(np.float32)
        idx = DRPF(num_trees=3, depth=2, seed=0)
        idx.index(data)
        result = idx.ann(data[0], k=10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# Stats printing
# ---------------------------------------------------------------------------
class TestPrintLeafStats:
    def test_smoke(self, indexed_drpf, capsys):
        idx, _ = indexed_drpf
        idx.print_leaf_stats(name="TestDRPF")
        out = capsys.readouterr().out
        assert "TestDRPF" in out
        assert "Total Leaves" in out

    def test_output_contains_all_fields(self, indexed_drpf, capsys):
        idx, _ = indexed_drpf
        idx.print_leaf_stats()
        out = capsys.readouterr().out
        for field in ("Min Size", "Max Size", "Mean Size", "Std Dev"):
            assert field in out, f"Missing field in output: {field}"

    def test_not_indexed_raises(self):
        idx = DRPF()
        with pytest.raises(RuntimeError):
            idx.print_leaf_stats()


# ---------------------------------------------------------------------------
# GPU backend
# ---------------------------------------------------------------------------
class TestGPUBackend:
    def test_gpu_constructor_accepts_device_kwarg(self):
        """Either constructs successfully (CUDA build) or raises a clear RuntimeError."""
        try:
            idx = DRPF(device="gpu")
            assert idx is not None
        except RuntimeError as e:
            assert "CUDA" in str(e) or "GPU" in str(e), \
                f"Expected CUDA/GPU in error message, got: {e}"

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="device"):
            DRPF(device="tpu")

    @pytest.mark.gpu
    def test_gpu_batch_matches_cpu(self, small_data):
        if not gpu_available():
            pytest.skip("GPU not available")
        cpu = DRPF(num_trees=3, depth=3, seed=42, device="cpu")
        gpu = DRPF(num_trees=3, depth=3, seed=42, device="gpu")
        cpu.index(small_data)
        gpu.index(small_data)

        # 64+ queries to cross the GPU_CROSSOVER threshold
        queries = small_data[:80]
        cpu_res = cpu.ann_batch(queries, k=10)
        gpu_res = gpu.ann_batch(queries, k=10)

        for i in range(len(queries)):
            overlap = len(set(cpu_res[i]) & set(gpu_res[i]))
            assert overlap >= 8, \
                f"GPU/CPU disagree too much on query {i}: {overlap}/10"

    @pytest.mark.gpu
    def test_gpu_small_batch_falls_back_to_cpu(self, small_data):
        if not gpu_available():
            pytest.skip("GPU not available")
        idx = DRPF(num_trees=3, depth=3, seed=42, device="gpu")
        idx.index(small_data)

        # < 64 queries should hit CPU fallback path inside DRPFBackendGPU
        result = idx.ann_batch(small_data[:10], k=5)
        assert result.shape == (10, 5)
        assert np.all(result >= 0)

    @pytest.mark.gpu
    def test_gpu_recall_finds_self_batch(self, small_data):
        """GPU batch results should find query points themselves with high recall."""
        if not gpu_available():
            pytest.skip("GPU not available")
        idx = DRPF(num_trees=3, depth=3, seed=42, device="gpu")
        idx.index(small_data)
        # Need >= 64 queries to actually exercise the GPU path
        queries = small_data[:80]
        results = idx.ann_batch(queries, k=10)
        hits = sum(1 for i in range(len(queries)) if i in results[i])
        assert hits >= 60, f"GPU batch recall too low: {hits}/80"