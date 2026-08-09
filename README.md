# DRPF: Dense Random Projection Forest
[![PyPI](https://img.shields.io/pypi/v/drpf?color=blue)](https://pypi.org/project/drpf/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/drpf)](https://pypi.org/project/drpf/)
[![CI](https://github.com/PanagiotisPapakonstantinou/drpf/actions/workflows/ci.yml/badge.svg)](https://github.com/PanagiotisPapakonstantinou/drpf/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/PanagiotisPapakonstantinou/drpf/branch/main/graph/badge.svg)](https://codecov.io/gh/PanagiotisPapakonstantinou/drpf)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/YOUR_PROJECT_ID)](https://app.codacy.com/gh/PanagiotisPapakonstantinou/drpf/dashboard)
[![Documentation Status](https://readthedocs.org/projects/drpf/badge/?version=latest)](https://drpf.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



This repository presents the DRPF package, an open-source, C++ accelerated Python library that provides an efficient and scalable implementation for Approximate Nearest Neighbor (ANN) search. It advances traditional Random Projection Trees by incorporating Kernel Density Estimation (KDE) to intelligently partition high-dimensional space. Instead of splitting random projections at the median, DRPF uses KDE to find natural "valleys" in the data distribution. This data-driven splitting creates highly balanced trees, reducing boundary errors and improving recall.

This package is highly suited for large-scale data applications as the focus has been given to the computational efficiency of the implemented search methodologies. All core linear algebra and matrix projection operations are powered by the [Eigen3](https://eigen.tuxfamily.org/) C++ library, ensuring cache-friendly math operations. Furthermore, both index construction and batch query evaluations (`ann_batch`) are fully multi-threaded, utilizing all available CPU cores via OpenMP and include optional CUDA support for accelerated batch queries on GPU backends. The Cython integration allows it to operate directly on NumPy `float32` arrays with zero-copy overhead. The software is provided under the MIT license.

## Installation

For the installation of the package, the necessary actions and requirements are a version of Python higher or equal to 3.8, a C++20 compatible compiler with OpenMP support, and the execution of the following commands. The Eigen library (3.4.0+) dependency is handled automatically by the setup script.

```bash
git clone https://github.com/PanagiotisPapakonstantinou/drpf.git
cd drpf
pip install .
```


### Requirements
* Python 3.8+
* C++20 compatible compiler with OpenMP
* Eigen 3.4.0+ (handled automatically)
* **macOS only:** [Homebrew](https://brew.sh/) is required — `setup.py` uses it to locate or install an OpenMP-capable compiler (LLVM/Clang or GCC) and the `libomp` runtime
* **Optional:** CUDA Toolkit 11.8+ for GPU backend

Python package dependencies (NumPy, Cython) are listed in [`requirements.txt`](requirements.txt) and are installed automatically via `pip install .`.

### Optional: GPU Support

DRPF includes an optional CUDA backend for accelerated batch queries on large
workloads. To build with GPU support:

1. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.8+ or 12.x).
2. Ensure `nvcc` is on your `PATH`, or set `CUDA_HOME` / `CUDA_PATH`.
3. Build normally — `setup.py` auto-detects CUDA and compiles `drpf_cuda.cu`:

```bash
   pip install .
```

To force a CPU-only build even with CUDA installed:

```bash
DRPF_DISABLE_CUDA=1 pip install .
```

To target a specific GPU architecture:

```bash
DRPF_CUDA_ARCH=sm_86 pip install .
```

#### Using the GPU backend

```python
index = drpf.DRPF(num_trees=50, depth=8, device="gpu")
index.index(data)

results = index.ann_batch(queries, k=10)
```

**When GPU helps:** large batch queries (≥ 64), large `k`, large datasets.
**When CPU is faster:** single-query `ann()` calls, small batches, small datasets.
Single-query `ann()` always runs on CPU regardless of `device`.

## Usage

### Batch Search Example (`ann_batch`)
The most efficient way to use `drpf` is by processing queries in batches.

```python
import numpy as np
import drpf

# 1. Prepare your data (100k vectors, 128 dimensions)
# Note: Data must be C-contiguous float32 for optimal performance
data = np.random.random((100000, 128)).astype(np.float32)

# 2. Initialize the index
# All parameters are optional and have sensible defaults:
# - num_trees: Larger forest = higher accuracy (default: 5)
# - depth: Greater depth = smaller leaves (default: 3)
# - bw_modifier: KDE smoothing. < 1.0 is highly sensitive; > 1.0 over-smooths. (default: 0.1)
# - min_ratio: Valley search constraint. Restricts split search to the middle mass. (default: 0.33333)
# - seed: Guarantees reproducible projection matrices. (default: 0)
# - num_threads: 0 auto-detects and uses all available CPU cores. (default: 0)

# Example overriding the defaults for a larger, custom-tuned forest:
index = drpf.DRPF(
    num_trees=50,
    depth=8,
    bw_modifier=0.5,
    min_ratio=0.33,
    seed=42,
    num_threads=0,
    device="cpu",
)

# 3. Build the index
index.index(data)

# (Optional) Check the leaf statistics to verify your tuning parameters
index.print_leaf_stats("My Data Index")

# 4. Batch Search
# Pass multiple queries at once to utilize parallel C++ execution
queries = np.random.random((100, 128)).astype(np.float32)
indices = index.ann_batch(queries, k=10, votes=2)

print(f"Nearest neighbors for first query: {indices[0]}")
```