GPU Backend
===========

DRPF ships with an optional CUDA backend for accelerating large batch
nearest-neighbor queries on NVIDIA GPUs.

Building with CUDA
------------------

The build system auto-detects CUDA when ``nvcc`` is on ``PATH`` or when
``CUDA_HOME`` / ``CUDA_PATH`` is set. No extra flags are needed:

.. code-block:: bash

   pip install .

Environment variables:

- ``DRPF_DISABLE_CUDA=1`` — force a CPU-only build.
- ``DRPF_CUDA_ARCH=sm_86`` — target a specific compute capability (default ``sm_75``).
- ``DRPF_CUDA_HOST_COMPILER=/path/to/g++`` — override the host compiler used by ``nvcc``.

Using the GPU backend
---------------------

Pass ``device="gpu"`` to the constructor:

.. code-block:: python

   import drpf
   index = drpf.DRPF(num_trees=50, depth=8, device="gpu")
   index.index(data)
   results = index.ann_batch(queries, k=10)

Performance characteristics
---------------------------

The GPU backend is designed for **throughput**, not latency:

- ``ann_batch()`` runs on the GPU when the batch size is at least 64.
- Smaller batches transparently fall back to the optimized CPU path.
- Single-query ``ann()`` always runs on CPU.

In practice, GPU mode shines for batches of thousands of queries against
datasets with hundreds of thousands of points or more. For interactive
single-query workloads, ``device="cpu"`` is usually faster.

Error handling
--------------

Constructing a GPU index on a build without CUDA raises ``RuntimeError``
with a clear message. To check at runtime whether GPU support is compiled in:

.. code-block:: python

   try:
       drpf.DRPF(device="gpu")
       gpu_supported = True
   except RuntimeError:
       gpu_supported = False