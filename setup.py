from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import urllib.request
import zipfile
import os
import subprocess
import sys
import shutil
import re 
import numpy as np

# Global variables for compiler flags
compile_flags = []
link_flags = []

# ===============================================================
# 1. Configuration
# ===============================================================

EIGEN_VERSION = "3.4.0"
EIGEN_DIR = os.path.join("src", "third_party", "eigen")
EIGEN_HEADER = os.path.join(EIGEN_DIR, "Eigen", "Dense")
EIGEN_URL = f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_VERSION}/eigen-{EIGEN_VERSION}.zip"


def eigen_version_at_least(eigen_path, required=(3, 4, 0)):
    macros_path = os.path.join(eigen_path, "Eigen", "src", "Core", "util", "Macros.h")
    if not os.path.exists(macros_path):
        return False

    world = major = minor = None
    with open(macros_path, "r", encoding="utf-8") as f:
        for line in f:
            if "#define EIGEN_WORLD_VERSION" in line:
                world = int(line.split()[-1])
            elif "#define EIGEN_MAJOR_VERSION" in line:
                major = int(line.split()[-1])
            elif "#define EIGEN_MINOR_VERSION" in line:
                minor = int(line.split()[-1])
            if world is not None and major is not None and minor is not None:
                break

    if world is None or major is None or minor is None:
        return False
    return (world, major, minor) >= required

# ===============================================================
# 2. Detect or download Eigen
# ===============================================================
def ensure_eigen():
    eigen_env = os.environ.get("EIGEN_PATH")
    if eigen_env and os.path.exists(os.path.join(eigen_env, "Eigen")):
        if eigen_version_at_least(eigen_env, (3, 4, 0)):
            return eigen_env
        else:
            raise RuntimeError(f"Eigen at {eigen_env} is too old. Require ≥ 3.4.0.")

    candidates = [
        "/opt/homebrew/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
        r"C:\Libraries\Eigen"
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "Eigen")):
            if eigen_version_at_least(c, (3, 4, 0)):
                return c

    if not os.path.exists(EIGEN_HEADER):
        print(f"Eigen {EIGEN_VERSION} not found. Downloading...")
        os.makedirs(EIGEN_DIR, exist_ok=True)
        zip_path = "eigen.zip"
        urllib.request.urlretrieve(EIGEN_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(EIGEN_DIR)
        os.remove(zip_path)

        nested = os.path.join(EIGEN_DIR, f"eigen-{EIGEN_VERSION}")
        if os.path.exists(nested):
            for item in os.listdir(nested):
                shutil.move(os.path.join(nested, item), EIGEN_DIR)
            shutil.rmtree(nested)

    return EIGEN_DIR

eigen_include = ensure_eigen()
print(f"Using Eigen from: {eigen_include}")



# ===============================================================
# 3. CUDA detection and pre-build
# ===============================================================
def find_cuda():
    """Return (cuda_home, nvcc_path) or (None, None) if not found."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_home = os.path.dirname(os.path.dirname(nvcc))
        return cuda_home, nvcc

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidate = os.path.join(cuda_home, "bin", "nvcc")
        if sys.platform == "win32":
            candidate += ".exe"
        if os.path.exists(candidate):
            return cuda_home, candidate

    guesses = ["/usr/local/cuda", "/opt/cuda"]
    for guess in guesses:
        candidate = os.path.join(guess, "bin", "nvcc")
        if os.path.exists(candidate):
            return guess, candidate

    return None, None


def cuda_lib_dir(cuda_home):
    """Return the platform-appropriate CUDA library directory."""
    if sys.platform == "win32":
        return os.path.join(cuda_home, "lib", "x64")
    # Linux: prefer lib64, fall back to lib
    lib64 = os.path.join(cuda_home, "lib64")
    if os.path.exists(lib64):
        return lib64
    return os.path.join(cuda_home, "lib")


def build_cuda_object(nvcc, src, obj, arch="sm_75", host_compiler=None, include_dirs=None):
    """Compile a .cu source into a relocatable object file with nvcc."""
    os.makedirs(os.path.dirname(obj) or ".", exist_ok=True)

    cmd = [
        nvcc,
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-DUSE_CUDA",
        f"-arch={arch}",
    ]

    if include_dirs:
        for inc_dir in include_dirs:
            cmd += [f"-I{inc_dir}"]

    if sys.platform != "win32":
        cmd += ["-Xcompiler", "-fPIC"]

    if host_compiler:
        cmd += ["-ccbin", host_compiler]

    cmd += ["-c", src, "-o", obj]

    print("nvcc:", " ".join(cmd))
    subprocess.check_call(cmd)


HERE = os.path.dirname(os.path.abspath(__file__))
CU_SRC = os.path.join(HERE, "src", "drpf", "drpf_cuda.cu")

CUDA_HOME, NVCC = find_cuda()

USE_CUDA = (
    CUDA_HOME is not None
    and not os.environ.get("DRPF_DISABLE_CUDA")
    and os.path.exists(CU_SRC)
)

cuda_objects = []
cuda_include_dirs = []
cuda_library_dirs = []
cuda_libraries = []
cuda_runtime_dirs = []

if USE_CUDA:
    print(f"CUDA found at {CUDA_HOME}; building GPU backend.")
    cuda_obj_name = "drpf_cuda.obj" if sys.platform == "win32" else "drpf_cuda.o"
    cuda_obj = os.path.join(HERE, "build", cuda_obj_name)
    arch = os.environ.get("DRPF_CUDA_ARCH", "sm_75")
    host_cxx = os.environ.get("DRPF_CUDA_HOST_COMPILER")  # optional override

    build_cuda_object(
        NVCC, 
        CU_SRC, 
        cuda_obj, 
        arch=arch, 
        host_compiler=host_cxx, 
        include_dirs=[os.path.join(CUDA_HOME, "include")]
    )

    cuda_objects.append(cuda_obj)
    cuda_include_dirs.append(os.path.join(CUDA_HOME, "include"))

    libdir = cuda_lib_dir(CUDA_HOME)
    cuda_library_dirs.append(libdir)
    cuda_runtime_dirs.append(libdir)
    cuda_libraries += ["cudart", "cublas"]
else:
    if os.environ.get("DRPF_DISABLE_CUDA"):
        print("DRPF_DISABLE_CUDA set; building CPU-only.")
    elif CUDA_HOME is None:
        print("CUDA not found (nvcc not on PATH); building CPU-only.")
    else:
        print("drpf_cuda.cu not present; building CPU-only.")


# ===============================================================
# 3. Compiler selection (macOS specific)
# ===============================================================
def configure_macos_compiler():
    global compile_flags, link_flags

    # 0) Respect explicit user choice first
    if os.environ.get("CC") and os.environ.get("CXX"):
        print(f"Using user-specified compiler: {os.environ['CC']} / {os.environ['CXX']}")
        return

    try:
        brew_prefix = subprocess.check_output(["brew", "--prefix"], text=True).strip()
    except Exception:
        raise RuntimeError("Homebrew is required on macOS for OpenMP support.")

    brew_bin = os.path.join(brew_prefix, "bin")

    # 1) Prefer Homebrew LLVM/Clang on macOS
    llvm_root = os.path.join(brew_prefix, "opt", "llvm")
    llvm_clang = os.path.join(llvm_root, "bin", "clang")
    llvm_clangxx = os.path.join(llvm_root, "bin", "clang++")
    omp_root = os.path.join(brew_prefix, "opt", "libomp")

    if os.path.exists(llvm_clang) and os.path.exists(llvm_clangxx):
        if not os.path.exists(omp_root):
            print("libomp not found. Attempting to install via Homebrew...")
            subprocess.check_call(["brew", "install", "libomp"])

        print(f"Using Homebrew LLVM Clang: {llvm_clangxx}")
        os.environ["CC"] = llvm_clang
        os.environ["CXX"] = llvm_clangxx

        compile_flags = [
            "-std=c++20",
            "-O3",
            "-ffast-math",
            "-funroll-loops",
            "-fno-strict-aliasing",
            "-DNDEBUG",
            "-fopenmp",
            f"-I{llvm_root}/include",
            f"-I{omp_root}/include",
        ]
        link_flags = [
            "-fopenmp",
            f"-L{llvm_root}/lib",
            f"-L{omp_root}/lib",
            "-lomp",
        ]
        return

    # 2) Fallback to Homebrew GCC, but pick newest by numeric version
    try:
        gcc_candidates = [f for f in os.listdir(brew_bin) if re.fullmatch(r"g\+\+-\d+", f)]
    except FileNotFoundError:
        gcc_candidates = []

    if gcc_candidates:
        gcc_candidates.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
        gpp_bin = os.path.join(brew_bin, gcc_candidates[0])
        gcc_bin = gpp_bin.replace("g++", "gcc")

        print(f"Using Homebrew GCC: {gpp_bin}")
        os.environ["CC"] = gcc_bin
        os.environ["CXX"] = gpp_bin

        compile_flags = [
            "-std=c++20",
            "-O3",
            "-ffast-math",
            "-funroll-loops",
            "-fno-strict-aliasing",
            "-DNDEBUG",
            "-fopenmp",
        ]
        link_flags = ["-fopenmp"]
        return

    # 3) Last resort: Apple Clang + libomp
    print("Homebrew LLVM/GCC not found. Falling back to Apple Clang...")
    clang_bin = shutil.which("clang")
    clangpp_bin = shutil.which("clang++")

    if not clang_bin or not clangpp_bin:
        raise RuntimeError("Apple Clang not found. Please install Xcode Command Line Tools.")

    if not os.path.exists(omp_root):
        print("libomp not found. Attempting to install via Homebrew...")
        subprocess.check_call(["brew", "install", "libomp"])

    os.environ["CC"] = clang_bin
    os.environ["CXX"] = clangpp_bin

    compile_flags = [
        "-std=c++20",
        "-O3",
        "-ffast-math",
        "-funroll-loops",
        "-fno-strict-aliasing",
        "-DNDEBUG",
        "-Xpreprocessor",
        "-fopenmp",
        f"-I{omp_root}/include",
    ]
    link_flags = [
        f"-L{omp_root}/lib",
        "-lomp",
    ]

# ===============================================================
# 4. Apply platform flags
# ===============================================================
if sys.platform == "darwin":
    configure_macos_compiler()
elif sys.platform.startswith("linux"):
    compile_flags = [
        "-std=c++20", "-O3", "-march=native", "-ffast-math",
        "-funroll-loops", "-fno-strict-aliasing", "-DNDEBUG", "-fopenmp",
    ]
    link_flags = ["-fopenmp"]
else:  # Windows
    compile_flags = [
        "/std:c++20",
        "/O2",
        "/fp:fast",
        "/DNDEBUG",
        "/openmp:experimental",
        "/Zc:cplusplus",
    ]
    link_flags = []

if USE_CUDA:
    if sys.platform == "win32":
        compile_flags.append("/DUSE_CUDA")
    else:
        compile_flags.append("-DUSE_CUDA")
        for d in cuda_runtime_dirs:
            link_flags.append(f"-Wl,-rpath,{d}")

# ===============================================================
# 5. Extension module definition
# ===============================================================
ext = Extension(
    "drpf.drpf",
    sources=["src/drpf/drpf.pyx"],
    language="c++",
    include_dirs=[
        eigen_include,
        np.get_include(),
    ] + cuda_include_dirs,       
    library_dirs=cuda_library_dirs,
    libraries=cuda_libraries,      
    runtime_library_dirs=cuda_runtime_dirs if sys.platform != "win32" else None,
    extra_objects=cuda_objects,
    extra_compile_args=compile_flags,
    extra_link_args=link_flags,
)

# ===============================================================
# 6. Build
# ===============================================================
setup(
    name="drpf",
    version="0.9.5",
    author="Panagiotis Papakonstantinou",
    author_email="panagiotispapakonstantinou15@gmail.com",
    description="Dense Random Projection Forest for Fast ANN Search",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/Pappan24/drpf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    license="MIT",
)