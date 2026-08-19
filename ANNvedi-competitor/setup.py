import shutil
import subprocess
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Command, setup

HERE = Path(__file__).resolve().parent
NVCC_CMD = [
    "nvcc", "-O3", "-shared", "-Xcompiler", "-fPIC",
    # PTX so the image builds on GPU-less machines and JITs on the target GPU
    "-gencode", "arch=compute_80,code=compute_80",
    str(HERE / "src" / "gpu_filter.cu"), "-o", str(HERE / "libgpufilter.so"),
]


def build_gpu_lib():
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        for cand in sorted(Path("/usr/local").glob("cuda*/bin/nvcc")):
            nvcc = str(cand)
            break
    if nvcc is None:
        print("setup.py: nvcc not found — skipping libgpufilter.so "
              "(the algorithm falls back to hnsw at runtime)", file=sys.stderr)
        return False
    cmd = [nvcc] + NVCC_CMD[1:]
    print("setup.py:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"setup.py: nvcc failed ({e}) — skipping libgpufilter.so "
              "(the algorithm falls back to hnsw at runtime)", file=sys.stderr)
        return False
    return True


class BuildGpu(Command):
    """python3 setup.py build_gpu -> compiles src/gpu_filter.cu to ./libgpufilter.so"""
    description = "compile the CUDA GPU-filter library with nvcc"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        if not build_gpu_lib():
            raise SystemExit("nvcc not found — install a CUDA toolkit to build the GPU library")


class BuildExtAndGpu(build_ext):
    """build_ext also attempts the GPU library (skips gracefully without nvcc)."""

    def run(self):
        super().run()
        build_gpu_lib()


ext_modules = [
    Pybind11Extension(
        "hnsw_cpp",
        ["src/hnsw.cpp"],
        cxx_std=17,
        include_dirs=["src"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-pthread"],
        extra_link_args=["-pthread"],
    ),
]

setup(
    name="annvedi_competitor",
    version="1.0",
    author="ANNvedi",
    description="Self-calibrating ANN contestant: GPU quantized filter + CPU HNSW",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtAndGpu, "build_gpu": BuildGpu},
    zip_safe=False,
)
