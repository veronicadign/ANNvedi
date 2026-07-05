from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "linear_ann_cpp",
        ["src/linear_index.cpp"],
        cxx_std=14,
        include_dirs=["src"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-pthread"],
        extra_link_args=["-pthread"],
    ),
    Pybind11Extension(
        "lsh_cpp_simple",
        ["src/lsh_index_simple.cpp"],
        cxx_std=14,
        include_dirs=["src"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-pthread"],
        extra_link_args=["-pthread"],
    ),
    Pybind11Extension(
        "lsh_cpp_optimized",
        ["src/lsh_index_optimized.cpp"],
        cxx_std=14,
        include_dirs=["src"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-pthread"],
        extra_link_args=["-pthread"],
    ),
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
    name="ann_indices",
    version="0.1",
    author="ANNvedi",
    description="Unified ANN Indices: Linear, IVF-LSH, HNSW, MultiProbe LSH",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)