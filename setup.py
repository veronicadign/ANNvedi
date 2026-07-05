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
        "lsh_cpp_module",
        ["src/lsh_index.cpp"],
        cxx_std=14,
        include_dirs=["src"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-pthread"],
        extra_link_args=["-pthread"],
    ),
    Pybind11Extension(
        "hnsw_cpp",
        ["HNSW/src/hnsw.cpp"],
        cxx_std=17,
        include_dirs=["HNSW/src"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-pthread"],
        extra_link_args=["-pthread"],
    ),
    Pybind11Extension(
        "multiprobe_lsh_cpp",
        ["MultiProbe_LSH/src/multiprobe_lsh.cpp"],
        cxx_std=14,
        include_dirs=["MultiProbe_LSH/src"],
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