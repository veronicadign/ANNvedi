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
    Pybind11Extension(
        "multiprobe_lsh_cpp",
        ["src/multiprobe_lsh.cpp"],
        cxx_std=14,
        include_dirs=["src"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-pthread"],
        extra_link_args=["-pthread"],
    ),
    # The newest IVF-LSH fork (per-dimension SQ8 scales, mean-centered
    # projections) lives only in the ivf_lsh bundle. Build it here too so the
    # dev facade / tuner can use it. NOTE: it must compile against ITS OWN
    # simd.h (incompatible per-dim signature) - quoted #include resolves to
    # the sibling header, so keep its include dir, never src/.
    Pybind11Extension(
        "ivf_lsh_cpp",
        ["competitors/ivf_lsh/src/lsh_index_optimized.cpp"],
        cxx_std=14,
        include_dirs=["competitors/ivf_lsh/src"],
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