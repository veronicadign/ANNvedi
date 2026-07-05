from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "lsh_cpp_optimized",
        ["src/lsh_index_optimized.cpp"],
        cxx_std=14,
        include_dirs=["src"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-pthread"],
        extra_link_args=["-pthread"],
    ),
]

setup(
    name="lsh_cpp_optimized",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
