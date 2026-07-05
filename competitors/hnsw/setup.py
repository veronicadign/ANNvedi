from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

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
    name="hnsw_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
