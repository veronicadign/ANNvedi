from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "lsh_cpp_module",
        ["src/lsh_index.cpp"],
        cxx_std=14,
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    ),
]

setup(
    name="lsh_cpp_module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
