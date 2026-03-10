from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "linear_ann_cpp",
        ["src/linear_index.cpp"],
        cxx_std=14,
    ),
]

setup(
    name="linear_ann_cpp",
    version="0.1",
    author="Tu",
    description="Modulo ANN lineare",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)