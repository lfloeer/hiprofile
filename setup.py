from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy
import platform
import os

"""
Requires fftw (brew install fftw)
"""
EXTRA_COMPILE_ARGS = ['-O3', '-march=native']
EXTRA_LINK_ARGS = ['-lfftw3', '-lm']
INCLUDE_DIRS = [numpy.get_include()]

MODEL_MODULE = Extension(
    "lineprofile.model",
    ["lineprofile/model.pyx"],
    extra_compile_args=EXTRA_COMPILE_ARGS,
    extra_link_args=EXTRA_LINK_ARGS,
    include_dirs=INCLUDE_DIRS,
)

FITTER_MODULE = Extension(
    "lineprofile.fitter",
    ["lineprofile/fitter.pyx"],
    extra_compile_args=EXTRA_COMPILE_ARGS,
    extra_link_args=EXTRA_LINK_ARGS,
    include_dirs=INCLUDE_DIRS,
)

setup(
    name='lineprofile',
    version='0.0.1',
    cmdclass={'build_ext': build_ext},
    ext_modules=[MODEL_MODULE, FITTER_MODULE],
    packages=['lineprofile'],
)
