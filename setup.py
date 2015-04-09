from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

model_module = Extension(
    "lineprofile.model",
    ["lineprofile/model.pyx"],
    extra_compile_args=['-O3','-ffast-math'],
    extra_link_args=['-lfftw3'],
    include_dirs=[numpy.get_include()]
)

fitter_module = Extension(
    "lineprofile.fitter",
    ["lineprofile/fitter.pyx"],
    extra_compile_args=['-O3','-ffast-math'],
    extra_link_args=['-lfftw3'],
    include_dirs=[numpy.get_include()]
)

setup(
    name='lineprofile',
    version='0.0.1',
    cmdclass={'build_ext': build_ext},
    ext_modules=[model_module, fitter_module],
    packages=['lineprofile']
)
