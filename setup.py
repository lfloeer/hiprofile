from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_module = Extension(
    "cyprofile",
    ["profile.pyx"],
    #extra_compile_args=['-O3'],
    extra_compile_args=['-O3','-fopenmp'],
    extra_link_args=['-fopenmp'],
    include_dirs = [numpy.get_include()]
)

setup(
    name = 'lineprofile',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)
