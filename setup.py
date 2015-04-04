from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_module = Extension(
    "lineprofile.profile",
    ["lineprofile/profile.pyx"],
    extra_compile_args=['-O3'],
    extra_link_args=['-lfftw3'],
    include_dirs = [numpy.get_include()]
)

setup(
    name = 'lineprofile',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
    packages=['lineprofile']
)
