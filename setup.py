from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "_ferns",
        ["fc/_ferns.pyx"],
	include_dirs=[numpy.get_include()]
    ),
]

setup(
    name = "fc",
    version = "1.0",
    install_requires=["numpy", "scikit-learn", "cython"],
    packages = find_packages(),
    ext_modules = cythonize(extensions)
)
