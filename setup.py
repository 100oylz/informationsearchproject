from setuptools import setup
import numpy 
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("svd/FunkSVD.pyx"),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'numpy',
        # Add other dependencies here
    ]
)