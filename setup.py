from setuptools import setup
import numpy 
import tqdm
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("svd/FunkSVD.pyx"),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'numpy',
        'tqdm'
        # Add other dependencies here
    ]
)