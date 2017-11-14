import os
import numpy as np
from setuptools import setup, find_packages
from distutils.core import Extension
from distutils.sysconfig import get_python_inc


"""
### Dependencies
* python3
* matplotlib
* numpy
* scipy
* pandas
* scikit-learn
* libNearestNeighbors
* sortedcontainers

"""


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


nnExtension = Extension('libNearestNeighbor',
                        include_dirs=[get_python_inc(), np.get_include()],
                        libraries=[],
                        library_dirs=[],
                        extra_compile_args=['-O3'],
                        sources=['skmultiflow/classification/lazy/libNearestNeighbors/nearestNeighbor.cpp'])

setup(name="scikit-multiflow",
      version="0.1.0",
      author="Jacob MONTIEL",
      # author_email="",
      description="Setup for the scikit-multiflow package",
      packages=find_packages(),
      long_description=read('README.md'),ext_modules=[nnExtension], install_requires=['sortedcontainers', 'numpy']
      )
