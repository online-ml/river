import sys
from setuptools import setup, find_packages
from distutils.core import Extension
from distutils.sysconfig import get_python_inc
from os.path import basename
from os.path import splitext
from os.path import join
from os.path import dirname
from glob import glob


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

DISTNAME = 'scikit-multiflow'
DESCRIPTION = 'Stream data and multi-label framework'
MAINTAINER = 'Jacob MONTIEL'
MAINTAINER_EMAIL = ''
URL = ''
LICENSE = '3-Clause BSD'
VERSION = '0.1.0'


def read(fname):
    return open(join(dirname(__file__), fname)).read()


with open('requirements.txt') as fid:
    INSTALL_REQUIRES = [l.strip() for l in fid.readlines() if l]

if __name__ == "__main__":
    try:
        from numpy import get_include
    except ImportError:
        print('To install scikit-multiflow first install numpy.\n' +
              'For example, using pip:\n' +
              '$ pip install -U numpy')
        sys.exit(1)

    nnExtension = Extension('libNearestNeighbor',
                            include_dirs=[get_python_inc(), get_include()],
                            libraries=[],
                            library_dirs=[],
                            extra_compile_args=['-O3'],
                            sources=['src/skmultiflow/classification/lazy/libNearestNeighbors/nearestNeighbor.cpp'])

    setup(name=DISTNAME,
          version=VERSION,
          license=LICENSE,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=read('README.md'),
          packages=find_packages('src', exclude=['docs', 'tests', '*.tests', 'test_*']),
          package_dir={'': 'src'},
          py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
          include_package_data=True,
          zip_safe=False,
          ext_modules=[nnExtension],
          install_requires=INSTALL_REQUIRES,
          setup_requires=['pytest-runner'],
          tests_require=['pytest'],
          )
