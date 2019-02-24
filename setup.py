import sys
from setuptools import setup, find_packages
from distutils.core import Extension
from distutils.sysconfig import get_python_inc
from os import path
from glob import glob


DIST_NAME = 'scikit-multiflow'
DESCRIPTION = 'A multi-output/multi-label and stream data mining framework.'
MAINTAINER = 'Jacob Montiel'
MAINTAINER_EMAIL = ''
URL = 'https://scikit-multiflow.github.io/'
PROJECT_URLS = {'Packaging tutorial': 'https://packaging.python.org/tutorials/distributing-packages/',
                'Travis CI': 'https://travis-ci.org/scikit-multiflow/scikit-multiflow',
                'Documentation': 'https://scikit-multiflow.github.io/scikit-multiflow/',
                'Source code': 'https://github.com/scikit-multiflow/scikit-multiflow',
                }
DOWNLOAD_URL = 'https://pypi.org/project/scikit-multiflow/#files'
LICENSE = '3-Clause BSD'
VERSION = '0.1.2'


# read the contents of README file
pkg_directory = path.abspath(path.dirname(__file__))
with open(path.join(pkg_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

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

    nn_extension = Extension('libNearestNeighbor',
                             include_dirs=[get_python_inc(), get_include()],
                             libraries=[],
                             library_dirs=[],
                             extra_compile_args=['-O3'],
                             sources=['src/skmultiflow/lazy/src/libNearestNeighbors/nearestNeighbor.cpp'])

    setup(name=DIST_NAME,
          version=VERSION,
          license=LICENSE,
          url=URL,
          download_url=DOWNLOAD_URL,
          project_urls=PROJECT_URLS,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=long_description,
          long_description_content_type='text/markdown',
          packages=find_packages('src', exclude=['docs', 'tests', '*.tests', 'test_*']),
          package_dir={'': 'src'},
          py_modules=[path.splitext(path.basename(src_path))[0] for src_path in glob('src/*.py')],
          include_package_data=True,
          zip_safe=False,
          ext_modules=[nn_extension],
          install_requires=INSTALL_REQUIRES,
          setup_requires=['pytest-runner'],
          tests_require=['pytest'],
          classifiers=["Intended Audience :: Developers",
                       "Intended Audience :: Science/Research",
                       "Programming Language :: Python :: 3 :: Only",
                       "Topic :: Scientific/Engineering",
                       "Topic :: Scientific/Engineering :: Artificial Intelligence",
                       "Topic :: Software Development",
                       "License :: OSI Approved :: BSD License"
                       ]
          )
