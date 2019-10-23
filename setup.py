#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import Extension, find_packages, setup, Command

try:
    from Cython.Build import cythonize
except ImportError:
    # Create closure for deferred import
    def cythonize(*args, **kwargs):
        from Cython.Build import cythonize
        return cythonize(*args, ** kwargs)


# Package meta-data.
NAME = 'creme'
DESCRIPTION = 'Incremental machine learning in Python'
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
URL = 'https://github.com/creme-ml/creme'
EMAIL = 'maxhalford25@gmail.com'
AUTHOR = 'Max Halford'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None

# Package requirements.
base_packages = ['numpy>=1.16.4', 'scipy>=1.3.0', 'scikit-learn>=0.21.2']

dev_packages = [
    'Cython>=0.29.6',
    'graphviz>=0.10.1',
    'matplotlib>=3.0.2',
    'nbval>=0.9.1',
    'pytest>=4.5.0',
    'pytest-cov>=2.6.1',
    'pytest-cython>=0.1.0',
]

docs_packages = dev_packages + [
    'ipykernel>=4.8.2',
    'jupyter-client>=5.2.3',
    'm2r>=0.2.1',
    'nbsphinx>=0.4.2',
    'Sphinx>=2.2.0',
    'sphinx-material>=0.0.12'
]

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=base_packages,
    extras_require={'dev': dev_packages, 'docs': docs_packages},
    include_package_data=True,
    license='BSD-3',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    ext_modules=cythonize([Extension('*', sources=['**/*.pyx'], libraries=['m'])])
)
