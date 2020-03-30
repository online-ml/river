import io
import platform
import os
import setuptools
import sys

try:
    from Cython.Build import cythonize
except ImportError:
    import subprocess

    errno = subprocess.call([sys.executable, '-m', 'pip', 'install', 'Cython'])
    if errno:
        print('Please install Cython')
        raise SystemExit(errno)
    else:
        from Cython.Build import cythonize


# Package meta-data.
NAME = 'creme'
DESCRIPTION = 'Online machine learning in Python'
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
URL = 'https://github.com/creme-ml/creme'
EMAIL = 'maxhalford25@gmail.com'
AUTHOR = 'Max Halford'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None

# Package requirements.
base_packages = ['mmh3==2.5.1', 'numpy>=1.18.1', 'scipy>=1.4.1']

compat_packages = base_packages + [
    'scikit-learn>=0.22.1',
    'pandas>=1.0.1',
    'torch>=1.4.0'
]

dev_packages = [
    'flake8>=3.7.9',
    'graphviz>=0.10.1',
    'matplotlib>=3.0.2',
    'mypy>=0.761',
    'nbval>=0.9.1',
    'pytest>=4.5.0',
    'pytest-cov>=2.6.1',
    'pytest-cython>=0.1.0',
    'scikit-learn>=0.22.1'
]

docs_packages = dev_packages + [
    'ipykernel>=4.8.2',
    'jupyter-client>=5.2.3',
    'nbsphinx>=0.5.1',
    'Sphinx>=2.2.0',
    'sphinx-autobuild>=0.7.1',
    'sphinx-material>=0.0.21',
    'sphinx-copybutton>=0.2.8'
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
setuptools.setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(exclude=('tests',)),
    install_requires=base_packages,
    extras_require={
        'dev': dev_packages,
        'compat': compat_packages,
        'docs': docs_packages
    },
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
    ext_modules=cythonize(
        module_list=[
            setuptools.Extension(
                '*',
                sources=['**/*.pyx'],
                libraries=[] if platform.system() == 'Windows' else ['m']
            )
        ],
        compiler_directives={'language_level': 3}
    )
)
