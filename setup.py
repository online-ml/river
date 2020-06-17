import sys
import os
import platform
import builtins
from os import path

from distutils.command.sdist import sdist

# Optional setuptools features
# We need to import setuptools early, if we want setuptools features,
# as it monkey-patches the 'setup' function
import setuptools    # Actually used, do not remove

# This is a bit (!) hackish: we are setting a global variable so that the
# main skmultiflow __init__ can detect if it is being loaded by the setup
# routine, to avoid attempting to load components that aren't built yet:
# the numpy distutils extensions that are used by scikit-multiflow to
# recursively build the compiled extensions in sub-packages is based on the
# Python import machinery.
builtins.__SKMULTIFLOW_SETUP__ = True

DIST_NAME = 'scikit-multiflow'
DESCRIPTION = 'A machine learning package for streaming data in Python.'
MAINTAINER = 'Jacob Montiel'
MAINTAINER_EMAIL = 'jacob.montiel@waikato.ac.nz'
URL = 'https://scikit-multiflow.github.io/'
PROJECT_URLS = {
    'Documentation': 'https://scikit-multiflow.github.io/scikit-multiflow/',
    'Source code': 'https://github.com/scikit-multiflow/scikit-multiflow',
    'Bug Tracker': 'https://github.com/scikit-multiflow/scikit-multiflow/issues',
    'Azure Pipelines': 'https://dev.azure.com/scikit-multiflow/scikit-multiflow',
}
DOWNLOAD_URL = 'https://pypi.org/project/scikit-multiflow/#files'
LICENSE = '3-Clause BSD'

# get __version__ from _version.py
ver_file = os.path.join('src/skmultiflow', '_version.py')
with open(ver_file) as f:
    exec(f.read())
VERSION = __version__  # noqa

# read the contents of README file
pkg_directory = path.abspath(path.dirname(__file__))
with open(path.join(pkg_directory, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt') as fid:
    INSTALL_REQUIRES = [line.strip() for line in fid.readlines() if line]


def configuration(parent_package='', top_path=None):
    # Returns a dictionary suitable for passing to numpy.distutils.core.setup(..)

    from numpy.distutils.misc_util import Configuration

    # BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
    # properly updated when the contents of directories change (true for distutils,
    # not sure about setuptools).
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('skmultiflow', subpackage_path='src/skmultiflow')

    return config


def setup_package():
    try:
        from numpy import get_include
    except ImportError:
        print('To install scikit-multiflow first install numpy.\n' +
              'For example, using pip:\n' +
              '$ pip install -U numpy')
        sys.exit(1)

    metadata = dict(name=DIST_NAME,
                    version=VERSION,
                    license=LICENSE,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    project_urls=PROJECT_URLS,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    long_description=LONG_DESCRIPTION,
                    long_description_content_type='text/markdown',
                    package_dir={'': 'src'},
                    install_requires=INSTALL_REQUIRES,
                    setup_requires=['pytest-runner'],
                    tests_require=['pytest'],
                    classifiers=["Intended Audience :: Developers",
                                 "Intended Audience :: Science/Research",
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 "Programming Language :: Python :: 3",
                                 'Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.6',
                                 'Programming Language :: Python :: 3.7',
                                 'Programming Language :: Python :: 3.8',
                                 "Topic :: Scientific/Engineering",
                                 "Topic :: Scientific/Engineering :: Artificial Intelligence",
                                 "Topic :: Software Development",
                                 "License :: OSI Approved :: BSD License",
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Operating System :: Microsoft :: Windows'
                                 ],
                    python_requires=">=3.5",
                    zip_safe=False,  # the package can run out of an .egg file
                    include_package_data=True,
                    cmdclass={'sdist': sdist})

    if sys.version_info < (3, 5):
        raise RuntimeError("scikit-multiflow requires Python 3.5 or later. "
                           "The current Python version is {} installed in {}}.".
                           format(platform.python_version(), sys.executable))

    from numpy.distutils.core import setup

    metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
    del builtins.__SKMULTIFLOW_SETUP__   # noqa
