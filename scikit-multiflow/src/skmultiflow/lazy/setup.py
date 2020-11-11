import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("lazy", parent_package, top_path)

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config.add_extension('libNearestNeighbor',
                          sources=[os.path.join('src', 'libNearestNeighbors', 'nearestNeighbor.cpp')],
                          include_dirs=[numpy.get_include()],
                          libraries=libraries,
                          language='c++')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
