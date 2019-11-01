import os

from distutils.version import LooseVersion


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('skmultiflow', parent_package, top_path)

    # submodules which do not have their own setup.py
    config.add_subpackage('anomaly_detection')
    config.add_subpackage('bayes')
    config.add_subpackage('core')
    config.add_subpackage('data')
    config.add_subpackage('drift_detection')
    config.add_subpackage('evaluation')
    config.add_subpackage('meta')
    config.add_subpackage('neural_networks')
    config.add_subpackage('prototype')
    config.add_subpackage('rules')
    config.add_subpackage('transform')
    config.add_subpackage('trees')
    config.add_subpackage('utils')
    config.add_subpackage('visualization')

    # submodules which have their own setup.py
    config.add_subpackage('lazy')
    config.add_subpackage('metrics')

    # Check if should run cythonize or build from source files
    maybe_cythonize_extensions(top_path, config)

    return config


########################################################################################################################
# Utilities useful during the build. Based on sklearn._build_utils                                                     #
########################################################################################################################

DEFAULT_ROOT = 'skmultiflow'
# on conda, this is the latest for python 3.5
CYTHON_MIN_VERSION = '0.28.5'


def build_from_c_and_cpp_files(extensions):
    """Modify the extensions to build from the .c and .cpp files.

    This is useful for releases, this way cython is not required to
    run python setup.py install.
    """
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources = sources


def maybe_cythonize_extensions(top_path, config):
    """Tweaks for building extensions between release and development mode."""
    is_release = os.path.exists(os.path.join(top_path, 'PKG-INFO'))
    if is_release:
        build_from_c_and_cpp_files(config.ext_modules)
    else:
        message = ('Please install cython with a version >= {0} in order '
                   'to build a scikit-multiflow development version.').format(
                       CYTHON_MIN_VERSION)
        try:
            import Cython
            if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
                message += ' Your version of Cython was {0}.'.format(
                    Cython.__version__)
                raise ValueError(message)
            from Cython.Build import cythonize
        except ImportError as exc:
            exc.args += (message,)
            raise
        config.ext_modules = cythonize(config.ext_modules,
                                       compiler_directives={'language_level': 3})


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
