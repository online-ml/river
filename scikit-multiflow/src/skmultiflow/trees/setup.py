from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration('trees', parent_package, top_path)

    # submodules which do not have their own setup.py
    config.add_subpackage('attribute_observer')
    config.add_subpackage('attribute_test')
    config.add_subpackage('nodes')
    config.add_subpackage('split_criterion')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
