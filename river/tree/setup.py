from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("trees", parent_package, top_path)

    # submodules which do not have their own setup.py
    config.add_subpackage("_attribute_observer")
    config.add_subpackage("_attribute_test")
    config.add_subpackage("_nodes")
    config.add_subpackage("_split_criterion")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())
