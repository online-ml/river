import platform

import numpy
import setuptools
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError
from setuptools_rust import Binding, RustExtension

ext_modules = cythonize(
    module_list=[
        setuptools.Extension(
            "*",
            sources=["river/**/*.pyx"],
            include_dirs=[numpy.get_include()],
            libraries=[] if platform.system() == "Windows" else ["m"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    ],
    compiler_directives={
        "binding": True,
        "embedsignature": True,
    },
)

rust_extensions = [RustExtension("river.stats._rust_stats", binding=Binding.PyO3)]


class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):
    def run(self):
        try:
            build_ext.run(self)
        except (FileNotFoundError):
            raise BuildFailed("File not found. Could not compile C extension.")

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, ValueError):
            raise BuildFailed("Could not compile C extension.")


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": ExtBuilder},
            "rust_extensions": rust_extensions,
            "zip_safe": False,
            "include_package_data": True,
        }
    )
