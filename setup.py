from __future__ import annotations

import setuptools
from setuptools_rust import Binding, RustExtension

rust_extensions = [RustExtension("river.stats._rust_stats", binding=Binding.PyO3, debug=False)]

setuptools.setup(
    rust_extensions=rust_extensions,
    zip_safe=False,
    include_package_data=True,
)
