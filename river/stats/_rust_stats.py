from __future__ import annotations

# Pure-Python fallback loaded when the compiled Rust extension is absent.
# The compiled extension (_rust_stats.*.so) takes precedence over this file
# when Rust was available at build time.

_MSG = (
    "The '_rust_stats' Rust extension was not compiled when river was installed. "
    "A Rust compiler is required to build it. "
    "Install Rust from https://rustup.rs, then reinstall river from source: "
    "pip install --no-binary river river"
)


class _RustNotAvailable:
    def __init__(self, *args, **kwargs):
        raise ImportError(_MSG)


RsQuantile = type("RsQuantile", (_RustNotAvailable,), {})
RsRollingQuantile = type("RsRollingQuantile", (_RustNotAvailable,), {})
RsEWMean = type("RsEWMean", (_RustNotAvailable,), {})
RsEWVar = type("RsEWVar", (_RustNotAvailable,), {})
RsIQR = type("RsIQR", (_RustNotAvailable,), {})
RsRollingIQR = type("RsRollingIQR", (_RustNotAvailable,), {})
RsKurtosis = type("RsKurtosis", (_RustNotAvailable,), {})
RsSkew = type("RsSkew", (_RustNotAvailable,), {})
RsPeakToPeak = type("RsPeakToPeak", (_RustNotAvailable,), {})
