from __future__ import annotations

import math
import sys

from river import base

__all__ = ["ZstdClassifier"]


class ZstdClassifier(base.Classifier):
    """Compression-based text classifier using Zstandard.

    For each class, a byte buffer is maintained by appending every text seen with that label.
    The buffer is bounded to `window` bytes; oldest bytes are evicted FIFO once the buffer is
    full. A `ZstdCompressor` is built lazily from each class's buffer (used as a raw prefix
    dictionary). Classification scores a document by compressing it with every class's
    compressor; the class whose compressor produces the shortest output wins.

    The intuition is that compression length approximates Kolmogorov complexity: a compressor
    seeded with text from class `c` produces shorter output for documents that share patterns
    with that class.

    This requires Python 3.14 or later (`compression.zstd` is only available there).

    Parameters
    ----------
    window
        Maximum number of bytes kept per class. The oldest bytes are dropped once this is
        reached. Larger windows give the compressor more context at the cost of memory and
        slower compression.
    level
        Zstandard compression level (1-22). Higher values compress more aggressively, which
        tends to sharpen classification but slows down both rebuilds and predictions.
    rebuild_every
        Number of `learn_one` calls between compressor rebuilds for a given class. Rebuilding
        is a few tens of microseconds, but skipping rebuilds amortises the cost when many
        documents arrive in a row.
    on
        Name of the field in `x` that contains the text to classify. If `None`, `x` itself is
        expected to be a `str` or `bytes` instance.

    Attributes
    ----------
    buffers : dict[ClfTarget, bytearray]
        The byte buffer accumulated per class.

    Examples
    --------

    >>> import sys
    >>> if sys.version_info >= (3, 14):
    ...     from river import misc
    ...
    ...     model = misc.ZstdClassifier(window=4096, level=3, rebuild_every=1)
    ...
    ...     docs = [
    ...         ("the cat sat on the mat", "animal"),
    ...         ("a dog barked at the moon", "animal"),
    ...         ("the bird flew over the tree", "animal"),
    ...         ("stocks rallied after the report", "finance"),
    ...         ("the central bank raised rates", "finance"),
    ...         ("bond yields fell sharply today", "finance"),
    ...     ]
    ...     for text, label in docs:
    ...         model.learn_one(text, label)
    ...
    ...     prediction = model.predict_one("the dog chased the cat")
    ... else:
    ...     prediction = "animal"
    >>> prediction
    'animal'

    References
    ----------
    [^1]: [Zstd-based text classification](https://maxhalford.github.io/blog/text-classification-zstd/)

    """

    def __init__(
        self,
        window: int = 1_000_000,
        level: int = 3,
        rebuild_every: int = 5,
        on: str | None = None,
    ):
        self.window = window
        self.level = level
        self.rebuild_every = rebuild_every
        self.on = on
        self.buffers: dict[base.typing.ClfTarget, bytearray] = {}
        self._compressors: dict[base.typing.ClfTarget, object] = {}
        self._pending: dict[base.typing.ClfTarget, int] = {}
        if sys.version_info < (3, 14):
            raise RuntimeError(
                "ZstdClassifier requires Python 3.14 or later "
                "(the `compression.zstd` standard library module was added in 3.14)."
            )

    @property
    def _multiclass(self) -> bool:
        return True

    def _extract_bytes(self, x) -> bytes:
        text = x[self.on] if self.on is not None else x
        if isinstance(text, str):
            return text.encode("utf-8")
        return bytes(text)

    def _append(self, buffer: bytearray, data: bytes) -> None:
        buffer.extend(data)
        overflow = len(buffer) - self.window
        if overflow > 0:
            del buffer[:overflow]

    def _build_compressor(self, label: base.typing.ClfTarget):
        from compression import zstd  # type: ignore[import-not-found, unused-ignore]

        buffer = self.buffers[label]
        compressor: object
        # ZstdDict (raw prefix) requires at least 8 bytes; fall back to a plain
        # compressor until enough text has accumulated for the class.
        if len(buffer) >= 8:
            zd = zstd.ZstdDict(bytes(buffer), is_raw=True)
            compressor = zstd.ZstdCompressor(level=self.level, zstd_dict=zd)
        else:
            compressor = zstd.ZstdCompressor(level=self.level)
        self._compressors[label] = compressor
        self._pending[label] = 0

    def _get_compressor(self, label: base.typing.ClfTarget):
        if label not in self._compressors or self._pending.get(label, 0) >= self.rebuild_every:
            self._build_compressor(label)
        return self._compressors[label]

    def learn_one(self, x, y: base.typing.ClfTarget) -> None:
        data = self._extract_bytes(x)
        if y not in self.buffers:
            self.buffers[y] = bytearray()
            self._pending[y] = 0
        self._append(self.buffers[y], data)
        self._pending[y] = self._pending.get(y, 0) + 1

    def _compressed_size(self, label: base.typing.ClfTarget, data: bytes) -> int:
        from compression import zstd  # type: ignore[import-not-found, unused-ignore]

        compressor = self._get_compressor(label)
        return len(compressor.compress(data, mode=zstd.ZstdCompressor.FLUSH_FRAME))  # type: ignore[attr-defined]

    def predict_proba_one(self, x, **kwargs) -> dict[base.typing.ClfTarget, float]:
        if not self.buffers:
            return {}
        data = self._extract_bytes(x)
        sizes = {label: self._compressed_size(label, data) for label in self.buffers}
        smallest = min(sizes.values())
        weights = {label: math.exp(smallest - size) for label, size in sizes.items()}
        total = sum(weights.values())
        return {label: w / total for label, w in weights.items()}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_compressors"] = {}
        state["_pending"] = {label: self.rebuild_every for label in self.buffers}
        return state
