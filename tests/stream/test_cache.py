from __future__ import annotations

import functools
import gc
import platform
import typing

import pytest

from river import stream, utils
from river.stream.cache import CACHE_SUFFIX

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


def keyless_function() -> Iterator[int]:
    """A generator function, i.e. the only thing `Cache.__call__` guesses a key from."""
    yield 1


def fill(cache: stream.Cache, key: str, elements: list[typing.Any]) -> None:
    """Runs a full first pass, so that `key` ends up cached."""
    assert list(cache(elements, key=key)) == elements


def recording_source(consumed: list[int], n: int = 3) -> Iterator[int]:
    """A stream of `n` integers that appends to `consumed` as it is iterated."""
    for i in range(n):
        consumed.append(i)
        yield i


def read_key(cache: stream.Cache, key: str) -> None:
    """Iterates a cached stream, which is when its file is actually opened."""
    _ = list(cache[key])


def patch_system(monkeypatch: pytest.MonkeyPatch, system: str) -> None:
    """Makes `platform.system` report `system`, whatever the host actually is."""

    def fake_system() -> str:
        return system

    monkeypatch.setattr(platform, "system", fake_system)


ROUND_TRIP_CASES: dict[str, list[typing.Any]] = {
    "empty": [],
    "scalars": [1, 2, 3],
    "strings": ["a", "b"],
    "river-rows": [({"a": 1.0, "b": 2}, True), ({"a": 3.0, "b": 4}, False)],
    "nested": [{"a": [1, {"b": (2, 3)}]}],
}

DISCOVERY_CASES: dict[str, tuple[list[str], set[str]]] = {
    "nothing": ([], set()),
    "one-key": (["k.river_cache.pkl"], {"k"}),
    "dotted-key": (["phishing.v2.river_cache.pkl"], {"phishing.v2"}),
    "other-extensions": (["notes.txt", "k.pkl"], set()),
    "mixed": (["k.river_cache.pkl", "notes.txt"], {"k"}),
}


KEYLESS_CASES: dict[str, tuple[typing.Any, type[Exception], str]] = {
    "an-iterable": ([1, 2], ValueError, "No default key could be guessed"),
    "a-function": (keyless_function, TypeError, "'function' object is not iterable"),
}

UNKNOWN_KEY_CASES: dict[str, Callable[[stream.Cache], None]] = {
    "getitem": functools.partial(read_key, key="nope"),
    "clear": functools.partial(stream.Cache.clear, key="nope"),
}

CLEAR_CASES: dict[str, tuple[list[str], Callable[[stream.Cache], None], set[str]]] = {
    "the-only-key": (["a"], functools.partial(stream.Cache.clear, key="a"), set()),
    "one-of-two-keys": (["a", "b"], functools.partial(stream.Cache.clear, key="a"), {"b"}),
    "all-of-two-keys": (["a", "b"], stream.Cache.clear_all, set()),
    "all-of-nothing": ([], stream.Cache.clear_all, set()),
}


@pytest.mark.parametrize("elements", ROUND_TRIP_CASES.values(), ids=ROUND_TRIP_CASES)
def test_elements_round_trip(tmp_path: Path, elements: list[typing.Any]) -> None:
    cache = stream.Cache(directory=tmp_path)
    fill(cache, "k", elements)
    # An empty second source, so that whatever comes out can only have come from the file.
    assert list(cache([], key="k")) == elements


def test_cache_hit_ignores_source(tmp_path: Path) -> None:
    first: list[int] = []
    second: list[int] = []
    cache = stream.Cache(directory=tmp_path)

    assert list(cache(recording_source(first), key="k")) == [0, 1, 2]
    assert first == [0, 1, 2]

    assert list(cache(recording_source(second, n=10), key="k")) == [0, 1, 2]
    assert second == []


def test_first_pass_is_lazy(tmp_path: Path) -> None:
    consumed: list[int] = []
    rows = stream.Cache(directory=tmp_path)(recording_source(consumed), key="k")

    assert consumed == []
    assert next(rows) == 0
    assert consumed == [0]


def test_key_registers_once_the_stream_is_exhausted(tmp_path: Path) -> None:
    cache = stream.Cache(directory=tmp_path)
    rows = cache([1, 2], key="k")

    assert next(rows) == 1
    assert cache.keys == set()

    assert list(rows) == [2]
    assert cache.keys == {"k"}


@pytest.mark.parametrize(("filenames", "expected"), DISCOVERY_CASES.values(), ids=DISCOVERY_CASES)
def test_keys_are_discovered_on_construction(
    tmp_path: Path, filenames: list[str], expected: set[str]
) -> None:
    for name in filenames:
        _ = (tmp_path / name).write_bytes(b"")
    assert stream.Cache(directory=tmp_path).keys == expected


def test_dotted_key_survives_full_round_trip(tmp_path: Path) -> None:
    fill(stream.Cache(directory=tmp_path), "phishing.v2", [1, 2])

    reopened = stream.Cache(directory=tmp_path)
    assert list(reopened["phishing.v2"]) == [1, 2]

    reopened.clear("phishing.v2")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("source", "error", "match"), KEYLESS_CASES.values(), ids=KEYLESS_CASES)
def test_keyless_call_raises(
    tmp_path: Path, source: typing.Any, error: type[Exception], match: str
) -> None:
    """A key is in practice mandatory: the only thing `__call__` guesses one from is a function,
    which is not iterable. `list` because both failures need iteration to start.
    """
    with pytest.raises(error, match=match):
        _ = list(stream.Cache(directory=tmp_path)(source))


def test_getitem_iterates_a_cached_key(tmp_path: Path) -> None:
    cache = stream.Cache(directory=tmp_path)
    fill(cache, "k", [1, 2])
    assert list(cache["k"]) == [1, 2]


@pytest.mark.parametrize("operation", UNKNOWN_KEY_CASES.values(), ids=UNKNOWN_KEY_CASES)
def test_unknown_key_raises(tmp_path: Path, operation: Callable[[stream.Cache], None]) -> None:
    with pytest.raises(FileNotFoundError):
        operation(stream.Cache(directory=tmp_path))


@pytest.mark.parametrize(("keys", "clear", "remaining"), CLEAR_CASES.values(), ids=CLEAR_CASES)
def test_clearing_removes_files_and_keys(
    tmp_path: Path, keys: list[str], clear: Callable[[stream.Cache], None], remaining: set[str]
) -> None:
    cache = stream.Cache(directory=tmp_path)
    for key in keys:
        fill(cache, key, [1])

    clear(cache)

    assert cache.keys == remaining
    assert {path.name for path in tmp_path.iterdir()} == {
        f"{key}{CACHE_SUFFIX}" for key in remaining
    }


def test_cleared_key_is_streamed_from_the_source(tmp_path: Path) -> None:
    cache = stream.Cache(directory=tmp_path)
    fill(cache, "k", [1, 2, 3])
    cache.clear("k")
    assert list(cache([4, 5, 6], key="k")) == [4, 5, 6]


@pytest.mark.parametrize(
    ("system", "expected"),
    [("Linux", "/tmp"), ("Darwin", "/tmp"), ("Windows", "C:\\TEMP")],
)
def test_default_directory_from_system(
    monkeypatch: pytest.MonkeyPatch, system: str, expected: str
) -> None:
    patch_system(monkeypatch, system)
    assert stream.Cache().directory == expected


def test_unknown_system_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_system(monkeypatch, "Plan9")
    with pytest.raises(ValueError, match="no default directory defined for Plan9"):
        _ = stream.Cache()


@pytest.mark.parametrize("as_str", [False, True], ids=["pathlike", "str"])
def test_directory_normalization_to_str(tmp_path: Path, as_str: bool) -> None:
    """`__repr__` joins `directory` with the key lines, which a `PathLike` would break."""
    cache = stream.Cache(directory=str(tmp_path) if as_str else tmp_path)
    assert cache.directory == str(tmp_path)

    fill(cache, "k", [1])
    assert repr(cache).splitlines()[0] == str(tmp_path)


def test_missing_directory_is_not_created(tmp_path: Path) -> None:
    cache = stream.Cache(directory=tmp_path / "nope")
    assert cache.keys == set()
    with pytest.raises(FileNotFoundError):
        _ = list(cache([1], key="k"))


@pytest.mark.parametrize("keys", [[], ["k"], ["a", "b"]], ids=["empty", "one-key", "two-keys"])
def test_repr(tmp_path: Path, keys: list[str]) -> None:
    cache = stream.Cache(directory=tmp_path)
    for key in keys:
        fill(cache, key, list(range(100)))

    directory, *lines = repr(cache).splitlines()

    assert directory == str(tmp_path)
    # `keys` is a set, so the lines come out in no particular order.
    assert sorted(lines) == sorted(
        f"{key} - {utils.pretty.humanize_bytes((tmp_path / f'{key}{CACHE_SUFFIX}').stat().st_size)}"
        for key in keys
    )


@pytest.mark.xfail(strict=True, reason="an abandoned first pass leaves a truncated cache file")
def test_abandoned_stream_does_not_poison_the_cache(tmp_path: Path) -> None:
    cache = stream.Cache(directory=tmp_path)
    rows = cache([1, 2, 3, 4, 5], key="k")
    assert [next(rows), next(rows)] == [1, 2]

    del rows
    _ = gc.collect()

    assert list(cache([1, 2, 3, 4, 5], key="k")) == [1, 2, 3, 4, 5]
