"""Exercise migrate_pickle_river_rust.py against synthetic legacy pickles.

We hand-craft pickle byte streams that reference the legacy module paths by
producing a current-river pickle and then byte-replacing the new module
string with the legacy one; that avoids depending on an installed old river.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from scripts.migrate_pickle_river_rust import REWRITES, migrate_bytes, migrate_dir


def _make_legacy(obj: object, new_module: bytes, legacy_module: bytes) -> bytes:
    """Rewrite the GLOBAL opcode (`c<module>\\n<class>\\n`) that names the class.

    Pickled at protocol 2: the GLOBAL opcode is newline-terminated rather than
    length-prefixed, so byte-replacing the module name doesn't require
    patching any length fields (or the FRAME opcode that protocol 4+ adds).
    """
    raw = pickle.dumps(obj, protocol=2)
    needle = b"c" + new_module + b"\n"
    replacement = b"c" + legacy_module + b"\n"
    assert needle in raw, (
        f"expected GLOBAL({new_module!r}) in pickle of "
        f"{type(obj).__name__}; prefix was {raw[:120]!r}"
    )
    return raw.replace(needle, replacement)


def test_rewrite_table_covers_every_breaking_class() -> None:
    """The rewrite table must list every pyclass whose __module__ changed."""
    expected_old_pairs = {
        ("river.stats._rust_stats", cls)
        for cls in [
            "RsQuantile",
            "RsEWMean",
            "RsEWVar",
            "RsIQR",
            "RsKurtosis",
            "RsPeakToPeak",
            "RsSkew",
            "RsRollingQuantile",
            "RsRollingIQR",
            "RsRollingROCAUC",
            "RsRollingPRAUC",
            "AdaptiveWindowing",
        ]
    } | {("river.utils.vectordict", "VectorDict")}
    assert set(REWRITES.keys()) == expected_old_pairs


def test_rewrites_a_real_quantile_pickle() -> None:
    from river.stats import Quantile

    q = Quantile(0.5)
    for x in range(50):
        q.update(x)

    legacy = _make_legacy(q, b"river._river_rust.stats", b"river.stats._rust_stats")
    migrated = migrate_bytes(legacy)
    roundtrip = pickle.loads(migrated)
    assert roundtrip.get() == q.get()


def test_rewrites_an_adwin_pickle() -> None:
    from river.drift import ADWIN

    a = ADWIN()
    for x in range(50):
        a.update(float(x))

    legacy = _make_legacy(a, b"river._river_rust.drift", b"river.stats._rust_stats")
    migrated = migrate_bytes(legacy)
    # ADWIN must unpickle and behave like an ADWIN.
    revived = pickle.loads(migrated)
    revived.update(1.0)


def test_rewrites_a_vectordict_pickle() -> None:
    from river.utils import VectorDict

    v = VectorDict({"a": 1.0, "b": 2.0})

    legacy = _make_legacy(v, b"river._river_rust.vectordict", b"river.utils.vectordict")
    migrated = migrate_bytes(legacy)
    revived = pickle.loads(migrated)
    assert dict(revived) == dict(v)


def test_directory_mode(tmp_path: Path) -> None:
    from river.stats import Quantile

    src_dir = tmp_path / "in"
    dst_dir = tmp_path / "out"
    src_dir.mkdir()

    q = Quantile(0.5)
    for x in range(20):
        q.update(x)
    legacy = _make_legacy(q, b"river._river_rust.stats", b"river.stats._rust_stats")
    (src_dir / "q.pkl").write_bytes(legacy)
    (src_dir / "sub").mkdir()
    (src_dir / "sub" / "q2.pkl").write_bytes(legacy)

    count = migrate_dir(src_dir, dst_dir)
    assert count == 2
    assert (dst_dir / "q.pkl").is_file()
    assert (dst_dir / "sub" / "q2.pkl").is_file()
    revived = pickle.loads((dst_dir / "sub" / "q2.pkl").read_bytes())
    assert revived.get() == q.get()


def test_idempotent_on_already_migrated_pickle() -> None:
    from river.stats import Quantile

    q = Quantile(0.5)
    for x in range(10):
        q.update(x)
    new_bytes = pickle.dumps(q, protocol=pickle.HIGHEST_PROTOCOL)
    once = migrate_bytes(new_bytes)
    twice = migrate_bytes(once)
    # The unpickle-then-redump round-trip is byte-stable: re-running it
    # produces the same bytes (the rewrite table doesn't match anything in
    # an already-migrated pickle, so it's a pure re-serialisation).
    assert once == twice
