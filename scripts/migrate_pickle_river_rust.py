"""Migrate pickles produced by river <= 0.24.x to the new `river._river_rust.*` layout.

The native Rust extension moved from `river.stats._rust_stats` to `river._river_rust`
with semantic submodules. Pickled instances bake the old `__module__` strings, so they
no longer unpickle directly. This script rewrites those strings in-place.

Usage:
    python scripts/migrate_pickle_river_rust.py old.pkl new.pkl
    python scripts/migrate_pickle_river_rust.py old_pickles_dir/ new_pickles_dir/

The new river must be importable in the python env that runs the script (because the
rewriting unpickler instantiates classes from the new locations as it loads).
"""

from __future__ import annotations

import argparse
import io
import pickle
from pathlib import Path

# (old_module, class_name) -> new_module
REWRITES: dict[tuple[str, str], str] = {
    # Stats classes
    ("river.stats._rust_stats", "RsQuantile"): "river._river_rust.stats",
    ("river.stats._rust_stats", "RsEWMean"): "river._river_rust.stats",
    ("river.stats._rust_stats", "RsEWVar"): "river._river_rust.stats",
    ("river.stats._rust_stats", "RsIQR"): "river._river_rust.stats",
    ("river.stats._rust_stats", "RsKurtosis"): "river._river_rust.stats",
    ("river.stats._rust_stats", "RsPeakToPeak"): "river._river_rust.stats",
    ("river.stats._rust_stats", "RsSkew"): "river._river_rust.stats",
    ("river.stats._rust_stats", "RsRollingQuantile"): "river._river_rust.stats",
    ("river.stats._rust_stats", "RsRollingIQR"): "river._river_rust.stats",
    ("river.stats._rust_stats", "RsRollingROCAUC"): "river._river_rust.stats",
    ("river.stats._rust_stats", "RsRollingPRAUC"): "river._river_rust.stats",
    # Drift
    ("river.stats._rust_stats", "AdaptiveWindowing"): "river._river_rust.drift",
    # VectorDict: legacy pickles carry the (mis-)declared "river.utils.vectordict"
    ("river.utils.vectordict", "VectorDict"): "river._river_rust.vectordict",
}


class PickleMigrationError(RuntimeError):
    """Raised for migration failures that are not bad CLI input."""


class _RewritingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # type: ignore[override]
        new_module = REWRITES.get((module, name), module)
        return super().find_class(new_module, name)


def migrate_bytes(data: bytes) -> bytes:
    obj = _RewritingUnpickler(io.BytesIO(data)).load()
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def migrate_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(migrate_bytes(src.read_bytes()))
    print(f"migrated {src} -> {dst}")


def migrate_dir(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in src_dir.rglob("*.pkl"):
        dst = dst_dir / src.relative_to(src_dir)
        migrate_file(src, dst)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "src",
        type=Path,
        help="input pickle file, or directory of pickles to migrate recursively",
    )
    parser.add_argument(
        "dst",
        type=Path,
        help="output path; mirrors `src` (file -> file, directory -> directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.src.is_file():
        migrate_file(args.src, args.dst)
        return
    if args.src.is_dir():
        count = migrate_dir(args.src, args.dst)
        print(f"migrated {count} pickle(s) from {args.src} to {args.dst}")
        return
    raise PickleMigrationError(f"{args.src}: not a regular file or directory")


if __name__ == "__main__":
    main()
