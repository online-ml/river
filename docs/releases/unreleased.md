# Unreleased

## stream

- Enabled strict typing for `stream.iter_array`. It is now overloaded on its inputs: an `X` of texts is typed to yield `str` rows, and a plain sequence `y` propagates its own target type, while numpy arrays keep yielding `Any`. Typing-only, apart from the fixes below.
- Fixed `stream.iter_array` for plain Python inputs, which are documented as supported: `shuffle=True` no longer raises a `TypeError` on lists, and a multi-output `y` given as a list of lists no longer raises an `AttributeError`.
- `stream.iter_array` now yields an empty stream when `X` is empty, instead of raising an `IndexError`, and comes out slightly faster on every input shape.
- Added dedicated tests for `stream.iter_array`, covering every supported input container (numpy, list, tuple), shuffling, multi-output targets, text arrays, and the static types its overloads advertise.
