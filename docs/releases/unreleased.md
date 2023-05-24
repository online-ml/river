# Unreleased

## compose

- `compose.TransformerProduct` now correctly returns a `compose.TransformerUnion` when a transformer is added to it.
- Fixed `compose.TransformerProduct`'s `transform_many` behavior.
- `compose.TransformerUnion` and `compose.TransformerProduct` will now clone the provided estimators, so that shallow copies aren't shared in different places.

## multioutput

- `metrics.multioutput.MacroAverage` and `metrics.multioutput.MicroAverage` now loop over the keys of `y_true` instead of `y_pred`. This ensures a `KeyError` is correctly raised if `y_pred` is missing an output that is present in `y_true`.
