# Unreleased

## base

- Introduce `base.MiniBatchTransformer`. Add support for mini-batches to `compose.TransformerUnion`, `compose.Select`, and `preprocessing.OneHotEncoder`.

## checks

- Created this module to store estimator unit testing, rather than having it in the `utils` module.

## compose

- Split `compose.Renamer` into `compose.Prefixer` and `compose.Suffixer` that respectively prepend and append a string to the features' name.
- Changed `compose.Renamer` to allow feature renaming following a mapping.

## evaluate

- Refactored `evaluate.progressive_validation` to work with `base.AnomalyDetector`s.

## facto

- Added `debug_one` method to `BaseFM`.

## feature_extraction

- Make the `by` parameter in `feature_extraction.Agg` and `feature_extraction.TargetAgg` to be optional, allowing to calculate aggregates over the whole data.
- Removed `feature_extraction.Lagger` and `feature_extraction.TargetLagger`. Their functionality can be reproduced by combining `feature_extraction.Agg` and `stats.Shift`.
- `feature_extraction.Agg` and `feature_extraction.Target` now have a `state` property. It returns a `pandas.Series` representing the current aggregates values within each group.

## metrics

- `metrics.ROCAUC` works with `base.AnomalyDetectors`s.

## reco

- Renamed the `Recommender` base class into `Ranker`.
- Added a `rank` method to each recommender.
- Removed `reco.SurpriseWrapper` as it wasn't really useful.
- Added an `is_contextual` property to each ranker to indicate if a model makes use of contextual features or not.

## special

- Created this module to store some stuff that was in the `utils` module.

## stats

- `stats.Mean`, `stats.Var`, and `stats.Cov` each now have an `update_many` method which accepts numpy arrays.

## utils

- Removed `utils.Window` and use `collections.deque` instead where necessary.
