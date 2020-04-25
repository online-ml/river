# Unreleased

## compose

- Added `compose.SelectType`, which allows selecting feature subsets based on their type.

## metrics

- In `metrics.SMAPE`, the convention is now to use 0 when both `y_true` and `y_pred` are equal to 0, instead of raising a `ZeroDivisionError`.

## multioutput

- Fixed a bug where `multioutput.ClassifierChain` and `multioutput.RegressorChain` could not be pickled.

## stats

- Added `stats.Shift`, which can be used to compute statistics over a shifted version of a variable.
- Added `stats.Link`, which can be used to compose univariate statistics. Univariate statistics can now be composed via the `|` operator.

## stream

- Added a `stream.iter_sql` utility method to work with SQLAlchemy.
