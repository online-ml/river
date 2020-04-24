# Unreleased

## metrics

- In `metrics.SMAPE`, the convention is now to use 0 when both `y_true` and `y_pred` are equal to 0, instead of raising a `ZeroDivisionError`.

## multioutput

- Fixed a bug where `multioutput.ClassifierChain` and `multioutput.RegressorChain` could not be pickled.

## stream

- Added a `iter_sql` utility method to work with SQLAlchemy.
