`Version 0.5.2 - 2020-03-29 <https://pypi.org/project/creme/0.5.2/>`_
=====================================================================

:mod:`metrics`

- In `metrics.SMAPE`, the convention is now to use 0 when both ``y_true`` and ``y_pred`` are equal to 0, instead of raising a ``ZeroDivisionError``.

:mod:`multioutput`

- Fixed a bug where `multioutput.ClassifierChain` and `multioutput.RegressorChain` could not be pickled.
