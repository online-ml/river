# Unreleased

* Add `ppc64le` architecture to Linux wheel builds. (@ChidiebereNjoku)

## rules

- Fixed `RecursionError` in `AMRules` on long streams: `tree.splitter.EBSTSplitter` (and `TEBSTSplitter`) now traverses its binary search tree iteratively and the BST nodes carry a custom iterative `__deepcopy__`, so deeply-skewed trees no longer blow Python's recursion limit when rules are cloned during expansion. `tree.splitter.ExhaustiveSplitter` received the same treatment (iterative split-search, iterative node insertion, and iterative `__deepcopy__`).
- Fixed an `AMRules` memory leak where `HoeffdingRule.expand` appended a redundant `NumericLiteral` whenever a new split shared the feature and direction of an existing literal but did not tighten the threshold.

## stream

- Added `stream.iter_frame`, a dataframe-agnostic row iterator powered by [Narwhals](https://narwhals-dev.github.io/narwhals/) that works with any eager dataframe (pandas, polars, PyArrow, Modin, cuDF, ...).
- Deprecated `stream.iter_pandas` and `stream.iter_polars` in favour of `stream.iter_frame`. They now emit a `DeprecationWarning` and will be removed in a future release.