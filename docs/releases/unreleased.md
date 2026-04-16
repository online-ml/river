# Unreleased

## tree

- Fixed `MondrianNodeClassifier.replant` not copying the `counts` attribute when promoting a leaf to a branch, leaving the new branch with `n_samples != 0` but empty class counts. The fix mirrors the regressor's `_mean` copy and matches the reference [`onelearn`](https://github.com/onelearn/onelearn) implementation. Addresses [#1823](https://github.com/online-ml/river/issues/1823).
