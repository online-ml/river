# Unreleased

## compose

- Fixed some bugs related to mini-batching in `compose.Pipeline`.

## multiclass

- `multiclass.OneVsRestClassifier` now supports mini-batching.

## optim

- Removed `optim.MiniBatcher`.
- Implemented `optim.Averager`, which allows doing averaged stochastic gradient descent.
