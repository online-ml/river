# Unreleased

## compose

- Fixed some bugs related to mini-batching in `compose.Pipeline`.

## datasets

- Added `datasets.SolarFlare`, which is a small multi-output regression dataset.

## decomposition

- `decomposition.LDA` now takes as input word counts instead of raw text.

## linear_model

- Added `linear_model.Perceptron`, which is implemented as a special case of logistic regression.

## multiclass

- `multiclass.OneVsRestClassifier` now supports mini-batching.

## optim

- Removed `optim.MiniBatcher`.
- Implemented `optim.Averager`, which allows doing averaged stochastic gradient descent.
- Removed `optim.Perceptron`.
