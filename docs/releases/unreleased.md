# Unreleased

- Moved all the public modules imports from `river/__init__.py` to `river/api.py` and removed unnecessary dependencies between modules enabling faster cherry-picked import times (≈3x).

## base

- Introduced an `edit` method to the `base.Base` class. This allows setting attributes in a controlled manner, which paves the way for online AutoML. See [/recipes/cloning-and-editing-parameters] for more information.

## compat

- Moved the PyTorch wrappers to river-extra.

## compose

- Moved `utils.pure_inference_mode` to `compose.pure_inference_mode` and `utils.warm_up_mode` to `compose.warm_up_mode`.
- Pipeline parts can now be accessed by integer positions as well as by name.

## datasets

- Imports `synth`, enabling `from river import datasets; datasets.synth`.

## metrics

- Removed dependency to `optim`.

## utils

- Removed dependencies to `anomaly` and `compose`.
