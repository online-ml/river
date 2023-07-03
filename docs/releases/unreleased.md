# Unreleased

Calling `learn_one` in a pipeline will now update each part of the pipeline in turn. Before the unsupervised parts of the pipeline were updated during `predict_one`. This is more intuitive for new users. The old behavior, which yields better results, can be restored by calling `learn_one` with the new `compose.pure_inference_mode` context manager.

## compose

- Removed the `compose.warm_up_mode` context manager.
- Removed the `compose.pure_inference_mode` context manager.

## forest

- Fixed issue with `forest.ARFClassifier` which couldn't be passed a `CrossEntropy` metric.
