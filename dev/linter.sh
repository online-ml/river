#!/bin/bash

# This is a proxy PEP8 linter. It's used to lint some modules.
# Note that not all the project respects PEP8 style yet.
# This script will be removed once https://github.com/scikit-multiflow/scikit-multiflow/issues/219 resolved.
# Usage: Execute from the root project this script (./dev/linter.sh)

python3 -m flake8 \
  src/skmultiflow/anomaly_detection \
  src/skmultiflow/bayes \
  src/skmultiflow/core \
  src/skmultiflow/data \
  src/skmultiflow/drift_detection
