#!/usr/bin/env bash

# Install and update Miniconda
if [ $TRAVIS_OS_NAME = "osx" ]
  then
    fname=Miniconda3-latest-MacOSX-x86_64.sh
  else
    fname=Miniconda3-latest-Linux-x86_64.sh
  fi
wget https://repo.continuum.io/miniconda/$fname \
    -O miniconda.sh
MINICONDA_PATH=$HOME/miniconda
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH=$MINICONDA_PATH/bin:$PATH
conda update --yes conda

# Create and activate a test environment
PYTHON_VERSION=$1
conda create --yes --name testenv python=PYTHON_VERSION
source activate testenv

# Install dependencies required for testing
python setup.py develop
pip install -e ".[dev]"
pip install codecov

# Run tests and coverage
pytest --cov=creme
codecov
