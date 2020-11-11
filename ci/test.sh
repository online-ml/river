#!/usr/bin/env bash

# Exit when any command fails
set -e

# Download Miniconda if necessary
mkdir -p downloads
if [ ! -f downloads/miniconda.sh ]
then
  if [ $TRAVIS_OS_NAME = "osx" ]
  then
    fname=Miniconda3-latest-MacOSX-x86_64.sh
  else
    fname=Miniconda3-latest-Linux-x86_64.sh
  fi
  wget https://repo.continuum.io/miniconda/$fname -q -O downloads/miniconda.sh
fi

# Install and update Miniconda
MINICONDA_PATH=$HOME/miniconda
chmod +x downloads/miniconda.sh && ./downloads/miniconda.sh -b -p $MINICONDA_PATH
export PATH=$MINICONDA_PATH/bin:$PATH
conda update --yes conda

# Create and activate a test environment
PYTHON_VERSION=$1
conda create --yes --name testenv python=$PYTHON_VERSION
source activate testenv

# Install the development dependencies
if [ "$TRAVIS_TAG" = "" ]
then
  echo "Installing dev dependencies"
  pip install -e ".[dev]" codecov
else
  echo "Installing dev and compat dependencies"
  pip install -e ".[compat,dev]" codecov
fi

echo "Downloading the datasets that are used for testing"
python -c "from river import datasets; datasets.CreditCard().download()"

echo "Running flake8"
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude scikit-multiflow
#mypy river

echo "Running pytest"
pytest --cov=river -m "not datasets"

echo "Running codecov"
codecov
