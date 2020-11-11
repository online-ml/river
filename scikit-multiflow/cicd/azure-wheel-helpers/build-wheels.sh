#!/bin/bash

# Copyright (c) 2019, Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/azure-wheel-helpers for details.

# Based on https://github.com/pypa/python-manylinux-demo/blob/master/travis/build-wheels.sh
# with CC0 license here: https://github.com/pypa/python-manylinux-demo/blob/master/LICENSE

set -e -x
echo "$dev_requirements_file, $test_requirements_file"

# Collect the pythons
pys=(/opt/python/*/bin)

# Print list of Python's available
echo "All Pythons: ${pys[@]}"

# Filter out Python 2.7, 3.4 and 3.5
pys=(${pys[@]//*27*/})
pys=(${pys[@]//*34*/})
pys=(${pys[@]//*35*/})
# Filter out Python 3.9 temporarily
pys=(${pys[@]//*39*/})

# Print list of Python's being used
echo "Using Pythons: ${pys[@]}"

# Compile wheels
for PYBIN in "${pys[@]}"; do
    "${PYBIN}/pip" install -r /io/$dev_requirements_file
    "${PYBIN}/pip" wheel -v /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/$package_name-*.whl; do
    auditwheel repair --plat $PLAT "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in "${pys[@]}"; do
    "${PYBIN}/pip" install -r /io/$test_requirements_file
    "${PYBIN}/pip" install $package_name --no-index -f /io/wheelhouse
    if [ -d "/io/tests" ]; then
        "${PYBIN}/pytest" /io/tests
    else
        "${PYBIN}/pytest" --pyargs $package_name
    fi
done
