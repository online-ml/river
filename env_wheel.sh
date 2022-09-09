export CIBW_BUILD="cp38-* cp39-* cp310-* cp311-*"
export CIBW_BEFORE_BUILD="pip install cython numpy setuptools wheel setuptools-rust && rustup default nightly && rustup show"
export CIBW_BEFORE_BUILD_LINUX="pip install -U cython numpy setuptools wheel setuptools-rust && curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain=nightly --profile=minimal -y && rustup show"
export CIBW_ENVIRONMENT='PATH="$HOME/.cargo/bin:$PATH"'
export CIBW_MANYLINUX_X86_64_IMAGE='manylinux2014'
export CIBW_MANYLINUX_I686_IMAGE='manylinux2014'
export CIBW_BEFORE_ALL='curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain=nightly --profile=minimal -y && rustup show'

ARCHFLAGS='-arch arm64' CIBW_BUILD="cp38-* cp39-* cp310-* cp311-*" CIBW_ENVIRONMENT='PATH="$HOME/.cargo/bin:$PATH"'  CIBW_BEFORE_BUILD="pip install cython numpy setuptools wheel setuptools-rust &&  curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain=nightly --profile=minimal -y && rustup show"  CIBW_MANYLINUX_X86_64_IMAGE='manylinux2014' cibuildwheel --platform linux --archs x86_64 


ARCHFLAGS='-arch arm64' CIBW_BUILD="cp38-* cp39-* cp310-* cp311-*" CIBW_ENVIRONMENT='PATH="$HOME/.cargo/bin:$PATH"'  CIBW_BEFORE_BUILD="pip install cython numpy setuptools wheel setuptools-rust &&  curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain=nightly --profile=minimal -y && rustup show"  CIBW_MANYLINUX_X86_64_IMAGE='manylinux2014' cibuildwheel --platform linux --archs aarch64

CIBW_BUILD="cp38-* cp39-* cp310-* cp311-*" CIBW_ENVIRONMENT='PATH="$HOME/.cargo/bin:$PATH"'  CIBW_BEFORE_BUILD="pip install cython numpy setuptools wheel setuptools-rust &&  curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain=nightly --profile=minimal -y && rustup show"  CIBW_MANYLINUX_X86_64_IMAGE='manylinux2014' cibuildwheel --platform linux --archs aarch64
CIBW_BUILD="cp38-*" CIBW_ENVIRONMENT='PATH="$HOME/.cargo/bin:$PATH"' CIBW_SKIP: '*-musllinux_i686'   CIBW_BEFORE_BUILD="pip install cython numpy setuptools wheel setuptools-rust &&  rustup default nightly && rustup show" CIBW_BEFORE_BUILD_LINUX='pip install cython numpy setuptools wheel setuptools-rust &&  curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain=nightly --profile=minimal -y && rustup show'  CIBW_MANYLINUX_X86_64_IMAGE='manylinux2014' CIBW_MANYLINUX_X86_64_IMAGE='manylinux2014' CIBW_MUSLLINUX_X86_64_IMAGE='musllinux_1_1' CIBW_MANYLINUX_AARCH64_IMAGE='manylinux2014' CIBW_MUSLLINUX_AARCH64_IMAGE='musllinux_1_1' cibuildwheel --platform linux --archs aarch64

CIBW_ENVIRONMENT='PATH="$HOME/.cargo/bin:$PATH"'
CIBW_SKIP: '*-musllinux_i686'
CIBW_BEFORE_BUILD="pip install cython numpy setuptools wheel setuptools-rust &&  rustup default nightly && rustup show"
CIBW_BEFORE_BUILD_LINUX='pip install cython numpy setuptools wheel setuptools-rust &&  curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain=nightly --profile=minimal -y && rustup show'
CIBW_MANYLINUX_X86_64_IMAGE='manylinux2014'
CIBW_MANYLINUX_X86_64_IMAGE='manylinux2014' 
CIBW_MUSLLINUX_X86_64_IMAGE='musllinux_1_1' 
CIBW_MANYLINUX_AARCH64_IMAGE='manylinux2014' 
CIBW_MUSLLINUX_AARCH64_IMAGE='musllinux_1_1'