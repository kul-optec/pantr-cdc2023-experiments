#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"/..

set -ex

# Download compiler and dependencies
download_url="https://github.com/tttapa/cross-python/releases/download/0.0.16"
tools_dir="$PWD/toolchains"
triple="x86_64-centos7-linux-gnu"
pfx="$tools_dir/$triple"
mkdir -p "$tools_dir"
if [ ! -d "$pfx" ]; then
    wget "$download_url/full-$triple.tar.xz" -O- | tar xJ -C "$tools_dir"
fi
pkg_path="$pfx/eigen-master;$pfx/casadi;$pfx/ipopt;$pfx/pybind11"
pfx_path="$pfx/ipopt/usr/local"

# Use ccache to cache compilation
if { which ccache > /dev/null; }; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
fi
# Compile for this system's processor for optimal performance
export CFLAGS="-march=native"
export CXXFLAGS="-march=native"
export FCFLAGS="-march=native"
export LDFLAGS="-static-libstdc++"

# Create Python virtual environment
[ -d .venv ] || python3.11 -m venv .venv
. ./.venv/bin/activate
pip install -U pip

# Download alpaqa
git submodule update --init

# Configure, build and install alpaqa
pushd alpaqa
cmake -S. -Bbuild \
    --toolchain "$pfx/cmake/$triple.toolchain.cmake" \
    -DCMAKE_FIND_ROOT_PATH="$pkg_path" \
    -DCMAKE_PREFIX_PATH="$pfx_path" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=On \
    -DBUILD_SHARED_LIBS=Off \
    -DALPAQA_WITH_DRIVERS=On \
    -DALPAQA_WITH_EXAMPLES=Off \
    -DALPAQA_WITH_TESTS=Off \
    -DALPAQA_WITH_PYTHON=Off \
    -DALPAQA_WITH_OCP=Off \
    -DCMAKE_STAGING_PREFIX="$PWD/staging" \
    -G "Ninja Multi-Config"
# Build
for cfg in Release; do
    cmake --build build -j --config $cfg
    cmake --install build --config $cfg
    cmake --install build --config $cfg --component debug
done
# Build alpaqa Python package
config="$triple.py-build-cmake.config.toml"
cat <<- EOF > "$config"
[cmake]
config = ["Release"]
generator = "Ninja Multi-Config"
[cmake.options]
CMAKE_FIND_ROOT_PATH = "$pkg_path"
CMAKE_PREFIX_PATH = "$pfx_path"
USE_GLOBAL_PYBIND11 = "On"
ALPAQA_WITH_OCP = "Off"
EOF
LDFLAGS='-static-libgcc -static-libstdc++' \
pip install '.[test]' -v \
    --config-settings=--cross="$pfx/cmake/$triple.py-build-cmake.cross.toml" \
    --config-settings=--local="$PWD/$config"
pytest
popd

# Install benchmark Python package
pushd python
pip install '.[test]'
pytest
popd

# Install benchmark requirements
pip install -r benchmarks-paper/requirements.txt
