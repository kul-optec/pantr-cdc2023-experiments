#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"/..

set -ex

# Paths to compiler and dependencies
tools_dir="$PWD/toolchains"
triple="x86_64-centos7-linux-gnu"
pfx="$tools_dir/$triple"
pkg_path="$pfx/eigen-master;$pfx/casadi;$pfx/openblas;$pfx/mumps;$pfx/ipopt;$pfx/pybind11"

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

# Activate Python virtual environment
. ./.venv/bin/activate

# Configure and build benchmark problems and benchmark driver
cmake -S. -Bbuild \
    --toolchain "$pfx/cmake/$triple.toolchain.cmake" \
    -DCMAKE_FIND_ROOT_PATH="$pkg_path" \
    -DBUILD_SHARED_LIBS=Off \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DPython3_ROOT_DIR="$PWD/.venv" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
    -G "Ninja Multi-Config"
# Build
for cfg in Release; do
    cmake --build build -j --config $cfg
done
