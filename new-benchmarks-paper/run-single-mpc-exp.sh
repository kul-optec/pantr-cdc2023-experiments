#!/usr/bin/env zsh
cd "$(dirname "$0")"
set -e

export LD_PRELOAD="$PWD/../toolchains/x86_64-centos7-linux-gnu/x-tools/x86_64-centos7-linux-gnu/x86_64-centos7-linux-gnu/lib64/libgfortran.so.5.0.0"
problem_dir="$PWD/../build/problems/Release"
# problem_dir="$PWD/../compiled-problems"
build_dir="$PWD/../build/drivers/Release"

output="$1"; shift;
name="$1"; shift;
solver="$1"; shift;
problem="$1"; shift;
formulation="$1"; shift;
num_sim="$1"; shift;
horizon="$1"; shift;

run () { pushd "$output"; set -ex;
    /usr/bin/time -f 'max_memory: %M' "$build_dir/benchmark-mpc-driver" "$problem_dir/$problem" $horizon $formulation method="$solver" num_sim=$num_sim results_name="$name" "$@"
    popd
}
mkdir -p "$output"
output_file="$output/$name"
( run "$@" ) > "${output_file}.txt" 2>&1
grep "^max_memory: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_mem.txt"
