#!/usr/bin/env zsh
cd "$(dirname "$0")"
set -e

export LD_PRELOAD="$PWD/../toolchains/x86_64-centos7-linux-gnu/x-tools/x86_64-centos7-linux-gnu/x86_64-centos7-linux-gnu/lib64/libgfortran.so.5.0.0"
problem_dir="$PWD/../build/problems/Release"
build_dir="$PWD/../build/drivers/Release"

output="$1"; shift;
name="$1"; shift;
solver="$1"; shift;
problem="$1"; shift;
formulation="$1"; shift;
delta="$1"; shift;
run () { pushd "$output"; set -ex;
    for i in {$delta..10..$delta}; do
        /usr/bin/time -f 'max_memory: %M' "$build_dir/benchmark-mpc-driver" "$problem_dir/$problem" $i $formulation method="$solver" num_sim=30 "$@"
    done
    popd
}
mkdir -p "$output"
output_file="$output/mpc-Δ$delta-$problem-$name"
( run "$@" ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
grep "^max_memory: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_mem.txt"
echo "${output_file}_ids.txt"
