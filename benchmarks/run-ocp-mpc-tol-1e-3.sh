#!/usr/bin/env zsh
cd "$(dirname "$0")"
set -e

export LD_PRELOAD="$PWD/../../toolchains/x86_64-centos7-linux-gnu/x-tools/x86_64-centos7-linux-gnu/x86_64-centos7-linux-gnu/lib64/libgfortran.so.5.0.0"

delta=5
tol=1e-5
problem="${1:-hanging_chain}"

build_type=${2:-Release}
cd "../../build/examples/benchmarks/${build_type}"
cmake --build ../../.. --config ${build_type} -j --target benchmark-ocp-mpc-driver
run () { solver="$1"; shift; variant="$1"; shift; set -ex;
    for i in {$delta..60..$delta}; do
        ./benchmark-ocp-mpc-driver ${variant}${problem} $i method="$solver" num_sim=60 "$@"
    done
}

output_file="mpc-tol$tol-${problem}-gauss-newton-lqr-cold-Δ$delta"
( run panococp '' solver.gn_interval=15 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol warm=0 ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
echo "${output_file}_ids.txt"

output_file="mpc-tol$tol-${problem}-gauss-newton-lqr-warm-Δ$delta"
( run panococp '' solver.gn_interval=15 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol warm=1 ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
echo "${output_file}_ids.txt"
