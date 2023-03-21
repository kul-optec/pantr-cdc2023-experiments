#!/usr/bin/env zsh
cd "$(dirname "$0")"
set -e

build_type=${1:-Release}
cd "../../build/examples/benchmarks/${build_type}"
cmake --build ../../.. --config ${build_type} -j --target benchmark-new-driver
run () { solver="$1"; shift; variant="$1"; shift; formulation="$1"; shift; set -ex;
    for i in {6..60..1}; do
        ./benchmark-new-driver ${variant}hanging_chain $i $formulation method="$solver" num_exp=4 "$@"
    done
}

output_file="ipopt"
( run ipopt 'dl:' ss2 ipopt.tol=1e-8 ipopt.constr_viol_tol=1e-8 ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
echo "${output_file}_ids.txt"

output_file="newton-tr"
( run fbetrust '' ss2p dir.exact_newton=1 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
echo "${output_file}_ids.txt"

output_file="strucpanoc"
( run strucpanoc '' ss2p lbfgs.memory=15 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
echo "${output_file}_ids.txt"
