#!/usr/bin/env zsh
cd "$(dirname "$0")"
set -e

export LD_PRELOAD="$PWD/../../toolchains/x86_64-centos7-linux-gnu/x-tools/x86_64-centos7-linux-gnu/x86_64-centos7-linux-gnu/lib64/libgfortran.so.5.0.0"

delta=1

build_type=${1:-Release}
cd "../../build/examples/benchmarks/${build_type}"
cmake --build ../../.. --config ${build_type} -j --target benchmark-mpc-driver
run () { solver="$1"; shift; variant="$1"; shift; formulation="$1"; shift; set -ex;
    for i in {1..60..$delta}; do
        ./benchmark-mpc-driver ${variant}quadcopter $i $formulation method="$solver" num_sim=60 "$@"
    done
}

# output_file="mpc-quadcopter-ipopt-precompiled-Δ$delta"
# ( run ipopt '' ss2 ipopt.tol=1e-8 ipopt.constr_viol_tol=1e-8 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-ipopt-Δ$delta"
# ( run ipopt 'dl:' ss2 ipopt.tol=1e-8 ipopt.constr_viol_tol=1e-8 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-lbfgs-tr-mem15-Δ$delta"
# ( run fbetrust '' ss2p dir.exact_newton=0 lbfgs.memory=15 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-newton-tr-Δ$delta"
# ( run fbetrust '' ss2p dir.exact_newton=1 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-strucpanoc-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=15 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-strucpanoc-mem30-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=30 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-strucpanoc-mem50-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=50 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-strucpanoc-mem70-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=70 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-strucpanoc-mem90-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=90 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-strucpanoc-mem110-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=110 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-strucpanoc-mem130-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=130 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-strucpanoc-mem150-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=150 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-strucpanoc-mem170-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=170 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-true-newton-tr-Δ$delta"
# ( run truefbetrust '' ss2p alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-newton-tr-0hessvec-Δ$delta"
# ( run fbetrust '' ss2p dir.exact_newton=1 dir.hessian_vec_factor=0 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-newton-tr-1hessvec-Δ$delta"
# ( run fbetrust '' ss2p dir.exact_newton=1 dir.hessian_vec_factor=1 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-panoc-mem50-Δ$delta"
# ( run panoc '' ss2p lbfgs.memory=50 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-panoc-mem150-Δ$delta"
# ( run panoc '' ss2p lbfgs.memory=150 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

output_file="mpc-quadcopter-lbfgsb-mem50-Δ$delta"
( run lbfgsb '' ss2p solver.m=50 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 solver.past=0 ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
echo "${output_file}_ids.txt"

# output_file="mpc-quadcopter-lbfgsb-mem15-Δ$delta"
# ( run lbfgsb '' ss2p solver.m=15 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 solver.past=0 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"
