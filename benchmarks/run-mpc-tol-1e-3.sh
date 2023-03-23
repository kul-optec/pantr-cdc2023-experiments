#!/usr/bin/env zsh
cd "$(dirname "$0")"
set -e

export LD_PRELOAD="$PWD/../../toolchains/x86_64-centos7-linux-gnu/x-tools/x86_64-centos7-linux-gnu/x86_64-centos7-linux-gnu/lib64/libgfortran.so.5.0.0"

delta=1
tol=1e-5
problem=quadcopter

build_type=${1:-Release}
cd "../../build/examples/benchmarks/${build_type}"
cmake --build ../../.. --config ${build_type} -j --target benchmark-mpc-driver
run () { solver="$1"; shift; variant="$1"; shift; formulation="$1"; shift; set -ex;
    for i in {$delta..60..$delta}; do
        ./benchmark-mpc-driver ${variant}${problem} $i $formulation method="$solver" num_sim=60 "$@"
    done
}

# output_file="mpc-tol$tol-${problem}-ipopt-precompiled-cold-Δ$delta"
# ( run ipopt '' ss2 ipopt.tol=$tol ipopt.constr_viol_tol=$tol warm=0 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-ipopt-precompiled-warm-Δ$delta"
# ( run ipopt '' ss2 ipopt.tol=$tol ipopt.constr_viol_tol=$tol warm=1 solver.warm_start_init_point=yes) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-ipopt-precompiled-ms-Δ$delta"
# ( run ipopt '' ms2 ipopt.tol=$tol ipopt.constr_viol_tol=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-ipopt-precompiled-ms-warm-Δ$delta"
# ( run ipopt '' ms2 ipopt.tol=$tol ipopt.constr_viol_tol=$tol warm=1 solver.warm_start_init_point=yes ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-ipopt-cold-Δ$delta"
# ( run ipopt 'dl:' ss2 ipopt.tol=$tol ipopt.constr_viol_tol=$tol warm=0 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-ipopt-warm-Δ$delta"
# ( run ipopt 'dl:' ss2 ipopt.tol=$tol ipopt.constr_viol_tol=$tol warm=1 solver.warm_start_init_point=yes ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-newton-tr-cold-Δ$delta"
# ( run fbetrust '' ss2p dir.exact_newton=1 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol alm.max_time=30s warm=0) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-newton-tr-warm-Δ$delta"
# ( run fbetrust '' ss2p dir.exact_newton=1 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol alm.max_time=30s warm=1 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

output_file="mpc-tol$tol-${problem}-strucpanoc-mem50-cold-Δ$delta"
( run strucpanoc '' ss2p lbfgs.memory=50 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol alm.max_time=30s warm=0 ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
echo "${output_file}_ids.txt"

output_file="mpc-tol$tol-${problem}-strucpanoc-mem50-warm-Δ$delta"
( run strucpanoc '' ss2p lbfgs.memory=50 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol alm.max_time=30s warm=1 ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
echo "${output_file}_ids.txt"


# output_file="mpc-tol$tol-${problem}-lbfgsb-mem15-cold-Δ$delta"
# ( run lbfgsb '' ss2p alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol alm.max_time=30s solver.memory=15 warm=0 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-lbfgsb-mem15-warm-Δ$delta"
# ( run lbfgsb '' ss2p alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol alm.max_time=30s solver.memory=15 warm=1 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-lbfgsb-mem50-Δ$delta"
# ( run lbfgsb '' ss2p alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol solver.memory=50 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-lbfgspp-mem15-Δ$delta"
# ( run lbfgspp '' ss2p alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol solver.m=15 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-lbfgspp-mem50-Δ$delta"
# ( run lbfgspp '' ss2p alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol solver.m=50 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-strucpanoc-mem15-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=15 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-strucpanoc-mem30-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=30 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-strucpanoc-mem50-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=50 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-panoc-mem50-Δ$delta"
# ( run panoc '' ss2p lbfgs.memory=50 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"




# output_file="mpc-tol$tol-${problem}-strucpanoc-mem70-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=70 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-strucpanoc-mem90-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=90 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-strucpanoc-mem110-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=110 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-strucpanoc-mem130-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=130 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-strucpanoc-mem150-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=150 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-strucpanoc-mem170-Δ$delta"
# ( run strucpanoc '' ss2p lbfgs.memory=170 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-true-newton-tr-Δ$delta"
# ( run truefbetrust '' ss2p alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-newton-tr-0hessvec-Δ$delta"
# ( run fbetrust '' ss2p dir.exact_newton=1 dir.hessian_vec_factor=0 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-newton-tr-1hessvec-Δ$delta"
# ( run fbetrust '' ss2p dir.exact_newton=1 dir.hessian_vec_factor=1 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="mpc-tol$tol-${problem}-panoc-mem50-Δ$delta"
# ( run panoc '' ss2p lbfgs.memory=50 alm.Δ=5 alm.ρ=0.1 alm.ε_0=1e-1 alm.Σ_0=1e3 alm.ε=$tol alm.δ=$tol ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f2 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"