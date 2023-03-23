#!/usr/bin/env zsh
cd "$(dirname "$0")"
set -e

build_type=${1:-Release}
cd "../../build/examples/benchmarks/${build_type}"
cmake --build ../../.. --config ${build_type} --target benchmark-driver -j
run () { set -ex;
    for i in {20..40..2}; do
        ./benchmark-driver hermans_bicycle $i ss2 "$@"
    done
    # for i in {2..10..2}; do
    #     ./benchmark-driver hanging_chain $i ss2 "$@"
    # done
    for i in {20..40..2}; do
        ./benchmark-driver quadcopter $i ss2 "$@"
    done
}

# output_file="newtontr"
# ( run solver=fbetrust dir.exact_newton=true solver.c1=0.1 solver.c3=25 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f3 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

output_file="newtontr-compute_ratio_using_new_γ"
( run solver=fbetrust dir.exact_newton=true solver.c1=0.1 solver.c3=25 solver.compute_ratio_using_new_γ=true ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f3 | tee "${output_file}_ids.txt"
echo "${output_file}_ids.txt"

# output_file="lbfgstr-5"
# ( run solver=fbetrust dir.exact_newton=false solver.c1=0.35 solver.c3=2.5 lbfgs.memory=5 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f3 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

output_file="lbfgstr-compute_ratio_using_new_γ-5"
( run solver=fbetrust dir.exact_newton=false solver.c1=0.35 solver.c3=2.5 lbfgs.memory=5 solver.compute_ratio_using_new_γ=true ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f3 | tee "${output_file}_ids.txt"
echo "${output_file}_ids.txt"

# output_file="lbfgstr-15"
# ( run solver=fbetrust dir.exact_newton=false solver.c1=0.35 solver.c3=2.5 lbfgs.memory=15 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f3 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="lbfgstr-25"
# ( run solver=fbetrust dir.exact_newton=false solver.c1=0.35 solver.c3=2.5 lbfgs.memory=25 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f3 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

output_file="strucpanoc-5"
( run solver=strucpanoc lbfgs.memory=5 ) > "${output_file}.txt" 2>&1
grep "^results: " "${output_file}.txt" | cut -d' ' -f3 | tee "${output_file}_ids.txt"
echo "${output_file}_ids.txt"

# output_file="strucpanoc-15"
# ( run solver=strucpanoc lbfgs.memory=15 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f3 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="strucpanoc-25"
# ( run solver=strucpanoc lbfgs.memory=25 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f3 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="strucpanoc-50"
# ( run solver=strucpanoc lbfgs.memory=50 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f3 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"

# output_file="strucpanoc-150"
# ( run solver=strucpanoc lbfgs.memory=150 ) > "${output_file}.txt" 2>&1
# grep "^results: " "${output_file}.txt" | cut -d' ' -f3 | tee "${output_file}_ids.txt"
# echo "${output_file}_ids.txt"
