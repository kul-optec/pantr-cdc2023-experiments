import importlib
import alpaqa.casadi_generator as cg
from sys import argv
import numpy as np

formulations = {
    "ocp": "no",
    "ss2p": "psi_prod", "ss2": "full", "ss": "no",
    "ms2p": "psi_prod", "ms2": "full", "ms": "no",
}

if len(argv) < 4 or argv[2] not in formulations:
    print(f"Usage:    {argv[0]} <problem-name> ocp|ss|ms <horizon>")
    print("          ocp:  stage-wise optimal control formulation")
    print("          ss:   single-shooting formulation (first order)")
    print("          ss2:  single-shooting formulation (second order, full hessians)")
    print("          ss2p: single-shooting formulation (second order, hessian-vector products)")
    print("          ms:   multiple-shooting formulation (first order)")
    print("          ms2:  multiple-shooting formulation (second order, full hessians)")
    print("          ms2p: multiple-shooting formulation (second order, hessian-vector products)")
    exit(0)

name, formulation = argv[1:3]
N = int(argv[3])
module = importlib.import_module("alpaqa_mpc_benchmarks.problems." + name)
if N:
    name += "_" + formulation + "_" + str(N)
else:
    name += "_" + formulation
ocp = module.Problem(N=N if N > 0 else 1)
penalty_alm_split = ocp.nc * ocp.N + ocp.nc_N

if formulation == "ocp":
    cg.generate_casadi_control_problem(
        f=ocp.f_dynamics,
        l=ocp.stage_cost,
        l_N=ocp.term_cost,
        c=ocp.stage_constr,
        c_N=ocp.term_constr,
        name=name,
    ).generate()

    with open(f"{name}.tsv", "w") as f:
        np.savetxt(f, ocp.input_constr_box, delimiter="\t", newline="\n")
        np.savetxt(f, ocp.stage_constr_box, delimiter="\t", newline="\n")
        np.savetxt(f, ocp.term_constr_box, delimiter="\t", newline="\n")
        np.savetxt(f, [ocp.initial_state], delimiter="\t", newline="\n")
        np.savetxt(f, [ocp.initial_guess], delimiter="\t", newline="\n")
        np.savetxt(f, [penalty_alm_split], delimiter="\t", newline="\n")
elif formulation in ["ss", "ss2", "ss2p"]:
    from alpaqa_mpc_benchmarks.formulations.ss import ocp_to_ss

    ss = ocp_to_ss(ocp)
    cg.generate_casadi_problem(
        f=ss.cost,
        g=ss.constr,
        name=name,
        second_order=formulations[formulation],
    ).generate()
    cg.write_casadi_problem_data(
        name + ".so",
        C=ss.C,
        D=ss.D,
        param=ss.initial_state,
    )

    with open(f"{name}.guess.tsv", "w") as f:
        np.savetxt(f, [ss.initial_guess], delimiter="\t", newline="\n")
        np.savetxt(f, [penalty_alm_split], delimiter="\t", newline="\n")
elif formulation in ["ms", "ms2", "ms2p"]:
    from alpaqa_mpc_benchmarks.formulations.ms import ocp_to_ms

    ms = ocp_to_ms(ocp)
    cg.generate_casadi_problem(
        f=ms.cost,
        g=ms.constr,
        name=name,
        second_order=formulations[formulation],
    ).generate()
    cg.write_casadi_problem_data(
        name + ".so",
        C=ms.C,
        D=ms.D,
        param=ms.initial_state,
    )

    with open(f"{name}.guess.tsv", "w") as f:
        np.savetxt(f, [ms.initial_guess], delimiter="\t", newline="\n")
        np.savetxt(f, [penalty_alm_split], delimiter="\t", newline="\n")
else:
    assert False, "Invalid formulation"
