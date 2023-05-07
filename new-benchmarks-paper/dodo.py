#!/usr/bin/env doit

import os.path as osp

Δ = 1
TOL = 1e-8
OUTPUT = "output"
N = 50
NUM_SIM = 60

OPT_ALM = [
    "alm.max_iter=1000",
    "alm.max_time=30s",
    "alm.initial_penalty=1e4",
    "alm.penalty_update_factor=5",
    "alm.initial_tolerance=1e-1",
    "alm.tolerance_update_factor=0.1",
    "solver.stop_crit=ProjGradUnitNorm",
    "solver.max_iter=250",
    f"alm.tolerance={TOL}",
    f"alm.dual_tolerance={TOL}",
]

SOLVER_OPT = {
    "pantr": OPT_ALM + [
        "solver.radius_factor_rejected=0.35",
        "solver.radius_factor_acceptable=0.99",
        "solver.radius_factor_good=2.5",
        "solver.ratio_threshold_acceptable=0.2",
        "solver.ratio_threshold_good=0.8",
        "solver.max_iter=250",
        "dir.hessian_vec_factor=0",
    ],
    "panoc": OPT_ALM + [
        "accel.memory=50",
        "solver.max_iter=250",
    ],
    "strucpanoc": OPT_ALM + [
        "accel.memory=50",
        "dir.hessian_vec=0",
        "solver.max_iter=250",
    ],
    "ipopt": [
        f"ipopt.tol={TOL}",
        f"ipopt.constr_viol_tol={TOL}"
    ]
}

SOLVER_NAMES = {
    "pantr": "PANTR",
    "panoc": "PANOC$^+$ (50)",
    "strucpanoc": "Struc. PANOC$^+$ (50)",
    "ipopt": "Ipopt",
}


def task_quadcopter():
    problem = "quadcopter"
    for warm in (True, False):
        for horizon in range(Δ, N+1, Δ):
            for solver, opt in SOLVER_OPT.items():
                if solver == "ipopt" and warm:
                    opt = opt + ["solver.warm_start_init_point=yes"]
                formulation = "ss2" if solver == "ipopt" else "ss2p"
                outfile = problem_name(problem, warm, solver, horizon)
                cmd = [
                    "zsh",
                    "run-single-mpc-exp.sh",
                    OUTPUT,
                    outfile,
                    solver,
                    problem,
                    formulation,
                    str(NUM_SIM),
                    str(horizon),
                    "num_dist=0",
                    "warm=" + str(warm).lower(),
                ]
                yield {
                    "name": outfile,
                    "file_dep": ["run-mpc-exp.sh"],
                    "targets": [osp.join(OUTPUT, outfile + ".py")],
                    "actions": [cmd + opt],
                }


def problem_name(problem, warm, solver, horizon):
    warm_str = "warm" if warm else "cold"
    name = f"{warm_str}-{solver}"
    outfile = f"mpc-{problem}-{horizon},{NUM_SIM}-{name}"
    return outfile

def plot_name(problem, warm, solver):
    warm_str = "warm" if warm else "cold"
    name = f"{warm_str}-{solver}"
    outfile = f"mpc-{problem}-{NUM_SIM}-{name}"
    return outfile


def task_perfplot():
    problem = "quadcopter"
    for warm in (True, False):
        deps = []
        results = []
        for solver in SOLVER_OPT:
            for horizon in range(Δ, N+1, Δ):
                outfile = problem_name(problem, warm, solver, horizon)
                deps += [osp.join(OUTPUT, outfile + ".py")]
            outfile = problem_name(problem, warm, solver, horizon='{}')
            results += [f"{SOLVER_NAMES[solver]}:{outfile}"]
        name = plot_name(problem, warm, solver)
        cmd = [
            "python3",
            "perfplot.py",
            OUTPUT,
            name,
            str(Δ),
            str(N),
        ]
        yield {
            "name": name,
            "file_dep": deps + ["perfplot.py"],
            "targets": [osp.join(OUTPUT, name + "-med-runtimes.pdf")],
            "actions": [cmd + results],
        }


# def task_trajplot():
#     problem = "quadcopter"
#     deps = []
#     results = []
#     warm = True
#     warm_str = "warm" if warm else "cold"
#     solver = "pantr"
#         name = f"{warm_str}-{solver}"
#         outfile = f"mpc-Δ{Δ}-{problem}-{name}"
#         deps += [osp.join(OUTPUT, outfile + "_ids.txt")]
#         results += [outfile]
#     name = f"mpc-Δ{Δ}-{problem}-{warm_str}"
#     cmd = [
#         "python3",
#         "../benchmarks-paper/visualize_mpc_solution.py",
#         OUTPUT,
#         name,
#     ]
#     yield {
#         "name": name,
#         "file_dep": deps,
#         "targets": [osp.join(OUTPUT, name + "-solution.pdf")],
#         "actions": [cmd + results],
#     }
