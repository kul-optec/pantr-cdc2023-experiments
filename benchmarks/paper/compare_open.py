import casadi as cs
import numpy as np
import numpy.linalg as la

from alpaqa_mpc_benchmarks.problems.quadcopter import Problem
from alpaqa_mpc_benchmarks.formulations.ss import ocp_to_ss

ocp = Problem(N=20)
ss = ocp_to_ss(ocp)

ss_cost = ss.cost(ss.input_var, ss.initial_state_var)
ss_constr = ss.constr(ss.input_var, ss.initial_state_var)
y_sym = cs.SX.sym("y", ss.constr.size1_out(0))
ss_grad_lagr = cs.Function(
    "grad_f",
    [ss.input_var, y_sym, ss.initial_state_var],
    [cs.gradient(ss_cost, ss.input_var) + cs.jtimes(ss_constr, ss.input_var, y_sym, True)],
)

# KKT error
def compute_kkt_error(ss, x, y):
    # Gradient of Lagrangian, ∇ℒ(x,y) = ∇f(x) + ∇g(x) y
    grad_Lx = ss_grad_lagr(x, y, ss.initial_state)
    # Eliminate normal cone of bound constraints, z = Π(x - ∇ℒ(x,y)) - x
    z = np.fmax(ss.C[0], np.fmin(x - grad_Lx, ss.C[1])) - x
    # Stationarity, ‖Π(x - ∇ℒ(x,y)) - x‖
    stationarity = la.norm(z, ord=np.inf)
    # Constraints, g(x)
    g = ss.constr(x, ss.initial_state)
    # Distance to feasible set, v = g(x) - Π(g(x))
    v = g - np.fmax(ss.D[0], np.fmin(g, ss.D[1]))
    # Constraint violation, ‖g(x) - Π(g(x))‖
    constr_violation = la.norm(v, ord=np.inf)
    # Complementary slackness
    complementarity = la.norm(y * v, ord=np.inf)

    return stationarity, constr_violation, complementarity

# %% OpEn

import opengen as og

og_box_C = og.constraints.Rectangle(xmin=list(ss.C[0]), xmax=list(ss.C[1]))
og_box_D = og.constraints.Rectangle(xmin=list(ss.D[0]), xmax=list(ss.D[1]))
og_problem = (
    og.builder.Problem(
        ss.input_var,
        ss.initial_state_var,
        ss_cost,
    )
    .with_constraints(og_box_C)
    .with_aug_lagrangian_constraints(
        ss_constr,
        og_box_D,
    )
)

meta = og.config.OptimizerMeta().with_optimizer_name("the_optimizer")

build_config = (
    og.config.BuildConfiguration()
    .with_build_directory("python_build")
    .with_build_mode(og.config.BuildConfiguration.RELEASE_MODE)
    .with_tcp_interface_config()
)

tol=1e-7
solver_config = (
    og.config.SolverConfiguration()
    .with_lbfgs_memory(50)
    .with_delta_tolerance(tol)
    .with_tolerance(tol)
    .with_initial_tolerance(1e-1)
    .with_initial_penalty(1e4)
    .with_max_inner_iterations(250 * 2000)
    .with_max_outer_iterations(200)
    .with_penalty_weight_update_factor(5)
    .with_inner_tolerance_update_factor(0.1)
    .with_max_duration_micros(30_000_000)
    .with_preconditioning(False)
)

builder = og.builder.OpEnOptimizerBuilder(
    og_problem,
    metadata=meta,
    build_configuration=build_config,
    solver_configuration=solver_config,
)
builder.build()

mng = og.tcp.OptimizerTcpManager("python_build/the_optimizer")
mng.start()

pong = mng.ping()  # check if the server is alive
print(pong)
response = mng.call(ss.initial_state, ss.initial_guess)  # call the solver over TCP


if response.is_ok():
    # Solver returned a solution
    solution_data = response.get()
    u_star = solution_data.solution
    y_star = solution_data.lagrange_multipliers
    exit_status = solution_data.exit_status
    solver_time = solution_data.solve_time_ms
    print(f"OpEn success ({exit_status}) in {solver_time*1e-3} s.")
    stationarity, constr_violation, complementarity = compute_kkt_error(ss, u_star, y_star)
    print(
        f"Cost: {solution_data.cost}, Infeasibility: {solution_data.f1_infeasibility}, Penalty: {solution_data.penalty}\n"
        f"Stationarity: {stationarity}, Constr. Violation: {constr_violation}, Complementarity: {complementarity}\n"
        f"Inner iter {solution_data.num_inner_iterations}, Outer iter {solution_data.num_outer_iterations}"
    )
    print("solution:", np.array(u_star), sep='\n')
    print("multipliers:", np.array(y_star), sep='\n')
    # print(dir(solution_data))
else:
    # Invocation failed - an error report is returned
    solver_error = response.get()
    print(f"({solver_error.code}) {solver_error.message}")


mng.kill()
