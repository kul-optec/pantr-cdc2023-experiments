import numpy as np
import casadi as cs
from typing import Tuple
from dataclasses import dataclass

from .ocp import OCProblem

@dataclass
class SSProblem:
    ocp: OCProblem
    input_var: cs.SX
    cost: cs.Function
    constr: cs.Function
    C: Tuple[np.ndarray, np.ndarray]
    D: Tuple[np.ndarray, np.ndarray]
    initial_guess: np.ndarray
    initial_state: np.ndarray
    initial_state_var: cs.SX

def ocp_to_ss(ocp: OCProblem) -> SSProblem:
    # Problem variables
    u_mat = cs.SX.sym("u", ocp.nu, ocp.N)
    x0 = cs.SX.sym("x0", ocp.nx)
    x_mat = ocp.f_dynamics.mapaccum(ocp.N)(x0, u_mat)
    x_mat = cs.horzcat(x0, x_mat)
    xu_mat = cs.vertcat(x_mat[:, :-1], u_mat)
    xN = x_mat[:, -1]

    # Input constraints
    C_lb = np.tile(ocp.input_constr_box[0], ocp.N)
    C_ub = np.tile(ocp.input_constr_box[1], ocp.N)

    # Cost
    stage_cost = cs.sum2(ocp.stage_cost.map(ocp.N)(xu_mat))
    term_cost = ocp.term_cost(xN)
    u_vec = cs.vec(u_mat)
    ss_cost = cs.Function('f_ss', [u_vec, x0], [stage_cost + term_cost])

    g = cs.vertcat(cs.vec(ocp.stage_constr.map(ocp.N)(x_mat[:, :-1])),
                   ocp.term_constr(xN))
    ss_constr = cs.Function("g_ss", [u_vec, x0], [g])
    D_lb = np.concatenate((np.tile(ocp.stage_constr_box[0], ocp.N), ocp.term_constr_box[0]))
    D_ub = np.concatenate((np.tile(ocp.stage_constr_box[1], ocp.N), ocp.term_constr_box[1]))

    return SSProblem(
        ocp=ocp,
        input_var=u_vec,
        cost=ss_cost,
        constr=ss_constr,
        C=(C_lb, C_ub),
        D=(D_lb, D_ub),
        initial_guess=ocp.initial_guess.copy(),
        initial_state=ocp.initial_state.copy(),
        initial_state_var=x0,
    )


def ocp_to_ss_soft(ocp: OCProblem, μ: np.ndarray) -> SSProblem:
    ss_hard = ocp_to_ss(ocp)
    u_vec, x0 = ss_hard.input_var, ss_hard.initial_state_var
    g = ss_hard.constr(u_vec, x0)
    f2 = g - cs.fmax(ss_hard.D[0], cs.fmin(g, ss_hard.D[1]))
    constr_cost = 0.5 * cs.sum1(μ * f2**2)
    cost = cs.Function('f_ss_soft', [u_vec, x0], [ss_hard.cost(u_vec, x0) + constr_cost])

    return SSProblem(
        ocp=ocp,
        input_var=u_vec,
        cost=cost,
        constr=cs.Function('g_ss_soft', [u_vec, x0], [cs.vertcat()]),
        C=ss_hard.C,
        D=(np.zeros((0,)), np.zeros((0,))),
        initial_guess=ocp.initial_guess.copy(),
        initial_state=ocp.initial_state.copy(),
        initial_state_var=x0,
    )

def ss_solver_ipopt(ss: SSProblem, opts: dict):
    u = ss.input_var
    p = ss.initial_state_var
    nlp = {"x": u, "p": p, "f": ss.cost(u, p), "g": ss.constr(u, p)}
    bounds = {"lbx": ss.C[0], "ubx": ss.C[1], "lbg": ss.D[0], "ubg": ss.D[1]}
    guess = {"x0": ss.initial_guess}
    return cs.nlpsol("ss_nlp", "ipopt", nlp, opts), bounds, guess

__all__ = [
    'SSProblem',
    'ocp_to_ss',
    'ocp_to_ss_soft',
    'ss_solver_ipopt',
]
