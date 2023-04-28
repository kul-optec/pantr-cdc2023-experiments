import numpy as np
import casadi as cs
from typing import Tuple
from dataclasses import dataclass

from .ocp import OCProblem

@dataclass
class MSProblem:
    ocp: OCProblem
    state_input_var: cs.MX
    cost: cs.Function
    constr: cs.Function
    C: Tuple[np.ndarray, np.ndarray]
    D: Tuple[np.ndarray, np.ndarray]
    initial_guess: np.ndarray
    initial_state: np.ndarray
    initial_state_var: cs.MX

def ocp_to_ms(ocp: OCProblem) -> MSProblem:
    # Problem variables
    u_mat = cs.MX.sym("u", ocp.nu, ocp.N)
    x_mat = cs.MX.sym("x", ocp.nx, ocp.N)
    x0 = cs.MX.sym("x0", ocp.nx)
    x_mat = cs.horzcat(x0, x_mat)
    xu_mat = cs.vertcat(x_mat[:, :-1], u_mat)
    xN = x_mat[:, -1]

    # Input constraints
    inf_x = np.inf * np.ones((ocp.nx,))
    C_lb = np.tile(np.concatenate((ocp.input_constr_box[0], -inf_x)), ocp.N)
    C_ub = np.tile(np.concatenate((ocp.input_constr_box[1], +inf_x)), ocp.N)

    # Cost
    stage_cost = cs.sum2(ocp.stage_cost.map(ocp.N)(xu_mat))
    term_cost = ocp.term_cost(xN)
    ms_vars = cs.vec(cs.vertcat(u_mat, x_mat[:, 1:]))
    ms_cost = cs.Function('f_ms', [ms_vars, x0], [stage_cost + term_cost])

    # Constraints
    g = cs.vertcat(cs.vec(ocp.stage_constr.map(ocp.N)(x_mat[:, :-1])),
                   ocp.term_constr(xN))

    # Dynamics
    f_dyn = ocp.f_dynamics.map(ocp.N)
    g_dyn = cs.vec(x_mat[:, 1:] - f_dyn(x_mat[:, :-1], u_mat))
    ms_constr = cs.Function("g_ms", [ms_vars, x0], [cs.vertcat(g, g_dyn)])
    D_lb = np.concatenate((np.tile(ocp.stage_constr_box[0], ocp.N), ocp.term_constr_box[0], np.zeros((ocp.nx * ocp.N,))))
    D_ub = np.concatenate((np.tile(ocp.stage_constr_box[1], ocp.N), ocp.term_constr_box[1], np.zeros((ocp.nx * ocp.N,))))

    # Initial guess
    u0_mat = ocp.initial_guess.reshape((ocp.nu, ocp.N))
    x0_mat = ocp.f_dynamics.mapaccum(ocp.N)(ocp.initial_state, u0_mat)
    initial_guess = cs.vec(cs.vertcat(u0_mat, x0_mat)).full().ravel()

    return MSProblem(
        ocp=ocp,
        state_input_var=ms_vars,
        cost=ms_cost,
        constr=ms_constr,
        C=(C_lb, C_ub),
        D=(D_lb, D_ub),
        initial_guess=initial_guess,
        initial_state=ocp.initial_state.copy(),
        initial_state_var=x0,
    )

def ocp_to_ms_soft(ocp: OCProblem, μ: np.ndarray) -> MSProblem:
    # Problem variables
    u_mat = cs.MX.sym("u", ocp.nu, ocp.N)
    x_mat = cs.MX.sym("x", ocp.nx, ocp.N)
    x0 = cs.MX.sym("x0", ocp.nx)
    x_mat = cs.horzcat(x0, x_mat)
    xu_mat = cs.vertcat(x_mat[:, :-1], u_mat)
    xN = x_mat[:, -1]

    # Input constraints
    inf_x = np.inf * np.ones((ocp.nx,))
    C_lb = np.tile(np.concatenate((ocp.input_constr_box[0], -inf_x)), ocp.N)
    C_ub = np.tile(np.concatenate((ocp.input_constr_box[1], +inf_x)), ocp.N)

    # Cost
    stage_cost = cs.sum2(ocp.stage_cost.map(ocp.N)(xu_mat))
    term_cost = ocp.term_cost(xN)
    ms_vars = cs.vec(cs.vertcat(u_mat, x_mat[:, 1:]))

    # Constraints
    g = cs.vertcat(cs.vec(ocp.stage_constr.map(ocp.N)(x_mat[:, :-1])),
                   ocp.term_constr(xN))
    D_lb = np.concatenate((np.tile(ocp.stage_constr_box[0], ocp.N), ocp.term_constr_box[0]))
    D_ub = np.concatenate((np.tile(ocp.stage_constr_box[1], ocp.N), ocp.term_constr_box[1]))
    f2 = g - cs.fmax(D_lb, cs.fmin(g, D_ub))
    constr_cost = 0.5 * cs.sum1(μ * f2**2)
    ms_cost = cs.Function('f_ms', [ms_vars, x0], [stage_cost + term_cost + constr_cost])

    # Dynamics
    f_dyn = ocp.f_dynamics.map(ocp.N)
    g_dyn = cs.vec(x_mat[:, 1:] - f_dyn(x_mat[:, :-1], u_mat))
    ms_constr = cs.Function("g_ms", [ms_vars, x0], [g_dyn])
    D_lb = np.zeros((ocp.nx * ocp.N,))
    D_ub = np.zeros((ocp.nx * ocp.N,))

    # Initial guess
    u0_mat = ocp.initial_guess.reshape((ocp.nu, ocp.N))
    x0_mat = ocp.f_dynamics.mapaccum(ocp.N)(ocp.initial_state, u0_mat)
    initial_guess = cs.vec(cs.vertcat(u0_mat, x0_mat)).full().ravel()

    return MSProblem(
        ocp=ocp,
        state_input_var=ms_vars,
        cost=ms_cost,
        constr=ms_constr,
        C=(C_lb, C_ub),
        D=(D_lb, D_ub),
        initial_guess=initial_guess,
        initial_state=ocp.initial_state.copy(),
        initial_state_var=x0,
    )

def ms_solver_ipopt(ms: MSProblem, opts: dict):
    u = ms.state_input_var
    p = ms.initial_state_var
    nlp = {"x": u, "p": p, "f": ms.cost(u, p), "g": ms.constr(u, p)}
    bounds = {"lbx": ms.C[0], "ubx": ms.C[1], "lbg": ms.D[0], "ubg": ms.D[1]}
    guess = {"x0": ms.initial_guess}
    nlpsol = cs.nlpsol("ss_nlp", "ipopt", nlp, opts)
    return nlpsol, bounds, guess

def ms_extract_states_inputs(ms: MSProblem, sol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, nu, nx = ms.ocp.N, ms.ocp.nu, ms.ocp.nx
    nxu = nu + nx
    ux = np.reshape(sol, (nxu, N), order='F')
    return np.ravel(ux[nu:nxu, :], order='F'), np.ravel(ux[:nu, :], order='F')

__all__ = [
    'MSProblem',
    'ocp_to_ms',
    'ocp_to_ms_soft',
    'ms_solver_ipopt',
    'ms_extract_states_inputs',
]
