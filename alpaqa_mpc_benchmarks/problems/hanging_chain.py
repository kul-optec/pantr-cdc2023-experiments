from ..formulations.ocp import OCProblem
import numpy as np
import casadi as cs
from dataclasses import dataclass


@dataclass
class Params:
    N_balls: int = 9  #           Number of balls
    n_dim: int = 3  #             Number of spatial dimensions
    α: float = 25
    β: float = 1
    γ: float = 0.01
    m: float = 0.03  #            mass
    D: float = 0.1  #             spring constant
    L: float = 0.033  #           spring length
    v_max: float = 1  #           maximum actuator velocity
    g_grav: float = 9.81  #       Gravitational acceleration       [m/s²]


class Problem(OCProblem):
    Params = Params

    def __init__(self, Ts: float = 0.1, N: int = 30, params: Params = None):
        self.Ts = Ts
        self.N = N
        self.params = params or Params()
        params = self.params

        d, Nb = params.n_dim, params.N_balls
        # State and input vectors
        x = cs.SX.sym("x", d, (Nb + 1))  # state: balls 1→N+1 positions
        v = cs.SX.sym("v", d, Nb)  # state: balls 1→N velocities
        self.input_var = input = cs.SX.sym("u", d)  # input: ball 1+N velocity
        self.state_var = state = cs.vertcat(cs.vec(x), cs.vec(v))
        self.state_names = ["x"] * (d * (Nb + 1)) + ["v"] * (d * Nb)
        self.input_names = ["u"] * d
        nu = np.product(input.shape)  # Number of inputs

        # Parameters
        m = params.m  # mass
        D = params.D  # spring constant
        L = params.L  # spring length
        g = params.g_grav * np.array([0, 0, -1] if d == 3 else [0, -1])  # gravity
        x_end = np.eye(1, d, 0).ravel()  # ball N+1 reference position

        # Continuous-time dynamics y' = f(y, u; p)
        f1 = [cs.vec(v), input]
        dist_vect = cs.horzcat(x[:, 0], x[:, 1:] - x[:, :-1])
        dist_norm = cs.sqrt(cs.sum1(dist_vect * dist_vect))

        F = dist_vect @ cs.diag(D * (1 - L / dist_norm).T)
        fs = cs.horzcat(F[:, 1:] - F[:, :-1]) / m + cs.repmat(g, (1, Nb))

        f_c_expr = cs.vertcat(*f1, cs.vec(fs))
        f_c = cs.Function("f", [state, input], [f_c_expr])

        # Runge-Kutta integrator
        k1 = f_c(state, input)
        k2 = f_c(state + Ts * k1 / 2, input)
        k3 = f_c(state + Ts * k2 / 2, input)
        k4 = f_c(state + Ts * k3, input)

        # Discrete-time dynamics
        f_d_expr = state + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.f_dynamics = f_d = cs.Function("f", [state, input], [f_d_expr])

        # MPC cost
        xt = cs.SX.sym("xt", d * (Nb + 1), 1)
        vt = cs.SX.sym("vt", d * Nb, 1)
        ut = cs.SX.sym("ut", d, 1)
        yt = cs.vertcat(xt, vt)

        L_cost_x = params.α * cs.sumsqr(xt[-d:] - x_end)
        for i in range(Nb):
            xdi = vt[d * i : d * i + d]
            L_cost_x += params.β * cs.sumsqr(xdi)
        L_cost_u = params.γ * cs.sumsqr(ut)
        self.stage_cost = cs.Function("l", [cs.vertcat(yt, ut)], [L_cost_x + L_cost_u])
        self.term_cost = cs.Function("l_N", [yt], [L_cost_x])

        # Box constraints on the actuator velocity:
        v_max = params.v_max * np.ones((nu,))
        self.input_constr_box = -v_max, +v_max

        # No general constraints
        self.stage_constr = cs.Function("c", [yt], [cs.vertcat()])
        self.stage_constr_box = np.zeros((0,)), np.zeros((0,))
        self.term_constr = cs.Function("c", [yt], [cs.vertcat()])
        self.term_constr_box = np.zeros((0,)), np.zeros((0,))

        # Initial state
        x_0 = np.zeros((d * (Nb + 1),))
        x_0[0::d] = np.arange(1, Nb + 2) / (Nb + 1)
        v_0 = np.zeros((d * Nb,))
        self.initial_state = np.concatenate((x_0, v_0))

        self.initial_guess = np.zeros((N * nu,))
