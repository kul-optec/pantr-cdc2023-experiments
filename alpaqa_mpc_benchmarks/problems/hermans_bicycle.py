from ..formulations.ocp import OCProblem
import numpy as np
import casadi as cs
from math import prod

class Problem(OCProblem):
    def __init__(self, Ts: float = 0.050, N: int = 30):
        self.Ts = Ts
        self.N = N

        # Parameters
        lr = 1.17
        lf = 1.77

        # Dynamics
        p = cs.SX.sym("p", 2)
        v = cs.SX.sym("v", 1)
        θ = cs.SX.sym("theta", 1)
        a = cs.SX.sym("a", 1)
        δ = cs.SX.sym("delta", 1)
        self.state_var = state = cs.vertcat(p, v, θ)
        self.input_var = input = cs.vertcat(a, δ)
        self.state_names = [r"$p_x$", r"$p_y$", r"$v$", r"$\theta$"]
        self.input_names = [r"$a$", r"$\delta$"]

        β = cs.atan(lr / (lr + lf) * cs.tan(δ))
        dpx = v * cs.cos(θ + β)
        dpy = v * cs.sin(θ + β)
        dv = a
        dθ = v / lr * cs.sin(β)

        self.f_c = cs.Function("f_c", [state, input],
            [cs.vertcat(dpx, dpy, dv, dθ)])

        # Discretization
        k1 = self.f_c(state, input)
        k2 = self.f_c(state + Ts * k1 / 2, input)
        k3 = self.f_c(state + Ts * k2 / 2, input)
        k4 = self.f_c(state + Ts * k3, input)

        f_d_expr = state + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.f_dynamics = cs.Function("f", [state, input], [f_d_expr])

        # Input constraints
        self.input_constr_box = (
            np.array([-10, -np.pi / 4]),
            np.array([+10, +np.pi / 4]),
        )

        # State constraints
        O1 = [
            p[0],
            5 - p[0],
            p[1] + 2,
            2 + 1.5 * cs.sin(2 * np.pi * p[0] / 5) - p[1],
        ]
        O2 = [
            p[0],
            5 - p[0],
            p[1] - 4 - 1.5 * cs.sin(2 * np.pi * p[0] / 5),
            8 - p[1],
        ]
        object_prod = lambda O: prod(map(lambda h: cs.fmax(0, h), O))
        self.stage_constr = cs.Function("c", [state], [cs.vertcat(
            object_prod(O1),
            object_prod(O2),
        )])
        self.stage_constr_box = (
            -np.inf * np.ones((self.stage_constr.size1_out(0),)),
            0 * np.ones((self.stage_constr.size1_out(0),)),
        )
        self.term_constr = self.stage_constr
        self.term_constr_box = self.stage_constr_box

        # Initial state
        self.p0 = p0 = np.array([-2., 5.])
        self.v0 = v0 = np.array([0.])
        self.θ0 = θ0 = np.array([0.])
        self.initial_state = np.concatenate((p0, v0, θ0))

        self.pf = pf = np.array([6., 3.])

        # Objective
        fudge = 1e1
        self.stage_cost_state = cs.Function("lx", [state], [
            0.02 * fudge * cs.sum1((p - pf)**2) +
            0.0002 * fudge * cs.sum1(v**2) +
            0.0002 * fudge * cs.sum1(θ**2) +
            0
        ])
        self.stage_cost_input = cs.Function("lu", [input], [
            0.01 * fudge * input.T @ input +
            0
        ])
        self.stage_cost = cs.Function("l", [cs.vertcat(state, input)], [
            self.stage_cost_input(input) + self.stage_cost_state(state)
        ])
        self.term_cost = cs.Function("l_N", [state], [
            2 * fudge * cs.sum1((p - pf)**2) +
            20 * fudge * cs.sum1(v**2) +
            0.02 * fudge * cs.sum1(θ**2) +
            0
        ])
        self.initial_guess = np.tile(np.array([0., 0.]), N)

        self.plot_2d = True
        self.plot_figsize = (8, 10)
        self.plot_x = 0
        self.plot_y = 1
        self.plot_collision_constr = (0, 1)
        self.plot_constr_xlim = (0, 5)
        self.plot_constr_ylim = (-2, 8)
        self.plot_constr_num = 200
