from ..formulations.ocp import OCProblem
import numpy as np
import casadi as cs

class Problem(OCProblem):
    def __init__(self, Ts: float = 0.1, N: int = 30):
        self.Ts = Ts
        self.N = N

        # Parameters
        a_min = -1.
        a_max = -a_min

        # Dynamics
        p = cs.SX.sym("p", 3)
        v = cs.SX.sym("v", 3)
        a = cs.SX.sym("a", 3)
        self.state_var = state = cs.vertcat(p, v)
        self.input_var = input = a
        self.state_names = ["$p_x$", "$p_y$", "$p_z$",
                            "$v_x$", "$v_y$", "$v_z$"]
        self.input_names = [r"$a_x$", r"$a_y$", r"$a_z$"]

        self.f_c = cs.Function("f_c", [state, input], [cs.vertcat(v, a)])

        # Discretization
        k1 = self.f_c(state, input)
        k2 = self.f_c(state + Ts * k1 / 2, input)
        k3 = self.f_c(state + Ts * k2 / 2, input)
        k4 = self.f_c(state + Ts * k3, input)

        f_d_expr = state + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.f_dynamics = cs.Function("f", [state, input], [f_d_expr])

        # Input constraints
        self.input_constr_box = (
            np.array([a_min, a_min, a_min]),
            np.array([a_max, a_max, a_max]),
        )

        # State constraints
        self.stage_constr = cs.Function("c", [state], [cs.vertcat()])
        self.stage_constr_box = (
            np.array([]),
            np.array([]),
        )
        self.term_constr = self.stage_constr
        self.term_constr_box = self.stage_constr_box

        # Initial state
        self.p0 = p0 = np.array([-0.2, -0.25, 0.5])
        self.v0 = v0 = np.array([0., 0., 0.])
        self.initial_state = np.concatenate((p0, v0))

        self.pf = pf = np.array([0.25, 0.25, 0.5])

        # Objective
        alpha = 1.
        beta = 0.1
        gamma = 10.
        self.stage_cost_state = cs.Function("lx", [state], [
            alpha * cs.sum1(v**2) +
            gamma * cs.sum1((p - pf)**2) +
            0
        ])
        self.stage_cost_input = cs.Function("lu", [input], [
            beta * cs.sum1(a**2) +
            0
        ])
        self.stage_cost = cs.Function("l", [cs.vertcat(state, input)], [
            self.stage_cost_input(input) + self.stage_cost_state(state)
        ])
        self.term_cost = cs.Function("l_N", [state], [
            25 * alpha * cs.sum1(v**2) +
            25 * gamma * cs.sum1((p - pf)**2) +
            0
        ])
        self.initial_guess = np.tile(np.array([0, 0, 0]), N)

