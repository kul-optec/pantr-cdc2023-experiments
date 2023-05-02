from ..formulations.ocp import OCProblem
import numpy as np
import casadi as cs

class Problem(OCProblem):
    def __init__(self, Ts: float = 0.1, N: int = 30):
        self.Ts = Ts
        self.N = N

        # Parameters
        atmin = 0
        atmax = 9.81 * 5
        tiltmax = 1.1 / 2
        dtiltmax = 0.1

        # Dynamics
        p = cs.SX.sym("p", 3)
        v = cs.SX.sym("v", 3)
        θ = cs.SX.sym("theta", 3)
        at = cs.SX.sym("at", 1)
        ω = cs.SX.sym("omega", 3)
        self.state_var = state = cs.vertcat(p, v, θ)
        self.input_var = input = cs.vertcat(at, ω)
        self.state_names = ["$p_x$", "$p_y$", "$p_z$",
                            "$v_x$", "$v_y$", "$v_z$",
                            r"$\theta_x$", r"$\theta_y$", r"$\theta_z$"]
        self.input_names = [r"$a_t$",
                            r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

        cr = np.cos(θ[0])
        sr = np.sin(θ[0])
        cp = np.cos(θ[1])
        sp = np.sin(θ[1])
        cy = np.cos(θ[2])
        sy = np.sin(θ[2])
        R = cs.vertcat(cs.horzcat(cy*cp, cy*sp*sr-sy*cr, cy*sp*cr + sy*sr),
                       cs.horzcat(sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
                       cs.horzcat(-sp, cp*sr, cp*cr))

        at_ = R @ cs.vertcat(0, 0, at)
        g = cs.vertcat(0, 0, -9.81)
        a = at_ + g

        self.f_c = cs.Function("f_c", [state, input], [cs.vertcat(v, a, ω)])

        # Discretization
        k1 = self.f_c(state, input)
        k2 = self.f_c(state + Ts * k1 / 2, input)
        k3 = self.f_c(state + Ts * k2 / 2, input)
        k4 = self.f_c(state + Ts * k3, input)

        f_d_expr = state + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.f_dynamics = cs.Function("f", [state, input], [f_d_expr])

        # Input constraints
        self.input_constr_box = (
            np.array([atmin, -dtiltmax, -dtiltmax, -dtiltmax]),
            np.array([atmax, +dtiltmax, +dtiltmax, +dtiltmax]),
        )

        # State constraints
        self.stage_constr = cs.Function("c", [state], [cs.vertcat(
            θ[0],
            θ[1],
            cs.cos(θ[0]) * cs.cos(θ[1]),
            # 1e4 * cs.fmax(0, 0.1**2 - p[0]**2) * cs.fmax(0, 0.1**2 - p[1]**2),
            # x²+y²>r²   ~   r²-x²-y²<0
            0.1**2 - p[0]**2 - p[1]**2,
        )])
        self.stage_constr_box = (
            np.array([-np.pi/2, -np.pi/2, np.cos(tiltmax), -np.inf]),
            np.array([+np.pi/2, +np.pi/2, +np.inf, 0]),
        )
        self.term_constr = self.stage_constr
        self.term_constr_box = self.stage_constr_box

        # Initial state
        self.p0 = p0 = np.array([-0.2, -0.25, 0.5])
        # self.v0 = v0 = np.array([1., 1., 1.])
        self.v0 = v0 = np.array([0., 0., 0.])
        # self.θ0 = θ0 = np.array([np.pi/10, np.pi/10, np.pi/10])
        self.θ0 = θ0 = np.array([0., 0., 0.])
        self.initial_state = np.concatenate((p0, v0, θ0))

        self.pf = pf = np.array([0.25, 0.25, 0.5])

        # Objective
        alpha = 1.
        beta = 10.
        gamma = 10.
        delta = 1.
        self.stage_cost_state = cs.Function("lx", [state], [
            alpha * cs.sum1(v**2) +
            gamma * cs.sum1((p - pf)**2) +
            delta * cs.sum1(θ**2) +
            0
        ])
        self.stage_cost_input = cs.Function("lu", [input], [
            1e-4 * input.T @ input +
            beta * cs.sum1(ω**2) + 
            0
        ])
        self.stage_cost = cs.Function("l", [cs.vertcat(state, input)], [
            self.stage_cost_input(input) + self.stage_cost_state(state)
        ])
        self.term_cost = cs.Function("l_N", [state], [
            25 * alpha * cs.sum1(v**2) +
            25 * gamma * cs.sum1((p - pf)**2) +
            10 * delta * cs.sum1(θ**2) +
            0
        ])
        self.initial_guess = np.tile(np.array([9.81, 0, 0, 0]), N)

        self.plot_2d = True
        self.plot_figsize = (4, 4)
        self.plot_x = 0
        self.plot_y = 1
        self.plot_collision_constr = (3,)
        self.plot_constr_xlim = (-0.35, 0.35)
        self.plot_constr_ylim = (-0.35, 0.35)
        self.plot_constr_num = 200
