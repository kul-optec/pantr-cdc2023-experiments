from dataclasses import dataclass
import casadi as cs
import numpy as np
from typing import List, Tuple

@dataclass
class OCProblem:
    N: int
    Ts: float
    state_var: cs.SX
    input_var: cs.SX
    state_names: List[str]
    input_names: List[str]
    initial_guess: np.ndarray
    initial_state: np.ndarray
    f_dynamics: cs.Function
    stage_constr: cs.Function
    stage_constr_box: Tuple[np.ndarray, np.ndarray]
    term_constr: cs.Function
    term_constr_box: Tuple[np.ndarray, np.ndarray]
    input_constr_box: Tuple[np.ndarray, np.ndarray]
    stage_cost: cs.Function
    term_cost: cs.Function

    plot_2d: bool = False
    plot_figsize: Tuple[int, int] = (6, 8)
    plot_x: int = 0
    plot_y: int = 1
    plot_collision_constr: Tuple[int, ...] = (0,)
    plot_constr_xlim: Tuple[int, int] = (-1, 1)
    plot_constr_ylim: Tuple[int, int] = (-1, 1)
    plot_constr_num: int = 128

    @property
    def nu(self) -> int:
        assert self.input_var.shape[1] == 1
        return self.input_var.shape[0]

    @property
    def nx(self) -> int:
        assert self.state_var.shape[1] == 1
        return self.state_var.shape[0]

    @property
    def nc(self) -> int:
        return self.stage_constr_box[0].shape[0]

    @property
    def nc_N(self) -> int:
        return self.term_constr_box[0].shape[0]

    def eval_constr_xy(self, i):
        def func(x, y):
            inputs = [0] * self.nx
            inputs[self.plot_x] = x
            inputs[self.plot_y] = y
            constr = self.stage_constr(inputs)
            return constr[i]
        return np.vectorize(func)

    def simulate(self, uk):
        nx, nu = self.nx, self.nu
        N = len(uk) // nu
        result = np.empty((N * (nu + nx) + nx,))
        result[:nx] = self.initial_state
        for i in range(N):
            result[i * (nx + nu) + nx:i * (nx + nu) + nx + nu] = uk[i * nu: i * nu + nu]
            result[(i + 1) * (nx + nu):(i + 1) * (nx + nu) + nx] = \
                self.f_dynamics(result[i * (nx + nu):i * (nx + nu) + nx], 
                                result[i * (nx + nu) + nx:i * (nx + nu) + nx + nu]).full().squeeze()
        return result

    def simulate_states(self, uk):
        nx, nu = self.nx, self.nu
        N = len(uk) // nu
        uk = np.reshape(uk, (nu, N), order='F')
        result = np.empty((nx, N + 1), order='F')
        result[:, 0] = self.initial_state
        for i in range(N):
            x_next = self.f_dynamics(result[:, i], uk[:, i])
            result[:, i + 1] = x_next.full().squeeze()
        return result


__all__ = [
    'OCProblem',
]
