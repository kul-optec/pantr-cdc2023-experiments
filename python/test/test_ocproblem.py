from alpaqa_mpc_benchmarks.problems import QuadcopterProblem
from alpaqa_mpc_benchmarks.formulations.ss import *
from alpaqa_mpc_benchmarks.formulations.ms import *
import numpy.linalg as la
import numpy as np

def test_ocproblem():
    ocp = QuadcopterProblem(0.1, 8)
    ss = ocp_to_ss(ocp)
    ipoptopt = {"ipopt": {"tol": 1e-8, "constr_viol_tol": 1e-6}}
    nlp, bounds, guess = ss_solver_ipopt(ss, ipoptopt)
    ss_sol = nlp(p=ss.initial_state, **bounds, **guess)["x"]

    ocp.initial_guess = ss_sol.full().ravel()
    ms = ocp_to_ms(ocp)
    nlp, bounds, guess = ms_solver_ipopt(ms, ipoptopt)
    ms_sol = nlp(p=ms.initial_state, **bounds, **guess)["x"]
    _, ms_sol = ms_extract_states_inputs(ms, ms_sol)

    assert la.norm(ss_sol - ms_sol, ord=np.inf) < 1e-5


def test_ocproblem_soft():
    ocp = QuadcopterProblem(0.1, 20)
    μ = 1e3 * np.ones((ocp.N * ocp.nc + ocp.nc_N,))
    ss = ocp_to_ss_soft(ocp, μ)
    ipoptopt = {"ipopt": {"tol": 1e-10, "constr_viol_tol": 1e-10}}
    nlp, bounds, guess = ss_solver_ipopt(ss, ipoptopt)
    ss_sol = nlp(p=ss.initial_state, **bounds, **guess)["x"]

    ocp.initial_guess = ss_sol.full().ravel()
    ms = ocp_to_ms_soft(ocp, μ)
    nlp, bounds, guess = ms_solver_ipopt(ms, ipoptopt)
    ms_sol = nlp(p=ms.initial_state, **bounds, **guess)["x"]
    _, ms_sol = ms_extract_states_inputs(ms, ms_sol)

    assert la.norm(ss_sol - ms_sol, ord=np.inf) < 1e-5
