from alpaqa_mpc_benchmarks.problems import HangingChainProblem, QuadcopterProblem
from alpaqa_mpc_benchmarks.formulations.ss import ocp_to_ss, ss_solver_ipopt
from alpaqa_mpc_benchmarks.formulations.ms import ocp_to_ms, ms_solver_ipopt

# ocp = HangingChainProblem(N=4)
ocp = QuadcopterProblem(N=12)
ipoptopt = {"ipopt": {"tol": 1e-8, "constr_viol_tol": 1e-8, "print_timing_statistics": "yes"}}

ss = ocp_to_ss(ocp)
nlp, bounds, guess = ss_solver_ipopt(ss, ipoptopt)
ss_sol = nlp(p=ss.initial_state, **bounds, **guess)["x"]

ms = ocp_to_ms(ocp)
nlp, bounds, guess = ms_solver_ipopt(ms, ipoptopt)
ms_sol = nlp(p=ms.initial_state, **bounds, **guess)["x"]
