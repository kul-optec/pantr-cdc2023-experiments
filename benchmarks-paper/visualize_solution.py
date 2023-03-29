from benchmark_util import load_results
from importlib import import_module
import sys

if len(sys.argv) > 1:
    results = load_results(sys.argv[1])
else:
    results = load_results('1678901391005_49bf0460')

module = import_module("alpaqa_mpc_benchmarks.problems." + results["problem"])
ocp = module.Problem(N=results["horizon"])
assert ocp.plot_2d, "Problem cannot be visualized in 2D"

# Plot iterate data
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np

plt.figure(figsize=ocp.plot_figsize)
sol = results["solution"]
if results.get("formulation", "").startswith("ms"):
    import alpaqa_mpc_benchmarks.formulations.ms as form_ms
    xs, _ = form_ms.ms_extract_states_inputs(form_ms.ocp_to_ms(ocp), sol)
    xs = np.concatenate((ocp.initial_state, xs))
    xs = xs.reshape((ocp.nx, ocp.N + 1), order='F')
else:
    xs = ocp.simulate_states(sol)
x_plt = np.linspace(*ocp.plot_constr_xlim, num=ocp.plot_constr_num)
y_plt = np.linspace(*ocp.plot_constr_ylim, num=ocp.plot_constr_num)
X, Y = np.meshgrid(x_plt, y_plt)
for i in ocp.plot_collision_constr:
    C = ocp.eval_constr_xy(i)(X, Y)
    plt.contour(X, Y, C)
    fx = [patheffects.withTickedStroke(spacing=7, linewidth=0.8)]
    eps = 0
    if np.isfinite(lb := ocp.stage_constr_box[0][i]):
        cgc = plt.contour(X, Y, -C, [-lb - eps], colors='k', linewidths=0.8)
        plt.setp(cgc.collections, path_effects=fx)
    if np.isfinite(ub := ocp.stage_constr_box[1][i]):
        cgc = plt.contour(X, Y, +C, [+ub + eps], colors='k', linewidths=0.8)
        plt.setp(cgc.collections, path_effects=fx)
plt.plot(xs[0, :], xs[1, :], 'x-', color='tab:red')
plt.xlabel(ocp.state_names[0])
plt.ylabel(ocp.state_names[1])
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
