from benchmark_util import load_experiment
from importlib import import_module
import casadi as cs
import sys
from os.path import join

outdir = sys.argv[1]
outname = sys.argv[2]
name = sys.argv[3]
results = load_experiment(outdir, f"{name}_ids.txt")[-1]

module = import_module("alpaqa_mpc_benchmarks.problems." + results["problem"])
ocp = module.Problem(N=results["horizon"])
assert ocp.plot_2d, "Problem cannot be visualized in 2D"

# Plot iterate data
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\renewcommand{\sfdefault}{phv}\renewcommand{\rmdefault}{ptm}",
        "font.family": "ptm",
        "font.size": 14,
        "lines.linewidth": 1,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)

plt.figure(figsize=ocp.plot_figsize)
if results.get("formulation", "").startswith("ms"):
    assert False
else:
    xs = np.array(results["states"], order='F')
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
plt.plot(xs[0, :], xs[1, :], '4-', color='tab:red')
plt.xlabel(ocp.state_names[0])
plt.ylabel(ocp.state_names[1])
plt.gca().set_aspect('equal')
plt.xticks([-0.2, 0, 0.2])
plt.yticks([-0.2, 0, 0.2])
plt.tight_layout()
plt.savefig(join(outdir, outname + f"-solution.pdf"))
plt.show()
