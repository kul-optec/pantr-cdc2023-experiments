from benchmark_util import load_results, total_evals
from datetime import datetime, timedelta
from pprint import pprint
import sys
from os.path import join

outdir = sys.argv[1]
outname = sys.argv[2]
Δ = int(sys.argv[3])
N = int(sys.argv[4])
names = sys.argv[5:]

warm_results = {
    key: [load_results(outdir, name.format(i)) for i in range(Δ, N + 1, Δ)]
    for temp, key, name in map(lambda x: x.split(':', 2), names)
    if temp == 'warm'
}
cold_results = {
    key: [load_results(outdir, name.format(i)) for i in range(Δ, N + 1, Δ)]
    for temp, key, name in map(lambda x: x.split(':', 2), names)
    if temp == 'cold'
}

# Plot iterate data
import matplotlib.pyplot as plt
import numpy as np

figsize = 2 * 4.5, 4
with_suptitle = False
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\renewcommand{\sfdefault}{phv}\renewcommand{\rmdefault}{ptm}",
        "font.family": "ptm",
        "font.size": 15,
        "lines.linewidth": 1,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "lines.markersize": 4,
        "legend.framealpha": 0.7,
        "legend.borderpad": 0.25,
        "legend.handlelength": 1,
        "legend.labelspacing": 0.25,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.25,
    }
)
for temp_results in (warm_results, cold_results):
    for results in temp_results.values():
        for el in results:
            el["num_eval_total"] = total_evals(el["evaluations"])
            abs_runtimes = abs(el["runtimes"])
            el["runtime"] = abs_runtimes.sum()
            el["max_runtime"] = np.max(abs_runtimes)
            el["avg_runtime"] = abs_runtimes.sum() / len(abs_runtimes)
            el["geomean_runtime"] = np.power(abs_runtimes.prod(), 1. / len(abs_runtimes))
            el["p95_runtime"] = np.quantile(abs_runtimes, .95)
            el["p90_runtime"] = np.quantile(abs_runtimes, .90)
            el["p5_runtime"] = np.quantile(abs_runtimes, .05)
            el["med_runtime"] = np.median(abs_runtimes)
            el["success"] = (el["runtimes"] < 0).sum() == 0

metrics = [
    "num_eval_total",
    "avg_runtime",
    "geomean_runtime",
    # "max_runtime",
    # "med_runtime",
    # "iter",
    # "f",
]
metric_names = {
    "num_eval_total": "Total problem function evaluations",
    "runtime": "Total solver run time",
    "max_runtime": "Maximum solver run time",
    "avg_runtime": "Average solver run time",
    "geomean_runtime": "Geometric mean of solver run time",
    "med_runtime": "Median solver run time",
    "iter": "Total number of iterations",
    "f": "Optimal cost",
}

ylabel_names = {
    "num_eval_total": "Evaluations",
    "runtime": "Solver run time [s]",
    "max_runtime": "Solver run time [s]",
    "avg_runtime": "Solver run time [s]",
    "geomean_runtime": "Solver run time [s]",
    "med_runtime": "Solver run time [s]",
    "iter": "Iterations",
    "f": r"Cost $f(x^\star)$",
}

fig, (ax_cold, ax_warm) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
metric = "avg_runtime"

min_y = np.inf
def do_plot_avg(temp_results, ax):
    global min_y
    for lbl, result in temp_results.items():
        reduce_data = lambda f: list(map(f, result))
        valid = np.array(reduce_data(lambda x: x["success"]))
        invalid = np.logical_not(valid)
        t_avg = np.array(reduce_data(lambda x: x[metric])).astype(np.float64)
        min_y = np.fmin(min_y, np.min(t_avg, initial=min_y))
        t_p95 = np.array(reduce_data(lambda x: x["p95_runtime"])).astype(np.float64)
        t_p90 = np.array(reduce_data(lambda x: x["p90_runtime"])).astype(np.float64)
        t_p5 = np.array(reduce_data(lambda x: x["p5_runtime"])).astype(np.float64)
        ind = np.arange(1, len(t_avg) + 1)
        p, = ax.semilogy(ind[valid], t_avg[valid], '.', label=lbl)
        ax.semilogy(ind[invalid], t_avg[invalid], 'x', color=p.get_color())
        ax.semilogy(ind, t_avg, ':', linewidth=1, color=p.get_color())
        ax.fill_between(ind, t_p5, t_p95, color=p.get_color(), alpha=0.25)
    ax.set_xlabel("Problem horizon $N$")
    ax.set_xlim(ind[0] - 1, ind[-1] + 1)

do_plot_avg(cold_results, ax_cold)
ax_cold.set_title("Cold start")
do_plot_avg(warm_results, ax_warm)
ax_warm.set_title("Warm start")
if with_suptitle:
    plt.suptitle(metric_names[metric], size=18)

if ylbl := ylabel_names.get(metric):
    ax_cold.set_ylabel(ylbl)
ax_cold.set_ylim(min_y / 2, None)
ax_cold.legend()
plt.tight_layout()
if with_suptitle:
    plt.subplots_adjust(top=0.85)
plt.savefig(join(outdir, outname + "-avg-runtimes-quantiles-cold-warm.pdf"))

fig, (ax_cold, ax_warm) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
def do_plot_mpc_runtimes(temp_results, ax):
    for lbl, result in temp_results.items():
        reduce_data = lambda f: list(map(f, result))
        ts = np.array(reduce_data(lambda x: x["runtimes"]))[-1].astype(np.float64)
        valid = ts > 0
        invalid = np.logical_not(valid)
        ts = abs(ts)
        ind = np.arange(1, ts.shape[-1] + 1)
        p, = ax.semilogy(ind[valid], ts[valid], '.', label=lbl)
        ax.semilogy(ind[invalid], ts[invalid], 'x', color=p.get_color())
        ax.semilogy(ind, ts, ':', linewidth=1, color=p.get_color())
    ax.set_xlabel("MPC time step")
    ax.set_xlim(ind[0] - 1, ind[-1] + 1)

do_plot_mpc_runtimes(cold_results, ax_cold)
ax_cold.set_title("Cold start")
do_plot_mpc_runtimes(warm_results, ax_warm)
ax_warm.set_title("Warm start")
if with_suptitle:
    plt.suptitle("Solver run times", size=18)

if ylbl := ylabel_names.get(metric):
    ax_cold.set_ylabel(ylbl)
ax_cold.legend(loc='upper right')
plt.tight_layout()
if with_suptitle:
    plt.subplots_adjust(top=0.85)
plt.savefig(join(outdir, outname + "-runtimes-mpc-last-cold-warm.pdf"))

plt.show()
