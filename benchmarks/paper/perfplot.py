from benchmark_util import load_results, load_experiment, total_evals
from datetime import datetime, timedelta
from pprint import pprint
import sys
from os.path import join

outdir = sys.argv[1]
outname = sys.argv[2]
delta = int(sys.argv[3])
names = sys.argv[4:]
all_results = {
    key: load_experiment(outdir, f"{name}_ids.txt")
    for key, name in map(lambda x: x.split(':', 1), names)
}

# Print problem info
def print_solver_info(results):
    solvers = set(res['solver'] for res in results)
    assert len(solvers) == 1, "Different solvers"
    first_solver = next(iter(solvers))
    opts = set(tuple(res['opts']) for res in results)
    assert len(opts) == 1, "Different solver options"
    first_opts = next(iter(opts))
    print(f"  {first_solver}")
    if first_opts:
        for opt in first_opts[:-1]:
            print(f"  ├─ {opt}")
        print(f"  └─ {first_opts[-1]}")

for lbl, results in all_results.items():
    print(f"\033[1m{lbl}\033[0m:")
    print_solver_info(results)

problem_paths = [tuple(x["path"] for x in res) for res in all_results.values()]
if len(set(problem_paths)) != 1:
    print("Warning: Mismatching problems")
print(f"\n\033[1mProblems\033[0m:")
pprint(set(problem_paths))

# Plot iterate data
import matplotlib.pyplot as plt
import numpy as np

figsize = 4.5, 3.75
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

for results in all_results.values():
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
    "num_eval_total": "Total evaluations",
    "runtime": "Total run time",
    "max_runtime": "Maximum run time",
    "avg_runtime": "Average run time",
    "geomean_runtime": "Geometric mean of run time",
    "med_runtime": "Median run time",
    "iter": "Total number of iterations",
    "f": "Optimal cost",
}

ylabel_names = {
    "num_eval_total": "Evaluations",
    "runtime": "Run time [s]",
    "max_runtime": "Run time [s]",
    "avg_runtime": "Run time [s]",
    "geomean_runtime": "Run time [s]",
    "med_runtime": "Run time [s]",
    "iter": "Iterations",
    "f": r"Cost $f(x^\star)$",
}


def do_perf_plot(metric):
    reduce_data = lambda f: [list(map(f, df)) for df in all_results.values()]
    valid = np.array(reduce_data(lambda x: x.get("success", True)))
    invalid = np.logical_not(valid)
    t_sp_orig = np.array(reduce_data(lambda x: x[metric])).astype(np.float64)
    t_sp = t_sp_orig.copy()
    t_sp[invalid] = np.inf
    n_p = t_sp.shape[1]
    n_s = t_sp.shape[0]
    tm_sp = np.min(t_sp, axis=0)
    tm_sp = np.tile(tm_sp, (n_s, 1))
    r_sp = t_sp / tm_sp

    plt.figure(figsize=figsize)
    plt.title('Performance profile: ' + metric_names[metric])

    τ_max = np.max(r_sp[np.isfinite(r_sp)]) * 2
    for ρ_solver, label in zip(r_sp, all_results):
        ρ_solver = ρ_solver[np.isfinite(ρ_solver)]
        x, y = np.unique(ρ_solver, return_counts=True)
        y = (1. / n_p) * np.cumsum(y)
        if len(x) > 0:
            x = np.append(x, [τ_max])
            y = np.append(y, [y[-1]])
        else:
            x = np.array([1, τ_max])
            y = np.array([0, 0])
        if x[0] != 1:
            x = np.concatenate(([1], x))
            y = np.concatenate(([0], y))
        plt.step(x, y, '-', where='post', label=label)

    plt.xscale("log")
    plt.xlim(1, τ_max * 0.99)
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\rho_s(\tau)$')
    # plt.ylim(0, None)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(join(outdir, outname + f"-perfplot-{metric}.pdf"))

    plt.figure(figsize=figsize)
    plt.title(metric_names[metric])
    for t, vld, invld, lbl in zip(t_sp_orig, valid, invalid, all_results):
        ind = np.arange(1, len(t) + 1)
        p, = plt.semilogy(ind[vld], t[vld], '.', label=lbl)
        plt.semilogy(ind[invld], t[invld], 'x', color=p.get_color())
        plt.semilogy(ind, t, ':', linewidth=1, color=p.get_color())
    plt.xlabel("Problem horizon $N$")
    if ylbl := ylabel_names.get(metric):
        plt.ylabel(ylbl)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(outdir, outname + f"-{metric}.pdf"))

for m in metrics:
    do_perf_plot(m)

metric = "med_runtime"
plt.figure(figsize=figsize)
plt.title(metric_names[metric])
for lbl, result in all_results.items():
    reduce_data = lambda f: list(map(f, result))
    valid = np.array(reduce_data(lambda x: x["success"]))
    invalid = np.logical_not(valid)
    t_med = np.array(reduce_data(lambda x: x[metric])).astype(np.float64)
    t_p95 = np.array(reduce_data(lambda x: x["p95_runtime"])).astype(np.float64)
    t_p90 = np.array(reduce_data(lambda x: x["p90_runtime"])).astype(np.float64)
    t_p5 = np.array(reduce_data(lambda x: x["p5_runtime"])).astype(np.float64)
    ind = np.arange(1, len(t_med) + 1)
    p, = plt.plot(ind[valid], t_med[valid], '.', label=lbl)
    plt.plot(ind[invalid], t_med[invalid], 'x', color=p.get_color())
    plt.plot(ind, t_med, ':', linewidth=1, color=p.get_color())
    plt.fill_between(ind, t_p5, t_p95, color=p.get_color(), alpha=0.25)
if ylbl := ylabel_names.get(metric):
    plt.ylabel(ylbl)
plt.xlabel("Problem horizon $N$")
plt.legend()
plt.tight_layout()
plt.savefig(join(outdir, outname + "-med-runtimes-lin.pdf"))

metric = "med_runtime"
plt.figure(figsize=figsize)
plt.title(metric_names[metric])
for lbl, result in all_results.items():
    reduce_data = lambda f: list(map(f, result))
    valid = np.array(reduce_data(lambda x: x["success"]))
    invalid = np.logical_not(valid)
    t_med = np.array(reduce_data(lambda x: x[metric])).astype(np.float64)
    ind = np.arange(1, len(t_med) + 1)
    p, = plt.semilogy(ind[valid], t_med[valid], '.', label=lbl)
    plt.semilogy(ind[invalid], t_med[invalid], 'x', color=p.get_color())
    plt.semilogy(ind, t_med, ':', linewidth=1, color=p.get_color())
plt.xlabel("Problem horizon $N$")
if ylbl := ylabel_names.get(metric):
    plt.ylabel(ylbl)
plt.legend()
plt.tight_layout()
plt.savefig(join(outdir, outname + "-med-runtimes.pdf"))

metric = "med_runtime"
plt.figure(figsize=figsize)
plt.title(metric_names[metric])
for lbl, result in all_results.items():
    reduce_data = lambda f: list(map(f, result))
    valid = np.array(reduce_data(lambda x: x["success"]))
    invalid = np.logical_not(valid)
    t_med = np.array(reduce_data(lambda x: x[metric])).astype(np.float64)
    t_p95 = np.array(reduce_data(lambda x: x["p95_runtime"])).astype(np.float64)
    t_p90 = np.array(reduce_data(lambda x: x["p90_runtime"])).astype(np.float64)
    t_p5 = np.array(reduce_data(lambda x: x["p5_runtime"])).astype(np.float64)
    ind = np.arange(1, len(t_med) + 1)
    p, = plt.semilogy(ind[valid], t_med[valid], '.', label=lbl)
    plt.semilogy(ind[invalid], t_med[invalid], 'x', color=p.get_color())
    plt.semilogy(ind, t_med, ':', linewidth=1, color=p.get_color())
    plt.fill_between(ind, t_p5, t_p95, color=p.get_color(), alpha=0.25)
plt.xlabel("Problem horizon $N$")
if ylbl := ylabel_names.get(metric):
    plt.ylabel(ylbl)
plt.legend()
plt.tight_layout()
plt.savefig(join(outdir, outname + "-med-runtimes-quantiles.pdf"))

plt.figure(figsize=figsize)
plt.title("Runtimes")
for lbl, result in all_results.items():
    reduce_data = lambda f: list(map(f, result))
    ts = np.array(reduce_data(lambda x: x["runtimes"]))[-1].astype(np.float64)
    valid = ts > 0
    invalid = np.logical_not(valid)
    ts = abs(ts)
    ind = np.arange(1, ts.shape[-1] + 1)
    p, = plt.semilogy(ind[valid], ts[valid], '.', label=lbl)
    plt.semilogy(ind[invalid], ts[invalid], 'x', color=p.get_color())
    plt.semilogy(ind, ts, ':', linewidth=1, color=p.get_color())
plt.xlabel("MPC time step")
if ylbl := ylabel_names.get("avg_runtime"):
    plt.ylabel(ylbl)
plt.legend()
plt.tight_layout()
plt.savefig(join(outdir, outname + "-runtimes-mpc-last.pdf"))

plt.show()
