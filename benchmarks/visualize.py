from benchmark_util import load_results, load_experiment
from datetime import datetime, timedelta

# all_results = {
#     "TR 1.00000005": load_results(1676767875063),
#     "TR 1.00000006": load_results(1676767875357),
#     "PANOC 1.00000005": load_results(1676767875525),
#     "PANOC 1.00000006": load_results(1676767875709),
#     "Str. PANOC 1.00000005": load_results(1676767875767),
#     "Str. PANOC 1.00000006": load_results(1676767875820),
# }

# all_results = {
#     "TR 1.00000005": load_results(1676773661798),
#     "TR 1.00000006": load_results(1676773662609),
#     "PANOC 1.00000005": load_results(1676773664022),
#     "PANOC 1.00000006": load_results(1676773665385),
#     "Str. PANOC 1.00000005": load_results(1676773665767),
#     "Str. PANOC 1.00000006": load_results(1676773666154),
# }

# all_results = {
#     "Newt. PANOC 0.0000041": load_results('1677081402190_ccbf4bda'),
#     "Newt. PANOC 0.0000046": load_results('1677081372077_12ba1f3d'),
# }

# # Integrator 30 comparing updating L-BFGS in candidate vs accepted step
# all_results = {
#     "PANOC candidate": load_results(1676837138961),
#     "PANOC accepted": load_results(1676837138970),
#     "Str. PANOC candidate": load_results(1676837138978),
#     "Str. PANOC accepted": load_results(1676837138987),
# }

# # Quadcopter 12 comparing updating L-BFGS in candidate vs accepted step
# all_results = {
#     "PANOC candidate": load_results(1676837300647),
#     "PANOC accepted": load_results(1676837300787),
#     "Str. PANOC candidate": load_results(1676837300830),
#     "Str. PANOC accepted": load_results(1676837300863),
# }

# # Quadcopter 30 comparing updating L-BFGS in candidate vs accepted step
# all_results = {
#     "PANOC candidate": load_results(1676837734855),
#     "PANOC accepted": load_results(1676837736357),
#     "Str. PANOC candidate": load_results(1676837737928),
#     "Str. PANOC accepted": load_results(1676837738460),
# }

# # Quadcopter 24 comparing updating L-BFGS in candidate vs accepted step
# all_results = {
#     "PANOC candidate": load_results(1676837984031),
#     "PANOC accepted": load_results(1676837985272),
#     "Str. PANOC candidate": load_results(1676837985580),
#     "Str. PANOC accepted": load_results(1676837985841),
# }

# # Quadcopter 26 comparing updating L-BFGS in candidate vs accepted step
# all_results = {
#     "PANOC candidate": load_results(1676838136778),
#     "PANOC accepted": load_results(1676838138274),
#     "Str. PANOC candidate": load_results(1676838138796),
#     "Str. PANOC accepted": load_results(1676838139151),
# }

# all_results = {
#     "TR 1.00000005": load_results(1676774534763),
#     "TR 1.00000006": load_results(1676774534789),
#     "PANOC 1.00000005": load_results(1676774534798),
#     "PANOC 1.00000006": load_results(1676774534806),
#     "Str. PANOC 1.00000005": load_results(1676774534814),
#     "Str. PANOC 1.00000006": load_results(1676774534822),
# }

# all_results = {
#     "TR 1.00000005": load_results(1676773149013, 'Debug'),
#     "TR 1.00000006": load_results(1676773153142, 'Debug'),
#     "PANOC 1.00000005": load_results(1676773156713, 'Debug'),
#     "PANOC 1.00000006": load_results(1676773160578, 'Debug'),
#     "Str. PANOC 1.00000005": load_results(1676773162492, 'Debug'),
#     "Str. PANOC 1.00000006": load_results(1676773164012, 'Debug'),
# }

# Quadcopter 40, c1=0.1, c3=25
all_results = {
    # "Str. PANOC": load_experiment("strucpanoc-15_ids.txt")[-1],
    # "L-BFGS TR": load_experiment("lbfgstr-5_ids.txt")[-1],
    # "Newton TR": load_experiment("newtontr_ids.txt")[-1],
}

all_results = {
    "quadcopter num_dist=0": load_results("1678867564613_5331d603"),
}

# Print problem info
problem_paths = set(res['path'] for res in all_results.values())
assert len(problem_paths) == 1, "Different problems"
first_result = next(iter(all_results.values()))
print(f"\033[1mProblem\033[0m: {first_result['problem']} "
      f"{first_result['horizon']} (\"{first_result['path']}\")")
print()

# Print solver info
for lbl, res in all_results.items():
    print(f"\033[1m{lbl}\033[0m: "
          f"({datetime.fromtimestamp(res['time_utc_ms'] // 1000)})")
    print(f"  {res['solver']}")
    if res['opts']:
        for opt in res['opts'][:-1]:
            print(f"  ├─ {opt}")
        print(f"  └─ {res['opts'][-1]}")
    print(f"  status:    {res['status']:<17}"
          f"  run time:     {timedelta(microseconds=res['runtime'] * 1e-3)}")
    print(f"  iterations:       {res['iter']:10}")
    print(f"  lbfgs rejected:   {res['lbfgs_rejected']:10}"
          f"  lbfgs fail:       {res['lbfgs_failures']:10}")
    print(f"  line search fail: {res['linesearch_failures']:10}"
          f"  backtrack/reject: {res['linesearch_backtracks']:10}")
    print()

# Print number of evaluations
lbl_w = max(map(len, all_results))
w = 10
print("\033[1mEvaluations\033[0m:")
print(f"  {'':{lbl_w}} "
      f"┌─\033[3m{'prox':─<{w}}\033[0m"
      f"┬─\033[3m{'cost─only':─<{w}}\033[0m"
      f"┬─\033[3m{'grad─tot':─<{w}}\033[0m"
      f"┬─\033[3m{'grad─only':─<{w}}\033[0m"
      f"┬─\033[3m{'cost+grad':─<{w}}\033[0m"
      f"┬─\033[3m{'hess─prod':─<{w}}\033[0m┐")
for lbl, res in all_results.items():
    evals = res["evaluations"]
    grad_only = evals['grad_f'] + evals['grad_ψ'] + evals['grad_ψ_from_ŷ'] + evals['grad_f_grad_g_prod']
    grad_cost = evals['f_grad_f'] + evals['ψ_grad_ψ'] + evals['f_grad_f_g']
    grad_tot = grad_only + grad_cost
    print(f"  \033[3m{lbl:>{lbl_w}}\033[0m "
          f"│{evals['prox_grad_step']:>{w}} "
          f"│{evals['f'] + evals['ψ'] + evals['f_g']:>{w}} "
          f"│{grad_tot:>{w}} "
          f"│{grad_only:>{w}} "
          f"│{grad_cost:>{w}} "
          f"│{evals['hess_ψ_prod'] + evals['hess_L_prod']:>{w}} │")
print(f"  {'':{lbl_w}} "
      f"└─{'':─<{w}}"
      f"┴─{'':─<{w}}"
      f"┴─{'':─<{w}}"
      f"┴─{'':─<{w}}"
      f"┴─{'':─<{w}}"
      f"┴─{'':─<{w}}┘")

# Plot iterate data
import matplotlib.pyplot as plt
import numpy as np

x_key = "it_time"
xlabels = {
    "it_time": "Run time",
    "grad_eval": "Total gradient evaluations",
}

if True:
    itstart = np.flatnonzero(res[x_key] == 0)
    itstart = np.append(itstart, len(res[x_key]))
    for i in range(len(itstart) - 1):
        x_start = res[x_key][max(0, itstart[i] - 1)]
        res[x_key][itstart[i]:itstart[i+1]] += x_start

    def plt_alm_iters():
        for i in itstart:
            if i > 0 and i < len(res[x_key]):
                plt.axvline(res[x_key][i], color='k', linestyle=':', linewidth=1)
else:
    def plt_alm_iters(): pass

plt.figure()
plt.title("FBE")
phi_star = min(map(np.min, (x["fbe"] for x in all_results.values())))
for lbl, res in all_results.items():
    plt.semilogy(res[x_key], res["fbe"] - phi_star, label=lbl)
plt_alm_iters()
plt.legend()
plt.xlabel(xlabels[x_key])
plt.tight_layout()

plt.figure()
plt.title("FPR")
for lbl, res in all_results.items():
    plt.semilogy(res[x_key], res["eps"], label=lbl)
plt_alm_iters()
plt.legend()
plt.xlabel(xlabels[x_key])
plt.tight_layout()

plt.figure()
plt.title("Line search parameter")
min_nonzero = lambda x: np.min(x[np.logical_and(x != 0, np.isfinite(x))])
taus = (x["tau"] for x in all_results.values() if x["tau"].size > 0)
min_tau = min(map(min_nonzero, taus), default=0)
for lbl, res in all_results.items():
    if len(res["tau"]) > 0:
        taus = res["tau"]
        zero_taus = (taus == 0).astype(np.float64)
        zero_taus[zero_taus == 0] = np.nan
        p, = plt.semilogy(res[x_key], taus, '.', label=lbl)
        plt.semilogy(res[x_key], zero_taus * min_tau / 2, 'x', color=p.get_color())
plt_alm_iters()
plt.legend()
plt.xlabel(xlabels[x_key])
plt.tight_layout()

plt.figure()
plt.title("FBS step size")
for lbl, res in all_results.items():
    if len(res["gamma"]) > 0:
        plt.semilogy(res[x_key], res["gamma"], '-', label=lbl)
plt_alm_iters()
plt.legend()
plt.xlabel(xlabels[x_key])
plt.tight_layout()

plt.figure()
plt.title("Accelerated step size and trust radius")
for lbl, res in all_results.items():
    p, = plt.semilogy(res[x_key], res["norm_q"], '-', label=lbl)
    if len(res["radius"]) > 0:
        plt.semilogy(res[x_key], res["radius"], '--', label=lbl + " (radius)", color=p.get_color())
plt_alm_iters()
plt.legend()
plt.xlabel(xlabels[x_key])
plt.tight_layout()

plt.figure()
plt.title("Accelerated step reduction ratio")
for lbl, res in all_results.items():
    if len(res["ratio"]) > 0:
        plt.plot(res[x_key], res["ratio"], '-', label=lbl)
plt_alm_iters()
plt.legend()
plt.xlabel(xlabels[x_key])
plt.tight_layout()

plt.show()
