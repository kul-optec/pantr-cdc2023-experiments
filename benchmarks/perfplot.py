from benchmark_util import load_results, load_experiment, total_evals
from datetime import datetime, timedelta
from pprint import pprint

# names_A = [
#     '1676910946918_6768d9d4',
#     '1676910946923_cf679036',
#     '1676910946928_f32fc857',
#     '1676910946933_eb15f4dd',
#     '1676910946940_07e76a2e',
#     '1676910946947_b12bf77a',
#     '1676910946956_a63b9f13',
#     '1676910946966_b5c64b8b',
#     '1676910946976_93d08268',
#     '1676910946989_f1b8a4b5',
#     '1676910947004_21d18416',
#     '1676910947019_8b0caf51',
#     '1676910947037_e9806006',
#     '1676910947059_e3e9ea02',
#     '1676910947081_32e83fff',
#     '1676910947106_29ad2712',
#     '1676910947136_9f327b35',
#     '1676910947164_0aeb6ef2',
#     '1676910947197_ef45ea33',
#     '1676910947236_55a780d0',
#     '1676910947241_77a8be55',
#     '1676910947245_d1b5c0f0',
#     '1676910947250_824db71d',
#     '1676910947255_6b115cca',
#     '1676910947264_5cf3bb45',
#     '1676910947294_d4092c92',
#     '1676910947346_074e54e6',
#     '1676910947429_95a680f7',
#     '1676910947544_99d75a66',
#     '1676910947677_21ace986',
#     '1676910947884_dcc934a8',
#     '1676910948138_1636cb68',
#     '1676910948493_8c31bec6',
#     '1676910948912_9531062f',
#     '1676910949425_78106f13',
#     '1676910950028_9c00ee37',
#     '1676910950770_7e835d9d',
#     '1676910951838_4d5fc34f',
#     '1676910953079_edea260b',
#     '1676910954312_054d5294',
# ]
# names_B = [
#     '1676911013995_586cee3f',
#     '1676911014001_4eeb4ed1',
#     '1676911014006_80f0b75c',
#     '1676911014013_6c68ff18',
#     '1676911014020_c17cac81',
#     '1676911014027_7f284795',
#     '1676911014036_ba34d3aa',
#     '1676911014047_0ed8942a',
#     '1676911014058_68b672d7',
#     '1676911014070_b5d0ee3b',
#     '1676911014086_a5e78cd4',
#     '1676911014101_d7dffdc6',
#     '1676911014119_d60c5c92',
#     '1676911014141_a6f174e8',
#     '1676911014165_a374ee03',
#     '1676911014191_6a37e970',
#     '1676911014221_4751e64c',
#     '1676911014252_25ad3593',
#     '1676911014287_b01bc9d3',
#     '1676911014325_c4c3a4e6',
#     '1676911014330_b768c989',
#     '1676911014334_f983e33b',
#     '1676911014338_3a493d6f',
#     '1676911014343_11ec4936',
#     '1676911014353_d3aa78f8',
#     '1676911014384_e301e77f',
#     '1676911014444_97f82716',
#     '1676911014525_72248758',
#     '1676911014641_91cc10ca',
#     '1676911014781_a99f7bad',
#     '1676911014986_8bf47b5b',
#     '1676911015226_c23b4129',
#     '1676911015631_c93bb1ac',
#     '1676911016456_07b6d03c',
#     '1676911017811_d90d1ca8',
#     '1676911018824_12ecbed4',
#     '1676911020608_30fbba0f',
#     '1676911025453_2d1735ea',
#     '1676911029111_b318659c',
#     '1676911031283_f012c7c2',
# ]
# all_results = {
#     "L-BFGS in accepted":
#         [load_results(ts) for i, ts in enumerate(names_A)],
#     "L-BFGS in candidate":
#         [load_results(ts) for i, ts in enumerate(names_B)],
# }

# names_A = [
#     '1676911329800_eb103e7f',
#     '1676911329818_48c2f564',
#     '1676911329836_71483931',
#     '1676911329859_bbe021b9',
#     '1676911329883_3989dc8b',
#     '1676911329909_2be9a6d8',
#     '1676911329935_0229a7cb',
#     '1676911329970_2a7dcb66',
#     '1676911330005_c51752c0',
#     '1676911330048_60642804',
#     '1676911330095_3c4e299d',
#     '1676911330237_d348a32a',
#     '1676911330443_a13c1012',
#     '1676911330670_2b164043',
#     '1676911331023_781e67a8',
#     '1676911331442_9baa8cf5',
#     '1676911331934_0968ad16',
#     '1676911332552_90b3b3a0',
#     '1676911333327_be359290',
#     '1676911334422_7e1f7008',
#     '1676911335726_a67edb7e',
#     '1676911336994_fbf1c218',
#     '1676911337021_42a53640',
#     '1676911337027_900903d9',
#     '1676911337053_99e6b967',
#     '1676911343733_d50df1ec',
#     '1676911343847_0362b1f0',
#     '1676911343956_741ff907',
#     '1676911344086_abf34aad',
#     '1676911344307_a4492648',
#     '1676911344524_e924a85e',
#     '1676911344957_07512d8b',
#     '1676911345170_d758b9fa',
# ]
# names_B = [
#     '1676911450770_a3d1716e',
#     '1676911450786_5a5e719b',
#     '1676911450802_b7816afd',
#     '1676911450820_511bf9b5',
#     '1676911450842_e2225500',
#     '1676911450861_bde01331',
#     '1676911450882_b16dd4c1',
#     '1676911450911_6b11db24',
#     '1676911450940_f209ac27',
#     '1676911450972_83c0b3c4',
#     '1676911451009_20a8450f',
#     '1676911451129_ce5ad22b',
#     '1676911451296_38719f87',
#     '1676911451541_9109fa62',
#     '1676911451845_f036a52a',
#     '1676911452496_a251d067',
#     '1676911453269_e2150658',
#     '1676911453946_20176692',
#     '1676911454680_3febd3ef',
#     '1676911455814_fa756b68',
#     '1676911457053_fc83ddff',
#     '1676911458298_11d47f6d',
#     '1676911458320_65a9711c',
#     '1676911458326_51cec955',
#     '1676911458370_48f60c0c',
#     '1676911458396_1395dacd',
#     '1676911458554_bf2ba95f',
#     '1676911458739_14012fe9',
#     '1676911458956_9ae5de29',
#     '1676911459497_5a3c820e',
#     '1676911459746_847b4a94',
#     '1676911460616_ffb4500b',
#     '1676911461046_8a84b97d',
# ]
# all_results = {
#     "L-BFGS(15)":
#         [load_results(ts) for i, ts in enumerate(names_A)],
#     "L-BFGS(25)":
#         [load_results(ts) for i, ts in enumerate(names_B)],
# }

# names_A = [
#     '1676911329800_eb103e7f',
#     '1676911329818_48c2f564',
#     '1676911329836_71483931',
#     '1676911329859_bbe021b9',
#     '1676911329883_3989dc8b',
#     '1676911329909_2be9a6d8',
#     '1676911329935_0229a7cb',
#     '1676911329970_2a7dcb66',
#     '1676911330005_c51752c0',
#     '1676911330048_60642804',
#     '1676911330095_3c4e299d',
#     '1676911330237_d348a32a',
#     '1676911330443_a13c1012',
#     '1676911330670_2b164043',
#     '1676911331023_781e67a8',
#     '1676911331442_9baa8cf5',
#     '1676911331934_0968ad16',
#     '1676911332552_90b3b3a0',
#     '1676911333327_be359290',
#     '1676911334422_7e1f7008',
#     '1676911335726_a67edb7e',
#     '1676911336994_fbf1c218',
#     '1676911337021_42a53640',
#     '1676911337027_900903d9',
#     '1676911337053_99e6b967',
#     '1676911343733_d50df1ec',
#     '1676911343847_0362b1f0',
#     '1676911343956_741ff907',
#     '1676911344086_abf34aad',
#     '1676911344307_a4492648',
#     '1676911344524_e924a85e',
#     '1676911344957_07512d8b',
#     '1676911345170_d758b9fa',
# ]
# names_B = [
#     '1676926551315_f4a3f164',
#     '1676926551336_fbd62b93',
#     '1676926551356_7f428d7e',
#     '1676926551380_9fc5d281',
#     '1676926551411_8494821d',
#     '1676926551446_d738ffea',
#     '1676926551486_63a3c634',
#     '1676926551525_073f12d9',
#     '1676926551570_b08bf74c',
#     '1676926551619_8e17103b',
#     '1676926551675_c9502746',
#     '1676926551870_df41fcbb',
#     '1676926552114_6c93f1b9',
#     '1676926552425_af4edb10',
#     '1676926552842_cb9d8f05',
#     '1676926553345_05dca473',
#     '1676926554032_5837f698',
#     '1676926554803_fdfbfd28',
#     '1676926555704_6a4b8edc',
#     '1676926556817_7ca36fb1',
#     '1676926558141_8710f77b',
#     '1676926559847_b4f1c7e9',
#     '1676926559875_38e33c76',
#     '1676926559885_b10713f7',
#     '1676926559900_96fefa81',
#     '1676926559932_7b5be7c9',
#     '1676926560041_7d61049d',
#     '1676926560153_30b98265',
#     '1676926560404_9a9f2f6a',
#     '1676926560637_2a88bda9',
#     '1676926560878_482636ac',
#     '1676926561148_03c870d9',
#     '1676926561397_f9411840',
# ]
# all_results = {
#     "Str. PANOC":
#         [load_results(ts) for i, ts in enumerate(names_A)],
#     "Str. ZeroFPR":
#         [load_results(ts) for i, ts in enumerate(names_B)],
# }

# all_results = {
#     "Str. PANOC (no update in prox)":
#         load_experiment("struczfpr-no-upd-prox_ids.txt"),
#     "Str. PANOC (update in prox)":
#         load_experiment("struczfpr-upd-prox_ids.txt"),
# }

# all_results = {
#     "PANOC":
#         load_experiment("lasso-panoc_ids.txt"),
#     "ZeroFPR":
#         load_experiment("lasso-zfpr_ids.txt"),
# }

# all_results = {
#     "PANOC":
#         load_experiment("logistic-panoc_ids.txt"),
#     "ZeroFPR":
#         load_experiment("logistic-zfpr_ids.txt"),
#     "FBETrust":
#         load_experiment("logistic-fbetrust_ids.txt"),
# }

# all_results = {
#     "Str. PANOC":
#         load_experiment("strucpanoc-nonewton_ids.txt"),
#     "Newton TR":
#         load_experiment("tr-newton_ids.txt"),
# }

# all_results = {
#     "Str. PANOC": load_experiment("strucpanoc_ids.txt"),
#     # "L-BFGS TR": load_experiment("lbfgstr_ids.txt"),
#     "Newton TR": load_experiment("newtontr_ids.txt"),
# }

# # Compare L-BFGS memories for PANOC
# all_results = {
#     # "Str. PANOC (5)": load_experiment("strucpanoc-5_ids.txt"),
#     "Str. PANOC (15)": load_experiment("strucpanoc-15_ids.txt"),
#     # "Str. PANOC (25)": load_experiment("strucpanoc-25_ids.txt"),
#     # "Str. PANOC (50)": load_experiment("strucpanoc-50_ids.txt"),
#     # "Str. PANOC (150)": load_experiment("strucpanoc-150_ids.txt"),
#     # "L-BFGS TR (5)": load_experiment("lbfgstr-5_ids.txt"),
#     # "L-BFGS TR (15)": load_experiment("lbfgstr-15_ids.txt"),
#     # "L-BFGS TR (25)": load_experiment("lbfgstr-25_ids.txt"),
#     "Newton TR": load_experiment("newtontr_ids.txt"),
# }

# all_results = {
#     # "Newton TR (old γ)": load_experiment("newtontr_ids.txt"),
#     "Newton TR (new γ)": load_experiment("newtontr-compute_ratio_using_new_γ_ids.txt"),
#     # "L-BFGS(5) TR (new γ)": load_experiment("lbfgstr-compute_ratio_using_new_γ-5_ids.txt"),
#     "L-BFGS(25) Str. PANOC": load_experiment("strucpanoc-25_ids.txt"),
# }

# all_results = {
#     "Ipopt": load_experiment("ipopt_ids.txt"),
#     "Newton TR": load_experiment("newton-tr_ids.txt"),
#     "Struc. PANOC(15)": load_experiment("strucpanoc_ids.txt"),
# }

# all_results = {
#     "Ipopt": load_experiment("mpc-ipopt_ids.txt"),
#     "Newton TR": load_experiment("mpc-newton-tr_ids.txt"),
#     "Struc. PANOC(15)": load_experiment("mpc-strucpanoc_ids.txt"),
# }

delta = 1
# name = "hanging_chain"
name = "tol1e-5-quadcopter"
# name = "tol1e-5-hanging_chain"
warm = False
all_results = {
    "Newton TR": load_experiment(f"mpc-{name}-newton-tr-{'warm' if warm else 'cold'}-Δ{delta}_ids.txt"),
    # "Newton TR (0)": load_experiment(f"mpc-{name}-newton-tr-0hessvec-Δ{delta}_ids.txt"),
    # "Newton TR (1)": load_experiment(f"mpc-{name}-newton-tr-1hessvec-Δ{delta}_ids.txt"),
    # "Struc. PANOC(15)": load_experiment(f"mpc-{name}-strucpanoc-Δ{delta}_ids.txt"),
    # "Struc. PANOC(30)": load_experiment(f"mpc-{name}-strucpanoc-mem30-Δ{delta}_ids.txt"),
    # "PANOC(50)": load_experiment(f"mpc-{name}-panoc-mem50-Δ{delta}_ids.txt"),
    "Struc. PANOC(50)": load_experiment(f"mpc-{name}-strucpanoc-mem50-{'warm' if warm else 'cold'}-Δ{delta}_ids.txt"),
    # "L-BFGS-B(15)": load_experiment(f"mpc-{name}-lbfgsb-mem15-Δ{delta}_ids.txt"),
    # "L-BFGS-B(50)": load_experiment(f"mpc-{name}-lbfgsb-mem50-Δ{delta}_ids.txt"),
    # "L-BFGS-B++(15)": load_experiment(f"mpc-{name}-lbfgspp-mem15-Δ{delta}_ids.txt"),
    # "L-BFGS-B++(50)": load_experiment(f"mpc-{name}-lbfgspp-mem50-Δ{delta}_ids.txt"),
    # "Struc. PANOC(70)": load_experiment(f"mpc-{name}-strucpanoc-mem70-Δ{delta}_ids.txt"),
    # "Struc. PANOC(90)": load_experiment(f"mpc-{name}-strucpanoc-mem90-Δ{delta}_ids.txt"),
    # "Struc. PANOC(110)": load_experiment(f"mpc-{name}-strucpanoc-mem110-Δ{delta}_ids.txt"),
    # "Struc. PANOC(130)": load_experiment(f"mpc-{name}-strucpanoc-mem130-Δ{delta}_ids.txt"),
    # "Struc. PANOC(150)": load_experiment(f"mpc-{name}-strucpanoc-mem150-Δ{delta}_ids.txt"),
    # "Struc. PANOC(170)": load_experiment(f"mpc-{name}-strucpanoc-mem170-Δ{delta}_ids.txt"),
    # "PANOC(50)": load_experiment(f"mpc-{name}-panoc-mem50-Δ{delta}_ids.txt"),
    # "PANOC(150)": load_experiment(f"mpc-{name}-panoc-mem150-Δ{delta}_ids.txt"),
    # "PATRINOC": load_experiment(f"mpc-{name}-true-newton-tr-Δ{delta}_ids.txt"),
    # "Ipopt (SS)": load_experiment(f"mpc-{name}-ipopt-precompiled-{'warm' if warm else 'cold'}-Δ{delta}_ids.txt"),
    # "Ipopt (SS, warm)": load_experiment(f"mpc-{name}-ipopt-precompiled-warm-Δ{delta}_ids.txt"),
    # "Ipopt (MS)": load_experiment(f"mpc-{name}-ipopt-precompiled-ms-Δ{delta}_ids.txt"),
    # "Ipopt (MS, warm)": load_experiment(f"mpc-{name}-ipopt-precompiled-ms-warm-Δ{delta}_ids.txt"),
    "Ipopt (SS, VM)": load_experiment(f"mpc-{name}-ipopt-{'warm' if warm else 'cold'}-Δ{delta}_ids.txt"),
    "PANOC GN": load_experiment(f"mpc-{name}-gauss-newton-lqr-{'warm' if warm else 'cold'}-Δ{delta}_ids.txt")
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

    plt.figure()
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

    plt.figure()
    plt.title(metric_names[metric])
    for t, vld, invld, lbl in zip(t_sp_orig, valid, invalid, all_results):
        ind = np.arange(1, len(t) + 1)
        p, = plt.semilogy(ind[vld], t[vld], '.', label=lbl)
        plt.semilogy(ind[invld], t[invld], 'x', color=p.get_color())
        plt.semilogy(ind, t, ':', linewidth=1, color=p.get_color())
    plt.xlabel("Problem")
    plt.legend()
    plt.tight_layout()

for m in metrics:
    do_perf_plot(m)

metric = "med_runtime"
plt.figure()
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
plt.xlabel("Problem")
plt.legend()
plt.tight_layout()

metric = "med_runtime"
plt.figure()
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
plt.xlabel("Problem")
plt.legend()
plt.tight_layout()


plt.figure()
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
plt.legend()
plt.tight_layout()

plt.show()
