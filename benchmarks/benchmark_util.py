from os.path import join, dirname, realpath
import importlib.util

BUILD_TYPE = "Release"
benchmarks_folder = dirname(__file__)
build_folder = "..", "..", "build", "examples", "benchmarks"
results_path = realpath(join(benchmarks_folder, *build_folder))


def load_results(name, build_type=BUILD_TYPE):
    name = f"results_{name}"
    mod_file = join(results_path, build_type, name + ".py")
    spec = importlib.util.spec_from_file_location("", mod_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.results


def load_experiment(name, build_type=BUILD_TYPE):
    exp_file = join(results_path, build_type, name)
    with open(exp_file) as file:
        return [load_results(line.rstrip(), build_type) for line in file]


def total_evals(results):
    return (
        results.get("f", 0)
        + results.get("grad_f", 0)
        + results.get("f_grad_f", 0)
        + results.get("ψ", 0)
        + results.get("grad_ψ", 0)
        + results.get("grad_ψ_from_ŷ", 0)
        + results.get("ψ_grad_ψ", 0)
        + results.get("hess_ψ", 0)
        + results.get("hess_ψ_prod", 0)
        + results.get("grad_L", 0)
        + results.get("hess_L", 0)
        + results.get("hess_L_prod", 0)
        + results.get("prox_grad_step", 0)
    )
