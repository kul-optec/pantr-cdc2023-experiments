from os.path import join
import importlib.util

def load_results(path, name):
    mod_file = join(path, f"results_{name}.py")
    spec = importlib.util.spec_from_file_location("", mod_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.results


def load_experiment(path, name):
    exp_file = join(path, name)
    with open(exp_file) as file:
        return [load_results(path, line.rstrip()) for line in file]


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
