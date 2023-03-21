#include <alpaqa/alm-fbe-trust-tr.hpp>
#include <alpaqa/alm-panoc-tr.hpp>
#include <alpaqa/config/config.hpp>
#include <alpaqa/implementation/outer/alm.tpp>
#include <alpaqa/inner/directions/panoc/lbfgs.hpp>
#include <alpaqa/inner/directions/panoc/structured-lbfgs.hpp>
#include <alpaqa/inner/directions/panoc/structured-newton.hpp>
#include <alpaqa/inner/directions/panoc/tr.hpp>
#include <alpaqa/inner/fbe-trust.hpp>
#include <alpaqa/inner/panoc.hpp>
#include <alpaqa/inner/zerofpr.hpp>
#include <alpaqa/structured-panoc-alm.hpp>
#include <alpaqa/util/demangled-typename.hpp>
#include <alpaqa/util/print.hpp>

#include "output.hpp"
#include "params.hpp"
#include <casadi-dll-wrapper.hpp>

#include <bit>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace fs = std::filesystem;

void print_usage(const char *a0) {
    std::cout << "Usage: " << a0
              << " [<path>/]<name> <horizon> <second-order> [solver=<solver>] "
                 "[<key>=<value>...]\n";
}

void set_params(auto &) {}
template <alpaqa::Config Conf>
void set_params(alpaqa::FBETrustParams<Conf> &params) {
    params.update_lbfgs_on_prox_step                  = true;
    params.recompute_last_prox_step_after_lbfgs_flush = false;
}
template <alpaqa::Config Conf>
void set_params(alpaqa::StructuredLBFGSDirectionParams<Conf> &params) {
    params.hessian_vec = false;
}

USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);

template <class InnerSolver>
auto make_inner_solver(const auto &extra_opts) {
    using Accelerator = typename InnerSolver::Direction;
    // Settings for the solver
    typename InnerSolver::Params solver_param;
    solver_param.max_iter       = 50'000;
    solver_param.print_interval = 500;
    solver_param.stop_crit      = alpaqa::PANOCStopCrit::FPRNorm2;
    set_params(solver_param);
    alpaqa::set_params(solver_param, "solver", extra_opts);
    // Settings for the direction provider
    typename Accelerator::DirectionParams dir_param;
    set_params(dir_param);
    alpaqa::set_params(dir_param, "dir", extra_opts);
    if constexpr (requires { typename Accelerator::LBFGSParams; }) {
        // Settings for the L-BFGS accelerator
        typename Accelerator::LBFGSParams lbfgs_param;
        lbfgs_param.memory = 5;
        if (std::is_same_v<Accelerator, alpaqa::TRDirection<config_t>>)
            lbfgs_param.powell.damped = true;
        alpaqa::set_params(lbfgs_param, "lbfgs", extra_opts);
        return InnerSolver{solver_param, {lbfgs_param, dir_param}};
    } else {
        return InnerSolver{solver_param, {dir_param}};
    }
}

template <class InnerSolver>
auto make_solver(const auto &extra_opts, length_t penalty_alm_split) {
    // Settings for the ALM solver
    using ALMSolver = alpaqa::ALMSolver<InnerSolver>;
    typename ALMSolver::Params alm_param;
    alm_param.max_iter          = 100;
    alm_param.ε                 = 1e-8;
    alm_param.δ                 = 1e-8;
    alm_param.print_interval    = 1;
    alm_param.penalty_alm_split = penalty_alm_split;
    alpaqa::set_params(alm_param, "alm", extra_opts);
    return ALMSolver{alm_param, make_inner_solver<InnerSolver>(extra_opts)};
}

template <class Solver>
void do_experiment(auto &problem, Solver &solver,
                   std::span<const std::string_view> extra_opts) {
    // Initial guess
    vec x = problem.initial_guess, y = problem.y;

    auto evals = problem.evaluations;
    std::vector<real_t> gr_eval, it_time;
    std::vector<real_t> fbe, norm_q, eps, gamma, tau, ratio, radius;
    auto t0       = std::chrono::steady_clock::now();
    auto callback = [&](const typename Solver::InnerSolver::ProgressInfo &i) {
        auto tk = std::chrono::steady_clock::now();
        if (i.k == 0)
            t0 = tk;
        it_time.push_back(std::chrono::duration<double>(tk - t0).count());
        gr_eval.push_back(static_cast<real_t>(
            evals->grad_f + evals->f_grad_f + evals->f_grad_f_g +
            evals->grad_f_grad_g_prod + evals->grad_g_prod + evals->grad_gi +
            evals->hess_L_prod + evals->hess_ψ_prod + evals->grad_ψ +
            evals->grad_ψ_from_ŷ + evals->ψ_grad_ψ));
        fbe.push_back(i.φγ);
        norm_q.push_back(i.q.norm());
        eps.push_back(i.ε);
        gamma.push_back(i.γ);
        if constexpr (requires { i.τ; })
            tau.push_back(i.τ);
        if constexpr (requires { i.ρ; })
            ratio.push_back(i.ρ);
        if constexpr (requires { i.δ; })
            radius.push_back(i.δ);
    };
    solver.inner_solver.set_progress_callback(callback);

    // Solve the problem
    auto stats = solver(problem.problem, x, y);

    // Print the results
    evals = std::make_shared<alpaqa::EvalCounter>(*evals);
    std::cout << '\n' << *evals << '\n';
    vec g(problem.m);
    if (problem.m > 0)
        problem.problem.eval_g(x, g);
    auto f      = problem.problem.eval_f(x);
    auto time_s = std::chrono::duration<double>(stats.elapsed_time).count();
    std::cout << "solver:  " << solver.get_name() << '\n'
              << "problem: " << problem.path.filename().c_str() << " "
              << problem.horizon << " (from " << problem.path.parent_path()
              << ")" << '\n'
              << "status:  " << stats.status << '\n'
              << "num var: " << problem.problem.get_n() << '\n'
              << "num con: " << problem.problem.get_m() << '\n'
              << "f = " << alpaqa::float_to_str(f) << '\n'
              << "ε = " << alpaqa::float_to_str(stats.ε) << '\n'
              << "δ = " << alpaqa::float_to_str(stats.δ) << '\n'
              << "γ = " << alpaqa::float_to_str(stats.inner.final_γ) << '\n'
              << "Σ = " << alpaqa::float_to_str(stats.norm_penalty) << '\n'
              << "time: " << alpaqa::float_to_str(time_s, 3) << " s\n"
              << "alm iter:  " << std::setw(6) << stats.outer_iterations << '\n'
              << "iter:      " << std::setw(6) << stats.inner.iterations << '\n'
              << "backtrack: " << std::setw(6)
              << stats.inner.linesearch_backtracks << '\n'
              << "+ _______: " << std::setw(6)
              << (stats.inner.iterations + stats.inner.linesearch_backtracks)
              << '\n'
              << "dir fail: " << std::setw(6) << stats.inner.lbfgs_failures
              << '\n'
              << "dir rej:  " << std::setw(6) << stats.inner.lbfgs_rejected
              << '\n'
              << std::endl;

    // Write results to file
    using ms     = std::chrono::milliseconds;
    auto now     = std::chrono::system_clock::now();
    auto now_ms  = std::chrono::duration_cast<ms>(now.time_since_epoch());
    auto gen     = std::random_device();
    auto rnd     = std::uniform_int_distribution<uint32_t>()(gen);
    auto rnd_str = std::string(8, '0');
    std::to_chars(rnd_str.data() + std::countl_zero(rnd) / 4,
                  rnd_str.data() + rnd_str.size(), rnd, 16);
    auto results_name =
        "results_" + std::to_string(now_ms.count()) + '_' + rnd_str;
    std::cout << "results: " << results_name << ' ' << now_ms.count() << '_'
              << rnd_str << '\n'
              << std::endl;
    std::ofstream res_file{results_name + ".py"};
    auto dict_elem = [&res_file]<class... A>(std::string_view k, A &&...a) {
        res_file << "    " << std::quoted(k) << ": ";
        python_literal(res_file, std::forward<A>(a)...);
        res_file << ",\n";
    };
    res_file << "# " << results_name << "\n"
             << "from numpy import nan, inf\n"
                "import numpy as np\n"
                "__all__ = ['results']\n"
                "results = {\n";
    dict_elem("grad_eval", gr_eval);
    dict_elem("it_time", it_time);
    dict_elem("fbe", fbe);
    dict_elem("norm_q", norm_q);
    dict_elem("eps", eps);
    dict_elem("gamma", gamma);
    dict_elem("tau", tau);
    dict_elem("ratio", ratio);
    dict_elem("radius", radius);
    dict_elem("solution", x);
    res_file << std::quoted("evaluations") << ": {\n";
#define EVAL(name) dict_elem(#name, evals->name)
    EVAL(proj_diff_g);
    EVAL(proj_multipliers);
    EVAL(prox_grad_step);
    EVAL(f);
    EVAL(grad_f);
    EVAL(f_grad_f);
    EVAL(f_g);
    EVAL(f_grad_f_g);
    EVAL(grad_f_grad_g_prod);
    EVAL(g);
    EVAL(grad_g_prod);
    EVAL(grad_gi);
    EVAL(grad_L);
    EVAL(hess_L_prod);
    EVAL(hess_L);
    EVAL(hess_ψ_prod);
    EVAL(hess_ψ);
    EVAL(ψ);
    EVAL(grad_ψ);
    EVAL(grad_ψ_from_ŷ);
    EVAL(ψ_grad_ψ);
#undef EVAL
    res_file << "},\n";
    dict_elem("total_cost_eval", evals->ψ);
    dict_elem("total_grad_eval", evals->grad_ψ);
    dict_elem("total_cost_grad_eval", evals->ψ_grad_ψ);
    dict_elem("total_prox_eval", evals->prox_grad_step);
    dict_elem("runtime", stats.elapsed_time.count());
    dict_elem("iter", stats.inner.iterations);
    dict_elem("outer_iter", stats.outer_iterations);
    dict_elem("linesearch_failures", stats.inner.linesearch_failures);
    dict_elem("linesearch_backtracks", stats.inner.linesearch_backtracks);
    dict_elem("stepsize_backtracks", stats.inner.stepsize_backtracks);
    dict_elem("lbfgs_failures", stats.inner.lbfgs_failures);
    dict_elem("lbfgs_rejected", stats.inner.lbfgs_rejected);
    dict_elem("status", enum_name(stats.status));
    dict_elem("solver", solver.get_name());
    dict_elem("problem", problem.name);
    dict_elem("path", problem.full_path);
    dict_elem("horizon", problem.horizon);
    dict_elem("second_order", problem.second_order);
    dict_elem("formulation", problem.formulation);
    dict_elem("opts", extra_opts);
    dict_elem("time_utc_ms", now_ms.count());
    res_file << "}\n";
}

int main(int argc, char *argv[]) try {
    // Find the problem to load
    if (argc < 1)
        return -1;
    if (argc == 1)
        return print_usage(argv[0]), 0;
    if (argc < 4)
        return -1;
    fs::path path{argv[1]};
    if (!path.has_parent_path())
        path = fs::canonical(fs::path(argv[0])).parent_path() / path;
    if (!path.has_parent_path() || !path.has_filename())
        return (std::cerr << "invalid problem path\n"), -1;
    length_t horizon             = std::stoi(argv[2]);
    std::string_view formulation = argv[3];

    std::vector<std::string_view> extra_opts;
    std::copy(argv + 4, argv + argc, std::back_inserter(extra_opts));

    // Disable Constraints
    ConstraintsOpt constr_opt;
    auto unconstr_it = std::ranges::find(extra_opts, "no-collision-constr");
    if (unconstr_it != extra_opts.end()) {
        std::cout << "Disabling collision constraints\n";
        constr_opt.with_collision = false;
        extra_opts.erase(unconstr_it);
    }
    auto no_bounds_it = std::ranges::find(extra_opts, "no-bound-constr");
    if (no_bounds_it != extra_opts.end()) {
        std::cout << "Disabling bound constraints\n";
        constr_opt.with_bounds = false;
        extra_opts.erase(no_bounds_it);
    }

    // Load the problem
    auto problem = load_benchmark_problem(path.parent_path().c_str(),
                                          path.filename().c_str(), horizon,
                                          formulation, constr_opt);

#if 0 // No constraints
    problem.problem.C.lowerbound.setConstant(-alpaqa::inf<config_t>);
    problem.problem.C.upperbound.setConstant(+alpaqa::inf<config_t>);
#endif

    // Check which solver to use
    std::string_view solver = "fbetrust";
    auto solver_it = std::ranges::find_if(extra_opts, [](std::string_view sv) {
        return sv.starts_with("solver=");
    });
    if (solver_it != extra_opts.end()) {
        solver = *solver_it;
        solver.remove_prefix(solver.find('=') + 1);
        extra_opts.erase(solver_it);
    }

    // Check ALM split
    const auto penalty_alm_split = problem.penalty_alm_split;

    // Run experiment
    if (solver == "fbetrust") {
        using Accelerator = alpaqa::TRDirection<config_t>;
        using InnerSolver = alpaqa::FBETrustSolver<Accelerator>;
        auto solver = make_solver<InnerSolver>(extra_opts, penalty_alm_split);
        do_experiment(problem, solver, extra_opts);
    } else if (solver == "strucpanoc") {
        using Accelerator = alpaqa::StructuredLBFGSDirection<config_t>;
        using InnerSolver = alpaqa::PANOCSolver<Accelerator>;
        auto solver = make_solver<InnerSolver>(extra_opts, penalty_alm_split);
        do_experiment(problem, solver, extra_opts);
    } else if (solver == "newtpanoc") {
        using Accelerator = alpaqa::StructuredNewtonDirection<config_t>;
        using InnerSolver = alpaqa::PANOCSolver<Accelerator>;
        auto solver = make_solver<InnerSolver>(extra_opts, penalty_alm_split);
        do_experiment(problem, solver, extra_opts);
    } else if (solver == "panoc") {
        using Accelerator = alpaqa::LBFGSDirection<config_t>;
        using InnerSolver = alpaqa::PANOCSolver<Accelerator>;
        auto solver = make_solver<InnerSolver>(extra_opts, penalty_alm_split);
        do_experiment(problem, solver, extra_opts);
    } else if (solver == "struczfpr") {
        using Accelerator = alpaqa::StructuredLBFGSDirection<config_t>;
        using InnerSolver = alpaqa::ZeroFPRSolver<Accelerator>;
        auto solver = make_solver<InnerSolver>(extra_opts, penalty_alm_split);
        do_experiment(problem, solver, extra_opts);
    } else if (solver == "zfpr") {
        using Accelerator = alpaqa::LBFGSDirection<config_t>;
        using InnerSolver = alpaqa::ZeroFPRSolver<Accelerator>;
        auto solver = make_solver<InnerSolver>(extra_opts, penalty_alm_split);
        do_experiment(problem, solver, extra_opts);
    } else {
        throw std::invalid_argument("Unknown solver '" + std::string(solver) +
                                    "'");
    }
} catch (std::exception &e) {
    std::cerr << "Error: " << demangled_typename(typeid(e)) << ":\n  "
              << e.what() << std::endl;
    return -1;
}
