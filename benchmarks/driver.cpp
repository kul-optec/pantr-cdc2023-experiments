#include <alpaqa/alm-fbe-trust-tr.hpp>
#include <alpaqa/alm-panoc-tr.hpp>
#include <alpaqa/config/config.hpp>
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
#include "results.hpp"
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

template <class InnerSolver, class Accelerator>
auto make_solver(const auto &extra_opts) {
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

template <class Solver>
void do_experiment(auto &problem, Solver &solver,
                   std::span<const std::string_view> extra_opts) {
    // Initial guess
    vec x = problem.initial_guess, y = problem.y, err_z(problem.m);

    auto evals = problem.evaluations;
    std::vector<real_t> gr_eval, it_time;
    std::vector<real_t> fbe, norm_q, eps, gamma, tau, ratio, radius;
    auto t0       = std::chrono::steady_clock::now();
    auto callback = [&](const typename Solver::ProgressInfo &i) {
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
    solver.set_progress_callback(callback);

    // Solve the problem
    auto stats =
        solver(problem.problem, {.tolerance = 1e-8}, x, y, problem.μ, err_z);

    // Print the results
    evals = std::make_shared<alpaqa::EvalCounter>(*evals);
    std::cout << '\n' << *evals << '\n';
    vec g(problem.m);
    if (problem.m > 0)
        problem.problem.eval_g(x, g);
    auto f      = problem.problem.eval_f(x);
    auto δ      = err_z.lpNorm<Eigen::Infinity>();
    auto time_s = std::chrono::duration<double>(stats.elapsed_time).count();
    std::cout << "solver:  " << solver.get_name() << '\n'
              << "problem: " << problem.path.filename().c_str() << " "
              << problem.horizon << " (from " << problem.path.parent_path()
              << ")" << '\n'
              << "status:  " << stats.status << '\n'
              << "f = " << alpaqa::float_to_str(f) << '\n'
              << "ε = " << alpaqa::float_to_str(stats.ε) << '\n'
              << "δ = " << alpaqa::float_to_str(δ) << '\n'
              << "γ = " << alpaqa::float_to_str(stats.final_γ) << '\n'
              << "time: " << alpaqa::float_to_str(time_s, 3) << " s\n"
              << "iter:      " << std::setw(6) << stats.iterations << '\n'
              << "backtrack: " << std::setw(6) << stats.linesearch_backtracks
              << '\n'
              << "+ _______: " << std::setw(6)
              << (stats.iterations + stats.linesearch_backtracks) << '\n'
              << "dir fail: " << std::setw(6) << stats.lbfgs_failures << '\n'
              << "dir rej:  " << std::setw(6) << stats.lbfgs_rejected << '\n'
              << std::endl;

    // Write results to file
    auto now_ms        = timestamp_ms<std::chrono::system_clock>();
    auto timestamp_str = std::to_string(now_ms.count());
    auto rnd_str       = random_hex_string(std::random_device());
    auto suffix        = timestamp_str + '_' + rnd_str;
    auto results_name  = "results_" + suffix;
    std::cout << "results: " << suffix << std::endl;
    std::ofstream res_file{results_name + ".py"};
    write_results(
        res_file,
        {
            .problem     = problem,
            .evals       = *evals,
            .duration    = stats.elapsed_time,
            .status      = enum_name(stats.status),
            .success     = stats.status == alpaqa::SolverStatus::Converged,
            .solver      = solver.get_name(),
            .f           = f,
            .δ           = δ,
            .ε           = stats.ε,
            .γ           = stats.final_γ,
            .solution    = x,
            .multipliers = y,
            .outer_iter  = 1,
            .inner_iter  = stats.iterations,
            .extra =
                {
                    {"grad_eval", gr_eval},
                    {"it_time", it_time},
                    {"fbe", fbe},
                    {"norm_q", norm_q},
                    {"eps", eps},
                    {"gamma", gamma},
                    {"tau", tau},
                    {"ratio", ratio},
                    {"radius", radius},
                    {"linesearch_failures", stats.linesearch_failures},
                    {"linesearch_backtracks", stats.linesearch_backtracks},
                    {"stepsize_backtracks", stats.stepsize_backtracks},
                    {"lbfgs_failures", stats.lbfgs_failures},
                    {"lbfgs_rejected", stats.lbfgs_rejected},
                },
            .options   = extra_opts,
            .timestamp = now_ms.count(),
        });
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

    // Check penalty factor
    auto pen_it = std::ranges::find_if(extra_opts, [](std::string_view sv) {
        return sv.starts_with("penalty=");
    });
    if (pen_it != extra_opts.end()) {
        pen_it->remove_prefix(pen_it->find('=') + 1);
        extra_opts.erase(pen_it);
        double penalty      = 1;
        const auto *val_end = pen_it->data() + pen_it->size();
        auto [ptr, ec]      = std::from_chars(pen_it->data(), val_end, penalty);
        if (ec != std::errc())
            throw std::invalid_argument("Invalid value '" +
                                        std::string(*pen_it) +
                                        "' for type 'double' in 'penalty': " +
                                        std::make_error_code(ec).message());
        if (ptr != val_end)
            throw std::invalid_argument("Invalid suffix '" +
                                        std::string(ptr, val_end) +
                                        "' for type 'double' in 'penalty'");
        problem.μ.setConstant(penalty);
    }

    // Run experiment
    if (solver == "fbetrust") {
        using Accelerator = alpaqa::TRDirection<config_t>;
        using InnerSolver = alpaqa::FBETrustSolver<Accelerator>;
        auto solver       = make_solver<InnerSolver, Accelerator>(extra_opts);
        do_experiment(problem, solver, extra_opts);
    } else if (solver == "strucpanoc") {
        using Accelerator = alpaqa::StructuredLBFGSDirection<config_t>;
        using InnerSolver = alpaqa::PANOCSolver<Accelerator>;
        auto solver       = make_solver<InnerSolver, Accelerator>(extra_opts);
        do_experiment(problem, solver, extra_opts);
    } else if (solver == "newtpanoc") {
        using Accelerator = alpaqa::StructuredNewtonDirection<config_t>;
        using InnerSolver = alpaqa::PANOCSolver<Accelerator>;
        auto solver       = make_solver<InnerSolver, Accelerator>(extra_opts);
        do_experiment(problem, solver, extra_opts);
    } else if (solver == "panoc") {
        using Accelerator = alpaqa::LBFGSDirection<config_t>;
        using InnerSolver = alpaqa::PANOCSolver<Accelerator>;
        auto solver       = make_solver<InnerSolver, Accelerator>(extra_opts);
        do_experiment(problem, solver, extra_opts);
    } else if (solver == "struczfpr") {
        using Accelerator = alpaqa::StructuredLBFGSDirection<config_t>;
        using InnerSolver = alpaqa::ZeroFPRSolver<Accelerator>;
        auto solver       = make_solver<InnerSolver, Accelerator>(extra_opts);
        do_experiment(problem, solver, extra_opts);
    } else if (solver == "zfpr") {
        using Accelerator = alpaqa::LBFGSDirection<config_t>;
        using InnerSolver = alpaqa::ZeroFPRSolver<Accelerator>;
        auto solver       = make_solver<InnerSolver, Accelerator>(extra_opts);
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
