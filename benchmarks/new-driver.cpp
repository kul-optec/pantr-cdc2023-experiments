#include <alpaqa/alm-fbe-trust-tr.hpp>
#include <alpaqa/alm-panoc-tr.hpp>
#include <alpaqa/alm-true-fbe-trust-tr.hpp>
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
#include "results.hpp"
#include <casadi-dll-wrapper.hpp>

#if WITH_IPOPT
#include <alpaqa/ipopt-adapter.hpp>
#include "ipopt-helpers.hpp"
#include <IpIpoptApplication.hpp>
#endif

#if WITH_LBFGSPP
#include <alpaqa/lbfgsb-adapter.hpp>
#endif

#if WITH_LBFGSB
#include <alpaqa/lbfgsb/lbfgsb-adapter.hpp>
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
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
              << " [<path>/]<name> <horizon> <second-order> [method=<solver>] "
                 "[<key>=<value>...]\n";
}

void set_specific_params(auto &) {}
template <alpaqa::Config Conf>
void set_specific_params(alpaqa::FBETrustParams<Conf> &params) {
    params.update_lbfgs_on_prox_step                  = true;
    params.recompute_last_prox_step_after_lbfgs_flush = false;
}
template <alpaqa::Config Conf>
void set_specific_params(alpaqa::StructuredLBFGSDirectionParams<Conf> &params) {
    params.hessian_vec = false;
}

USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);

template <class InnerSolver>
auto make_inner_solver(std::span<const std::string_view> extra_opts) {
    // Settings for the solver
    typename InnerSolver::Params solver_param;
    solver_param.max_iter       = 50'000;
    solver_param.print_interval = 0;
    solver_param.stop_crit      = alpaqa::PANOCStopCrit::FPRNorm;
    set_specific_params(solver_param);
    alpaqa::set_params(solver_param, "solver", extra_opts);

    if constexpr (requires { typename InnerSolver::Direction; }) {
        // Settings for the direction provider
        using Accelerator = typename InnerSolver::Direction;
        typename Accelerator::DirectionParams dir_param;
        set_specific_params(dir_param);
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
    } else {
        return InnerSolver{solver_param};
    }
}

#if WITH_LBFGSPP
using InnerLBFGSppSolver = alpaqa::lbfgspp::LBFGSBSolver<alpaqa::DefaultConfig>;
template <>
auto make_inner_solver<InnerLBFGSppSolver>(
    std::span<const std::string_view> extra_opts) {
    // Settings for the solver
    InnerLBFGSppSolver::Params solver_param;
    solver_param.max_iterations = 50'000;
    set_specific_params(solver_param);
    alpaqa::set_params(solver_param, "solver", extra_opts);
    return InnerLBFGSppSolver{solver_param};
}
#endif

#if WITH_LBFGSB
using InnerLBFGSBSolver = alpaqa::lbfgsb::LBFGSBSolver;
template <>
auto make_inner_solver<InnerLBFGSBSolver>(
    std::span<const std::string_view> extra_opts) {
    // Settings for the solver
    InnerLBFGSBSolver::Params solver_param;
    solver_param.max_iter       = 50'000;
    solver_param.print_interval = 0;
    solver_param.stop_crit      = alpaqa::PANOCStopCrit::ProjGradUnitNorm;
    set_specific_params(solver_param);
    alpaqa::set_params(solver_param, "solver", extra_opts);
    return InnerLBFGSBSolver{solver_param};
}
#endif

template <class InnerSolver>
auto make_solver(const auto &extra_opts) {
    // Settings for the ALM solver
    using ALMSolver = alpaqa::ALMSolver<InnerSolver>;
    typename ALMSolver::Params alm_param;
    alm_param.max_iter        = 200;
    alm_param.ε               = 1e-8;
    alm_param.δ               = 1e-8;
    alm_param.print_interval  = 1;
    alm_param.print_precision = 1;
    alpaqa::set_params(alm_param, "alm", extra_opts);
    return ALMSolver{alm_param, make_inner_solver<InnerSolver>(extra_opts)};
}

template <class Solver>
BenchmarkResults
do_experiment_impl(auto &problem, Solver &solver,
                   std::span<const std::string_view> extra_opts) {
    // Initial guess
    vec x = problem.initial_guess, y = problem.y;

    // Gather information about intermediate iterates
    auto evals = problem.evaluations;
    std::vector<real_t> gr_eval, it_time;
    std::vector<real_t> fbe, norm_q, eps, gamma, tau, ratio, radius;
    auto t0 = std::chrono::steady_clock::now();
    if constexpr (requires { typename Solver::InnerSolver::ProgressInfo; }) {
        auto callback =
            [&](const typename Solver::InnerSolver::ProgressInfo &i) {
                auto tk = std::chrono::steady_clock::now();
                if (i.k == 0)
                    t0 = tk;
                it_time.push_back(
                    std::chrono::duration<double>(tk - t0).count());
                gr_eval.push_back(static_cast<real_t>(
                    evals->grad_f + evals->f_grad_f + evals->f_grad_f_g +
                    evals->grad_f_grad_g_prod + evals->grad_g_prod +
                    evals->grad_gi + evals->hess_L_prod + evals->hess_ψ_prod +
                    evals->hess_L + evals->hess_ψ + evals->grad_ψ +
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
    }

    // Solve the problem
    auto stats = solver(problem.problem, x, y);
    evals      = std::make_shared<alpaqa::EvalCounter>(*evals);

    // Solve the problems again to average runtimes
    auto avg_duration = stats.elapsed_time;
    length_t N_exp    = 1;
    alpaqa::set_params(N_exp, "num_exp", extra_opts);
    std::cout.setstate(std::ios_base::badbit);
    for (index_t i = 0; i < N_exp - 1; ++i) {
        vec x = problem.initial_guess, y = problem.y;
        avg_duration += solver(problem.problem, x, y).elapsed_time;
    }
    std::cout.clear();
    avg_duration /= N_exp;

    // Results
    auto now_ms    = timestamp_ms<std::chrono::system_clock>();
    auto f         = problem.problem.eval_f(x);
    auto kkt_err   = compute_kkt_error(problem.problem, x, y);
    real_t final_γ = 0;
    if constexpr (requires { stats.inner.final_γ; })
        final_γ = stats.inner.final_γ;
    decltype(BenchmarkResults::extra) extra{
        {"grad_eval", gr_eval}, {"it_time", it_time}, {"fbe", fbe},
        {"norm_q", norm_q},     {"eps", eps},         {"gamma", gamma},
        {"tau", tau},           {"ratio", ratio},     {"radius", radius},
    };
    if constexpr (requires { stats.inner.linesearch_failures; })
        extra.emplace_back("linesearch_failures",
                           stats.inner.linesearch_failures);
    if constexpr (requires { stats.inner.linesearch_backtracks; })
        extra.emplace_back("linesearch_backtracks",
                           stats.inner.linesearch_backtracks);
    if constexpr (requires { stats.inner.stepsize_backtracks; })
        extra.emplace_back("stepsize_backtracks",
                           stats.inner.stepsize_backtracks);
    if constexpr (requires { stats.inner.lbfgs_failures; })
        extra.emplace_back("lbfgs_failures", stats.inner.lbfgs_failures);
    if constexpr (requires { stats.inner.lbfgs_rejected; })
        extra.emplace_back("lbfgs_rejected", stats.inner.lbfgs_rejected);
    return BenchmarkResults{
        .problem          = problem,
        .evals            = *evals,
        .duration         = avg_duration,
        .status           = enum_name(stats.status),
        .success          = stats.status == alpaqa::SolverStatus::Converged,
        .solver           = solver.get_name(),
        .f                = f,
        .δ                = stats.δ,
        .ε                = stats.ε,
        .γ                = final_γ,
        .Σ                = stats.norm_penalty,
        .stationarity     = kkt_err.stationarity,
        .constr_violation = kkt_err.constr_violation,
        .complementarity  = kkt_err.complementarity,
        .solution         = x,
        .multipliers      = y,
        .outer_iter       = stats.outer_iterations,
        .inner_iter       = stats.inner.iterations,
        .extra            = std::move(extra),
        .options          = extra_opts,
        .timestamp        = now_ms.count(),
    };
}

#if WITH_IPOPT

auto make_ipopt_solver(std::span<const std::string_view> extra_opts) {
    using namespace Ipopt;

    // We are using the factory, since this allows us to compile this
    // example with an Ipopt Windows DLL
    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    app->RethrowNonIpoptException(true);

    app->Options()->SetNumericValue("tol", 1e-8);
    app->Options()->SetNumericValue("constr_viol_tol", 1e-8);
    app->Options()->SetStringValue("linear_solver", "mumps");
    app->Options()->SetStringValue("print_timing_statistics", "yes");
    app->Options()->SetStringValue("timing_statistics", "yes");
    app->Options()->SetStringValue("hessian_approximation", "exact");

    alpaqa::set_params(*app, "solver", extra_opts);

    // Initialize the IpoptApplication and process the options
    ApplicationReturnStatus status = app->Initialize();
    if (status != Solve_Succeeded)
        throw std::runtime_error("Error during Ipopt initialization");

    return app;
}

BenchmarkResults
do_experiment_impl(auto &problem,
                   Ipopt::SmartPtr<Ipopt::IpoptApplication> &solver,
                   std::span<const std::string_view> extra_opts) {
    // Ipopt problem adapter
    using Problem                    = alpaqa::IpoptAdapter;
    Ipopt::SmartPtr<Ipopt::TNLP> nlp = new Problem(problem.problem);
    auto *my_nlp                     = dynamic_cast<Problem *>(GetRawPtr(nlp));

    // Initial guess
    my_nlp->initial_guess = problem.initial_guess;

    // Solve the problem
    auto t0     = std::chrono::steady_clock::now();
    auto status = solver->OptimizeTNLP(nlp);
    auto t1     = std::chrono::steady_clock::now();
    auto evals  = *problem.evaluations;

    // Solve the problems again to average runtimes
    using ns          = std::chrono::nanoseconds;
    auto avg_duration = duration_cast<ns>(t1 - t0);
    length_t N_exp    = 4;
    alpaqa::set_params(N_exp, "num_exp", extra_opts);
    std::cout.setstate(std::ios_base::badbit);
    for (index_t i = 0; i < N_exp - 1; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        solver->OptimizeTNLP(nlp);
        auto t1 = std::chrono::steady_clock::now();
        avg_duration += duration_cast<ns>(t1 - t0);
    }
    std::cout.clear();
    avg_duration /= N_exp;

    // Results
    auto &nlp_res = my_nlp->results;
    auto kkt_err  = compute_kkt_error(problem.problem, nlp_res.solution_x,
                                      nlp_res.solution_y);
    auto now_ms   = timestamp_ms<std::chrono::system_clock>();
    BenchmarkResults results{
        .problem          = problem,
        .evals            = evals,
        .duration         = avg_duration,
        .status           = enum_name(status),
        .success          = status == Ipopt::Solve_Succeeded,
        .solver           = "Ipopt",
        .f                = nlp_res.solution_f,
        .δ                = nlp_res.infeasibility,
        .ε                = nlp_res.nlp_error,
        .γ                = 0,
        .Σ                = 0,
        .stationarity     = kkt_err.stationarity,
        .constr_violation = kkt_err.constr_violation,
        .complementarity  = kkt_err.complementarity,
        .solution         = nlp_res.solution_x,
        .multipliers      = nlp_res.solution_y,
        .outer_iter       = nlp_res.iter_count,
        .inner_iter       = nlp_res.iter_count,
        .extra            = {{"grad_eval", vec{}},
                             {"it_time", vec{}},
                             {"fbe", vec{}},
                             {"norm_q", vec{}},
                             {"eps", vec{}},
                             {"gamma", vec{}},
                             {"tau", vec{}},
                             {"ratio", vec{}},
                             {"radius", vec{}}},
        .options          = extra_opts,
        .timestamp        = now_ms.count(),
    };
    return results;
}

#endif

void do_experiment(auto &problem, auto &solver,
                   std::span<const std::string_view> extra_opts) {
    // Run experiment
    BenchmarkResults results = do_experiment_impl(problem, solver, extra_opts);

    // Print results to output
    print_results(std::cout, results);

    // Write results to file
    auto timestamp_str = std::to_string(results.timestamp);
    auto rnd_str       = random_hex_string(std::random_device());
    auto suffix        = timestamp_str + '_' + rnd_str;
    auto results_name  = "results_" + suffix;
    std::cout << "results: " << suffix << std::endl;
    std::ofstream res_file{results_name + ".py"};
    write_results(res_file, results);
}

int main(int argc, char *argv[]) try {
    // Find the problem to load
    if (argc < 1)
        return -1;
    if (argc == 1)
        return print_usage(argv[0]), 0;
    if (argc < 4)
        return print_usage(argv[0]), -1;
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
    std::string_view solver_name = "fbetrust";
    alpaqa::set_params(solver_name, "method", extra_opts);

    // Available solvers
    std::map<std::string_view, std::function<void()>> solvers {
        {"fbetrust",
         [&] {
             using Accelerator = alpaqa::TRDirection<config_t>;
             using InnerSolver = alpaqa::FBETrustSolver<Accelerator>;
             auto solver       = make_solver<InnerSolver>(extra_opts);
             do_experiment(problem, solver, extra_opts);
         }},
            {"truefbetrust",
             [&] {
                 using InnerSolver = alpaqa::TrueFBETrustSolver<config_t>;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, solver, extra_opts);
             }},
            {"strucpanoc",
             [&] {
                 using Accelerator = alpaqa::StructuredLBFGSDirection<config_t>;
                 using InnerSolver = alpaqa::PANOCSolver<Accelerator>;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, solver, extra_opts);
             }},
            {"newtpanoc",
             [&] {
                 using Accelerator =
                     alpaqa::StructuredNewtonDirection<config_t>;
                 using InnerSolver = alpaqa::PANOCSolver<Accelerator>;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, solver, extra_opts);
             }},
            {"panoc",
             [&] {
                 using Accelerator = alpaqa::LBFGSDirection<config_t>;
                 using InnerSolver = alpaqa::PANOCSolver<Accelerator>;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, solver, extra_opts);
             }},
            {"struczfpr",
             [&] {
                 using Accelerator = alpaqa::StructuredLBFGSDirection<config_t>;
                 using InnerSolver = alpaqa::ZeroFPRSolver<Accelerator>;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, solver, extra_opts);
             }},
            {"zfpr",
             [&] {
                 using Accelerator = alpaqa::LBFGSDirection<config_t>;
                 using InnerSolver = alpaqa::ZeroFPRSolver<Accelerator>;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, solver, extra_opts);
             }},
#if WITH_IPOPT
            {"ipopt",
             [&] {
                 auto solver = make_ipopt_solver(extra_opts);
                 do_experiment(problem, solver, extra_opts);
             }},
#endif
#if WITH_LBFGSPP
            {"lbfgspp",
             [&] {
                 using InnerSolver = InnerLBFGSppSolver;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, solver, extra_opts);
             }},
#endif
#if WITH_LBFGSB
            {"lbfgsb", [&] {
                 using InnerSolver = InnerLBFGSBSolver;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, solver, extra_opts);
             }},
#endif
    };

    // Run experiment
    auto solver_it = solvers.find(solver_name);
    if (solver_it != solvers.end())
        solver_it->second();
    else
        throw std::invalid_argument(
            "Unknown solver '" + std::string(solver_name) + "'\n" +
            "  Available solvers: " + [&] {
                if (solvers.empty())
                    return std::string{};
                auto penult       = std::prev(solvers.end());
                auto quote_concat = [](std::string &&a, auto b) {
                    return a + "'" + std::string(b.first) + "', ";
                };
                return std::accumulate(solvers.begin(), penult, std::string{},
                                       quote_concat) +
                       "'" + std::string(penult->first) + "'";
            }());
} catch (std::exception &e) {
    std::cerr << "Error: " << demangled_typename(typeid(e)) << ":\n  "
              << e.what() << std::endl;
    return -1;
}
