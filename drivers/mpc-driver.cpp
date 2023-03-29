#include <alpaqa/config/config.hpp>
#include <alpaqa/newton-tr-pantr-alm.hpp>
#include <alpaqa/panoc-alm.hpp>
#include <alpaqa/params/params.hpp>
#include <alpaqa/structured-panoc-alm.hpp>
#include <alpaqa/structured-zerofpr-alm.hpp>
#include <alpaqa/util/demangled-typename.hpp>
#include <alpaqa/util/print.hpp>
#include <alpaqa/zerofpr-alm.hpp>

#include "output.hpp"
#include "results.hpp"
#include <casadi-dll-wrapper.hpp>

#if WITH_IPOPT
#include <alpaqa/ipopt/ipopt-adapter.hpp>
#include <alpaqa/ipopt/ipopt-enums.hpp>
#include <IpIpoptApplication.hpp>
#endif

#if WITH_LBFGSB
#include "lbfgsb-alm.hpp"
#endif

#include <chrono>
#include <cmath>
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

USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);

template <class InnerSolver>
auto make_inner_solver(std::span<const std::string_view> extra_opts) {
    // Settings for the solver
    typename InnerSolver::Params solver_param;
    solver_param.max_iter       = 50'000;
    solver_param.print_interval = 0;
    solver_param.stop_crit      = alpaqa::PANOCStopCrit::FPRNorm;
    alpaqa::params::set_params(solver_param, "solver", extra_opts);

    if constexpr (requires { typename InnerSolver::Direction; }) {
        // Settings for the direction provider
        using Direction = typename InnerSolver::Direction;
        typename Direction::DirectionParams dir_param;
        typename Direction::AcceleratorParams accel_param;
        alpaqa::params::set_params(dir_param, "dir", extra_opts);
        alpaqa::params::set_params(accel_param, "accel", extra_opts);
        return InnerSolver{
            solver_param,
            Direction{{
                .accelerator = accel_param,
                .direction   = dir_param,
            }},
        };
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
    alpaqa::params::set_params(solver_param, "solver", extra_opts);
    return InnerLBFGSBSolver{solver_param};
}
#endif

template <class InnerSolver>
auto make_solver(const auto &extra_opts) {
    // Settings for the ALM solver
    using ALMSolver = alpaqa::ALMSolver<InnerSolver>;
    typename ALMSolver::Params alm_param;
    alm_param.max_iter       = 200;
    alm_param.ε              = 1e-8;
    alm_param.δ              = 1e-8;
    alm_param.print_interval = 0;
    alpaqa::params::set_params(alm_param, "alm", extra_opts);
    return ALMSolver{alm_param, make_inner_solver<InnerSolver>(extra_opts)};
}

struct MPCResults {
    mat states;
    mat inputs;
    vec runtimes;
};

MPCResults do_mpc_experiment(BenchmarkProblem &problem,
                             BenchmarkProblem &dl_problem, auto &solver,
                             std::span<const std::string_view> extra_opts) {
    rvec x0       = problem.initial_state();
    const auto nu = dl_problem.nu, nx = dl_problem.nx, nc = dl_problem.nc,
               nc_N = dl_problem.nc_N, N = dl_problem.horizon;
    if (x0.size() != nx)
        throw std::logic_error("Invalid x0 dimension");
    const auto n = problem.problem->get_n(), m = problem.problem->get_m();
    const bool ss = n == nu * N;
    const bool ms = n == nu * N + nx * N;
    assert(ss || ms);
    vec u_initial = problem.initial_guess;
    vec u         = u_initial;
    vec z_l       = vec::Zero(n);
    vec z_u       = vec::Zero(n);
    vec y         = vec::Zero(m);
    vec u_dist    = dl_problem.problem->get_box_C().upperbound.topRows(nu);
    for (real_t &ud : u_dist)
        if (!std::isfinite(ud))
            ud = 0;
    alpaqa::params::set_params(u_dist, "u_dist", extra_opts);
    if (u_dist.size() != nu)
        throw std::out_of_range("Invalid u_dist size");
    std::cout << "u_dist = " << u_dist.transpose() << '\n';
    length_t num_dist = 3;
    alpaqa::params::set_params(num_dist, "num_dist", extra_opts);
    length_t num_sim = N;
    alpaqa::params::set_params(num_sim, "num_sim", extra_opts);
    // Initial disturbance
    for (index_t i = 0; i < num_dist; ++i)
        x0 = dl_problem.simulate(x0, u_dist);
    // Initial solve
    bool warm_start = true;
    alpaqa::params::set_params(warm_start, "warm", extra_opts);
    if (warm_start)
        solver(u, y, z_l, z_u);
    // Simulate
    auto u_first = ss ? u.topRows(nu) : u.segment(nx, nu);
    x0           = dl_problem.simulate(x0, u_first);
    problem.evaluations->reset();
    vec runtimes = vec::Constant(num_sim, alpaqa::NaN<config_t>);
    mat states(nx, num_sim + 1);
    mat inputs(n, num_sim);
    for (index_t i = 0; i < num_sim; ++i) {
        // Shift previous solution for warm start
        if (warm_start) {
            if (ss) {
                u.topRows(n - nu)   = u.bottomRows(n - nu);
                z_l.topRows(n - nu) = z_l.bottomRows(n - nu);
                z_u.topRows(n - nu) = z_u.bottomRows(n - nu);
                if (nc != nc_N)
                    throw std::logic_error("invalid multiplier shift");
                y.topRows(m - nc_N) = y.bottomRows(m - nc_N);
            } else if (ms) {
                u.topRows(n - nx - nu)   = u.bottomRows(n - nx - nu);
                z_l.topRows(n - nx - nu) = z_l.bottomRows(n - nx - nu);
                z_u.topRows(n - nx - nu) = z_u.bottomRows(n - nx - nu);
                const auto m_dyn         = nx * N;
                const auto m_con         = m - m_dyn;
                if (nc != nc_N)
                    throw std::logic_error("invalid multiplier shift");
                if (m_con != nc * N + nc_N)
                    throw std::logic_error("invalid number of constraints");
                auto y_constr = y.topRows(m_con);
                auto y_dyn    = y.bottomRows(m_dyn);
                y_constr.topRows(m_con - nc_N) =
                    y_constr.bottomRows(m_con - nc_N);
                y_dyn.topRows(m_dyn - nx) = y_dyn.topRows(m_dyn - nx);
            }
        } else {
            u = u_initial;
            z_l.setZero();
            z_u.setZero();
            y.setZero();
        }
        // Solve
        auto runtime = solver(u, y, z_l, z_u);
        // Accumulate stats
        runtimes(i)   = std::chrono::duration<real_t>{runtime}.count();
        states.col(i) = x0;
        inputs.col(i) = u;
        // Simulate
        x0 = dl_problem.simulate(x0, u_first);
    }
    states.col(num_sim) = x0;

    return {.states = states, .inputs = inputs, .runtimes = runtimes};
}

template <class Solver>
MPCResults do_experiment_impl(BenchmarkProblem &problem,
                              BenchmarkProblem &dl_problem, Solver &solver,
                              std::span<const std::string_view> extra_opts) {
    auto solve = [&](rvec u, rvec y, const rvec &, const rvec &) {
        auto stats = solver(*problem.problem, u, y);
        return stats.status == alpaqa::SolverStatus::Converged
                   ? stats.elapsed_time
                   : -stats.elapsed_time;
    };
    return do_mpc_experiment(problem, dl_problem, solve, extra_opts);
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
    app->Options()->SetIntegerValue("print_level", 0);
    // app->Options()->SetStringValue("print_timing_statistics", "yes");
    // app->Options()->SetStringValue("timing_statistics", "yes");
    app->Options()->SetStringValue("hessian_approximation", "exact");
    // app->Options()->SetStringValue("warm_start_init_point", "yes");

    alpaqa::params::set_params(*app, "solver", extra_opts);

    // Initialize the IpoptApplication and process the options
    ApplicationReturnStatus status = app->Initialize();
    if (status != Solve_Succeeded)
        throw std::runtime_error("Error during Ipopt initialization");

    return app;
}

MPCResults do_experiment_impl(BenchmarkProblem &problem,
                              BenchmarkProblem &dl_problem,
                              Ipopt::SmartPtr<Ipopt::IpoptApplication> &solver,
                              std::span<const std::string_view> extra_opts) {
    // Ipopt problem adapter
    using Problem                    = alpaqa::IpoptAdapter;
    Ipopt::SmartPtr<Ipopt::TNLP> nlp = new Problem(*problem.problem);
    auto *my_nlp                     = dynamic_cast<Problem *>(GetRawPtr(nlp));
    bool first                       = true;
    auto solve                       = [&](rvec u, rvec y, rvec z_L, rvec z_U) {
        my_nlp->initial_guess                      = u;
        my_nlp->initial_guess_multipliers          = y;
        my_nlp->initial_guess_bounds_multipliers_l = z_L;
        my_nlp->initial_guess_bounds_multipliers_l = z_U;
        auto t0 = std::chrono::steady_clock::now();
        auto status =
            first ? solver->OptimizeTNLP(nlp) : solver->ReOptimizeTNLP(nlp);
        auto t1 = std::chrono::steady_clock::now();
        first   = false;
        u       = my_nlp->results.solution_x;
        y       = my_nlp->results.solution_y;
        z_L     = my_nlp->results.solution_z_L;
        z_U     = my_nlp->results.solution_z_U;
        std::cout << enum_name(status) << ": "
                  << std::chrono::duration<double>{t1 - t0}.count() << '\n';
        return status == Ipopt::Solve_Succeeded ? t1 - t0 : t0 - t1;
    };
    return do_mpc_experiment(problem, dl_problem, solve, extra_opts);
}

#endif

void do_experiment(BenchmarkProblem &problem, BenchmarkProblem &dl_problem,
                   auto &solver, std::span<const std::string_view> extra_opts) {
    // Run experiment
    MPCResults mpc_res =
        do_experiment_impl(problem, dl_problem, solver, extra_opts);

    // Print results to output
    std::string solver_name = "Ipopt";
    if constexpr (requires { solver.get_name(); })
        solver_name = solver.get_name();

    MPCBenchmarkResults results{
        .problem   = problem,
        .evals     = *problem.evaluations,
        .solver    = solver_name,
        .runtimes  = mpc_res.runtimes,
        .states    = mpc_res.states,
        .inputs    = mpc_res.inputs,
        .extra     = {},
        .options   = extra_opts,
        .timestamp = timestamp_ms<std::chrono::system_clock>().count(),
    };
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

    // Load the problem
    BenchmarkProblem problem = load_benchmark_problem(
        path.parent_path().c_str(), path.filename().c_str(), horizon,
        formulation, extra_opts);
    std::optional<BenchmarkProblem> dl_problem_sto;
    if (problem.nx == 0) {
        std::cout << "Loading 'dl:' variant ..." << std::endl;
        dl_problem_sto = load_benchmark_dl_problem(path.parent_path().c_str(),
                                                   path.filename().c_str(),
                                                   horizon, extra_opts);
    }
    auto &dl_problem = dl_problem_sto ? *dl_problem_sto : problem;

    // Check which solver to use
    std::string_view solver_name = "fbetrust";
    alpaqa::params::set_params(solver_name, "method", extra_opts);

    // Available solvers
    std::map<std::string_view, std::function<void()>> solvers {
        {"pantr",
         [&] {
             using Accelerator = alpaqa::NewtonTRDirection<config_t>;
             using InnerSolver = alpaqa::PANTRSolver<Accelerator>;
             auto solver       = make_solver<InnerSolver>(extra_opts);
             do_experiment(problem, dl_problem, solver, extra_opts);
         }},
            {"strucpanoc",
             [&] {
                 using Accelerator = alpaqa::StructuredLBFGSDirection<config_t>;
                 using InnerSolver = alpaqa::PANOCSolver<Accelerator>;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, dl_problem, solver, extra_opts);
             }},
            {"panoc",
             [&] {
                 using Accelerator = alpaqa::LBFGSDirection<config_t>;
                 using InnerSolver = alpaqa::PANOCSolver<Accelerator>;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, dl_problem, solver, extra_opts);
             }},
            {"struczfpr",
             [&] {
                 using Accelerator = alpaqa::StructuredLBFGSDirection<config_t>;
                 using InnerSolver = alpaqa::ZeroFPRSolver<Accelerator>;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, dl_problem, solver, extra_opts);
             }},
            {"zfpr",
             [&] {
                 using Accelerator = alpaqa::LBFGSDirection<config_t>;
                 using InnerSolver = alpaqa::ZeroFPRSolver<Accelerator>;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, dl_problem, solver, extra_opts);
             }},
#if WITH_IPOPT
            {"ipopt",
             [&] {
                 auto solver = make_ipopt_solver(extra_opts);
                 do_experiment(problem, dl_problem, solver, extra_opts);
             }},
#endif
#if WITH_LBFGSPP
            {"lbfgspp",
             [&] {
                 using InnerSolver = InnerLBFGSppSolver;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, dl_problem, solver, extra_opts);
             }},
#endif
#if WITH_LBFGSB
            {"lbfgsb", [&] {
                 using InnerSolver = InnerLBFGSBSolver;
                 auto solver       = make_solver<InnerSolver>(extra_opts);
                 do_experiment(problem, dl_problem, solver, extra_opts);
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
