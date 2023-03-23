#include <alpaqa/config/config.hpp>
#include <alpaqa/implementation/outer/alm.tpp>
#include <alpaqa/inner/panoc-ocp.hpp>
#include <alpaqa/util/demangled-typename.hpp>
#include <alpaqa/util/print.hpp>

#include "output.hpp"
#include "params.hpp"
#include "results.hpp"
#include <casadi-dll-wrapper.hpp>

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
              << " [<path>/]<name> <horizon> [method=<solver>] "
                 "[<key>=<value>...]\n";
}

void set_specific_params(auto &) {}

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
    return InnerSolver{solver_param};
}

template <class InnerSolver>
auto make_solver(length_t penalty_alm_split, const auto &extra_opts) {
    // Settings for the ALM solver
    using ALMSolver = alpaqa::ALMSolver<InnerSolver>;
    typename ALMSolver::Params alm_param;
    alm_param.max_iter          = 200;
    alm_param.ε                 = 1e-8;
    alm_param.δ                 = 1e-8;
    alm_param.print_interval    = 0;
    alm_param.penalty_alm_split = penalty_alm_split;
    alpaqa::set_params(alm_param, "alm", extra_opts);
    return ALMSolver{alm_param, make_inner_solver<InnerSolver>(extra_opts)};
}

struct MPCResults {
    mat states;
    mat inputs;
    vec runtimes;
};

MPCResults do_mpc_experiment(BenchmarkControlProblem &problem,
                             BenchmarkProblem &dl_problem, auto &solver,
                             std::span<const std::string_view> extra_opts) {
    rvec x0       = problem.initial_state();
    const auto nu = dl_problem.nu, nx = dl_problem.nx, nc = dl_problem.nc,
               nc_N = dl_problem.nc_N, N = problem.horizon;
    if (x0.size() != nx)
        throw std::logic_error("Invalid x0 dimension");
    const bool ss    = true;
    const length_t n = problem.n, m = problem.m;
    vec u_initial(n);
    for (index_t t = 0; t < N; ++t)
        u_initial.segment(t * nu, nu) = problem.initial_guess;
    vec u      = u_initial;
    vec z_l    = vec::Zero(n);
    vec z_u    = vec::Zero(n);
    vec y      = vec::Zero(m);
    vec u_dist = dl_problem.problem.get_box_C().upperbound.topRows(nu);
    for (real_t &ud : u_dist)
        if (!std::isfinite(ud))
            ud = 0;
    alpaqa::set_params(u_dist, "u_dist", extra_opts);
    if (u_dist.size() != nu)
        throw std::out_of_range("Invalid u_dist size");
    std::cout << "u_dist = " << u_dist.transpose() << '\n';
    length_t num_dist = 3;
    alpaqa::set_params(num_dist, "num_dist", extra_opts);
    length_t num_sim = N;
    alpaqa::set_params(num_sim, "num_sim", extra_opts);
    // Initial disturbance
    for (index_t i = 0; i < num_dist; ++i)
        x0 = dl_problem.simulate(x0, u_dist);
    // Initial solve
    bool warm_start = true;
    alpaqa::set_params(warm_start, "warm", extra_opts);
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
            u.topRows(n - nu)   = u.bottomRows(n - nu);
            z_l.topRows(n - nu) = z_l.bottomRows(n - nu);
            z_u.topRows(n - nu) = z_u.bottomRows(n - nu);
            if (nc != nc_N)
                throw std::logic_error("invalid multiplier shift");
            y.topRows(m - nc_N) = y.bottomRows(m - nc_N);
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
MPCResults do_experiment_impl(BenchmarkControlProblem &problem,
                              BenchmarkProblem &dl_problem, Solver &solver,
                              std::span<const std::string_view> extra_opts) {
    auto solve = [&](rvec u, rvec y, const rvec &, const rvec &) {
        auto stats = solver(problem.problem, u, y);
        return stats.status == alpaqa::SolverStatus::Converged
                   ? stats.elapsed_time
                   : -stats.elapsed_time;
    };
    return do_mpc_experiment(problem, dl_problem, solve, extra_opts);
}

void do_experiment(BenchmarkControlProblem &problem,
                   BenchmarkProblem &dl_problem, auto &solver,
                   std::span<const std::string_view> extra_opts) {
    // Run experiment
    MPCResults mpc_res =
        do_experiment_impl(problem, dl_problem, solver, extra_opts);

    // Print results to output
    std::string solver_name = "Ipopt";
    if constexpr (requires { solver.get_name(); })
        solver_name = solver.get_name();

    OCPMPCBenchmarkResults results{
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
    if (argc < 3)
        return print_usage(argv[0]), -1;
    fs::path path{argv[1]};
    if (!path.has_parent_path())
        path = fs::canonical(fs::path(argv[0])).parent_path() / path;
    if (!path.has_parent_path() || !path.has_filename())
        return (std::cerr << "invalid problem path\n"), -1;
    length_t horizon = std::stoi(argv[2]);

    std::vector<std::string_view> extra_opts;
    std::copy(argv + 3, argv + argc, std::back_inserter(extra_opts));

    // Disable Constraints
    ConstraintsOpt constr_opt;
    auto unconstr_it = std::ranges::find(extra_opts, "no-collision-constr");
    if (unconstr_it != extra_opts.end()) {
        std::cout << "Disabling collision constraints\n";
        constr_opt.with_collision = false;
    }
    auto no_bounds_it = std::ranges::find(extra_opts, "no-bound-constr");
    if (no_bounds_it != extra_opts.end()) {
        std::cout << "Disabling bound constraints\n";
        constr_opt.with_bounds = false;
    }

    // Load the problem
    BenchmarkControlProblem problem = load_benchmark_control_problem(
        path.parent_path().c_str(), path.filename().c_str(), horizon,
        constr_opt);
    std::cout << "Loading 'dl:' variant ..." << std::endl;
    BenchmarkProblem dl_problem =
        load_benchmark_dl_problem(path.parent_path().c_str(),
                                  path.filename().c_str(), horizon, constr_opt);
    auto penalty_alm_split = problem.penalty_alm_split;
    penalty_alm_split      = 0;

    // Check which solver to use
    std::string_view solver_name = "panococp";
    alpaqa::set_params(solver_name, "method", extra_opts);

    // Available solvers
    std::map<std::string_view, std::function<void()>> solvers{
        {"panococp",
         [&] {
             using InnerSolver = alpaqa::PANOCOCPSolver<config_t>;
             auto solver = make_solver<InnerSolver>(penalty_alm_split,
                                                    extra_opts);
             do_experiment(problem, dl_problem, solver, extra_opts);
         }},
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
