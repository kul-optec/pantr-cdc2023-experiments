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
#include "results.hpp"
#include <casadi-dll-wrapper.hpp>

#include <IpIpoptApplication.hpp>
#include <alpaqa/ipopt-adapter.hpp>

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
              << " [<path>/]<name> <horizon> <second-order> "
                 "[<key>=<value>...]\n";
}

USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);

auto make_solver(std::span<const std::string_view> extra_opts) {
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

std::string_view enum_name(Ipopt::ApplicationReturnStatus s) {
    using enum Ipopt::ApplicationReturnStatus;
    switch (s) {
        case Solve_Succeeded: return "Solve_Succeeded";
        case Solved_To_Acceptable_Level: return "Solved_To_Acceptable_Level";
        case Infeasible_Problem_Detected: return "Infeasible_Problem_Detected";
        case Search_Direction_Becomes_Too_Small:
            return "Search_Direction_Becomes_Too_Small";
        case Diverging_Iterates: return "Diverging_Iterates";
        case User_Requested_Stop: return "User_Requested_Stop";
        case Feasible_Point_Found: return "Feasible_Point_Found";
        case Maximum_Iterations_Exceeded: return "Maximum_Iterations_Exceeded";
        case Restoration_Failed: return "Restoration_Failed";
        case Error_In_Step_Computation: return "Error_In_Step_Computation";
        case Maximum_CpuTime_Exceeded: return "Maximum_CpuTime_Exceeded";
        case Maximum_WallTime_Exceeded: return "Maximum_WallTime_Exceeded";
        case Not_Enough_Degrees_Of_Freedom:
            return "Not_Enough_Degrees_Of_Freedom";
        case Invalid_Problem_Definition: return "Invalid_Problem_Definition";
        case Invalid_Option: return "Invalid_Option";
        case Invalid_Number_Detected: return "Invalid_Number_Detected";
        case Unrecoverable_Exception: return "Unrecoverable_Exception";
        case NonIpopt_Exception_Thrown: return "NonIpopt_Exception_Thrown";
        case Insufficient_Memory: return "Insufficient_Memory";
        case Internal_Error: return "Internal_Error";
        default: return "<unknown>";
    }
}

template <class Solver>
void do_experiment(auto &problem, Solver &solver,
                   std::span<const std::string_view> extra_opts) {
    // Ipopt problem adapter
    using Problem                    = alpaqa::IpoptAdapter;
    Ipopt::SmartPtr<Ipopt::TNLP> nlp = new Problem(problem.problem);
    auto *my_nlp                     = dynamic_cast<Problem *>(GetRawPtr(nlp));
    // Initial guess
    my_nlp->initial_guess = problem.initial_guess;

    auto evals = problem.evaluations;
    std::vector<real_t> gr_eval, it_time;
    std::vector<real_t> fbe, norm_q, eps, gamma, tau, ratio, radius;

    // Solve the problem
    auto t0     = std::chrono::steady_clock::now();
    auto status = solver->OptimizeTNLP(nlp);
    auto t1     = std::chrono::steady_clock::now();

    // Print the results
    evals         = std::make_shared<alpaqa::EvalCounter>(*evals);
    auto &results = my_nlp->results;
    std::cout << '\n' << *evals << '\n';
    auto time_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "solver:  "
              << "Ipopt" << '\n'
              << "problem: " << problem.path.filename().c_str() << " "
              << problem.horizon << " (from " << problem.path.parent_path()
              << ")" << '\n'
              << "status:  " << enum_name(status) << '\n'
              << "num var: " << problem.problem.get_n() << '\n'
              << "num con: " << problem.problem.get_m() << '\n'
              << "f = " << alpaqa::float_to_str(results.solution_f) << '\n'
              << "ε = " << alpaqa::float_to_str(results.nlp_error) << '\n'
              << "δ = " << alpaqa::float_to_str(results.infeasibility) << '\n'
              << "γ = " << '-' << '\n'
              << "Σ = " << '-' << '\n'
              << "time: " << alpaqa::float_to_str(time_s, 3) << " s\n"
              << "alm iter:  " << std::setw(6) << '-' << '\n'
              << "iter:      " << std::setw(6) << results.iter_count << '\n'
              << "backtrack: " << std::setw(6) << '-' << '\n'
              << "+ _______: " << std::setw(6) << results.iter_count << '\n'
              << "dir fail:  " << std::setw(6) << '-' << '\n'
              << "dir rej:   " << std::setw(6) << '-' << '\n'
              << std::endl;

    // Write results to file
    auto now_ms        = timestamp_ms<std::chrono::system_clock>();
    auto timestamp_str = std::to_string(now_ms.count());
    auto rnd_str       = random_hex_string(std::random_device());
    auto suffix        = timestamp_str + '_' + rnd_str;
    auto results_name  = "results_" + suffix;
    std::cout << "results: " << suffix << std::endl;
    std::ofstream res_file{results_name + ".py"};
    using ns = std::chrono::nanoseconds;
    write_results(res_file,
                  {
                      .problem     = problem,
                      .evals       = *evals,
                      .duration    = std::chrono::duration_cast<ns>(t1 - t0),
                      .status      = enum_name(status),
                      .success     = status == Ipopt::Solve_Succeeded,
                      .solver      = "Ipopt",
                      .f           = results.solution_f,
                      .δ           = results.infeasibility,
                      .ε           = results.nlp_error,
                      .γ           = 0,
                      .solution    = results.solution_x,
                      .multipliers = results.solution_y,
                      .outer_iter  = results.iter_count,
                      .inner_iter  = results.iter_count,
                      .extra =
                          {
                              {"grad_eval", vec{}},
                              {"it_time", vec{}},
                              {"fbe", vec{}},
                              {"norm_q", vec{}},
                              {"eps", vec{}},
                              {"gamma", vec{}},
                              {"tau", vec{}},
                              {"ratio", vec{}},
                              {"radius", vec{}},
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
    std::string_view solver = "ipopt";
    auto solver_it = std::ranges::find_if(extra_opts, [](std::string_view sv) {
        return sv.starts_with("solver=");
    });
    if (solver_it != extra_opts.end()) {
        solver = *solver_it;
        solver.remove_prefix(solver.find('=') + 1);
        extra_opts.erase(solver_it);
    }

    // Run experiment
    if (solver == "ipopt") {
        auto solver = make_solver(extra_opts);
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
