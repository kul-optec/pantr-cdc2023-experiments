/**
 * @file
 * Loads a compiled CasADi problem and problem values from a .tsv file and
 * exposes it through a C interface that can be called from other languages.
 */

#include <alpaqa/dl/dl-problem.hpp>
#include <alpaqa/util/io/csv.hpp>
#include <casadi-dll-wrapper.hpp>

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <stdexcept>

#include "../problems/problem-config.hpp"

namespace fs = std::filesystem;
USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);

BenchmarkProblem load_benchmark_casadi_problem(
    std::string_view path, std::string_view name, length_t horizon,
    std::string_view formulation,
    std::span<const std::string_view> problem_options) {
    fs::path full_path =
        fs::path{path} / (std::string(name) + "_" + std::string(formulation) +
                          "_" + std::to_string(horizon) + ".so");
    bool second_order = formulation.ends_with("2");

    static std::mutex mtx;
    std::unique_lock lck{mtx};
    // Load CasADi problem and allocate workspaces
    using Problem    = alpaqa::TypeErasedProblem<config_t>;
    using CsProblem  = alpaqa::CasADiProblem<config_t>;
    using CntProblem = alpaqa::ProblemWithCounters<CsProblem>;
    BenchmarkProblem problem{
        .problem = std::make_unique<Problem>(
            Problem::make<CntProblem>(std::in_place, full_path.c_str())),
        .path         = full_path,
        .name         = std::string(name),
        .full_path    = full_path,
        .formulation  = std::string(formulation),
        .horizon      = horizon,
        .second_order = second_order,
    };
    lck.unlock();

    auto &cnt_problem   = problem.problem->as<CntProblem>();
    problem.evaluations = cnt_problem.evaluations;
    auto &cs_problem    = cnt_problem.problem;

    // Load numeric data
    std::ifstream guess{fs::path{full_path}.replace_extension(".guess.tsv")};
    if (!guess)
        throw std::runtime_error("Failed to open guess file");
    alpaqa::csv::read_row(guess, problem.initial_guess, '\t');
    guess >> problem.penalty_alm_split;

    // Get mutable view of initial state
    problem.initial_state = [&]() -> rvec { return cs_problem.param; };

    return problem;
}

BenchmarkProblem
load_benchmark_dl_problem(std::string_view path, std::string_view name,
                          length_t size,
                          std::span<const std::string_view> problem_options) {
    std::string_view pfx = "dl:";
    while (name.starts_with(pfx))
        name = name.substr(pfx.size());
    fs::path full_path = fs::path{path} / (std::string(name) + ".so");
    std::any problem_options_any{
        ProblemConfig{static_cast<int32_t>(size), problem_options},
    };
    // Load CasADi problem and allocate workspaces
    using Problem    = alpaqa::TypeErasedProblem<config_t>;
    using DLProblem  = alpaqa::dl::DLProblem;
    using CntProblem = alpaqa::ProblemWithCounters<DLProblem>;
    BenchmarkProblem problem{
        .problem      = std::make_unique<Problem>(Problem::make<CntProblem>(
            std::in_place, full_path.c_str(), "benchmark_problem",
            &problem_options_any)),
        .path         = full_path,
        .name         = std::string(name),
        .full_path    = full_path,
        .formulation  = "unknown",
        .horizon      = size,
        .second_order = false,
    };
    auto &cnt_problem   = problem.problem->as<CntProblem>();
    problem.evaluations = cnt_problem.evaluations;
    auto &dl_problem    = cnt_problem.problem;

    // Load numeric data
    try {
        using func_t = void(const DLProblem::instance_t *, real_t *);
        dl_problem.call_extra_func<func_t>("initial_guess",
                                           problem.initial_guess.data());
    } catch (std::out_of_range &e) {
        problem.initial_guess.setZero();
    }

    // Get mutable view of initial state
    using len_func_t      = length_t(const DLProblem::instance_t *);
    problem.nx            = dl_problem.call_extra_func<len_func_t>("get_nx");
    problem.nu            = dl_problem.call_extra_func<len_func_t>("get_nu");
    problem.nc            = dl_problem.call_extra_func<len_func_t>("get_nc");
    problem.nc_N          = dl_problem.call_extra_func<len_func_t>("get_nc_N");
    problem.initial_state = [&]() -> rvec {
        using func_t = std::span<real_t>(DLProblem::instance_t *);
        auto p = dl_problem.call_extra_func<func_t>("get_initial_state_ptr");
        return mvec{p.data(), static_cast<length_t>(p.size())};
    };
    problem.simulate = [&dl_problem, nx{problem.nx}](crvec x, crvec u) -> vec {
        vec x_next(nx);
        using func_t = void(const DLProblem::instance_t *, const real_t *,
                            const real_t *, real_t *);
        dl_problem.call_extra_func<func_t>("simulate_dynamics", x.data(),
                                           u.data(), x_next.data());
        return x_next;
    };

    return problem;
}

BenchmarkProblem
load_benchmark_problem(std::string_view path, std::string_view name,
                       length_t horizon, std::string_view formulation,
                       std::span<const std::string_view> problem_options) {
    std::string_view pfx = "dl:";
    if (std::string_view{name}.starts_with(pfx))
        return load_benchmark_dl_problem(path, name.substr(pfx.size()), horizon,
                                         problem_options);
    return load_benchmark_casadi_problem(path, name, horizon, formulation,
                                         problem_options);
};
