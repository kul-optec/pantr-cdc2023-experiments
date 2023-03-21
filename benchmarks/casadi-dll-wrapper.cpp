/**
 * @file
 * Loads a compiled CasADi problem and problem values from a .tsv file and
 * exposes it through a C interface that can be called from other languages.
 */

#include <alpaqa/dl/dl-problem.hpp>
#include <casadi-dll-wrapper.hpp>

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <stdexcept>

#include "problems/config.hpp"

namespace fs = std::filesystem;
USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);

namespace {

/// Hacky way to read a vector from a .tsv file
void read_vector(std::istream &is, rvec v) {
    std::string s;
    s.reserve(32);
    for (auto &vv : v)
        if (!(is >> s))
            throw std::runtime_error("read_vector extraction failed");
        else if (std::from_chars(&*s.begin(), &*s.end(), vv).ec != std::errc{})
            throw std::runtime_error("read_vector conversion failed");
    if (is.get() != '\n' && is)
        throw std::runtime_error("read_vector line not finished");
}

} // namespace

BenchmarkProblem load_benchmark_casadi_problem(std::string_view path,
                                               std::string_view name,
                                               length_t horizon,
                                               std::string_view formulation,
                                               ConstraintsOpt constr_opt) {
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
        .problem = Problem::make<CntProblem>(std::in_place, full_path.c_str()),
        .path    = full_path,
        .name    = std::string(name),
        .full_path    = full_path,
        .formulation  = std::string(formulation),
        .horizon      = horizon,
        .second_order = second_order,
    };
    lck.unlock();

    auto &cnt_problem   = problem.problem.as<CntProblem>();
    problem.evaluations = cnt_problem.evaluations;
    auto &cs_problem    = cnt_problem.problem;

    // Load numeric data
    std::ifstream bounds{fs::path{full_path}.replace_extension("tsv")};
    if (!bounds)
        throw std::runtime_error("Failed to open bounds file");
    read_vector(bounds, cs_problem.C.lowerbound);
    read_vector(bounds, cs_problem.C.upperbound);
    read_vector(bounds, cs_problem.D.lowerbound);
    read_vector(bounds, cs_problem.D.upperbound);
    read_vector(bounds, cs_problem.param);
    read_vector(bounds, problem.initial_guess);
    real_t penalty_alm_split;
    read_vector(bounds, mvec{&penalty_alm_split, 1});
    problem.penalty_alm_split = static_cast<length_t>(penalty_alm_split);

    // Get mutable view of initial state
    problem.initial_state = [&]() -> rvec { return cs_problem.param; };

    // Disable constraints
    if (!constr_opt.with_bounds)
        cs_problem.C = alpaqa::Box<config_t>(cs_problem.n);
    if (!constr_opt.with_collision) {
        cs_problem.D.lowerbound.topRows(problem.penalty_alm_split)
            .setConstant(-alpaqa::inf<config_t>);
        cs_problem.D.upperbound.topRows(problem.penalty_alm_split)
            .setConstant(+alpaqa::inf<config_t>);
    }

    return problem;
}

BenchmarkProblem load_benchmark_dl_problem(std::string_view path,
                                           std::string_view name, length_t size,
                                           ConstraintsOpt constr_opt) {
    std::string_view pfx = "dl:";
    while (name.starts_with(pfx))
        name = name.substr(pfx.size());
    fs::path full_path = fs::path{path} / (std::string(name) + ".so");
    ProblemConfig config{.size             = static_cast<int32_t>(size),
                         .collision_constr = constr_opt.with_collision};

    // Load CasADi problem and allocate workspaces
    using Problem    = alpaqa::TypeErasedProblem<config_t>;
    using DLProblem  = alpaqa::dl::DLProblem;
    using CntProblem = alpaqa::ProblemWithCounters<DLProblem>;
    BenchmarkProblem problem{
        .problem   = Problem::make<CntProblem>(std::in_place, full_path.c_str(),
                                             "benchmark_problem", &config),
        .path      = full_path,
        .name      = std::string(name),
        .full_path = full_path,
        .formulation  = "unknown",
        .horizon      = size,
        .second_order = false,
    };
    auto &cnt_problem   = problem.problem.as<CntProblem>();
    problem.evaluations = cnt_problem.evaluations;
    auto &dl_problem    = cnt_problem.problem;

    // Disable constraints
    if (!constr_opt.with_bounds)
        dl_problem.C = alpaqa::Box<config_t>(dl_problem.n);

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
    problem.simulate = [&](crvec x, crvec u) -> vec {
        vec x_next(problem.nx);
        using func_t = void(const DLProblem::instance_t *, const real_t *,
                            const real_t *, real_t *);
        dl_problem.call_extra_func<func_t>("simulate_dynamics", x.data(),
                                           u.data(), x_next.data());
        return x_next;
    };

    return problem;
}

BenchmarkProblem load_benchmark_problem(std::string_view path,
                                        std::string_view name, length_t horizon,
                                        std::string_view formulation,
                                        ConstraintsOpt constr_opt) {
    std::string_view pfx = "dl:";
    if (std::string_view{name}.starts_with(pfx))
        return load_benchmark_dl_problem(path, name.substr(pfx.size()), horizon,
                                         constr_opt);
    return load_benchmark_casadi_problem(path, name, horizon, formulation,
                                         constr_opt);
};

#if ALPAQA_WITH_OCP

BenchmarkControlProblem
load_benchmark_control_problem(std::string_view path, std::string_view name,
                               length_t horizon, ConstraintsOpt constr_opt) {
    fs::path full_path = fs::path{path} / (std::string(name) + "_ocp.so");

    static std::mutex mtx;
    std::unique_lock lck{mtx};
    // Load CasADi problem and allocate workspaces
    using Problem    = alpaqa::TypeErasedControlProblem<config_t>;
    using CsProblem  = alpaqa::CasADiControlProblem<config_t>;
    using CntProblem = alpaqa::ControlProblemWithCounters<CsProblem>;
    BenchmarkControlProblem problem{
        .problem   = Problem::make<CntProblem>(std::in_place, full_path.c_str(),
                                             horizon),
        .path      = full_path,
        .name      = std::string(name),
        .full_path = full_path,
        .formulation = "ocp",
    };
    lck.unlock();

    auto &cnt_problem   = problem.problem.as<CntProblem>();
    problem.evaluations = cnt_problem.evaluations;
    auto &cs_problem    = cnt_problem.problem;

    // Load numeric data
    std::ifstream bounds{fs::path{full_path}.replace_extension("tsv")};
    if (!bounds)
        throw std::runtime_error("Failed to open bounds file");
    read_vector(bounds, cs_problem.U.lowerbound);
    read_vector(bounds, cs_problem.U.upperbound);
    read_vector(bounds, cs_problem.D.lowerbound);
    read_vector(bounds, cs_problem.D.upperbound);
    read_vector(bounds, cs_problem.D_N.lowerbound);
    read_vector(bounds, cs_problem.D_N.upperbound);
    read_vector(bounds, cs_problem.x_init);
    read_vector(bounds, problem.initial_guess);
    real_t penalty_alm_split;
    read_vector(bounds, mvec{&penalty_alm_split, 1});
    problem.penalty_alm_split = static_cast<length_t>(penalty_alm_split) / 2;

    // Disable constraints
    if (!constr_opt.with_bounds)
        cs_problem.U = alpaqa::Box<config_t>(cs_problem.nu);
    if (!constr_opt.with_collision) {
        cs_problem.D   = alpaqa::Box<config_t>(cs_problem.nc);
        cs_problem.D_N = alpaqa::Box<config_t>(cs_problem.nc_N);
    }

    problem.initial_state = [&]() -> rvec { return cs_problem.x_init; };
    return problem;
}

#endif