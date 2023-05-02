#pragma once

#include <span>
#if ALPAQA_WITH_OCP
#include <alpaqa/casadi/CasADiControlProblem.hpp>
#include <alpaqa/problem/ocproblem.hpp>
#endif
#include <alpaqa/casadi/CasADiProblem.hpp>
#include <alpaqa/problem/problem-counters.hpp>
#include <alpaqa/problem/type-erased-problem.hpp>
#include <filesystem>
#include <string_view>

namespace fs = std::filesystem;

struct BenchmarkProblem {
    USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);
    std::unique_ptr<alpaqa::TypeErasedProblem<config_t>> problem;
    fs::path path;
    std::string name;
    fs::path full_path;
    std::string formulation;
    length_t horizon  = -1;
    bool second_order = false;
    length_t n = problem->get_n(), m = problem->get_m();
    vec initial_guess                                = vec(n);
    vec y                                            = vec::Zero(m);
    vec μ                                            = vec::Constant(m, 1e2);
    std::shared_ptr<alpaqa::EvalCounter> evaluations = nullptr;
    length_t penalty_alm_split                       = 0;
    length_t nx = 0, nu = 0, nc = 0, nc_N = 0;
    std::function<rvec()> initial_state           = {};
    std::function<vec(crvec x, crvec u)> simulate = {};
};

BenchmarkProblem
load_benchmark_problem(std::string_view path, std::string_view name,
                       alpaqa::length_t<alpaqa::DefaultConfig> horizon,
                       std::string_view formulation,
                       std::span<const std::string_view> = {});

BenchmarkProblem
load_benchmark_dl_problem(std::string_view path, std::string_view name,
                          alpaqa::length_t<alpaqa::DefaultConfig> size,
                          std::span<const std::string_view> = {});
