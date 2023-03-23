#pragma once

#if ALPAQA_WITH_OCP
#include <alpaqa/casadi/CasADiControlProblem.hpp>
#include <alpaqa/problem/ocproblem.hpp>
#endif
#include <alpaqa/casadi/CasADiProblem.hpp>
#include <alpaqa/problem/problem-counters.hpp>
#include <casadi-dll-wrapper-export.h>
#include <filesystem>
#include <string_view>

namespace fs = std::filesystem;

struct BenchmarkProblem {
    USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);
    alpaqa::TypeErasedProblem<config_t> problem;
    fs::path path;
    std::string name;
    fs::path full_path;
    std::string formulation;
    length_t horizon  = -1;
    bool second_order = false;
    length_t n = problem.get_n(), m = problem.get_m();
    vec initial_guess                                = vec(n);
    vec y                                            = vec::Zero(m);
    vec μ                                            = vec::Constant(m, 1e2);
    std::shared_ptr<alpaqa::EvalCounter> evaluations = nullptr;
    length_t penalty_alm_split                       = 0;
    length_t nx = 0, nu = 0, nc = 0, nc_N = 0;
    std::function<rvec()> initial_state           = {};
    std::function<vec(crvec x, crvec u)> simulate = {};
};

struct ConstraintsOpt {
    bool with_bounds    = true;
    bool with_collision = true;
};

CASADI_DLL_WRAPPER_EXPORT BenchmarkProblem load_benchmark_problem(
    std::string_view path, std::string_view name,
    alpaqa::length_t<alpaqa::DefaultConfig> horizon,
    std::string_view formulation, ConstraintsOpt constr_opt = {});

CASADI_DLL_WRAPPER_EXPORT BenchmarkProblem
load_benchmark_dl_problem(std::string_view path, std::string_view name,
                          alpaqa::length_t<alpaqa::DefaultConfig> size,
                          ConstraintsOpt constr_opt = {});

#if ALPAQA_WITH_OCP
struct BenchmarkControlProblem {
    USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);
    alpaqa::TypeErasedControlProblem<config_t> problem;
    fs::path path;
    std::string name;
    fs::path full_path;
    std::string formulation;
    length_t horizon  = problem.get_N();
    length_t n        = problem.get_N() * problem.get_nu(),
             m        = problem.get_N() * problem.get_nc() + problem.get_nc_N();
    vec initial_guess = vec(problem.get_nu());
    vec y             = vec::Zero(m);
    vec μ             = vec::Constant(m, 1e2);
    std::shared_ptr<alpaqa::OCPEvalCounter> evaluations = nullptr;
    length_t penalty_alm_split                          = 0;
    std::function<rvec()> initial_state                 = {};
};

CASADI_DLL_WRAPPER_EXPORT BenchmarkControlProblem
load_benchmark_control_problem(std::string_view path, std::string_view name,
                               alpaqa::length_t<alpaqa::DefaultConfig> horizon,
                               ConstraintsOpt constr_opt);
#endif
