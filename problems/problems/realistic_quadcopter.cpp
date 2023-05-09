#include <realistic_quadcopter-export.h>

#include <alpaqa/dl/dl-problem.h>
#include <casadi/casadi.hpp>
#include <any>
#include <cstring>
#include <iostream>
#include <limits>
#include <numbers>
#include <type_traits>
#include <vector>

#include "../problem-config.hpp"
#include "formulation.hpp"

using real_t   = alpaqa_real_t;
using length_t = alpaqa_length_t;
using index_t  = alpaqa_index_t;

namespace cs = casadi;
using cs::DM;
using cs::SX;
using std::vector;

inline constexpr auto inf = std::numeric_limits<real_t>::infinity();
inline constexpr auto pi  = std::numbers::pi_v<real_t>;

struct QuadcopterConfig {
    length_t N            = 30;
    real_t Ts             = 0.1;
    real_t at_min         = 0;
    real_t g              = 9.81;
    real_t at_max         = g * 5;
    real_t tilt_max       = 1.1 / 2;
    real_t d_tilt_max     = 0.1;
    real_t q_v            = 1.;
    real_t q_p            = 10.;
    real_t q_θ            = 1.;
    real_t r              = 1e-4;
    real_t r_ω            = 10.;
    real_t qf_v           = 25 * q_v;
    real_t qf_p           = 25 * q_p;
    real_t qf_θ           = 10 * q_θ;
    bool collision_constr = true;
};

struct QuadcopterModel {

    static constexpr length_t nx = 9;
    static constexpr length_t nu = 4;
    length_t nc                  = 4;
    length_t nc_N                = nc;
    length_t N;

    cs::Function f_d;
    cs::Function c;
    cs::Function c_N;
    cs::Function l;
    cs::Function l_N;
    vector<real_t> U_lb   = vector<real_t>(nu);
    vector<real_t> U_ub   = vector<real_t>(nu);
    vector<real_t> D_lb   = vector<real_t>(nc);
    vector<real_t> D_ub   = vector<real_t>(nc);
    vector<real_t> D_N_lb = vector<real_t>(nc_N);
    vector<real_t> D_N_ub = vector<real_t>(nc_N);
    vector<real_t> initial_guess;
    vector<real_t> initial_state;

    QuadcopterModel(const QuadcopterConfig &conf) : N{conf.N} {
        // State and input variables
        auto state = SX::sym("state", 9);
        SX px = state(0), py = state(1), pz = state(2);
        SX p  = SX::vertcat({px, py, pz});
        SX vx = state(3), vy = state(4), vz = state(5);
        SX v = SX::vertcat({vx, vy, vz});
        SX φ = state(6), θ = state(7), ψ = state(8);
        SX orient  = SX::vertcat({φ, θ, ψ});
        auto input = SX::sym("input", 4);
        SX at = input(0), ωx = input(1), ωy = input(2), ωz = input(3);
        SX ω             = SX::vertcat({ωx, ωy, ωz});
        auto state_input = SX::vertcat({state, input});

        // Rotation matrix
        auto cψ = SX::cos(ψ), cφ = SX::cos(φ), cθ = SX::cos(θ);
        auto sψ = SX::sin(ψ), sφ = SX::sin(φ), sθ = SX::sin(θ);
        SX ARB      = SX::vertcat({
            SX::horzcat(
                {cψ * cθ - sφ * sψ * sθ, -cφ * sψ, cψ * sθ + cθ * sφ * sψ}),
            SX::horzcat(
                {cθ * sψ + cψ * sφ * sθ, cφ * cψ, sψ * sθ - cψ * cθ * sφ}),
            SX::horzcat({-cφ * sθ, sφ, cφ * cθ}),
        });
        SX Ω        = SX::vertcat({
            SX::horzcat({cθ, 0, -cφ * sθ}),
            SX::horzcat({0, 1, sφ}),
            SX::horzcat({sθ, 0, cφ * cθ}),
        });
        SX d_orient = SX::solve(Ω, ω);

        // Continuous-time dynamics
        auto g   = SX::vertcat({0, 0, conf.g});
        auto a   = SX::mtimes(ARB, SX::vertcat({0, 0, at})) - g;
        auto f_c = cs::Function("f_c", {state, input},
                                {SX::vertcat({v, a, d_orient})});
        // Discrete-time dynamics (RK4)
        f_d = cs::Function("f_d", {state, input},
                           {discretize_rk4(f_c, state, input, conf.Ts)});

        // Input constraints
        U_lb[0] = conf.at_min;
        U_ub[0] = conf.at_max;
        U_lb[1] = -conf.d_tilt_max;
        U_ub[1] = +conf.d_tilt_max;
        U_lb[2] = -conf.d_tilt_max;
        U_ub[2] = +conf.d_tilt_max;
        U_lb[3] = -conf.d_tilt_max;
        U_ub[3] = +conf.d_tilt_max;

        // State constraints
        std::vector constr_v{
            φ,
            θ,
            SX::cos(φ) * SX::cos(θ),
        };
        if (conf.collision_constr)
            // constr_v.push_back(1e4 * SX::fmax(0, cs::sq(0.1) - SX::sq(px)) *
            //                    SX::fmax(0, cs::sq(0.1) - SX::sq(py)));
            constr_v.push_back(cs::sq(0.1) - cs::sq(px) - cs::sq(py));
        auto constr = SX::vertcat(constr_v);
        nc = nc_N = constr.size1();
        c = c_N = cs::Function("c", {state}, {constr});

        D_lb = {-pi / 2, -pi / 2, cs::cos(conf.tilt_max)};
        D_ub = {+pi / 2, +pi / 2, +inf};
        if (conf.collision_constr) {
            D_lb.push_back(-inf);
            D_ub.push_back(0);
        }
        D_N_lb = D_lb;
        D_N_ub = D_ub;

        // Initial state
        auto p0       = DM::vertcat({-0.20, -0.25, +0.50});
        auto v0       = DM::vertcat({0., 0., 0.});
        auto θ0       = DM::vertcat({0., 0., 0.});
        initial_state = vector<real_t>{DM::vertcat({p0, v0, θ0})};

        // Target state and cost
        auto pf = SX::vertcat({+0.25, +0.25, +0.50});
        auto lx = conf.q_p * SX::sumsqr(p - pf) + conf.q_v * SX::sumsqr(v) +
                  conf.q_θ * SX::sumsqr(θ);
        auto lu = conf.r * SX::sumsqr(input) + conf.r_ω * SX::sumsqr(ω);
        l       = cs::Function("l", {state_input}, {lx + lu});
        auto lN = conf.qf_p * SX::sumsqr(p - pf) + conf.qf_v * SX::sumsqr(v) +
                  conf.qf_θ * SX::sumsqr(θ);
        l_N = cs::Function("l_N", {state}, {lN});

        // Initial guess
        initial_guess = vector<real_t>(nu * N);
        for (index_t t = 0; t < N; ++t)
            initial_guess[nu * t] = conf.g;
    }
};

extern "C" REALISTIC_QUADCOPTER_EXPORT alpaqa_problem_register_t
benchmark_problem_register(void *user_data) {
    if (!user_data)
        throw std::invalid_argument("Missing user data");
    const auto *user        = reinterpret_cast<std::any *>(user_data);
    const auto &user_config = any_cast<const ProblemConfig &>(*user);
    using Problem           = SingleShootingProblem<QuadcopterModel>;
    QuadcopterConfig config;
    if (user_config.size > 0)
        config.N = user_config.size;
    auto *problem = new Problem{QuadcopterModel{config}};
    alpaqa_problem_register_t result;
    std::memset(&result, 0, sizeof(result));
    alpaqa::register_member_function(result, "initial_guess",
                                     &Problem::get_initial_guess);
    alpaqa::register_member_function(result, "get_initial_state_ptr",
                                     &Problem::get_initial_state_ptr);
    alpaqa::register_member_function(result, "get_initial_state",
                                     &Problem::get_initial_state);
    alpaqa::register_member_function(result, "set_initial_state",
                                     &Problem::set_initial_state);
    alpaqa::register_member_function(result, "get_nx", &Problem::get_nx);
    alpaqa::register_member_function(result, "get_nu", &Problem::get_nu);
    alpaqa::register_member_function(result, "get_nc", &Problem::get_nc);
    alpaqa::register_member_function(result, "get_nc_N", &Problem::get_nc_N);
    alpaqa::register_member_function(result, "simulate_dynamics",
                                     &Problem::simulate_dynamics);
    result.instance = problem;
    result.cleanup  = [](void *instance) {
        delete reinterpret_cast<Problem *>(instance);
    };
    result.functions = &problem->funcs;
    return result;
}
