#include <hanging_chain-export.h>

#include <alpaqa/dl/dl-problem.h>
#include <casadi/casadi.hpp>
#include <cstring>
#include <limits>
#include <numbers>
#include <type_traits>
#include <vector>

#include "config.hpp"
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

struct HangingChainConfig {
    length_t N            = 30;
    real_t Ts             = 0.1;
    length_t N_balls      = 9; ///< Number of balls
    length_t n_dim        = 3; ///< Number of spatial dimensions
    real_t α              = 25;
    real_t β              = 1;
    real_t γ              = 0.01;
    real_t m              = 0.03;  ///< mass
    real_t D              = 0.1;   ///< spring constant
    real_t L              = 0.033; ///< spring length
    real_t v_max          = 1;     ///< maximum actuator velocity
    real_t g_grav         = 9.81;  ///< Gravitational acceleration       [m/s²]
    bool collision_constr = true;
};

struct HangingChainModel {

    length_t nx   = 0;
    length_t nu   = 0;
    length_t nc   = 4;
    length_t nc_N = nc;
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

    HangingChainModel(const HangingChainConfig &conf) : N{conf.N} {
        // State and input variables
        auto d = conf.n_dim, Nb = conf.N_balls;
        auto x           = SX::sym("x", d, (Nb + 1));
        auto v           = SX::sym("v", d, Nb);
        auto state       = SX::vertcat({SX::vec(x), SX::vec(v)});
        auto input       = SX::sym("input", d);
        auto state_input = SX::vertcat({state, input});
        nx               = state.size1();
        nu               = input.size1();

        // Parameters
        auto m     = conf.m; // mass
        auto D     = conf.D; // spring constant
        auto L     = conf.L; // spring length
        auto g     = d == 2 ? DM{0, -conf.g_grav} : DM{0, 0, -conf.g_grav};
        auto x_end = d == 2 ? DM{1, 0} : DM{1, 0, 0};

        // Continuous-time dynamics
        auto f1        = SX::vertcat({SX::vec(v), input});
        using ivec     = vector<casadi_int>;
        auto x_spl_bgn = SX::horzsplit(x, ivec{0, 1, Nb + 1}); // [x[1], x[1:]]
        auto x_spl_end = SX::horzsplit(x, Nb); // [x[:-1], x[-1]]
        auto dist_vect = SX::horzcat({
            x_spl_bgn[0],
            x_spl_bgn[1] - x_spl_end[0],
        });
        auto dist_norm = SX::sqrt(SX::sum1(dist_vect * dist_vect));
        auto F = SX::mtimes(dist_vect, SX::diag(D * (1 - L / dist_norm.T())));
        auto F_spl_bgn = SX::horzsplit(F, ivec{0, 1, Nb + 1}); // [F[1], F[1:]]
        auto F_spl_end = SX::horzsplit(F, Nb); // [F[:-1], F[-1]]
        auto fs        = SX::horzcat({F_spl_bgn[1] - F_spl_end[0]}) / m +
                  SX::repmat(g, 1, Nb);
        auto f_c = cs::Function("f_c", {state, input},
                                {SX::vertcat({f1, SX::vec(fs)})});

        // Discrete-time dynamics (RK4)
        f_d = cs::Function("f_d", {state, input},
                           {discretize_rk4(f_c, state, input, conf.Ts)});

        // Input constraints
        U_lb.resize(nu);
        std::fill(U_lb.begin(), U_lb.end(), -conf.v_max);
        U_ub.resize(nu);
        std::fill(U_ub.begin(), U_ub.end(), +conf.v_max);

        // State constraints
        nc = nc_N = 0;
        c = c_N = cs::Function("c", {state}, {SX::vertcat({})});
        D_lb.resize(nc);
        D_ub.resize(nc);
        D_N_lb = D_lb;
        D_N_ub = D_ub;

        // Initial state
        initial_state.resize(nx);
        for (index_t i = 0; i < Nb + 1; ++i)
            initial_state[i * d] =
                static_cast<real_t>(i + 1) / static_cast<real_t>(Nb + 1);

        // Target state and cost
        cs::Slice all_d{static_cast<casadi_int>(0), static_cast<casadi_int>(d)};
        auto lx = conf.α * SX::sumsqr(x(all_d, Nb) - SX{x_end}) +
                  conf.β * SX::sumsqr(SX::vec(v));
        auto lu = conf.γ * SX::sumsqr(input);
        l       = cs::Function("l", {state_input}, {lx + lu});
        l_N     = cs::Function("l_N", {state}, {lx});

        // Initial guess
        initial_guess.resize(nu * N);
    }
};

extern "C" HANGING_CHAIN_EXPORT alpaqa_problem_register_t
benchmark_problem_register(void *user_data) {
    const auto *user = reinterpret_cast<const ProblemConfig *>(user_data);
    using Problem    = SingleShootingProblem<HangingChainModel>;
    HangingChainConfig config;
    if (user->size > 0)
        config.N = user->size;
    config.collision_constr = user->collision_constr;
    auto *problem           = new Problem{HangingChainModel{config}};
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
