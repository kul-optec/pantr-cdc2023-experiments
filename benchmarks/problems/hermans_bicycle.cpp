#include <hermans_bicycle-export.h>

#include <alpaqa/dl/dl-problem.h>
#include <casadi/casadi.hpp>
#include <cstring>
#include <limits>
#include <numbers>
#include <numeric>
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

struct HermansBicycleConfig {
    length_t N            = 30;
    real_t Ts             = 0.05;
    real_t lr             = 1.17;
    real_t lf             = 1.77;
    bool collision_constr = true;
};

struct HermansBicycleModel {

    static constexpr length_t nx = 4;
    static constexpr length_t nu = 2;
    length_t nc                  = 2;
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

    HermansBicycleModel(const HermansBicycleConfig &conf) : N{conf.N} {
        // State and input variables
        auto state = SX::sym("state", 4);
        SX px = state(0), py = state(1), v = state(2), θ = state(3);
        auto input = SX::sym("input", 2);
        SX a = input(0), δ = input(1);
        SX p             = SX::vertcat({px, py});
        auto state_input = SX::vertcat({state, input});

        // Continuous-time dynamics
        auto lr = conf.lr, lf = conf.lf;
        auto β   = SX::atan(lr / (lr + lf) * SX::tan(δ));
        auto dpx = v * SX::cos(θ + β);
        auto dpy = v * SX::sin(θ + β);
        auto dv  = a;
        auto dθ  = v / lr * SX::sin(β);
        auto f_c = cs::Function("f_c", {state, input},
                                {SX::vertcat({dpx, dpy, dv, dθ})});
        // Discrete-time dynamics (RK4)
        f_d = cs::Function("f_d", {state, input},
                           {discretize_rk4(f_c, state, input, conf.Ts)});

        // Input constraints
        U_lb[0] = -10;
        U_ub[0] = +10;
        U_lb[1] = -pi / 4;
        U_ub[1] = +pi / 4;

        // State constraints
        std::vector O1{
            px,
            5 - px,
            py + 2,
            2 + 1.5 * SX::sin(2 * pi * px / 5) - py,
        };
        std::vector O2{
            px,
            5 - px,
            py - 4 - 1.5 * SX::sin(2 * pi * px / 5),
            8 - py,
        };
        auto obstacle_prod = [](std::span<const SX> O) {
            return std::transform_reduce(
                O.begin(), O.end(), SX{1}, std::multiplies<>{},
                [](const SX &a) { return SX::fmax(0, a); });
        };
        std::vector<SX> constr_v;
        if (conf.collision_constr) {
            constr_v.push_back(obstacle_prod(O1));
            constr_v.push_back(obstacle_prod(O2));
        }
        auto constr = SX::vertcat(constr_v);
        nc = nc_N = constr.size1();
        c = c_N = cs::Function("c", {state}, {constr});

        if (conf.collision_constr) {
            D_lb = {-inf, -inf};
            D_ub = {0, 0};
        } else {
            D_lb = {};
            D_ub = {};
        }
        D_N_lb = D_lb;
        D_N_ub = D_ub;

        // Initial state
        auto p0       = DM::vertcat({-2, +5.});
        initial_state = vector<real_t>{DM::vertcat({p0, 0, 0})};

        // Target state and cost
        auto pf      = SX::vertcat({+6., +3.});
        real_t fudge = 1e1;
        auto lx      = 0.02 * SX::sumsqr(p - pf) + 0.0002 * SX::sumsqr(v) +
                  0.0002 * SX::sumsqr(θ);
        auto lu = 0.01 * SX::sumsqr(input);
        l       = cs::Function("l", {state_input}, {fudge * (lx + lu)});
        auto lN =
            2 * SX::sumsqr(p - pf) + 20 * SX::sumsqr(v) + 0.02 * SX::sumsqr(θ);
        l_N = cs::Function("l_N", {state}, {fudge * lN});

        // Initial guess
        initial_guess = vector<real_t>(nu * N);
    }
};

extern "C" HERMANS_BICYCLE_EXPORT alpaqa_problem_register_t
benchmark_problem_register(void *user_data) {
    const auto *user = reinterpret_cast<const ProblemConfig *>(user_data);
    using Problem    = SingleShootingProblem<HermansBicycleModel>;
    HermansBicycleConfig config;
    if (user->size > 0)
        config.N = user->size;
    config.collision_constr = user->collision_constr;
    auto *problem           = new Problem{HermansBicycleModel{config}};
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
