#include <alpaqa/dl/dl-problem.h>
#include <lasso-export.h>
#include <algorithm>
#include <cstdint>
#include <random>

#include <Eigen/Dense>

using real_t   = alpaqa_real_t;
using index_t  = alpaqa_index_t;
using length_t = alpaqa_length_t;
using mat      = Eigen::MatrixX<real_t>;
using vec      = Eigen::VectorX<real_t>;
using mvec     = Eigen::Map<vec>;
using cmvec    = Eigen::Map<const vec>;

inline constexpr auto inf = std::numeric_limits<real_t>::infinity();

struct ProblemConfig {
    uint64_t seed   = 12345;
    length_t sc     = 16;
    length_t m      = 0;
    length_t n      = 0;
    real_t sparsity = 0.1;
    real_t λ_factor = 0.05;
};

struct Problem {
    ProblemConfig config;
    alpaqa_problem_functions_t funcs;
    length_t n, m;
    real_t λ;
    mat A;          ///< m×n
    vec b;          ///< m
    vec x_exact;    ///< n
    vec Aᵀb;        ///< n
    mutable vec Ax; ///< m

    void get_x_exact(real_t *xe) const { mvec{xe, n} = x_exact; }

    real_t eval_f(const real_t *x_) const {
        cmvec x{x_, n};
        Ax.noalias() = A * x;
        return (Ax - b).squaredNorm();
    }

    void eval_grad_f(const real_t *x_, real_t *g_) const {
        cmvec x{x_, n};
        mvec g{g_, n};
        Ax.noalias() = A * x;
        g.noalias()  = 2 * (A.transpose() * Ax);
        g -= 2 * Aᵀb;
    }

    real_t eval_f_grad_f(const real_t *x_, real_t *g_) const {
        cmvec x{x_, n};
        mvec g{g_, n};
        Ax.noalias() = A * x;
        g.noalias()  = 2 * (A.transpose() * Ax);
        g -= 2 * Aᵀb;
        return (Ax - b).squaredNorm();
    }

    void eval_g(const real_t *, real_t *) const {}

    void eval_grad_g_prod(const real_t *, const real_t *, real_t *gr_) const {
        mvec{gr_, n}.setZero();
    }

    real_t eval_prox_grad_step(real_t γ, const real_t *x_, const real_t *gr_,
                               real_t *x̂_, real_t *p_) const {
        cmvec x{x_, n};
        cmvec gr{gr_, n};
        mvec x̂{x̂_, n};
        auto &&x̂a = x̂.array();
        mvec p{p_, n};

        x̂  = x - γ * gr;
        x̂a = (x̂a.abs() - λ * γ).max(0) * x̂a.sign();
        p  = x̂ - x;
        return λ * x̂.lpNorm<1>();
    }

    Problem(const ProblemConfig &conf) : config{conf} {
        // Functions
        std::memset(&funcs, 0, sizeof(funcs));
        n = config.n ? n : config.sc * 32;
        m = config.m ? m : config.sc * 64;

        funcs.n = n;
        funcs.m = 0;

        std::mt19937 rng(config.seed);
        std::uniform_real_distribution<real_t> uniform{0, 1};
        A            = mat::NullaryExpr(m, n, [&] { return uniform(rng); });
        x_exact      = vec::NullaryExpr(n, [&] {
            return uniform(rng) <= config.sparsity ? uniform(rng) : 0;
        });
        b            = vec::NullaryExpr(m, [&] { return uniform(rng) / 10; });
        Ax.noalias() = A * x_exact;
        b.noalias() += Ax;
        Aᵀb.noalias() = A.transpose() * b;
        real_t λ_max  = Aᵀb.lpNorm<Eigen::Infinity>();
        λ             = config.λ_factor * λ_max;

        using alpaqa::member_caller;
        using P                   = Problem;
        funcs.eval_f              = member_caller<&P::eval_f>();
        funcs.eval_grad_f         = member_caller<&P::eval_grad_f>();
        funcs.eval_f_grad_f       = member_caller<&P::eval_f_grad_f>();
        funcs.eval_g              = member_caller<&P::eval_g>();
        funcs.eval_grad_g_prod    = member_caller<&P::eval_grad_g_prod>();
        funcs.eval_prox_grad_step = member_caller<&P::eval_prox_grad_step>();
    }
};

extern "C" LASSO_EXPORT alpaqa_problem_register_t
benchmark_problem_register(void *user_data) {
    const auto *scale = reinterpret_cast<const int32_t *>(user_data);
    auto *problem     = new Problem{ProblemConfig{.sc = scale ? *scale : 16}};
    alpaqa_problem_register_t result;
    std::memset(&result, 0, sizeof(result));
    alpaqa::register_member_function(result, "get_x_exact",
                                     &Problem::get_x_exact);
    result.instance = problem;
    result.cleanup  = [](void *instance) {
        delete reinterpret_cast<Problem *>(instance);
    };
    result.functions = &problem->funcs;
    return result;
}
