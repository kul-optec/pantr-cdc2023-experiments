#include <alpaqa/dl/dl-problem.h>
#include <sp-log-regr-export.h>
#include <algorithm>
#include <cstdint>
#include <random>

#include <Eigen/Dense>

using real_t   = alpaqa_real_t;
using index_t  = alpaqa_index_t;
using length_t = alpaqa_length_t;
using mat      = Eigen::MatrixX<real_t>;
using vec      = Eigen::VectorX<real_t>;
using rvec     = Eigen::Ref<vec>;
using crvec    = Eigen::Ref<const vec>;
using mvec     = Eigen::Map<vec>;
using cmvec    = Eigen::Map<const vec>;

inline constexpr auto inf = std::numeric_limits<real_t>::infinity();

struct ProblemConfig {
    uint64_t seed   = 12345;
    length_t sc     = 16;
    length_t m      = 0;
    length_t n      = 0;
    real_t λ_factor = 0.1;
};

struct Problem {
    ProblemConfig config;
    alpaqa_problem_functions_t funcs;
    length_t n, m;
    real_t λ, μ;
    mat A;          ///< m×n
    vec b;          ///< m
    vec x_exact;    ///< n
    vec Aᵀb;        ///< n
    mutable vec Ax; ///< m

    void get_x_exact(real_t *xe) const { mvec{xe, n} = x_exact; }

    /// f(x) = μ ∑ ln(1 + exp(-b x))
    real_t logistic_loss(crvec x) const {
        auto &&xa = x.array();
        auto &&ba = b.array();
        return μ * ((-ba * xa).exp() + 1).log().sum();
    }

    /// df(x)/dx = -μ b / (exp(b x) + 1)
    void deriv_logistic_loss(crvec x, rvec g) const {
        auto &&xa = x.array();
        auto &&ba = b.array();
        g         = -μ * ba / ((ba * xa).exp() + 1);
    }

    real_t eval_f(const real_t *x_) const {
        cmvec x{x_, n};
        Ax.noalias() = A * x;
        return logistic_loss(Ax);
    }

    void eval_grad_f(const real_t *x_, real_t *g_) const {
        cmvec x{x_, n};
        mvec g{g_, n};
        Ax.noalias() = A * x;
        // ∇(f∘A)(x) = Aᵀ∇f(Ax)
        deriv_logistic_loss(Ax, Ax);      // Ax ← ∇f(Ax)
        g.noalias() = A.transpose() * Ax; // g ← Aᵀ∇f(Ax)
    }

    real_t eval_f_grad_f(const real_t *x_, real_t *g_) const {
        cmvec x{x_, n};
        mvec g{g_, n};
        Ax.noalias() = A * x;
        real_t f     = logistic_loss(Ax);
        // ∇(f∘A)(x) = Aᵀ∇f(Ax)
        deriv_logistic_loss(Ax, Ax);      // Ax ← ∇f(Ax)
        g.noalias() = A.transpose() * Ax; // g ← Aᵀ∇f(Ax)
        return f;
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
        n = config.n ? n : config.sc * 100;
        m = config.m ? m : config.sc * 4;

        funcs.n = n;
        funcs.m = 0;

        std::mt19937 rng(config.seed);
        std::uniform_real_distribution<real_t> unifrm{0, 1};
        std::normal_distribution<real_t> nrml;
        A            = mat::NullaryExpr(m, n, [&] { return unifrm(rng); });
        x_exact      = vec::NullaryExpr(n, [&] {
            return unifrm(rng) <= 20. / static_cast<real_t>(n) ? nrml(rng) : 0;
        });
        Ax.noalias() = A * x_exact;
        b            = Ax.unaryExpr([&](real_t Axi) -> real_t {
            return unifrm(rng) <= 1. / (1 + std::exp(-Axi)) ? 1 : 0;
        });

        Aᵀb.noalias() = A.transpose() * b;
        real_t λ_max = Aᵀb.lpNorm<Eigen::Infinity>() / static_cast<real_t>(m);
        λ            = config.λ_factor * λ_max;
        μ            = 1. / static_cast<real_t>(m);

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

extern "C" SP_LOG_REGR_EXPORT alpaqa_problem_register_t
benchmark_problem_register(void *user_data) {
    const auto *scale = reinterpret_cast<const int32_t *>(user_data);
    auto *problem     = new Problem{ProblemConfig{.sc = scale ? *scale : 5}};
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
