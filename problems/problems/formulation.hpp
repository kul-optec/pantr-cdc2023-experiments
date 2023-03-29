#pragma once

#include <alpaqa/dl/dl-problem.h>
#include <casadi/casadi.hpp>
#include <algorithm>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>
#include <any>

using real_t   = alpaqa_real_t;
using length_t = alpaqa_length_t;
using index_t  = alpaqa_index_t;

namespace cs = casadi;
using cs::DM;
using cs::SX;
using std::vector;

auto discretize_rk4(const cs::Function &f_c, const auto &state,
                    const auto &input, real_t Ts) {
    auto k1 = f_c(vector{state, input})[0];
    auto k2 = f_c(vector{state + Ts * k1 / 2, input})[0];
    auto k3 = f_c(vector{state + Ts * k2 / 2, input})[0];
    auto k4 = f_c(vector{state + Ts * k3, input})[0];
    return state + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}

template <class OptimalControlProblem>
class SingleShootingProblem {
  public:
    SingleShootingProblem(OptimalControlProblem &&ocp)
        : ocp{std::forward<OptimalControlProblem>(ocp)} {
        build();
    }

    alpaqa_problem_functions_t funcs;

  private:
    OptimalControlProblem ocp;
    cs::Function f, grad_f, g, grad_g_prod, jac_g, ψ, ψ_grad_ψ, hess_ψ_prod,
        grad_L, hess_L;

    void build() {
        // Inputs and simulated states
        const auto n = ocp.N * ocp.nu;
        const auto m = ocp.N * ocp.nc + ocp.nc_N;
        auto u_mat   = SX::sym("u", ocp.nu, ocp.N);
        auto u_vec   = SX::vec(u_mat);
        auto x_0     = SX::sym("x0", ocp.nx);
        auto x_mat   = ocp.f_d.mapaccum(ocp.N)(vector{x_0, u_mat})[0];
        x_mat        = SX::horzcat(vector{x_0, x_mat});
        auto x_split = SX::horzsplit(x_mat, ocp.N);
        auto xu_mat  = SX::vertcat(vector{x_split[0], u_mat});
        auto x_term  = x_split[1];

        // Cost
        auto stage_cost = SX::sum2(ocp.l.map(ocp.N)(vector{xu_mat})[0]);
        auto term_cost  = ocp.l_N(vector{x_term})[0];
        auto full_cost  = stage_cost + term_cost;
        this->f = cs::Function("f_ss", vector{u_vec, x_0}, vector{full_cost});
        this->grad_f = cs::Function("grad_f_ss", vector{u_vec, x_0},
                                    vector{SX::gradient(full_cost, u_vec)});

        // Constraints
        auto stage_constr = SX::vec(ocp.c.map(ocp.N)(vector{x_split[0]})[0]);
        auto term_constr  = ocp.c_N(vector{x_term})[0];
        auto constr       = SX::vertcat({stage_constr, term_constr});
        this->g = cs::Function("g_ss", vector{u_vec, x_0}, vector{constr});
        auto y  = SX::sym("y", m);
        this->grad_g_prod =
            cs::Function("grad_g_ss_prod", vector{u_vec, x_0, y},
                         vector{SX::jtimes(constr, u_vec, y, true)});
        this->jac_g = cs::Function("jac_g", vector{u_vec, x_0},
                                   vector{SX::jacobian(constr, u_vec)});

        // (Augmented) Lagrangian
        auto s  = SX::sym("s");
        auto Σ  = SX::sym("Σ", m);
        auto zl = SX::sym("zl", m), zu = SX::sym("zu", m);
        auto ζ        = constr + y / Σ;
        auto ẑ        = SX::fmax(zl, SX::fmin(ζ, zu));
        auto d        = ζ - ẑ;
        auto ŷ        = Σ * d;
        auto aug_lagr = s * full_cost + 0.5 * SX::dot(ŷ, d);
        auto lagr     = s * full_cost + SX::dot(y, constr);
        this->ψ = cs::Function("psi", vector{u_vec, x_0, y, Σ, s, zl, zu},
                               vector{aug_lagr, ŷ});
        this->ψ_grad_ψ =
            cs::Function("psi_grad_psi", vector{u_vec, x_0, y, Σ, s, zl, zu},
                         vector{aug_lagr, SX::gradient(aug_lagr, u_vec)});
        auto v              = SX::sym("v", n);
        auto jac_aug_lagr_v = SX::jtimes(aug_lagr, u_vec, v, false);
        this->hess_ψ_prod   = cs::Function(
            "hess_psi_prod", vector{u_vec, x_0, y, Σ, s, zl, zu, v},
            vector{SX::gradient(jac_aug_lagr_v, u_vec)});
        this->grad_L = cs::Function("grad_L", vector{u_vec, x_0, y, s},
                                    vector{SX::gradient(lagr, u_vec)});
        this->hess_L = cs::Function("hess_L", vector{u_vec, x_0, y, s},
                                    vector{SX::hessian(lagr, u_vec)});

        // Function bindings for alpaqa
        std::memset(&funcs, 0, sizeof(funcs));
        using alpaqa::member_caller;
        using P                = SingleShootingProblem;
        funcs.n                = n;
        funcs.m                = m;
        funcs.get_C            = member_caller<&P::get_C>();
        funcs.get_D            = member_caller<&P::get_D>();
        funcs.eval_f           = member_caller<&P::eval_f>();
        funcs.eval_grad_f      = member_caller<&P::eval_grad_f>();
        funcs.eval_g           = member_caller<&P::eval_g>();
        funcs.eval_grad_g_prod = member_caller<&P::eval_grad_g_prod>();
        funcs.eval_ψ           = member_caller<&P::eval_ψ>();
        funcs.eval_ψ_grad_ψ    = member_caller<&P::eval_ψ_grad_ψ>();
        funcs.eval_hess_ψ_prod = member_caller<&P::eval_hess_ψ_prod>();
        funcs.eval_grad_L      = member_caller<&P::eval_grad_L>();
        funcs.eval_jac_g       = member_caller<&P::eval_jac_g>();
        funcs.eval_hess_L      = member_caller<&P::eval_hess_L>();
        funcs.get_jac_g_num_nonzeros =
            member_caller<&P::get_jac_g_num_nonzeros>();
        funcs.get_hess_L_num_nonzeros =
            member_caller<&P::get_hess_L_num_nonzeros>();
    }

    real_t eval_f(const real_t *x) const {
        real_t ret;
        f({x, ocp.initial_state.data()}, {&ret});
        return ret;
    }

    void eval_grad_f(const real_t *x, real_t *gr) const {
        grad_f({x, ocp.initial_state.data()}, {gr});
    }

    void eval_g(const real_t *x, real_t *gx) const {
        g({x, ocp.initial_state.data()}, {gx});
    }

    void eval_grad_g_prod(const real_t *x, const real_t *y, real_t *gy) const {
        grad_g_prod({x, ocp.initial_state.data(), y}, {gy});
    }

    real_t eval_ψ(const real_t *x, const real_t *y, const real_t *Σ,
                  const real_t *zl, const real_t *zu, real_t *ŷ) const {
        const real_t scale = 1;
        real_t ret;
        ψ({x, ocp.initial_state.data(), y, Σ, &scale, zl, zu}, {&ret, ŷ});
        return ret;
    }

    real_t eval_ψ_grad_ψ(const real_t *x, const real_t *y, const real_t *Σ,
                         const real_t *zl, const real_t *zu, real_t *gr,
                         real_t *, real_t *) const {
        const real_t scale = 1;
        real_t ret;
        ψ_grad_ψ({x, ocp.initial_state.data(), y, Σ, &scale, zl, zu},
                 {&ret, gr});
        return ret;
    }

    void eval_hess_ψ_prod(const real_t *x, const real_t *y, const real_t *Σ,
                          real_t scale, const real_t *zl, const real_t *zu,
                          const real_t *v, real_t *Hv) const {
        hess_ψ_prod({x, ocp.initial_state.data(), y, Σ, &scale, zl, zu, v},
                    {Hv});
    }

    void eval_grad_L(const real_t *x, const real_t *y, real_t *gr,
                     real_t *) const {
        const real_t scale = 1;
        grad_L({x, ocp.initial_state.data(), y, &scale}, {gr});
    }

    static index_t casadi_to_index(casadi_int i) {
        return static_cast<index_t>(i);
    }

    static void copy_sparsity(const cs::Function &func, index_t *inner_idx,
                              index_t *outer_ptr) {
        auto &&sparsity = func.sparsity_out(0);
        if (!sparsity.is_dense()) {
            std::transform(sparsity.row(), sparsity.row() + sparsity.nnz(),
                           inner_idx, casadi_to_index);
            std::transform(sparsity.colind(),
                           sparsity.colind() + func.size2_out(0) + 1, outer_ptr,
                           casadi_to_index);
        }
    }

    static length_t sparsity_nonzeros(const cs::Function &func) {
        auto &&sparsity = func.sparsity_out(0);
        return sparsity.is_dense() ? 0 : casadi_to_index(sparsity.nnz());
    }

    length_t get_jac_g_num_nonzeros() const { return sparsity_nonzeros(jac_g); }

    void eval_jac_g(const real_t *x, index_t *inner_idx, index_t *outer_ptr,
                    real_t *J_values) const {
        if (J_values)
            jac_g({x, ocp.initial_state.data()}, {J_values});
        else
            copy_sparsity(jac_g, inner_idx, outer_ptr);
    }

    length_t get_hess_L_num_nonzeros() const {
        return sparsity_nonzeros(hess_L);
    }

    void eval_hess_L(const real_t *x, const real_t *y, real_t scale,
                     index_t *inner_idx, index_t *outer_ptr,
                     real_t *H_values) const {
        if (H_values)
            hess_L({x, ocp.initial_state.data(), y, &scale}, {H_values});
        else
            copy_sparsity(hess_L, inner_idx, outer_ptr);
    }

    void get_C(real_t *lb, real_t *ub) const {
        for (index_t t = 0; t < ocp.N; ++t) {
            std::copy(ocp.U_lb.begin(), ocp.U_lb.end(), lb);
            std::copy(ocp.U_ub.begin(), ocp.U_ub.end(), ub);
            lb += ocp.nu;
            ub += ocp.nu;
        }
    }

    void get_D(real_t *lb, real_t *ub) const {
        for (index_t t = 0; t < ocp.N; ++t) {
            std::copy(ocp.D_lb.begin(), ocp.D_lb.end(), lb);
            std::copy(ocp.D_ub.begin(), ocp.D_ub.end(), ub);
            lb += ocp.nc;
            ub += ocp.nc;
        }
        std::copy(ocp.D_N_lb.begin(), ocp.D_N_lb.end(), lb);
        std::copy(ocp.D_N_ub.begin(), ocp.D_N_ub.end(), ub);
    }

  public:
    void get_initial_guess(real_t *u0) const {
        std::copy(ocp.initial_guess.begin(), ocp.initial_guess.end(), u0);
    }

    void get_initial_state(real_t *x0) const {
        std::copy(ocp.initial_state.begin(), ocp.initial_state.end(), x0);
    }
    void set_initial_state(const real_t *x0) {
        std::copy_n(x0, ocp.nx, ocp.initial_state.begin());
    }
    [[nodiscard]] length_t get_nx() const { return ocp.nx; }
    [[nodiscard]] length_t get_nu() const { return ocp.nu; }
    [[nodiscard]] length_t get_nc() const { return ocp.nc; }
    [[nodiscard]] length_t get_nc_N() const { return ocp.nc_N; }
    std::span<real_t> get_initial_state_ptr() {
        return {ocp.initial_state.data(), ocp.initial_state.size()};
    }
    void simulate_dynamics(const real_t *x, const real_t *u,
                           real_t *x_next) const {
        ocp.f_d({x, u}, {x_next});
    }
};
