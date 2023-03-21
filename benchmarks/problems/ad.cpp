#include <alpaqa/config/config.hpp>
#include <alpaqa/util/demangled-typename.hpp>
#include <alpaqa/util/print.hpp>
#include <array>
#include <iostream>
#include <optional>
#include <type_traits>

USING_ALPAQA_CONFIG(alpaqa::DefaultConfig);

struct DualReal {
    constexpr explicit DualReal() : DualReal{0} {}
    constexpr DualReal(real_t value) : DualReal{value, 0} {}
    constexpr DualReal(real_t value, real_t deriv) : values{value, deriv} {}
    std::array<real_t, 2> values;
    [[nodiscard]] constexpr real_t &value() { return values[0]; }
    [[nodiscard]] constexpr const real_t &value() const { return values[0]; }
    [[nodiscard]] constexpr real_t &deriv() { return values[1]; }
    [[nodiscard]] constexpr const real_t &deriv() const { return values[1]; }
};

DualReal operator+(DualReal a, DualReal b) {
    return {a.value() + b.value(), a.deriv() + b.deriv()};
}
DualReal operator-(DualReal a, DualReal b) {
    return {a.value() - b.value(), a.deriv() - b.deriv()};
}
DualReal operator-(DualReal a) { return {-a.value(), -a.deriv()}; }

DualReal operator*(DualReal a, DualReal b) {
    return {a.value() * b.value(),
            a.value() * b.deriv() + b.value() * a.deriv()};
}
DualReal sin(DualReal a) {
    using std::cos;
    using std::sin;
    return {sin(a.value()), cos(a.value()) * a.deriv()};
}
DualReal cos(DualReal a) {
    using std::cos;
    using std::sin;
    return {cos(a.value()), -sin(a.value()) * a.deriv()};
}

struct DualStruc {
    constexpr explicit DualStruc() : DualStruc{false} {}
    constexpr DualStruc(bool value) : DualStruc{value, false} {}
    constexpr DualStruc(real_t value) : DualStruc{value != 0, false} {}
    constexpr DualStruc(int value) : DualStruc{value != 0, false} {}
    constexpr DualStruc(bool value, bool deriv) : values{value, deriv} {}
    std::array<bool, 2> values;
    [[nodiscard]] constexpr bool &value() { return values[0]; }
    [[nodiscard]] constexpr const bool &value() const { return values[0]; }
    [[nodiscard]] constexpr bool &deriv() { return values[1]; }
    [[nodiscard]] constexpr const bool &deriv() const { return values[1]; }
};

DualStruc operator+(DualStruc a, DualStruc b) {
    return {a.value() || b.value(), a.deriv() || b.deriv()};
}
DualStruc operator-(DualStruc a, DualStruc b) {
    return {a.value() || b.value(), a.deriv() || b.deriv()};
}
DualStruc operator-(DualStruc a) { return {a.value(), a.deriv()}; }

DualStruc operator*(DualStruc a, DualStruc b) {
    return {a.value() && b.value(),
            (a.value() && b.deriv()) || (b.value() && a.deriv())};
}
DualStruc sin(DualStruc a) {
    using std::cos;
    using std::sin;
    return {a.value(), a.deriv()};
}
DualStruc cos(DualStruc a) {
    using std::cos;
    using std::sin;
    return {true, a.value() && a.deriv()};
}

constexpr DualReal dx{0, 1};

static constexpr length_t nx = 9;
static constexpr length_t nu = 4;

template <class T = DualReal>
using xuvec = Eigen::Vector<T, nx + nu>;
template <class T = DualReal>
using xvec = Eigen::Vector<T, nx>;
template <class T = DualReal>
using uvec = Eigen::Vector<T, nu>;

using Eigen::Ref;

const auto f_c = [](const auto &x, const auto &u, auto &x_deriv) {
    using value_t = typename std::remove_cvref_t<decltype(x)>::Scalar;
    const auto v  = x.segment(3, 3);
    const auto θx = x(6), θy = x(7), θz = x(8);
    const auto at = u(0);
    const auto ω  = u.segment(1, 3);
    const auto cr = cos(θx);
    const auto sr = sin(θx);
    const auto cp = cos(θy);
    const auto sp = sin(θy);
    const auto cy = cos(θz);
    const auto sy = sin(θz);
    Eigen::Matrix<value_t, 3, 3> R;
    R << cy * cp,               // 0
        cy * sp * sr - sy * cr, //
        cy * sp * cr + sy * sr, //
        sy * cp,                // 1
        sy * sp * sr + cy * cr, //
        sy * sp * cr - cy * sr, //
        -sp,                    // 2
        cp * sr,                //
        cp * cr;                //
    Eigen::Vector<value_t, 3> atv{0, 0, at};
    Eigen::Vector<value_t, 3> g{0, 0, 9.81};
    auto a = R * atv - g;
    x_deriv << v, a, ω;
};

auto discretize_rk4(const auto &f_c, real_t Ts) {
    return [&, Ts]<class T>(const xvec<T> &x, const uvec<T> &u,
                            xvec<T> &x_next) -> void {
        xvec<T> k1, k2, k3, k4;
        f_c(x, u, k1);
        f_c(x + (Ts / 2) * k1, u, k2);
        f_c(x + (Ts / 2) * k2, u, k3);
        f_c(x + Ts * k3, u, k4);
        x_next =
            x + (Ts / 6) * k1 + (Ts / 3) * k2 + (Ts / 3) * k3 + (Ts / 6) * k4;
    };
}

auto values(auto &x) {
    return x.unaryExpr([](auto &a) -> decltype(auto) { return a.value(); });
}
auto derivs(auto &x) {
    return x.unaryExpr([](auto &a) -> decltype(auto) { return a.deriv(); });
}

template <class T>
concept adj_op = T::is_adj_op;
template <class T>
concept adj_op_or_ref = adj_op<std::remove_reference_t<T>>;

namespace detail {
using std::cos;
using std::exp;
using std::pow;
using std::sin;
auto transpose(real_t v) { return v; }
template <class T>
auto transpose(T &v)
    requires requires { v.transpose(); }
{
    return v.transpose();
}
template <class T>
auto exp(T &v)
    requires requires { v.array().exp().matrix(); }
{
    return v.array().exp().matrix();
}
template <class T>
auto sin(T &v)
    requires requires { v.array().sin().matrix(); }
{
    return v.array().sin().matrix();
}
template <class T>
auto cos(T &v)
    requires requires { v.array().cos().matrix(); }
{
    return v.array().cos().matrix();
}
} // namespace detail

template <class L, class R>
struct AddOp {
    L left;
    R right;
    using left_type  = typename std::remove_reference_t<L>::value_type;
    using right_type = typename std::remove_reference_t<R>::value_type;
    using value_type =
        decltype(std::declval<left_type>() + std::declval<right_type>());
    auto forward() { return left.forward() + right.forward(); }
    void reverse(const auto &seed) {
        left.reverse(seed);
        right.reverse(seed);
    }
    static constexpr bool is_adj_op = true;
};

template <class L, class R>
struct MulOp {
    L left;
    R right;
    using left_type  = typename std::remove_reference_t<L>::value_type;
    using right_type = typename std::remove_reference_t<R>::value_type;
    using value_type = decltype(std::declval<left_type>() * std::declval<right_type>());
    std::optional<left_type> left_value   = std::nullopt;
    std::optional<right_type> right_value = std::nullopt;
    auto forward() {
        left_value.emplace(left.forward());
        right_value.emplace(right.forward());
        return *left_value * *right_value;
    }
    void reverse(const auto &seed) {
        using detail::transpose;
        left.reverse(seed * transpose(*right_value));
        right.reverse(transpose(*left_value) * seed);
    }
    static constexpr bool is_adj_op = true;
};

template <class L>
struct SinOp {
    L left;
    using left_type  = typename std::remove_reference_t<L>::value_type;
    using value_type = decltype(detail::sin(std::declval<left_type>()));
    using cache_type = decltype(detail::cos(std::declval<left_type>()));
    std::optional<cache_type> cache_value = std::nullopt;
    auto forward() {
        left_type left_value = left.forward();
        cache_value.emplace(detail::cos(left_value));
        return detail::sin(left_value);
    }
    void reverse(const auto &seed) {
        left.reverse(seed * *cache_value);
    }
    static constexpr bool is_adj_op = true;
};

template <class L>
struct ExpOp {
    L left;
    using left_type  = typename std::remove_reference_t<L>::value_type;
    using value_type = decltype(detail::exp(std::declval<left_type>()));
    using cache_type = decltype(detail::exp(std::declval<left_type>()));
    std::optional<cache_type> cache_value = std::nullopt;
    auto forward() {
        left_type left_value = left.forward();
        cache_value.emplace(detail::exp(left_value));
        return *cache_value;
    }
    void reverse(const auto &seed) { left.reverse(seed * *cache_value); }
    static constexpr bool is_adj_op = true;
};

template <class L>
struct IndexOp {
    L left;
    index_t start;
    length_t len;
    using left_type  = typename std::remove_reference_t<L>::value_type;
    using value_type = decltype(std::declval<left_type>().segment(start, len));
    length_t left_size = 0;
    auto forward() {
        left_type left_value = left.forward();
        left_size            = left_value.size();
        return left_value.segment(start, len);
    }
    void reverse(const auto &seed) {
        vec tmp                 = vec::Zero(left_size);
        tmp.segment(start, len) = seed;
        left.reverse(tmp);
    }
    static constexpr bool is_adj_op = true;
};

template <class T>
struct AdjVar {
    AdjVar &operator=(const T &v) {
        value.emplace(v);
        return *this;
    }
    using value_type           = const T &;
    std::optional<T> value     = std::nullopt;
    std::optional<T> adj_value = std::nullopt;
    value_type forward() { return *value; }
    void reverse(const auto &seed) { *adj_value += seed; }
    static constexpr bool is_adj_op = true;
};

template <adj_op_or_ref L>
IndexOp<L> index(L &&l, index_t start, length_t len = 1) {
    return {std::forward<L>(l), start, len};
}

template <adj_op_or_ref L, adj_op_or_ref R>
AddOp<L, R> operator+(L &&l, R &&r) {
    return {std::forward<L>(l), std::forward<R>(r)};
}
template <adj_op_or_ref L, adj_op_or_ref R>
MulOp<L, R> operator*(L &&l, R &&r) {
    return {std::forward<L>(l), std::forward<R>(r)};
}
template <adj_op_or_ref L>
SinOp<L> sin(L &&l) {
    return {std::forward<L>(l)};
}
template <adj_op_or_ref L>
ExpOp<L> exp(L &&l) {
    return {std::forward<L>(l)};
}

AdjVar<real_t> operator""_var(long double v) {
    return {static_cast<real_t>(v)};
}
AdjVar<real_t> operator""_var(long long unsigned v) {
    return {static_cast<real_t>(v)};
}

int main() {
    real_t Ts = 0.1;
    auto f_d  = discretize_rk4(f_c, Ts);
    xvec<> x{0.10, -0.11, 0.12, -0.13, 0.14, -0.15, 0.16, -0.17, 0.18 + dx};
    uvec<> u{0.9, 0.05, 0.06, 0.07};
    xvec<> x_next;
    f_d(x, u, x_next);
    std::cout << values(x_next) << "\n\n" << derivs(x_next) << std::endl;
    Eigen::Matrix<bool, nx, nx + nu> jac;
    for (index_t i = 0; i < nx + nu; ++i) {
        xuvec<DualStruc> xu = xuvec<DualStruc>::Ones();
        xu(i).deriv()       = true;
        xvec<DualStruc> x   = xu.topRows(nx);
        uvec<DualStruc> u   = xu.bottomRows(nu);
        xvec<DualStruc> x_next;
        f_d(x, u, x_next);
        jac.col(i) = derivs(x_next);
    }
    std::cout << jac << std::endl;
    {
        AdjVar<vec> x{vec(3)};
        *x.value << 1.1, 1.2, 1.3;
        auto x1 = index(x, 0);
        auto x2 = index(x, 1);
        auto x3 = index(x, 2);
        auto y  = sin(x1 * x2) + exp(x1 * x2 * x3);
        std::cout << "y  = " << y.forward() << std::endl;
        x.adj_value.emplace(vec::Zero(3));
        y.reverse(1);
        std::cout << "∇y = " << x.adj_value->transpose() << std::endl;
        std::cout << demangled_typename(typeid(y)) << std::endl;
    }
}
