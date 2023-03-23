#include <alpaqa/config/config.hpp>
#include <alpaqa/util/demangled-typename.hpp>

#include <charconv>
#include <chrono>
#include <concepts>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>

#include "params.hpp"

namespace alpaqa {

using config_t = DefaultConfig;

template <class T>
void assert_key_empty(ParamString s) {
    if (!s.key.empty())
        throw std::invalid_argument("Type '" + demangled_typename(typeid(T)) +
                                    "' cannot be indexed in '" +
                                    std::string(s.full_key) + "'");
}

inline auto split_key(std::string_view full, char tok = '.') {
    auto dot_pos = full.find(tok);
    if (dot_pos == full.npos)
        return std::make_tuple(full, std::string_view{});
    std::string_view key{full.begin(), full.begin() + dot_pos};
    std::string_view rem{full.begin() + dot_pos + 1, full.end()};
    return std::make_tuple(key, rem);
}

template <class T>
void set_param(T &,
               [[maybe_unused]] ParamString s); // deliberately undefined

template <class T>
void unsupported_type(T &, [[maybe_unused]] ParamString s) {
    throw std::invalid_argument("Unknown parameter type '" +
                                demangled_typename(typeid(T)) + "' in '" +
                                std::string(s.full_key) + "'");
}

template <>
void set_param(bool &b, ParamString s);

template <>
void set_param(std::string_view &v, ParamString s);

template <>
void set_param(std::string &v, ParamString s);

template <class T>
    requires((std::floating_point<T> || std::integral<T>) && !std::is_enum_v<T>)
void set_param(T &f, ParamString s);

template <>
void set_param(vec<config_t> &v, ParamString s);

template <class Rep, class Period>
void set_param(std::chrono::duration<Rep, Period> &t, ParamString s);

template <class T, class A>
auto param_setter(A T::*attr) {
    return [attr](T &t, ParamString s) { return set_param(t.*attr, s); };
}

template <class T>
struct param_setter_fun_t {
    template <class A>
    param_setter_fun_t(A T::*attr) : set(param_setter(attr)) {}
    std::function<void(T &, ParamString)> set;
};

template <class T>
using dict_to_struct_table_t =
    std::map<std::string_view, param_setter_fun_t<T>>;

template <class T>
struct dict_to_struct_table {};

template <class T>
auto possible_keys() {
    const auto &tbl = dict_to_struct_table<T>::table;
    if (tbl.empty())
        return std::string{};
    auto penult       = std::prev(tbl.end());
    auto quote_concat = [](std::string &&a, auto b) {
        return a + "'" + std::string(b.first) + "', ";
    };
    return std::accumulate(tbl.begin(), penult, std::string{}, quote_concat) +
           "'" + std::string(penult->first) + "'";
}

template <class T>
    requires requires { dict_to_struct_table<T>::table; }
void set_param(T &t, ParamString s) {
    const auto &m         = dict_to_struct_table<T>::table;
    auto [key, remainder] = split_key(s.key);
    auto it               = m.find(key);
    if (it == m.end())
        throw std::invalid_argument(
            "Invalid key '" + std::string(key) + "' for type '" +
            demangled_typename(typeid(T)) + "' in '" + std::string(s.full_key) +
            "',\n  possible keys are: " + possible_keys<T>());
    s.key = remainder;
    it->second.set(t, s);
}

#define PARAMS_TABLE(type_, ...)                                               \
    template <>                                                                \
    struct dict_to_struct_table<type_> {                                       \
        using type = type_;                                                    \
        inline static const dict_to_struct_table_t<type> table{__VA_ARGS__};   \
    }

#define PARAMS_MEMBER(name)                                                    \
    {                                                                          \
#name, &type::name                                                     \
    }

template <class T>
void set_params(T &t, std::string_view prefix,
                std::span<const std::string_view> opts) {
    for (const auto &kv : opts) {
        auto [key, value]     = split_key(kv, '=');
        auto [pfx, remainder] = split_key(key);
        if (pfx != prefix)
            continue;
        set_param(t, {.full_key = kv, .key = remainder, .value = value});
    }
}

} // namespace alpaqa