#pragma once

#include <span>
#include <string_view>

namespace alpaqa {

struct ParamString {
    std::string_view full_key, key, value;
};

template <class T>
void set_params(T &, std::string_view prefix,
                std::span<const std::string_view>);

} // namespace alpaqa
