#pragma once

#include <cstdint>
#include <span>
#include <string_view>

struct ProblemConfig {
    int32_t size = 0;
    std::span<const std::string_view> options = {};
};
