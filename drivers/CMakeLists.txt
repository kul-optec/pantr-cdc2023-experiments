find_package(alpaqa REQUIRED)
find_package(Threads REQUIRED)
find_package(Ipopt REQUIRED)

# CasADi wrapper for loading CasADi problems
add_library(casadi-dll-wrapper "casadi-dll-wrapper.cpp")
target_link_libraries(casadi-dll-wrapper
    PUBLIC alpaqa::alpaqa alpaqa::casadi-loader PRIVATE alpaqa::dl-loader)
if (TARGET alpaqa::casadi-ocp-loader)
    target_link_libraries(casadi-dll-wrapper PUBLIC alpaqa::casadi-ocp-loader)
endif()
target_include_directories(casadi-dll-wrapper PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

# Benchmark driver executable
add_executable(benchmark-mpc-driver "mpc-driver.cpp")
target_link_libraries(benchmark-mpc-driver PRIVATE casadi-dll-wrapper)

if (TARGET alpaqa::ipopt-adapter)
    message(STATUS "Enabled Ipopt driver")
    target_link_libraries(benchmark-mpc-driver PRIVATE alpaqa::ipopt-adapter)
    target_compile_definitions(benchmark-mpc-driver PRIVATE WITH_IPOPT)
endif()

if (TARGET alpaqa::lbfgspp-adapter)
    message(STATUS "Enabled LBFGS++ driver")
    target_link_libraries(benchmark-mpc-driver PRIVATE alpaqa::lbfgspp-adapter)
    target_compile_definitions(benchmark-mpc-driver PRIVATE WITH_LBFGSPP)
endif()

if (TARGET alpaqa::lbfgsb-adapter)
    message(STATUS "Enabled L-BFGS-B driver")
    target_link_libraries(benchmark-mpc-driver PRIVATE alpaqa::lbfgsb-adapter)
    target_compile_definitions(benchmark-mpc-driver PRIVATE WITH_LBFGSB)
    target_sources(benchmark-mpc-driver PRIVATE "lbfgsb-alm.cpp")
endif()
