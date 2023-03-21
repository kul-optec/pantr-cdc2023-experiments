find_package(casadi REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Uses Python to generate a CasADi problem
function(casadi_function_codegen_python problem pythonfile formulation horizon)
    if (${horizon})
        set(problem_name "${problem}_${formulation}_${horizon}")
    else()
        set(problem_name "${problem}_${formulation}")
    endif()
    if (NOT TARGET ${problem_name})
        add_custom_command(OUTPUT "${problem_name}.c"
                        COMMAND Python3::Interpreter
                        ARGS "${CMAKE_CURRENT_SOURCE_DIR}/${pythonfile}"
                            "${problem}" "${formulation}" "${horizon}"
                        BYPRODUCTS "${problem_name}.tsv"
                        MAIN_DEPENDENCY "${pythonfile}")
        add_library("${problem_name}" MODULE "${problem_name}.c")
        set_target_properties("${problem_name}" PROPERTIES
                            PREFIX ""
                            RELEASE_POSTFIX ""
                            DEBUG_POSTFIX ""
                            RELWITHDEBINFO_POSTFIX ""
                            MINSIZEREL_POSTFIX ""
                            OUTPUT_NAME ${problem_name})
        add_custom_command(TARGET "${problem_name}" POST_BUILD
                        COMMAND ${CMAKE_COMMAND}
                        ARGS -E copy "${problem_name}.tsv"
                                "$<TARGET_FILE_DIR:${problem_name}>")
    endif()
endfunction()
    
# Python CasADi problem selection
set(PROBLEM_NAME_OPTIONS "quadcopter" "hermans_bicycle" "hanging_chain" "integrator")
set(PROBLEM_NAME "quadcopter" CACHE STRING "Problem to benchmark")
set_property(CACHE PROBLEM_NAME PROPERTY STRINGS ${PROBLEM_NAME_OPTIONS})
if(NOT PROBLEM_NAME IN_LIST PROBLEM_NAME_OPTIONS)
    message(FATAL_ERROR "PROBLEM_NAME must be one of ${PROBLEM_NAME_OPTIONS}")
endif()
# Python CasADi problem formulation selection
set(PROBLEM_FORM_OPTIONS "ocp" "ocp2" "ss" "ss2" "ss2p" "ms" "ms2" "ms2p")
set(PROBLEM_FORM "ss2p" CACHE STRING "Problem to benchmark")
set_property(CACHE PROBLEM_FORM PROPERTY STRINGS ${PROBLEM_FORM_OPTIONS})
if(NOT PROBLEM_FORM IN_LIST PROBLEM_FORM_OPTIONS)
    message(FATAL_ERROR "PROBLEM_FORM must be one of ${PROBLEM_FORM_OPTIONS}")
endif()
# Python CasADi problem horizon lenght selection
set(PROBLEM_HORIZON 12 CACHE STRING "Horizon length")
set(FULL_PROBLEM_NAME "${PROBLEM_NAME}_${PROBLEM_FORM}_${PROBLEM_HORIZON}")

# Python CasADi problem generation
casadi_function_codegen_python(${PROBLEM_NAME} "util/generate_problem.py"
    ${PROBLEM_FORM} ${PROBLEM_HORIZON})

foreach(N RANGE 1 60 1)
    casadi_function_codegen_python("quadcopter" "util/generate_problem.py"
                                   "ss2p" ${N})
    casadi_function_codegen_python("quadcopter" "util/generate_problem.py"
                                   "ms2p" ${N})
endforeach()
casadi_function_codegen_python("quadcopter" "util/generate_problem.py"
                               "ocp" 0)

foreach(N RANGE 1 60 1)
    casadi_function_codegen_python("hermans_bicycle" "util/generate_problem.py"
                                   "ss2p" ${N})
endforeach()

foreach(N RANGE 1 60 1)
    casadi_function_codegen_python("hanging_chain" "util/generate_problem.py"
                                   "ss2p" ${N})
    casadi_function_codegen_python("hanging_chain" "util/generate_problem.py"
                                   "ms2" ${N})
endforeach()
casadi_function_codegen_python("hanging_chain" "util/generate_problem.py"
                               "ocp" 0)
casadi_function_codegen_python("hermans_bicycle" "util/generate_problem.py"
                               "ocp" 0)

foreach(N RANGE 2 40 2)
    casadi_function_codegen_python("integrator" "util/generate_problem.py"
                                   "ss2p" ${N})
endforeach()

foreach(N RANGE 1 60 1)
    casadi_function_codegen_python("quadcopter" "util/generate_problem.py"
                                   "ss2" ${N})
    casadi_function_codegen_python("quadcopter" "util/generate_problem.py"
                                   "ms2" ${N})
endforeach()

foreach(N RANGE 1 40 2)
    casadi_function_codegen_python("hermans_bicycle" "util/generate_problem.py"
                                   "ss2" ${N})
    casadi_function_codegen_python("hermans_bicycle" "util/generate_problem.py"
                                   "ms2" ${N})
endforeach()

# foreach(N RANGE 10 40 10)
#     casadi_function_codegen_python("hanging_chain" "util/generate_problem.py"
#                                    "ss2" ${N})
#     casadi_function_codegen_python("hanging_chain" "util/generate_problem.py"
#                                    "ms2" ${N})
# endforeach()

foreach(N RANGE 2 40 2)
    casadi_function_codegen_python("integrator" "util/generate_problem.py"
                                   "ss2" ${N})
    casadi_function_codegen_python("integrator" "util/generate_problem.py"
                                   "ms2" ${N})
endforeach()

casadi_function_codegen_python("hanging_chain" "util/generate_problem.py" "ms2" 4)
casadi_function_codegen_python("hanging_chain" "util/generate_problem.py" "ss2" 4)

# CasADi wrapper for loading CasADi problems in Julia
add_library(casadi-dll-wrapper SHARED "casadi-dll-wrapper.cpp")
target_link_libraries(casadi-dll-wrapper
    PUBLIC alpaqa::alpaqa alpaqa::casadi-loader 
    PRIVATE alpaqa::warnings alpaqa::dl-loader)
if (ALPAQA_WITH_OCP)
    target_link_libraries(casadi-dll-wrapper PUBLIC alpaqa::casadi-ocp-loader)
endif()
target_compile_definitions(casadi-dll-wrapper PRIVATE
    PROBLEM_DLL=\"$<TARGET_FILE_NAME:${FULL_PROBLEM_NAME}>\"
    $<$<STREQUAL:${PROBLEM_FORM},ocp>:"CONTROL_PROBLEM">
    PROBLEM_SECOND_ORDER=$<IF:$<STREQUAL:${PROBLEM_FORM},ss2>,1,0>
    PROBLEM_HORIZON=${PROBLEM_HORIZON})
add_dependencies(casadi-dll-wrapper ${FULL_PROBLEM_NAME})
include(GenerateExportHeader)
generate_export_header(casadi-dll-wrapper
    EXPORT_FILE_NAME export/casadi-dll-wrapper-export.h)
target_include_directories(casadi-dll-wrapper PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/export>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
set_target_properties(casadi-dll-wrapper PROPERTIES
    C_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN true)
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(casadi-dll-wrapper PRIVATE "LINKER:--exclude-libs,ALL")
endif()

add_executable(benchmark-driver "driver.cpp" "params.cpp")
target_link_libraries(benchmark-driver
    PRIVATE casadi-dll-wrapper alpaqa::warnings)

add_executable(benchmark-alm-driver "alm-driver.cpp" "params.cpp")
target_link_libraries(benchmark-alm-driver
    PRIVATE casadi-dll-wrapper alpaqa::warnings)

add_executable(benchmark-new-driver "new-driver.cpp" "params.cpp")
target_link_libraries(benchmark-new-driver
    PRIVATE casadi-dll-wrapper alpaqa::warnings)

add_executable(benchmark-mpc-driver "mpc-driver.cpp" "params.cpp")
target_link_libraries(benchmark-mpc-driver
    PRIVATE casadi-dll-wrapper alpaqa::warnings)

if (ALPAQA_WITH_OCP)
    add_executable(benchmark-ocp-mpc-driver "ocp-mpc-driver.cpp" "params.cpp")
    target_link_libraries(benchmark-ocp-mpc-driver
        PRIVATE casadi-dll-wrapper alpaqa::warnings)
endif()

if (TARGET alpaqa::ipopt-adapter)
    message(STATUS "Compiling Ipopt driver")
    add_executable(benchmark-ipopt-driver
        "ipopt-driver.cpp" "ipopt-params.cpp" "params.cpp")
    target_link_libraries(benchmark-ipopt-driver PRIVATE
        alpaqa::ipopt-adapter casadi-dll-wrapper alpaqa::warnings)

    target_sources(benchmark-new-driver PRIVATE "ipopt-params.cpp")
    target_link_libraries(benchmark-new-driver PRIVATE alpaqa::ipopt-adapter)
    target_compile_definitions(benchmark-new-driver PRIVATE WITH_IPOPT)

    target_sources(benchmark-mpc-driver PRIVATE "ipopt-params.cpp")
    target_link_libraries(benchmark-mpc-driver PRIVATE alpaqa::ipopt-adapter)
    target_compile_definitions(benchmark-mpc-driver PRIVATE WITH_IPOPT)
endif()

if (TARGET alpaqa::lbfgspp-adapter)
    message(STATUS "Compiling LBFGS++ driver")

    target_sources(benchmark-new-driver PRIVATE "lbfgspp-params.cpp")
    target_link_libraries(benchmark-new-driver PRIVATE alpaqa::lbfgspp-adapter)
    target_compile_definitions(benchmark-new-driver PRIVATE WITH_LBFGSPP)

    target_sources(benchmark-mpc-driver PRIVATE "lbfgspp-params.cpp")
    target_link_libraries(benchmark-mpc-driver PRIVATE alpaqa::lbfgspp-adapter)
    target_compile_definitions(benchmark-mpc-driver PRIVATE WITH_LBFGSPP)
endif()

if (TARGET alpaqa::lbfgsb-adapter)
    message(STATUS "Compiling L-BFGS-B driver")

    target_sources(benchmark-new-driver PRIVATE "lbfgsb-params.cpp")
    target_link_libraries(benchmark-new-driver PRIVATE alpaqa::lbfgsb-adapter)
    target_compile_definitions(benchmark-new-driver PRIVATE WITH_LBFGSB)

    target_sources(benchmark-mpc-driver PRIVATE "lbfgsb-params.cpp")
    target_link_libraries(benchmark-mpc-driver PRIVATE alpaqa::lbfgsb-adapter)
    target_compile_definitions(benchmark-mpc-driver PRIVATE WITH_LBFGSB)
endif()

# More problems
find_package(Eigen3 REQUIRED)
find_package(casadi REQUIRED)

function(add_problem_module name)
    add_library(${name} MODULE "problems/${name}.cpp")
    target_link_libraries(${name} PRIVATE alpaqa::dl-api Eigen3::Eigen)
    alpaqa_configure_visibility(${name})
    set_target_properties(${name} PROPERTIES
        PREFIX ""
        RELEASE_POSTFIX ""
        DEBUG_POSTFIX ""
        RELWITHDEBINFO_POSTFIX ""
        MINSIZEREL_POSTFIX "")
    include(GenerateExportHeader)
    generate_export_header(${name}
        EXPORT_FILE_NAME export/${name}-export.h)
    target_include_directories(${name} PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/export>)
    target_compile_features(${name} PRIVATE cxx_std_20)
endfunction()

add_problem_module(lasso)
add_problem_module(sp-log-regr)
add_problem_module(quadcopter)
target_link_libraries(quadcopter PRIVATE casadi)
add_problem_module(hanging_chain)
target_link_libraries(hanging_chain PRIVATE casadi)
add_problem_module(hermans_bicycle)
target_link_libraries(hermans_bicycle PRIVATE casadi)

add_executable(ad "problems/ad.cpp")
target_link_libraries(ad PRIVATE alpaqa::alpaqa alpaqa::warnings)
