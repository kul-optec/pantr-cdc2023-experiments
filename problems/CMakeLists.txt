find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Uses Python to generate a CasADi problem
function(generate_casadi_problem problem pythonfile formulation horizon)
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
                        BYPRODUCTS "${problem_name}.csv"
                        MAIN_DEPENDENCY "${pythonfile}")
        add_library("${problem_name}" MODULE "${problem_name}.c")
        set_target_properties("${problem_name}" PROPERTIES
                            PREFIX "" RELEASE_POSTFIX "" DEBUG_POSTFIX ""
                            RELWITHDEBINFO_POSTFIX "" MINSIZEREL_POSTFIX ""
                            OUTPUT_NAME ${problem_name})
        add_custom_command(TARGET "${problem_name}" POST_BUILD
                        COMMAND ${CMAKE_COMMAND}
                        ARGS -E copy
                            "${problem_name}.csv"
                            "${problem_name}.guess.csv"
                            "$<TARGET_FILE_DIR:${problem_name}>")
    endif()
endfunction()

# Compile the CasADi problems
foreach(N RANGE 1 60 1)
    generate_casadi_problem("quadcopter" "generate_problem.py" "ss2p" ${N})
    # generate_casadi_problem("quadcopter" "generate_problem.py" "ss2" ${N})
    generate_casadi_problem("hanging_chain" "generate_problem.py" "ss2p" ${N})
    # fails to compile (too big)
    # generate_casadi_problem("hanging_chain" "generate_problem.py" "ss2" ${N})
endforeach()

# C++ problems
find_package(Eigen3 REQUIRED)
find_package(alpaqa REQUIRED)
find_package(casadi REQUIRED)

function(add_problem_module name)
    add_library(${name} MODULE "problems/${name}.cpp")
    target_link_libraries(${name} PRIVATE alpaqa::dl-api Eigen3::Eigen)
    pantr_configure_visibility(${name})
    set_target_properties(${name} PROPERTIES
        PREFIX "" RELEASE_POSTFIX "" DEBUG_POSTFIX "" RELWITHDEBINFO_POSTFIX ""
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
