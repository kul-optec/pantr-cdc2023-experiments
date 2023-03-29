cmake_minimum_required(VERSION 3.24)
project(pantr-cdc2023-experiments LANGUAGES CXX C Fortran)
list(APPEND CMAKE_FIND_ROOT_PATH "${PROJECT_SOURCE_DIR}/alpaqa/staging")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/alpaqa/cmake/find")

function(pantr_configure_visibility target)
    set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                               C_VISIBILITY_PRESET "hidden"
                                               VISIBILITY_INLINES_HIDDEN true)
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_options(${target} PRIVATE
            $<$<LINK_LANGUAGE:C,CXX>:LINKER:--exclude-libs,ALL>)
    endif()
endfunction()

add_subdirectory(problems)
add_subdirectory(drivers)
