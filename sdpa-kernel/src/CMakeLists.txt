cmake_minimum_required(VERSION 3.21)

project(sdpa_src LANGUAGES CXX)

# Ensure amdclang++ is used
set(CMAKE_CXX_COMPILER "/usr/bin/amdclang++")

# Source files
file(GLOB_RECURSE SRC_SOURCES 
    "*.cpp"
    "*.cc"
    "kernel/*.cpp"
    "kernel/*.cc"
)

file(GLOB_RECURSE SRC_HEADERS 
    "../include/*.h"
    "../include/*.hpp"
    "*.h"
    "*.hpp"
    "kernel/*.h"
    "kernel/*.hpp"
)

# Create library
add_library(sdpa_src STATIC ${SRC_SOURCES} ${SRC_HEADERS})

# High-performance compiler flags
target_compile_options(sdpa_src PRIVATE
    -O2
    --offload-arch=gfx1100
    -ffast-math
    -funroll-loops
    -fno-finite-math-only
    -munsafe-fp-atomics
    -Wall
    -Wextra
    -fPIC
    -march=native
)

# Include directories
target_include_directories(sdpa_src 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include/sdpa>
        # ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
        # ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
        ${ROCM_PATH}/include
)

# Link libraries
target_link_libraries(sdpa_src 
    PUBLIC 
        sdpa_common
        sdpa_utils
        amdhip64
)

# Properties
set_target_properties(sdpa_src PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Installation
install(TARGETS sdpa_src
    EXPORT sdpa_src_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ../include/
    DESTINATION include/sdpa
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(EXPORT sdpa_src_targets
    FILE sdpa_src_targets.cmake
    NAMESPACE sdpa::
    DESTINATION lib/cmake/sdpa_src
)