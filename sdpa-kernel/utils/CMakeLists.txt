cmake_minimum_required(VERSION 3.21)

project(sdpa_utils LANGUAGES CXX)

# Ensure amdclang++ is used
set(CMAKE_CXX_COMPILER "/usr/bin/amdclang++")

# Source files
file(GLOB_RECURSE UTILS_SOURCES 
    "src/*.cpp"
    "src/*.cc"
)

file(GLOB_RECURSE UTILS_HEADERS 
    "include/*.h"
    "include/*.hpp"
)

# Create library
add_library(sdpa_utils STATIC ${UTILS_SOURCES} ${UTILS_HEADERS})

add_definitions(-D__HIP_PLATFORM_AMD__)


# Compiler flags for maximum performance
target_compile_options(sdpa_utils PRIVATE
    -O2
    --offload-arch=gfx1100
    -ffast-math
    -funroll-loops
    -Wall
    -Wextra
    -fPIC
    -march=native
)

# Include directories
target_include_directories(sdpa_utils 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/sdpa/utils>
        # ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
        ${ROCM_PATH}/include
)

# Link libraries
target_link_libraries(sdpa_utils 
    PUBLIC 
        sdpa_common
        amdhip64
)

# Properties
set_target_properties(sdpa_utils PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Installation
install(TARGETS sdpa_utils
    EXPORT sdpa_utils_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/sdpa/utils
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(EXPORT sdpa_utils_targets
    FILE sdpa_utils_targets.cmake
    NAMESPACE sdpa::
    DESTINATION lib/cmake/sdpa_utils
)