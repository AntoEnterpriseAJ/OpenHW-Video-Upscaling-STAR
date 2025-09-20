cmake_minimum_required(VERSION 3.21)

project(sdpa_common LANGUAGES CXX)

# Ensure amdclang++ is used
set(CMAKE_CXX_COMPILER "/usr/bin/amdclang++")

# Source files
file(GLOB_RECURSE COMMON_SOURCES 
    "src/*.cpp"
    "src/*.cc"
)

file(GLOB_RECURSE COMMON_HEADERS 
    "include/*.h"
    "include/*.hpp"
)

# Create library
add_library(sdpa_common STATIC ${COMMON_SOURCES} ${COMMON_HEADERS})

add_definitions(-D__HIP_PLATFORM_AMD__)

# Compiler flags
target_compile_options(sdpa_common PRIVATE
    -O2
    --offload-arch=gfx1100
    -Wall
    -Wextra
    -fPIC
)

# Include directories
target_include_directories(sdpa_common 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/sdpa/common>
        ${ROCM_PATH}/include
)

# Link libraries
target_link_libraries(sdpa_common 
    PUBLIC 
        amdhip64
)

# Properties
set_target_properties(sdpa_common PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Installation
install(TARGETS sdpa_common
    EXPORT sdpa_common_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/sdpa/common
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(EXPORT sdpa_common_targets
    FILE sdpa_common_targets.cmake
    NAMESPACE sdpa::
    DESTINATION lib/cmake/sdpa_common
)
