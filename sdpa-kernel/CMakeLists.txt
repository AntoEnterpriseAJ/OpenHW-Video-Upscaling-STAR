cmake_minimum_required(VERSION 3.21)

project(sdpa_kernel 
    VERSION 1.0.0 
    DESCRIPTION "Scaled Dot-Product Attention Kernel with HIP and BFloat16"
    LANGUAGES CXX
)

add_definitions(-D__HIP_PLATFORM_AMD__)

# Force amdclang++ compiler
set(CMAKE_CXX_COMPILER "/usr/bin/amdclang++")
set(CMAKE_HIP_COMPILER "/opt/rocm-6.4.1/bin/amdclang++")

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find ROCm installation
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
endif()

# Verify ROCm path exists
if(NOT EXISTS ${ROCM_PATH})
    message(FATAL_ERROR "ROCm installation not found at ${ROCM_PATH}")
endif()

message(STATUS "Using ROCm at: ${ROCM_PATH}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")

# Global compiler and linker flags
set(AMD_ARCH_FLAGS "-x hip --offload-arch=gfx1100")
set(AMD_LIB_FLAGS "-lamdhip64 -L${ROCM_PATH}/lib")
set(AMD_OPT_FLAGS "-O2")

# Set compile flags for all targets
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AMD_OPT_FLAGS} ${AMD_ARCH_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${AMD_LIB_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${AMD_LIB_FLAGS}")

# Include directories
include_directories(
    ${ROCM_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/include
)

# Link directories
link_directories(${ROCM_PATH}/lib)

# Build options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Add subdirectories
add_subdirectory(common)
add_subdirectory(utils)
add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Main library target
add_library(sdpa_kernel INTERFACE)
target_link_libraries(sdpa_kernel 
    INTERFACE 
        sdpa_common
        sdpa_utils
        sdpa_src
        amdhip64
)

# Installation
install(TARGETS sdpa_kernel
    EXPORT sdpa_kernel_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(EXPORT sdpa_kernel_targets
    FILE sdpa_kernel_targets.cmake
    NAMESPACE sdpa::
    DESTINATION lib/cmake/sdpa_kernel
)

# Configuration summary
message(STATUS "")
message(STATUS "=== SDPA Kernel Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "ROCm path: ${ROCM_PATH}")
message(STATUS "Architecture: gfx1100")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "=================================")