cmake_minimum_required(VERSION 3.21)

project(sdpa_tests LANGUAGES CXX)

# Ensure amdclang++ is used
set(CMAKE_CXX_COMPILER "/usr/bin/amdclang++")

# Find all test source files
file(GLOB_RECURSE TEST_SOURCES 
    "unit/*.cpp"
    "integration/*.cpp"
)

# Create test executables
foreach(TEST_SOURCE ${TEST_SOURCES})
    # Get the test name from file path
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # Create executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # Compiler flags
    target_compile_options(${TEST_NAME} PRIVATE
        -O2
        --offload-arch=gfx1100
        -Wall
        -Wextra
        -g
    )
    
    # Include directories
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
        ${ROCM_PATH}/include
    )
    
    # Link libraries
    target_link_libraries(${TEST_NAME} 
        sdpa_kernel
        amdhip64
    )
    
    # Add to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    # Set test properties
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 300
        ENVIRONMENT "HIP_VISIBLE_DEVICES=0"
    )
endforeach()

# Global test configuration
set_property(DIRECTORY PROPERTY 
    COMPILE_DEFINITIONS HIP_PLATFORM_AMD=1
)
