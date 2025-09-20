cmake_minimum_required(VERSION 3.21)

project(sdpa_examples LANGUAGES CXX)

# Ensure /usr/bin/amdclang++ is used
set(CMAKE_CXX_COMPILER "/usr/bin/amdclang++")

# Find all example source files
file(GLOB EXAMPLE_SOURCES "*.cpp" "*.cc")

# Create example executables
foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    # Get the example name from file path
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    
    # Create executable
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    
    # High-performance compiler flags
    target_compile_options(${EXAMPLE_NAME} PRIVATE
        -O2
        --offload-arch=gfx1100
        -ffast-math
        -funroll-loops
        -Wall
        -Wextra
    )
    
    # Include directories
    target_include_directories(${EXAMPLE_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
        ${ROCM_PATH}/include
    )
    
    # Link libraries
    target_link_libraries(${EXAMPLE_NAME} 
        sdpa_kernel
        amdhip64
    )
    
    # Set properties
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    
    # Installation
    install(TARGETS ${EXAMPLE_NAME}
        RUNTIME DESTINATION bin/examples
    )
endforeach()