#!/bin/bash

# SDPA Kernel Build Script for amdclang++

set -e

# Configuration
BUILD_TYPE=${1:-Release}
NUM_JOBS=${2:-$(nproc)}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}

echo "Building SDPA Kernel with amdclang++"
echo "Build type: $BUILD_TYPE"
echo "Jobs: $NUM_JOBS"
echo "ROCm path: $ROCM_PATH"

# Check if amdclang++ is available
if ! command -v amdclang++ &> /dev/null; then
    echo "Error: amdclang++ not found. Please ensure ROCm is installed."
    exit 1
fi

# Check ROCm installation
if [ ! -d "$ROCM_PATH" ]; then
    echo "Error: ROCm installation not found at $ROCM_PATH"
    exit 1
fi

# Create build directory
BUILD_DIR="build-${BUILD_TYPE,,}"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DROCM_PATH=$ROCM_PATH \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
make -j$NUM_JOBS

echo "Build completed successfully!"
echo "Executables and libraries are in: $BUILD_DIR"

# Run tests if requested
if [ "$3" == "test" ]; then
    echo "Running tests..."
    ctest --output-on-failure
fi