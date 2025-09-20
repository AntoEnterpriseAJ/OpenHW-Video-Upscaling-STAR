#!/bin/bash

# Clean build artifacts

echo "Cleaning SDPA Kernel build artifacts..."

# Remove build directories
rm -rf build*
rm -rf cmake-build-*

# Remove generated files
find . -name "*.o" -delete
find . -name "*.a" -delete
find . -name "*.so" -delete
find . -name "CMakeCache.txt" -delete
find . -name "cmake_install.cmake" -delete
find . -name "Makefile" -delete
find . -name "compile_commands.json" -delete

# Remove CMake generated directories
find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Clean completed!"