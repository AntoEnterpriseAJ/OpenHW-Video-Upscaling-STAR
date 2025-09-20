#include "../include/platform.h"
#include <iostream>
#include <cstring>

namespace sdpa {
namespace common {

Platform detectPlatform() {
    int device_count = 0;
    hipError_t error = hipGetDeviceCount(&device_count);
    
    if (error != hipSuccess || device_count == 0) {
        return Platform::UNKNOWN;
    }
    
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
        printf("Failed to get device properties\n");
        exit(23);
    }
    
    // Check for AMD ROCm
    if (strstr(props.name, "gfx") != nullptr || 
        strstr(props.name, "AMD") != nullptr ||
        strstr(props.name, "Radeon") != nullptr) {
        return Platform::AMD_ROCM;
    }
    
    // For HIP, we're primarily targeting AMD ROCm
    return Platform::AMD_ROCM;
}

Architecture detectArchitecture() {
    int device_count = 0;
    hipError_t error = hipGetDeviceCount(&device_count);
    
    if (error != hipSuccess || device_count == 0) {
        return Architecture::UNKNOWN;
    }
    
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
        printf("Failed to get device properties\n");
        exit(23);
    }
    // AMD architecture detection based on device name
    if (strstr(props.name, "gfx908") != nullptr || 
        strstr(props.name, "MI100") != nullptr) {
        return Architecture::CDNA;
    }
    
    if (strstr(props.name, "gfx90a") != nullptr || 
        strstr(props.name, "MI200") != nullptr ||
        strstr(props.name, "MI210") != nullptr ||
        strstr(props.name, "MI250") != nullptr) {
        return Architecture::CDNA2;
    }
    
    if (strstr(props.name, "gfx10") != nullptr ||
        strstr(props.name, "RX 6") != nullptr) {
        return Architecture::RDNA2;
    }
    
    if (strstr(props.name, "gfx11") != nullptr ||
        strstr(props.name, "RX 7") != nullptr) {
        return Architecture::RDNA3;
    }
    
    // Default to GCN for older AMD cards
    if (strstr(props.name, "gfx") != nullptr) {
        return Architecture::GCN;
    }
    
    return Architecture::UNKNOWN;
}

PlatformInfo getPlatformInfo() {
    PlatformInfo info;
    info.platform = detectPlatform();
    info.architecture = detectArchitecture();
    
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
        info.device_name = std::string(props.name);
        info.major_version = props.major;
        info.minor_version = props.minor;
        info.compute_units = props.multiProcessorCount;
    }
    else {
        printf("Failed to get device properties\n");
        exit(23);
    }
    info.global_memory_size = props.totalGlobalMem;
    info.shared_memory_per_block = props.sharedMemPerBlock;
    info.warp_size = props.warpSize;
    
    // Feature detection
    info.supports_bfloat16 = (info.architecture == Architecture::CDNA2 || 
                             info.architecture == Architecture::RDNA3);
    info.supports_fp16 = (info.major_version >= 9);  // GFX9 and later
    info.supports_int8 = (info.architecture == Architecture::CDNA || 
                         info.architecture == Architecture::CDNA2);
    info.supports_cooperative_groups = (info.major_version >= 9);
    
    return info;
}

void PlatformInfo::print() const {
    std::cout << "=== Platform Information ===" << std::endl;
    std::cout << "Platform: ";
    switch (platform) {
        case Platform::AMD_ROCM: std::cout << "AMD ROCm"; break;
        case Platform::NVIDIA_CUDA: std::cout << "NVIDIA CUDA"; break;
        case Platform::INTEL_ONEAPI: std::cout << "Intel OneAPI"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;
    
    std::cout << "Architecture: ";
    switch (architecture) {
        case Architecture::GCN: std::cout << "GCN"; break;
        case Architecture::CDNA: std::cout << "CDNA"; break;
        case Architecture::CDNA2: std::cout << "CDNA2"; break;
        case Architecture::RDNA: std::cout << "RDNA"; break;
        case Architecture::RDNA2: std::cout << "RDNA2"; break;
        case Architecture::RDNA3: std::cout << "RDNA3"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;
    
    std::cout << "Device: " << device_name << std::endl;
    std::cout << "Compute Version: " << major_version << "." << minor_version << std::endl;
    std::cout << "Compute Units: " << compute_units << std::endl;
    std::cout << "Global Memory: " << (global_memory_size / (1024 * 1024 * 1024)) << " GB" << std::endl;
    std::cout << "Shared Memory per Block: " << (shared_memory_per_block / 1024) << " KB" << std::endl;
    std::cout << "Warp Size: " << warp_size << std::endl;
    
    std::cout << "Feature Support:" << std::endl;
    std::cout << "  BFloat16: " << (supports_bfloat16 ? "Yes" : "No") << std::endl;
    std::cout << "  FP16: " << (supports_fp16 ? "Yes" : "No") << std::endl;
    std::cout << "  INT8: " << (supports_int8 ? "Yes" : "No") << std::endl;
    std::cout << "  Cooperative Groups: " << (supports_cooperative_groups ? "Yes" : "No") << std::endl;
    std::cout << "=============================" << std::endl;
}

bool supportsBFloat16() {
    PlatformInfo info = getPlatformInfo();
    return info.supports_bfloat16;
}

bool supportsFP16() {
    PlatformInfo info = getPlatformInfo();
    return info.supports_fp16;
}

bool supportsInt8() {
    PlatformInfo info = getPlatformInfo();
    return info.supports_int8;
}

bool supportsCooperativeGroups() {
    PlatformInfo info = getPlatformInfo();
    return info.supports_cooperative_groups;
}

bool supportsAtomicAdd() {
    // Most modern AMD GPUs support atomic operations
    return true;
}

bool supportsFastMath() {
    // Fast math is generally available on AMD GPUs
    return true;
}

int getOptimalBlockSize(int problem_size) {
    PlatformInfo info = getPlatformInfo();
    
    // For small problems, use smaller blocks
    if (problem_size < 1024) {
        return 64;
    } else if (problem_size < 4096) {
        return 128;
    } else {
        return 256;
    }
}

int getOptimalGridSize(int problem_size, int block_size) {
    PlatformInfo info = getPlatformInfo();
    int num_blocks = (problem_size + block_size - 1) / block_size;
    
    // Limit grid size to maximize occupancy
    int max_blocks = info.compute_units * 4;  // 4 blocks per CU as a heuristic
    return std::min(num_blocks, max_blocks);
}

size_t getMaxSharedMemorySize() {
    PlatformInfo info = getPlatformInfo();
    return info.shared_memory_per_block;
}

int getMaxRegistersPerThread() {
    // AMD GPUs typically have 256 VGPRs available
    return 256;
}

int getMaxThreadsPerBlock() {
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    return props.maxThreadsPerBlock;
}

double estimateMemoryBandwidth() {
    PlatformInfo info = getPlatformInfo();
    
    // Rough estimates based on architecture
    switch (info.architecture) {
        case Architecture::CDNA2:
            return 1600.0;  // GB/s for MI250X
        case Architecture::CDNA:
            return 1200.0;  // GB/s for MI100
        case Architecture::RDNA3:
            return 960.0;   // GB/s for RX 7900 XTX
        case Architecture::RDNA2:
            return 512.0;   // GB/s for RX 6800 XT
        default:
            return 400.0;   // Conservative estimate
    }
}

double estimateComputeThroughput() {
    PlatformInfo info = getPlatformInfo();
    
    // Rough TFLOPS estimates for FP16/BF16
    switch (info.architecture) {
        case Architecture::CDNA2:
            return 380.0;   // TFLOPS for MI250X
        case Architecture::CDNA:
            return 185.0;   // TFLOPS for MI100
        case Architecture::RDNA3:
            return 123.0;   // TFLOPS for RX 7900 XTX
        case Architecture::RDNA2:
            return 41.0;    // TFLOPS for RX 6800 XT
        default:
            return 20.0;    // Conservative estimate
    }
}

} // namespace common
} // namespace sdpa