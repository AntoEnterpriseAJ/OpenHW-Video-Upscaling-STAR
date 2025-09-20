#ifndef PLATFORM_H
#define PLATFORM_H

#include <hip/hip_runtime.h>
#include <string>

namespace sdpa {
namespace common {

// Platform detection
enum class Platform {
    UNKNOWN,
    AMD_ROCM,
    NVIDIA_CUDA,
    INTEL_ONEAPI
};

enum class Architecture {
    UNKNOWN,
    GCN,
    CDNA,
    CDNA2,
    RDNA,
    RDNA2,
    RDNA3
};

struct PlatformInfo {
    Platform platform;
    Architecture architecture;
    std::string device_name;
    int major_version;
    int minor_version;
    int compute_units;
    size_t global_memory_size;
    size_t shared_memory_per_block;
    int warp_size;
    bool supports_bfloat16;
    bool supports_fp16;
    bool supports_int8;
    bool supports_cooperative_groups;
    
    void print() const;
};

// Platform detection functions
Platform detectPlatform();
Architecture detectArchitecture();
PlatformInfo getPlatformInfo();

// Capability queries
bool supportsBFloat16();
bool supportsFP16();
bool supportsInt8();
bool supportsCooperativeGroups();
bool supportsAtomicAdd();
bool supportsFastMath();

// Performance characteristics
int getOptimalBlockSize(int problem_size);
int getOptimalGridSize(int problem_size, int block_size);
size_t getMaxSharedMemorySize();
int getMaxRegistersPerThread();
int getMaxThreadsPerBlock();

// Memory bandwidth estimation
double estimateMemoryBandwidth();
double estimateComputeThroughput();

} // namespace common
} // namespace sdpa

#endif // PLATFORM_H