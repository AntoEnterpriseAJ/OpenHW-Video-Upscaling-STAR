#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstddef>

namespace sdpa {
namespace utils {

// Memory alignment constants
constexpr size_t WARP_SIZE = 64;
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t MEMORY_ALIGN = 128;

// HIP memory management utilities
class HipMemoryManager {
public:
    // Device memory allocation with alignment
    template<typename T>
    static hipError_t allocateDevice(T** ptr, size_t count, size_t alignment = MEMORY_ALIGN);
    
    // Host memory allocation (pinned)
    template<typename T>
    static hipError_t allocateHost(T** ptr, size_t count);
    
    // Unified memory allocation
    template<typename T>
    static hipError_t allocateManaged(T** ptr, size_t count);
    
    // Memory copy utilities
    template<typename T>
    static hipError_t copyHostToDevice(T* dst, const T* src, size_t count);
    
    template<typename T>
    static hipError_t copyDeviceToHost(T* dst, const T* src, size_t count);
    
    template<typename T>
    static hipError_t copyDeviceToDevice(T* dst, const T* src, size_t count);
    
    // Memory deallocation
    static hipError_t deallocateDevice(void* ptr);
    static hipError_t deallocateHost(void* ptr);
    
    // Memory info
    static hipError_t getMemoryInfo(size_t* free, size_t* total);
    
    // Memory prefetch for managed memory
    template<typename T>
    static hipError_t prefetchToDevice(T* ptr, size_t count, int device);
    
    template<typename T>
    static hipError_t prefetchToHost(T* ptr, size_t count);
};

// Memory pool for efficient allocation/deallocation
class MemoryPool {
private:
    struct Block {
        void* ptr;
        size_t size;
        bool is_free;
        Block* next;
    };
    
    Block* head_;
    size_t total_size_;
    size_t used_size_;
    
public:
    explicit MemoryPool(size_t initial_size);
    ~MemoryPool();
    
    void* allocate(size_t size, size_t alignment = MEMORY_ALIGN);
    void deallocate(void* ptr);
    void reset();
    
    size_t getTotalSize() const { return total_size_; }
    size_t getUsedSize() const { return used_size_; }
    size_t getFreeSize() const { return total_size_ - used_size_; }
};

// Shared memory utilities for kernels
template<typename T>
__device__ T* getSharedMemory();

template<>
__device__ __hip_bfloat16* getSharedMemory<__hip_bfloat16>();

template<>
__device__ float* getSharedMemory<float>();

// Memory pattern utilities
__device__ void prefetchL1(const void* ptr);
__device__ void prefetchL2(const void* ptr);
__device__ void invalidateCache(const void* ptr, size_t size);

} // namespace utils
} // namespace sdpa

#endif // MEMORY_UTILS_H
