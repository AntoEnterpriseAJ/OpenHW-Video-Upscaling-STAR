#include "memory_utils.h"
#include <cassert>
#include <cstdlib>
#include <new>

namespace sdpa {
namespace utils {

template <typename T>
hipError_t HipMemoryManager::allocateDevice(T **ptr, size_t count,
                                            size_t alignment) {
  size_t size = count * sizeof(T);
  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
  return hipMalloc(reinterpret_cast<void **>(ptr), aligned_size);
}

template <typename T>
hipError_t HipMemoryManager::allocateHost(T **ptr, size_t count) {
  size_t size = count * sizeof(T);
  return hipHostMalloc(reinterpret_cast<void **>(ptr), size,
                       hipHostMallocDefault);
}

template <typename T>
hipError_t HipMemoryManager::allocateManaged(T **ptr, size_t count) {
  size_t size = count * sizeof(T);
  return hipMallocManaged(reinterpret_cast<void **>(ptr), size,
                          hipMemAttachGlobal);
}

template <typename T>
hipError_t HipMemoryManager::copyHostToDevice(T *dst, const T *src,
                                              size_t count) {
  return hipMemcpy(dst, src, count * sizeof(T), hipMemcpyHostToDevice);
}

template <typename T>
hipError_t HipMemoryManager::copyDeviceToHost(T *dst, const T *src,
                                              size_t count) {
  return hipMemcpy(dst, src, count * sizeof(T), hipMemcpyDeviceToHost);
}

template <typename T>
hipError_t HipMemoryManager::copyDeviceToDevice(T *dst, const T *src,
                                                size_t count) {
  return hipMemcpy(dst, src, count * sizeof(T), hipMemcpyDeviceToDevice);
}

hipError_t HipMemoryManager::deallocateDevice(void *ptr) {
  return hipFree(ptr);
}

hipError_t HipMemoryManager::deallocateHost(void *ptr) {
  return hipHostFree(ptr);
}

hipError_t HipMemoryManager::getMemoryInfo(size_t *free, size_t *total) {
  return hipMemGetInfo(free, total);
}

template <typename T>
hipError_t HipMemoryManager::prefetchToDevice(T *ptr, size_t count,
                                              int device) {
  return hipMemPrefetchAsync(ptr, count * sizeof(T), device, nullptr);
}

template <typename T>
hipError_t HipMemoryManager::prefetchToHost(T *ptr, size_t count) {
  return hipMemPrefetchAsync(ptr, count * sizeof(T), hipCpuDeviceId, nullptr);
}

// Explicit template instantiations
template hipError_t
HipMemoryManager::allocateDevice<hip_bfloat16>(hip_bfloat16 **, size_t,
                                                 size_t);
template hipError_t HipMemoryManager::allocateDevice<float>(float **, size_t,
                                                            size_t);
template hipError_t
HipMemoryManager::allocateHost<hip_bfloat16>(hip_bfloat16 **, size_t);
template hipError_t HipMemoryManager::allocateHost<float>(float **, size_t);

MemoryPool::MemoryPool(size_t initial_size)
    : total_size_(initial_size), used_size_(0) {
  hipError_t error =
      hipMalloc(reinterpret_cast<void **>(&head_), sizeof(Block));
  if (error != hipSuccess) {
    throw std::bad_alloc();
  }

  void *pool_memory;
  error = hipMalloc(&pool_memory, initial_size);
  if (error != hipSuccess) {
    hipFree(head_);
    throw std::bad_alloc();
  }

  head_->ptr = pool_memory;
  head_->size = initial_size;
  head_->is_free = true;
  head_->next = nullptr;
}

MemoryPool::~MemoryPool() {
  Block *current = head_;
  while (current) {
    Block *next = current->next;
    if (current->ptr) {
      hipFree(current->ptr);
    }
    hipFree(current);
    current = next;
  }
}

void *MemoryPool::allocate(size_t size, size_t alignment) {
  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

  Block *current = head_;
  while (current) {
    if (current->is_free && current->size >= aligned_size) {
      current->is_free = false;
      used_size_ += aligned_size;

      // Split block if necessary
      if (current->size > aligned_size + sizeof(Block)) {
        Block *new_block;
        hipMalloc(reinterpret_cast<void **>(&new_block), sizeof(Block));
        new_block->ptr = static_cast<char *>(current->ptr) + aligned_size;
        new_block->size = current->size - aligned_size;
        new_block->is_free = true;
        new_block->next = current->next;
        current->next = new_block;
        current->size = aligned_size;
      }

      return current->ptr;
    }
    current = current->next;
  }

  return nullptr; // No suitable block found
}

void MemoryPool::deallocate(void *ptr) {
  Block *current = head_;
  while (current) {
    if (current->ptr == ptr) {
      current->is_free = true;
      used_size_ -= current->size;

      // Merge with next block if free
      if (current->next && current->next->is_free) {
        Block *next = current->next;
        current->size += next->size;
        current->next = next->next;
        hipFree(next);
      }

      break;
    }
    current = current->next;
  }
}

void MemoryPool::reset() {
  Block *current = head_;
  while (current) {
    current->is_free = true;
    current = current->next;
  }
  used_size_ = 0;
}

// Device-side shared memory specializations
template <> __device__ hip_bfloat16 *getSharedMemory<hip_bfloat16>() {
  extern __shared__ hip_bfloat16 shared_bf16[];
  return shared_bf16;
}

template <> __device__ float *getSharedMemory<float>() {
  extern __shared__ float shared_float[];
  return shared_float;
}

// Cache management
__device__ void prefetchL1(const void *ptr) { __builtin_prefetch(ptr, 0, 1); }

__device__ void prefetchL2(const void *ptr) { __builtin_prefetch(ptr, 0, 2); }

__device__ void invalidateCache(const void *ptr, size_t size) {
  // HIP specific cache invalidation if available
  // This is a placeholder - actual implementation depends on architecture
}

} // namespace utils
} // namespace sdpa