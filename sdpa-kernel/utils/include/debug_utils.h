#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <string>

namespace sdpa {
namespace utils {
namespace debug {

// Debug levels
enum class DebugLevel {
    NONE = 0,
    ERROR = 1,
    WARNING = 2,
    INFO = 3,
    DEBUG = 4,
    VERBOSE = 5
};

// Global debug level (can be set at runtime)
extern DebugLevel g_debug_level;

// Debug macros
#define DEBUG_PRINT(level, fmt, ...) \
    do { \
        if (static_cast<int>(level) <= static_cast<int>(sdpa::utils::debug::g_debug_level)) { \
            printf("[%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        } \
    } while(0)

#define DEBUG_ERROR(fmt, ...) DEBUG_PRINT(sdpa::utils::debug::DebugLevel::ERROR, "ERROR: " fmt, ##__VA_ARGS__)
#define DEBUG_WARNING(fmt, ...) DEBUG_PRINT(sdpa::utils::debug::DebugLevel::WARNING, "WARNING: " fmt, ##__VA_ARGS__)
#define DEBUG_INFO(fmt, ...) DEBUG_PRINT(sdpa::utils::debug::DebugLevel::INFO, "INFO: " fmt, ##__VA_ARGS__)
#define DEBUG_DEBUG(fmt, ...) DEBUG_PRINT(sdpa::utils::debug::DebugLevel::DEBUG, "DEBUG: " fmt, ##__VA_ARGS__)
#define DEBUG_VERBOSE(fmt, ...) DEBUG_PRINT(sdpa::utils::debug::DebugLevel::VERBOSE, "VERBOSE: " fmt, ##__VA_ARGS__)

// HIP error checking
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            DEBUG_ERROR("HIP error at %s:%d - %s", __FILE__, __LINE__, hipGetErrorString(error)); \
            return error; \
        } \
    } while(0)

#define HIP_CHECK_THROW(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            DEBUG_ERROR("HIP error at %s:%d - %s", __FILE__, __LINE__, hipGetErrorString(error)); \
            throw std::runtime_error(hipGetErrorString(error)); \
        } \
    } while(0)

// Device info utilities
struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int warp_size;
    int max_threads_per_block;
    int max_blocks_per_multiprocessor;
    
    void print() const;
};

// Get current device information
DeviceInfo getCurrentDeviceInfo();

// Memory debugging utilities
class MemoryDebugger {
private:
    struct AllocationInfo {
        void* ptr;
        size_t size;
        std::string file;
        int line;
        bool is_freed;
    };
    
    std::vector<AllocationInfo> allocations_;
    size_t total_allocated_;
    size_t peak_allocated_;
    bool tracking_enabled_;
    
public:
    MemoryDebugger();
    ~MemoryDebugger();
    
    void enable_tracking(bool enable = true);
    void record_allocation(void* ptr, size_t size, const char* file, int line);
    void record_deallocation(void* ptr);
    
    void print_summary() const;
    void print_leaks() const;
    void reset_stats();
    
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_peak_allocated() const { return peak_allocated_; }
    size_t get_current_allocated() const;
};

// Global memory debugger instance
extern MemoryDebugger g_memory_debugger;

// Memory debugging macros
#define DEBUG_MALLOC(ptr, size) \
    do { \
        sdpa::utils::debug::g_memory_debugger.record_allocation(ptr, size, __FILE__, __LINE__); \
    } while(0)

#define DEBUG_FREE(ptr) \
    do { \
        sdpa::utils::debug::g_memory_debugger.record_deallocation(ptr); \
    } while(0)

// Kernel launch debugging
struct KernelLaunchInfo {
    std::string kernel_name;
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem_size;
    double execution_time_ms;
    
    void print() const;
};

// Kernel timing utilities
class KernelTimer {
private:
    hipEvent_t start_event_;
    hipEvent_t stop_event_;
    bool timing_active_;
    
public:
    KernelTimer();
    ~KernelTimer();
    
    void start();
    void stop();
    float get_elapsed_time_ms() const;
};

// Kernel profiling
#define PROFILE_KERNEL(kernel_name, grid, block, shared_mem, stream, ...) \
    do { \
        sdpa::utils::debug::KernelTimer timer; \
        timer.start(); \
        kernel_name<<<grid, block, shared_mem, stream>>>(__VA_ARGS__); \
        timer.stop(); \
        DEBUG_INFO("Kernel %s executed in %.3f ms", #kernel_name, timer.get_elapsed_time_ms()); \
    } while(0)

// Tensor debugging utilities
template<typename T>
void print_tensor_info(const T* data, const std::string& name, 
                      const std::vector<int>& shape, bool print_values = false);

template<typename T>
void print_tensor_stats(const T* data, int size, const std::string& name);

// Device-side debugging
__device__ void print_thread_info();
__device__ void print_block_info();
__device__ void print_bfloat16(__hip_bfloat16 val, const char* name = "value");

// Validation utilities
bool validate_tensor_shape(const std::vector<int>& shape);
bool validate_attention_inputs(int batch_size, int num_heads, int seq_len, int head_dim);

template<typename T>
bool check_for_nan_inf(const T* data, int size, const std::string& tensor_name);

template<typename T>
bool compare_tensors(const T* a, const T* b, int size, float tolerance = 1e-5f,
                    const std::string& name_a = "tensor_a", const std::string& name_b = "tensor_b");

// Performance monitoring
class PerformanceMonitor {
private:
    struct Metric {
        std::string name;
        double value;
        std::string unit;
    };
    
    std::vector<Metric> metrics_;
    
public:
    void add_metric(const std::string& name, double value, const std::string& unit = "");
    void print_metrics() const;
    void reset_metrics();
    
    // Common performance metrics
    void record_bandwidth(size_t bytes_transferred, double time_ms);
    void record_flops(size_t operations, double time_ms);
    void record_memory_usage(size_t bytes_used);
};

extern PerformanceMonitor g_performance_monitor;

} // namespace debug
} // namespace utils
} // namespace sdpa

#endif // DEBUG_UTILS_H