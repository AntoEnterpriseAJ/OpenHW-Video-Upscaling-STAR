#include "debug_utils.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace sdpa {
namespace utils {
namespace debug {

// Global debug level
DebugLevel g_debug_level = DebugLevel::INFO;

// Global memory debugger
MemoryDebugger g_memory_debugger;

// Global performance monitor
PerformanceMonitor g_performance_monitor;

void DeviceInfo::print() const {
    std::cout << "=== Device Information ===" << std::endl;
    std::cout << "Device ID: " << device_id << std::endl;
    std::cout << "Name: " << name << std::endl;
    std::cout << "Total Memory: " << (total_memory / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Free Memory: " << (free_memory / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Compute Capability: " << compute_capability_major << "." << compute_capability_minor << std::endl;
    std::cout << "Multiprocessor Count: " << multiprocessor_count << std::endl;
    std::cout << "Warp Size: " << warp_size << std::endl;
    std::cout << "Max Threads per Block: " << max_threads_per_block << std::endl;
    std::cout << "Max Blocks per Multiprocessor: " << max_blocks_per_multiprocessor << std::endl;
    std::cout << "==========================" << std::endl;
}

DeviceInfo getCurrentDeviceInfo() {
    DeviceInfo info;
    
    // Get current device
    hipGetDevice(&info.device_id);
    
    // Get device properties
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, info.device_id);
    
    info.name = std::string(props.name);
    info.total_memory = props.totalGlobalMem;
    info.compute_capability_major = props.major;
    info.compute_capability_minor = props.minor;
    info.multiprocessor_count = props.multiProcessorCount;
    info.warp_size = props.warpSize;
    info.max_threads_per_block = props.maxThreadsPerBlock;
    info.max_blocks_per_multiprocessor = props.maxBlocksPerMultiProcessor;
    
    // Get memory info
    hipMemGetInfo(&info.free_memory, &info.total_memory);
    
    return info;
}

MemoryDebugger::MemoryDebugger() 
    : total_allocated_(0), peak_allocated_(0), tracking_enabled_(false) {
}

MemoryDebugger::~MemoryDebugger() {
    if (tracking_enabled_) {
        print_leaks();
    }
}

void MemoryDebugger::enable_tracking(bool enable) {
    tracking_enabled_ = enable;
}

void MemoryDebugger::record_allocation(void* ptr, size_t size, const char* file, int line) {
    if (!tracking_enabled_) return;
    
    AllocationInfo info;
    info.ptr = ptr;
    info.size = size;
    info.file = std::string(file);
    info.line = line;
    info.is_freed = false;
    
    allocations_.push_back(info);
    total_allocated_ += size;
    peak_allocated_ = std::max(peak_allocated_, get_current_allocated());
}

void MemoryDebugger::record_deallocation(void* ptr) {
    if (!tracking_enabled_) return;
    
    for (auto& alloc : allocations_) {
        if (alloc.ptr == ptr && !alloc.is_freed) {
            alloc.is_freed = true;
            break;
        }
    }
}

void MemoryDebugger::print_summary() const {
    std::cout << "=== Memory Debug Summary ===" << std::endl;
    std::cout << "Total Allocated: " << (total_allocated_ / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Peak Allocated: " << (peak_allocated_ / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Current Allocated: " << (get_current_allocated() / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Total Allocations: " << allocations_.size() << std::endl;
    std::cout << "============================" << std::endl;
}

void MemoryDebugger::print_leaks() const {
    std::cout << "=== Memory Leaks ===" << std::endl;
    bool found_leaks = false;
    
    for (const auto& alloc : allocations_) {
        if (!alloc.is_freed) {
            std::cout << "LEAK: " << alloc.size << " bytes at " << alloc.ptr 
                      << " (" << alloc.file << ":" << alloc.line << ")" << std::endl;
            found_leaks = true;
        }
    }
    
    if (!found_leaks) {
        std::cout << "No memory leaks detected." << std::endl;
    }
    std::cout << "===================" << std::endl;
}

void MemoryDebugger::reset_stats() {
    allocations_.clear();
    total_allocated_ = 0;
    peak_allocated_ = 0;
}

size_t MemoryDebugger::get_current_allocated() const {
    size_t current = 0;
    for (const auto& alloc : allocations_) {
        if (!alloc.is_freed) {
            current += alloc.size;
        }
    }
    return current;
}

void KernelLaunchInfo::print() const {
    std::cout << "=== Kernel Launch Info ===" << std::endl;
    std::cout << "Kernel: " << kernel_name << std::endl;
    std::cout << "Grid: (" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << ")" << std::endl;
    std::cout << "Block: (" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << ")" << std::endl;
    std::cout << "Shared Memory: " << shared_mem_size << " bytes" << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(3) << execution_time_ms << " ms" << std::endl;
    std::cout << "===========================" << std::endl;
}

KernelTimer::KernelTimer() : timing_active_(false) {
    hipEventCreate(&start_event_);
    hipEventCreate(&stop_event_);
}

KernelTimer::~KernelTimer() {
    hipEventDestroy(start_event_);
    hipEventDestroy(stop_event_);
}

void KernelTimer::start() {
    hipEventRecord(start_event_, 0);
    timing_active_ = true;
}

void KernelTimer::stop() {
    if (timing_active_) {
        hipEventRecord(stop_event_, 0);
        hipEventSynchronize(stop_event_);
        timing_active_ = false;
    }
}

float KernelTimer::get_elapsed_time_ms() const {
    float elapsed = 0.0f;
    hipEventElapsedTime(&elapsed, start_event_, stop_event_);
    return elapsed;
}

template<typename T>
void print_tensor_info(const T* data, const std::string& name, 
                      const std::vector<int>& shape, bool print_values) {
    std::cout << "=== Tensor Info: " << name << " ===" << std::endl;
    
    // Print shape
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Calculate total elements
    int total_elements = 1;
    for (int dim : shape) {
        total_elements *= dim;
    }
    std::cout << "Total Elements: " << total_elements << std::endl;
    std::cout << "Memory Size: " << (total_elements * sizeof(T)) << " bytes" << std::endl;
    
    if (print_values && total_elements <= 100) {
        std::cout << "Values: ";
        for (int i = 0; i < total_elements; ++i) {
            if constexpr (std::is_same_v<T, hip_bfloat16>) {
                std::cout << __bfloat162float(data[i]);
            } else {
                std::cout << data[i];
            }
            if (i < total_elements - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "===============================" << std::endl;
}

template<typename T>
void print_tensor_stats(const T* data, int size, const std::string& name) {
    T min_val = data[0];
    T max_val = data[0];
    double sum = 0.0;
    
    for (int i = 0; i < size; ++i) {
        T val = data[i];
        if constexpr (std::is_same_v<T, hip_bfloat16>) {
            float f_val = __bfloat162float(val);
            sum += static_cast<double>(f_val);
            if (f_val < __bfloat162float(min_val)) min_val = val;
            if (f_val > __bfloat162float(max_val)) max_val = val;
        } else {
            sum += static_cast<double>(val);
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }
    
    double mean = sum / size;
    
    std::cout << "=== Tensor Stats: " << name << " ===" << std::endl;
    std::cout << "Size: " << size << std::endl;
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
        std::cout << "Min: " << __bfloat162float(min_val) << std::endl;
        std::cout << "Max: " << __bfloat162float(max_val) << std::endl;
    } else {
        std::cout << "Min: " << min_val << std::endl;
        std::cout << "Max: " << max_val << std::endl;
    }
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "===============================" << std::endl;
}

// Device-side debugging functions
__device__ void print_thread_info() {
    printf("Thread Info - Block: (%d,%d,%d), Thread: (%d,%d,%d), Global: (%d,%d,%d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x * blockDim.x + threadIdx.x,
           blockIdx.y * blockDim.y + threadIdx.y,
           blockIdx.z * blockDim.z + threadIdx.z);
}

__device__ void print_block_info() {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        printf("Block Info - Grid: (%d,%d,%d), Block: (%d,%d,%d), BlockDim: (%d,%d,%d)\n",
               gridDim.x, gridDim.y, gridDim.z,
               blockIdx.x, blockIdx.y, blockIdx.z,
               blockDim.x, blockDim.y, blockDim.z);
    }
}

__device__ void print_bfloat16(hip_bfloat16 val, const char* name) {
    printf("%s = %f\n", name, __bfloat162float(val));
}

bool validate_tensor_shape(const std::vector<int>& shape) {
    if (shape.empty()) {
        DEBUG_ERROR("Tensor shape is empty");
        return false;
    }
    
    for (int dim : shape) {
        if (dim <= 0) {
            DEBUG_ERROR("Invalid dimension size: %d", dim);
            return false;
        }
    }
    
    return true;
}

bool validate_attention_inputs(int batch_size, int num_heads, int seq_len, int head_dim) {
    if (batch_size <= 0) {
        DEBUG_ERROR("Invalid batch size: %d", batch_size);
        return false;
    }
    
    if (num_heads <= 0) {
        DEBUG_ERROR("Invalid number of heads: %d", num_heads);
        return false;
    }
    
    if (seq_len <= 0) {
        DEBUG_ERROR("Invalid sequence length: %d", seq_len);
        return false;
    }
    
    if (head_dim <= 0 || head_dim % 2 != 0) {
        DEBUG_ERROR("Invalid head dimension (must be positive and even): %d", head_dim);
        return false;
    }
    
    return true;
}

template<typename T>
bool check_for_nan_inf(const T* data, int size, const std::string& tensor_name) {
    bool found_issues = false;
    int nan_count = 0;
    int inf_count = 0;
    
    for (int i = 0; i < size; ++i) {
        if constexpr (std::is_same_v<T, hip_bfloat16>) {
            float val = __bfloat162float(data[i]);
            if (std::isnan(val)) {
                nan_count++;
                found_issues = true;
            } else if (std::isinf(val)) {
                inf_count++;
                found_issues = true;
            }
        } else {
            if (std::isnan(data[i])) {
                nan_count++;
                found_issues = true;
            } else if (std::isinf(data[i])) {
                inf_count++;
                found_issues = true;
            }
        }
    }
    
    if (found_issues) {
        DEBUG_WARNING("Tensor '%s' contains %d NaN values and %d Inf values", 
                     tensor_name.c_str(), nan_count, inf_count);
    }
    
    return !found_issues;
}

template<typename T>
bool compare_tensors(const T* a, const T* b, int size, float tolerance,
                    const std::string& name_a, const std::string& name_b) {
    bool tensors_match = true;
    int mismatch_count = 0;
    T max_diff = T(0);
    
    for (int i = 0; i < size; ++i) {
        T diff;
        if constexpr (std::is_same_v<T, hip_bfloat16>) {
            float val_a = __bfloat162float(a[i]);
            float val_b = __bfloat162float(b[i]);
            float f_diff = std::abs(val_a - val_b);
            diff = __float2bfloat16(f_diff);
            
            if (f_diff > tolerance) {
                tensors_match = false;
                mismatch_count++;
            }
        } else {
            diff = std::abs(a[i] - b[i]);
            if (static_cast<float>(diff) > tolerance) {
                tensors_match = false;
                mismatch_count++;
            }
        }
        
        if constexpr (std::is_same_v<T, hip_bfloat16>) {
            if (__bfloat162float(diff) > __bfloat162float(max_diff)) {
                max_diff = diff;
            }
        } else {
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    
    if (!tensors_match) {
        DEBUG_WARNING("Tensors '%s' and '%s' differ: %d/%d elements, max diff: %f", 
                     name_a.c_str(), name_b.c_str(), mismatch_count, size, 
                     std::is_same_v<T, hip_bfloat16> ? __bfloat162float(max_diff) : static_cast<float>(max_diff));
    } else {
        DEBUG_INFO("Tensors '%s' and '%s' match within tolerance %f", 
                  name_a.c_str(), name_b.c_str(), tolerance);
    }
    
    return tensors_match;
}

void PerformanceMonitor::add_metric(const std::string& name, double value, const std::string& unit) {
    Metric metric;
    metric.name = name;
    metric.value = value;
    metric.unit = unit;
    metrics_.push_back(metric);
}

void PerformanceMonitor::print_metrics() const {
    std::cout << "=== Performance Metrics ===" << std::endl;
    for (const auto& metric : metrics_) {
        std::cout << metric.name << ": " << std::fixed << std::setprecision(3) 
                  << metric.value;
        if (!metric.unit.empty()) {
            std::cout << " " << metric.unit;
        }
        std::cout << std::endl;
    }
    std::cout << "============================" << std::endl;
}

void PerformanceMonitor::reset_metrics() {
    metrics_.clear();
}

void PerformanceMonitor::record_bandwidth(size_t bytes_transferred, double time_ms) {
    double bandwidth_gb_s = (static_cast<double>(bytes_transferred) / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
    add_metric("Memory Bandwidth", bandwidth_gb_s, "GB/s");
}

void PerformanceMonitor::record_flops(size_t operations, double time_ms) {
    double flops = static_cast<double>(operations) / (time_ms / 1000.0);
    double gflops = flops / (1024.0 * 1024.0 * 1024.0);
    add_metric("Compute Performance", gflops, "GFLOPS");
}

void PerformanceMonitor::record_memory_usage(size_t bytes_used) {
    double memory_mb = static_cast<double>(bytes_used) / (1024.0 * 1024.0);
    add_metric("Memory Usage", memory_mb, "MB");
}

// Explicit template instantiations
template void print_tensor_info<hip_bfloat16>(const hip_bfloat16*, const std::string&, const std::vector<int>&, bool);
template void print_tensor_info<float>(const float*, const std::string&, const std::vector<int>&, bool);
template void print_tensor_stats<hip_bfloat16>(const hip_bfloat16*, int, const std::string&);
template void print_tensor_stats<float>(const float*, int, const std::string&);
template bool check_for_nan_inf<hip_bfloat16>(const hip_bfloat16*, int, const std::string&);
template bool check_for_nan_inf<float>(const float*, int, const std::string&);
template bool compare_tensors<hip_bfloat16>(const hip_bfloat16*, const hip_bfloat16*, int, float, const std::string&, const std::string&);
template bool compare_tensors<float>(const float*, const float*, int, float, const std::string&, const std::string&);

} // namespace debug
} // namespace utils
} // namespace sdpa