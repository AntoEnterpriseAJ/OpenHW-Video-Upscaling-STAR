#include "../include/error_codes.h"
#include <hip/hip_runtime.h>

namespace sdpa {
namespace common {

const char* getErrorString(ErrorCode error) {
    switch (error) {
        case ErrorCode::SUCCESS:
            return "Success";
            
        // Memory errors
        case ErrorCode::MEMORY_ALLOCATION_FAILED:
            return "Memory allocation failed";
        case ErrorCode::MEMORY_DEALLOCATION_FAILED:
            return "Memory deallocation failed";
        case ErrorCode::INVALID_MEMORY_ADDRESS:
            return "Invalid memory address";
        case ErrorCode::OUT_OF_MEMORY:
            return "Out of memory";
        case ErrorCode::MEMORY_COPY_FAILED:
            return "Memory copy failed";
            
        // Input validation errors
        case ErrorCode::INVALID_BATCH_SIZE:
            return "Invalid batch size";
        case ErrorCode::INVALID_SEQUENCE_LENGTH:
            return "Invalid sequence length";
        case ErrorCode::INVALID_HEAD_DIMENSION:
            return "Invalid head dimension";
        case ErrorCode::INVALID_NUM_HEADS:
            return "Invalid number of heads";
        case ErrorCode::INVALID_TENSOR_SHAPE:
            return "Invalid tensor shape";
        case ErrorCode::NULL_POINTER:
            return "Null pointer";
        case ErrorCode::DIMENSION_MISMATCH:
            return "Dimension mismatch";
            
        // Computation errors
        case ErrorCode::NUMERICAL_INSTABILITY:
            return "Numerical instability detected";
        case ErrorCode::OVERFLOW_ERROR:
            return "Overflow error";
        case ErrorCode::UNDERFLOW_ERROR:
            return "Underflow error";
        case ErrorCode::DIVISION_BY_ZERO:
            return "Division by zero";
        case ErrorCode::INVALID_OPERATION:
            return "Invalid operation";
            
        // Hardware/Runtime errors
        case ErrorCode::DEVICE_NOT_AVAILABLE:
            return "Device not available";
        case ErrorCode::INSUFFICIENT_DEVICE_MEMORY:
            return "Insufficient device memory";
        case ErrorCode::KERNEL_LAUNCH_FAILED:
            return "Kernel launch failed";
        case ErrorCode::SYNCHRONIZATION_FAILED:
            return "Synchronization failed";
        case ErrorCode::CONTEXT_ERROR:
            return "Context error";
        case ErrorCode::DRIVER_ERROR:
            return "Driver error";
            
        // Configuration errors
        case ErrorCode::INVALID_CONFIGURATION:
            return "Invalid configuration";
        case ErrorCode::UNSUPPORTED_FEATURE:
            return "Unsupported feature";
        case ErrorCode::VERSION_MISMATCH:
            return "Version mismatch";
        case ErrorCode::PLATFORM_NOT_SUPPORTED:
            return "Platform not supported";
            
        // Performance errors
        case ErrorCode::TIMEOUT_ERROR:
            return "Timeout error";
        case ErrorCode::PERFORMANCE_DEGRADATION:
            return "Performance degradation detected";
        case ErrorCode::RESOURCE_EXHAUSTION:
            return "Resource exhaustion";
            
        default:
            return "Unknown error";
    }
}

bool isRecoverable(ErrorCode error) {
    switch (error) {
        // Non-recoverable errors
        case ErrorCode::OUT_OF_MEMORY:
        case ErrorCode::DEVICE_NOT_AVAILABLE:
        case ErrorCode::INSUFFICIENT_DEVICE_MEMORY:
        case ErrorCode::DRIVER_ERROR:
        case ErrorCode::PLATFORM_NOT_SUPPORTED:
        case ErrorCode::VERSION_MISMATCH:
            return false;
            
        // Potentially recoverable errors
        case ErrorCode::NUMERICAL_INSTABILITY:
        case ErrorCode::PERFORMANCE_DEGRADATION:
        case ErrorCode::TIMEOUT_ERROR:
        case ErrorCode::SYNCHRONIZATION_FAILED:
        case ErrorCode::KERNEL_LAUNCH_FAILED:
            return true;
            
        // Input/validation errors (recoverable with correct input)
        case ErrorCode::INVALID_BATCH_SIZE:
        case ErrorCode::INVALID_SEQUENCE_LENGTH:
        case ErrorCode::INVALID_HEAD_DIMENSION:
        case ErrorCode::INVALID_NUM_HEADS:
        case ErrorCode::INVALID_TENSOR_SHAPE:
        case ErrorCode::NULL_POINTER:
        case ErrorCode::DIMENSION_MISMATCH:
        case ErrorCode::INVALID_CONFIGURATION:
            return true;
            
        default:
            return false;
    }
}

ErrorCode hipErrorToSdpaError(int hip_error) {
    switch (hip_error) {
        case hipSuccess:
            return ErrorCode::SUCCESS;
        case hipErrorOutOfMemory:
            return ErrorCode::OUT_OF_MEMORY;
        case hipErrorInvalidValue:
        case hipErrorInvalidDevice:
            return ErrorCode::INVALID_CONFIGURATION;
        case hipErrorInvalidMemcpyDirection:
            return ErrorCode::MEMORY_COPY_FAILED;
        case hipErrorLaunchFailure:
            return ErrorCode::KERNEL_LAUNCH_FAILED;
        case hipErrorNoDevice:
            return ErrorCode::DEVICE_NOT_AVAILABLE;
        case hipErrorInvalidContext:
            return ErrorCode::CONTEXT_ERROR;
        case hipErrorMapFailed:
        case hipErrorUnmapFailed:
            return ErrorCode::INVALID_MEMORY_ADDRESS;
        default:
            return ErrorCode::UNKNOWN_ERROR;
    }
}

} // namespace common
} // namespace sdpa
