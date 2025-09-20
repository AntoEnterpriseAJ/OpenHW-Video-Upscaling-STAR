#ifndef ERROR_CODES_H
#define ERROR_CODES_H

namespace sdpa {
namespace common {

enum class ErrorCode {
    SUCCESS = 0,
    
    // Memory errors
    MEMORY_ALLOCATION_FAILED = 1000,
    MEMORY_DEALLOCATION_FAILED = 1001,
    INVALID_MEMORY_ADDRESS = 1002,
    OUT_OF_MEMORY = 1003,
    MEMORY_COPY_FAILED = 1004,
    
    // Input validation errors
    INVALID_BATCH_SIZE = 2000,
    INVALID_SEQUENCE_LENGTH = 2001,
    INVALID_HEAD_DIMENSION = 2002,
    INVALID_NUM_HEADS = 2003,
    INVALID_TENSOR_SHAPE = 2004,
    NULL_POINTER = 2005,
    DIMENSION_MISMATCH = 2006,
    
    // Computation errors
    NUMERICAL_INSTABILITY = 3000,
    OVERFLOW_ERROR = 3001,
    UNDERFLOW_ERROR = 3002,
    DIVISION_BY_ZERO = 3003,
    INVALID_OPERATION = 3004,
    
    // Hardware/Runtime errors
    DEVICE_NOT_AVAILABLE = 4000,
    INSUFFICIENT_DEVICE_MEMORY = 4001,
    KERNEL_LAUNCH_FAILED = 4002,
    SYNCHRONIZATION_FAILED = 4003,
    CONTEXT_ERROR = 4004,
    DRIVER_ERROR = 4005,
    
    // Configuration errors
    INVALID_CONFIGURATION = 5000,
    UNSUPPORTED_FEATURE = 5001,
    VERSION_MISMATCH = 5002,
    PLATFORM_NOT_SUPPORTED = 5003,
    
    // Performance errors
    TIMEOUT_ERROR = 6000,
    PERFORMANCE_DEGRADATION = 6001,
    RESOURCE_EXHAUSTION = 6002,
    
    // Unknown error
    UNKNOWN_ERROR = 9999
};

const char* getErrorString(ErrorCode error);
bool isRecoverable(ErrorCode error);
ErrorCode hipErrorToSdpaError(int hip_error);

} // namespace common
} // namespace sdpa

#endif // ERROR_CODES_H
