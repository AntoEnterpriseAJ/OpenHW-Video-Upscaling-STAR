#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <hip/hip_bfloat16.h>
#include <cstddef>

namespace sdpa {
namespace common {

// Mathematical constants
constexpr float PI = 3.14159265358979323846f;
constexpr float E = 2.71828182845904523536f;
constexpr float SQRT_2 = 1.41421356237309504880f;
constexpr float SQRT_PI = 1.77245385090551602730f;
constexpr float INV_SQRT_2PI = 0.39894228040143267794f;

// Numerical limits for bfloat16
constexpr float BF16_MAX = 3.38953138925e38f;
constexpr float BF16_MIN = 1.17549435082e-38f;
constexpr float BF16_EPSILON = 7.81250000e-03f;  // 2^-7
constexpr float BF16_NEG_INF_VAL = -3.38953138925e38f;
constexpr float BF16_POS_INF_VAL = 3.38953138925e38f;

// Hardware constants
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int WARP_SIZE = 64;
constexpr int MAX_BLOCKS_PER_SM = 32;
constexpr int SHARED_MEMORY_SIZE = 65536;  // 64KB

// Attention computation constants
constexpr int MAX_SEQUENCE_LENGTH = 8192;
constexpr int MAX_HEAD_DIMENSION = 256;
constexpr int MAX_BATCH_SIZE = 64;
constexpr int MAX_NUM_HEADS = 32;

// Memory alignment constants
constexpr size_t MEMORY_ALIGNMENT = 128;
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t PAGE_SIZE = 4096;

// Performance tuning constants
constexpr int TILE_SIZE_M = 128;
constexpr int TILE_SIZE_N = 128;
constexpr int TILE_SIZE_K = 32;
constexpr int BLOCK_SIZE_X = 16;
constexpr int BLOCK_SIZE_Y = 16;
constexpr int THREAD_BLOCK_SIZE = 256;

// Kernel configuration constants
constexpr int MIN_BLOCKS_PER_SM = 1;
constexpr int OCCUPANCY_TARGET = 75;  // Target occupancy percentage
constexpr int REGISTERS_PER_THREAD_LIMIT = 64;

} // namespace common
} // namespace sdpa

#endif // CONSTANTS_H
