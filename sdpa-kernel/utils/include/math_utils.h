#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace sdpa {
namespace utils {

#ifndef hip_bfloat162
typedef struct {
    hip_bfloat16 x, y;
} hip_bfloat162;
#endif

// Mathematical constants in bfloat16
constexpr hip_bfloat16 BF16_ZERO = float2bfloat16(0.0f);
constexpr hip_bfloat16 BF16_ONE = float2bfloat16(1.0f);
constexpr hip_bfloat16 BF16_NEG_INF = float2bfloat16(-65504.0f);
constexpr hip_bfloat16 BF16_SQRT_2 = float2bfloat16(1.4142135f);
constexpr hip_bfloat16 BF16_INV_SQRT_2PI = float2bfloat16(0.39894228f);

// SIMD-style operations for bfloat16
struct BFloat16x2 {
    hip_bfloat162 data;

    __device__ __host__ BFloat16x2() : data(hip_bfloat162{BF16_ZERO, BF16_ZERO}) {}
    __device__ __host__ BFloat16x2(hip_bfloat16 a, hip_bfloat16 b) : data(hip_bfloat162{a, b}) {}
    __device__ __host__ explicit BFloat16x2(hip_bfloat162 d) : data(d) {}

    __device__ __host__ BFloat16x2 operator+(const BFloat16x2& other) const;
    __device__ __host__ BFloat16x2 operator-(const BFloat16x2& other) const;
    __device__ __host__ BFloat16x2 operator*(const BFloat16x2& other) const;
    __device__ __host__ BFloat16x2& operator+=(const BFloat16x2& other);
    __device__ __host__ BFloat16x2& operator*=(const BFloat16x2& other);
};

// Fast math functions optimized for attention computation
namespace fast_math {

// Fast exponential approximation using bfloat16
__device__ __forceinline__ hip_bfloat16 fast_exp(hip_bfloat16 x);
__device__ __forceinline__ BFloat16x2 fast_exp(const BFloat16x2& x);

// Fast division and reciprocal
__device__ __forceinline__ hip_bfloat16 fast_div(hip_bfloat16 a, hip_bfloat16 b);
__device__ __forceinline__ hip_bfloat16 fast_rcp(hip_bfloat16 x);

// Fast square root and inverse square root
__device__ __forceinline__ hip_bfloat16 fast_sqrt(hip_bfloat16 x);
__device__ __forceinline__ hip_bfloat16 fast_rsqrt(hip_bfloat16 x);

// Fast logarithm
__device__ __forceinline__ hip_bfloat16 fast_log(hip_bfloat16 x);

// Fast softmax computation
__device__ void fast_softmax(hip_bfloat16* input, hip_bfloat16* output, int size);
__device__ void fast_softmax_inplace(hip_bfloat16* data, int size);

// Fast matrix multiplication helpers
__device__ __forceinline__ hip_bfloat16 dot_product(const hip_bfloat16* a, const hip_bfloat16* b, int size);
__device__ __forceinline__ BFloat16x2 dot_product_2(const BFloat16x2* a, const BFloat16x2* b, int size);

} // namespace fast_math

// Warp-level primitives
namespace warp {

constexpr int WARP_SIZE = 64;

// Warp reduction operations
__device__ __forceinline__ hip_bfloat16 reduce_sum(hip_bfloat16 val);
__device__ __forceinline__ hip_bfloat16 reduce_max(hip_bfloat16 val);
__device__ __forceinline__ BFloat16x2 reduce_sum(const BFloat16x2& val);
__device__ __forceinline__ BFloat16x2 reduce_max(const BFloat16x2& val);

// Warp scan operations
__device__ __forceinline__ hip_bfloat16 inclusive_scan_sum(hip_bfloat16 val);
__device__ __forceinline__ hip_bfloat16 exclusive_scan_sum(hip_bfloat16 val);

// Warp shuffle operations
__device__ __forceinline__ hip_bfloat16 shfl(hip_bfloat16 val, int src_lane);
__device__ __forceinline__ hip_bfloat16 shfl_down(hip_bfloat16 val, int delta);
__device__ __forceinline__ hip_bfloat16 shfl_up(hip_bfloat16 val, int delta);
__device__ __forceinline__ hip_bfloat16 shfl_xor(hip_bfloat16 val, int mask);

} // namespace warp

// Block-level primitives
namespace block {

// Block reduction with shared memory
template<int BLOCK_SIZE>
__device__ hip_bfloat16 reduce_sum(hip_bfloat16 val);

template<int BLOCK_SIZE>
__device__ hip_bfloat16 reduce_max(hip_bfloat16 val);

// Block scan operations
template<int BLOCK_SIZE>
__device__ hip_bfloat16 inclusive_scan_sum(hip_bfloat16 val);

template<int BLOCK_SIZE>
__device__ hip_bfloat16 exclusive_scan_sum(hip_bfloat16 val);

} // namespace block

// Utility functions for attention computation
namespace attention {

// Scale factor computation
__device__ __forceinline__ hip_bfloat16 compute_scale_factor(int head_dim);

// Causal mask application
__device__ __forceinline__ hip_bfloat16 apply_causal_mask(hip_bfloat16 score, int row, int col);

// Dropout simulation (for training)
__device__ __forceinline__ hip_bfloat16 apply_dropout(hip_bfloat16 val, float dropout_prob, unsigned int seed);

// Attention weight computation
__device__ void compute_attention_weights(
    const hip_bfloat16* q, const hip_bfloat16* k, hip_bfloat16* scores,
    int seq_len, int head_dim, hip_bfloat16 scale, bool causal_mask = false
);

// Attention output computation
__device__ void compute_attention_output(
    const hip_bfloat16* weights, const hip_bfloat16* v, hip_bfloat16* output,
    int seq_len, int head_dim
);

} // namespace attention

} // namespace utils
} // namespace sdpa

#endif // MATH_UTILS_H