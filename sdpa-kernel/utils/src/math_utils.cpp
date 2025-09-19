#include "math_utils.h"
#include <hip/hip_runtime.h>

namespace sdpa {
namespace utils {

// BFloat16x2 operations
__device__ __host__ BFloat16x2 BFloat16x2::operator+(const BFloat16x2& other) const {
    return BFloat16x2(__hadd2(data, other.data));
}

__device__ __host__ BFloat16x2 BFloat16x2::operator-(const BFloat16x2& other) const {
    return BFloat16x2(__hsub2(data, other.data));
}

__device__ __host__ BFloat16x2 BFloat16x2::operator*(const BFloat16x2& other) const {
    return BFloat16x2(__hmul2(data, other.data));
}

__device__ __host__ BFloat16x2& BFloat16x2::operator+=(const BFloat16x2& other) {
    data = __hadd2(data, other.data);
    return *this;
}

__device__ __host__ BFloat16x2& BFloat16x2::operator*=(const BFloat16x2& other) {
    data = __hmul2(data, other.data);
    return *this;
}

namespace fast_math {

__device__ __forceinline__ __hip_bfloat16 fast_exp(__hip_bfloat16 x) {
    // Fast exponential approximation using bit manipulation
    float f = __bfloat162float(x);
    
    // Clamp to avoid overflow
    f = fminf(f, 88.0f);
    f = fmaxf(f, -88.0f);
    
    // Use hardware exp if available, otherwise polynomial approximation
    return __float2bfloat16(expf(f));
}

__device__ __forceinline__ BFloat16x2 fast_exp(const BFloat16x2& x) {
    __hip_bfloat16 x1 = __low2bfloat16(x.data);
    __hip_bfloat16 x2 = __high2bfloat16(x.data);
    return BFloat16x2(fast_exp(x1), fast_exp(x2));
}

__device__ __forceinline__ __hip_bfloat16 fast_div(__hip_bfloat16 a, __hip_bfloat16 b) {
    return __hdiv(a, b);
}

__device__ __forceinline__ __hip_bfloat16 fast_rcp(__hip_bfloat16 x) {
    return __hdiv(BF16_ONE, x);
}

__device__ __forceinline__ __hip_bfloat16 fast_sqrt(__hip_bfloat16 x) {
    return __float2bfloat16(sqrtf(__bfloat162float(x)));
}

__device__ __forceinline__ __hip_bfloat16 fast_rsqrt(__hip_bfloat16 x) {
    return __float2bfloat16(rsqrtf(__bfloat162float(x)));
}

__device__ __forceinline__ __hip_bfloat16 fast_log(__hip_bfloat16 x) {
    return __float2bfloat16(logf(__bfloat162float(x)));
}

__device__ void fast_softmax(__hip_bfloat16* input, __hip_bfloat16* output, int size) {
    // Find maximum for numerical stability
    __hip_bfloat16 max_val = input[0];
    for (int i = 1; i < size; i++) {
        max_val = __hmax(max_val, input[i]);
    }
    
    // Compute exponentials and sum
    __hip_bfloat16 sum = BF16_ZERO;
    for (int i = 0; i < size; i++) {
        output[i] = fast_exp(__hsub(input[i], max_val));
        sum = __hadd(sum, output[i]);
    }
    
    // Normalize
    __hip_bfloat16 inv_sum = fast_rcp(sum);
    for (int i = 0; i < size; i++) {
        output[i] = __hmul(output[i], inv_sum);
    }
}

__device__ void fast_softmax_inplace(__hip_bfloat16* data, int size) {
    fast_softmax(data, data, size);
}

__device__ __forceinline__ __hip_bfloat16 dot_product(const __hip_bfloat16* a, const __hip_bfloat16* b, int size) {
    __hip_bfloat16 result = BF16_ZERO;
    for (int i = 0; i < size; i++) {
        result = __hfma(a[i], b[i], result);
    }
    return result;
}

__device__ __forceinline__ BFloat16x2 dot_product_2(const BFloat16x2* a, const BFloat16x2* b, int size) {
    BFloat16x2 result;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

} // namespace fast_math

namespace warp {

__device__ __forceinline__ __hip_bfloat16 reduce_sum(__hip_bfloat16 val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = __hadd(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ __forceinline__ __hip_bfloat16 reduce_max(__hip_bfloat16 val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = __hmax(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ __forceinline__ BFloat16x2 reduce_sum(const BFloat16x2& val) {
    BFloat16x2 result = val;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        BFloat16x2 other;
        other.data = __shfl_down(result.data, offset);
        result = result + other;
    }
    return result;
}

__device__ __forceinline__ BFloat16x2 reduce_max(const BFloat16x2& val) {
    BFloat16x2 result = val;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        BFloat16x2 other;
        other.data = __shfl_down(result.data, offset);
        result.data = __hmax2(result.data, other.data);
    }
    return result;
}

__device__ __forceinline__ __hip_bfloat16 inclusive_scan_sum(__hip_bfloat16 val) {
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        __hip_bfloat16 temp = __shfl_up(val, offset);
        if (threadIdx.x % WARP_SIZE >= offset) {
            val = __hadd(val, temp);
        }
    }
    return val;
}

__device__ __forceinline__ __hip_bfloat16 exclusive_scan_sum(__hip_bfloat16 val) {
    __hip_bfloat16 inclusive = inclusive_scan_sum(val);
    __hip_bfloat16 previous = __shfl_up(inclusive, 1);
    return (threadIdx.x % WARP_SIZE == 0) ? BF16_ZERO : previous;
}

__device__ __forceinline__ __hip_bfloat16 shfl(__hip_bfloat16 val, int src_lane) {
    return __shfl(val, src_lane);
}

__device__ __forceinline__ __hip_bfloat16 shfl_down(__hip_bfloat16 val, int delta) {
    return __shfl_down(val, delta);
}

__device__ __forceinline__ __hip_bfloat16 shfl_up(__hip_bfloat16 val, int delta) {
    return __shfl_up(val, delta);
}

__device__ __forceinline__ __hip_bfloat16 shfl_xor(__hip_bfloat16 val, int mask) {
    return __shfl_xor(val, mask);
}

} // namespace warp

namespace block {

template<int BLOCK_SIZE>
__device__ __hip_bfloat16 reduce_sum(__hip_bfloat16 val) {
    __shared__ __hip_bfloat16 shared_mem[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    shared_mem[tid] = val;
    __syncthreads();
    
    // Tree reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] = __hadd(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    
    return shared_mem[0];
}

template<int BLOCK_SIZE>
__device__ __hip_bfloat16 reduce_max(__hip_bfloat16 val) {
    __shared__ __hip_bfloat16 shared_mem[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    shared_mem[tid] = val;
    __syncthreads();
    
    // Tree reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] = __hmax(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    
    return shared_mem[0];
}

// Explicit template instantiations for common block sizes
template __device__ __hip_bfloat16 reduce_sum<256>(__hip_bfloat16);
template __device__ __hip_bfloat16 reduce_sum<512>(__hip_bfloat16);
template __device__ __hip_bfloat16 reduce_max<256>(__hip_bfloat16);
template __device__ __hip_bfloat16 reduce_max<512>(__hip_bfloat16);

} // namespace block

namespace attention {

__device__ __forceinline__ __hip_bfloat16 compute_scale_factor(int head_dim) {
    return fast_math::fast_rsqrt(__float2bfloat16(static_cast<float>(head_dim)));
}

__device__ __forceinline__ __hip_bfloat16 apply_causal_mask(__hip_bfloat16 score, int row, int col) {
    return (col > row) ? BF16_NEG_INF : score;
}

__device__ __forceinline__ __hip_bfloat16 apply_dropout(__hip_bfloat16 val, float dropout_prob, unsigned int seed) {
    // Simple linear congruential generator
    unsigned int state = seed + threadIdx.x + blockIdx.x * blockDim.x;
    state = (state * 1664525u + 1013904223u);
    float random = static_cast<float>(state) / static_cast<float>(UINT32_MAX);
    
    return (random < dropout_prob) ? BF16_ZERO : __hdiv(val, __float2bfloat16(1.0f - dropout_prob));
}

__device__ void compute_attention_weights(
    const __hip_bfloat16* q, const __hip_bfloat16* k, __hip_bfloat16* scores,
    int seq_len, int head_dim, __hip_bfloat16 scale, bool causal_mask
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < seq_len && col < seq_len) {
        // Compute dot product
        __hip_bfloat16 score = fast_math::dot_product(
            &q[row * head_dim], &k[col * head_dim], head_dim
        );
        
        // Apply scaling
        score = __hmul(score, scale);
        
        // Apply causal mask if needed
        if (causal_mask) {
            score = apply_causal_mask(score, row, col);
        }
        
        scores[row * seq_len + col] = score;
    }
}

__device__ void compute_attention_output(
    const __hip_bfloat16* weights, const __hip_bfloat16* v, __hip_bfloat16* output,
    int seq_len, int head_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < seq_len && col < head_dim) {
        __hip_bfloat16 result = BF16_ZERO;
        
        for (int i = 0; i < seq_len; i++) {
            result = __hfma(weights[row * seq_len + i], v[i * head_dim + col], result);
        }
        
        output[row * head_dim + col] = result;
    }
}

} // namespace attention

} // namespace utils
} // namespace sdpa