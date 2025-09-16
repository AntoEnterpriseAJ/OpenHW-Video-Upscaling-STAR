# triton_sdpa.py (optimized)
from __future__ import annotations
import contextlib
import enum
import typing as t

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


class SDPBackend(enum.Enum):
    EFFICIENT_ATTENTION = 1


def sdpa_kernel(backend: SDPBackend):
    """
    Return a context manager which monkey-patches torch.nn.functional.scaled_dot_product_attention
    with a Triton-backed implementation when backend == SDPBackend.EFFICIENT_ATTENTION and Triton is available.
    Otherwise returns a nullcontext.
    """
    if backend != SDPBackend.EFFICIENT_ATTENTION or not _TRITON_AVAILABLE:
        return contextlib.nullcontext()

    return _TritonSDPAContext()


# -----------------------
# Triton kernels & helpers
# -----------------------

if _TRITON_AVAILABLE:
    # --- TUNABLE PARAMETERS (change for your hardware) ---
    # Typical good starting points: BLOCK_M=128, BLOCK_N=128, BLOCK_D=64 for Ampere+.
    # For very small D (<64), reduce BLOCK_D. For huge D, increase/block differently.
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = 64

    @triton.jit
    def _kernel_sdpa_blocked(
        Q_ptr, K_ptr, V_ptr, OUT_ptr,
        B, H, S_q, S_k, D,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        scale,
        is_causal: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        """
        Blocked, numerically stable SDPA.
        - Each program handles one (batch * head, row-block) pair.
        - We iterate over key-blocks (N direction) and D-blocks for partial matmuls.
        - The kernel maintains running_max and running_sumexp per query row to compute stable softmax across blocks.
        Notes:
          * Q/K/V/OUT ptrs are pointers to element 0. Strides are in elements.
          * This implementation favors clarity and correctness; it also reduces redundant memory ops vs naive code.
        """
        pid = tl.program_id(0)

        num_m_blocks = (S_q + BLOCK_M - 1) // BLOCK_M
        total_heads = B * H
        head_index = pid // num_m_blocks
        m_block = pid % num_m_blocks
        if head_index >= total_heads:
            return

        b = head_index // H
        h = head_index % H

        row_start = m_block * BLOCK_M
        row_range = tl.arange(0, BLOCK_M)
        q_row_idx = row_start + row_range  # (BLOCK_M,)
        row_mask = q_row_idx < S_q

        # initialize accumulation buffer for output in D-block increments
        # We'll accumulate and store per-d_block to avoid huge registers when D large
        # but keep running softmax stats across N-blocks.
        running_max = tl.full((BLOCK_M,), -1e9, dtype=tl.float32)
        running_sumexp = tl.zeros((BLOCK_M,), dtype=tl.float32)

        num_n_blocks = (S_k + BLOCK_N - 1) // BLOCK_N

        # precompute base pointers that do not change across loops
        q_base = Q_ptr + b * stride_qb + h * stride_qh + row_start * stride_qs
        out_base = OUT_ptr + b * stride_ob + h * stride_oh + row_start * stride_os

        # We'll accumulate per D-block and write them out when finished iterating over N-blocks.
        # For each d_block, accumulate acc over N-blocks into a local buffer.
        for d_off in range(0, D, BLOCK_D):
            d_block = min(BLOCK_D, D - d_off)
            # accumulator for this d_block: shape (BLOCK_M, d_block)
            acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

            # iterate over key blocks
            for n_block in range(num_n_blocks):
                col_start = n_block * BLOCK_N
                col_range = tl.arange(0, BLOCK_N)
                k_col_idx = col_start + col_range  # (BLOCK_N,)
                col_mask = k_col_idx < S_k

                # load Q slice for this d_block: (BLOCK_M, d_block)
                # Q offsets: q_base + (row_offset * stride_qs) + (d_index * stride_qd)
                q_ptr = q_base + (row_range[:, None] * stride_qs) + (d_off + tl.arange(0, d_block))[None, :] * stride_qd
                q = tl.load(q_ptr, mask=(row_mask[:, None]), other=0.0)  # (BLOCK_M, d_block)

                # load K slice for this d_block: (BLOCK_N, d_block)
                k_ptr = K_ptr + b * stride_kb + h * stride_kh + (col_start + col_range)[:, None] * stride_ks + (d_off + tl.arange(0, d_block))[None, :] * stride_kd
                k = tl.load(k_ptr, mask=(col_mask[:, None]), other=0.0)  # (BLOCK_N, d_block)

                # compute logits partial: q @ k^T -> (BLOCK_M, BLOCK_N)
                # We need q (BLOCK_M, d), k (BLOCK_N, d) -> use tl.dot with transpose of k
                logits_block = tl.dot(q, tl.transpose(k))  # (BLOCK_M, BLOCK_N)
                # scale
                logits_block = logits_block * scale

                # causal mask: set logits to -inf where key_pos > query_pos
                if is_causal:
                    # q_row_idx shape (BLOCK_M,), k_col_idx shape (BLOCK_N,)
                    q_idx = q_row_idx[:, None]
                    k_idx = k_col_idx[None, :]
                    causal_mask = k_idx > q_idx
                    # mask out (note: use a large negative)
                    logits_block = tl.where(causal_mask, -1e9, logits_block)

                # numerically stable online softmax update:
                # block_max : max logits across current block (per row)
                block_max = tl.max(logits_block, axis=1)
                new_max = tl.maximum(running_max, block_max)

                exp_block = tl.exp(logits_block - new_max[:, None])
                block_sumexp = tl.sum(exp_block, axis=1)

                # update running_sumexp and running_max consistently:
                running_sumexp = running_sumexp * tl.exp(running_max - new_max) + block_sumexp
                running_max = new_max

                # compute normalized probabilities for this block
                p_block = exp_block / running_sumexp[:, None]  # (BLOCK_M, BLOCK_N)

                # load V slice for this d_block: (BLOCK_N, d_block)
                v_ptr = V_ptr + b * stride_vb + h * stride_vh + (col_start + col_range)[:, None] * stride_vs + (d_off + tl.arange(0, d_block))[None, :] * stride_vd
                v = tl.load(v_ptr, mask=(col_mask[:, None]), other=0.0)  # (BLOCK_N, d_block)

                # accumulate acc += p_block @ v -> (BLOCK_M, d_block)
                pv = tl.dot(p_block, v)  # (BLOCK_M, d_block)
                # pv may be d_block columns (< BLOCK_D). store into acc[:, :d_block]
                acc += pv

            # After iterating all N-blocks, write acc into OUT for rows that exist and d offsets
            # out_row_ptr = out_base + (row_idx * stride_os) + (d_off + dd) * stride_od
            for i in range(0, BLOCK_M):
                idx = row_start + i
                if idx >= S_q:
                    break
                # For each d in the d_block write scalar
                # we stride through D dimension
                base_row_ptr = out_base + i * stride_os + d_off * stride_od
                # write contiguous d_block floats
                tl.store(base_row_ptr + tl.arange(0, d_block) * stride_od, acc[i, :d_block], mask=(tl.arange(0, d_block) < d_block))

        # end kernel


    def _triton_sdpa_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
        """
        q,k,v: tensors on CUDA with shape (B, H, S, D) or (B, S, D).
        Returns: out with same shape as input (B, H, S, D) or (B, S, D)
        """
        was_3d = False
        if q.ndim == 3:
            was_3d = True
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

        B, H, S_q, D = q.shape
        S_k = k.shape[2]
        assert k.shape == (B, H, S_k, D)
        assert v.shape == (B, H, S_k, D)

        # choose compute dtype: if input is fp16 prefer fp16 compute to use Tensor Cores; circuit casts inside kernel are float32 for accumulation
        orig_dtype = q.dtype
        compute_dtype = torch.float16 if orig_dtype == torch.float16 else torch.float32

        # create contiguous tensors of compute dtype
        q_cont = q.contiguous().to(compute_dtype)
        k_cont = k.contiguous().to(compute_dtype)
        v_cont = v.contiguous().to(compute_dtype)

        out = torch.empty_like(q_cont, device=q.device, dtype=q_cont.dtype)

        # strides in elements
        stride_qb, stride_qh, stride_qs, stride_qd = q_cont.stride()
        stride_kb, stride_kh, stride_ks, stride_kd = k_cont.stride()
        stride_vb, stride_vh, stride_vs, stride_vd = v_cont.stride()
        stride_ob, stride_oh, stride_os, stride_od = out.stride()

        scale = 1.0 / (D ** 0.5)

        num_m_blocks = (S_q + BLOCK_M - 1) // BLOCK_M
        grid = (B * H * num_m_blocks,)

        _kernel_sdpa_blocked[grid](
            q_cont, k_cont, v_cont, out,
            B, H, S_q, S_k, D,
            stride_qb, stride_qh, stride_qs, stride_qd,
            stride_kb, stride_kh, stride_ks, stride_kd,
            stride_vb, stride_vh, stride_vs, stride_vd,
            stride_ob, stride_oh, stride_os, stride_od,
            scale,
            bool(is_causal),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )

        # cast back to original dtype if needed
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        if was_3d:
            out = out.squeeze(1)
        return out


# -----------------------
# Context manager that patches F.scaled_dot_product_attention
# -----------------------

class _TritonSDPAContext:
    def __init__(self):
        self._orig = None

    def __enter__(self):
        self._orig = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = _patched_sdpa
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig is not None:
            F.scaled_dot_product_attention = self._orig
        return False


# patched implementation visible to Python (used in context manager)
def _patched_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False):
    """
    A drop-in replacement for torch.nn.functional.scaled_dot_product_attention that routes to Triton kernel for CUDA tensors.
    Supports q/k/v shaped (B, S, D) or (B, H, S, D) and returns same shape as inputs.
    attn_mask: not implemented in Triton path. dropout_p > 0 falls back to PyTorch.
    """
    # Fallback conditions
    if (not _TRITON_AVAILABLE) or (not q.is_cuda) or (attn_mask is not None) or (dropout_p != 0.0):
        return _fallback_sdpa(q, k, v, attn_mask, dropout_p, is_causal)

    if q.ndim not in (3, 4):
        return _fallback_sdpa(q, k, v, attn_mask, dropout_p, is_causal)

    # If inputs are not contiguous, make them contiguous
    # Prefer to keep fp16 compute when user passed fp16
    orig_dtype = q.dtype
    out = _triton_sdpa_forward(q, k, v, is_causal=is_causal)
    return out


def _fallback_sdpa(q, k, v, attn_mask, dropout_p, is_causal):
    # simple fallback using PyTorch existing implementation
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)


# If Triton not available, sdpa_kernel returns nullcontext
if not _TRITON_AVAILABLE:
    pass


__all__ = ["sdpa_kernel", "SDPBackend"]
