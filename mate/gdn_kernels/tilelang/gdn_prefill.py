import functools
import math
from typing import Optional, Tuple

import torch
import tilelang
import tilelang.language as T

_LOG2E = 1.4426950408889634


@functools.lru_cache
def _torch_dtype_to_tl(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype {dtype}.")
    return mapping[dtype]


@functools.lru_cache
def _build_chunk_metadata(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build per-chunk metadata once on host, then upload:
      - chunk_offsets: [batch + 1]
      - chunk_token_starts: [total_chunks]
      - chunk_token_lens: [total_chunks]
    """
    dev = cu_seqlens.device
    dtype = torch.int32

    cu = cu_seqlens.to(dtype=dtype)
    batch = cu.numel() - 1
    seq_lens = cu[1:] - cu[:-1]
    chunks_per_batch = (seq_lens + chunk_size - 1) // chunk_size

    chunk_offsets = torch.cat(
        [
            torch.zeros(1, dtype=dtype, device=dev),
            chunks_per_batch.cumsum(0, dtype=dtype),
        ]
    )

    total_chunks = int(chunk_offsets[-1].item())

    if total_chunks == 0:
        empty = torch.empty(0, dtype=dtype, device=dev)
        return empty, empty, empty, 0

    batch_idx = torch.arange(batch, device=dev).repeat_interleave(
        chunks_per_batch
    )  # [total_chunks]
    local_chunk_idx = (
        torch.arange(total_chunks, dtype=dtype, device=dev) - chunk_offsets[batch_idx]
    )  # [total_chunks]
    starts = cu[:-1][batch_idx] + local_chunk_idx * chunk_size
    lens = torch.clamp(
        seq_lens[batch_idx] - local_chunk_idx * chunk_size, min=0, max=chunk_size
    )

    return (
        chunk_offsets,
        starts,
        lens,
        total_chunks,
    )


@functools.lru_cache(maxsize=32)
def fused_prepare_compute_w_u_tl(
    total_chunks: int,
    total_tokens: int,
    head_sab: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    head_k: Optional[int] = None,
    dtype: str = "float16",
):
    if head_k is None:
        head_k = head_sab
    if head_sab % head_k != 0:
        raise ValueError("head_sab must be divisible by head_k.")
    sab_to_k_group_size = head_sab // head_k

    accum_dtype = "float32"
    block_C = chunk_size
    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0

    @tilelang.jit(
        target="musa",
        out_idx=[-3, -2, -1],  # output Tensor：w, u, cu_g
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128, block_DK=64, block_DV=64):
        assert dim_k % block_DK == 0, "dim_k must be divisible by block_DK"
        assert dim_v % block_DV == 0, "dim_v must be divisible by block_DV"

        @T.prim_func
        def fused_prepare_compute_w_u(
            k: T.Tensor([total_tokens, head_k, dim_k], dtype),
            v: T.Tensor([total_tokens, head_sab, dim_v], dtype),
            alpha: T.Tensor([total_tokens, head_sab], "float32"),
            beta: T.Tensor([total_tokens, head_sab], "float32"),
            chunk_token_starts: T.Tensor([total_chunks], "int32"),
            chunk_token_lens: T.Tensor([total_chunks], "int32"),
            w: T.Tensor([total_tokens, head_sab, dim_k], dtype),
            u: T.Tensor([total_tokens, head_sab, dim_v], dtype),
            cu_g: T.Tensor([total_tokens, head_sab], "float32"),
        ):
            with T.Kernel(total_chunks, head_sab, threads=threads) as (tid, sab_hid):
                k_hid = sab_hid // sab_to_k_group_size
                chunk_token_start = chunk_token_starts[tid]
                actual_len = chunk_token_lens[tid]

                # share memory
                alpha_shared = T.alloc_shared([block_C], "float32")
                beta_shared = T.alloc_shared([block_C], "float32")
                cu_g_shared = T.alloc_shared([block_C], "float32")
                k_shared = T.alloc_shared([block_C, block_DK], dtype)
                k_beta_shared = T.alloc_shared([block_C, block_DK], dtype)
                v_beta_shared = T.alloc_shared([block_C, block_DV], dtype)
                S_left_shared = T.alloc_shared([block_C, block_C], dtype)
                S_right_shared = T.alloc_shared([block_C, block_C], dtype)
                P_left_shared = T.alloc_shared([block_C, block_C], dtype)
                P_right_shared = T.alloc_shared([block_C, block_C], dtype)

                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                temp_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                w_frag = T.alloc_fragment([block_C, block_DK], accum_dtype)
                u_frag = T.alloc_fragment([block_C, block_DV], accum_dtype)

                for i in T.Parallel(block_C):
                    valid_flag = i < actual_len
                    alpha_shared[i] = T.if_then_else(
                        valid_flag, alpha[chunk_token_start + i, sab_hid], 1.0
                    )
                    beta_shared[i] = T.if_then_else(
                        valid_flag, beta[chunk_token_start + i, sab_hid], 1.0
                    )

                acc = T.alloc_var("float32", init=0.0)
                for i in T.serial(block_C):
                    acc = acc + T.log(alpha_shared[i])
                    cu_g_shared[i] = acc

                T.clear(gram_frag)
                for i_k in T.Pipelined(dim_k // block_DK, num_stages=num_stages):
                    k_offset = i_k * block_DK
                    for n, kk in T.Parallel(block_C, block_DK):
                        k_shared[n, kk] = T.if_then_else(
                            n < actual_len,
                            k[chunk_token_start + n, k_hid, k_offset + kk],
                            0.0,
                        )
                    T.gemm(
                        k_shared,
                        k_shared,
                        gram_frag,
                        transpose_B=True,
                    )

                for i, j in T.Parallel(block_C, block_C):
                    p_val = T.if_then_else(
                        i < actual_len and j < i,
                        T.cast(
                            -gram_frag[i, j]
                            * beta_shared[i]
                            * T.exp2((cu_g_shared[i] - cu_g_shared[j]) * _LOG2E),
                            dtype,
                        ),
                        0.0,
                    )
                    P_left_shared[i, j] = p_val
                    P_right_shared[i, j] = p_val

                T.clear(S_left_shared)
                T.clear(S_right_shared)
                for i in T.serial(block_C):
                    S_left_shared[i, i] = T.if_then_else(i < actual_len, 1.0, 0.0)
                    S_right_shared[i, i] = T.if_then_else(i < actual_len, 1.0, 0.0)

                for _r in T.serial(num_rounds):
                    T.gemm(P_left_shared, S_right_shared, temp_frag, clear_accum=True)
                    for i, j in T.Parallel(block_C, block_C):
                        s_next = S_left_shared[i, j] + temp_frag[i, j]
                        S_left_shared[i, j] = s_next
                        S_right_shared[i, j] = s_next
                    T.gemm(P_left_shared, P_right_shared, temp_frag, clear_accum=True)
                    T.copy(temp_frag, P_left_shared)
                    T.copy(temp_frag, P_right_shared)
                for i_k in T.Pipelined(dim_k // block_DK, num_stages=num_stages):
                    k_offset = i_k * block_DK
                    for n, kk in T.Parallel(block_C, block_DK):
                        k_beta_shared[n, kk] = T.if_then_else(
                            n < actual_len,
                            k_shared[n, kk] * beta_shared[n],
                            0.0,
                        )
                    T.gemm(
                        S_left_shared,
                        k_beta_shared,
                        w_frag,
                        clear_accum=True,
                    )
                    for n, kk in T.Parallel(block_C, block_DK):
                        if n < actual_len:
                            w[chunk_token_start + n, sab_hid, k_offset + kk] = w_frag[
                                n, kk
                            ]

                for i_v in T.Pipelined(dim_v // block_DV, num_stages=num_stages):
                    v_start = i_v * block_DV
                    for n, vv in T.Parallel(block_C, block_DV):
                        valid_flag = n < actual_len
                        v_beta_shared[n, vv] = T.if_then_else(
                            valid_flag,
                            v[chunk_token_start + n, sab_hid, v_start + vv]
                            * beta_shared[n],
                            0.0,
                        )
                    T.gemm(
                        S_left_shared,
                        v_beta_shared,
                        u_frag,
                        clear_accum=True,
                    )
                    for n, vv in T.Parallel(block_C, block_DV):
                        if n < actual_len:
                            u[chunk_token_start + n, sab_hid, v_start + vv] = u_frag[
                                n, vv
                            ]

                    for n in T.Parallel(block_C):
                        if n < actual_len:
                            cu_g[chunk_token_start + n, sab_hid] = cu_g_shared[n]

        return fused_prepare_compute_w_u

    return _func


@functools.lru_cache(maxsize=32)
def _h_recurrence_tl(
    total_chunks: int,
    total_tokens: int,
    batch: int,
    head_sab: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    head_k: Optional[int] = None,
    dtype: str = "float16",
):
    if head_k is None:
        head_k = head_sab
    if head_sab % head_k != 0:
        raise ValueError("head_sab must be divisible by head_k.")
    sab_to_k_group_size = head_sab // head_k

    accum_dtype = "float32"
    block_C = chunk_size

    @tilelang.jit(
        target="musa",
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128, block_DV=64):
        assert dim_v % block_DV == 0, "dim_v must be divisible by block_DV"

        @T.prim_func
        def h_recurrence(
            k: T.Tensor([total_tokens, head_k, dim_k], dtype),
            cu_g: T.Tensor([total_tokens, head_sab], "float32"),
            w: T.Tensor([total_tokens, head_sab, dim_k], dtype),
            u: T.Tensor([total_tokens, head_sab, dim_v], dtype),
            cu_seqlens: T.Tensor([batch + 1], "int32"),
            chunk_offsets: T.Tensor([batch + 1], "int32"),
            initial_state: T.Tensor([batch, head_sab, dim_v, dim_k], "float32"),
            output_state: T.Tensor([batch, head_sab, dim_v, dim_k], "float32"),
            S: T.Tensor([total_chunks, head_sab, dim_v, dim_k], "float32"),
            v_new: T.Tensor([total_tokens, head_sab, dim_v], dtype),
        ):
            with T.Kernel(dim_v // block_DV, batch, head_sab, threads=threads) as (
                vid,
                bid,
                sab_hid,
            ):
                k_hid = sab_hid // sab_to_k_group_size
                seq_start = cu_seqlens[bid]
                seq_end = cu_seqlens[bid + 1]
                cid = chunk_offsets[bid]
                seqlen = seq_end - seq_start
                num_chunks = (seqlen + block_C - 1) // block_C

                v_offset = vid * block_DV

                g_c = T.alloc_shared([block_C], "float32")
                g_exp_c = T.alloc_shared([block_C], "float32")
                g_decay_c = T.alloc_shared([block_C], "float32")
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                h_c = T.alloc_shared([block_DV, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, block_DV], dtype)
                v_scaled_c = T.alloc_shared([block_C, block_DV], dtype)
                ws_frag = T.alloc_fragment([block_C, block_DV], accum_dtype)
                h_next_frag = T.alloc_fragment([block_DV, dim_k], accum_dtype)

                # initialize h
                T.copy(
                    initial_state[bid, sab_hid, v_offset : v_offset + block_DV, :],
                    h_c,
                    disable_tma=True,
                )

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    chunk_token_start = seq_start + t * block_C
                    chunk_token_end = T.min(chunk_token_start + block_C, seq_end)
                    actual_len = chunk_token_end - chunk_token_start

                    for n in T.Parallel(block_C):
                        g_c[n] = T.if_then_else(
                            n < actual_len,
                            cu_g[chunk_token_start + n, sab_hid],
                            0.0,
                        )

                    g_last_val = g_c[actual_len - 1]

                    for n in T.Parallel(block_C):
                        g_exp_c[n] = T.exp2(g_c[n] * _LOG2E)
                        g_decay_c[n] = T.exp2((g_last_val - g_c[n]) * _LOG2E)

                    for n, kk in T.Parallel(block_C, dim_k):
                        valid_flag = n < actual_len
                        w_c[n, kk] = T.if_then_else(
                            valid_flag, w[chunk_token_start + n, sab_hid, kk], 0.0
                        )
                        k_c[n, kk] = T.if_then_else(
                            valid_flag, k[chunk_token_start + n, k_hid, kk], 0.0
                        )
                    T.gemm(
                        w_c,
                        h_c,
                        ws_frag,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    for n, vv in T.Parallel(block_C, block_DV):
                        valid_flag = n < actual_len
                        v_new_c[n, vv] = (
                            T.if_then_else(
                                valid_flag,
                                u[chunk_token_start + n, sab_hid, v_offset + vv],
                                0.0,
                            )
                            - ws_frag[n, vv] * g_exp_c[n]
                        )
                        v_scaled_c[n, vv] = v_new_c[n, vv] * g_decay_c[n]
                        if valid_flag:
                            v_new[chunk_token_start + n, sab_hid, v_offset + vv] = (
                                v_new_c[n, vv]
                            )

                    for i, j in T.Parallel(block_DV, dim_k):
                        h_next_frag[i, j] = h_c[i, j] * T.exp2(g_last_val * _LOG2E)
                    T.gemm(
                        v_scaled_c,
                        k_c,
                        h_next_frag,
                        transpose_A=True,
                        clear_accum=True,
                    )
                    T.copy(h_next_frag, h_c, disable_tma=True)
                    T.copy(
                        h_next_frag,
                        S[cid + t, sab_hid, v_offset : v_offset + block_DV, :],
                        disable_tma=True,
                    )
                # write final state
                T.copy(
                    h_next_frag,
                    output_state[bid, sab_hid, v_offset : v_offset + block_DV, :],
                    disable_tma=True,
                )

        return h_recurrence

    return _func


@functools.lru_cache(maxsize=32)
def _output_o_tl(
    total_chunks: int,
    total_tokens: int,
    head_q: int,
    head_kv: int,
    group_size: int,
    chunk_size: int,
    scale: float,
    dim_k: int,
    dim_v: int,
    head_o: Optional[int] = None,
    head_sab: Optional[int] = None,
    q_group_size: Optional[int] = None,
    sab_group_size: Optional[int] = None,
    head_k: Optional[int] = None,
    dtype: str = "float32",
):
    if head_o is None:
        head_o = head_q
    if head_sab is None:
        head_sab = head_kv
    if head_k is None:
        head_k = head_kv
    if q_group_size is None:
        q_group_size = head_o // head_q
    if sab_group_size is None:
        sab_group_size = group_size

    if head_o % head_q != 0:
        raise ValueError("head_o must be divisible by head_q.")
    if head_o % head_sab != 0:
        raise ValueError("head_o must be divisible by head_sab.")
    if head_sab % head_k != 0:
        raise ValueError("head_sab must be divisible by head_k.")
    if q_group_size != head_o // head_q:
        raise ValueError("q_group_size mismatch with head_o/head_q.")
    if sab_group_size != head_o // head_sab:
        raise ValueError("sab_group_size mismatch with head_o/head_sab.")

    sab_to_k_group_size = head_sab // head_k

    accum_dtype = "float32"
    block_C = chunk_size

    @tilelang.jit(
        target="musa",
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128, block_DV=64):
        assert dim_v % block_DV == 0, "dim_v must be divisible by block_DV"

        @T.prim_func
        def output_o(
            q: T.Tensor([total_tokens, head_q, dim_k], dtype),
            k: T.Tensor([total_tokens, head_k, dim_k], dtype),
            g: T.Tensor([total_tokens, head_sab], "float32"),
            chunk_token_starts: T.Tensor([total_chunks], "int32"),
            chunk_token_lens: T.Tensor([total_chunks], "int32"),
            S: T.Tensor([total_chunks, head_sab, dim_v, dim_k], "float32"),
            v_new: T.Tensor([total_tokens, head_sab, dim_v], dtype),
            o: T.Tensor([total_tokens, head_o, dim_v], dtype),
        ):
            with T.Kernel(total_chunks, head_o, dim_v // block_DV, threads=threads) as (
                tid,
                out_hid,
                vid,
            ):
                q_hid = out_hid // q_group_size
                sab_hid = out_hid // sab_group_size
                k_hid = sab_hid // sab_to_k_group_size

                chunk_token_start = chunk_token_starts[tid]
                actual_len = chunk_token_lens[tid]
                v_offset = vid * block_DV

                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                h_c = T.alloc_shared([block_DV, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, block_DV], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)
                o_c = T.alloc_shared([block_C, block_DV], dtype)
                o_frag = T.alloc_fragment([block_C, block_DV], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)

                for n, kk in T.Parallel(block_C, dim_k):
                    valid_flag = n < actual_len
                    q_c[n, kk] = T.if_then_else(
                        valid_flag, q[chunk_token_start + n, q_hid, kk], 0.0
                    )
                    k_c[n, kk] = T.if_then_else(
                        valid_flag, k[chunk_token_start + n, k_hid, kk], 0.0
                    )

                for i in T.Parallel(block_C):
                    g_c[i] = g[chunk_token_start + i, sab_hid]

                T.copy(
                    S[tid, sab_hid, v_offset : v_offset + block_DV, :],
                    h_c,
                    disable_tma=True,
                )
                for n, vv in T.Parallel(block_C, block_DV):
                    valid_flag = n < actual_len
                    v_new_c[n, vv] = T.if_then_else(
                        valid_flag,
                        v_new[chunk_token_start + n, sab_hid, v_offset + vv],
                        0.0,
                    )

                T.gemm(q_c, h_c, o_frag, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_C, block_DV):
                    o_frag[i, j] = o_frag[i, j] * T.exp2(g_c[i] * _LOG2E)

                T.gemm(q_c, k_c, attn_frag, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(
                        j <= i,
                        attn_frag[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                        0.0,
                    )

                T.gemm(attn, v_new_c, o_frag)
                for i, j in T.Parallel(block_C, block_DV):
                    o_frag[i, j] = scale * o_frag[i, j]
                T.copy(o_frag, o_c)
                for n, vv in T.Parallel(block_C, block_DV):
                    if n < actual_len:
                        o[chunk_token_start + n, out_hid, v_offset + vv] = o_c[n, vv]

        return output_o

    return _func


def gdn_prefill(
    output: torch.Tensor,
    output_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    chunk_size: int = 64,
):
    total_tokens = q.size(0)
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, total_tokens], device=q.device, dtype=torch.int32)
    else:
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32)

    batch = cu_seqlens.size(0) - 1
    head_q = q.size(1)
    head_k = k.size(1)
    head_v = v.size(1)
    dim_k = k.size(2)
    dim_v = v.size(2)

    is_gqa = head_v == head_k and head_q % head_k == 0
    is_gva = head_q == head_k and head_v % head_q == 0
    if not (is_gqa or is_gva):
        raise ValueError(
            "Unsupported head configuration. "
            "Supported: GQA (head_q % head_k == 0 and head_v == head_k) or "
            "GVA (head_q == head_k and head_v % head_q == 0)."
        )
    head_o = max(head_q, head_v)
    head_sab = max(head_k, head_v)
    if head_sab % head_k != 0:
        raise ValueError("head_sab must be divisible by head_k.")
    if head_o % head_q != 0 or head_o % head_sab != 0:
        raise ValueError(
            "head mapping requires head_o divisible by head_q and head_sab."
        )

    q_group_size = head_o // head_q
    sab_group_size = head_o // head_sab
    group_size = sab_group_size

    if alpha is None:
        alpha = torch.ones(total_tokens, head_sab, dtype=torch.float32, device=q.device)
    if beta is None:
        beta = torch.ones(total_tokens, head_sab, dtype=torch.float32, device=q.device)

    if initial_state is None:
        initial_state = torch.zeros(
            batch, head_sab, dim_v, dim_k, dtype=torch.float32, device=q.device
        )

    if output.shape[1] != head_o:
        raise ValueError("output must have shape [total_tokens, head_o, dim_v].")
    if output_state.shape[1] < head_sab:
        raise ValueError("output_state must have at least head_sab heads on dim=1.")

    # compute chunk metadata once (avoid per-block binary search in kernels)
    chunk_offsets, chunk_token_starts, chunk_token_lens, total_chunks = (
        _build_chunk_metadata(cu_seqlens, chunk_size)
    )

    tl_dtype = _torch_dtype_to_tl(q.dtype)
    # ---------- kernel1：w, u ----------
    fused_fn = fused_prepare_compute_w_u_tl(
        total_chunks,
        total_tokens,
        head_sab,
        chunk_size,
        dim_k,
        dim_v,
        head_k=head_k,
        dtype=tl_dtype,
    )(num_stages=2, threads=128, block_DK=64, block_DV=64)
    w, u, cu_g = fused_fn(k, v, alpha, beta, chunk_token_starts, chunk_token_lens)

    # ---------- kernel 2：recurrent state ----------
    h_fn = _h_recurrence_tl(
        total_chunks,
        total_tokens,
        batch,
        head_sab,
        chunk_size,
        dim_k,
        dim_v,
        head_k=head_k,
        dtype=tl_dtype,
    )(num_stages=2, threads=128, block_DV=32)
    S_buf, v_new = h_fn(
        k,
        cu_g,
        w,
        u,
        cu_seqlens,
        chunk_offsets,
        initial_state,
        output_state,
    )

    # ---------- kernel 3：output ----------
    if scale is None:
        scale = dim_k**-0.5

    o_fn = _output_o_tl(
        total_chunks,
        total_tokens,
        head_q,
        head_sab,
        group_size,
        chunk_size,
        scale,
        dim_k,
        dim_v,
        head_o=head_o,
        head_sab=head_sab,
        q_group_size=q_group_size,
        sab_group_size=sab_group_size,
        head_k=head_k,
        dtype=tl_dtype,
    )(num_stages=1, threads=128, block_DV=64)
    o_fn(
        q,
        k,
        cu_g,
        chunk_token_starts,
        chunk_token_lens,
        S_buf,
        v_new,
        output,
    )
