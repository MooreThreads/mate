import torch
import mate._C  # noqa: F401
from typing import Tuple, Optional
from mate.api_logging import mate_api
from mate.gemm import (
    ragged_m_moe_gemm_16bit,
    masked_moe_gemm_16bit,
    ragged_k_moe_gemm_16bit,
    ragged_m_moe_gemm_8bit,
    masked_moe_gemm_8bit,
    gemm_fp8_nt_groupwise,
    ragged_k_moe_gemm_8bit,
)


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    m_indices: torch.Tensor,
    alignment_m: int = 128,
):
    ragged_m_moe_gemm_16bit(
        a,
        b,
        m_indices,
        d,
        alignment_m=alignment_m,
    )


def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    compiled_dims: str = "nk",
    enable_overlap: bool = False,
    signal: torch.Tensor = None,
):
    res = masked_moe_gemm_16bit(
        a,
        b,
        masked_m,
        d,
        expect_tokens=expected_m,
        enable_overlap=enable_overlap,
        signal=signal,
    )

    return res[2:] if enable_overlap else None


def m_grouped_fp8_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    m_indices: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
    alignment_m: int = 128,
):
    if not disable_ue8m0_cast:
        raise Exception("m_grouped_fp8_gemm_nt_contiguous UE8M0 cast is not supported!")

    ragged_m_moe_gemm_8bit(a, b, m_indices, d, scale_granularity_mnk=recipe, alignment_m=alignment_m)


def m_grouped_fp8_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
    enable_overlap: bool = False,
    signal: torch.Tensor = None,
):
    if not disable_ue8m0_cast:
        raise Exception("m_grouped_fp8_gemm_nt_masked UE8M0 cast is not supported!")

    res = masked_moe_gemm_8bit(
        a,
        b,
        masked_m,
        d,
        recipe,
        expected_m,
        enable_overlap=enable_overlap,
        signal=signal,
    )

    return res[2:] if enable_overlap else None

def k_grouped_fp8_gemm_tn_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks: Optional[list[int]] = None,
    ks_tensor: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
):
    
    if ks_tensor is None:
        if ks is None:
            raise Exception("Must give the ks whether on host or device!")
        ks_tensor = torch.tensor(ks, device="musa", dtype=torch.int32)
    
    res = ragged_k_moe_gemm_8bit(
        a,
        b,
        ks_tensor,
        d,
        scale_granularity_mnk = recipe,
    )

    return d

def k_grouped_bf16_gemm_tn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    ks: Optional[list[int]] = None,
    ks_tensor: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
):
    
    if ks_tensor is None:
        if ks is None:
            raise Exception("Must give the ks whether on host or device!")
        ks_tensor = torch.tensor(ks, device="musa", dtype=torch.int32)

    res = ragged_k_moe_gemm_16bit(
        a,
        b,
        ks_tensor,
        d,
    )

    return d

# legacy deepgemm api
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked


def fp8_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
):
    assert c is None, "Not support GEMM with C"

    return gemm_fp8_nt_groupwise(
        a[0], b[0], a[1], b[1], scale_granularity_mnk=recipe, out=d
    )


@mate_api
def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_kv: int, num_mps: int = 0
) -> torch.Tensor:
    r"""Get metadata for paged MQA logits

    Parameters
    ----------
    context_lens: Tensor
        Context lengths of each query, shape ``(batch_size)``
    block_kv: Tensor
        Block size of kv cache, **must be 64 now**.
    num_mps: int
        Number of MP to execute. 0 means use all MPs of the current device

    Returns
    -------
    Tensor
        Schedule metadata, shape ``(num_mps + 1, 2)``
    """
    return torch.ops.mate.get_paged_mqa_logits_metadata(context_lens, block_kv, num_mps)


@mate_api
def fp8_paged_mqa_logits(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    r"""FP8 Paged MQA logits

    Parameters
    ----------
    q: Tensor
        The FP8 query tensor with shape ``(batch_size, next_n, heads, index_dim)``
    fused_kv_cache: Tensor
        The FP8 kv cache with fp32 scale, shape ``(num_blocks, block_size, 1, index_dim + 4)``
    weights: Tensor
        The FP32 weight tensor for each query, shape ``(batch_size * next_n, heads)``
    context_lens: Tensor
        Context lengths tensor, supports two layouts:

        - **1D** ``(batch_size,)`` — all ``next_n`` draft tokens of request ``i`` share the
          same context length ``context_lens[i]``.  The visible KV range for draft token
          ``j`` is implicitly ``[0, context_lens[i] - next_n + j]``.
        - **2D** ``(batch_size, next_n)`` — each draft token has an independent context
          length ``context_lens[i, j]``, with visible KV range ``[0, context_lens[i, j] - 1]``.
          Useful for tree-based speculative decoding (e.g. Medusa / EAGLE) where tokens
          on different branches see different KV prefixes.

        The shape is auto-detected; ``get_paged_mqa_logits_metadata`` must be called with
        the same ``context_lens`` tensor.
    block_table: Tensor
        Block table tensor with shape ``(batch_size, max_blocks)``
    schedule_meta: Tensor
        Schedule metadata tensor with shape ``(num_mps + 1, 2)``, produced by
        :func:`get_paged_mqa_logits_metadata`
    max_context_len: int
        Maximum context length
    clean_logits: bool
        Whether to zero-fill logit positions that are out of the valid KV range

    Returns
    -------
    Tensor
        FP32 logits, shape ``(batch_size * next_n, max_context_len)``
    """
    return torch.ops.mate.fp8_paged_mqa_logits(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
    )


@mate_api
def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
) -> torch.Tensor:
    r"""FP8 MQA logits.

    This operator computes MQA (multi-query attention) logits for a query sequence against
    a *non-paged* KV tensor. It supports both full logits and "compressed logits" mode
    (when ``max_seqlen_k > 0``), where the output width is limited to a window size.

    Parameters
    ----------
    q : torch.Tensor
        FP8 query tensor with shape ``(seq_len, heads, head_dim)`` and dtype
        ``torch.float8_e4m3fn``.
    kv : tuple[torch.Tensor, torch.Tensor]
        A tuple ``(kv_fp8, kv_scale)``:
        - ``kv_fp8``: FP8 KV tensor with shape ``(seq_len_kv, head_dim)`` and dtype
          ``torch.float8_e4m3fn``. (MQA uses a single KV head.)
        - ``kv_scale``: FP32 scale tensor with shape ``(seq_len_kv,)`` and dtype
          ``torch.float32``.
    weights : torch.Tensor
        FP32 weight tensor with shape ``(seq_len, heads)`` and dtype ``torch.float32``.
    cu_seq_len_k_start : torch.Tensor
        Per-row valid KV start offsets (inclusive) for each query row, with shape
        ``(seq_len,)`` and dtype ``torch.int32``.
    cu_seq_len_k_end : torch.Tensor
        Per-row valid KV end offsets (exclusive) for each query row, with shape
        ``(seq_len,)`` and dtype ``torch.int32``.
    clean_logits : bool, default=False
        Whether to clean logits outside valid KV range. Must be ``False`` when
        ``max_seqlen_k > 0``.
    max_seqlen_k : int, default=0
        If > 0, enables compressed logits mode. The output width becomes ``max_seqlen_k``
        (a windowed logits range per row). In this mode, ``clean_logits`` must be ``False``.

    Returns
    -------
    torch.Tensor
        FP32 logits tensor with shape:
        - ``(seq_len, seq_len_kv)`` if ``max_seqlen_k == 0``
        - ``(seq_len, max_seqlen_k)`` if ``max_seqlen_k > 0``
        and dtype ``torch.float32``.
    """
    kv_fp8, kv_scale = kv
    if max_seqlen_k > 0 and clean_logits:
        raise ValueError("max_seq_len_k is not supported with clean_logits")
    return torch.ops.mate.fp8_mqa_logits(
        q,
        kv_fp8,
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        kv_scale,
        clean_logits,
        int(max_seqlen_k),
    )
