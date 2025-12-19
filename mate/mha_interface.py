import torch
import mate._C  # noqa: F401
from typing import Optional, Union, Tuple


def _check_valid_asm_input(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
):
    enable_mubin = True

    enable_mubin &= q.is_musa
    enable_mubin &= k.is_musa
    enable_mubin &= v.is_musa

    enable_mubin &= q.dtype == torch.float16 or q.dtype == torch.bfloat16
    enable_mubin &= k.dtype == torch.float16 or k.dtype == torch.bfloat16
    enable_mubin &= v.dtype == torch.float16 or v.dtype == torch.bfloat16
    enable_mubin &= q.dtype == k.dtype and q.dtype == v.dtype

    enable_mubin &= q.dim() == 3 or q.dim() == 4
    enable_mubin &= k.dim() == 3 or k.dim() == 4
    enable_mubin &= v.dim() == 3 or v.dim() == 4
    enable_mubin &= q.dim() == k.dim() and q.dim() == v.dim()

    headdim_qk = q.shape[-1]
    enable_mubin &= headdim_qk == 192 or headdim_qk == 128

    headdim_v = v.shape[-1]
    enable_mubin &= headdim_v == 128

    if q.dim() == 3:
        total_seq_q, nr_heads, headdim_qk = q.shape
        total_seq_kv, nr_heads_kv, _ = k.shape
        _, _, headdim_v = v.shape

        enable_mubin &= k.shape == (total_seq_kv, nr_heads_kv, headdim_qk)
        enable_mubin &= v.shape == (total_seq_kv, nr_heads_kv, headdim_v)

        enable_mubin &= cu_seqlens_q.is_musa
        enable_mubin &= cu_seqlens_k.is_musa

        enable_mubin &= cu_seqlens_q is not None
        enable_mubin &= cu_seqlens_k is not None
        enable_mubin &= cu_seqlens_k.numel() == cu_seqlens_q.numel()

        enable_mubin &= max_seqlen_q is not None
        enable_mubin &= max_seqlen_k is not None

    if q.dim() == 4:
        batch, seq_q, nr_heads, headdim_qk = q.shape
        _, seq_kv, nr_heads_kv, _ = k.shape
        _, _, _, headdim_v = v.shape

        enable_mubin &= k.shape == (batch, seq_kv, nr_heads_kv, headdim_qk)
        enable_mubin &= v.shape == (batch, seq_kv, nr_heads_kv, headdim_v)

    return enable_mubin


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_varlen_asm_forward(
    q,
    k,
    v,
    input_cu_seqlen_q,
    input_cu_seqlen_k,
    max_seqlen_q,
    max_seqlen_kv,
    out,
    out_lse,
):
    return torch.ops.mate.flash_atten_varlen_asm(
        q,
        k,
        v,
        input_cu_seqlen_q,
        input_cu_seqlen_k,
        max_seqlen_q,
        max_seqlen_kv,
        out,
        out_lse,
    )


def _flash_attn_forward(
    q,
    k,
    v,
    k_new,
    v_new,
    qv,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqlens_k_new,
    seqused_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    page_table,
    kv_batch_idx,
    leftpad_k,
    rotary_cos,
    rotary_sin,
    seqlens_rotary,
    q_descale,
    k_descale,
    v_descale,
    softmax_scale,
    causal,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
):
    q, k, k_new, v_new = [maybe_contiguous(x) for x in (q, k, k_new, v_new)]
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
    cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new)
    ]
    seqused_q, seqused_k = [maybe_contiguous(x) for x in (seqused_q, seqused_k)]
    page_table, kv_batch_idx, leftpad_k = [
        maybe_contiguous(x) for x in (page_table, kv_batch_idx, leftpad_k)
    ]
    rotary_cos, rotary_sin = [maybe_contiguous(x) for x in (rotary_cos, rotary_sin)]
    seqlens_rotary = maybe_contiguous(seqlens_rotary)
    out, softmax_lse, *rest = torch.ops.mate.fmha_fwd(
        q,
        k,
        v,
        k_new,
        v_new,
        qv,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
    )
    return out, softmax_lse, *rest


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Tuple = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa=None,
    deterministic: bool = False,
    sm_margin=0,
    return_attn_probs: bool = False,
    backend: str = "auto",  # "auto", "mutlass", "mubin"
):
    r"""
    FlashAttention3 compaitible API: forward with varlen or non-varlen inputs

    Parameters
    ----------
    q : Tensor
        The query tensor with shape ``(batch_size, seqlen, nheads, headdim)`` if cu_seqlen_q is None,
        or ``(total_q, nheads, headdim)`` if cu_seqlen_q is not None.
    k : Tensor
        The key tensor with shape ``(batch_size, seqlen_k, nheads_k, headdim)`` if cu_seqlen_k is None,
        or ``(total_k, nheads_k, headdim)`` if cu_seqlen_k is not None.
    v : Tensor
        The value tensor with shape ``(batch_size, seqlen_k, nheads_k, headdim)`` if cu_seqlen_k is None,
        or ``(total_k, nhead_k, headdim_v)`` if cu_seqlen_k is not None.
    cu_seqlens_q : Optional[Tensor]
        The cumulative sequence length tensor for query, shape ``(batch_size + 1)``
    cu_seqlens_k : Optional[Tensor]
        The cumulative sequence length tensor for key/value, shape ``(batch_size + 1)``
    max_seqlen_q : Optional[int]
        The maximum sequence length for query, must provided if varlen forward
    max_seqlen_k : Optional[int]
        The maximum sequence length for key/value
    seqused_q: Optional[Tensor]
        Tensor with shape ``(batch_size)``
        If given, only this many element of each batch element's queries and outputs are used.

        *Not supported now.*
    seqused_k: Optional[Tensor]
        Tensor with shape ``(batch_size)``
        If given, only this many element of each batch element's keys and values are used.

        *Not supported now.*
    softmax_scale: Optional[float]
        The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
    causal: bool
        Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: Tuple[int, int]
        The size of the sliding window. If not (-1, -1), implements sliding window local attention.

        *Only (-1, -1) is supported now.*
    softcap: float
        Anything > 0 activates softcapping attent

        *Only 0.0 is supported now.*
    return_attn_probs: bool
        Whether to return the logsumexp of the attention scores.
    backend: str
        The backend to use. It's recommend to use the default ``auto``.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor]]
        If :attr:`return_attn_probs` is ``False``, the attention output, shape ``(total_q, nheads, headdim_v)``

        If :attr:`return_attn_probs` is ``True``, a tuple of two tensors:

        * The attention output, shape ``(total_q, nheads, headdim_v)``
        * The log sum exp value, shape ``(nheads, total_q)``
    """
    if backend == "auto":
        enable_mubin = _check_valid_asm_input(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        )

        if enable_mubin:
            backend = "mubin"
        else:
            backend = "mutlass"

    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )

    if backend == "mutlass":
        # out, q, k, v, out_padded, softmax_lse = _flash_attn_varlen_forward(
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,
            None,  # rotary_cos/sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )

    elif backend == "mubin":
        is_varlen = (
            cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or max_seqlen_k is not None
            or max_seqlen_q is not None
            or seqused_q is not None
            or seqused_k is not None
        )

        if is_varlen:
            assert cu_seqlens_q is not None
            assert cu_seqlens_k is not None
            assert max_seqlen_k is not None
            assert max_seqlen_q is not None

            total_seqlen, nr_heads, _ = q.shape
            headdim_v = v.shape[-1]

            out = torch.empty(
                (total_seqlen, nr_heads, headdim_v), dtype=q.dtype, device=q.device
            )

            batch = cu_seqlens_q.shape[0] - 1
            softmax_lse = torch.zeros(
                (nr_heads, total_seqlen),
                dtype=torch.float32,
                device=q.device,
            )

        else:
            # is no varlen
            # bshd
            batch, seq_q, nr_heads, _ = q.shape
            headdim_v = v.shape[-1]

            out = torch.empty(
                (batch, seq_q, nr_heads, headdim_v), dtype=q.dtype, device=q.device
            )

            softmax_lse = torch.empty(
                (batch, nr_heads, seq_q), dtype=torch.float32, device=q.device
            )

        out, softmax_lse = torch.ops.mate.flash_atten_varlen_asm(
            q,
            k,
            v,
            softmax_scale,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        )

    else:
        raise ValueError(
            f"Only support backend 'mutlass', 'mubin' and 'auto'! Get unknown backend {backend}!"
        )

    return (out, softmax_lse) if return_attn_probs else out


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple = (-1, -1),  # -1 means infinite context window
    attention_chunk: int = 0,
    softcap: float = 0.0,  # 0.0 means deactivated
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse: bool = False,
):
    r"""FlashAttention3 compatible API: forward with kv cache

    Parameters
    ----------
    q : Tensor
        The query tensor with shape ``(batch_size, seqlen, nheads, headdim)`` if cu_seqlens_q is None,
        or ``(total_q, nheads, headdim)`` if cu_seqlens_q is not None
    k_cache : Tensor
        The key cache tensor with shape ``(batch_size_cache, seqlen_cache, nheads_k, headdim)`` if there's no page_table,
        or ``(num_blocks, page_block_size, nheads_k, headdim)`` if there's a page_table (i.e. paged KV cache)

        *page_block_size must be 64 and only paged KV Cache mode is supported now.*
    v_cache : Tensor
        The value cache tensor with shape ``(batch_size_cache, seqlen_cache, nheads_k, headdim_v)`` if there's no page_table,
        or ``(num_blocks, page_block_size, nheads_k, headdim_v)`` if there's a page_table (i.e. paged KV cache)

        *page_block_size must be 64 and only paged KV Cache mode is supported now.*
    k : Optional[Tensor]
        The key tensor with shape ``(batch_size, seqlen_new, nheads_k, headdim)``. If not None, we concatenate
        k with k_cache, starting at the indices specified by cache_seqlens.

        *Not supported now.*
    v : Optional[Tensor]
        The value tensor with shape ``(batch_size, seqlen_new, nheads_k, headdim_v)``. Similar to k.

        *Not supported now.*
    rotary_cos: Optional[Tensor]
        Tensor with shape ``(seqlen_ro, rotary_dim / 2)``. If not None, we apply rotary embedding to k and q.
        Only applicable if k and v are passed in. rotary_dim must be divisible by 16.

        *Not supported now.*
    rotary_sin: Optional[Tensor]
        Tensor with shape ``(seqlen_ro, rotary_dim / 2)``. Similar to rotary_cos.

        *Not supported now.*
    cache_seqlens: Union[int, Tensor]
        The sequence lengths of the KV cache, shape ``(batch_size)`` if it is tensor.
    cache_batch_idx: Optional[Tensor]
        The indices used to index into the KV cache, shape ``(batch_size)``.
        If the indices are not distinct, and k and v are provided, the values updated in the cache might come from any of the duplicate indices.

        *Not supported now.*
    cache_leftpad: Optional[Tensor]
        The index that the KV cache starts. If None, assume 0.

        *Not supported now.*
    page_table: Optional[Tensor]
        The page table tensor with shape ``(batch_size, max_num_blocks_per_seq)``

        *Must provide now.*
    cu_seqlens_q: Optional[Tensor]
        The cumulative sequence lengths of the query, shape ``(batch_size + 1)``.

        *Not supported now.*
    cu_seqlens_k_new: Optional[Tensor]
        The cumulative sequence lengths of the new KV, shape ``(batch_size + 1)``.

        *Not supported now.*
    softmax_scale: Optional[float]
        The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
    causal: bool
        Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: Tuple[int, int]
        The size of the sliding window. If not (-1, -1), implements sliding window local attention.

        *Only (-1, -1) is supported now.*
    softcap: float
        Anything > 0 activates softcapping attent

        *Only 0.0 is supported now.*
    rotary_interleaved: bool
        If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
        rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
        (i.e. GPT-NeoX style).
    num_splits: int
        If > 1, split the key/value into this many chunks along the sequence.
        If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
        to automatically determine the number of splits.
        Don't change this unless you know what you are doing.
    return_softmax_lse: bool
        Whether to return the logsumexp of the attention scores.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor]]
        If :attr:`return_softmax_lse` is ``False``, the attention output, shape ``(batch_size, seqlen, nheads, headdim_v)`` if cu_seqlens_q is None,
        or ``(batch_size, seqlen, nheads, headdim_v)`` if cu_seqlens_q is not None

        If :attr:`return_softmax_lse` is ``True``, a tuple of two tensors:

        * The attention output, shape ``(batch_size, seqlen, nheads, headdim_v)`` if cu_seqlens_q is None,
          or ``(batch_size, seqlen, nheads, headdim_v)`` if cu_seqlens_q is not None
        * The log sum exp value, shape ``(batch_size, nheads, seqlen)`` if cu_seqlens_q is None,
          or ``(nheads, total_q)`` if cu_seqlens_q is not None

    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)

    out, softmax_lse, *rest = _flash_attn_forward(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        None,  # out
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens,
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        rotary_seqlens,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )

    return (out, softmax_lse) if return_softmax_lse else out
