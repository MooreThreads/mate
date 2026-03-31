import torch
import mate._C  # noqa: F401
from typing import Optional, Union, Tuple

from mate.api_logging import mate_api
from .jit.attention.fmha import _fmha_get_metadata as jit_fmha_get_metadata  # noqa: F401
from .jit.attention.fmha import _fmha_fwd as jit_fmha_fwd  # noqa: F401


def _check_valid_asm_input(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q,
    seqused_k,
    window_size,
    learnable_sink,
    cp_world_size=1,
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
    headdim_v = v.shape[-1]

    is_192_128 = headdim_qk == 192 and headdim_v == 128
    is_128_128_or_less = headdim_qk == headdim_v and headdim_qk <= 128

    enable_mubin &= is_192_128 or is_128_128_or_less

    enable_mubin &= seqused_q is None
    enable_mubin &= seqused_k is None

    window_size_left, window_size_right = window_size
    enable_mubin &= window_size_left is None or window_size_left < 0
    enable_mubin &= window_size_right is None or window_size_right < 0

    enable_mubin &= learnable_sink is None

    enable_mubin &= cp_world_size == 1

    if not enable_mubin:
        return enable_mubin

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
    learnable_sink=None,
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=-1,
    pack_gqa=None,
    sm_margin=0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
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

    out, softmax_lse, *rest = jit_fmha_fwd(
        q=q,
        k=k,
        v=v,
        k_new=k_new,
        v_new=v_new,
        q_v=qv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_k_new=cu_seqlens_k_new,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        kv_batch_idx=kv_batch_idx,
        leftpad_k=leftpad_k,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        seqlens_rotary=seqlens_rotary,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        attention_chunk=attention_chunk,
        learnable_sink=learnable_sink,
        softcap=softcap,
        is_rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        mp_margin=sm_margin,
        return_lse=True,
        lse=None,
        out=out,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
    )

    return out, softmax_lse, *rest


def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d = torch.ops.mate.dnn_mha_varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        None,
        None,
    )
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
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
        learnable_sink: Optional[torch.Tensor] = None,
        attention_chunk: int = 0,
        softcap: float = 0.0,
        scheduler_metadata: Optional[torch.Tensor] = None,
        num_splits: int = -1,
        pack_gqa=None,
        deterministic: bool = False,
        sm_margin=0,
        return_attn_probs: bool = False,
        return_softmax_lse: bool = False,
        backend: str = "auto",  # "auto", "mutlass", "mubin"
        cp_world_size: int = 1,
        cp_rank: int = 0,
        cp_tot_seqused_k: Optional[torch.Tensor] = None,
    ):
        select_backend = backend
        if select_backend == "auto":
            enable_mubin = _check_valid_asm_input(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                window_size=window_size,
                learnable_sink=learnable_sink,
                cp_world_size=cp_world_size,
            )

            if enable_mubin:
                select_backend = "mubin"
            else:
                select_backend = "mutlass"

            # assert not enable_mubin

        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
                -0.5
            )

        if select_backend == "mutlass":
            out, softmax_lse, *rest = _flash_attn_forward(
                q=q,
                k=k,
                v=v,
                k_new=None,
                v_new=None,
                qv=qv,
                out=None,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                cu_seqlens_k_new=None,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                page_table=None,
                kv_batch_idx=None,
                leftpad_k=None,
                rotary_cos=None,
                rotary_sin=None,
                seqlens_rotary=None,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                learnable_sink=learnable_sink,
                attention_chunk=attention_chunk,
                softcap=softcap,
                rotary_interleaved=True,
                scheduler_metadata=scheduler_metadata,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                sm_margin=sm_margin,
                cp_world_size=cp_world_size,
                cp_rank=cp_rank,
                cp_tot_seqused_k=cp_tot_seqused_k,
            )

        elif select_backend == "mubin":
            is_varlen = cu_seqlens_q is not None and cu_seqlens_k is not None

            window_size_left, window_size_right = window_size
            assert seqused_q is None
            assert seqused_k is None

            assert window_size_left is None or window_size_left < 0
            assert window_size_right is None or window_size_right < 0

            assert learnable_sink is None

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
                softmax_lse = torch.empty(
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
                causal,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
            )

        else:
            raise ValueError(
                f"Only support backend 'mutlass', 'mubin' and 'auto'! Get unknown backend {select_backend}!"
            )

        is_grad = any(x.requires_grad for x in [q, k, v])
        if is_grad:
            ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.deterministic = deterministic

        should_return_lse = return_attn_probs or return_softmax_lse

        return (out, softmax_lse) if should_return_lse else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors

        # dnn bwd need (b,h max_q) lse but fwd lse is (h, total_q) currently!
        def lse_varlen_to_padded(lse_flat, cu_seqlen_q, max_q, pad_value=0.0):
            # lse_flat: [H, total_Q]
            # lse_padded: [H, b, max_q]
            H, total_Q = lse_flat.shape
            # b = cu_seqlen_q.shape[0] - 1
            device = lse_flat.device
            seq_ids = torch.arange(max_q, device=device).unsqueeze(0)
            offsets = cu_seqlen_q[:-1].unsqueeze(1)
            indices = seq_ids + offsets
            seqlens = cu_seqlen_q[1:] - cu_seqlen_q[:-1]
            valid_mask = seq_ids < seqlens.unsqueeze(1)
            indices = torch.clamp(indices, 0, total_Q - 1)
            lse_gathered = lse_flat[:, indices]
            lse_padded = torch.where(
                valid_mask.unsqueeze(0),
                lse_gathered,
                torch.tensor(pad_value, device=device, dtype=lse_flat.dtype),
            )
            return lse_padded.permute(1, 0, 2).contiguous()

        softmax_lse = lse_varlen_to_padded(softmax_lse, cu_seqlens_q, ctx.max_seqlen_q)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _flash_attn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            softcap=ctx.softcap,
            alibi_slopes=None,
            deterministic=ctx.deterministic,
            rng_state=None,
        )
        dq = dq[..., : dout.shape[-1]]
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@mate_api
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
    learnable_sink: Optional[torch.Tensor] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    pack_gqa=None,
    deterministic: bool = False,
    sm_margin=0,
    return_attn_probs: bool = False,
    return_softmax_lse: bool = False,
    backend: str = "auto",  # "auto", "mutlass", "mubin"
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
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

    seqused_k: Optional[Tensor]
        Tensor with shape ``(batch_size)``
        If given, only this many element of each batch element's keys and values are used.

    softmax_scale: Optional[float]
        The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
    causal: bool
        Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: Tuple[int, int]
        The size of the sliding window. If not (-1, -1), implements sliding window local attention.
    learnable_sink: Optional[Tensor]
        The Learnable Sink tensor for attention, shape ``(nheads, )``.

    softcap: float
        Anything > 0 activates softcapping attention, applied as

        ``logits = softcap * tanh(logits / softcap)`` before the softmax.
        0.0 (default) disables softcapping.
    return_attn_probs: bool
        Whether to return the logsumexp of the attention scores.
    return_softmax_lse: bool
        Whether to return the logsumexp of the attention scores.
        Same as :attr:`return_attn_probs`. Some frameworks (e.g. SGLang) use this name.

    backend: str
        The backend to use. It's recommend to use the default ``auto``.
    cp_world_size: int
        Total number of ranks in the Context Parallelism (CP) group. Default 1 (CP disabled).
        When > 1, the global sequence is assumed to be distributed across ranks using an
        interleaved token pattern, where rank ``r`` holds tokens at positions
        ``[r, r + cp_world_size, r + 2*cp_world_size, ...]``.
    cp_rank: int
        The rank of the current device within the CP group. Default 0.
    cp_tot_seqused_k: Optional[Tensor]
        The **global** (across all CP ranks) cumulative key sequence lengths, shape
        ``(batch_size + 1,)``, dtype ``int32``. Required when CP is enabled (``cp_world_size > 1``)
        so that each rank can correctly compute causal masking boundaries against the full
        key sequence. Ignored when ``cp_world_size == 1``.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor]]
        If :attr:`return_attn_probs` is ``False``, the attention output, shape ``(total_q, nheads, headdim_v)``

        If :attr:`return_attn_probs` is ``True``, a tuple of two tensors:

        * The attention output, shape ``(total_q, nheads, headdim_v)``
        * The log sum exp value, shape ``(nheads, total_q)``
    """

    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        seqused_q,
        seqused_k,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        learnable_sink,
        attention_chunk,
        softcap,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
        return_softmax_lse,
        backend,
        cp_world_size,
        cp_rank,
        cp_tot_seqused_k,
    )


@mate_api
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
    learnable_sink: Optional[torch.Tensor] = None,
    attention_chunk: int = 0,
    softcap: float = 0.0,  # 0.0 means deactivated
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse: bool = False,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
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

    v_cache : Tensor
        The value cache tensor with shape ``(batch_size_cache, seqlen_cache, nheads_k, headdim_v)`` if there's no page_table,
        or ``(num_blocks, page_block_size, nheads_k, headdim_v)`` if there's a page_table (i.e. paged KV cache)

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

    cu_seqlens_k_new: Optional[Tensor]
        The cumulative sequence lengths of the new KV, shape ``(batch_size + 1)``.

        *Not supported now.*
    softmax_scale: Optional[float]
        The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
    causal: bool
        Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: Tuple[int, int]
        The size of the sliding window. If not (-1, -1), implements sliding window local attention.
    learnable_sink: Optional[Tensor]
        The Learnable Sink tensor for attention, shape ``(nheads, )``.

    softcap: float
        Anything > 0 activates softcapping attention, applied as

        ``logits = softcap * tanh(logits / softcap)`` before the softmax.
        0.0 (default) disables softcapping.
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
    cp_world_size: int
        Total number of ranks in the Context Parallelism (CP) group. Default 1 (CP disabled).
        When > 1, the global sequence is assumed to be distributed across ranks using an
        interleaved token pattern, where rank ``r`` holds tokens at positions
        ``[r, r + cp_world_size, r + 2*cp_world_size, ...]``.
    cp_rank: int
        The rank of the current device within the CP group. Default 0.
    cp_tot_seqused_k: Optional[Tensor]
        The **global** (across all CP ranks) cumulative key sequence lengths, shape
        ``(batch_size + 1,)``, dtype ``int32``. Required when CP is enabled (``cp_world_size > 1``)
        so that each rank can correctly compute causal masking boundaries against the full
        key sequence. Ignored when ``cp_world_size == 1``.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor]]
        If :attr:`return_softmax_lse` is ``False``, the attention output, shape ``(batch_size, seqlen, nheads, headdim_v)`` if cu_seqlens_q is None,
        or ``(total_q, nheads, headdim_v)`` if cu_seqlens_q is not None

        If :attr:`return_softmax_lse` is ``True``, a tuple of two tensors:

        * The attention output, shape ``(batch_size, seqlen, nheads, headdim_v)`` if cu_seqlens_q is None,
          or ``(total_q, nheads, headdim_v)`` if cu_seqlens_q is not None
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

    is_mla_decode = qv is not None and qv.shape[-1] == 512 and q.shape[-1] == 64

    if is_mla_decode:
        out, softmax_lse, *rest = torch.ops.mate.dispatch_mla_impl_for_fa_interface(
            qv,
            q,
            v_cache,
            k_cache,
            cache_seqlens,
            page_table,
            softmax_scale,
            causal,
            cu_seqlens_q,
            max_seqlen_q,
            scheduler_metadata[0] if scheduler_metadata is not None else None,
            scheduler_metadata[1] if scheduler_metadata is not None else False,
        )
    else:
        out, softmax_lse, *rest = jit_fmha_fwd(
            q=q,
            k=k_cache,
            v=v_cache,
            k_new=None,
            v_new=None,
            q_v=qv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            cu_seqlens_k_new=cu_seqlens_k_new,
            seqused_q=None,
            seqused_k=cache_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=None,
            page_table=page_table,
            kv_batch_idx=cache_batch_idx,
            leftpad_k=cache_leftpad,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            seqlens_rotary=rotary_seqlens,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=softmax_scale,
            is_causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
            is_rotary_interleaved=rotary_interleaved,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mp_margin=sm_margin,
            return_lse=return_softmax_lse,
            lse=None,
            out=None,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
            cp_tot_seqused_k=cp_tot_seqused_k,
        )

    return (out, softmax_lse, *rest) if return_softmax_lse else out


@mate_api
def get_scheduler_metadata(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_kv,
    headdim,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size=None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    has_softcap=False,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    mp_margin=0,  # Can be tuned if some MPs are used for communication):
):
    seqused_q = maybe_contiguous(seqused_q)
    seqused_k = maybe_contiguous(seqused_k)
    if headdim_v is None:
        headdim_v = headdim
    scheduler_metadata = jit_fmha_get_metadata(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        headdim=headdim,
        headdim_v=headdim_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        causal=causal,
        num_splits=num_splits,
        packgqa=pack_gqa,
        mp_margin=mp_margin,
    )
    return scheduler_metadata
