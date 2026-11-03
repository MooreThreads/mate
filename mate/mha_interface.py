import functools
import torch
from typing import List, Optional, Union, Tuple

from mate.api_logging import mate_api
from mate.mate_runtime import get_physical_num_mps
from mate.utils import ceil_div
from .jit.mubin.flash_attention import flash_atten_varlen_asm_mubin
from .jit.mubin.flash_mla import flash_mla_asm_mubin
from .jit.mla_ops import get_mla_ops_module
from .jit.attention.fmha import (
    _fmha_get_metadata as jit_fmha_get_metadata,
    _fmha_fwd as jit_fmha_fwd,
)  # noqa: F401
from .jit.attention.fmha.fmha_combine import _flash_attn_combine


@functools.cache
def _get_mla_ops():
    return get_mla_ops_module()


def _check_valid_asm_input(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    page_table,
    seqused_q,
    seqused_k,
    qv,
    window_size,
    learnable_sink,
    attention_chunk,
    softcap,
    cp_world_size=1,
):
    if not (q.is_musa and k.is_musa and v.is_musa):
        return False

    supported_dtypes = (torch.float16, torch.bfloat16)
    if (
        q.dtype not in supported_dtypes
        or k.dtype not in supported_dtypes
        or v.dtype not in supported_dtypes
        or q.dtype != k.dtype
        or q.dtype != v.dtype
    ):
        return False

    if (
        q.dim() not in (3, 4)
        or k.dim() not in (3, 4)
        or v.dim() not in (3, 4)
        or q.dim() != k.dim()
        or q.dim() != v.dim()
    ):
        return False

    headdim_qk = q.shape[-1]
    headdim_v = v.shape[-1]

    is_192_128 = headdim_qk == 192 and headdim_v == 128
    is_128_128_or_less = headdim_qk == headdim_v and headdim_qk <= 128

    window_size_left, window_size_right = window_size
    if (
        not (is_192_128 or is_128_128_or_less)
        or page_table is not None
        or seqused_q is not None
        or seqused_k is not None
        or (window_size_left is not None and window_size_left >= 0)
        or (window_size_right is not None and window_size_right >= 0)
        or qv is not None
        or softcap != 0.0
        or learnable_sink is not None
        or attention_chunk != 0
        or cp_world_size != 1
    ):
        return False

    if q.dim() == 3:
        total_seq_q, nr_heads, headdim_qk = q.shape
        total_seq_kv, nr_heads_kv, _ = k.shape
        _, _, headdim_v = v.shape

        if (
            cu_seqlens_q is None
            or cu_seqlens_k is None
            or max_seqlen_q is None
            or max_seqlen_k is None
        ):
            return False

        return (
            k.shape == (total_seq_kv, nr_heads_kv, headdim_qk)
            and v.shape == (total_seq_kv, nr_heads_kv, headdim_v)
            and cu_seqlens_q.is_musa
            and cu_seqlens_k.is_musa
            and cu_seqlens_k.numel() == cu_seqlens_q.numel()
        )

    batch, seq_q, nr_heads, headdim_qk = q.shape
    _, seq_kv, nr_heads_kv, _ = k.shape
    _, _, _, headdim_v = v.shape

    return k.shape == (
        batch,
        seq_kv,
        nr_heads_kv,
        headdim_qk,
    ) and v.shape == (batch, seq_kv, nr_heads_kv, headdim_v)


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _reduce_max_logits(
    max_logits: torch.Tensor,
    seqused_q: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if max_logits.numel() == 0:
        return max_logits.new_full((max_logits.shape[-2],), float("-inf"))
    if seqused_q is not None:
        query = torch.arange(max_logits.shape[-1], device=max_logits.device)
        if cu_seqlens_q is None:
            invalid = (query[None, :] >= seqused_q[:, None])[:, None, :]
        else:
            batch = torch.searchsorted(cu_seqlens_q[1:], query, right=True)
            invalid = (query - cu_seqlens_q[batch] >= seqused_q[batch])[None, :]
        max_logits = max_logits.masked_fill(invalid, float("-inf"))
    dims = (0, 2) if max_logits.ndim == 3 else 1
    maximum = max_logits.amax(dim=dims)
    # Preserve all-masked heads when MUSA amax returns a finite sentinel.
    valid_count = torch.count_nonzero(~torch.isneginf(max_logits), dim=dims)
    return maximum.masked_fill(valid_count == 0, float("-inf"))


def _allocate_mla_decode_outputs(q: torch.Tensor, head_dim_v: int):
    if q.dim() == 4:
        out = torch.empty(
            (*q.shape[:-1], head_dim_v),
            dtype=q.dtype,
            device=q.device,
        )
        softmax_lse = torch.empty(
            q.shape[:-1],
            dtype=torch.float32,
            device=q.device,
        )
    else:
        out = torch.empty(
            (q.shape[0], q.shape[1], head_dim_v),
            dtype=q.dtype,
            device=q.device,
        )
        softmax_lse = torch.empty(
            (q.shape[1], q.shape[0]),
            dtype=torch.float32,
            device=q.device,
        )
    return out, softmax_lse


_FLASH_MLA_METADATA_SIZE = 8


def _prepare_mla_scheduler_metadata_from_workspace(
    workspace: torch.Tensor,
    is_inited: bool,
    q_nope: torch.Tensor,
    ckv: torch.Tensor,
    seqlens_k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
):
    if workspace.dim() != 1:
        raise ValueError("MLA scheduler workspace must be 1D")
    if workspace.dtype != torch.uint8:
        raise ValueError("MLA scheduler workspace must have dtype torch.uint8")
    if not workspace.is_contiguous():
        raise ValueError("MLA scheduler workspace must be contiguous")
    if workspace.device != q_nope.device:
        raise ValueError("MLA scheduler workspace must be on the query device")
    if cu_seqlens_q is not None and max_seqlen_q is None:
        raise ValueError("max_seqlen_q must be provided when cu_seqlens_q is set")

    seqlen_q = q_nope.shape[1] if cu_seqlens_q is None else max_seqlen_q
    num_heads_q = q_nope.shape[-2]
    num_heads_k = ckv.shape[-2] if ckv.dim() == 4 else 1
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    num_mp_parts = max(
        get_physical_num_mps(q_nope.device)
        // num_heads_k
        // ceil_div(q_seq_per_hk, 128),
        1,
    )
    batch = seqlens_k.shape[0]
    metadata_words = num_mp_parts * _FLASH_MLA_METADATA_SIZE
    num_splits_words = batch + 1
    needed_bytes = (metadata_words + num_splits_words) * 4
    if workspace.numel() < needed_bytes:
        raise ValueError("MLA scheduler workspace is too small")

    workspace_i32 = workspace[:needed_bytes].view(torch.int32)
    tile_scheduler_metadata = workspace_i32[:metadata_words].view(
        num_mp_parts,
        _FLASH_MLA_METADATA_SIZE,
    )
    num_splits = workspace_i32[metadata_words : metadata_words + num_splits_words]

    if not is_inited:
        _get_mla_ops().get_function("get_mla_decoding_metadata")(
            seqlens_k,
            q_seq_per_hk,
            num_heads_k,
            None,
            False,
            None,
            tile_scheduler_metadata,
            num_splits,
            None,
            None,
            None,
            None,
            None,
        )
    return tile_scheduler_metadata, num_splits


def _prepare_mla_query_input(
    x: torch.Tensor, *, require_seq_dense: bool
) -> torch.Tensor:
    # Match Python-side materialization to the exact MLA backend stride contract.
    if x.stride(-1) != 1:
        return x.contiguous()
    if require_seq_dense and x.dim() == 4:
        if x.stride(1) != x.shape[-2] * x.stride(2):
            return x.contiguous()
    if require_seq_dense and x.dim() == 3:
        if x.stride(0) != x.shape[-2] * x.stride(1):
            return x.contiguous()
    return x


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
    return_max_logits: bool = False,
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
        return_max_logits=return_max_logits,
    )

    return out, softmax_lse, *rest


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
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        qv: Optional[torch.Tensor] = None,
        q_descale: Optional[torch.Tensor] = None,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        window_size: Union[Tuple, List, None] = (-1, -1),
        learnable_sink: Optional[torch.Tensor] = None,
        attention_chunk: Optional[int] = 0,
        softcap: float = 0.0,
        scheduler_metadata: Optional[torch.Tensor] = None,
        num_splits: int = -1,
        pack_gqa=None,
        deterministic: bool = False,
        sm_margin=0,
        return_softmax_lse: bool = False,
        backend: str = "auto",  # "auto", "mutlass", "mubin"
        cp_world_size: int = 1,
        cp_rank: int = 0,
        cp_tot_seqused_k: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        return_max_logits: bool = False,
        bwd_backend: str = "auto",
    ):
        if window_size is None:
            window_size = (-1, -1)
        if attention_chunk is None:
            attention_chunk = 0
        select_backend = backend
        if return_max_logits:
            if select_backend == "tilelang":
                raise ValueError(
                    "max logits are supported by the mubin and mutlass backends"
                )
            if isinstance(scheduler_metadata, tuple):
                raise ValueError("max logits do not support MLA scheduler metadata")
        if select_backend == "auto":
            enable_mubin = _check_valid_asm_input(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                page_table=page_table,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                qv=qv,
                window_size=window_size,
                learnable_sink=learnable_sink,
                attention_chunk=attention_chunk,
                softcap=softcap,
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
        if return_max_logits and softmax_scale < 0 and softcap <= 0:
            raise ValueError("return_max_logits requires nonnegative softmax_scale")

        if select_backend == "mutlass":
            out, softmax_lse, *rest = _flash_attn_forward(
                q=q,
                k=k,
                v=v,
                k_new=None,
                v_new=None,
                qv=qv,
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                cu_seqlens_k_new=None,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                page_table=page_table,
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
                return_max_logits=return_max_logits,
            )

        elif select_backend == "mubin":
            is_varlen = cu_seqlens_q is not None and cu_seqlens_k is not None

            window_size_left, window_size_right = window_size
            assert seqused_q is None
            assert seqused_k is None

            if not (
                (window_size_left is None or window_size_left < 0)
                and (window_size_right is None or window_size_right < 0)
            ):
                raise ValueError("MUBIN backend does not support windowed attention")

            assert qv is None
            assert learnable_sink is None
            assert softcap == 0.0
            assert attention_chunk == 0

            assert q.dtype in [torch.float16, torch.bfloat16]
            assert q.dtype == k.dtype and q.dtype == v.dtype

            assert page_table is None

            if is_varlen:
                assert cu_seqlens_q is not None
                assert cu_seqlens_k is not None
                assert max_seqlen_k is not None
                assert max_seqlen_q is not None

                total_seqlen, nr_heads, _ = q.shape
                headdim_v = v.shape[-1]

                if out is None:
                    out = torch.empty(
                        (total_seqlen, nr_heads, headdim_v),
                        dtype=q.dtype,
                        device=q.device,
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

                if out is None:
                    out = torch.empty(
                        (batch, seq_q, nr_heads, headdim_v),
                        dtype=q.dtype,
                        device=q.device,
                    )

                softmax_lse = torch.empty(
                    (batch, nr_heads, seq_q), dtype=torch.float32, device=q.device
                )

            max_logits_rows = (
                torch.empty_like(softmax_lse) if return_max_logits else None
            )
            rest = (max_logits_rows,)
            flash_atten_varlen_asm_mubin(
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
                max_logits=max_logits_rows,
            )

        else:
            raise ValueError(
                f"Only support backend 'mutlass', 'mubin' and 'auto'! Get unknown backend {select_backend}!"
            )

        is_grad = any(x.requires_grad for x in [q, k, v])
        if is_grad:
            ctx.save_for_backward(
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
            )
            ctx.has_qv = qv is not None
            ctx.has_page_table = page_table is not None
            ctx.has_learnable_sink = learnable_sink is not None
            ctx.has_q_descale = q_descale is not None
            ctx.has_k_descale = k_descale is not None
            ctx.has_v_descale = v_descale is not None
            ctx.attention_chunk = attention_chunk
            ctx.cp_world_size = cp_world_size
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.deterministic = deterministic
            ctx.bwd_backend = bwd_backend

        should_return_lse = return_softmax_lse

        if return_max_logits:
            max_logits = _reduce_max_logits(rest[-1], seqused_q, cu_seqlens_q)
            ctx.mark_non_differentiable(max_logits)
            return (
                (out, softmax_lse, max_logits)
                if should_return_lse
                else (out, max_logits)
            )

        return (out, softmax_lse) if should_return_lse else out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
        ) = ctx.saved_tensors
        bwd_backend = ctx.bwd_backend
        if bwd_backend not in ("auto", "mubin", "tilelang"):
            raise ValueError(f"Unknown FlashAttention backward backend: {bwd_backend}")
        # These forward contracts are absent from the ASM backward ABI. Keep
        # auto on the existing TileLang path rather than ignoring their inputs.
        extra_bwd_features = (
            ctx.has_qv
            or ctx.has_page_table
            or ctx.has_learnable_sink
            or bool(ctx.attention_chunk)
            or ctx.cp_world_size != 1
            or ctx.has_q_descale
            or ctx.has_k_descale
            or ctx.has_v_descale
        )
        if extra_bwd_features:
            if bwd_backend == "mubin":
                raise ValueError(
                    "MUBIN backward does not support qv, paged KV, sinks, "
                    "chunked attention, context parallelism, or descales"
                )
            if bwd_backend == "auto":
                bwd_backend = "tilelang"
        from .flash_attention.backward import (
            flashattn_varlen_bwd_interface,
        )

        dq, dk, dv = flashattn_varlen_bwd_interface(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            is_causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            smscale=ctx.softmax_scale,
            dtype=None,
            is_bhsd=False,
            deterministic=ctx.deterministic,
            backend=bwd_backend,
        )
        return (dq, dk, dv) + (None,) * (len(ctx.needs_input_grad) - 3)


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
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Union[Tuple, List, None] = (-1, -1),
    learnable_sink: Optional[torch.Tensor] = None,
    attention_chunk: Optional[int] = 0,
    softcap: float = 0.0,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    pack_gqa=None,
    deterministic: bool = False,
    sm_margin=0,
    return_softmax_lse: bool = False,
    backend: str = "auto",  # "auto", "mutlass", "mubin"
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_max_logits: bool = False,
    bwd_backend: str = "auto",
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
    return_softmax_lse: bool
        Whether to return the logsumexp of the attention scores.

    backend: str
        The backend to use: ``auto``, ``mutlass``, or ``mubin``.
        ``auto`` preserves the existing MUBIN-then-MUTLASS selection.
    cp_world_size: int
        Total number of ranks in the Context Parallelism (CP) group. Default 1 (CP disabled).
        When > 1, the global sequence is assumed to be distributed across ranks using an
        interleaved token pattern, where rank ``r`` holds tokens at positions
        ``[r, r + cp_world_size, r + 2*cp_world_size, ...]``.
    cp_rank: int
        The rank of the current device within the CP group. Default 0.
    cp_tot_seqused_k: Optional[Tensor]
        The final **global** key sequence length for each batch across all CP ranks, shape
        ``(batch_size,)``, dtype ``int32``. Required when CP is enabled (``cp_world_size > 1``)
        so that each rank can compute masking boundaries against the full key sequence.
        Ignored when ``cp_world_size == 1``.
    return_max_logits: bool
        Return FP32, non-differentiable per-head maxima with shape ``(nheads,)``,
        reduced over valid batch/query/key positions after scale/softcap/mask,
        excluding sinks. Heads with no valid attention scores are ``-inf``.
        Return ``(out, max_logits)`` or ``(out, lse, max_logits)`` when LSE is requested.
        ``auto`` selects ASM where supported, otherwise MUTLASS. CP maxima are rank-local. FP8 Q/K descales
        and, without softcap, softmax scale must be non-negative.

    bwd_backend: str
        Backward backend: ``auto`` prefers MUBIN on supported MP31 inputs and
        otherwise uses TileLang. ``mubin`` requires ASM support; ``tilelang``
        explicitly selects TileLang. Independent of the forward backend.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
        If :attr:`return_softmax_lse` is ``False``, the attention output, shape ``(total_q, nheads, headdim_v)``

        If :attr:`return_softmax_lse` is ``True``, a tuple of two tensors:

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
        page_table,
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
        return_softmax_lse,
        backend,
        cp_world_size,
        cp_rank,
        cp_tot_seqused_k,
        out,
        return_max_logits,
        bwd_backend,
    )


@mate_api
def fmha_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
    *,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute attention gradients from saved forward output and natural-log LSE.

    Parameters
    ----------
    q, k, v : Tensor
        Dense BSHD tensors or packed THD tensors, as for
        :func:`flash_attn_varlen_func`. Only supplied gradient buffers are written.
    out, dout : Tensor
        Saved forward output and its gradient, in the same layout as Q.
    softmax_lse : Tensor
        FP32 natural-log LSE: (B, H, S) for dense, (H, T) for packed Q.
        O/LSE must come from forward with the same scale, mask and softcap;
        they are logical values and are not tied to a particular backend.
    cu_seqlens_q, cu_seqlens_k : Tensor, optional
        INT32 cumulative lengths for packed Q/KV. Supply both or neither.
    max_seqlen_q, max_seqlen_k : int, optional
        Maximum sequence lengths. Required for packed inputs; inferred for dense.
    seqused_q, seqused_k : Tensor, optional
        Per-sequence used lengths, subject to backend support.
    softmax_scale : float, optional
        Scale applied to QK; defaults to 1/sqrt(Q head dimension).
    is_causal : bool
        Apply bottom-right causal masking.
    window_size_left, window_size_right : int
        Inclusive left/right offsets around the bottom-right diagonal;
        -1 means unbounded on that side.
    softcap : float
        Forward logit softcap; zero disables it.
    deterministic : bool
        Request deterministic gradients; unsupported backends reject it.
    backend : str
        ``auto`` currently selects ``tilelang``. Explicit selection never
        falls back.
    sm_margin : int
        Reserved compute units for backend scheduling. TileLang ignores this hint.
    dq, dk, dv : Tensor, optional
        Caller-owned gradient buffers matching each input's shape, dtype and
        device. Their strides may differ. Supplied buffers are written in place
        and returned by identity; omitted buffers are allocated by the backend.
        Buffers must not overlap inputs or each other. TileLang requires unit
        inner stride and, for aligned head dimensions, outer strides divisible
        by 8. Aligned outputs are written directly; head-dimension padding may
        require temporary storage.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        dQ, dK, dV and FP32 softmax_d in logical LSE layout, with unused query
        rows zeroed. This explicit backward does not recompute forward or support
        higher-order gradients. Backend shape/architecture restrictions
        propagate as errors. This release provides the MP31 TileLang backend.
    """
    if backend == "auto":
        backend = "tilelang"
    if backend == "tilelang":
        from .flash_attention.tilelang.flash_attention_varlen_bwd import (  # type: ignore[attr-defined]
            flashattn_varlen_bwd_interface,
        )

        # sm_margin is a scheduling hint; TileLang currently ignores it.
        return flashattn_varlen_bwd_interface(
            q=q,
            k=k,
            v=v,
            out=out,
            dout=dout,
            softmax_lse=softmax_lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            smscale=softmax_scale,
            is_causal=is_causal,
            window_size=(window_size_left, window_size_right),
            softcap=softcap,
            deterministic=deterministic,
            is_bhsd=False,
            dq=dq,
            dk=dk,
            dv=dv,
            return_softmax_d=True,
        )
    raise ValueError(f"Unknown attention backward backend: {backend}")


@mate_api
def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    return _flash_attn_combine(out_partial, lse_partial, out, out_dtype)


@mate_api
def flash_attn_with_kvcache(
    q: Optional[torch.Tensor],
    k_cache: Optional[torch.Tensor],
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
    window_size: Union[Tuple, List, None] = (
        -1,
        -1,
    ),  # -1 means infinite context window
    learnable_sink: Optional[torch.Tensor] = None,
    attention_chunk: Optional[int] = 0,
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
    only_qv: bool = False,
    return_max_logits: bool = False,
):
    r"""FlashAttention3 compatible API: forward with kv cache

    Parameters
    ----------
    q : Optional[Tensor]
        The query tensor with shape ``(batch_size, seqlen, nheads, headdim)`` if cu_seqlens_q is None,
        or ``(total_q, nheads, headdim)`` if cu_seqlens_q is not None. May be ``None`` when
        ``only_qv=True``.
    k_cache : Optional[Tensor]
        The key cache tensor with shape ``(batch_size_cache, seqlen_cache, nheads_k, headdim)`` if there's no page_table,
        or ``(num_blocks, page_block_size, nheads_k, headdim)`` if there's a page_table (i.e. paged KV cache).
        May be ``None`` when ``only_qv=True``.

    v_cache : Tensor
        The value cache tensor with shape ``(batch_size_cache, seqlen_cache, nheads_k, headdim_v)`` if there's no page_table,
        or ``(num_blocks, page_block_size, nheads_k, headdim_v)`` if there's a page_table (i.e. paged KV cache)

    k : Optional[Tensor]
        The key tensor with shape ``(batch_size, seqlen_new, nheads_k, headdim)`` if cu_seqlens_k_new is None,
        or ``(total_k_new, nheads_k, headdim)`` if cu_seqlens_k_new is not None.
        If k is not None, we concatenate k with k_cache, starting at the indices specified by cache_seqlens.

    v : Optional[Tensor]
        The value tensor with shape ``(batch_size, seqlen_new, nheads_k, headdim_v)`` if cu_seqlens_k_new is None.
        or ``(total_k_new, nheads_k, headdim_v)`` if cu_seqlens_k_new is not None.
        Similar to k.

    rotary_cos: Optional[Tensor]
        Tensor with shape ``(seqlen_ro, rotary_dim / 2)``. If not None, we apply rotary embedding to k and q.
        Only applicable if k and v are passed in. ``rotary_dim`` must be ``<= headdim`` and divisible by 16.
        ``rotary_cos`` must be on MUSA and have the same dtype as q.
    rotary_sin: Optional[Tensor]
        Tensor with shape ``(seqlen_ro, rotary_dim / 2)``. Similar to rotary_cos and must have the same shape and dtype.
    cache_seqlens: Union[int, Tensor]
        The sequence lengths of the KV cache, shape ``(batch_size)`` if it is tensor.

    cache_batch_idx: Optional[Tensor]
        The int32 indices used to index into the KV cache, shape ``(batch_size,)``.
        The tensor must be on MUSA and contiguous.
        If the indices are not distinct, and k and v are provided, the values updated in the cache might come from any of the duplicate indices.
    cache_leftpad: Optional[Tensor]
        The int32 left padding offset where the KV cache starts for each batch, shape ``(batch_size,)``.
        The tensor must be on MUSA and contiguous. If None, assume 0.
    page_table: Optional[Tensor]
        The page table tensor with shape ``(batch_size, max_num_blocks_per_seq)``

    cu_seqlens_q: Optional[Tensor]
        The cumulative sequence lengths of the query, shape ``(batch_size + 1)``.

    cu_seqlens_k_new: Optional[Tensor]
        The cumulative sequence lengths of the new KV, shape ``(batch_size + 1)``.

    rotary_seqlens: Optional[Tensor]
        Optional int32 tensor with shape ``(batch_size,)`` used as the rotary position length for each batch.

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
        If True, rotary embedding uses GPT-J style and combines dimensions 0 & 1, 2 & 3, etc. If False,
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
        The final **global** key sequence length for each batch across all CP ranks, shape
        ``(batch_size,)``, dtype ``int32``. Required when CP is enabled (``cp_world_size > 1``).
        With AppendKV, each value must include the KV appended by all ranks in this call;
        ``cache_seqlens`` remains the rank-local pre-append cache position. With rotary,
        ``rotary_seqlens`` is the global pre-append position and each rank's new KV must use
        the cyclic global positions owned by that rank. Ignored when ``cp_world_size == 1``.
    only_qv: bool
        Skip the QK score and use only the QV score.
    return_max_logits: bool
        Return FP32 per-head maxima with shape ``(nheads,)``, reduced over valid
        batch/query/key positions after scale/softcap/mask, excluding sinks.
        Heads with no valid attention scores are ``-inf``. CP maxima are rank-local.
        Return ``(out, max_logits)`` or ``(out, lse, max_logits)`` when LSE is requested.
        FP8 Q/K descales and, without softcap, softmax scale must be non-negative.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, ...]]
        If :attr:`return_softmax_lse` is ``False``, the attention output, shape ``(batch_size, seqlen, nheads, headdim_v)`` if cu_seqlens_q is None,
        or ``(total_q, nheads, headdim_v)`` if cu_seqlens_q is not None

        If :attr:`return_softmax_lse` is ``True``, a tuple of two tensors:

        * The attention output, shape ``(batch_size, seqlen, nheads, headdim_v)`` if cu_seqlens_q is None,
          or ``(total_q, nheads, headdim_v)`` if cu_seqlens_q is not None
        * The log sum exp value, shape ``(batch_size, nheads, seqlen)`` if cu_seqlens_q is None,
          or ``(nheads, total_q)`` if cu_seqlens_q is not None

    """
    if return_max_logits and isinstance(scheduler_metadata, tuple):
        raise ValueError("max logits do not support MLA scheduler metadata")
    if only_qv:
        if qv is None:
            raise ValueError("only_qv=True requires qv")
        if isinstance(scheduler_metadata, tuple):
            raise ValueError("only_qv=True does not support MLA ASM scheduler metadata")
        if q is None:
            q = torch.empty((*qv.shape[:-1], 64), dtype=qv.dtype, device=qv.device)
        if k_cache is None:
            k_cache = torch.empty(
                (*v_cache.shape[:-1], 64),
                dtype=v_cache.dtype,
                device=v_cache.device,
            )
    else:
        if q is None:
            raise ValueError("q can only be None when only_qv=True")
        if k_cache is None:
            raise ValueError("k_cache can only be None when only_qv=True")
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if window_size is None:
        window_size = (-1, -1)
    if attention_chunk is None:
        attention_chunk = 0
    if softmax_scale is None:
        softmax_scale = (
            qv.shape[-1]
            if only_qv
            else q.shape[-1] + (qv.shape[-1] if qv is not None else 0)
        ) ** -0.5
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)

    is_mla_decode = qv is not None and qv.shape[-1] == 512 and q.shape[-1] == 64

    if is_mla_decode and isinstance(scheduler_metadata, tuple):
        mla_qv = qv
        mla_q = q
        use_flash_mla_asm = mla_qv.shape[-2] == 128
        require_seq_dense = not use_flash_mla_asm
        mla_qv = _prepare_mla_query_input(mla_qv, require_seq_dense=require_seq_dense)
        mla_q = _prepare_mla_query_input(mla_q, require_seq_dense=require_seq_dense)
        out, softmax_lse = _allocate_mla_decode_outputs(mla_q, mla_qv.shape[-1])
        tile_scheduler_metadata, mla_num_splits = (
            _prepare_mla_scheduler_metadata_from_workspace(
                scheduler_metadata[0],
                scheduler_metadata[1],
                mla_qv,
                v_cache,
                cache_seqlens,
                cu_seqlens_q,
                max_seqlen_q,
            )
        )
        if use_flash_mla_asm:
            flash_mla_asm_mubin(
                mla_qv,
                mla_q,
                v_cache,
                k_cache,
                cache_seqlens,
                page_table,
                tile_scheduler_metadata,
                mla_num_splits,
                out,
                softmax_lse,
                softmax_scale,
                causal,
                cu_seqlens_q,
                max_seqlen_q,
            )
        else:
            _get_mla_ops().get_function("mla_with_kvcache")(
                mla_qv,
                mla_q,
                v_cache,
                k_cache,
                cache_seqlens,
                cu_seqlens_q,
                max_seqlen_q,
                page_table,
                tile_scheduler_metadata,
                mla_num_splits,
                out,
                softmax_lse,
                softmax_scale,
                causal,
            )
        rest = []
    else:
        out, softmax_lse, *rest = jit_fmha_fwd(
            q=q,
            k=k_cache,
            v=v_cache,
            k_new=k,
            v_new=v,
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
            return_lse=return_softmax_lse or return_max_logits,
            lse=None,
            out=None,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
            cp_tot_seqused_k=cp_tot_seqused_k,
            only_qv=only_qv,
            return_max_logits=return_max_logits,
        )

    if return_max_logits:
        max_logits = _reduce_max_logits(rest[-1], cu_seqlens_q=cu_seqlens_q)
        return (
            (out, softmax_lse, max_logits) if return_softmax_lse else (out, max_logits)
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
    has_qv=False,
    mp_margin=0,  # Can be tuned if some MPs are used for communication):
):
    r"""
    Build scheduler metadata for ``flash_attn_with_kvcache``.

    Use this helper to precompute the tensor passed through the
    ``scheduler_metadata`` argument of ``flash_attn_with_kvcache``. This is the
    direct top-level API exposed as ``mate.get_scheduler_metadata``.

    Parameters
    ----------
    batch_size : int
        Batch size for the scheduled attention workload.
    max_seqlen_q : int
        Maximum query sequence length used by the target
        ``flash_attn_with_kvcache`` call.
    max_seqlen_k : int
        Maximum key sequence length already present in the cache.
    num_heads_q : int
        Number of query heads.
    num_heads_kv : int
        Number of key / value heads.
    headdim : int
        Query and key head dimension.
    seqused_q : Optional[Tensor]
        Optional per-batch query lengths with shape ``(batch_size,)``.
    seqused_k : Optional[Tensor]
        Optional per-batch key lengths with shape ``(batch_size,)``.
    qkv_dtype : torch.dtype
        Data type of the scheduled QKV path. Default ``torch.bfloat16``.
    headdim_v : Optional[int]
        Value head dimension. Defaults to ``headdim``.
    cu_seqlens_q : Optional[Tensor]
        Optional cumulative query sequence lengths with shape
        ``(batch_size + 1,)``.
    cu_seqlens_k : Optional[Tensor]
        Optional cumulative cached key sequence lengths with shape
        ``(batch_size + 1,)``.
    cu_seqlens_k_new : Optional[Tensor]
        Optional cumulative new-KV sequence lengths with shape
        ``(batch_size + 1,)``.
    cache_leftpad : Optional[Tensor]
        Optional per-batch left padding offsets for the KV cache.
    page_size : Optional[int]
        Page size for paged KV-cache scheduling.
    max_seqlen_k_new : int
        Maximum number of newly appended KV tokens.
    causal : bool
        Whether the target attention call uses causal masking.
    window_size : Tuple[int, int]
        Sliding-window attention bounds. ``(-1, -1)`` means full context.
    attention_chunk : int
        Chunk size used by chunked attention scheduling.
    has_softcap : bool
        Whether the target attention call enables softcapping.
    num_splits : int
        Requested split count for key / value scheduling.
    pack_gqa : Optional[bool]
        Optional GQA packing mode.
    has_qv : bool
        Whether the target attention call includes the optional ``qv`` input.
    mp_margin : int
        Number of MPs reserved for communication or other work.

    Returns
    -------
    Tensor
        Scheduler metadata tensor to pass to
        ``flash_attn_with_kvcache(..., scheduler_metadata=...)``.

    Notes
    -----
    Keep the scheduling-related arguments aligned with the corresponding
    ``flash_attn_with_kvcache`` call so the generated metadata matches the
    actual workload.
    """
    if window_size is None:
        window_size = (-1, -1)
    seqused_q = maybe_contiguous(seqused_q)
    seqused_k = maybe_contiguous(seqused_k)
    if headdim_v is None:
        headdim_v = headdim
    scheduler_metadata = jit_fmha_get_metadata(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        max_seqlen_k_new=max_seqlen_k_new,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        headdim=headdim,
        headdim_v=headdim_v,
        qkv_dtype=qkv_dtype,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        is_causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        attention_chunk=attention_chunk,
        leftpad_k=cache_leftpad,
        num_splits=num_splits,
        packgqa=pack_gqa,
        has_qv=has_qv,
        mp_margin=mp_margin,
        cu_seqlens_k_new=cu_seqlens_k_new,
    )
    return scheduler_metadata
