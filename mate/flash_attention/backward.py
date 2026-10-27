"""FlashAttention backward dispatch between MATE-owned ASM and TileLang."""

import torch

from mate.execution_context import skip_kernel_launch_if_dry_run


def mubin_backward_unsupported_reason(
    q,
    k,
    v,
    out,
    dout,
    softmax_lse,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    is_causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    smscale=None,
    dtype=None,
    is_bhsd=False,
    deterministic=False,
):
    if q.device.type != "musa":
        return "requires MUSA tensors"
    props = torch.musa.get_device_properties(q.device)
    if (props.major, props.minor) != (3, 1):
        return "requires MP31"
    if is_bhsd:
        return "requires BSHD or packed THD layout"
    if deterministic:
        return "does not support deterministic accumulation"
    if seqused_q is not None or seqused_k is not None:
        return "does not support seqused"
    if softcap != 0 or any(x is not None and x >= 0 for x in (window_size or (-1, -1))):
        return "does not support softcap or local windows"
    if dtype is not None:
        return "does not support a dtype override"
    if q.ndim not in (3, 4) or any(t.ndim != q.ndim for t in (k, v, out, dout)):
        return "requires matching 3D or 4D input ranks"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return "requires FP16 or BF16"
    # Delta runs before the ASM launcher and consumes out/dout. K/V dtype,
    # device and full shape contracts are checked by the C++ launcher.
    if any(t.dtype != q.dtype or t.device != q.device for t in (out, dout)):
        return "requires output and output-gradient matching input dtype and device"
    if any(not t.is_contiguous() for t in (q, k, v, out)):
        return "requires contiguous Q/K/V and output"
    dim, vdim = q.shape[-1], v.shape[-1]
    if not (
        8 <= dim <= 128 and dim % 8 == 0 and vdim == dim or dim == 192 and vdim == 128
    ):
        return "supports D=8..128 (multiples of 8) or QK192/V128"
    h, hkv = q.shape[-2], k.shape[-2]
    if hkv <= 0 or h <= 0 or h % hkv:
        return "requires integral grouped query heads"
    if out.shape != (*q.shape[:-1], vdim) or dout.shape != out.shape:
        return "invalid output or output-gradient shape"
    if q.ndim == 4:
        if (
            cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or k.shape[0] != q.shape[0]
        ):
            return "dense batches require matching batch sizes without cu_seqlens"
        lse_shape = (q.shape[0], h, q.shape[1])
    else:
        if cu_seqlens_q is None or cu_seqlens_k is None:
            return "packed inputs require both cu_seqlens"
        for cu in (cu_seqlens_q, cu_seqlens_k):
            if (
                cu.ndim != 1
                or cu.dtype != torch.int32
                or cu.device != q.device
                or not cu.is_contiguous()
            ):
                return "cu_seqlens must be contiguous MUSA int32 vectors"
        if cu_seqlens_q.numel() <= 1 or cu_seqlens_q.shape != cu_seqlens_k.shape:
            return "cu_seqlens require matching nonempty batches"
        if max_seqlen_q is None or max_seqlen_k is None:
            return "packed inputs require max_seqlen_q/k"
        lse_shape = (h, q.shape[0])
    if (
        softmax_lse.shape != lse_shape
        or softmax_lse.dtype != torch.float32
        or softmax_lse.device != q.device
    ):
        return "requires MATE FP32 LSE layout"
    if (
        q.numel() == 0
        or k.numel() == 0
        or (q.ndim == 3 and min(max_seqlen_q, max_seqlen_k) <= 0)
    ):
        return "empty inputs use TileLang"
    # The imported launcher has int32 element strides and byte offsets.
    if any(
        t.numel() * t.element_size() >= 2**31 for t in (q, k, v, out, dout, softmax_lse)
    ):
        return "input byte offsets exceed int32"
    batch = q.shape[0] if q.ndim == 4 else cu_seqlens_q.numel() - 1
    max_q = q.shape[1] if q.ndim == 4 else max_seqlen_q
    total_k = k.shape[0] * k.shape[1] if q.ndim == 4 else k.shape[0]
    block_dim = 128 if dim <= 128 else 192
    padded_q = (max_q + 63) // 64 * 64
    if batch * h * padded_q * block_dim * 4 >= 2**31 or (
        h != hkv and total_k * h * block_dim * 4 >= 2**31
    ):
        return "workspace byte offsets exceed int32"
    from mate.artifacts import ArtifactPath

    try:
        ArtifactPath.for_module("flash_attention_bwd")
    except ValueError:
        return "backward ASM artifacts have not been published and pinned"
    return None


def _persistent_backward_eligible(q, k, v, cu_q, cu_k, is_causal):
    dense = q.ndim == 4
    singleton = (
        q.ndim == 3
        and cu_q is not None
        and cu_k is not None
        and cu_q.numel() == cu_k.numel() == 2
    )
    return (
        (dense or singleton)
        and q.numel() > 0
        and k.numel() > 0
        and v.shape[-1] == 128
        and q.shape[-1] in (128, 192)
    )


def _mubin_persistent_backward(q, k, v, out, dout, softmax_lse, is_causal, smscale):
    from mate.jit.mubin.flash_attention_bwd import (
        get_flash_attention_bwd_mubin_launch_plan,
    )
    from .tilelang._flash_attention_bwd_post import (
        compute_delta_ws,
        pack_dq_from_accum_ws,
        reduce_kv_grads_ws,
        to_tilelang_dtype,
    )

    original_shapes = (q.shape, k.shape, v.shape)
    if q.ndim == 3:
        q, k, v, out, dout = (t.unsqueeze(0) for t in (q, k, v, out, dout))
        softmax_lse = softmax_lse.unsqueeze(0)
    batch, sq, heads, dim = q.shape
    sk, hkv = k.shape[1:3]
    padded = (sq + 63) // 64 * 64
    gqa = heads != hkv
    dtype = to_tilelang_dtype(q.dtype)
    dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
    dq_accum = torch.empty(
        (batch, heads, padded, dim), device=q.device, dtype=torch.float32
    )
    if gqa:
        dk_accum = torch.empty(
            (batch, sk, heads, dim), device=q.device, dtype=torch.float32
        )
        dv_accum = torch.empty_like(dk_accum)
    else:
        dk_accum, dv_accum = dk, dv
    # Dense kernels speculatively read complete Q tiles. Pad per batch/head.
    lse = torch.full(
        (batch, heads, padded), float("inf"), device=q.device, dtype=torch.float32
    )
    delta = torch.zeros_like(lse)
    delta_unpadded = torch.empty(
        (batch, heads, sq), device=q.device, dtype=torch.float32
    )
    unused_cu = torch.empty(batch + 1, device=q.device, dtype=torch.int32)
    unused_seq = torch.empty(batch, device=q.device, dtype=torch.int32)
    compute = compute_delta_ws(128, is_varlen=False, dtype=dtype, head_major=True)
    pack = pack_dq_from_accum_ws(dim, is_varlen=False, dtype=dtype)
    reduce = (
        reduce_kv_grads_ws(heads // hkv, dim, is_varlen=False, output_dtype=dtype)
        if gqa
        else None
    )
    path, launch = get_flash_attention_bwd_mubin_launch_plan(
        q.dtype, is_causal, dim, gqa, is_persistence=True
    )
    if not skip_kernel_launch_if_dry_run():
        lse[..., :sq].copy_(softmax_lse)
        do = dout.contiguous()
        compute(out, do, unused_cu, unused_seq, delta_unpadded, sq)
        delta[..., :sq].copy_(delta_unpadded)
        dq_accum.zero_()
        dk_accum.zero_()
        dv_accum.zero_()
        launch(
            path,
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq_accum,
            dk_accum,
            dv_accum,
            dim**-0.5 if smscale is None else smscale,
        )
        pack(dq_accum, unused_cu, unused_seq, dq)
        if reduce is not None:
            reduce(dk_accum, dv_accum, dk, dv)
    return tuple(t.view(shape) for t, shape in zip((dq, dk, dv), original_shapes))


def _mubin_backward(
    q,
    k,
    v,
    out,
    dout,
    softmax_lse,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    is_causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    smscale=None,
    dtype=None,
    is_bhsd=False,
    deterministic=False,
):
    from mate.jit.mubin.flash_attention_bwd import (
        get_flash_attention_bwd_mubin_launch_plan,
    )
    from .tilelang._flash_attention_bwd_post import (
        compute_delta_ws,
        pack_dq_from_accum_ws,
        reduce_kv_grads_ws,
        to_tilelang_dtype,
    )

    if _persistent_backward_eligible(q, k, v, cu_seqlens_q, cu_seqlens_k, is_causal):
        if q.ndim == 3 and not skip_kernel_launch_if_dry_run():
            # Singleton packed adaptation requires the complete tensor segment.
            # Validate on the device, without .item() or a host synchronization.
            for cu, tensor in ((cu_seqlens_q, q), (cu_seqlens_k, k)):
                torch._assert_async((cu[0] == 0) & (cu[1] == tensor.shape[0]))
        return _mubin_persistent_backward(
            q, k, v, out, dout, softmax_lse, is_causal, smscale
        )

    is_dense = q.ndim == 4
    if is_dense:
        batch, max_seqlen_q, heads, _ = q.shape
        max_seqlen_k = k.shape[1]
        cu_seqlens_q = (
            torch.arange(batch + 1, device=q.device, dtype=torch.int32) * max_seqlen_q
        )
        cu_seqlens_k = (
            torch.arange(batch + 1, device=q.device, dtype=torch.int32) * max_seqlen_k
        )
        lse = softmax_lse.permute(1, 0, 2).contiguous().view(heads, -1)
        q_flat, k_flat, v_flat, out_flat = (t.flatten(0, 1) for t in (q, k, v, out))
    else:
        batch, heads = cu_seqlens_q.numel() - 1, q.shape[1]
        lse = softmax_lse.contiguous()
        q_flat, k_flat, v_flat, out_flat = q, k, v, out
    dout_flat = dout.contiguous().flatten(0, 1) if is_dense else dout.contiguous()
    dim, vdim, hkv = q.shape[-1], v.shape[-1], k.shape[-2]
    block_dim = 128 if dim <= 128 else 192
    padded_q = (max_seqlen_q + 63) // 64 * 64
    is_gqa = heads != hkv
    dq = torch.empty_like(q_flat)
    dk, dv = torch.empty_like(k_flat), torch.empty_like(v_flat)
    dq_accum = torch.empty(
        (batch, heads, padded_q, block_dim), device=q.device, dtype=torch.float32
    )
    if is_gqa:
        dk_accum = torch.empty(
            (k_flat.shape[0], heads, block_dim), device=q.device, dtype=torch.float32
        )
        dv_accum = torch.empty_like(dk_accum)
    else:
        dk_accum, dv_accum = dk, dv
    delta = torch.empty_like(lse)
    unused_seqused = torch.empty(batch, device=q.device, dtype=torch.int32)
    kernel_dtype = to_tilelang_dtype(q.dtype)
    compute_delta = compute_delta_ws(
        128, is_varlen=True, dtype=kernel_dtype, head_major=True
    )
    pack_dq = pack_dq_from_accum_ws(block_dim, is_varlen=True, dtype=kernel_dtype)
    reduce_kv = (
        reduce_kv_grads_ws(
            heads // hkv, block_dim, is_varlen=True, output_dtype=kernel_dtype
        )
        if is_gqa
        else None
    )
    path, launch = get_flash_attention_bwd_mubin_launch_plan(
        q.dtype, is_causal, dim, is_gqa
    )
    if skip_kernel_launch_if_dry_run():
        return dq.view(q.shape), dk.view(k.shape), dv.view(v.shape)
    compute_delta(
        out_flat, dout_flat, cu_seqlens_q, unused_seqused, delta, max_seqlen_q
    )
    dq_accum.zero_()
    dk_accum.zero_()
    dv_accum.zero_()
    launch(
        path,
        q_flat,
        k_flat,
        v_flat,
        dout_flat,
        lse,
        delta,
        dq_accum,
        dk_accum,
        dv_accum,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dim**-0.5 if smscale is None else smscale,
    )
    pack_dq(dq_accum, cu_seqlens_q, unused_seqused, dq)
    if reduce_kv is not None:
        reduce_kv(dk_accum, dv_accum, dk, dv)
    return dq.view(q.shape), dk.view(k.shape), dv.view(v.shape)


def flashattn_varlen_bwd_interface(
    q,
    k,
    v,
    out,
    dout,
    softmax_lse,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    is_causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    smscale=None,
    dtype=None,
    is_bhsd=False,
    deterministic=False,
    backend="auto",
):
    """Prefer ASM for supported MP31 inputs; explicitly select either backend."""
    if backend not in ("auto", "mubin", "tilelang"):
        raise ValueError(f"Unknown FlashAttention backward backend: {backend}")
    args = (q, k, v, out, dout, softmax_lse, max_seqlen_q, max_seqlen_k)
    kwargs = dict(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        smscale=smscale,
        dtype=dtype,
        is_bhsd=is_bhsd,
        deterministic=deterministic,
    )
    if backend != "tilelang":
        reason = mubin_backward_unsupported_reason(*args, **kwargs)
        if reason is None:
            return _mubin_backward(*args, **kwargs)
        if backend == "mubin":
            raise ValueError(f"FlashAttention backward MUBIN: {reason}")
    from .tilelang.flash_attention_varlen_bwd import (
        flashattn_varlen_bwd_interface as tilelang_backward,
    )

    return tilelang_backward(*args, **kwargs)
