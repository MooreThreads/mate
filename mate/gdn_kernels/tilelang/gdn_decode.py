"""TileLang backend for the active GDN unified decode path.

This module intentionally stays backend-oriented: it builds and runs the active
VK-layout, FP32-state decode kernel and assumes API-level validation is handled
by ``mate.gdn_decode``.
"""

import os

import tilelang
import tilelang.language as T
import torch

__all__ = ["run_gated_delta_rule_decode_vk_fp32"]

_LOG2E = 1.4426950408889634
_DEFAULT_V_TILE = 16
_SOFTPLUS_BETA = 1.0
_SOFTPLUS_THRESHOLD = 20.0


def _parse_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.")
    return value


def _resolve_kernel_config_from_env() -> dict | None:
    num_stages = _parse_positive_int_env("GDN_TL_NUM_STAGES")
    threads = _parse_positive_int_env("GDN_TL_THREADS")
    v_tile = _parse_positive_int_env("GDN_TL_V_TILE")

    config = {}
    if num_stages is not None:
        config["num_stages"] = num_stages
    if threads is not None:
        config["threads"] = threads
    if v_tile is not None:
        config["v_tile"] = v_tile
    return config or None


def _build_decode_fp32_vk_kernel_factory(
    batch: int,
    qk_head: int,
    head: int,
    dim_k: int,
    dim_v: int,
    input_dtype: str,
    gate_batch_dtype: str,
    scale: float,
    use_qk_l2norm: bool,
    v_tile: int,
):
    if qk_head <= 0:
        raise ValueError("qk_head must be positive.")
    if head % qk_head != 0:
        raise ValueError(
            f"state/value heads={head} must be divisible by q/k heads={qk_head}."
        )
    if dim_v % v_tile != 0:
        raise ValueError(f"dim_v={dim_v} must be divisible by v_tile={v_tile}")
    if input_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"Unsupported input_dtype={input_dtype}")
    if gate_batch_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"Unsupported gate_batch_dtype={gate_batch_dtype}")

    head_group_size = head // qk_head

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
        compile_flags=["-O3"],
    )
    def _decode_func(num_stages, threads=128):
        @T.prim_func
        def gated_deltanet_decode_fp32_vk_qknorm(
            q: T.Tensor([batch, qk_head, dim_k], input_dtype),
            k: T.Tensor([batch, qk_head, dim_k], input_dtype),
            v: T.Tensor([batch, head, dim_v], input_dtype),
            A_log: T.Tensor([head], "float32"),
            a: T.Tensor([batch, head], gate_batch_dtype),
            dt_bias: T.Tensor([head], "float32"),
            b: T.Tensor([batch, head], gate_batch_dtype),
            state: T.Tensor([batch, head, dim_v, dim_k], "float32"),
            o: T.Tensor([batch, head, dim_v], input_dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([v_tile, dim_k], "float32")
                sk_frag = T.alloc_fragment([dim_v], "float32")
                sq_frag = T.alloc_fragment([dim_v], "float32")
                v_new = T.alloc_shared([dim_v], "float32")
                qk_dot = T.alloc_local([1], "float32")
                sum_q = T.alloc_local([1], "float32")
                sum_k = T.alloc_local([1], "float32")
                inv_norm_q = T.alloc_local([1], "float32")
                inv_norm_k = T.alloc_local([1], "float32")
                qk_hid = hid // head_group_size

                A_log_val = T.cast(A_log[hid], "float32")
                a_val = T.cast(a[bid, hid], "float32")
                dt_bias_val = T.cast(dt_bias[hid], "float32")
                b_val = T.cast(b[bid, hid], "float32")

                x = a_val + dt_bias_val
                beta_x = _SOFTPLUS_BETA * x
                softplus_x = T.if_then_else(
                    beta_x <= _SOFTPLUS_THRESHOLD,
                    (1.0 / _SOFTPLUS_BETA) * T.log(1.0 + T.exp(beta_x)),
                    x,
                )
                g_val = -T.exp(A_log_val) * softplus_x
                beta_val = 1.0 / (1.0 + T.exp(-b_val))
                alpha = T.exp2(g_val * _LOG2E)
                alpha_beta = alpha * beta_val

                sum_q[0] = 0.0
                sum_k[0] = 0.0
                if use_qk_l2norm:
                    for kk in T.serial(dim_k):
                        q_raw = T.cast(q[bid, qk_hid, kk], "float32")
                        k_raw = T.cast(k[bid, qk_hid, kk], "float32")
                        sum_q[0] += q_raw * q_raw
                        sum_k[0] += k_raw * k_raw
                    inv_norm_q[0] = T.rsqrt(sum_q[0] + 1e-6)
                    inv_norm_k[0] = T.rsqrt(sum_k[0] + 1e-6)
                else:
                    inv_norm_q[0] = 1.0
                    inv_norm_k[0] = 1.0

                T.fill(sk_frag, 0.0)
                T.fill(sq_frag, 0.0)

                for kk in T.serial(dim_k):
                    k_val = T.cast(k[bid, qk_hid, kk], "float32") * inv_norm_k[0]
                    q_val = (
                        T.cast(q[bid, qk_hid, kk], "float32") * inv_norm_q[0] * scale
                    )
                    for j in T.Parallel(dim_v):
                        h_val = state[bid, hid, j, kk]
                        sk_frag[j] = sk_frag[j] + k_val * h_val
                        sq_frag[j] = sq_frag[j] + q_val * h_val

                qk_dot[0] = 0.0
                for kk in T.serial(dim_k):
                    q_val = (
                        T.cast(q[bid, qk_hid, kk], "float32") * inv_norm_q[0] * scale
                    )
                    k_val = T.cast(k[bid, qk_hid, kk], "float32") * inv_norm_k[0]
                    qk_dot[0] += q_val * k_val

                for j in T.Parallel(dim_v):
                    v_new[j] = (
                        beta_val * T.cast(v[bid, hid, j], "float32")
                        - alpha_beta * sk_frag[j]
                    )

                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = T.cast(
                        alpha * sq_frag[j] + qk_dot[0] * v_new[j], input_dtype
                    )

                for vt in T.Pipelined(dim_v // v_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, vt * v_tile, 0], h_tile)
                    for jj, kk in T.Parallel(v_tile, dim_k):
                        state[bid, hid, vt * v_tile + jj, kk] = alpha * h_tile[
                            jj, kk
                        ] + v_new[vt * v_tile + jj] * (
                            T.cast(k[bid, qk_hid, kk], "float32") * inv_norm_k[0]
                        )

        return gated_deltanet_decode_fp32_vk_qknorm

    return _decode_func


def _get_decode_fp32_vk_kernel(
    batch: int,
    qk_head: int,
    head: int,
    dim_k: int,
    dim_v: int,
    input_dtype: str,
    gate_batch_dtype: str,
    scale: float,
    use_qk_l2norm: bool,
    num_stages: int,
    threads: int,
    v_tile: int,
):
    return _build_decode_fp32_vk_kernel_factory(
        batch=batch,
        qk_head=qk_head,
        head=head,
        dim_k=dim_k,
        dim_v=dim_v,
        input_dtype=input_dtype,
        gate_batch_dtype=gate_batch_dtype,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        v_tile=v_tile,
    )(num_stages, threads)


def run_gated_delta_rule_decode_vk_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    *,
    scale: float,
    use_qk_l2norm: bool,
):
    """Run the active VK-layout FP32-state decode backend.

    Inputs are expected to be validated by ``mate.gdn_decode.gated_delta_rule_decode``.
    Returns FP32 output with shape ``[B, 1, HV, V]`` while updating ``state`` in-place.
    """
    B, _, Hq, K = q.shape
    _, _, HV, V = v.shape
    kernel_config = _resolve_kernel_config_from_env() or {}

    kernel_fn = _get_decode_fp32_vk_kernel(
        batch=B,
        qk_head=Hq,
        head=HV,
        dim_k=K,
        dim_v=V,
        input_dtype=str(q.dtype).split(".")[-1],
        gate_batch_dtype=str(a.dtype).split(".")[-1],
        scale=float(scale),
        use_qk_l2norm=bool(use_qk_l2norm),
        num_stages=kernel_config.get("num_stages", 3),
        threads=kernel_config.get("threads", 128),
        v_tile=kernel_config.get("v_tile", _DEFAULT_V_TILE),
    )

    kernel_fn(
        q.squeeze(1),
        k.squeeze(1),
        v.squeeze(1),
        A_log,
        a.squeeze(1),
        dt_bias,
        b.squeeze(1),
        state,
        output,
    )
