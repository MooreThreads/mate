"""FMHA operator definition shared by generated and direct-input workflows."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Sequence

import torch
from einops import rearrange

from mate.execution_context import is_fake_mode
from mate.mha_interface import (
    flash_attn_combine,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from mate.testing.flash_attn import (
    _combine_cp_partials,
    apply_rotary_emb,
    attention_ref,
    generate_block_kvcache,
    lse_ref_from_score,
    unpad_input,
)

from .operator import (
    Operator,
    OperatorInputs,
    OperatorOutputs,
    OperatorReference,
    OperatorWorkload,
)

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


class FmhaApi(Enum):
    VARLEN = "varlen"
    KVCACHE = "kvcache"
    COMBINE = "combine"


class FmhaLayout(Enum):
    NORMAL = "normal"
    PADDED = "padded"
    RAGGED = "ragged"
    PAGED = "paged"


class FmhaVerifyMode(Enum):
    THREASHOLD = "threashold"
    DIFF = "diff"


FmhaTensorInitializer = Callable[[torch.Tensor], torch.Tensor]


def default_tensor_init(tensor: torch.Tensor) -> torch.Tensor:
    """Use the generated tensor without changing its storage layout."""
    return tensor


def noncontiguous_tensor_init(tensor: torch.Tensor) -> torch.Tensor:
    """Copy values into a view with a non-contiguous leading dimension."""
    for dim, size in enumerate(tensor.shape[:-1]):
        if size <= 1:
            continue
        storage_shape = list(tensor.shape)
        storage_shape[dim] *= 2
        storage = torch.empty(
            storage_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(None, None, 2)
        view = storage[tuple(slices)]
        view.copy_(tensor)
        return view
    raise ValueError(
        "noncontiguous tensor initialization requires a non-singleton "
        "dimension before the head dimension"
    )


def large_stride_tensor_init(tensor: torch.Tensor) -> torch.Tensor:
    """Give a singleton leading dimension a stride larger than int32."""
    if tensor.dim() == 0 or tensor.shape[0] != 1:
        raise ValueError("large-stride tensor initialization requires shape[0] == 1")
    return tensor.as_strided(
        tensor.shape,
        (2**31 + 1024, *tensor.stride()[1:]),
    )


@dataclass(frozen=True, kw_only=True)
class FmhaTensorSpec:
    """Describe one tensor's logical layout and value-preserving initializer."""

    layout: FmhaLayout = FmhaLayout.NORMAL
    init: FmhaTensorInitializer = default_tensor_init


class UnsupportedFmhaWorkload(ValueError):
    """Raised when an FMHA workload describes an unsupported feature combination."""


@dataclass(frozen=True, kw_only=True)
class FmhaWorkload(OperatorWorkload):
    api: FmhaApi
    batch_size: int
    seqlen_q: int
    num_heads_q: int
    head_dim_v: int

    seqlen_k: int | None = None
    num_heads_kv: int | None = None
    head_dim_qk: int | None = None
    dtype: torch.dtype = torch.bfloat16
    reference_dtype: torch.dtype = torch.bfloat16
    device: str | torch.device = "musa"
    seed: int | None = 666
    label: str | None = None

    verify_mode: FmhaVerifyMode = FmhaVerifyMode.THREASHOLD
    q: FmhaTensorSpec | None = None
    k: FmhaTensorSpec | None = None
    v: FmhaTensorSpec | None = None
    qv: FmhaTensorSpec | None = None
    k_cache: FmhaTensorSpec | None = None
    v_cache: FmhaTensorSpec | None = None
    combine_layout: FmhaLayout = FmhaLayout.NORMAL
    q_lengths: Sequence[int] | None = None
    kv_lengths: Sequence[int] | None = None
    q_used_lengths: Sequence[int] | None = None
    kv_used_lengths: Sequence[int] | None = None
    page_size: int | None = None

    causal: bool = False
    window_size: tuple[int | None, int | None] = (None, None)
    attention_chunk: int = 0
    softcap: float = 0.0
    softmax_scale: float | None = None
    num_splits: int = 0
    pack_gqa: bool | None = None
    backend: str = "mutlass"
    return_softmax_lse: bool = True
    compare_lse: bool = True
    atol: float = 1.5e-2
    rtol: float = 1e-2

    cp_world_size: int = 1
    learnable_sink: bool = False
    expect_single_split: bool = False

    only_qv: bool = False
    seqlen_new_eq_seqlen_q: bool = False
    has_cache_batch_idx: bool = False
    has_cache_leftpad: bool = False
    rotary_fraction: float = 0.0
    rotary_interleaved: bool = True
    has_rotary_seqlens: bool = False

    randomize_splits: bool = False
    reuse_output: bool = False

    def __post_init__(self) -> None:
        for name in ("batch_size", "seqlen_q", "num_heads_q", "head_dim_v"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.backend not in ("auto", "mutlass", "mubin"):
            raise ValueError("backend must be 'auto', 'mutlass', or 'mubin'")
        if self.cp_world_size <= 0:
            raise ValueError("cp_world_size must be positive")
        if self.only_qv and self.api is not FmhaApi.KVCACHE:
            raise ValueError("only_qv is only supported by the kvcache API")
        self._validate_lengths("q_lengths", self.q_lengths, self.seqlen_q)
        self._validate_used_lengths("q_used_lengths", self.q_used_lengths)

        if self.api is FmhaApi.COMBINE:
            if self.num_splits <= 0:
                raise ValueError("combine workloads require num_splits > 0")
            if self.combine_layout is FmhaLayout.PAGED:
                raise ValueError("combine workloads cannot use a paged layout")
            return

        if self.q is None:
            raise ValueError("attention workloads require a q tensor spec")
        if self.seqlen_k is None or self.seqlen_k <= 0:
            raise ValueError("seqlen_k must be positive for attention workloads")
        if self.num_heads_kv is None or self.num_heads_kv <= 0:
            raise ValueError("num_heads_kv must be positive for attention workloads")
        if self.head_dim_qk is None or self.head_dim_qk <= 0:
            raise ValueError("head_dim_qk must be positive for attention workloads")
        if self.num_heads_q % self.num_heads_kv != 0:
            raise ValueError("num_heads_q must be divisible by num_heads_kv")
        self._validate_lengths("kv_lengths", self.kv_lengths, self.seqlen_k)
        self._validate_used_lengths("kv_used_lengths", self.kv_used_lengths)

        for name in ("q", "k", "v", "qv"):
            spec = getattr(self, name)
            if spec is not None and spec.layout is FmhaLayout.PAGED:
                raise ValueError(f"{name} cannot use a paged layout")
        if (self.k is None) != (self.v is None):
            raise ValueError("k and v tensor specs must be provided together")
        if (self.k_cache is None) != (self.v_cache is None):
            raise ValueError(
                "k_cache and v_cache tensor specs must be provided together"
            )
        if self.qv is not None and self.qv.layout is not self.q.layout:
            raise ValueError("q and qv must use the same layout")
        if self.k is not None and self.k.layout is not self.v.layout:
            raise ValueError("k and v must use the same layout")
        if self.k_cache is not None and self.k_cache.layout is not self.v_cache.layout:
            raise ValueError("k_cache and v_cache must use the same layout")
        for name in ("k_cache", "v_cache"):
            spec = getattr(self, name)
            if spec is not None and spec.layout not in (
                FmhaLayout.NORMAL,
                FmhaLayout.PAGED,
            ):
                raise ValueError(f"{name} layout must be normal or paged")

        if self.api is FmhaApi.VARLEN:
            if self.k is None:
                raise ValueError("varlen workloads require k and v tensor specs")
            if self.k_cache is not None:
                raise ValueError("varlen workloads cannot use cache tensor specs")
        elif self.api is FmhaApi.KVCACHE:
            if self.k_cache is None:
                raise ValueError("kvcache workloads require cache tensor specs")

        cache_layout = (
            self.k_cache.layout if self.k_cache is not None else FmhaLayout.NORMAL
        )
        if cache_layout is FmhaLayout.PAGED:
            if self.page_size is None or self.page_size <= 0:
                raise ValueError("paged KV workloads require a positive page_size")
        elif self.page_size is not None:
            raise ValueError("page_size is only valid for paged KV workloads")
        if self.only_qv and self.qv is None:
            raise ValueError("only_qv requires a qv tensor spec")
        if self.seqlen_new_eq_seqlen_q and not self.appends_kv:
            raise ValueError("seqlen_new_eq_seqlen_q requires appended k and v")

    def _validate_lengths(
        self,
        name: str,
        lengths: Sequence[int] | None,
        maximum: int,
    ) -> None:
        if lengths is None:
            return
        if len(lengths) != self.batch_size:
            raise ValueError(f"{name} must contain one value per batch item")
        if any(length < 0 or length > maximum for length in lengths):
            raise ValueError(f"{name} values must be in [0, {maximum}]")

    def _validate_used_lengths(
        self,
        name: str,
        lengths: Sequence[int] | None,
    ) -> None:
        if lengths is not None and len(lengths) != self.batch_size:
            raise ValueError(f"{name} must contain one value per batch item")

    def __str__(self) -> str:
        if self.label is not None:
            return self.label
        parts = [
            self.api.value,
            f"b{self.batch_size}",
            f"sq{self.seqlen_q}",
        ]
        if self.api is FmhaApi.COMBINE:
            parts.append(f"combine-{self.combine_layout.value}")
        for name in ("q", "k", "v", "qv", "k_cache", "v_cache"):
            spec = getattr(self, name)
            if spec is None:
                continue
            tensor_part = f"{name}-{spec.layout.value}"
            if spec.init is not default_tensor_init:
                initializer_name = getattr(
                    spec.init, "__name__", type(spec.init).__name__
                )
                tensor_part += f"-{initializer_name.removesuffix('_tensor_init')}"
            parts.append(tensor_part)
        if self.seqlen_k is not None:
            parts.append(f"sk{self.seqlen_k}")
        heads = f"h{self.num_heads_q}"
        if self.num_heads_kv is not None:
            heads += f"x{self.num_heads_kv}"
        dimensions = f"d{self.head_dim_v}"
        if self.head_dim_qk is not None:
            dimensions = f"d{self.head_dim_qk}x{self.head_dim_v}"
        parts.extend((heads, dimensions))
        if self.page_size is not None:
            parts.append(f"page{self.page_size}")
        if self.causal:
            parts.append("causal")
        elif self.window_size not in ((None, None), (-1, -1)):
            parts.append(f"local{self.window_size[0]}x{self.window_size[1]}")
        if self.attention_chunk > 0:
            parts.append(f"chunk{self.attention_chunk}")
        if self.softcap > 0:
            parts.append(f"softcap{self.softcap:g}")
        parts.append(f"split{self.num_splits}")
        if self.pack_gqa:
            parts.append("pack")
        if self.cp_world_size > 1:
            parts.append(f"cp{self.cp_world_size}")
        if self.api is FmhaApi.VARLEN:
            parts.append(self.backend)
        return "-".join(parts)

    @property
    def appends_kv(self) -> bool:
        return self.k_cache is not None and self.k is not None


@dataclass(kw_only=True)
class FmhaInputs(OperatorInputs):
    api: FmhaApi
    workload: FmhaWorkload | None = None

    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    qv: torch.Tensor | None = None
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None
    k_cache_reference: torch.Tensor | None = None
    v_cache_reference: torch.Tensor | None = None
    out_partial: torch.Tensor | None = None
    lse_partial: torch.Tensor | None = None
    out: torch.Tensor | None = None

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    cu_seqlens_k_new: torch.Tensor | None = None
    seqused_q: torch.Tensor | None = None
    seqused_k: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | int | None = None
    cache_batch_idx: torch.Tensor | None = None
    cache_leftpad: torch.Tensor | None = None
    page_table: torch.Tensor | None = None
    scheduler_metadata: torch.Tensor | None = None
    prepare_scheduler_metadata: bool = False

    rotary_cos: torch.Tensor | None = None
    rotary_sin: torch.Tensor | None = None
    rotary_cos_reference: torch.Tensor | None = None
    rotary_sin_reference: torch.Tensor | None = None
    rotary_seqlens: torch.Tensor | None = None
    rotary_seqlens_reference: torch.Tensor | None = None
    q_descale: torch.Tensor | None = None
    k_descale: torch.Tensor | None = None
    v_descale: torch.Tensor | None = None
    learnable_sink: torch.Tensor | None = None
    cp_tot_seqused_k: torch.Tensor | None = None

    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    max_seqlen_k_new: int = 0
    softmax_scale: float | None = None
    causal: bool = False
    window_size: tuple[int | None, int | None] = (None, None)
    attention_chunk: int = 0
    softcap: float = 0.0
    num_splits: int = 0
    pack_gqa: bool | None = None
    deterministic: bool = False
    sm_margin: int = 0
    backend: str = "mutlass"
    rotary_interleaved: bool = True
    cp_world_size: int = 1
    cp_rank: int = 0
    only_qv: bool = False
    return_softmax_lse: bool = True
    out_dtype: torch.dtype | None = None

    q_padded: torch.Tensor | None = None
    k_padded: torch.Tensor | None = None
    v_padded: torch.Tensor | None = None
    qv_padded: torch.Tensor | None = None
    query_padding_mask: torch.Tensor | None = None
    key_padding_mask: torch.Tensor | None = None
    key_new_padding_mask: torch.Tensor | None = None
    q_indices: torch.Tensor | None = None
    k_indices: torch.Tensor | None = None
    q_lengths: tuple[int, ...] | None = None
    kv_lengths: tuple[int, ...] | None = None
    rotary_dim: int = 0


@dataclass(frozen=True, kw_only=True)
class FmhaOutputs(OperatorOutputs):
    out: torch.Tensor
    lse: torch.Tensor | None = None
    out_accum: torch.Tensor | None = None
    lse_accum: torch.Tensor | None = None
    auxiliary: tuple[Any, ...] = ()


@dataclass(frozen=True, kw_only=True)
class FmhaReference(OperatorReference):
    out: torch.Tensor
    lse: torch.Tensor | None = None
    high_precision_out: torch.Tensor | None = None
    expected_k_cache: torch.Tensor | None = None
    expected_v_cache: torch.Tensor | None = None


class FmhaOperator(Operator):
    """Generate, invoke, reference, and verify FMHA workloads."""

    def generate(self, workload):
        unsupported = self._unsupported_reason(workload)
        if unsupported is not None:
            raise UnsupportedFmhaWorkload(unsupported)
        self._seed(workload.seed, workload.device)
        if workload.api is FmhaApi.COMBINE:
            return self._generate_combine(workload)
        return self._generate_attention(workload)

    def call(self, inputs):
        if inputs.only_qv and inputs.api is not FmhaApi.KVCACHE:
            raise ValueError("only_qv is only supported by the kvcache API")
        if inputs.api is FmhaApi.COMBINE:
            return self._call_combine(inputs)
        if inputs.cp_world_size > 1:
            return self._call_cp(inputs)
        return self._call_attention_once(inputs)

    def reference(self, inputs):
        if inputs.api is FmhaApi.COMBINE:
            return self._reference_combine(inputs)
        return self._reference_attention(inputs)

    def verify(
        self,
        inputs,
        outputs,
        reference,
    ) -> None:
        workload = inputs.workload
        if workload is not None and workload.verify_mode is FmhaVerifyMode.DIFF:
            self._verify_diff(inputs, outputs, reference)
        else:
            self._verify_threashold(inputs, outputs, reference)
        self._verify_appended_cache(inputs, reference)

    def _call_attention_once(self, inputs: FmhaInputs) -> FmhaOutputs:
        metadata = inputs.scheduler_metadata
        if (
            metadata is None
            and inputs.prepare_scheduler_metadata
            and inputs.num_splits >= 0
        ):
            metadata = self._make_scheduler_metadata(inputs)
            inputs.scheduler_metadata = metadata

        if inputs.api is FmhaApi.VARLEN:
            if inputs.q is None or inputs.k is None or inputs.v is None:
                raise ValueError("varlen calls require q, k, and v")
            result = flash_attn_varlen_func(
                q=inputs.q,
                k=inputs.k,
                v=inputs.v,
                cu_seqlens_q=inputs.cu_seqlens_q,
                cu_seqlens_k=inputs.cu_seqlens_k,
                max_seqlen_q=inputs.max_seqlen_q,
                max_seqlen_k=inputs.max_seqlen_k,
                seqused_q=inputs.seqused_q,
                seqused_k=inputs.seqused_k,
                page_table=inputs.page_table,
                softmax_scale=inputs.softmax_scale,
                causal=inputs.causal,
                qv=inputs.qv,
                q_descale=inputs.q_descale,
                k_descale=inputs.k_descale,
                v_descale=inputs.v_descale,
                window_size=inputs.window_size,
                learnable_sink=inputs.learnable_sink,
                attention_chunk=inputs.attention_chunk,
                softcap=inputs.softcap,
                scheduler_metadata=metadata,
                num_splits=inputs.num_splits,
                pack_gqa=inputs.pack_gqa,
                deterministic=inputs.deterministic,
                sm_margin=inputs.sm_margin,
                return_softmax_lse=inputs.return_softmax_lse,
                backend=inputs.backend,
                cp_world_size=inputs.cp_world_size,
                cp_rank=inputs.cp_rank,
                cp_tot_seqused_k=inputs.cp_tot_seqused_k,
                out=inputs.out,
            )
        else:
            if inputs.q is None or inputs.k_cache is None or inputs.v_cache is None:
                raise ValueError("kvcache calls require q, k_cache, and v_cache")
            if inputs.only_qv and inputs.qv is None:
                raise ValueError("only_qv kvcache calls require qv")
            result = flash_attn_with_kvcache(
                q=inputs.q,
                k_cache=inputs.k_cache,
                v_cache=inputs.v_cache,
                k=inputs.k,
                v=inputs.v,
                qv=inputs.qv,
                rotary_cos=inputs.rotary_cos,
                rotary_sin=inputs.rotary_sin,
                cache_seqlens=inputs.cache_seqlens,
                cache_batch_idx=inputs.cache_batch_idx,
                cache_leftpad=inputs.cache_leftpad,
                page_table=inputs.page_table,
                cu_seqlens_q=inputs.cu_seqlens_q,
                cu_seqlens_k_new=inputs.cu_seqlens_k_new,
                max_seqlen_q=inputs.max_seqlen_q,
                rotary_seqlens=inputs.rotary_seqlens,
                q_descale=inputs.q_descale,
                k_descale=inputs.k_descale,
                v_descale=inputs.v_descale,
                softmax_scale=inputs.softmax_scale,
                causal=inputs.causal,
                window_size=inputs.window_size,
                learnable_sink=inputs.learnable_sink,
                attention_chunk=inputs.attention_chunk,
                softcap=inputs.softcap,
                rotary_interleaved=inputs.rotary_interleaved,
                scheduler_metadata=metadata,
                num_splits=inputs.num_splits,
                pack_gqa=inputs.pack_gqa,
                sm_margin=inputs.sm_margin,
                return_softmax_lse=inputs.return_softmax_lse,
                cp_world_size=inputs.cp_world_size,
                cp_rank=inputs.cp_rank,
                cp_tot_seqused_k=inputs.cp_tot_seqused_k,
                only_qv=inputs.only_qv,
            )
        return self._normalize_outputs(result, inputs.return_softmax_lse)

    def _call_cp(self, inputs: FmhaInputs) -> FmhaOutputs:
        if inputs.q is None or inputs.cu_seqlens_q is None:
            raise ValueError("CP workloads require ragged q and cu_seqlens_q")
        if inputs.kv_lengths is None:
            raise ValueError("CP workloads require concrete KV lengths")
        if inputs.api is FmhaApi.KVCACHE and (
            inputs.k is not None or inputs.v is not None
        ):
            raise ValueError("append-KV is not supported with context parallelism")
        k_init = (
            inputs.workload.k.init
            if inputs.workload is not None and inputs.workload.k is not None
            else default_tensor_init
        )
        v_init = (
            inputs.workload.v.init
            if inputs.workload is not None and inputs.workload.v is not None
            else default_tensor_init
        )
        k_cache_init = (
            inputs.workload.k_cache.init
            if inputs.workload is not None and inputs.workload.k_cache is not None
            else default_tensor_init
        )
        v_cache_init = (
            inputs.workload.v_cache.init
            if inputs.workload is not None and inputs.workload.v_cache is not None
            else default_tensor_init
        )

        cache_leftpads = (0,) * len(inputs.kv_lengths)
        if inputs.api is FmhaApi.VARLEN:
            if inputs.seqused_k is not None and not is_fake_mode():
                used_kv_lengths = tuple(
                    int(value) for value in inputs.seqused_k.cpu().tolist()
                )
            elif (
                inputs.workload is not None
                and inputs.workload.kv_used_lengths is not None
            ):
                used_kv_lengths = self._resolve_used_lengths(
                    inputs.workload.kv_used_lengths, inputs.kv_lengths
                )
            else:
                used_kv_lengths = inputs.kv_lengths
            physical_kv_lengths = inputs.kv_lengths
        else:
            if inputs.cache_leftpad is not None and not is_fake_mode():
                cache_leftpads = tuple(
                    int(value) for value in inputs.cache_leftpad.cpu().tolist()
                )
            used_kv_lengths = tuple(
                length - leftpad
                for length, leftpad in zip(inputs.kv_lengths, cache_leftpads)
            )
            physical_kv_lengths = used_kv_lengths
        cp_tot_seqused_k = torch.tensor(
            used_kv_lengths, dtype=torch.int32, device=inputs.q.device
        )
        rank_inputs_keepalive: list[FmhaInputs] = []
        rank_outputs: list[FmhaOutputs] = []
        for cp_rank in range(inputs.cp_world_size):
            local_lengths = tuple(
                max(
                    0,
                    (length - cp_rank + inputs.cp_world_size - 1)
                    // inputs.cp_world_size,
                )
                for length in physical_kv_lengths
            )
            local_used_lengths = tuple(
                max(
                    0,
                    (length - cp_rank + inputs.cp_world_size - 1)
                    // inputs.cp_world_size,
                )
                for length in used_kv_lengths
            )
            if inputs.api is FmhaApi.VARLEN:
                if inputs.k_padded is None or inputs.v_padded is None:
                    raise ValueError("varlen CP requires padded source K and V")
                k_segments = [
                    inputs.k_padded[index, cp_rank : length : inputs.cp_world_size]
                    for index, length in enumerate(physical_kv_lengths)
                ]
                v_segments = [
                    inputs.v_padded[index, cp_rank : length : inputs.cp_world_size]
                    for index, length in enumerate(physical_kv_lengths)
                ]
                cu_seqlens_k = torch.tensor(
                    self._cumulative_lengths(local_lengths),
                    dtype=torch.int32,
                    device=inputs.q.device,
                )
                seqused_k = (
                    torch.tensor(
                        local_used_lengths,
                        dtype=torch.int32,
                        device=inputs.q.device,
                    )
                    if inputs.seqused_k is not None
                    else None
                )
                rank_inputs = replace(
                    inputs,
                    k=k_init(torch.cat(k_segments, dim=0).to(inputs.q.dtype)),
                    v=v_init(torch.cat(v_segments, dim=0).to(inputs.q.dtype)),
                    cu_seqlens_k=cu_seqlens_k,
                    seqused_k=seqused_k,
                    max_seqlen_k=max(local_lengths),
                    scheduler_metadata=None,
                    cp_rank=cp_rank,
                    cp_tot_seqused_k=cp_tot_seqused_k,
                    kv_lengths=local_lengths,
                )
            else:
                if (
                    inputs.k_cache is None
                    or inputs.v_cache is None
                    or inputs.page_table is None
                    or inputs.workload is None
                    or inputs.workload.page_size is None
                ):
                    raise ValueError("paged CP requires paged K/V and a page table")
                k_local, v_local, page_table_local, local_seqlens = (
                    self._make_cp_paged_cache(
                        inputs,
                        cp_rank=cp_rank,
                        local_lengths=local_lengths,
                        cache_leftpads=cache_leftpads,
                    )
                )
                rank_inputs = replace(
                    inputs,
                    k_cache=k_cache_init(k_local),
                    v_cache=v_cache_init(v_local),
                    page_table=page_table_local,
                    cache_seqlens=local_seqlens,
                    cache_batch_idx=None,
                    cache_leftpad=None,
                    max_seqlen_k=max(local_lengths),
                    scheduler_metadata=None,
                    cp_rank=cp_rank,
                    cp_tot_seqused_k=cp_tot_seqused_k,
                )
            rank_inputs = replace(rank_inputs, return_softmax_lse=True)
            rank_inputs_keepalive.append(rank_inputs)
            rank_outputs.append(self._call_attention_once(rank_inputs))
            if not is_fake_mode() and hasattr(torch, "musa"):
                torch.musa.synchronize()

        if is_fake_mode():
            output = rank_outputs[0]
            return replace(
                output,
                lse=output.lse if inputs.return_softmax_lse else None,
            )
        total_q = inputs.q.shape[0]
        combined_out = torch.zeros(
            total_q,
            inputs.q.shape[-2],
            rank_outputs[0].out.shape[-1],
            dtype=torch.float32,
            device=inputs.q.device,
        )
        combined_lse = torch.full(
            (inputs.q.shape[-2], total_q),
            float("-inf"),
            dtype=torch.float32,
            device=inputs.q.device,
        )
        cu_q = [int(value) for value in inputs.cu_seqlens_q.cpu().tolist()]
        used_q = (
            [int(value) for value in inputs.seqused_q.cpu().tolist()]
            if inputs.seqused_q is not None
            else [end - start for start, end in zip(cu_q, cu_q[1:])]
        )
        for start, physical_end, used_length in zip(cu_q, cu_q[1:], used_q):
            end = min(start + used_length, physical_end)
            if start == end:
                continue
            rank_lses = []
            for output in rank_outputs:
                if output.lse is None:
                    raise AssertionError("CP rank did not return its required LSE")
                rank_lses.append(output.lse[:, start:end].float())
            out_item, lse_item = _combine_cp_partials(
                [output.out[start:end].float() for output in rank_outputs],
                rank_lses,
            )
            combined_out[start:end] = out_item
            combined_lse[:, start:end] = lse_item
        return FmhaOutputs(
            out=combined_out.to(rank_outputs[0].out.dtype),
            lse=combined_lse if inputs.return_softmax_lse else None,
        )

    def _make_cp_paged_cache(
        self,
        inputs: FmhaInputs,
        *,
        cp_rank: int,
        local_lengths: tuple[int, ...],
        cache_leftpads: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            inputs.k_cache is None
            or inputs.v_cache is None
            or inputs.page_table is None
            or inputs.workload is None
            or inputs.workload.page_size is None
        ):
            raise ValueError("paged CP requires paged K/V and a page table")
        page_size = inputs.workload.page_size
        batch_size = len(local_lengths)
        max_local_length = max(local_lengths)
        pages_per_sequence = max(1, math.ceil(max_local_length / page_size))
        padded_length = pages_per_sequence * page_size
        k_dense = self._logical_cache(inputs.k_cache, inputs.page_table)
        v_dense = self._logical_cache(inputs.v_cache, inputs.page_table)
        if inputs.cache_batch_idx is not None:
            indices = inputs.cache_batch_idx.long()
            k_dense = k_dense[indices]
            v_dense = v_dense[indices]
        k_local_dense = torch.zeros(
            batch_size,
            padded_length,
            *k_dense.shape[2:],
            dtype=k_dense.dtype,
            device=k_dense.device,
        )
        v_local_dense = torch.zeros(
            batch_size,
            padded_length,
            *v_dense.shape[2:],
            dtype=v_dense.dtype,
            device=v_dense.device,
        )
        if not is_fake_mode():
            assert inputs.kv_lengths is not None
            for batch_index, global_length in enumerate(inputs.kv_lengths):
                local_length = local_lengths[batch_index]
                leftpad = cache_leftpads[batch_index]
                # CP positions are relative to the logical sequence, after the
                # physical cache's left padding has been removed.
                k_local_dense[batch_index, :local_length] = k_dense[
                    batch_index,
                    leftpad + cp_rank : global_length : inputs.cp_world_size,
                ]
                v_local_dense[batch_index, :local_length] = v_dense[
                    batch_index,
                    leftpad + cp_rank : global_length : inputs.cp_world_size,
                ]
        k_local = k_local_dense.reshape(
            batch_size * pages_per_sequence,
            page_size,
            *k_dense.shape[2:],
        )
        v_local = v_local_dense.reshape(
            batch_size * pages_per_sequence,
            page_size,
            *v_dense.shape[2:],
        )
        page_table = torch.arange(
            batch_size * pages_per_sequence,
            dtype=torch.int32,
            device=k_dense.device,
        ).reshape(batch_size, pages_per_sequence)
        seqlens = torch.tensor(local_lengths, dtype=torch.int32, device=k_dense.device)
        return k_local, v_local, page_table, seqlens

    def _verify_threashold(
        self,
        inputs: FmhaInputs,
        outputs: FmhaOutputs,
        reference: FmhaReference,
    ) -> None:
        workload = inputs.workload
        atol = workload.atol if workload is not None else 1.5e-2
        rtol = workload.rtol if workload is not None else 1e-2
        out = outputs.out
        out_ref = reference.out.to(dtype=out.dtype)
        valid_queries = self._valid_query_mask(inputs, out)
        if valid_queries is not None:
            out = out[valid_queries]
            out_ref = out_ref[valid_queries]
        torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

        compare_lse = workload.compare_lse if workload is not None else True
        if compare_lse and reference.lse is not None:
            if outputs.lse is None:
                raise AssertionError("FMHA did not return the requested softmax LSE")
            lse = outputs.lse
            lse_ref = reference.lse.to(dtype=lse.dtype)
            if valid_queries is not None:
                if valid_queries.dim() == 1 and lse.dim() == 2:
                    lse = lse[:, valid_queries]
                    lse_ref = lse_ref[:, valid_queries]
                elif inputs.api is FmhaApi.COMBINE:
                    valid_lse = valid_queries.unsqueeze(-1).expand_as(lse)
                    lse = lse[valid_lse]
                    lse_ref = lse_ref[valid_lse]
                elif valid_queries.dim() == 2 and lse.dim() == 3:
                    valid_lse = valid_queries.unsqueeze(1).expand_as(lse)
                    lse = lse[valid_lse]
                    lse_ref = lse_ref[valid_lse]
            torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)

        if (
            workload is not None
            and workload.reuse_output
            and outputs.out is not inputs.out
        ):
            raise AssertionError("FMHA did not reuse the supplied output tensor")
        if workload is not None and workload.expect_single_split:
            if inputs.scheduler_metadata is None:
                raise AssertionError("FMHA did not generate scheduler metadata")
            splits = inputs.scheduler_metadata[: workload.batch_size]
            torch.testing.assert_close(splits, torch.ones_like(splits), atol=0, rtol=0)

    def _verify_diff(
        self,
        inputs: FmhaInputs,
        outputs: FmhaOutputs,
        reference: FmhaReference,
    ) -> None:
        if reference.high_precision_out is None or inputs.workload is None:
            raise ValueError("diff verification requires a high-precision reference")
        out = outputs.out.float()
        out_ref = reference.high_precision_out.float()
        low_precision_out = reference.out.float()
        valid_queries = self._valid_query_mask(inputs, out)
        if valid_queries is not None:
            out = out[valid_queries]
            out_ref = out_ref[valid_queries]
            low_precision_out = low_precision_out[valid_queries]
        if out.numel() == 0:
            return
        max_multiplier = 4 if inputs.workload.dtype in _FP8_DTYPES else 2
        mean_multiplier = 3 if inputs.workload.dtype in _FP8_DTYPES else 1.5
        diff = (out - out_ref).abs()
        reference_diff = (low_precision_out - out_ref).abs()
        if diff.max().item() > max_multiplier * reference_diff.max().item() + 1e-5:
            raise AssertionError("FMHA maximum error exceeded the diff threshold")
        if diff.mean().item() > mean_multiplier * reference_diff.mean().item() + 1e-5:
            raise AssertionError("FMHA mean error exceeded the diff threshold")

    @staticmethod
    def _valid_query_mask(
        inputs: FmhaInputs, output: torch.Tensor
    ) -> torch.Tensor | None:
        mask = inputs.query_padding_mask
        if mask is not None:
            if inputs.q_indices is not None:
                return mask.flatten()[inputs.q_indices]
            return mask
        if inputs.seqused_q is None:
            return None

        seqused_q = inputs.seqused_q.to(device=output.device, dtype=torch.long)
        if inputs.cu_seqlens_q is None:
            positions = torch.arange(output.shape[1], device=output.device)
            return positions.unsqueeze(0) < seqused_q.unsqueeze(1)

        cu_seqlens_q = inputs.cu_seqlens_q.to(device=output.device, dtype=torch.long)
        physical_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        starts = torch.repeat_interleave(cu_seqlens_q[:-1], physical_lengths)
        used_lengths = torch.repeat_interleave(seqused_q, physical_lengths)
        positions = torch.arange(output.shape[0], device=output.device) - starts
        return positions < used_lengths

    def _verify_appended_cache(
        self,
        inputs: FmhaInputs,
        reference: FmhaReference,
    ) -> None:
        if inputs.k is None or inputs.k_cache is None:
            return
        if inputs.workload is None:
            raise ValueError("append-KV verification requires its workload")
        if reference.expected_k_cache is None or reference.expected_v_cache is None:
            raise ValueError("append-KV verification requires expected caches")
        k_actual = self._logical_cache(
            inputs.k_cache.to(inputs.workload.reference_dtype), inputs.page_table
        )
        v_actual = self._logical_cache(
            inputs.v_cache.to(inputs.workload.reference_dtype), inputs.page_table
        )
        if inputs.cache_batch_idx is not None:
            indices = inputs.cache_batch_idx.long()
            k_actual = k_actual[indices]
            v_actual = v_actual[indices]
        k_actual = k_actual[:, : inputs.workload.seqlen_k].to(
            inputs.workload.reference_dtype
        )
        v_actual = v_actual[:, : inputs.workload.seqlen_k].to(
            inputs.workload.reference_dtype
        )
        if inputs.workload.dtype in _FP8_DTYPES:
            torch.testing.assert_close(
                v_actual, reference.expected_v_cache, atol=1e-3, rtol=1e-3
            )
        else:
            torch.testing.assert_close(
                v_actual, reference.expected_v_cache, atol=0, rtol=0
            )
        if inputs.rotary_dim == 0:
            atol = rtol = 0.0
        elif inputs.workload.dtype in _FP8_DTYPES:
            atol = rtol = 1e-1
        else:
            atol, rtol = 1e-2, 1.5e-2
        torch.testing.assert_close(
            k_actual, reference.expected_k_cache, atol=atol, rtol=rtol
        )

    @staticmethod
    def _seed(seed: int | None, device: str | torch.device) -> None:
        if seed is None:
            return
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.device(device).type == "musa" and hasattr(torch, "musa"):
            torch.musa.manual_seed(seed)

    @staticmethod
    def _unsupported_reason(workload: FmhaWorkload) -> str | None:
        local = workload.window_size not in ((None, None), (-1, -1))
        if workload.learnable_sink and not local:
            return "learnable sink is only supported by these workloads with local attention"
        if workload.cp_world_size > 1:
            if workload.api is FmhaApi.COMBINE:
                return "combine does not support context parallelism"
            if workload.q is None or workload.q.layout is not FmhaLayout.RAGGED:
                return "context parallelism requires a ragged q layout"
            if local or workload.attention_chunk > 0:
                return "context parallelism does not support local or chunked attention"
            if workload.appends_kv:
                return "append-KV is not supported with context parallelism"
            if workload.api is FmhaApi.KVCACHE and (
                workload.k_cache is None
                or workload.k_cache.layout is not FmhaLayout.PAGED
            ):
                return "kvcache context parallelism requires a paged cache layout"
        dimensions = (workload.head_dim_qk, workload.head_dim_v)
        if (
            workload.appends_kv
            and workload.qv is None
            and dimensions not in ((192, 128), (128, 128))
        ):
            return "non-QV append-KV only supports d192x128 and d128x128"
        if (
            workload.appends_kv
            and workload.seqlen_k is not None
            and workload.seqlen_q > workload.seqlen_k
        ):
            return "append KV requires seqlen_q <= seqlen_k"
        if not workload.appends_kv and (
            workload.rotary_fraction > 0
            or workload.seqlen_new_eq_seqlen_q
            or workload.has_rotary_seqlens
        ):
            return "rotary append-KV options require appended k and v"
        if workload.rotary_fraction == 0 and (
            workload.has_rotary_seqlens or not workload.rotary_interleaved
        ):
            return "rotary layout options require rotary embeddings"
        if workload.dtype in _FP8_DTYPES and workload.qv is not None:
            if dimensions not in ((64, 256), (64, 512)):
                return "float8 QV only supports d64x256 and d64x512"
        return None

    def _generate_attention(self, workload: FmhaWorkload) -> FmhaInputs:
        assert workload.seqlen_k is not None
        assert workload.num_heads_kv is not None
        assert workload.head_dim_qk is not None
        assert workload.q is not None
        q_lengths = self._resolve_lengths(
            workload.q_lengths, workload.batch_size, workload.seqlen_q
        )
        q_used = self._resolve_used_lengths(workload.q_used_lengths, q_lengths)
        q, q_padded, q_mask, q_indices, cu_q, seqused_q = self._generate_tensor(
            workload.q,
            q_lengths,
            q_used,
            workload.seqlen_q,
            workload.num_heads_q,
            workload.head_dim_qk,
            workload.dtype,
            workload.reference_dtype,
            workload.device,
            use_seqused=workload.q_used_lengths is not None,
        )
        if workload.qv is not None:
            qv, qv_padded, _, _, _, _ = self._generate_tensor(
                workload.qv,
                q_lengths,
                q_used,
                workload.seqlen_q,
                workload.num_heads_q,
                workload.head_dim_v,
                workload.dtype,
                workload.reference_dtype,
                workload.device,
            )
        else:
            qv = qv_padded = None

        learnable_sink = (
            torch.randn(
                workload.num_heads_q, device=workload.device, dtype=workload.dtype
            )
            * 10
            if workload.learnable_sink
            else None
        )

        has_direct_kv = workload.k is not None and workload.v is not None
        has_cache = workload.k_cache is not None and workload.v_cache is not None
        appends_kv = has_direct_kv and has_cache
        k = v = k_padded = v_padded = None
        k_mask = k_indices = cu_k = seqused_k = None
        max_seqlen_k_new = 0
        direct_lengths: tuple[int, ...] | None = None
        direct_used: tuple[int, ...] | None = None
        if has_direct_kv:
            assert workload.k is not None and workload.v is not None
            if appends_kv:
                max_seqlen_k_new = (
                    workload.seqlen_q
                    if workload.seqlen_new_eq_seqlen_q
                    else random.randint(1, workload.seqlen_q)
                )
                direct_lengths = (
                    tuple(
                        random.randint(1, max_seqlen_k_new)
                        for _ in range(workload.batch_size)
                    )
                    if workload.k.layout is FmhaLayout.RAGGED
                    else (max_seqlen_k_new,) * workload.batch_size
                )
                direct_used = direct_lengths
                use_seqused_k = False
            else:
                direct_lengths = self._resolve_lengths(
                    workload.kv_lengths, workload.batch_size, workload.seqlen_k
                )
                direct_used = self._resolve_used_lengths(
                    workload.kv_used_lengths, direct_lengths
                )
                max_seqlen_k_new = 0
                use_seqused_k = workload.kv_used_lengths is not None
            direct_maximum = max_seqlen_k_new if appends_kv else workload.seqlen_k
            k, k_padded, k_mask, k_indices, cu_k, seqused_k = self._generate_tensor(
                workload.k,
                direct_lengths,
                direct_used,
                direct_maximum,
                workload.num_heads_kv,
                workload.head_dim_qk,
                workload.dtype,
                workload.reference_dtype,
                workload.device,
                use_seqused=use_seqused_k,
            )
            v, v_padded, _, _, _, _ = self._generate_tensor(
                workload.v,
                direct_lengths,
                direct_used,
                direct_maximum,
                workload.num_heads_kv,
                workload.head_dim_v,
                workload.dtype,
                workload.reference_dtype,
                workload.device,
            )

        k_cache = v_cache = None
        k_cache_reference = v_cache_reference = None
        page_table = cache_seqlens = cache_leftpad = cache_batch_idx = None
        cache_lengths: tuple[int, ...] | None = None
        num_pages = 0
        if has_cache:
            assert workload.k_cache is not None and workload.v_cache is not None
            batch_size_cache = (
                workload.batch_size * 2
                if workload.has_cache_batch_idx
                else workload.batch_size
            )
            if workload.k_cache.layout is FmhaLayout.PAGED:
                assert workload.page_size is not None
                (
                    _,
                    _,
                    page_table,
                    k_cache_reference,
                    v_cache_reference,
                    num_pages,
                ) = generate_block_kvcache(
                    workload.seqlen_k,
                    workload.page_size,
                    batch_size_cache,
                    workload.num_heads_kv,
                    workload.head_dim_qk,
                    workload.head_dim_v,
                    workload.device,
                    workload.dtype,
                    workload.reference_dtype,
                    torch.randn,
                )
            else:
                k_cache_reference = self._generate_reference_tensor(
                    (
                        batch_size_cache,
                        workload.seqlen_k,
                        workload.num_heads_kv,
                        workload.head_dim_qk,
                    ),
                    workload,
                )
                v_cache_reference = self._generate_reference_tensor(
                    (
                        batch_size_cache,
                        workload.seqlen_k,
                        workload.num_heads_kv,
                        workload.head_dim_v,
                    ),
                    workload,
                )
            k_cache = workload.k_cache.init(k_cache_reference.to(workload.dtype))
            v_cache = workload.v_cache.init(v_cache_reference.to(workload.dtype))
            cache_batch_idx = (
                torch.randperm(
                    batch_size_cache,
                    dtype=torch.int32,
                    device=workload.device,
                )[: workload.batch_size]
                if workload.has_cache_batch_idx
                else None
            )

            if appends_kv:
                effective_local = (
                    workload.window_size not in ((None, None), (-1, -1))
                    or workload.attention_chunk > 0
                )
                rotary_dim = (
                    math.floor(workload.rotary_fraction * workload.head_dim_qk / 16)
                    * 16
                )
                cache_upper = (
                    workload.seqlen_k
                    - (
                        workload.seqlen_q
                        if (workload.causal or effective_local) and rotary_dim > 1
                        else max_seqlen_k_new
                    )
                    + 1
                )
                cache_lengths = tuple(
                    random.randrange(max(cache_upper, 1))
                    for _ in range(workload.batch_size)
                )
            else:
                configured_cache_lengths = self._resolve_lengths(
                    workload.kv_lengths, workload.batch_size, workload.seqlen_k
                )
                cache_lengths = self._resolve_used_lengths(
                    workload.kv_used_lengths, configured_cache_lengths
                )
            cache_seqlens = torch.tensor(
                cache_lengths,
                dtype=torch.int32,
                device=workload.device,
            )
            cache_leftpad = self._generate_cache_leftpad(workload, cache_lengths)

        key_padding_mask = k_mask
        if has_cache:
            assert cache_lengths is not None
            arange = torch.arange(workload.seqlen_k, device=workload.device).reshape(
                1, -1
            )
            final_cache_lengths: Sequence[int] | torch.Tensor = cache_lengths
            if appends_kv:
                assert direct_lengths is not None
                final_cache_lengths = torch.tensor(
                    cache_lengths, dtype=torch.int32, device=workload.device
                ) + torch.tensor(
                    direct_lengths, dtype=torch.int32, device=workload.device
                )
            key_padding_mask = arange < torch.as_tensor(
                final_cache_lengths,
                dtype=torch.int32,
                device=workload.device,
            ).reshape(-1, 1)
            if cache_leftpad is not None:
                key_padding_mask &= arange >= cache_leftpad.unsqueeze(-1)

        rotary_dim = (
            math.floor(workload.rotary_fraction * workload.head_dim_qk / 16) * 16
        )
        rotary_seqlens = (
            cache_seqlens // 2 if workload.has_rotary_seqlens else cache_seqlens
        )
        if rotary_dim > 0:
            rotary_length = (
                num_pages * workload.page_size
                if workload.page_size is not None
                else workload.seqlen_k
            )
            angle = (
                torch.rand(
                    rotary_length,
                    rotary_dim // 2,
                    device=workload.device,
                )
                * 2
                * math.pi
            )
            rotary_cos_reference = (
                torch.cos(angle)
                .to(workload.reference_dtype)
                .to(workload.dtype)
                .to(workload.reference_dtype)
            )
            rotary_sin_reference = (
                torch.sin(angle)
                .to(workload.reference_dtype)
                .to(workload.dtype)
                .to(workload.reference_dtype)
            )
            rotary_cos = rotary_cos_reference.to(workload.dtype)
            rotary_sin = rotary_sin_reference.to(workload.dtype)
        else:
            rotary_cos = rotary_sin = None
            rotary_cos_reference = rotary_sin_reference = None

        if workload.dtype in _FP8_DTYPES:
            q_descale, k_descale, v_descale = [
                torch.rand(
                    workload.batch_size,
                    workload.num_heads_kv,
                    device=workload.device,
                    dtype=torch.float32,
                )
                * 2
                for _ in range(3)
            ]
        else:
            q_descale = k_descale = v_descale = None

        return FmhaInputs(
            api=workload.api,
            workload=workload,
            q=q,
            q_padded=q_padded,
            qv=qv,
            qv_padded=qv_padded,
            k=k,
            v=v,
            k_padded=k_padded,
            v_padded=v_padded,
            k_cache=k_cache,
            v_cache=v_cache,
            k_cache_reference=k_cache_reference,
            v_cache_reference=v_cache_reference,
            query_padding_mask=q_mask,
            key_padding_mask=key_padding_mask,
            key_new_padding_mask=k_mask if appends_kv else None,
            q_indices=q_indices,
            k_indices=k_indices,
            q_lengths=q_lengths,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=None if appends_kv else cu_k,
            cu_seqlens_k_new=cu_k if appends_kv else None,
            seqused_q=seqused_q,
            seqused_k=None if appends_kv else seqused_k,
            max_seqlen_q=(
                workload.seqlen_q if workload.q.layout is FmhaLayout.RAGGED else None
            ),
            max_seqlen_k=workload.seqlen_k,
            max_seqlen_k_new=max_seqlen_k_new,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            cache_leftpad=cache_leftpad,
            page_table=page_table,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            rotary_cos_reference=rotary_cos_reference,
            rotary_sin_reference=rotary_sin_reference,
            rotary_seqlens=(
                rotary_seqlens
                if (workload.qv is None or workload.has_rotary_seqlens)
                else None
            ),
            # QV kernels without an explicit rotary_seqlens use cache_seqlens as
            # their offset. The reference still needs that resolved value.
            rotary_seqlens_reference=rotary_seqlens,
            rotary_dim=rotary_dim,
            rotary_interleaved=workload.rotary_interleaved,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=workload.softmax_scale,
            causal=workload.causal,
            window_size=workload.window_size,
            learnable_sink=learnable_sink,
            attention_chunk=workload.attention_chunk,
            softcap=workload.softcap,
            num_splits=workload.num_splits,
            pack_gqa=workload.pack_gqa,
            backend=workload.backend,
            cp_world_size=workload.cp_world_size,
            only_qv=workload.only_qv,
            return_softmax_lse=workload.return_softmax_lse,
            prepare_scheduler_metadata=True,
            kv_lengths=(cache_lengths if has_cache else direct_lengths),
        )

    def _generate_combine(self, workload: FmhaWorkload) -> FmhaInputs:
        if workload.combine_layout is FmhaLayout.RAGGED:
            q_lengths = tuple(
                random.randint(1, workload.seqlen_q) for _ in range(workload.batch_size)
            )
            total_q = sum(q_lengths)
            shape_out = (
                workload.num_splits,
                1,
                total_q,
                workload.num_heads_q,
                workload.head_dim_v,
            )
            cu_q = torch.tensor(
                self._cumulative_lengths(q_lengths),
                dtype=torch.int32,
                device=workload.device,
            )
            seqused_q = None
        else:
            q_lengths = (workload.seqlen_q,) * workload.batch_size
            shape_out = (
                workload.num_splits,
                workload.batch_size,
                workload.seqlen_q,
                workload.num_heads_q,
                workload.head_dim_v,
            )
            cu_q = None
            seqused_q = (
                torch.tensor(
                    [
                        random.randint(1, workload.seqlen_q)
                        for _ in range(workload.batch_size)
                    ],
                    dtype=torch.int32,
                    device=workload.device,
                )
                if workload.combine_layout is FmhaLayout.PADDED
                else None
            )
        out_partial = torch.randn(
            shape_out, dtype=torch.float32, device=workload.device
        )
        lse_storage_shape = shape_out[:2] + (shape_out[3], shape_out[2])
        lse_partial = torch.randn(
            lse_storage_shape, dtype=torch.float32, device=workload.device
        ).transpose(2, 3)
        valid_splits = (
            tuple(
                random.randint(2, workload.num_splits)
                for _ in range(workload.batch_size)
            )
            if workload.randomize_splits
            else (workload.num_splits,) * workload.batch_size
        )
        if not is_fake_mode():
            for batch_index, split_count in enumerate(valid_splits):
                if workload.combine_layout is FmhaLayout.RAGGED:
                    assert cu_q is not None
                    start = int(cu_q[batch_index].item())
                    end = int(cu_q[batch_index + 1].item())
                    out_partial[split_count:, 0, start:end] = 0
                    lse_partial[split_count:, 0, start:end] = float("-inf")
                else:
                    out_partial[split_count:, batch_index] = 0
                    lse_partial[split_count:, batch_index] = float("-inf")
            if seqused_q is not None:
                for batch_index, length in enumerate(seqused_q.cpu().tolist()):
                    out_partial[:, batch_index, length:] = 0
                    lse_partial[:, batch_index, length:] = float("-inf")
        out = (
            torch.empty(
                shape_out[1:],
                dtype=workload.dtype,
                device=workload.device,
            )
            if workload.reuse_output
            else None
        )
        return FmhaInputs(
            api=FmhaApi.COMBINE,
            workload=workload,
            out_partial=out_partial.contiguous(),
            lse_partial=lse_partial,
            out=out,
            out_dtype=workload.dtype,
            num_splits=workload.num_splits,
            cu_seqlens_q=cu_q,
            seqused_q=seqused_q,
            q_lengths=q_lengths,
        )

    @staticmethod
    def _generate_tensor(
        spec: FmhaTensorSpec,
        lengths: tuple[int, ...],
        used_lengths: tuple[int, ...],
        maximum: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        reference_dtype: torch.dtype,
        device: str | torch.device,
        *,
        use_seqused: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        padded = (
            torch.randn(
                len(lengths),
                maximum,
                num_heads,
                head_dim,
                dtype=reference_dtype,
                device=device,
            )
            .to(dtype)
            .to(reference_dtype)
        )
        if spec.layout is FmhaLayout.NORMAL:
            return spec.init(padded.to(dtype)), padded, None, None, None, None
        used_mask = FmhaOperator._length_mask(used_lengths, maximum, device)
        seqused = torch.tensor(used_lengths, dtype=torch.int32, device=device)
        if spec.layout is FmhaLayout.PADDED:
            return spec.init(padded.to(dtype)), padded, used_mask, None, None, seqused
        if spec.layout is not FmhaLayout.RAGGED:
            raise ValueError(f"unsupported tensor layout: {spec.layout.value}")
        # cu_seqlens describes packed storage; seqused limits the logical prefix.
        storage_mask = FmhaOperator._length_mask(lengths, maximum, device)
        unused_mask = storage_mask & ~used_mask
        unpadded, indices, cu_seqlens, _, generated_seqused = unpad_input(
            padded, used_mask, unused_mask
        )
        return (
            spec.init(unpadded.to(dtype)),
            padded,
            used_mask,
            indices,
            cu_seqlens,
            generated_seqused if use_seqused else None,
        )

    @staticmethod
    def _generate_reference_tensor(
        shape: tuple[int, ...], workload: FmhaWorkload
    ) -> torch.Tensor:
        return (
            torch.randn(
                shape,
                dtype=workload.reference_dtype,
                device=workload.device,
            )
            .to(workload.dtype)
            .to(workload.reference_dtype)
        )

    @staticmethod
    def _generate_cache_leftpad(
        workload: FmhaWorkload,
        cache_lengths: Sequence[int],
    ) -> torch.Tensor | None:
        if not workload.has_cache_leftpad:
            return None
        if is_fake_mode():
            return torch.empty(
                workload.batch_size,
                dtype=torch.int32,
                device=workload.device,
            )
        return torch.tensor(
            [random.randrange(length) if length > 0 else 0 for length in cache_lengths],
            dtype=torch.int32,
            device=workload.device,
        )

    @staticmethod
    def _resolve_lengths(
        lengths: Sequence[int] | None,
        batch_size: int,
        maximum: int,
    ) -> tuple[int, ...]:
        return (
            (maximum,) * batch_size
            if lengths is None
            else tuple(int(length) for length in lengths)
        )

    @staticmethod
    def _resolve_used_lengths(
        used_lengths: Sequence[int] | None,
        lengths: tuple[int, ...],
    ) -> tuple[int, ...]:
        if used_lengths is None:
            return lengths
        return tuple(
            length if used < 0 else min(length, int(used))
            for length, used in zip(lengths, used_lengths)
        )

    @staticmethod
    def _length_mask(
        lengths: Sequence[int], maximum: int, device: str | torch.device
    ) -> torch.Tensor:
        return torch.arange(maximum, device=device).reshape(1, -1) < torch.tensor(
            lengths, dtype=torch.int32, device=device
        ).reshape(-1, 1)

    @staticmethod
    def _cumulative_lengths(lengths: Sequence[int]) -> list[int]:
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + int(length))
        return cumulative

    @staticmethod
    def _call_combine(inputs: FmhaInputs) -> FmhaOutputs:
        if inputs.out_partial is None or inputs.lse_partial is None:
            raise ValueError("combine calls require out_partial and lse_partial")
        out, lse = flash_attn_combine(
            inputs.out_partial,
            inputs.lse_partial,
            out=inputs.out,
            out_dtype=inputs.out_dtype,
        )
        return FmhaOutputs(out=out, lse=lse)

    @staticmethod
    def _normalize_outputs(result: Any, return_lse: bool) -> FmhaOutputs:
        if not return_lse:
            if not isinstance(result, torch.Tensor):
                raise TypeError("FMHA returned an unexpected output value")
            return FmhaOutputs(out=result)
        if not isinstance(result, tuple) or len(result) < 2:
            raise TypeError("FMHA did not return the requested output and LSE")
        out, lse, *auxiliary = result
        return FmhaOutputs(
            out=out,
            lse=lse,
            out_accum=auxiliary[0] if len(auxiliary) > 0 else None,
            lse_accum=auxiliary[1] if len(auxiliary) > 1 else None,
            auxiliary=tuple(auxiliary),
        )

    @staticmethod
    def _make_scheduler_metadata(inputs: FmhaInputs) -> torch.Tensor:
        if inputs.q is None:
            raise ValueError("scheduler metadata requires q")
        if inputs.q.dim() == 4:
            batch_size, max_seqlen_q = inputs.q.shape[:2]
        elif inputs.cu_seqlens_q is not None and inputs.max_seqlen_q is not None:
            batch_size = inputs.cu_seqlens_q.shape[0] - 1
            max_seqlen_q = inputs.max_seqlen_q
        else:
            raise ValueError("ragged q requires cu_seqlens_q and max_seqlen_q")
        kv = inputs.k if inputs.api is FmhaApi.VARLEN else inputs.k_cache
        vv = inputs.v if inputs.api is FmhaApi.VARLEN else inputs.v_cache
        if kv is None or vv is None:
            raise ValueError("scheduler metadata requires K and V")
        max_seqlen_k = inputs.max_seqlen_k or kv.shape[1]
        page_size = kv.shape[1] if inputs.page_table is not None else None
        return get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=inputs.q.shape[-2],
            num_heads_kv=kv.shape[-2],
            headdim=inputs.q.shape[-1],
            seqused_q=inputs.seqused_q,
            seqused_k=(
                inputs.seqused_k
                if inputs.api is FmhaApi.VARLEN
                else inputs.cache_seqlens
            ),
            qkv_dtype=inputs.q.dtype,
            headdim_v=vv.shape[-1],
            cu_seqlens_q=inputs.cu_seqlens_q,
            cu_seqlens_k=inputs.cu_seqlens_k,
            cu_seqlens_k_new=inputs.cu_seqlens_k_new,
            cache_leftpad=inputs.cache_leftpad,
            page_size=page_size,
            max_seqlen_k_new=inputs.max_seqlen_k_new,
            causal=inputs.causal,
            window_size=inputs.window_size,
            attention_chunk=inputs.attention_chunk,
            has_softcap=inputs.softcap != 0.0,
            num_splits=inputs.num_splits,
            pack_gqa=inputs.pack_gqa,
            has_qv=inputs.qv is not None,
            mp_margin=0,
        )

    def _reference_attention(self, inputs: FmhaInputs) -> FmhaReference:
        if inputs.workload is None or inputs.q_padded is None:
            raise ValueError("attention reference requires its workload and padded q")
        workload = inputs.workload
        has_cache = inputs.k_cache is not None and inputs.v_cache is not None
        appends_kv = has_cache and inputs.k_padded is not None
        if has_cache:
            k_cache_source = (
                inputs.k_cache_reference
                if inputs.k_cache_reference is not None
                else inputs.k_cache
            )
            v_cache_source = (
                inputs.v_cache_reference
                if inputs.v_cache_reference is not None
                else inputs.v_cache
            )
            k_ref = self._logical_cache(k_cache_source, inputs.page_table).to(
                workload.reference_dtype
            )[:, : workload.seqlen_k]
            v_ref = self._logical_cache(v_cache_source, inputs.page_table).to(
                workload.reference_dtype
            )[:, : workload.seqlen_k]
            if inputs.cache_batch_idx is not None:
                indices = inputs.cache_batch_idx.long()
                k_ref = k_ref[indices].clone()
                v_ref = v_ref[indices].clone()
            else:
                k_ref = k_ref[: workload.batch_size].clone()
                v_ref = v_ref[: workload.batch_size].clone()
        else:
            if inputs.k_padded is None or inputs.v_padded is None:
                raise ValueError("attention reference requires logical K and V")
            k_ref = inputs.k_padded
            v_ref = inputs.v_padded

        q_ref = inputs.q_padded
        k_new = inputs.k_padded
        v_new = inputs.v_padded
        if inputs.rotary_dim > 0:
            rotary_seqlens_reference = (
                inputs.rotary_seqlens_reference
                if inputs.rotary_seqlens_reference is not None
                else inputs.rotary_seqlens
            )
            if (
                inputs.rotary_cos_reference is None
                or inputs.rotary_sin_reference is None
                or rotary_seqlens_reference is None
            ):
                raise ValueError("rotary reference requires cos, sin, and seqlens")
            effective_local = (
                inputs.window_size not in ((None, None), (-1, -1))
                or inputs.attention_chunk > 0
            )
            if inputs.causal or effective_local:
                q_ref = apply_rotary_emb(
                    q_ref,
                    inputs.rotary_cos_reference,
                    inputs.rotary_sin_reference,
                    seqlen_offsets=rotary_seqlens_reference,
                    interleaved=inputs.rotary_interleaved,
                )
            else:
                q_ref = rearrange(
                    apply_rotary_emb(
                        rearrange(q_ref, "b s h d -> b 1 (s h) d"),
                        inputs.rotary_cos_reference,
                        inputs.rotary_sin_reference,
                        seqlen_offsets=rotary_seqlens_reference,
                        interleaved=inputs.rotary_interleaved,
                    ),
                    "b 1 (s h) d -> b s h d",
                    s=workload.seqlen_q,
                )
            if k_new is not None:
                k_new = apply_rotary_emb(
                    k_new,
                    inputs.rotary_cos_reference,
                    inputs.rotary_sin_reference,
                    seqlen_offsets=rotary_seqlens_reference,
                    interleaved=inputs.rotary_interleaved,
                )

        if appends_kv:
            if (
                k_new is None
                or v_new is None
                or not isinstance(inputs.cache_seqlens, torch.Tensor)
            ):
                raise ValueError(
                    "append-KV reference requires new K/V and cache lengths"
                )
            arange = torch.arange(workload.seqlen_k, device=q_ref.device).reshape(1, -1)
            cache_lengths = inputs.cache_seqlens.reshape(-1, 1)
            new_lengths: int | torch.Tensor = (
                inputs.key_new_padding_mask.sum(-1, keepdim=True)
                if inputs.key_new_padding_mask is not None
                else inputs.max_seqlen_k_new
            )
            update_mask = (cache_lengths <= arange) & (
                arange < cache_lengths + new_lengths
            )
            k_update = k_new.reshape(-1, *k_new.shape[2:])
            v_update = v_new.reshape(-1, *v_new.shape[2:])
            if inputs.k_indices is not None:
                k_update = k_update[inputs.k_indices]
                v_update = v_update[inputs.k_indices]
            k_ref[update_mask] = k_update
            v_ref[update_mask] = v_update

        high_precision_out, _, score_ref = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask=inputs.query_padding_mask,
            key_padding_mask=inputs.key_padding_mask,
            key_leftpad=inputs.cache_leftpad,
            causal=inputs.causal,
            qv=inputs.qv_padded,
            q_descale=inputs.q_descale,
            k_descale=inputs.k_descale,
            v_descale=inputs.v_descale,
            window_size=inputs.window_size,
            attention_chunk=inputs.attention_chunk,
            learnable_sink=inputs.learnable_sink,
            softcap=inputs.softcap,
            only_qv=inputs.only_qv,
            softmax_scale=inputs.softmax_scale,
        )
        if workload.verify_mode is FmhaVerifyMode.DIFF:
            out_ref, _, _ = attention_ref(
                q_ref,
                k_ref,
                v_ref,
                query_padding_mask=inputs.query_padding_mask,
                key_padding_mask=inputs.key_padding_mask,
                key_leftpad=inputs.cache_leftpad,
                causal=inputs.causal,
                qv=inputs.qv_padded,
                q_descale=inputs.q_descale,
                k_descale=inputs.k_descale,
                v_descale=inputs.v_descale,
                window_size=inputs.window_size,
                attention_chunk=inputs.attention_chunk,
                learnable_sink=inputs.learnable_sink,
                softcap=inputs.softcap,
                upcast=False,
                reorder_ops=True,
                intermediate_dtype=(
                    workload.dtype if workload.dtype in _FP8_DTYPES else None
                ),
                only_qv=inputs.only_qv,
                softmax_scale=inputs.softmax_scale,
            )
        else:
            out_ref = high_precision_out

        if inputs.q_indices is not None:
            high_precision_out = high_precision_out.reshape(
                -1, *high_precision_out.shape[2:]
            )[inputs.q_indices]
            out_ref = out_ref.reshape(-1, *out_ref.shape[2:])[inputs.q_indices]

        lse_ref = None
        if (
            workload.verify_mode is FmhaVerifyMode.THREASHOLD
            and inputs.return_softmax_lse
        ):
            lse_seqused_k = inputs.seqused_k
            if has_cache:
                if isinstance(inputs.cache_seqlens, torch.Tensor):
                    lse_seqused_k = inputs.cache_seqlens
                elif isinstance(inputs.cache_seqlens, int):
                    lse_seqused_k = torch.full(
                        (workload.batch_size,),
                        inputs.cache_seqlens,
                        dtype=torch.int32,
                        device=score_ref.device,
                    )
                else:
                    lse_seqused_k = None
                if appends_kv and lse_seqused_k is not None:
                    new_lengths = (
                        inputs.key_new_padding_mask.sum(-1)
                        if inputs.key_new_padding_mask is not None
                        else inputs.max_seqlen_k_new
                    )
                    lse_seqused_k = lse_seqused_k + new_lengths
            _, lse_padded = lse_ref_from_score(
                score_ref,
                is_causal=inputs.causal,
                cu_seqlens_q=inputs.cu_seqlens_q,
                cu_seqlens_k=inputs.cu_seqlens_k,
                seqused_q=inputs.seqused_q,
                seqused_k=lse_seqused_k,
                learnable_sink=inputs.learnable_sink,
            )
            if inputs.q_indices is not None:
                lse_ref = lse_padded.permute(0, 2, 1).reshape(-1, lse_padded.shape[1])[
                    inputs.q_indices
                ]
                lse_ref = lse_ref.transpose(0, 1)
            else:
                lse_ref = lse_padded
            self._register_dump_reference(out_ref, lse_ref)

        return FmhaReference(
            out=out_ref,
            lse=lse_ref,
            high_precision_out=(
                high_precision_out
                if workload.verify_mode is FmhaVerifyMode.DIFF
                else None
            ),
            expected_k_cache=(
                k_ref.to(workload.dtype).to(workload.reference_dtype)
                if appends_kv
                else None
            ),
            expected_v_cache=(
                v_ref.to(workload.dtype).to(workload.reference_dtype)
                if appends_kv
                else None
            ),
        )

    @staticmethod
    def _logical_cache(
        cache: torch.Tensor | None, page_table: torch.Tensor | None
    ) -> torch.Tensor:
        if cache is None:
            raise ValueError("cache tensor is required")
        if page_table is None:
            return cache
        batch_size, blocks = page_table.shape
        return cache[page_table.long().reshape(-1)].reshape(
            batch_size, blocks * cache.shape[1], *cache.shape[2:]
        )

    @staticmethod
    def _reference_combine(inputs: FmhaInputs) -> FmhaReference:
        if inputs.out_partial is None or inputs.lse_partial is None:
            raise ValueError("combine reference requires out_partial and lse_partial")
        lse_partial = inputs.lse_partial.clone(memory_format=torch.contiguous_format)
        lse = torch.logsumexp(lse_partial, dim=0)
        weights = torch.exp(lse_partial - lse.unsqueeze(0))
        weights = torch.where(torch.isfinite(weights), weights, 0.0)
        out = (weights.unsqueeze(-1) * inputs.out_partial).sum(dim=0)
        return FmhaReference(out=out, lse=lse)

    @staticmethod
    def _register_dump_reference(out: torch.Tensor, lse: torch.Tensor) -> None:
        if os.environ.get("MATE_JIT_DUMP_REFERENCE", "0") != "1" or not os.environ.get(
            "MATE_JIT_DUMP_DIR"
        ):
            return
        from mate.jit.dump import register_dump_reference

        register_dump_reference("out", out, atol=1.5e-2, rtol=1e-2)
        register_dump_reference("lse", lse, atol=1.5e-2, rtol=1e-2)


__all__ = [
    "FmhaApi",
    "FmhaInputs",
    "FmhaLayout",
    "FmhaOperator",
    "FmhaOutputs",
    "FmhaReference",
    "FmhaTensorInitializer",
    "FmhaTensorSpec",
    "FmhaVerifyMode",
    "FmhaWorkload",
    "UnsupportedFmhaWorkload",
    "large_stride_tensor_init",
    "noncontiguous_tensor_init",
    "default_tensor_init",
]
