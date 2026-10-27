from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist

from mate.jit.mega_moe_common import _align, _normalize_activation
from mate.jit.runtime import ffi_to_torch
from mate.jit.mega_moe import (
    MegaMoEKernelConfig,
    get_fp8_fp8_mega_moe_stage1_module,
    get_fp8_fp8_mega_moe_stage2_module,
    get_mega_moe_runtime_utils_module,
)
from mate.jit.mega_moe_prefill import (
    PREFILL_STAGE3_NUM_THREADS,
    PrefillMegaMoEKernelConfig,
    _validate_stage1_profile_mode,
    get_prefill_stage1_module,
    get_prefill_stage2_module,
    get_prefill_stage3_module,
)
from mate.mate_runtime import resolve_num_mps


def _get_prefill_worst_case_pool_tokens(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
) -> int:
    experts_per_rank = num_experts // num_ranks
    max_routes = num_ranks * num_max_tokens_per_rank * min(num_topk, experts_per_rank)
    return _align(max_routes + experts_per_rank * 255, 256)


def _resolve_prefill_pool_capacity_rows(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    ratio: float,
) -> tuple[float, int]:
    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError("prefill_pool_capacity_ratio must be in (0, 1]")
    worst_case_rows = _get_prefill_worst_case_pool_tokens(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
    )
    return ratio, _align(math.ceil(worst_case_rows * ratio), 256)


def _validate_prefill_memory_options(
    prefill_pool_capacity_ratio: float, prefill_l1_ring: bool
) -> None:
    """Ensure the two prefill activation-pool reduction modes are exclusive."""
    if prefill_l1_ring and prefill_pool_capacity_ratio != 1.0:
        raise ValueError(
            "prefill_l1_ring and prefill_pool_capacity_ratio < 1.0 are mutually exclusive"
        )


def _resolve_prefill_l1_ring_tokens(
    enable: bool,
    num_ranks: int,
    num_topk: int,
    num_max_tokens_per_rank: int,
    prefill_num_max_pool_tokens: int,
) -> int:
    """Pick the routed L1 ring capacity automatically; 0 disables recycling.

    Two requirements drive the size:

    * Safety: grouped swizzle can visit every M block of one expert before
      releasing its first block, so the ring must be large enough that the
      previous occupant of a slot always sits in an earlier expert. Legal,
      deduplicated top-k caps one expert at ``num_ranks * tokens`` rows, i.e.
      ``ceil(num_ranks * num_max_tokens_per_rank / 256)`` blocks.
    * Throughput: measurement on Kimi K3 shapes shows the stage 1 consumer frontier drifts
      across most of the routed pool, so the ring also covers roughly twice the
      expected per-rank routed block count.

    Both stay far below the cross-rank worst-case pool. If the capacity is too
    small to fit even the safety floor, recycling is disabled instead of running
    a ring that could deadlock.
    """
    if not enable:
        return 0
    expert_blocks = (num_ranks * num_max_tokens_per_rank + 255) // 256
    routed_blocks = (num_max_tokens_per_rank * num_topk + 127) // 128
    ring_blocks = 1
    while ring_blocks < max(routed_blocks, expert_blocks):
        ring_blocks <<= 1
    capacity_blocks = prefill_num_max_pool_tokens // 256
    while ring_blocks > capacity_blocks:
        ring_blocks >>= 1
    if ring_blocks < expert_blocks:
        return 0
    return ring_blocks * 256


def _get_unified_buffer_layout(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    prefill_num_max_pool_tokens: int,
    prefill_l1_ring_tokens: int,
):
    utils = get_mega_moe_runtime_utils_module()
    num_bytes, slice_input_buffers = utils.get_function(
        "mega_moe_unified_get_buffer_layout"
    )(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        prefill_num_max_pool_tokens,
        prefill_l1_ring_tokens,
    )
    return int(num_bytes), slice_input_buffers


def _view_fp8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1).view(torch.float8_e4m3fn).view(tensor.shape)


class _SymmHandle:
    def __init__(self, buffer_ptrs_tensor: torch.Tensor) -> None:
        self.buffer_ptrs_tensor = buffer_ptrs_tensor

    @property
    def buffer_ptrs(self) -> List[int]:
        return [int(value) for value in self.buffer_ptrs_tensor.tolist()]


class MegaMoESymmBuffer:
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        use_fp8_dispatch: bool = True,
        activation: str = "swiglu",
        prefill_pool_capacity_ratio: float = 1.0,
        prefill_l1_ring: bool = False,
    ) -> None:
        assert group.size() <= 8, (
            "MegaMoE symmetric buffer supports intranode groups only"
        )
        assert num_experts % group.size() == 0
        activation = _normalize_activation(activation)
        assert use_fp8_dispatch
        assert num_max_tokens_per_rank % 256 == 0
        assert num_max_tokens_per_rank <= 65536
        assert num_topk <= 32
        assert hidden % 128 == 0, "hidden must be divisible by 128"
        assert intermediate_hidden % 128 == 0, (
            "intermediate_hidden must be divisible by 128"
        )
        assert intermediate_hidden <= 8192
        prefill_pool_capacity_ratio, prefill_num_max_pool_tokens = (
            _resolve_prefill_pool_capacity_rows(
                group.size(),
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                prefill_pool_capacity_ratio,
            )
        )
        _validate_prefill_memory_options(prefill_pool_capacity_ratio, prefill_l1_ring)

        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.activation = activation
        self.prefill_pool_capacity_ratio = prefill_pool_capacity_ratio
        self.prefill_num_max_pool_tokens = prefill_num_max_pool_tokens
        self.prefill_l1_ring_tokens = _resolve_prefill_l1_ring_tokens(
            prefill_l1_ring,
            group.size(),
            num_topk,
            num_max_tokens_per_rank,
            prefill_num_max_pool_tokens,
        )

        num_bytes, slice_input_buffers = _get_unified_buffer_layout(
            group.size(),
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            prefill_num_max_pool_tokens,
            self.prefill_l1_ring_tokens,
        )
        self.num_bytes = num_bytes
        utils = get_mega_moe_runtime_utils_module()
        # The owning FFI tensor frees the IPC allocation after its last view.
        self.buffer = cast(
            torch.Tensor,
            ffi_to_torch(
                utils.get_function("mega_moe_alloc_buffer")(
                    num_bytes, torch.musa.current_device()
                )
            ),
        )
        views = list(ffi_to_torch(slice_input_buffers(self.buffer)))
        for index in (0, 4, 7, 10, 12):
            views[index] = _view_fp8(views[index])
        (
            self.x,
            self.x_sf,
            self.topk_idx,
            self.topk_weights,
            self.decode_l1_acts,
            self.decode_l1_acts_sf,
            self.decode_l1_topk_weights,
            self.decode_l2_acts,
            self.decode_l2_acts_sf,
            self.decode_combine_acts,
            self.prefill_l1_acts,
            self.prefill_l1_acts_sf,
            self.prefill_l2_acts,
            self.prefill_l2_acts_sf,
            self.prefill_combine_acts,
        ) = views
        if group.size() == 1:
            ptrs = torch.tensor(
                [self.buffer.data_ptr()], dtype=torch.int64, device="cpu"
            )
        else:
            handle_size = int(utils.get_function("mega_moe_get_ipc_handle_size")())
            local_handle = torch.empty(handle_size, dtype=torch.uint8, device="cpu")
            utils.get_function("mega_moe_get_ipc_handle")(self.buffer, local_handle)

            gathered: List[Optional[bytes]] = [None] * group.size()
            dist.all_gather_object(gathered, bytes(local_handle.tolist()), group)
            handles = torch.empty(
                (group.size(), handle_size), dtype=torch.uint8, device="cpu"
            )
            for idx, handle in enumerate(gathered):
                assert handle is not None and len(handle) == handle_size
                handles[idx].copy_(
                    torch.tensor(list(handle), dtype=torch.uint8, device="cpu")
                )

            ptrs = torch.empty(group.size(), dtype=torch.int64, device="cpu")
            utils.get_function("mega_moe_open_ipc_handles")(
                handles, self.buffer, group.rank(), ptrs
            )

        self.handle = _SymmHandle(ptrs)
        self.y = None
        self.buffer.zero_()
        dist.barrier(group=group)
        torch.musa.synchronize()

    def destroy(self) -> None:
        """Collectively close IPC mappings before releasing the local buffer."""
        if getattr(self, "handle", None) is not None and self.group is not None:
            torch.musa.synchronize(self.buffer.device)
            if self.group.size() > 1:
                dist.barrier(group=self.group)
                utils = get_mega_moe_runtime_utils_module()
                utils.get_function("mega_moe_close_ipc_handles")(
                    self.handle.buffer_ptrs_tensor,
                    self.group.rank(),
                )
                dist.barrier(group=self.group)
            self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.decode_l1_acts = None
        self.decode_l1_acts_sf = None
        self.decode_l1_topk_weights = None
        self.decode_l2_acts = None
        self.decode_l2_acts_sf = None
        self.decode_combine_acts = None
        self.prefill_l1_acts = None
        self.prefill_l1_acts_sf = None
        self.prefill_l2_acts = None
        self.prefill_l2_acts_sf = None
        self.prefill_combine_acts = None
        self.y = None
        self.activation = None
        self.prefill_pool_capacity_ratio = None
        self.prefill_num_max_pool_tokens = None
        self.prefill_l1_ring_tokens = None


# Compatibility aliases; internal dispatch uses MegaMoESymmBuffer.
MegaMoEPrefillSymmBuffer = MegaMoESymmBuffer
SymmBuffer = MegaMoESymmBuffer
PrefillSymmBuffer = MegaMoESymmBuffer


def _validate_activation_options(
    activation: str,
    activation_clamp: Optional[float],
    activation_beta: Optional[float],
    activation_linear_beta: Optional[float],
) -> str:
    activation = _normalize_activation(activation)
    if activation == "situ" and activation_clamp is not None:
        raise ValueError("activation_clamp is only valid for SwiGLU")
    if activation == "situ" and activation_beta is None:
        raise ValueError("activation_beta is required for SiTU")
    if activation == "situ" and (
        not math.isfinite(activation_beta) or activation_beta <= 0.0
    ):
        raise ValueError("SiTU beta must be finite and positive")
    if activation_linear_beta is not None and (
        not math.isfinite(activation_linear_beta) or activation_linear_beta <= 0.0
    ):
        raise ValueError("SiTU linear_beta must be finite and positive")
    return activation


def _kernel_config_kwargs(
    sym_buffer: MegaMoESymmBuffer, fast_math: bool, num_mps: int
) -> dict:
    major, minor = torch.musa.get_device_capability(sym_buffer.buffer.device)
    return dict(
        num_ranks=sym_buffer.group.size(),
        num_experts=sym_buffer.num_experts,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_topk=sym_buffer.num_topk,
        hidden=sym_buffer.hidden,
        intermediate_hidden=sym_buffer.intermediate_hidden,
        num_mps=num_mps,
        fast_math=fast_math,
        cc=int(major) * 10 + int(minor),
    )


def _prefill_kernel_config(
    sym_buffer: MegaMoESymmBuffer, fast_math: bool, num_mps: Optional[int] = None
) -> PrefillMegaMoEKernelConfig:
    return PrefillMegaMoEKernelConfig(
        **_kernel_config_kwargs(
            sym_buffer, fast_math, resolve_num_mps(sym_buffer.buffer.device, num_mps)
        ),
        num_max_pool_tokens=sym_buffer.prefill_num_max_pool_tokens,
        num_l1_ring_tokens=sym_buffer.prefill_l1_ring_tokens,
    )


def _prefill_stage1(
    sym_buffer: MegaMoESymmBuffer,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    num_tokens: int,
    num_mps: Optional[int] = None,
    fast_math: bool = True,
    activation_clamp: Optional[float] = None,
    activation: str = "swiglu",
    activation_beta: Optional[float] = None,
    activation_linear_beta: Optional[float] = None,
    profile_mode: str = "full",
) -> None:
    """Run the fused dispatch/GEMM/activation/FP8 prefill stage.

    Split debug modes (dispatch_only/compute_only) require a symmetric buffer
    created with prefill_l1_ring=False; only full mode supports ring recycling.
    """
    _validate_stage1_profile_mode(profile_mode, sym_buffer.prefill_l1_ring_tokens)
    weight, scale = l1_weights
    config = _prefill_kernel_config(sym_buffer, fast_math, num_mps)
    activation = _validate_activation_options(
        activation, activation_clamp, activation_beta, activation_linear_beta
    )
    name, module = get_prefill_stage1_module(
        config,
        activation=activation,
        profile_mode=profile_mode,
        use_situ_linear=activation_linear_beta is not None,
    )
    kernel_args = [
        sym_buffer.prefill_l1_acts,
        sym_buffer.prefill_l1_acts_sf,
        sym_buffer.prefill_l2_acts,
        sym_buffer.prefill_l2_acts_sf,
        weight,
        scale,
        sym_buffer.handle.buffer_ptrs_tensor,
        sym_buffer.group.rank(),
        num_tokens,
        -1.0 if activation_clamp is None else float(activation_clamp),
        1.0 if activation_beta is None else float(activation_beta),
        -1.0 if activation_linear_beta is None else float(activation_linear_beta),
    ]
    module.get_function(name)(*kernel_args)


def _prefill_stage2(
    sym_buffer: MegaMoESymmBuffer,
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    num_tokens: int,
    num_mps: Optional[int] = None,
    fast_math: bool = True,
) -> None:
    """Run tiled FP8 stage 2 and remote-store partials to each source rank."""
    l2_weight, l2_scale = l2_weights
    assert 0 <= num_tokens <= sym_buffer.num_max_tokens_per_rank
    config = _prefill_kernel_config(sym_buffer, fast_math, num_mps)
    name, module = get_prefill_stage2_module(config)
    module.get_function(name)(
        sym_buffer.prefill_l2_acts,
        sym_buffer.prefill_l2_acts_sf,
        l2_weight,
        l2_scale,
        sym_buffer.handle.buffer_ptrs_tensor,
        sym_buffer.group.rank(),
    )


def _prefill_stage3(
    sym_buffer: MegaMoESymmBuffer,
    num_tokens: int,
    num_mps: Optional[int] = None,
    num_threads: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    fast_math: bool = True,
) -> torch.Tensor:
    """Reduce remote-stored stage 2 top-k partials into source-local BF16 output."""
    if output is None:
        output = torch.empty(
            (num_tokens, sym_buffer.hidden),
            dtype=torch.bfloat16,
            device=sym_buffer.buffer.device,
        )
    else:
        assert output.dtype == torch.bfloat16 and output.is_contiguous()
        assert tuple(output.shape) == (num_tokens, sym_buffer.hidden)
        assert output.device == sym_buffer.buffer.device
    if num_threads not in (None, PREFILL_STAGE3_NUM_THREADS):
        raise ValueError(
            f"remote-store prefill stage 3 reduce uses {PREFILL_STAGE3_NUM_THREADS} threads"
        )
    if num_mps is None:
        # Stage 3 reduce is latency-bound; 2x MP oversubscription measured fastest.
        resolved_num_mps = 2 * resolve_num_mps(sym_buffer.buffer.device)
    else:
        resolved_num_mps = resolve_num_mps(sym_buffer.buffer.device, num_mps)
    config = _prefill_kernel_config(sym_buffer, fast_math, resolved_num_mps)
    name, module = get_prefill_stage3_module(config)
    module.get_function(name)(
        output,
        sym_buffer.handle.buffer_ptrs_tensor,
        sym_buffer.group.rank(),
        num_tokens,
    )
    sym_buffer.y = output
    return output


def get_symm_buffer_for_mega_moe(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
    prefill_pool_capacity_ratio: float = 1.0,
    prefill_l1_ring: bool = False,
) -> MegaMoESymmBuffer:
    """Allocate a symmetric buffer with optional prefill pool reduction.

    Values below 1.0 reduce the L1/L2 activation pools. If the actual sum of
    256-row-aligned local expert counts exceeds the resulting capacity, kernels
    may access outside the allocation; callers opting in own that risk.

    ``prefill_l1_ring`` recycles the routed L1 activation pool through an
    automatically sized ring while keeping routing metadata at the worst-case
    size. It defaults to ``False``. The ring and a pool ratio below ``1.0`` are
    mutually exclusive; enable at most one memory-saving mode at a time.
    """
    assert use_fp8_dispatch
    num_max_tokens_per_rank = _align(num_max_tokens_per_rank, 256)
    return MegaMoESymmBuffer(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        activation=activation,
        prefill_pool_capacity_ratio=prefill_pool_capacity_ratio,
        prefill_l1_ring=prefill_l1_ring,
    )


def _check_grouped_tensor(name: str, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    assert tensor.dim() == 3, f"{name} must be a 3D grouped tensor"
    assert tensor.dtype == dtype, f"{name} must use dtype {dtype}, got {tensor.dtype}"
    assert tensor.is_contiguous(), f"{name} must be contiguous"


def _check_scale_tensor(
    name: str, tensor: torch.Tensor, expected_shape: Tuple[int, int, int]
) -> None:
    assert tensor.dim() == 3, f"{name} must be a 3D grouped scale tensor"
    assert tensor.dtype == torch.float32, (
        f"{name} must use float32 scales, got {tensor.dtype}"
    )
    assert tensor.is_contiguous(), f"{name} must be contiguous"
    assert tuple(tensor.shape) == expected_shape, (
        f"{name} shape must be {expected_shape}, got {tuple(tensor.shape)}"
    )


def _check_logical_weight_pair_for_transform(
    name: str,
    weights: Tuple[torch.Tensor, torch.Tensor],
    *,
    require_even_n: bool,
) -> Tuple[int, int, int]:
    assert isinstance(weights, tuple) and len(weights) == 2, (
        f"{name} must be a (weight, scale) tuple"
    )
    weight, scale = weights
    _check_grouped_tensor(f"{name}[0]", weight, torch.float8_e4m3fn)
    num_groups, n, k = weight.shape
    if require_even_n:
        assert n % 2 == 0, f"{name}[0] N dimension must be even for gate/up weights"
        assert n % 256 == 0, f"{name}[0] N dimension must be a multiple of 256"
    assert n % 128 == 0, f"{name}[0] N dimension must be a multiple of 128"
    assert k % 128 == 0, f"{name}[0] K dimension must be a multiple of 128"
    _check_scale_tensor(f"{name}[1]", scale, (num_groups, n // 128, k // 128))
    return num_groups, n, k


def _interleave_l1_weights(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    def interleave_weight(tensor: torch.Tensor, gran: int = 64) -> torch.Tensor:
        g, n, *rest = tensor.shape
        half = n // 2
        gate = tensor[:, :half].reshape(g, half // gran, gran, *rest)
        up = tensor[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.stack((gate, up), dim=2).reshape(g, n, *rest).contiguous()

    def expand_scale(scale: torch.Tensor, gran: int = 64) -> torch.Tensor:
        g, n_blocks, *rest = scale.shape
        half_blocks = n_blocks // 2
        chunks = torch.arange(0, half_blocks * 128, gran, device=scale.device)
        gate_idx = chunks // 128
        up_idx = gate_idx + half_blocks
        idx = torch.stack((gate_idx, up_idx), dim=1).reshape(-1)
        return scale.index_select(1, idx).contiguous()

    weight, scale = l1_weights
    return interleave_weight(weight), expand_scale(scale)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
]:
    _check_logical_weight_pair_for_transform(
        "l1_weights", l1_weights, require_even_n=True
    )
    l1_w, l1_sf = _interleave_l1_weights(l1_weights)
    if l2_weights is None:
        return l1_w, l1_sf
    _check_logical_weight_pair_for_transform(
        "l2_weights", l2_weights, require_even_n=False
    )
    return (l1_w, l1_sf), l2_weights


def _check_transformed_mega_moe_weights(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: MegaMoESymmBuffer,
) -> None:
    assert isinstance(l1_weights, tuple) and len(l1_weights) == 2
    assert isinstance(l2_weights, tuple) and len(l2_weights) == 2

    l1_weight, l1_scale = l1_weights
    l2_weight, l2_scale = l2_weights
    _check_grouped_tensor("l1_weights[0]", l1_weight, torch.float8_e4m3fn)
    _check_grouped_tensor("l2_weights[0]", l2_weight, torch.float8_e4m3fn)

    num_experts_per_rank = sym_buffer.num_experts // sym_buffer.group.size()
    expected_l1_shape = (
        num_experts_per_rank,
        sym_buffer.intermediate_hidden * 2,
        sym_buffer.hidden,
    )
    expected_l2_shape = (
        num_experts_per_rank,
        sym_buffer.hidden,
        sym_buffer.intermediate_hidden,
    )
    assert tuple(l1_weight.shape) == expected_l1_shape, (
        f"l1_weights[0] shape must be {expected_l1_shape}, got {tuple(l1_weight.shape)}"
    )
    assert tuple(l2_weight.shape) == expected_l2_shape, (
        f"l2_weights[0] shape must be {expected_l2_shape}, got {tuple(l2_weight.shape)}"
    )

    _check_scale_tensor(
        "l1_weights[1]",
        l1_scale,
        (
            num_experts_per_rank,
            (sym_buffer.intermediate_hidden * 2) // 64,
            sym_buffer.hidden // 128,
        ),
    )
    _check_scale_tensor(
        "l2_weights[1]",
        l2_scale,
        (
            num_experts_per_rank,
            sym_buffer.hidden // 128,
            sym_buffer.intermediate_hidden // 128,
        ),
    )


# Open issue: repeated decode calls can show non-bitwise output differences for
# small hidden/intermediate shapes (observed at H=256/I=256 with SwiGLU).
# Production H=7168/I=2048 decode and the 4K/8K Prefill path are unaffected in
# current validation; keep this note until the small-shape Stage1 behavior is
# root-caused and fixed.
def _run_decode_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: MegaMoESymmBuffer,
    num_tokens: int,
    activation: str,
    activation_clamp: Optional[float],
    activation_beta: Optional[float],
    activation_linear_beta: Optional[float],
    fast_math: bool,
) -> None:
    config = MegaMoEKernelConfig(
        **_kernel_config_kwargs(sym_buffer, fast_math, resolve_num_mps(y.device))
    )
    activation = _validate_activation_options(
        activation, activation_clamp, activation_beta, activation_linear_beta
    )
    stage1_name, stage1_module = get_fp8_fp8_mega_moe_stage1_module(
        config,
        activation=activation,
        use_situ_linear=activation_linear_beta is not None,
    )
    stage1_module.get_function(stage1_name)(
        sym_buffer.decode_l1_acts,
        sym_buffer.decode_l1_acts_sf,
        sym_buffer.decode_l2_acts,
        sym_buffer.decode_l2_acts_sf,
        l1_weights[0],
        l1_weights[1],
        sym_buffer.handle.buffer_ptrs_tensor,
        sym_buffer.group.rank(),
        num_tokens,
        1.0 if activation_beta is None else float(activation_beta),
        -1.0 if activation_linear_beta is None else float(activation_linear_beta),
    )

    stage2_name, stage2_module = get_fp8_fp8_mega_moe_stage2_module(config)
    stage2_module.get_function(stage2_name)(
        sym_buffer.decode_l2_acts,
        sym_buffer.decode_l2_acts_sf,
        l2_weights[0],
        l2_weights[1],
        y,
        sym_buffer.handle.buffer_ptrs_tensor,
        sym_buffer.group.rank(),
        num_tokens,
    )


def _run_prefill_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: MegaMoESymmBuffer,
    num_tokens: int,
    activation: str,
    activation_clamp: Optional[float],
    activation_beta: Optional[float],
    activation_linear_beta: Optional[float],
    fast_math: bool,
) -> None:
    _prefill_stage1(
        sym_buffer,
        l1_weights,
        num_tokens,
        fast_math=fast_math,
        activation_clamp=activation_clamp,
        activation=activation,
        activation_beta=activation_beta,
        activation_linear_beta=activation_linear_beta,
    )
    _prefill_stage2(
        sym_buffer,
        l2_weights,
        num_tokens,
        fast_math=fast_math,
    )
    _prefill_stage3(
        sym_buffer,
        num_tokens,
        output=y,
        fast_math=fast_math,
    )


def fp8_fp8_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: MegaMoESymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    activation_beta: Optional[float] = None,
    activation_linear_beta: Optional[float] = None,
) -> None:
    """Run M32 decode for <=128 tokens, otherwise split-prefill stage 1/2/3."""
    assert isinstance(sym_buffer, MegaMoESymmBuffer)
    activation = _normalize_activation(activation)
    assert y.dim() == 2
    assert y.dtype == torch.bfloat16
    assert y.is_contiguous()
    assert y.size(1) == sym_buffer.hidden
    assert y.device == sym_buffer.buffer.device
    _check_transformed_mega_moe_weights(l1_weights, l2_weights, sym_buffer)
    if recipe != (1, 1, 32):
        raise ValueError("MegaMoE uses fixed M32 decode and M256 prefill schedules")
    num_tokens = int(y.size(0))
    if num_tokens <= 0 or num_tokens > sym_buffer.num_max_tokens_per_rank:
        raise ValueError("output token count is outside the symmetric-buffer capacity")
    _ = cumulative_local_expert_recv_stats
    if num_tokens <= 128:
        _run_decode_mega_moe(
            y,
            l1_weights,
            l2_weights,
            sym_buffer,
            num_tokens,
            activation,
            activation_clamp,
            activation_beta,
            activation_linear_beta,
            fast_math,
        )
        sym_buffer.y = y
        return

    _run_prefill_mega_moe(
        y,
        l1_weights,
        l2_weights,
        sym_buffer,
        num_tokens,
        activation,
        activation_clamp,
        activation_beta,
        activation_linear_beta,
        fast_math,
    )


def get_token_alignment_for_mega_moe() -> int:
    return 256


__all__ = [
    "MegaMoESymmBuffer",
    "MegaMoEPrefillSymmBuffer",
    "PrefillSymmBuffer",
    "SymmBuffer",
    "fp8_fp8_mega_moe",
    "get_symm_buffer_for_mega_moe",
    "get_token_alignment_for_mega_moe",
    "transform_weights_for_mega_moe",
]
