"""Tests for the native mamba/SSU family.

One module, eight grouped sections:

1. the frozen ``selective_state_update`` contract and the one-token kernel on
   MUSA hardware;
2. the packed multi-token (MTP) SSU path, including the cross-check against the
   stock Triton SSU it replaces;
3. the multi-token SSU oracle's semantics;
4. the SSD prefill stages: chunk cumsum, chunk state, state passing and BMM;
5. the SSD chunk scan;
6. the packed SSD orchestration end to end;
7. the SSD BMM kernel's strided-input contract on MUSA hardware;
8. the same strided-input contract for the rest of the TileLang family.

Sections 3-6 are **CPU-only**: they need no MUSA device and no TileLang kernel
module, so they run anywhere torch does. That is a structural property of this
file, not a coincidence -- those sections sit below the CPU-only divider and
reach no device-only import (``mate.testing.operators.mamba`` is a plain-torch
oracle module, and ``mate.mamba``'s TileLang kernel imports are lazy by design).
Every test that does need a device carries the
``supported_musa_compute_capability`` marker and skips elsewhere.

The oracle the device tests compare against lives in
``mate.testing.operators.mamba``, beside the other operator test harnesses.
"""

from __future__ import annotations

import importlib.util
import inspect
import itertools
import os
import pathlib
import re
import types

import pytest
import torch

import mate.mamba
from mate.testing import supported_musa_compute_capability
from mate.testing.operators.mamba import (
    selective_state_update_multi_token_reference,
    selective_state_update_one_token_reference,
    ssd_bmm_reference,
    ssd_chunk_cumsum_reference,
    ssd_chunk_scan_reference,
    ssd_chunk_state_reference,
    ssd_state_passing_reference,
)

# ---------------------------------------------------------------------------
# 1. Native one-token selective state update: the frozen contract and the kernel
#
# The contract test pins the frozen argument list shared with the FlashInfer-shaped
# compatibility wrapper. The numeric tests compare the tilelang kernel against the
# fp32 oracle on real MUSA hardware, and the gap tests assert that unsupported
# paths fail loudly instead of silently narrowing semantics.
# ---------------------------------------------------------------------------

# Frozen argument list: the wrapper forwards exactly these, in this order.
SSU_CONTRACT_PARAMETERS = (
    "state",
    "x",
    "dt",
    "A",
    "B",
    "C",
    "D",
    "z",
    "dt_bias",
    "dt_softplus",
    "state_batch_indices",
    "pad_slot_id",
    "state_scale",
    "out",
    "disable_state_update",
    "intermediate_states_buffer",
    "intermediate_state_indices",
    "intermediate_state_scales",
    "rand_seed",
    "philox_rounds",
    "cache_steps",
    "algorithm",
    "dst_state_batch_indices",
    "cu_seqlens",
    "num_accepted_tokens",
    "backend",
)

SSU_HEADS = 8
SSU_DIM = 8
SSU_DSTATE = 128
SSU_GROUPS = 2
SSU_BATCH = 2
SSU_SLOTS = 4


def test_selective_state_update_contract():
    parameters = inspect.signature(mate.mamba.selective_state_update).parameters
    assert tuple(parameters) == SSU_CONTRACT_PARAMETERS
    assert parameters["state"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["pad_slot_id"].default == -1
    assert parameters["philox_rounds"].default == 10
    assert parameters["algorithm"].default == "auto"
    assert parameters["backend"].default == "auto"


def _ssu_device() -> torch.device:
    return torch.device("musa")


def _ssu_case(
    *,
    with_D: bool = True,
    with_z: bool = False,
    with_bias: bool = True,
    softplus: bool = False,
    disable_state_update: bool = False,
    pad_slot_id: int = -1,
    src_slots: tuple[int, ...] = (0, 1),
    dst_slots: tuple[int, ...] | None = None,
):
    device = _ssu_device()
    torch.manual_seed(0)
    state = torch.randn(
        SSU_SLOTS, SSU_HEADS, SSU_DIM, SSU_DSTATE, dtype=torch.float32, device=device
    ).to(torch.float16)
    x = torch.randn(SSU_BATCH, SSU_HEADS, SSU_DIM, dtype=torch.float32, device=device).to(
        torch.bfloat16
    )
    dt = torch.rand(SSU_BATCH, SSU_HEADS, SSU_DIM, dtype=torch.float32, device=device) * 0.1
    A = -torch.rand(SSU_HEADS, dtype=torch.float32, device=device)
    B = torch.randn(SSU_BATCH, SSU_GROUPS, SSU_DSTATE, dtype=torch.float32, device=device).to(
        torch.bfloat16
    )
    C = torch.randn(SSU_BATCH, SSU_GROUPS, SSU_DSTATE, dtype=torch.float32, device=device).to(
        torch.bfloat16
    )
    D = torch.rand(SSU_HEADS, dtype=torch.float32, device=device) if with_D else None
    dt_bias = (
        torch.rand(SSU_HEADS, dtype=torch.float32, device=device) if with_bias else None
    )
    z = (
        torch.randn(SSU_BATCH, SSU_HEADS, SSU_DIM, dtype=torch.float32, device=device).to(
            torch.bfloat16
        )
        if with_z
        else None
    )
    src = torch.tensor(src_slots, dtype=torch.int32, device=device)
    dst = (
        torch.tensor(dst_slots, dtype=torch.int32, device=device) if dst_slots else None
    )
    return {
        "state": state,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "z": z,
        "dt_bias": dt_bias,
        "dt_softplus": softplus,
        "src_slots": src,
        "dst_slots": dst,
        "pad_slot_id": pad_slot_id,
        "disable_state_update": disable_state_update,
    }


def _ssu_run_kernel(case, *, out=None, **kwargs):
    # `kwargs` carry `algorithm`/`backend`, which only the vocabulary tests set: every
    # other test wants the production call shape, i.e. both left at their defaults.
    return mate.mamba.selective_state_update(
        case["state"],
        case["x"],
        case["dt"],
        case["A"],
        case["B"],
        case["C"],
        case["D"],
        z=case["z"],
        dt_bias=case["dt_bias"],
        dt_softplus=case["dt_softplus"],
        state_batch_indices=case["src_slots"],
        dst_state_batch_indices=case["dst_slots"],
        pad_slot_id=case["pad_slot_id"],
        disable_state_update=case["disable_state_update"],
        out=out,
        **kwargs,
    )


def _ssu_run_reference(case):
    return selective_state_update_one_token_reference(
        case["state"],
        case["x"],
        case["dt"],
        case["A"],
        case["B"],
        case["C"],
        case["D"],
        case["z"],
        case["dt_bias"],
        case["dt_softplus"],
        case["src_slots"],
        case["dst_slots"],
        case["pad_slot_id"],
        case["disable_state_update"],
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("with_D", [True, False])
@pytest.mark.parametrize("with_z", [True, False])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("softplus", [True, False])
@torch.inference_mode
def test_one_token_matches_reference(with_D, with_z, with_bias, softplus):
    case = _ssu_case(with_D=with_D, with_z=with_z, with_bias=with_bias, softplus=softplus)
    expected_state = case["state"].clone()
    expected = _ssu_run_reference({**case, "state": expected_state})
    actual_state = case["state"].clone()
    actual = _ssu_run_kernel({**case, "state": actual_state})

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-2, atol=1e-3)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_separate_destination_slots_and_padding():
    # Source slot 2 is the padding sentinel and slot 3 is an untouched source.
    case = _ssu_case(src_slots=(2, 3), dst_slots=(0, 1), pad_slot_id=2)
    original = case["state"].clone()
    expected_state = original.clone()
    expected = _ssu_run_reference({**case, "state": expected_state})
    actual_state = original.clone()
    actual = _ssu_run_kernel({**case, "state": actual_state})

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-2, atol=1e-3)
    # The padded source contributes a zero state and the untouched slot is intact.
    torch.testing.assert_close(actual_state[3], original[3])


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_disable_state_update_keeps_the_pool():
    case = _ssu_case(disable_state_update=True)
    original = case["state"].clone()
    expected = _ssu_run_reference({**case, "state": original.clone()})
    actual_state = original.clone()
    actual = _ssu_run_kernel({**case, "state": actual_state})

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_state, original)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_prewarm_then_run_uses_the_compiled_kernel():
    from mate.mamba import prewarm_selective_state_update

    prewarm_selective_state_update(
        state_dtype=torch.float16,
        io_dtype=torch.bfloat16,
        batch=SSU_BATCH,
        heads=SSU_HEADS,
        dim=SSU_DIM,
        dstate=SSU_DSTATE,
        groups=SSU_GROUPS,
        slot_dtype=torch.int32,
    )
    case = _ssu_case()
    expected = _ssu_run_reference({**case, "state": case["state"].clone()})
    actual = _ssu_run_kernel(case)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_out_buffer_is_written_in_place():
    case = _ssu_case()
    out = torch.empty(SSU_BATCH, SSU_HEADS, SSU_DIM, dtype=torch.bfloat16, device=_ssu_device())
    returned = _ssu_run_kernel(case, out=out)
    assert returned is out
    assert torch.isfinite(out).all()


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"rand_seed": torch.zeros(1, dtype=torch.int32)}, "stochastic rounding"),
        ({"state_scale": torch.ones(SSU_SLOTS, dtype=torch.float32)}, "quantized state"),
        ({"cache_steps": 4}, "intermediate"),
        # A packed call is no longer refused: one sequence of two tokens, which the
        # packed kernel handles (the numeric check lives in the packed MTP
        # section above).
        # The metadata tensors live on the device like every other input; a CPU
        # `cu_seqlens` in an otherwise device-side case fails for the wrong reason.
        (
            {"cu_seqlens": torch.tensor([0, 2], dtype=torch.int32, device=_ssu_device())},
            None,
        ),
        # vLLM's SSU dispatch forwards the slot-table width as `cache_steps` whenever
        # it passes cu_seqlens, including the plain one-token-per-sequence step. That
        # must keep running -- refusing it would take the whole
        # `--mamba-backend flashinfer` arm off the native path.
        (
            {
                "cache_steps": 2,
                "cu_seqlens": torch.tensor(
                    [0, 1, 2], dtype=torch.int32, device=_ssu_device()
                ),
            },
            None,
        ),
    ],
)
def test_unsupported_paths_fail_loudly(overrides, message):
    case = _ssu_case()
    kwargs = {
        "D": case["D"],
        "z": case["z"],
        "dt_bias": case["dt_bias"],
        "pad_slot_id": case["pad_slot_id"],
        **overrides,
    }
    if message is None:
        # A no-op override set must still run, so the test proves the guard list
        # is not simply rejecting everything.
        mate.mamba.selective_state_update(
            case["state"],
            case["x"],
            case["dt"],
            case["A"],
            case["B"],
            case["C"],
            **kwargs,
        )
        return
    with pytest.raises(NotImplementedError, match=message):
        mate.mamba.selective_state_update(
            case["state"],
            case["x"],
            case["dt"],
            case["A"],
            case["B"],
            case["C"],
            **kwargs,
        )


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_channel_wise_D_is_applied_per_channel():
    """A per-(head, dim) ``D`` is the shape vLLM's decode step passes, and it is used.

    The fp32 oracle models a per-head ``D`` only, so this is checked in two directions:
    channels that agree must reproduce the per-head result, and moving one channel must
    move the output -- a kernel that read a single column would pass the first half alone.
    """
    case = _ssu_case()
    per_head = case["D"]
    flat = per_head[:, None].repeat(1, SSU_DIM)

    # One case per run: the call updates the destination slots of `state` in place, so a
    # reused case would hand the next call an already-updated state.
    baseline = _ssu_run_kernel({**case, "D": per_head})
    broadcast = _ssu_run_kernel({**_ssu_case(), "D": flat})
    torch.testing.assert_close(broadcast, baseline, rtol=2e-2, atol=2e-2)

    varying = flat.clone()
    varying[:, -1] += 1.0
    moved = _ssu_run_kernel({**_ssu_case(), "D": varying})
    assert not torch.allclose(moved, baseline)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_ssu_algorithm_accepts_the_consumer_vocabulary():
    """vLLM forwards its ``MambaSSUAlgorithm`` choice; one implementation answers all of it.

    Every value that vocabulary can carry must run and agree, and a value outside it must
    still fail loudly: the argument is validated, it just does not select a kernel.
    """
    baseline = _ssu_run_kernel(_ssu_case())
    for algorithm in ("auto", "simple", "vertical", "horizontal"):
        actual = _ssu_run_kernel(_ssu_case(), algorithm=algorithm)
        torch.testing.assert_close(actual, baseline, rtol=2e-2, atol=2e-2)

    with pytest.raises(ValueError, match="implements"):
        _ssu_run_kernel(_ssu_case(), algorithm="bogus")


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_ssu_backend_names_the_implementation():
    """``tilelang`` names this kernel family; a consumer's dispatch name is not a backend.

    ``--mamba-backend flashinfer`` selects us where vLLM dispatches and never reaches this
    argument, so accepting ``flashinfer`` here would only hide a wrong call.
    """
    baseline = _ssu_run_kernel(_ssu_case())
    torch.testing.assert_close(
        _ssu_run_kernel(_ssu_case(), backend="tilelang"), baseline, rtol=2e-2, atol=2e-2
    )
    for backend in ("flashinfer", "musa", "native"):
        with pytest.raises(ValueError, match="accepts backends"):
            _ssu_run_kernel(_ssu_case(), backend=backend)


# ---------------------------------------------------------------------------
# 2. Packed multi-token (MTP) SSU path
#
# The packed kernel is what speculative decoding needs from the decode side: several
# tokens per sequence in one call, one destination state slot per speculative
# position, and a read slot seeded by the acceptance count. These tests compare it
# against ``mate.testing.operators.mamba`` over the call shapes the serving path can
# produce -- tokens per sequence 1..7, batch 1/2/8, every acceptance count in
# ``0..T``, shared and distinct slot tables, the flat ``[rows]`` tables vLLM's MTP6
# call passes, and null block entries on either side.
#
# The comparison is per element and against the oracle rather than against recorded
# numbers, because a state that is merely *close* is a behaviour change here: under
# MTP the state feeds the draft model, so it moves acceptance length and the served
# output. Empty sequences and variable-length packing are covered too -- the packed
# row layout is what makes an empty sequence reachable at all.
# ---------------------------------------------------------------------------

MTP_HEADS = 8
MTP_DIM = 8
MTP_DSTATE = 128
MTP_GROUPS = 2
MTP_IO_DTYPE = torch.bfloat16
MTP_STATE_DTYPE = torch.float16
MTP_SOFTPLUS = True
#: vLLM's ``NULL_BLOCK_ID``, which is also what its backend dispatch passes as the
#: pad slot; MATE refuses any other sentinel, so the tests use the serving one.
MTP_PAD = 0


def _mtp_device() -> torch.device:
    return torch.device("musa")


def _mtp_row_base(steps: list[int]) -> list[int]:
    """First packed row of each sequence."""
    return [0, *itertools.accumulate(steps)][: len(steps)]


def _mtp_case(
    steps: list[int],
    *,
    accepted: list[int] | None = None,
    heads: int = MTP_HEADS,
    dim: int = MTP_DIM,
    dstate: int = MTP_DSTATE,
    groups: int = MTP_GROUPS,
    state_dtype: torch.dtype = MTP_STATE_DTYPE,
    same_slots: bool = False,
    null_src: tuple[tuple[int, int], ...] = (),
    null_dst: tuple[tuple[int, int], ...] = (),
    dense: bool = False,
    disable_state_update: bool = False,
    flat: bool = False,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Build one packed decode call plus the oracle's description of the same call.

    Every ``(sequence, step)`` pair owns its own slot row, so a write to the wrong
    slot shows up as a large difference instead of a coincidence. ``null_src`` and
    ``null_dst`` poke the sentinel into chosen table entries. ``dense`` describes the
    same call through the 4-D ``[batch, steps, heads, dim]`` layout, and ``flat``
    describes it the way vLLM's MTP6 call does: one 1-D table with an entry per packed
    row, addressed as ``sequence + step``. A flat table only names every
    ``(sequence, step)`` pair separately when there is one sequence, so flat cases
    with several multi-token sequences are left to the non-spec regime, where only
    entry ``sequence`` is touched.
    """
    device = _mtp_device() if device is None else device
    # A CPU generator keeps the inputs of a case identical across runs, devices and
    # dtypes; the refusal tests then build the same tensors without a MUSA device.
    generator = torch.Generator(device="cpu").manual_seed(0)
    batch = len(steps)
    rows = sum(steps)
    width = max(steps)
    # Row 0 stays the sentinel; a flat table needs one entry per packed row, a 2-D
    # table one per (sequence, step) pair.
    slot_rows = 1 + 2 * (rows if flat else batch * width)

    def _rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.float32, generator=generator).to(device)

    state = (_rand(slot_rows, heads, dim, dstate) * 0.5).to(state_dtype)
    x = _rand(rows, heads, dim).to(MTP_IO_DTYPE)
    dt = (
        torch.rand(rows, heads, 1, dtype=torch.float32, generator=generator).to(device)
        * 0.3
        + 0.01
    )
    A = -torch.rand(heads, 1, dtype=torch.float32, generator=generator).to(device) - 0.1
    B = _rand(rows, groups, dstate).to(MTP_IO_DTYPE)
    C = _rand(rows, groups, dstate).to(MTP_IO_DTYPE)
    D = torch.rand(heads, 1, dtype=torch.float32, generator=generator).to(device)
    dt_bias = _rand(heads, 1) * 0.1
    z = _rand(rows, heads, dim).to(MTP_IO_DTYPE)

    table_elems = rows if flat else batch * width
    src_slots = torch.arange(1, 1 + table_elems, dtype=torch.int32, device=device)
    dst_slots = (
        src_slots.clone()
        if same_slots
        else torch.arange(
            1 + table_elems, 1 + 2 * table_elems, dtype=torch.int32, device=device
        )
    )
    if not flat:
        src_slots = src_slots.reshape(batch, width).contiguous()
        dst_slots = dst_slots.reshape(batch, width).contiguous()

    def _poke(table: torch.Tensor, entries: tuple[tuple[int, int], ...]) -> None:
        for sequence, step in entries:
            if flat:
                # A flat table is addressed by position: both coordinates select one
                # entry, which is why a flat multi-sequence table has to stay out of
                # the spec regime -- sequence 1's first token and sequence 0's second
                # would be the same slot.
                table[sequence + step] = MTP_PAD
            else:
                table[sequence, step] = MTP_PAD

    _poke(src_slots, null_src)
    _poke(dst_slots, null_dst)

    return {
        "state": state,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "z": z,
        "dt_bias": dt_bias,
        "src_slots": src_slots,
        "dst_slots": dst_slots,
        "cu_seqlens": torch.tensor(
            [0, *itertools.accumulate(steps)], dtype=torch.int32, device=device
        ),
        "num_accepted_tokens": (
            None
            if accepted is None
            else torch.tensor(accepted, dtype=torch.int32, device=device)
        ),
        "steps": steps,
        "row_base": _mtp_row_base(steps),
        "dense": dense,
        "disable_state_update": disable_state_update,
    }


def _mtp_dense_operand(case: dict[str, object], name: str) -> torch.Tensor:
    """View a packed operand as the dense ``[batch, steps, ...]`` form of itself."""
    tensor = case[name]
    steps = case["steps"]
    return tensor.view(len(steps), max(steps), *tensor.shape[1:])


def _mtp_run_kernel(case: dict[str, object], *, state: torch.Tensor | None = None):
    x = case["x"]
    dt = case["dt"]
    B = case["B"]
    C = case["C"]
    z = case["z"]
    cu_seqlens = case["cu_seqlens"]
    if case["dense"]:
        x = _mtp_dense_operand(case, "x")
        dt = _mtp_dense_operand(case, "dt")
        B = _mtp_dense_operand(case, "B")
        C = _mtp_dense_operand(case, "C")
        z = _mtp_dense_operand(case, "z")
        cu_seqlens = None
    return mate.mamba.selective_state_update(
        case["state"] if state is None else state,
        x,
        dt,
        case["A"],
        B,
        C,
        case["D"],
        z=z,
        dt_bias=case["dt_bias"],
        dt_softplus=MTP_SOFTPLUS,
        state_batch_indices=case["src_slots"],
        dst_state_batch_indices=case["dst_slots"],
        pad_slot_id=MTP_PAD,
        disable_state_update=case["disable_state_update"],
        num_accepted_tokens=case["num_accepted_tokens"],
        cu_seqlens=cu_seqlens,
    )


def _mtp_run_reference(case: dict[str, object], *, state: torch.Tensor | None = None):
    return selective_state_update_multi_token_reference(
        case["state"] if state is None else state,
        case["x"],
        case["dt"],
        case["A"],
        case["B"],
        case["C"],
        case["D"],
        case["z"],
        case["dt_bias"],
        MTP_SOFTPLUS,
        case["src_slots"],
        case["dst_slots"],
        MTP_PAD,
        case["disable_state_update"],
        case["steps"],
        case["row_base"],
        case["num_accepted_tokens"],
    )


def _mtp_compare(case: dict[str, object]) -> None:
    expected_state = case["state"].clone()
    expected = _mtp_run_reference(case, state=expected_state)
    actual_state = case["state"].clone()
    actual = _mtp_run_kernel(case, state=actual_state)

    # A dense call returns its own rank; the oracle speaks the flattened layout.
    actual = actual.reshape(expected.shape)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-2, atol=1e-3)


#: (tokens per sequence, acceptance count) for every value in 0..T.
MTP_TOKENS_AND_ACCEPTED = [
    (steps, accepted) for steps in range(1, 8) for accepted in range(steps + 1)
]


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("steps, accepted", MTP_TOKENS_AND_ACCEPTED)
def test_uniform_packed_sequences_match_reference(batch, steps, accepted):
    """The MTP6 shape family: uniform sequences, one slot table for both sides.

    ``accepted == 0`` covers the floor of the ``max(count - 1, 0)`` read rule, and
    the identity form -- the caller passing the read table as the destination table,
    which is what the mixer does with prefix caching off -- is what the call itself
    uses here.
    """
    case = _mtp_case([steps] * batch, accepted=[accepted] * batch, same_slots=True)
    case["dst_slots"] = case["src_slots"]
    _mtp_compare(case)


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("steps, accepted", MTP_TOKENS_AND_ACCEPTED)
def test_distinct_slot_tables_match_reference(batch, steps, accepted):
    """Prefix caching passes distinct read and destination tables.

    Every (sequence, step) pair then writes a row no other pair touches, so the
    oracle pins the write set exactly: a missed or extra store cannot hide.
    """
    _mtp_compare(_mtp_case([steps] * batch, accepted=[accepted] * batch))


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("steps", [1, 3, 7])
def test_null_block_entries_match_reference(batch, steps):
    """A null read slot reads as zero state; a null write slot is never written."""
    # Nulls on the first readable column: the seeded read when nothing was accepted.
    _mtp_compare(
        _mtp_case(
            [steps] * batch,
            accepted=[1] * batch,
            null_src=tuple((b, 0) for b in range(batch)),
            null_dst=tuple((b, steps - 1) for b in range(batch)),
        )
    )
    # Nulls on the column the acceptance counts select, with one shared table.
    _mtp_compare(
        _mtp_case(
            [steps] * batch,
            accepted=[steps] * batch,
            same_slots=True,
            null_src=tuple((b, steps - 1) for b in range(batch)),
            null_dst=tuple((b, 0) for b in range(batch)),
        )
    )


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize(
    "steps, accepted",
    [
        ([0, 3, 0, 4], [0, 2, 0, 4]),
        ([7, 1], [5, 1]),
        ([1, 6, 2, 5], [1, 3, 2, 5]),
        ([0, 1], [0, 1]),
    ],
)
def test_variable_length_sequences_match_reference(steps, accepted):
    """Packed rows of different lengths, including sequences of no tokens."""
    _mtp_compare(_mtp_case(steps, accepted=accepted))
    _mtp_compare(_mtp_case(steps, accepted=accepted, same_slots=True))


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("steps", [1, 7])
@pytest.mark.parametrize("accepted", [None, "full"])
def test_dense_four_dimensional_call_matches_reference(batch, steps, accepted):
    """The dense ``[batch, steps, heads, dim]`` form runs the same recurrence.

    That is the layout FlashInfer documents for multi-token SSU; the entry point
    flattens it into packed rows with the regular query starts that describe it.
    """
    _mtp_compare(
        _mtp_case(
            [steps] * batch,
            accepted=None if accepted is None else [steps] * batch,
            dense=True,
            same_slots=True,
        )
    )


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accepted", [None, [2, 2]])
def test_state_dtypes_match_reference(state_dtype, accepted):
    """The pool dtype only changes where the state is rounded, not the recurrence."""
    _mtp_compare(_mtp_case([2, 2], accepted=accepted, state_dtype=state_dtype))


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_production_shape_matches_reference():
    """Nemotron-3.5's MTP6 decode shape: B=8, T=7, H=64, D=64, N=128, G=8."""
    for accepted in (1, 4, 7):
        _mtp_compare(
            _mtp_case(
                [7] * 8,
                accepted=[accepted] * 8,
                heads=64,
                dim=64,
                dstate=128,
                groups=8,
                same_slots=True,
            )
        )


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("identical", [True, False])
@pytest.mark.parametrize("accepted", range(1, 8))
def test_flat_production_slot_tables_match_reference(accepted, identical):
    """The call ``--mamba-backend flashinfer`` dies on today: vLLM's MTP6 decode.

    ``mamba_mixer2`` passes ``x``/``dt``/``out`` of ``[total_tokens, heads, dim]``,
    flat 1-D ``state_batch_indices``/``dst_state_batch_indices``, an acceptance count
    and ``query_start_loc`` for one sequence of seven tokens at ``--max-num-seqs 1``.
    The tables carry one entry per packed row, so the accepted position is entry
    ``accepted - 1`` and token ``t`` publishes to entry ``t`` -- reading a 2-D table's
    ``[sequence, step]`` instead would decode from the wrong state, and refusing the
    flat form at all is what kept this configuration from starting.
    """
    case = _mtp_case([7], accepted=[accepted], flat=True)
    if identical:
        # The mixer hands the same table over as both source and destination when
        # prefix caching is off, so the chain moves forward in place.
        case["dst_slots"] = case["src_slots"]
    _mtp_compare(case)


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("steps", [1, 3, 7])
def test_flat_slot_tables_carry_the_plain_packed_call(batch, steps):
    """Without acceptance counts a flat table is one entry per sequence.

    Only entry ``sequence`` is read and written, so several multi-token sequences
    stay distinguishable -- which pins the sequence stride of the flat addressing.
    """
    _mtp_compare(_mtp_case([steps] * batch, flat=True))
    _mtp_compare(_mtp_case([steps] * batch, same_slots=True, flat=True))


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("null_entries", [((0, 0),), ((0, 6),)])
def test_null_entries_in_flat_slot_tables_match_reference(null_entries):
    """The sentinel reads as a zero state and is never written, in the flat form too."""
    _mtp_compare(
        _mtp_case(
            [7],
            accepted=[1],
            flat=True,
            null_src=null_entries,
            null_dst=null_entries,
        )
    )


@supported_musa_compute_capability([31])
@torch.inference_mode
@pytest.mark.parametrize("flat", [True, False])
def test_absent_destination_table_publishes_to_the_read_table(flat):
    """With no destination table Triton aliases the read table onto the write side.

    Every token then publishes to its own entry of that table rather than to the slot
    the sequence started from, which is the behaviour to match: the read slot is only
    the *initial* state.
    """
    case = _mtp_case([4], accepted=[2], flat=flat)
    case["dst_slots"] = None
    expected_state = case["state"].clone()
    _mtp_run_reference(case, state=expected_state)
    # Entries 0 and 1 are the ones the accepted position and the token loop reach, so
    # a store that ignored the table would leave them at their initial values.
    assert not torch.equal(expected_state[1:3], case["state"][1:3])
    _mtp_compare(case)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_disable_state_update_keeps_the_packed_pool():
    case = _mtp_case([3, 3], accepted=[2, 3], disable_state_update=True)
    original = case["state"].clone()
    expected = _mtp_run_reference({**case, "state": original.clone()})
    actual = _mtp_run_kernel(case, state=original.clone())
    torch.testing.assert_close(
        actual.reshape(expected.shape), expected, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(case["state"], original)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_dispatch_keeps_the_one_token_kernel_for_the_plain_step(monkeypatch):
    """A plain decode step must not be sent through the packed kernel.

    vLLM's Mamba2 mixer makes this call on every decode step of every layer, and it
    passes query starts plus one slot per sequence. That shape has to keep reaching
    the kernel built for it; the packed kernel is for the multi-token calls.
    """
    calls: list[str] = []
    one_token = mate.mamba._ssu_launch()
    packed = mate.mamba._ssu_packed_launch()

    def _record(name: str, launcher):
        def _launch(*args, **kwargs):
            calls.append(name)
            return launcher(*args, **kwargs)

        return _launch

    monkeypatch.setattr(
        mate.mamba, "_ssu_launch", lambda: _record("one_token", one_token)
    )
    monkeypatch.setattr(
        mate.mamba, "_ssu_packed_launch", lambda: _record("packed", packed)
    )

    plain = _mtp_case([1, 1], same_slots=True)
    plain["src_slots"] = plain["src_slots"].reshape(-1)
    plain["dst_slots"] = None
    _mtp_run_kernel(plain)
    assert calls == ["one_token"], "the plain decode step left the one-token kernel"

    calls.clear()
    # One speculative step later: acceptance counts plus a per-token destination
    # table, so the state has to start from the accepted position.
    mtp = _mtp_case([1, 1], accepted=[1, 1], same_slots=True)
    _mtp_run_kernel(mtp)
    assert calls == ["packed"], "a one-token MTP call reached the one-token kernel"

    calls.clear()
    # The dense single-token form with one slot per sequence is the plain call too.
    dense = _mtp_case([1, 1], dense=True, same_slots=True)
    dense["src_slots"] = dense["src_slots"].reshape(-1)
    dense["dst_slots"] = None
    _mtp_run_kernel(dense)
    assert calls == ["one_token"], "the dense single-token call left the fast path"


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_one_token_call_agrees_across_the_two_kernels():
    """A one-token MTP call takes the packed kernel and must still match the fast path.

    That is the boundary the dispatch rule moves calls across, and the only shape
    where the two kernels can be compared directly.
    """
    vector = _mtp_case([1, 1], same_slots=True)
    vector["src_slots"] = vector["src_slots"].reshape(-1)
    vector["dst_slots"] = None
    table = _mtp_case([1, 1], accepted=[1, 1], same_slots=True)

    vector_state = vector["state"].clone()
    vector_out = _mtp_run_kernel(vector, state=vector_state)
    table_state = table["state"].clone()
    table_out = _mtp_run_kernel(table, state=table_state)

    torch.testing.assert_close(table_out, vector_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(table_state, vector_state, rtol=1e-2, atol=1e-3)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_prewarm_packed_then_run_uses_the_compiled_kernel():
    from mate.mamba import prewarm_selective_state_update

    prewarm_selective_state_update(
        state_dtype=MTP_STATE_DTYPE,
        io_dtype=MTP_IO_DTYPE,
        batch=2,
        heads=MTP_HEADS,
        dim=MTP_DIM,
        dstate=MTP_DSTATE,
        groups=MTP_GROUPS,
        slot_dtype=torch.int32,
        packed=True,
    )
    _mtp_compare(_mtp_case([3, 3], accepted=[2, 1]))


def test_packed_contract_is_validated_before_the_kernel():
    """A call that cannot express the packed contract fails instead of guessing.

    These run on CPU tensors: the validation happens before any kernel is built, so
    they cost nothing and they pin the errors a caller sees. The Triton kernel
    asserts the first two of them as well.
    """
    # A packed multi-token call with neither query starts nor a table that describes
    # the row split cannot be decoded.
    without_starts = _mtp_case([2, 2], device=torch.device("cpu"))
    without_starts["cu_seqlens"] = None
    with pytest.raises(ValueError, match="cu_seqlens"):
        _mtp_run_kernel(without_starts)

    # Acceptance needs a read mechanism, and that mechanism is the read table.
    without_table = _mtp_case([2, 2], accepted=[1, 1], device=torch.device("cpu"))
    without_table["src_slots"] = None
    with pytest.raises(ValueError, match="state_batch_indices"):
        _mtp_run_kernel(without_table)


def test_flat_slot_tables_are_normalized_rather_than_refused():
    """The tables are addressed by strides, so a 1-D table is a call, not an error.

    The serving path passes flat tables together with an acceptance count, and the
    kernel reaches position ``t`` of sequence ``b`` at entry ``b + t``: the strides the
    normalizer hands over are what say so. A 2-D table keeps its own row stride, so the
    two forms only differ in addressing -- with one sequence they name the same entries.
    """
    flat = torch.tensor([3, 4, 5, 6, 7, 8, 9], dtype=torch.int32)
    table, seq_stride, step_stride = mate.mamba._slot_table(
        flat, name="state_batch_indices"
    )
    assert table.data_ptr() == flat.data_ptr(), "the flat table was copied"
    assert (seq_stride, step_stride) == (1, 1)

    wide = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    table, seq_stride, step_stride = mate.mamba._slot_table(
        wide, name="state_batch_indices"
    )
    assert table.shape == (6,)
    assert (seq_stride, step_stride) == (3, 1)

    with pytest.raises(ValueError, match="flat"):
        mate.mamba._slot_table(
            torch.zeros(2, 2, 2, dtype=torch.int32), name="state_batch_indices"
        )


# ---------------------------------------------------------------------------
# Cross-check against the implementation this kernel replaces.
#
# The oracle above is MATE's own reference, so it pins the semantics but not the
# replaceability. The arm the MTP6 config falls back to today is the stock Triton
# SSU, and agreeing with *that* under the same call is what makes the swap a swap
# rather than a behaviour change. The two reduce over dstate in a different order,
# so agreement here is tolerance-based, and the bitwise-equal count is a property
# of that order rather than a gate -- a state that is merely close is still a
# behaviour change for the draft model, which is why the tolerance is the same one
# the standalone validator uses.
# ---------------------------------------------------------------------------

try:  # the cross-check is optional: MATE must stay testable without vLLM-MUSA
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
        selective_state_update as MTP_TRITON_SSU,
    )
except Exception:  # pragma: no cover - environment probe
    MTP_TRITON_SSU = None


def _mtp_run_triton(case: dict[str, object]):
    """Run the stock Triton SSU on the same case, in the wrapper's calling style.

    The serving wrapper keeps ``A``, ``D`` and ``dt_bias`` tied per head as
    zero-stride views, and the Triton launcher detects exactly that; the expands
    here reproduce the same scalar decay rather than a per-(head, dim) one.
    """
    state = case["state"].clone()
    heads, dim, dstate = state.shape[1:]
    rows = case["x"].shape[0]
    out = torch.empty_like(case["x"])
    MTP_TRITON_SSU(
        state,
        case["x"],
        case["dt"].expand(rows, heads, dim),
        case["A"].view(heads, 1, 1).expand(heads, dim, dstate),
        case["B"],
        case["C"],
        case["D"].view(heads, 1).expand(heads, dim),
        case["dt_bias"].view(heads, 1).expand(heads, dim),
        z=case["z"],
        dt_softplus=MTP_SOFTPLUS,
        state_batch_indices=case["src_slots"],
        dst_state_batch_indices=case["dst_slots"],
        null_block_id=MTP_PAD,
        out=out,
        num_accepted_tokens=case["num_accepted_tokens"],
        cu_seqlens=case["cu_seqlens"],
    )
    return out, state


def _mtp_assert_parity(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float):
    """Fail with the size of the disagreement, not just its existence."""
    delta = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    limit = atol + rtol * expected.to(torch.float32).abs()
    bad = int((delta > limit).sum().item())
    if bad:
        raise AssertionError(
            f"{bad}/{delta.numel()} elements outside atol={atol} rtol={rtol}; "
            f"max abs difference {float(delta.max().item()):.4e}"
        )


def _mtp_compare_triton(case: dict[str, object]) -> None:
    """Both implementations answer the same call; the oracle is not involved.

    Two families are left out on purpose. Dense calls are MATE's own extra entry
    point and the serving path never uses them, so there is nothing for the
    fallback arm to agree with. Cases whose seeded read slot is the pad are left
    out for the opposite reason: the pad is a rule MATE's normalizer implements,
    and the stock kernel does not share it.
    """
    expected_out, expected_state = _mtp_run_triton(case)
    actual_state = case["state"].clone()
    actual_out = _mtp_run_kernel(case, state=actual_state).reshape(expected_out.shape)
    _mtp_assert_parity(actual_out, expected_out, 2e-2, 2e-2)
    _mtp_assert_parity(actual_state, expected_state, 1e-3, 1e-2)
    print(
        f"    bitwise-equal out "
        f"{int((actual_out == expected_out).sum().item())}/{expected_out.numel()}"
    )


#: (tokens per sequence, acceptance count) for the cross-check. Triton specializes
#: per shape, so this covers the ends of every regime instead of all of them.
MTP_TRITON_SHAPES = [
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (4, 1),
    (4, 4),
    (7, 0),
    (7, 3),
    (7, 6),
    (7, 7),
]


@supported_musa_compute_capability([31])
@pytest.mark.skipif(MTP_TRITON_SSU is None, reason="vLLM-MUSA's Triton SSU is unavailable")
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("steps, accepted", MTP_TRITON_SHAPES)
def test_packed_call_agrees_with_triton(batch, steps, accepted):
    """The MTP6 call form: one table for both sides, one acceptance count."""
    case = _mtp_case([steps] * batch, accepted=[accepted] * batch, same_slots=True)
    case["dst_slots"] = case["src_slots"]
    _mtp_compare_triton(case)


@supported_musa_compute_capability([31])
@pytest.mark.skipif(MTP_TRITON_SSU is None, reason="vLLM-MUSA's Triton SSU is unavailable")
@torch.inference_mode
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("steps, accepted", [(1, 1), (2, 2), (4, 4), (7, 3), (7, 7)])
def test_distinct_slot_tables_agree_with_triton(batch, steps, accepted):
    """Prefix caching's form: the read table and the write table are different."""
    _mtp_compare_triton(_mtp_case([steps] * batch, accepted=[accepted] * batch))


@supported_musa_compute_capability([31])
@pytest.mark.skipif(MTP_TRITON_SSU is None, reason="vLLM-MUSA's Triton SSU is unavailable")
@torch.inference_mode
def test_variable_length_call_agrees_with_triton():
    """Rows of different lengths: what the serving path builds from cu_seqlens."""
    _mtp_compare_triton(_mtp_case([1, 6, 2, 5], accepted=[1, 3, 2, 5]))


# ---------------------------------------------------------------------------
# CPU-ONLY. Everything below this line runs without a MUSA device: it reaches
# only torch and the plain-torch oracle in ``mate.testing.operators.mamba``, and
# never the TileLang kernel module. If a section below ever needs a device, it
# belongs above this divider instead.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. CPU-only: the multi-token SSU oracle's semantics
#
# These need no MUSA device: the multi-token oracle must reduce to the validated
# one-token oracle when the token loop is written out explicitly, and it must honour
# the slot regimes the consumers rely on. Sitting below the CPU-only divider,
# and importing no TileLang kernel module, is what lets them run anywhere.
# ---------------------------------------------------------------------------

MTP_ORACLE_HEADS, MTP_ORACLE_DIM, MTP_ORACLE_DSTATE, MTP_ORACLE_GROUPS, MTP_ORACLE_SLOTS, MTP_ORACLE_PAD = 4, 3, 8, 2, 8, -1
# An fp32 state pool keeps the token-by-token chain exact: the recurrence carries the
# state in registers between tokens, so rounding the pool between steps would differ
# by rounding only (covered by the last test in this module).
MTP_ORACLE_POOL_DTYPE = torch.float32
MTP_ORACLE_IO_DTYPE = torch.bfloat16


def _mtp_oracle_inputs(rows: int, device: str = "cpu") -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(0)
    return {
        "x": torch.randn(rows, MTP_ORACLE_HEADS, MTP_ORACLE_DIM, generator=generator).to(MTP_ORACLE_IO_DTYPE).to(device),
        "dt": (torch.rand(rows, MTP_ORACLE_HEADS, MTP_ORACLE_DIM, generator=generator) * 0.3 + 0.01).to(
            device
        ),
        "B": torch.randn(rows, MTP_ORACLE_GROUPS, MTP_ORACLE_DSTATE, generator=generator)
        .to(MTP_ORACLE_IO_DTYPE)
        .to(device),
        "C": torch.randn(rows, MTP_ORACLE_GROUPS, MTP_ORACLE_DSTATE, generator=generator)
        .to(MTP_ORACLE_IO_DTYPE)
        .to(device),
        "z": torch.randn(rows, MTP_ORACLE_HEADS, MTP_ORACLE_DIM, generator=generator).to(MTP_ORACLE_IO_DTYPE).to(device),
        "A": (-torch.rand(MTP_ORACLE_HEADS, generator=generator) - 0.1).to(device),
        "D": torch.randn(MTP_ORACLE_HEADS, generator=generator).to(device),
        "bias": (torch.randn(MTP_ORACLE_HEADS, generator=generator) * 0.1).to(device),
    }


def _mtp_oracle_pool(dtype: torch.dtype = MTP_ORACLE_POOL_DTYPE) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return (torch.randn(MTP_ORACLE_SLOTS, MTP_ORACLE_HEADS, MTP_ORACLE_DIM, MTP_ORACLE_DSTATE, generator=generator) * 0.5).to(dtype)


def _mtp_oracle_one_token(
    pool: torch.Tensor,
    inp: dict[str, torch.Tensor],
    row: int,
    src: int,
    dst: int | None,
    softplus: bool = False,
    disable: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-token oracle on a single row, keeping the batch dimension."""
    state = pool.clone()
    y = selective_state_update_one_token_reference(
        state,
        inp["x"][row : row + 1],
        inp["dt"][row : row + 1],
        inp["A"],
        inp["B"][row : row + 1],
        inp["C"][row : row + 1],
        inp["D"],
        inp["z"][row : row + 1],
        inp["bias"],
        softplus,
        torch.tensor([src]),
        None if dst is None else torch.tensor([dst]),
        MTP_ORACLE_PAD,
        disable,
    )
    return y[0], state


def _mtp_oracle_sequential(
    pool: torch.Tensor,
    inp: dict[str, torch.Tensor],
    chains: list[list[tuple[int, int, int | None]]],
    softplus: bool = False,
    disable: bool = False,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Run the one-token oracle token by token over explicit (row, src, dst) chains."""
    outputs: dict[int, torch.Tensor] = {}
    state = pool.clone()
    for chain in chains:
        for row, src, dst in chain:
            y, state = _mtp_oracle_one_token(state, inp, row, src, dst, softplus, disable)
            outputs[row] = y
    return outputs, state


def _mtp_oracle_multi(
    pool: torch.Tensor,
    inp: dict[str, torch.Tensor],
    steps: list[int],
    row_base: list[int],
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor | None,
    softplus: bool = False,
    disable: bool = False,
    accepted: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = pool.clone()
    y = selective_state_update_multi_token_reference(
        state,
        inp["x"],
        inp["dt"],
        inp["A"],
        inp["B"],
        inp["C"],
        inp["D"],
        inp["z"],
        inp["bias"],
        softplus,
        src_slots,
        dst_slots,
        MTP_ORACLE_PAD,
        disable,
        steps,
        row_base,
        accepted,
    )
    return y, state


def test_single_step_matches_one_token_oracle() -> None:
    inp, pool = _mtp_oracle_inputs(3), _mtp_oracle_pool()
    y, state = _mtp_oracle_multi(pool, inp, [1, 1, 1], [0, 1, 2], torch.tensor([1, 3, 5]), None)
    expected_y, expected_state = _mtp_oracle_sequential(
        pool, inp, [[(0, 1, None)], [(1, 3, None)], [(2, 5, None)]]
    )
    for row in range(3):
        assert torch.equal(y[row], expected_y[row])
    assert torch.equal(state, expected_state)


def test_mtp_plain_writes_the_read_slot_once() -> None:
    inp, pool = _mtp_oracle_inputs(2), _mtp_oracle_pool()
    y, state = _mtp_oracle_multi(pool, inp, [2], [0], torch.tensor([2]), None)
    expected_y, expected_state = _mtp_oracle_sequential(pool, inp, [[(0, 2, None), (1, 2, None)]])
    assert torch.equal(y[0], expected_y[0])
    assert torch.equal(y[1], expected_y[1])
    assert torch.equal(state, expected_state)


def test_packed_varlen_writes_the_final_slot_once() -> None:
    inp, pool = _mtp_oracle_inputs(3), _mtp_oracle_pool()
    y, state = _mtp_oracle_multi(
        pool, inp, [2, 1], [0, 2], torch.tensor([6, 7]), torch.tensor([6, 7])
    )
    expected_y, expected_state = _mtp_oracle_sequential(
        pool, inp, [[(0, 6, 6), (1, 6, 6)], [(2, 7, 7)]]
    )
    for row in range(3):
        assert torch.equal(y[row], expected_y[row])
    assert torch.equal(state, expected_state)


def test_speculative_decoding_chains_the_destination_slots() -> None:
    inp, pool = _mtp_oracle_inputs(2), _mtp_oracle_pool()
    y, state = _mtp_oracle_multi(
        pool,
        inp,
        [2],
        [0],
        torch.tensor([1]),
        torch.tensor([[4, 5]]),
        accepted=torch.tensor([1]),
    )
    expected_y, expected_state = _mtp_oracle_sequential(pool, inp, [[(0, 1, 4), (1, 4, 5)]])
    assert torch.equal(y[0], expected_y[0])
    assert torch.equal(y[1], expected_y[1])
    assert torch.equal(state, expected_state)
    # The source slot is only read: the chain moves forward slot by slot.
    assert torch.equal(state[1], pool[1])


def test_num_accepted_tokens_indexes_the_read_slot() -> None:
    inp, pool = _mtp_oracle_inputs(2), _mtp_oracle_pool()
    slots = torch.tensor([[0, 3]])
    # The consumer passes the accepted-token count, and the state is read at
    # ``count - 1`` so that the accepted token itself is the starting state. With no
    # destination table the write target is the read table's own entry for each
    # token, which is what the Triton kernel's ``dst = src`` default produces: the
    # second token writes column 1 whether or not the first one wrote column 0.
    for count, first_read in ((1, 0), (2, 3)):
        y, state = _mtp_oracle_multi(
            pool, inp, [2], [0], slots, None, accepted=torch.tensor([count])
        )
        expected_y, expected_state = _mtp_oracle_sequential(
            pool, inp, [[(0, first_read, 0), (1, 0, 3)]]
        )
        assert torch.equal(y[0], expected_y[0])
        assert torch.equal(y[1], expected_y[1])
        assert torch.equal(state, expected_state)


def test_flat_slot_tables_are_addressed_by_position() -> None:
    inp, pool = _mtp_oracle_inputs(3), _mtp_oracle_pool()
    # vLLM's MTP6 call passes one flat entry per packed row. Triton reaches it by
    # unsqueezing to [rows, 1], so both strides are 1 and sequence 0's position t is
    # entry t -- which is why the accepted position is also the entry read.
    slots = torch.tensor([2, 5, 6])
    y, state = _mtp_oracle_multi(pool, inp, [3], [0], slots, None, accepted=torch.tensor([1]))
    expected_y, expected_state = _mtp_oracle_sequential(
        pool, inp, [[(0, 2, 2), (1, 2, 5), (2, 5, 6)]]
    )
    for row in range(3):
        assert torch.equal(y[row], expected_y[row])
    assert torch.equal(state, expected_state)

    # A flat destination table is addressed the same way, independently of the read
    # table: token t writes its own entry. The pool is ``MTP_ORACLE_SLOTS`` rows
    # deep, so the entries have to stay inside it -- an out-of-pool slot is a caller
    # bug, and the oracle fails loudly on one instead of quietly indexing past the
    # end.
    y, state = _mtp_oracle_multi(
        pool,
        inp,
        [3],
        [0],
        slots,
        torch.tensor([3, 4, 7]),
        accepted=torch.tensor([1]),
    )
    expected_y, expected_state = _mtp_oracle_sequential(
        pool, inp, [[(0, 2, 3), (1, 3, 4), (2, 4, 7)]]
    )
    for row in range(3):
        assert torch.equal(y[row], expected_y[row])
    assert torch.equal(state, expected_state)


def test_flat_slot_tables_carry_one_entry_per_sequence_without_acceptance() -> None:
    inp, pool = _mtp_oracle_inputs(4), _mtp_oracle_pool()
    # Without acceptance counts a flat table is the per-sequence slot vector the
    # plain path passes, so sequence b reads and writes entry b even when its rows
    # span several tokens.
    y, state = _mtp_oracle_multi(pool, inp, [2, 2], [0, 2], torch.tensor([1, 3]), None)
    expected_y, expected_state = _mtp_oracle_sequential(
        pool, inp, [[(0, 1, 1), (1, 1, 1)], [(2, 3, 3), (3, 3, 3)]]
    )
    for row in range(4):
        assert torch.equal(y[row], expected_y[row])
    assert torch.equal(state, expected_state)


def test_pad_slots_disable_state_update_and_empty_sequences() -> None:
    inp, pool = _mtp_oracle_inputs(2), _mtp_oracle_pool()

    # A padded source reads as a zero state and a padded destination is never written.
    zero_pool = pool.clone()
    zero_pool[0] = 0.0
    y_pad, state_pad = _mtp_oracle_multi(
        pool, inp, [2], [0], torch.tensor([MTP_ORACLE_PAD]), torch.tensor([MTP_ORACLE_PAD])
    )
    expected_y, _ = _mtp_oracle_sequential(zero_pool, inp, [[(0, 0, 0), (1, 0, 0)]])
    assert torch.equal(state_pad, pool)
    assert torch.equal(y_pad[0], expected_y[0])
    assert torch.equal(y_pad[1], expected_y[1])

    # disable_state_update and empty sequences leave the pool untouched.
    _, state_disabled = _mtp_oracle_multi(
        pool, inp, [2], [0], torch.tensor([1]), torch.tensor([1]), disable=True
    )
    _, expected_disabled = _mtp_oracle_sequential(
        pool, inp, [[(0, 1, 1), (1, 1, 1)]], disable=True
    )
    assert torch.equal(state_disabled, expected_disabled)
    _, state_empty = _mtp_oracle_multi(pool, inp, [0], [0], torch.tensor([1]), torch.tensor([1]))
    assert torch.equal(state_empty, pool)


def test_optional_features_match_the_sequential_chain() -> None:
    inp, pool = _mtp_oracle_inputs(2), _mtp_oracle_pool()
    # dt_bias and dt_softplus are active in both runs; D and z are always passed.
    for softplus in (False, True):
        y, state = _mtp_oracle_multi(
            pool, inp, [2], [0], torch.tensor([1]), torch.tensor([1]), softplus=softplus
        )
        expected_y, expected_state = _mtp_oracle_sequential(
            pool, inp, [[(0, 1, 1), (1, 1, 1)]], softplus=softplus
        )
        assert torch.equal(y[0], expected_y[0])
        assert torch.equal(y[1], expected_y[1])
        assert torch.equal(state, expected_state)


def test_rounding_state_pool_differs_by_rounding_only() -> None:
    inp = _mtp_oracle_inputs(2)
    pool = _mtp_oracle_pool(torch.float16)
    y, state = _mtp_oracle_multi(pool, inp, [2], [0], torch.tensor([2]), None)
    expected_y, expected_state = _mtp_oracle_sequential(pool, inp, [[(0, 2, None), (1, 2, None)]])
    scale = float(expected_state[2].abs().max())
    assert torch.allclose(
        state[2].to(torch.float32),
        expected_state[2].to(torch.float32),
        atol=8 * 2**-11 * max(scale, 1.0),
    )
    assert torch.allclose(
        y[1].to(torch.float32), expected_y[1].to(torch.float32), atol=2e-2
    )


# ---------------------------------------------------------------------------
# 4. CPU-only: the SSD prefill stages (chunk cumsum, chunk state, state passing, BMM)
#
# These pin the *semantics* of the reference implementations that the device
# kernels are measured against, on any machine with torch. Kernel-level properties
# that need a device (masked stores leaving padded rows untouched, timing) live in
# the device tests above.
# ---------------------------------------------------------------------------

SSD_HEADS = 4
SSD_CHUNK = 8


def _ssd_inputs(offset_list, heads=SSD_HEADS, chunk=SSD_CHUNK, seed=0):
    generator = torch.Generator().manual_seed(seed)
    total = offset_list[-1]
    dt = torch.randn(total, heads, generator=generator)
    A = -torch.rand(heads, generator=generator) - 0.5
    bias = torch.randn(heads, generator=generator) * 0.1
    cu = torch.tensor(offset_list, dtype=torch.int32)
    return dt, A, bias, cu, chunk


def _ssd_layouts():
    """(name, chunk offsets) pairs: full chunks, partial chunks, single tokens."""
    return [
        ("full", [0, 8, 16, 24]),
        ("partial", [0, 8, 11, 19, 21]),
        ("single-token", [0, 1, 9, 10]),
        ("short-chunk", [0, 3, 4, 12]),
    ]


def test_inclusive_prefix_over_each_chunk():
    for _, offsets in _ssd_layouts():
        dt, A, bias, cu, chunk = _ssd_inputs(offsets)
        dA, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
        for c in range(len(offsets) - 1):
            length = offsets[c + 1] - offsets[c]
            expected = torch.cumsum(dt_out[:, c, :length] * A.view(-1, 1), dim=1)
            assert torch.allclose(dA[:, c, :length], expected, atol=1e-6)


def test_chunks_are_independent():
    dt, A, bias, cu, chunk = _ssd_inputs([0, 8, 11, 19, 21])
    base_dA, base_dt = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    perturbed = dt.clone()
    perturbed[0:8] += 5.0  # first chunk only
    new_dA, new_dt = ssd_chunk_cumsum_reference(perturbed, A, bias, cu, chunk, True)
    assert not torch.allclose(base_dA[:, 0, :8], new_dA[:, 0, :8])
    for c in (1, 2, 3):
        assert torch.allclose(base_dA[:, c], new_dA[:, c], atol=0.0)
        assert torch.allclose(base_dt[:, c], new_dt[:, c], atol=0.0)


def test_padding_saturates_dt_to_zero_and_da_to_the_chunk_total():
    """Padding is contractual: dt is 0 past the chunk end, so the scan saturates
    and the row's last position holds the chunk's total decay -- the value the
    downstream stages read unconditionally."""
    offsets = [0, 8, 11]
    dt, A, bias, cu, chunk = _ssd_inputs(offsets)
    dA, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)

    # dt is exactly zero beyond the chunk.
    assert torch.equal(dt_out[:, 1, 3:], torch.zeros_like(dt_out[:, 1, 3:]))
    assert torch.all(dt_out[:, 1, :3] > 0)

    # dA_cumsum is saturated at the chunk total past the chunk...
    total = dA[:, 1, 2]  # last valid position of the 3-token chunk
    assert torch.allclose(
        dA[:, 1, 3:], total.unsqueeze(1).expand_as(dA[:, 1, 3:]), atol=0.0
    )
    # ...and equals the row's last position, which is what callers read.
    assert torch.equal(dA[:, 1, 2], dA[:, 1, chunk - 1])

    # The total is the fp32 accumulation of processed_dt * A over valid tokens.
    expected = (dt_out[:, 1, :3] * A.view(-1, 1)).sum(dim=1)
    assert torch.allclose(total, expected, atol=1e-6)


def test_partial_chunk_equals_full_chunk_prefix():
    """A short chunk must produce exactly the prefix of the longer case."""
    dt, A, bias, cu, chunk = _ssd_inputs([0, 8, 8 + 5])
    short_dA, short_dt = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    full_dA, full_dt = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    assert torch.allclose(short_dA[:, 1, :5], full_dA[:, 1, :5], atol=0.0)
    assert torch.allclose(short_dt[:, 1, :5], full_dt[:, 1, :5], atol=0.0)


def test_dt_softplus_uses_the_threshold_rule():
    dt, A, bias, cu, chunk = _ssd_inputs([0, 8])
    dt = dt.clone()
    dt[0, :] = 30.0  # above the threshold: passed through
    dt[1, :] = -30.0  # below: log1p(exp(x))
    _, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True)
    assert torch.allclose(dt_out[:, 0, 0], dt[0] + bias, atol=1e-5)
    assert torch.allclose(
        dt_out[:, 0, 1], torch.log1p(torch.exp(dt[1] + bias)), atol=1e-5
    )


def test_dt_softplus_disabled_adds_bias_then_clamps():
    """`dt_limit` clamps unconditionally, so with softplus off a negative dt is
    clamped (the production `(0.0, inf)` limit makes that a no-op only while
    softplus is on, which is the model's configuration)."""
    dt, A, bias, cu, chunk = _ssd_inputs([0, 8])
    _, dt_out = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, False)
    assert torch.allclose(
        dt_out[:, 0, 0], (dt[0] + bias).clamp(min=0.0), atol=1e-6
    )
    # With an open limit the bias is the only change.
    _, unbounded = ssd_chunk_cumsum_reference(
        dt, A, bias, cu, chunk, False, dt_limit=(-float("inf"), float("inf"))
    )
    assert torch.allclose(unbounded[:, 0, 0], dt[0] + bias, atol=1e-6)


def test_dt_limit_clamps_before_the_scan():
    dt, A, bias, cu, chunk = _ssd_inputs([0, 8])
    _, dt_out = ssd_chunk_cumsum_reference(
        dt, A, bias, cu, chunk, True, dt_limit=(0.5, 1.5)
    )
    assert float(dt_out.min()) >= 0.5 - 1e-6
    assert float(dt_out.max()) <= 1.5 + 1e-6
    dA, _ = ssd_chunk_cumsum_reference(dt, A, bias, cu, chunk, True, dt_limit=(0.5, 1.5))
    expected = torch.cumsum(dt_out[:, 0, :8] * A.view(-1, 1), dim=1)
    assert torch.allclose(dA[:, 0, :8], expected, atol=1e-6)


def test_missing_dt_bias_is_allowed():
    dt, A, _, cu, chunk = _ssd_inputs([0, 8])
    dA, dt_out = ssd_chunk_cumsum_reference(dt, A, None, cu, chunk, True)
    assert torch.allclose(
        dt_out[:, 0, 0], torch.log1p(torch.exp(dt[0])), atol=1e-5
    )
    assert torch.allclose(dA[:, 0, 0], dt_out[:, 0, 0] * A, atol=1e-6)


# ---------------------------------------------------------------------------
# Stage 2 -- intra-chunk states.
#
# `ssd_chunk_state_reference` mislabels the token axis of its einsum (see
# `test_chunk_state_reference_pairs_the_token_axis`), so the tests below pin the
# properties that hold independently of that defect -- the padding contract, the
# chunk-total read, the single-token case -- plus one xfail that names it.
# ---------------------------------------------------------------------------

STATE_HEADS = 4
STATE_GROUPS = 2
STATE_DIM = 3
STATE_DSTATE = 4
STATE_CHUNK = 4


def _state_inputs(offsets, heads=STATE_HEADS, groups=STATE_GROUPS, seed=0):
    """Token-major ``x``/``B``, the head-major stage-1 outputs, and the metadata.

    ``dA_cumsum``/``dt_out`` come from the cumsum reference itself, so the chunk
    totals -- including the nonzero *tail* that stage 2 reads unconditionally --
    are exactly the ones stage 1 produces.
    """
    generator = torch.Generator().manual_seed(seed)
    dim, dstate, chunk = STATE_DIM, STATE_DSTATE, STATE_CHUNK
    tokens = offsets[-1]
    x = torch.randn(tokens, heads, dim, generator=generator)
    b = torch.randn(tokens, groups, dstate, generator=generator)
    dt = torch.randn(tokens, heads, generator=generator)
    a = -torch.rand(heads, generator=generator) - 0.5
    cu_chunk_seqlens = torch.tensor(offsets, dtype=torch.int32)
    dA_cumsum, dt_out = ssd_chunk_cumsum_reference(
        dt, a, None, cu_chunk_seqlens, chunk, True
    )
    return x, b, dt_out, dA_cumsum, cu_chunk_seqlens, chunk


def _state_explicit(x, b, dt_out, dA_cumsum, offsets, chunk, head_ratio):
    """The rank-one accumulation, written out per token in fp64.

    ``S_c[h,d,n] = sum_{t < limit} exp(min(total - dA_t, 0)) * dt_t * x_t B_t``
    over the *valid* token range only, so it cannot inherit any padding.
    """
    heads = x.shape[1]
    result = torch.zeros(
        (len(offsets) - 1, heads, STATE_DIM, STATE_DSTATE), dtype=torch.float64
    )
    for c in range(len(offsets) - 1):
        lo, hi = offsets[c], offsets[c + 1]
        limit = hi - lo
        if limit <= 0:
            continue
        total = dA_cumsum[:, c, chunk - 1].to(torch.float64)
        weight = torch.exp(
            (total.unsqueeze(1) - dA_cumsum[:, c, :limit].to(torch.float64)).clamp(
                max=0.0
            )
        ) * dt_out[:, c, :limit].to(torch.float64)
        for head in range(heads):
            group = head // head_ratio
            for t in range(limit):
                result[c, head] += (
                    weight[head, t] * x[lo + t, head].to(torch.float64)
                ).unsqueeze(1) * b[lo + t, group].to(torch.float64).unsqueeze(0)
    return result


def test_chunk_state_tail_of_a_partial_chunk_contributes_nothing():
    """``dt_out`` past a chunk's end is contractual zero, and stage 2 must ignore
    it even when a contract-violating nonzero tail is supplied: the decay factor
    there is ``exp(total - total) = 1``, so only the zero dt saves it."""
    offsets = [0, 3]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    base = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)
    assert torch.equal(dt_out[:, 0, 3:], torch.zeros_like(dt_out[:, 0, 3:]))

    dirty = dt_out.clone()
    dirty[:, 0, 3:] = 5.0  # the padded tail only
    assert torch.equal(ssd_chunk_state_reference(x, b, dirty, dA, cu, chunk), base)


def test_chunk_state_reads_the_chunk_total_from_the_padded_tail():
    """``dA_cumsum[:, c, L-1]`` holds the chunk's total decay for every chunk
    length and is read unconditionally, so perturbing only that entry moves the
    result of a partial chunk."""
    offsets = [0, 3]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    base = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)
    assert float(dA[:, 0, chunk - 1].abs().min()) > 0.0

    perturbed = dA.clone()
    perturbed[:, 0, chunk - 1] -= 0.5
    moved = ssd_chunk_state_reference(x, b, dt_out, perturbed, cu, chunk)
    assert not torch.allclose(base, moved)


def test_chunk_state_decay_is_a_difference_of_one_cumsum():
    """The decay weights are ``exp(total - dA_t)``, so shifting a chunk's whole
    ``dA_cumsum`` row -- valid part and padded tail together -- by a constant
    leaves the state untouched. An exclusive cumsum, a ``+dt`` shift or a zeroed
    tail would not survive this."""
    offsets = [0, 3, 7]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    base = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)

    # Shift one chunk's whole padded row -- valid entries and the total together.
    shifted = dA.clone()
    shifted[:, 0, :] += 2.0
    assert torch.allclose(
        ssd_chunk_state_reference(x, b, dt_out, shifted, cu, chunk).to(torch.float64),
        base.to(torch.float64),
        atol=1e-5,
    )


def test_chunk_state_single_token_chunk_is_the_rank_one_contribution():
    """With one valid token the reference's contraction is exact (there is only
    one token to sum over), so this pins the weight formula and the layout."""
    offsets = [0, 1]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    states = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)
    for head in range(STATE_HEADS):
        group = head // (STATE_HEADS // STATE_GROUPS)
        delta = float(dA[head, 0, chunk - 1] - dA[head, 0, 0])
        weight = float(torch.tensor(delta).clamp(max=0.0).exp()) * float(
            dt_out[head, 0, 0]
        )
        expected = torch.outer(x[0, head].double(), b[0, group].double()) * weight
        assert torch.allclose(states[0, head].to(torch.float64), expected, atol=1e-6)


def test_chunk_state_zero_length_chunk_is_zero():
    offsets = [0, 0, 4]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    states = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)
    assert torch.equal(states[0], torch.zeros_like(states[0]))
    assert bool((states[1].abs() > 1e-6).any())


def test_chunk_state_chunks_do_not_mix_sequences():
    """Perturbing one sequence's tokens must leave the other sequence's chunks
    bit-identical -- the token range comes from ``cu_chunk_seqlens`` alone."""
    offsets = [0, 3, 7]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    base = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)

    moved_x = x.clone()
    moved_b = b.clone()
    moved_x[0:3] += 1.0
    moved_b[0:3] += 1.0
    moved = ssd_chunk_state_reference(moved_x, moved_b, dt_out, dA, cu, chunk)

    assert not torch.allclose(base[0], moved[0])
    assert torch.equal(base[1], moved[1])


@pytest.mark.xfail(
    strict=False,
    reason="ssd_chunk_state_reference contracts the wrong axis: its einsum "
    "'thd,hln->hdn' leaves the token labels unshared, so it returns "
    "(sum_t x[t,h,d]) * (sum_l scaled[h,l,n]) instead of the rank-one "
    "accumulation over the chunk's tokens. Reported; the fix is "
    "'thd,htn->hdn'.",
)
def test_chunk_state_reference_pairs_the_token_axis():
    """The state must be the chunk's rank-one accumulation, not a product of two
    independent sums."""
    offsets = [0, 3, 4, 8]
    x, b, dt_out, dA, cu, chunk = _state_inputs(offsets)
    states = ssd_chunk_state_reference(x, b, dt_out, dA, cu, chunk)
    expected = _state_explicit(
        x, b, dt_out, dA, offsets, chunk, STATE_HEADS // STATE_GROUPS
    )
    assert torch.allclose(states.to(torch.float64), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Stage 3 -- inter-chunk state passing.
# ---------------------------------------------------------------------------

PASS_HEADS = 2
PASS_DIM = 3
PASS_DSTATE = 4
PASS_CHUNK = 4


def _passing_case(nchunks, last, decay=None, seed=0):
    """``states``/``dA_cumsum``/``last_chunk_indices`` for one chunk layout.

    ``dA_cumsum`` carries only the padded row's last entry: the chunk's total
    decay. ``decay`` is ``[heads, nchunks]`` (or a scalar), left at 0 -- i.e.
    ``exp(0) = 1``, no decay -- when omitted.
    """
    generator = torch.Generator().manual_seed(seed)
    heads, dim, dstate, chunk = PASS_HEADS, PASS_DIM, PASS_DSTATE, PASS_CHUNK
    states = torch.randn((nchunks, heads, dim, dstate), generator=generator)
    dA_cumsum = torch.zeros((heads, nchunks, chunk))
    if decay is not None:
        dA_cumsum[:, :, chunk - 1] = torch.as_tensor(decay, dtype=torch.float32)
    return states, dA_cumsum, torch.tensor(last, dtype=torch.int32)


def _passing_fp64(states, dA_cumsum, last, initial_states=None):
    """Independent fp64 recurrence over the chunk ranges ``last`` implies."""
    heads, dim, dstate = PASS_HEADS, PASS_DIM, PASS_DSTATE
    last = [int(value) for value in last]
    out = torch.zeros((states.shape[0], heads, dim, dstate), dtype=torch.float64)
    for sequence in range(len(last)):
        end = last[sequence] + 1
        start = (last[sequence - 1] + 1) if sequence > 0 else 0
        running = (
            torch.zeros((heads, dim, dstate), dtype=torch.float64)
            if initial_states is None
            else initial_states[sequence].to(torch.float64)
        )
        for chunk in range(start, end):
            running = torch.exp(
                dA_cumsum[:, chunk, PASS_CHUNK - 1].to(torch.float64)
            ).view(heads, 1, 1) * running + states[chunk].to(torch.float64)
            out[chunk] = running
    return out


def test_state_passing_without_decay_is_the_running_sum():
    """``dA_cumsum`` all zero makes the recurrence a plain prefix sum -- and the
    prefix restarts at every sequence boundary (sequence 0 owns chunk 0, sequence
    1 owns chunks 1..3)."""
    states, dA_cumsum, last = _passing_case(4, [0, 3])
    out = ssd_state_passing_reference(states, dA_cumsum, last)
    expected = torch.empty_like(out)
    expected[0] = states[0]
    expected[1:] = torch.cumsum(states[1:], dim=0)
    assert torch.allclose(out, expected, atol=1e-5)
    assert torch.equal(out[0], states[0])


def test_state_passing_matches_the_fp64_recurrence_with_decay():
    """Two sequences, a partial second one, per-(head, chunk) decay, and fp16
    initial states that must be upcast before the recurrence."""
    decay = -torch.rand(PASS_HEADS, 4) - 0.1
    states, dA_cumsum, last = _passing_case(4, [1, 3], decay=decay)
    generator = torch.Generator().manual_seed(7)
    init = torch.randn((2, PASS_HEADS, PASS_DIM, PASS_DSTATE), generator=generator).to(
        torch.float16
    )
    out = ssd_state_passing_reference(
        states, dA_cumsum, last, initial_states=init, state_dtype=torch.float32
    )
    assert out.dtype == torch.float32
    assert torch.allclose(
        out.to(torch.float64), _passing_fp64(states, dA_cumsum, last, init), atol=1e-5
    )


def test_state_passing_indexes_initial_states_by_sequence_position():
    states, dA_cumsum, last = _passing_case(3, [0, 2])
    init = torch.zeros(2, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    init[0] = 3.0
    init[1] = -7.0
    out = ssd_state_passing_reference(states, dA_cumsum, last, initial_states=init)
    assert torch.allclose(out[0], states[0] + 3.0, atol=1e-5)
    # Sequence 1 is seeded from row 1, not from row 0 and not from a seq_idx.
    assert torch.allclose(out[1], states[1] - 7.0, atol=1e-5)
    assert torch.allclose(out[2], states[1] + states[2] - 7.0, atol=1e-5)


def test_state_passing_zero_chunk_sequences_iterate_zero_times():
    """The middle sequence owns no chunk, so it must neither seed nor advance the
    chunk its right neighbour starts from."""
    states, dA_cumsum, last = _passing_case(4, [1, 1, 3])
    init = torch.zeros(3, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    init[0] = 1.0
    init[1] = 100.0
    init[2] = 1000.0
    out = ssd_state_passing_reference(states, dA_cumsum, last, initial_states=init)

    assert torch.allclose(
        out.to(torch.float64),
        _passing_fp64(states, dA_cumsum, last, init),
        atol=1e-5,
    )
    # Sequence 0 owns chunks 0 and 1 and carries its own running state.
    assert torch.allclose(out[0], states[0] + 1.0, atol=1e-5)
    assert torch.allclose(out[1], states[0] + states[1] + 1.0, atol=1e-5)
    # Chunks 2 and 3 belong to sequence 2 and start from 1000, never from the
    # empty sequence's 100.
    assert torch.allclose(out[2], states[2] + 1000.0, atol=1e-5)
    assert torch.allclose(out[3], states[2] + states[3] + 1000.0, atol=1e-5)


def test_state_passing_first_sequence_with_no_chunks():
    """``last_chunk_indices[0] == -1``: sequence 0 owns nothing and the neighbour
    still starts at chunk 0."""
    states, dA_cumsum, last = _passing_case(2, [-1, 1])
    init = torch.zeros(2, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    init[1] = 5.0
    out = ssd_state_passing_reference(states, dA_cumsum, last, initial_states=init)
    assert torch.allclose(out[0], states[0] + 5.0, atol=1e-5)
    assert torch.allclose(out[1], states[0] + states[1] + 5.0, atol=1e-5)


def test_state_passing_all_empty_sequences_leave_the_output_untouched():
    states, dA_cumsum, last = _passing_case(2, [-1, -1])
    out = ssd_state_passing_reference(states, dA_cumsum, last)
    assert torch.equal(out, torch.zeros_like(out))


def test_state_passing_absent_initial_states_equals_zero_initial_states():
    states, dA_cumsum, last = _passing_case(3, [2], decay=-0.3)
    zeros = torch.zeros(1, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    with_init = ssd_state_passing_reference(
        states, dA_cumsum, last, initial_states=zeros
    )
    without = ssd_state_passing_reference(states, dA_cumsum, last)
    assert torch.equal(with_init, without)


def test_state_passing_rounds_only_on_the_store():
    """The running state stays fp32 across chunk boundaries and only the store
    rounds; carrying the rounded value forward gives a different answer."""
    states = torch.zeros(4, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    states[0] = 1.0
    states[1:] = 4e-4
    _, dA_cumsum, last = _passing_case(4, [3])
    out = ssd_state_passing_reference(
        states, dA_cumsum, last, state_dtype=torch.float16
    )
    assert out.dtype == torch.float16

    running = torch.cumsum(states.to(torch.float32), dim=0)
    assert torch.equal(out, running.to(torch.float16))

    carried = torch.empty_like(running)
    previous = torch.zeros(PASS_HEADS, PASS_DIM, PASS_DSTATE)
    for chunk in range(4):
        previous = (previous.to(torch.float16) + states[chunk]).to(torch.float32)
        carried[chunk] = previous
    assert not torch.equal(out, carried.to(torch.float16))


def test_state_passing_state_dtype_resolution():
    states, dA_cumsum, last = _passing_case(2, [1])
    init = torch.zeros(1, PASS_HEADS, PASS_DIM, PASS_DSTATE, dtype=torch.bfloat16)
    # state_dtype wins...
    out = ssd_state_passing_reference(
        states, dA_cumsum, last, initial_states=init, state_dtype=torch.float32
    )
    assert out.dtype == torch.float32
    # ...then initial_states.dtype...
    out = ssd_state_passing_reference(states, dA_cumsum, last, initial_states=init)
    assert out.dtype == torch.bfloat16
    # ...then the input state's dtype.
    out = ssd_state_passing_reference(states, dA_cumsum, last)
    assert out.dtype == torch.float32


def _passing_fp64_by_seq_idx(states, dA_cumsum, seq_idx, initial_states=None):
    """Independent fp64 recurrence for the ``seq_idx`` walk.

    Chunks are visited in index order; a sequence's first chunk enters from
    ``initial_states[seq_idx[c]]``, later chunks of the same id continue it, and
    ids that never appear leave nothing written.
    """
    running: dict[int, torch.Tensor] = {}
    out = torch.zeros(states.shape, dtype=torch.float64)
    for chunk in range(states.shape[0]):
        sequence = int(seq_idx[chunk])
        if sequence not in running:
            running[sequence] = (
                torch.zeros((PASS_HEADS, PASS_DIM, PASS_DSTATE), dtype=torch.float64)
                if initial_states is None
                else initial_states[sequence].to(torch.float64)
            )
        decay = torch.exp(dA_cumsum[:, chunk, PASS_CHUNK - 1].to(torch.float64)).view(
            PASS_HEADS, 1, 1
        )
        running[sequence] = decay * running[sequence] + states[chunk].to(torch.float64)
        out[chunk] = running[sequence]
    return out


def test_state_passing_seq_idx_agrees_with_last_chunk_indices_when_contiguous():
    """The two boundary rules are interchangeable metadata for a packed,
    contiguous prefill: sequence 0 owns chunk 0, sequence 1 owns chunks 1..3."""
    states, dA_cumsum, last = _passing_case(4, [0, 3], decay=-0.4)
    seq_idx = torch.tensor([0, 1, 1, 1], dtype=torch.int32)
    init = torch.full((2, PASS_HEADS, PASS_DIM, PASS_DSTATE), 0.25)
    by_range = ssd_state_passing_reference(states, dA_cumsum, last, initial_states=init)
    by_seq_idx = ssd_state_passing_reference(
        states, dA_cumsum, last, initial_states=init, seq_idx=seq_idx
    )
    assert torch.equal(by_range, by_seq_idx)


def test_state_passing_seq_idx_keeps_interleaved_sequences_separate():
    """Interleaved chunks are where the rules disagree: ``last_chunk_indices``
    arithmetic would hand sequence 1 the whole tail from chunk 1, while the
    oracle gives it only its own chunks 1 and 3."""
    states, dA_cumsum, _ = _passing_case(4, [3], decay=-0.4)
    seq_idx = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    init = torch.full((2, PASS_HEADS, PASS_DIM, PASS_DSTATE), 0.5)

    out = ssd_state_passing_reference(
        states, dA_cumsum, None, initial_states=init, seq_idx=seq_idx
    )
    assert torch.allclose(
        out.to(torch.float64),
        _passing_fp64_by_seq_idx(states, dA_cumsum, seq_idx, init),
        atol=1e-5,
    )
    # Chunks 0 and 2 belong to sequence 0 and are chained through chunk 2 only.
    assert not torch.allclose(out[1], out[2], atol=1e-3)
    # Chunk 2 continues chunk 0's sequence: it must not see chunk 1's state.
    decay = torch.exp(dA_cumsum[:, 2, PASS_CHUNK - 1].to(torch.float64)).view(
        PASS_HEADS, 1, 1
    )
    expected = decay * out[0].to(torch.float64) + states[2].to(torch.float64)
    assert torch.allclose(out[2].to(torch.float64), expected, atol=1e-5)


def test_state_passing_seq_idx_skips_without_decaying():
    """A skipped chunk must not decay the resumed sequence's state, and the
    resumed chunk applies its own decay."""
    states, dA_cumsum, _ = _passing_case(3, [2], decay=-1.0)
    seq_idx = torch.tensor([0, 1, 0], dtype=torch.int32)

    out = ssd_state_passing_reference(states, dA_cumsum, None, seq_idx=seq_idx)
    assert torch.allclose(
        out[0].to(torch.float64), states[0].to(torch.float64), atol=1e-5
    )
    decay = torch.exp(dA_cumsum[:, 2, PASS_CHUNK - 1].to(torch.float64)).view(
        PASS_HEADS, 1, 1
    )
    # Only chunk 2's own decay applies -- chunk 1's is not applied on the way.
    resumed = decay * states[0].to(torch.float64) + states[2].to(torch.float64)
    assert torch.allclose(out[2].to(torch.float64), resumed, atol=1e-5)


def test_state_passing_seq_idx_indexes_initial_states_by_the_id():
    """``initial_states`` is indexed by the value of ``seq_idx``: chunk 0's
    sequence is id 1, so it enters from ``initial_states[1]``."""
    states, dA_cumsum, last = _passing_case(2, [0, 1], decay=-0.2)
    seq_idx = torch.tensor([1, 1], dtype=torch.int32)
    init = torch.zeros(2, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    init[0] = -3.0
    init[1] = 7.0

    out = ssd_state_passing_reference(
        states, dA_cumsum, last, initial_states=init, seq_idx=seq_idx
    )
    decay0 = torch.exp(dA_cumsum[:, 0, PASS_CHUNK - 1].to(torch.float64)).view(
        PASS_HEADS, 1, 1
    )
    assert torch.allclose(
        out[0].to(torch.float64),
        7.0 * decay0 + states[0].to(torch.float64),
        atol=1e-5,
    )
    assert not torch.allclose(
        out[0].to(torch.float64), -3.0 * decay0 + states[0].double()
    )


def test_state_passing_seq_idx_id_with_no_chunks_writes_nothing():
    """Id 1 never appears in ``seq_idx``, so its initial state can never enter a
    chain: the result is unchanged when that row is zeroed."""
    states, dA_cumsum, _ = _passing_case(3, [2], decay=-0.5)
    seq_idx = torch.tensor([0, 2, 2], dtype=torch.int32)
    init = torch.ones(3, PASS_HEADS, PASS_DIM, PASS_DSTATE)
    init[1] = 100.0

    out = ssd_state_passing_reference(
        states, dA_cumsum, None, initial_states=init, seq_idx=seq_idx
    )
    assert torch.allclose(
        out.to(torch.float64),
        _passing_fp64_by_seq_idx(states, dA_cumsum, seq_idx, init),
        atol=1e-5,
    )
    assert bool((out[2].abs() > 1e-6).any())

    zeroed = init.clone()
    zeroed[1] = 0.0
    assert torch.allclose(
        out.to(torch.float64),
        _passing_fp64_by_seq_idx(states, dA_cumsum, seq_idx, zeroed),
        atol=1e-5,
    )


def test_state_passing_seq_idx_must_index_initial_states():
    states, dA_cumsum, last = _passing_case(2, [0, 1])
    with pytest.raises(ValueError, match="must index initial_states"):
        ssd_state_passing_reference(
            states,
            dA_cumsum,
            last,
            initial_states=torch.zeros(2, PASS_HEADS, PASS_DIM, PASS_DSTATE),
            seq_idx=torch.tensor([0, 5], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="one sequence id per chunk"):
        ssd_state_passing_reference(
            states, dA_cumsum, last, seq_idx=torch.tensor([0], dtype=torch.int32)
        )


# ---------------------------------------------------------------------------
# Stage 4 -- chunk-level ``C @ B^T``.
# ---------------------------------------------------------------------------

BMM_GROUPS = 2
BMM_DSTATE = 4
BMM_CHUNK = 4


def _bmm_case(offsets, dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokens = offsets[-1]
    cmat = torch.randn((tokens, BMM_GROUPS, BMM_DSTATE), generator=generator)
    bmat = torch.randn((tokens, BMM_GROUPS, BMM_DSTATE), generator=generator)
    cu_chunk_seqlens = torch.tensor(offsets, dtype=torch.int32)
    return cmat.to(dtype), bmat.to(dtype), cu_chunk_seqlens


def _bmm_explicit(cmat, bmat, offsets, chunk=BMM_CHUNK, dot_dtype=None):
    """Independent fp64 double loop over individual inner products."""
    if dot_dtype is not None:
        cmat = cmat.to(dot_dtype)
        bmat = bmat.to(dot_dtype)
    cmat = cmat.to(torch.float64)
    bmat = bmat.to(torch.float64)
    out = torch.zeros((len(offsets) - 1, BMM_GROUPS, chunk, chunk), dtype=torch.float64)
    for c in range(len(offsets) - 1):
        lo, hi = offsets[c], offsets[c + 1]
        for i in range(hi - lo):
            for j in range(hi - lo):
                for group in range(BMM_GROUPS):
                    out[c, group, i, j] = (
                        cmat[lo + i, group] * bmat[lo + j, group]
                    ).sum()
    return out


def test_bmm_matches_explicit_inner_products_over_the_chunk_range():
    """Multi-sequence varlen: chunk lengths 3, 1 and 4 against chunk_size 4."""
    offsets = [0, 3, 4, 8]
    cmat, bmat, cu = _bmm_case(offsets)
    cb = ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)
    assert cb.shape == (3, BMM_GROUPS, BMM_CHUNK, BMM_CHUNK)
    assert cb.dtype == torch.float32
    assert torch.allclose(
        cb.to(torch.float64), _bmm_explicit(cmat, bmat, offsets), atol=1e-5
    )


def test_bmm_is_non_causal():
    offsets = [0, 4]
    cmat, bmat, cu = _bmm_case(offsets)
    cb = ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)
    # The strict upper triangle is materialized, not left at zero...
    assert bool((cb[:, :, 0, 1:].abs() > 1e-6).all())
    # ...and it is a genuine inner product, not a copy of the lower triangle.
    assert not torch.allclose(cb[:, :, 0, 1], cb[:, :, 1, 0])
    explicit = _bmm_explicit(cmat, bmat, offsets)
    for i in range(BMM_CHUNK):
        for j in range(BMM_CHUNK):
            assert torch.allclose(
                cb[:, :, i, j].to(torch.float64), explicit[:, :, i, j], atol=1e-5
            )


def test_bmm_partial_chunk_tail_is_exactly_zero():
    """A chunk shorter than ``chunk_size``: everything at or beyond the boundary
    is exactly zero while the valid block is untouched."""
    offsets = [0, 2]
    cmat, bmat, cu = _bmm_case(offsets)
    cb = ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)
    assert torch.equal(cb[:, :, 2:, :], torch.zeros_like(cb[:, :, 2:, :]))
    assert torch.equal(cb[:, :, :, 2:], torch.zeros_like(cb[:, :, :, 2:]))
    assert bool((cb[:, :, :2, :2].abs() > 1e-6).all())
    explicit = _bmm_explicit(cmat, bmat, offsets)
    assert torch.allclose(
        cb[:, :, :2, :2].to(torch.float64), explicit[:, :, :2, :2], atol=1e-5
    )


def test_bmm_single_token_and_zero_length_chunks():
    offsets = [0, 1, 1, 5]
    cmat, bmat, cu = _bmm_case(offsets)
    cb = ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)

    expected = torch.einsum("gn,gn->g", cmat[0], bmat[0]).to(torch.float64)
    assert torch.allclose(cb[0, :, 0, 0].to(torch.float64), expected, atol=1e-5)
    assert torch.equal(cb[0, :, 0, 1:], torch.zeros_like(cb[0, :, 0, 1:]))
    assert torch.equal(cb[0, :, 1:, :], torch.zeros_like(cb[0, :, 1:, :]))

    # A zero-length chunk contributes nothing at all.
    assert torch.equal(cb[1], torch.zeros_like(cb[1]))
    assert torch.allclose(
        cb[2].to(torch.float64), _bmm_explicit(cmat, bmat, offsets)[2], atol=1e-5
    )


def test_bmm_rounds_both_operands_to_the_activation_dtype_before_the_product():
    offsets = [0, 4]
    cmat32, bmat32, cu = _bmm_case(offsets)
    cmat = cmat32.to(torch.bfloat16)
    bmat = bmat32.to(torch.bfloat16)
    cb = ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)
    assert cb.dtype == torch.float32
    # Exactly the dot of the bf16-rounded operands...
    assert torch.allclose(
        cb.to(torch.float64),
        _bmm_explicit(cmat, bmat, offsets),
        atol=1e-4,
    )
    # ...which is a different answer from the fp32-exact product of the same
    # values, so the rounding point is observable.
    exact = _bmm_explicit(cmat32, bmat32, offsets)
    assert not torch.allclose(cb.to(torch.float64), exact, atol=1e-6)


def test_bmm_rejects_a_chunk_longer_than_chunk_size():
    offsets = [0, 5]
    cmat, bmat, cu = _bmm_case(offsets)
    with pytest.raises(ValueError, match="longer than chunk_size"):
        ssd_bmm_reference(cmat, bmat, cu, BMM_CHUNK)


# ---------------------------------------------------------------------------
# 5. CPU-only: the SSD chunk scan
#
# The scan is the one stage where a wrong result hides easily: the operands are
# small, the tolerances downstream are loose, and three separate things can be
# individually plausible and still wrong together -- the choice of the entering
# state, the causal masking, and the order of the ``D``/``z`` epilogue. So the
# reference is checked against a second, deliberately naive transcription of the
# specification's per-token formula, plus one test per contract clause.
# ---------------------------------------------------------------------------

SCAN_HEADS = 4
SCAN_DIM = 3
SCAN_DSTATE = 5
SCAN_GROUPS = 2
SCAN_CHUNK = 8
SCAN_HEAD_RATIO = SCAN_HEADS // SCAN_GROUPS


def _naive_scan(x, C, B, dt_out, dA_cumsum, states, initial_states, seq_idx, offsets, out):
    """The spec's formula as a plain per-token, per-j triple loop."""
    tokens, heads, dim = x.shape
    _, nchunks, block = dA_cumsum.shape
    c32 = C.to(torch.float32)
    b32 = B.to(torch.float32)
    x32 = x.to(torch.float32)
    dA = dA_cumsum.to(torch.float32)
    dt = dt_out.to(torch.float32)
    st = states.to(torch.float32)
    sequence = [int(v) for v in seq_idx.reshape(-1).tolist()]

    for chunk in range(nchunks):
        start, end = offsets[chunk], offsets[chunk + 1]
        limit = end - start
        if limit <= 0:
            continue
        opens = chunk == 0 or sequence[chunk] != sequence[chunk - 1]
        if opens:
            prev = (
                initial_states[sequence[chunk]].to(torch.float32)
                if initial_states is not None
                else torch.zeros((heads, dim, SCAN_DSTATE), dtype=torch.float32)
            )
        else:
            prev = st[chunk - 1]
        for head in range(heads):
            group = head // SCAN_HEAD_RATIO
            for t in range(limit):
                dA_t = float(dA[head, chunk, t])
                for d in range(dim):
                    acc = 0.0
                    for n in range(SCAN_DSTATE):
                        acc += (
                            float(c32[start + t, group, n])
                            * torch.exp(torch.tensor(dA_t)).item()
                            * float(prev[head, d, n])
                        )
                    for j in range(t + 1):
                        # the CB product, then the decay, then dt, then the
                        # round to the activation dtype -- production's order
                        cb = 0.0
                        for n in range(SCAN_DSTATE):
                            cb += float(c32[start + t, group, n]) * float(
                                b32[start + j, group, n]
                            )
                        dA_j = float(dA[head, chunk, j])
                        cb = cb * float(
                            torch.exp(torch.tensor(min(dA_t - dA_j, 0.0))).item()
                        )
                        cb = cb * float(dt[head, chunk, j])
                        cb = float(
                            torch.tensor(cb, dtype=torch.float32).to(x.dtype).to(torch.float32)
                        )
                        acc += cb * float(x32[start + j, head, d])
                    out[start + t, head, d] = acc
    return out


def _scan_case(offsets, seed=0, heads=SCAN_HEADS, chunk=SCAN_CHUNK, with_initial=True):
    generator = torch.Generator().manual_seed(seed)
    total = offsets[-1]
    x = torch.randn(total, heads, SCAN_DIM, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    C = torch.randn(total, SCAN_GROUPS, SCAN_DSTATE, generator=generator).to(torch.bfloat16)
    B = torch.randn(total, SCAN_GROUPS, SCAN_DSTATE, generator=generator).to(torch.bfloat16)
    nchunks = len(offsets) - 1
    dt_out = torch.zeros(heads, nchunks, chunk)
    dA_cumsum = torch.zeros(heads, nchunks, chunk)
    for c in range(nchunks):
        length = offsets[c + 1] - offsets[c]
        dt_out[:, c, :length] = torch.rand(heads, length, generator=generator) * 0.5 + 0.1
        dA_cumsum[:, c, :length] = torch.cumsum(
            -torch.rand(heads, length, generator=generator), dim=1
        )
        dA_cumsum[:, c, length:] = dA_cumsum[:, c, length - 1 : length]
    states = ssd_chunk_state_reference(
        x, B, dt_out, dA_cumsum, torch.tensor(offsets, dtype=torch.int32), chunk
    )
    initial_states = (
        torch.randn(len(offsets) - 1, heads, SCAN_DIM, SCAN_DSTATE, generator=generator)
        if with_initial
        else None
    )
    return x, C, B, dt_out, dA_cumsum, states, initial_states


def test_reference_matches_the_naive_per_token_formula():
    offsets = [0, 8]
    x, C, B, dt_out, dA_cumsum, states, initial = _scan_case(offsets)
    seq_idx = torch.zeros(1, dtype=torch.int32)
    cu = torch.tensor(offsets, dtype=torch.int32)

    fast = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, SCAN_CHUNK
    )
    slow = _naive_scan(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, offsets, torch.zeros_like(x)
    )
    assert torch.allclose(fast.float(), slow.float(), rtol=1e-4, atol=1e-4)


def test_second_sequence_and_bare_chunks_use_the_right_entering_state():
    offsets = [0, 8, 16]
    x, C, B, dt_out, dA_cumsum, states, initial = _scan_case(offsets)
    cu = torch.tensor(offsets, dtype=torch.int32)
    # Chunks 0 and 1 belong to two sequences: each takes its initial state.
    seq_idx = torch.tensor([0, 1], dtype=torch.int32)
    with_initial = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, SCAN_CHUNK
    )
    # With no initial states both chunks of that layout start from zero, so both
    # results must equal the same run against zeroed initial states.
    zeros = torch.zeros_like(initial)
    from_zeros = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, zeros, seq_idx, cu, SCAN_CHUNK
    )
    assert not torch.allclose(with_initial, from_zeros)
    # One sequence spanning both chunks: chunk 0 opens it, so only chunk 0 sees
    # the initial state; chunk 1 must consume states[0] and be identical either
    # way. That is exactly the boundary the two indexings disagree on.
    one_seq = torch.zeros(2, dtype=torch.int32)
    a = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, one_seq, cu, SCAN_CHUNK
    )
    b = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, zeros, one_seq, cu, SCAN_CHUNK
    )
    assert torch.equal(a[SCAN_CHUNK:], b[SCAN_CHUNK:])
    assert not torch.allclose(a[:SCAN_CHUNK], b[:SCAN_CHUNK])


def test_rows_past_a_partial_chunk_are_never_written():
    offsets = [0, 8, 11]
    x, C, B, dt_out, dA_cumsum, states, initial = _scan_case(offsets)
    seq_idx = torch.zeros(2, dtype=torch.int32)
    cu = torch.tensor(offsets, dtype=torch.int32)
    out = torch.full_like(x, float("nan"))
    ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, SCAN_CHUNK, out=out
    )
    assert not torch.isnan(out[:11]).any()
    assert out[11:].isnan().all()


def test_future_tokens_do_not_leak_through_the_causal_mask():
    offsets = [0, 8]
    x, C, B, dt_out, dA_cumsum, states, initial = _scan_case(offsets)
    seq_idx = torch.zeros(1, dtype=torch.int32)
    cu = torch.tensor(offsets, dtype=torch.int32)
    base = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, SCAN_CHUNK
    )
    perturbed = x.clone()
    perturbed[4:] += 10.0  # only tokens at or after row 4
    after = ssd_chunk_scan_reference(
        perturbed, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, SCAN_CHUNK
    )
    assert torch.allclose(base[:4], after[:4], atol=1e-6)
    assert not torch.allclose(base[4:], after[4:])


def test_epilogue_applies_d_before_the_z_gate():
    offsets = [0, 8]
    x, C, B, dt_out, dA_cumsum, states, initial = _scan_case(offsets)
    seq_idx = torch.zeros(1, dtype=torch.int32)
    cu = torch.tensor(offsets, dtype=torch.int32)
    generator = torch.Generator().manual_seed(7)
    D_param = torch.randn(SCAN_HEADS, SCAN_DIM, generator=generator)
    z = torch.randn(8, SCAN_HEADS, SCAN_DIM, generator=generator).to(torch.bfloat16)

    # fp32 outputs: a bf16 result quantizes to ~0.4 %, which would swamp the
    # comparison the test is actually making.
    trace = ssd_chunk_scan_reference(
        x,
        C,
        B,
        dt_out,
        dA_cumsum,
        states,
        initial,
        seq_idx,
        cu,
        SCAN_CHUNK,
        out=torch.zeros(x.shape, dtype=torch.float32),
    )
    both = ssd_chunk_scan_reference(
        x,
        C,
        B,
        dt_out,
        dA_cumsum,
        states,
        initial,
        seq_idx,
        cu,
        SCAN_CHUNK,
        D_param=D_param,
        z=z,
        out=torch.zeros(x.shape, dtype=torch.float32),
    )
    # Reconstruct: skip, then SiLU gate, both in fp32.
    z32 = z.to(torch.float32)
    expected = (
        trace.float() + x.float() * D_param.view(1, SCAN_HEADS, SCAN_DIM)
    ) * z32 * torch.sigmoid(z32)
    assert torch.allclose(both.float(), expected, rtol=1e-5, atol=1e-5)
    # And the order matters: applying the gate first would not match.
    wrong_order = trace.float() * z32 * torch.sigmoid(z32) + x.float() * D_param.view(
        1, SCAN_HEADS, SCAN_DIM
    )
    assert not torch.allclose(both.float(), wrong_order, rtol=1e-3, atol=1e-3)


def test_zero_chunk_sequence_is_skipped_without_reading_states():
    offsets = [0, 8]
    x, C, B, dt_out, dA_cumsum, states, initial = _scan_case(offsets)
    # Two sequences, but the second has no chunks at all (its state row is
    # never consumed); the run must still complete and match the single-sequence
    # behaviour.
    seq_idx = torch.zeros(1, dtype=torch.int32)
    cu = torch.tensor(offsets, dtype=torch.int32)
    initial = torch.randn(3, SCAN_HEADS, SCAN_DIM, SCAN_DSTATE)
    out = ssd_chunk_scan_reference(
        x, C, B, dt_out, dA_cumsum, states, initial, seq_idx, cu, SCAN_CHUNK
    )
    assert torch.isfinite(out.float()).all()


# ---------------------------------------------------------------------------
# 6. CPU-only: the packed SSD orchestration, end to end
#
# ``mate.mamba.ssd_combined_fwd_varlen`` is checked against the *independent*
# FlashInfer-MUSA reference implementation that lives in the fork -- a separate
# transcription of the same contract, written by a different author for a different
# runtime. A disagreement means one of them is wrong; agreement over partial chunks,
# sequence boundaries, initial states and the epilogue is real evidence.
#
# The native tilelang stages need MUSA, so the stage table is patched with
# reference-backed launchers that keep the native calling convention (preallocated
# outputs, keyword-only flags). The plumbing under test -- metadata handling, the
# workspace, the entering-state rule, the return contract -- is the same code that
# runs on device.
#
# The oracle's file is not part of this repository: these tests skip when it is
# absent so the suite stays runnable anywhere. Point ``MATE_SSD_ORACLE`` at it to
# run elsewhere.
# ---------------------------------------------------------------------------

COMBINED_DEFAULT_ORACLE = pathlib.Path(
    "/home/xiaodongye/ws/flashinfer-nemotron-musa/flashinfer/mamba/musa_reference.py"
)

# The tolerance the fork's own device tests use for exactly this comparison:
# the native path rounds the running state to state_dtype at every chunk
# boundary while the oracle keeps it in fp32 across the sequence.
RTOL = 0.05
ATOL = 0.02


def _combined_load_oracle():
    path = pathlib.Path(os.environ.get("MATE_SSD_ORACLE", COMBINED_DEFAULT_ORACLE))
    if not path.exists():
        pytest.skip(f"FlashInfer-MUSA oracle not present at {path}")
    spec = importlib.util.spec_from_file_location("musa_reference_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ssd_combined_fwd_varlen_musa_reference


@pytest.fixture(scope="module")
def combined_oracle():
    return _combined_load_oracle()


def combined_reference_stage_table(round_operands=True):
    """Reference-backed launchers that keep the native calling convention.

    ``round_operands=False`` drops the pre-dot rounding to the activation dtype,
    turning the chain into a pure fp32 transcription of the same arithmetic.
    """
    def cumsum(
        dt, A, dt_bias, cu_chunk_seqlens, chunk_size, *, dt_softplus, dt_limit,
        dA_cumsum, dt_out,
    ):
        dA, processed = ssd_chunk_cumsum_reference(
            dt, A, dt_bias, cu_chunk_seqlens, chunk_size, dt_softplus, dt_limit
        )
        dA_cumsum.copy_(dA)
        dt_out.copy_(processed)
        return dA_cumsum, dt_out

    def chunk_state(x, B, dt_out, dA_cumsum, cu_chunk_seqlens, chunk_size, *, states):
        computed = ssd_chunk_state_reference(
            x, B, dt_out, dA_cumsum, cu_chunk_seqlens, chunk_size, round_operands
        )
        states.copy_(computed)
        return states

    def state_passing(
        states, dA_cumsum, initial_states, last_chunk_indices, seq_idx, *, state_dtype
    ):
        nchunks, heads, dim, dstate = states.shape
        block = dA_cumsum.shape[2]
        for chunk in range(nchunks):
            opens_sequence = chunk == 0 or int(seq_idx[chunk]) != int(seq_idx[chunk - 1])
            if opens_sequence:
                if initial_states is None:
                    running = torch.zeros(
                        (heads, dim, dstate), dtype=torch.float32, device=states.device
                    )
                else:
                    running = initial_states[int(seq_idx[chunk])].to(torch.float32)
            # decay the entering state across the chunk, then add the chunk's
            # own contribution; rounded at every boundary, as the fast path does
            decay = torch.exp(dA_cumsum[:, chunk, block - 1]).to(torch.float32)
            running = (
                decay.unsqueeze(1).unsqueeze(2) * running
                + states[chunk].to(torch.float32)
            )
            states[chunk].copy_(running.to(state_dtype))
        return states

    def bmm(C, B, cu_chunk_seqlens, chunk_size, *, cb):
        offsets = [int(v) for v in cu_chunk_seqlens.reshape(-1).tolist()]
        c32, b32 = C.to(torch.float32), B.to(torch.float32)
        cb.zero_()
        for chunk in range(len(offsets) - 1):
            start, end = offsets[chunk], offsets[chunk + 1]
            limit = end - start
            cb[chunk, :, :limit, :limit] = torch.einsum(
                "ign,jgn->gij", c32[start:end], b32[start:end]
            )
        return cb

    def chunk_scan(
        x, C, CB, dt_out, dA_cumsum, states, initial_states, seq_idx,
        cu_chunk_seqlens, out, *, D_param, z,
    ):
        return ssd_chunk_scan_reference(
            x,
            C,
            torch.zeros_like(C),
            dt_out,
            dA_cumsum,
            states,
            initial_states,
            seq_idx,
            cu_chunk_seqlens,
            dA_cumsum.shape[2],
            D_param=D_param,
            z=z,
            out=out,
            CB=CB,
            round_operands=round_operands,
        )

    return {
        "cumsum": cumsum,
        "chunk_state": chunk_state,
        "state_passing": state_passing,
        "bmm": bmm,
        "chunk_scan": chunk_scan,
    }


@pytest.fixture
def combined_reference_stages(monkeypatch):
    """Patch the five native stage launchers with reference-backed equivalents.

    Requested with ``@pytest.mark.usefixtures`` on each test of this section
    rather than ``autouse=True``: the merged module also holds the sections
    above, and an autouse fixture here would patch the stage table under them.
    """
    from mate import mamba

    table = combined_reference_stage_table()
    monkeypatch.setattr(mamba, "_stage", lambda name: table[name])
    mamba.reset_ssd_workspaces()
    yield
    mamba.reset_ssd_workspaces()


def _combined_tensors(case, device="cpu", seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokens = case["cu_chunk_seqlens"][-1]
    heads, dim, groups, dstate = case["heads"], case["dim"], case["groups"], case["dstate"]
    x = torch.randn(tokens, heads, dim, generator=generator).to(torch.bfloat16).to(device)
    dt = torch.randn(tokens, heads, generator=generator).abs().to(device) + 0.1
    A = (-torch.rand(heads, generator=generator) - 0.5).to(device)
    B = torch.randn(tokens, groups, dstate, generator=generator).to(torch.bfloat16).to(device)
    C = torch.randn(tokens, groups, dstate, generator=generator).to(torch.bfloat16).to(device)
    return x, dt, A, B, C


# The recipe from the specification's oracle section: one 5-token sequence and one
# 17-token sequence, so the first chunk is partial and both shapes of chunk appear.
COMBINED_TINY = {
    # sequence 0 is 5 tokens (one partial chunk), sequence 1 is 17 (three chunks)
    "cu_seqlens": [0, 5, 22],
    "cu_chunk_seqlens": [0, 5, 8, 16, 22],
    "last_chunk_indices": [0, 3],
    "seq_idx": [0, 1, 1, 1],
    "heads": 4,
    "dim": 8,
    "groups": 2,
    "dstate": 16,
    "chunk_size": 8,
}

COMBINED_FULL_CHUNKS = {
    "cu_seqlens": [0, 16, 32],
    "cu_chunk_seqlens": [0, 8, 16, 24, 32],
    "last_chunk_indices": [1, 3],
    "seq_idx": [0, 0, 1, 1],
    "heads": 4,
    "dim": 8,
    "groups": 2,
    "dstate": 16,
    "chunk_size": 8,
}


def _combined_run(case, combined_oracle, *, with_D=True, with_z=True, with_initial=False, extra=None):
    from mate.mamba import ssd_combined_fwd_varlen

    device = "cpu"
    x, dt, A, B, C = _combined_tensors(case, device)
    meta = {
        "cu_seqlens": torch.tensor(case["cu_seqlens"], dtype=torch.int32),
        "cu_chunk_seqlens": torch.tensor(case["cu_chunk_seqlens"], dtype=torch.int32),
        "last_chunk_indices": torch.tensor(case["last_chunk_indices"], dtype=torch.int32),
        "seq_idx": torch.tensor(case["seq_idx"], dtype=torch.int32),
    }
    generator = torch.Generator().manual_seed(11)
    batch = len(case["cu_seqlens"]) - 1
    kwargs = dict(
        chunk_size=case["chunk_size"],
        D=torch.randn(case["heads"], case["dim"], generator=generator) if with_D else None,
        z=torch.randn_like(x).to(torch.bfloat16) if with_z else None,
        dt_bias=torch.randn(case["heads"], generator=generator) * 0.1,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        initial_states=(
            torch.randn(
                batch, case["heads"], case["dim"], case["dstate"], generator=generator
            ).to(torch.bfloat16)
            if with_initial
            else None
        ),
    )
    if extra:
        kwargs.update(extra)

    mine_out = torch.zeros_like(x)
    expected_out = torch.zeros_like(x)
    mine_states = ssd_combined_fwd_varlen(
        x, dt, A, B, C, out=mine_out, return_intermediate_states=True, **meta, **kwargs
    )
    expected_states = combined_oracle(
        x, dt, A, B, C, out=expected_out, return_intermediate_states=True, **meta, **kwargs
    )
    return mine_out, mine_states, expected_out, expected_states


def combined_assert_states_close(mine, expected, nchunks, case):
    """Compare states under the documented rounding difference.

    The native path rounds the running state to ``state_dtype`` at every chunk
    boundary while the oracle keeps it in fp32 across a sequence, so the two
    disagree by the accumulated bf16 resolution -- and elements produced by
    cancellation inherit the error of the *scale*, not of their own size. The
    bound therefore scales with the largest state and with the number of
    boundaries instead of being a per-element tolerance.

    The tolerance must stay far below the size of a structural error: a mis-
    contracted operand axis in the state stage measured 3.8 here, an order of
    magnitude above this bound.
    """
    scale = float(expected.float().abs().max())
    bound = ATOL + RTOL * scale + nchunks * 2 ** -9 * scale
    worst = float((mine.float() - expected.float()).abs().max())
    assert worst <= bound, (
        f"{case}: worst state difference {worst:.4f} exceeds {bound:.4f} "
        f"(state scale {scale:.3f}, {nchunks} chunk boundaries)"
    )


def combined_assert_out_close(mine, expected):
    """Compare ``out`` with the fork's tolerance applied to the output scale.

    The fork's device tests use ``rtol=0.05, atol=0.02`` per element, which holds
    for model-like activations where outputs are O(1). These cases use
    unnormalized random inputs whose outputs reach 33, and the error floor set by
    the bf16 output quantization plus the pre-dot operand rounding is inherited
    from that scale -- cancellation then makes it large *relatively* on elements
    near zero while staying at 0.05% of the scale in absolute terms. Measured:
    max absolute difference 1.56e-02 on a scale of 33, with no element above the
    fork's own atol. A structural error moves every element, which the mean bound
    below catches.
    """
    scale = float(expected.float().abs().max())
    worst = float((mine.float() - expected.float()).abs().max())
    mean = float((mine.float() - expected.float()).abs().mean())
    assert worst <= ATOL + RTOL * scale, (
        f"worst output difference {worst:.4e} exceeds {ATOL + RTOL * scale:.4e} "
        f"(output scale {scale:.2f})"
    )
    assert mean <= 1e-3 * scale, (
        f"mean output difference {mean:.4e} exceeds 0.1% of the scale {scale:.2f}; "
        f"this is not a cancellation artifact"
    )


@pytest.mark.usefixtures("combined_reference_stages")
@pytest.mark.parametrize(
    "case",
    [COMBINED_TINY, COMBINED_FULL_CHUNKS],
    ids=["partial-first-chunk", "full-chunks"],
)
def test_out_and_states_match_the_independent_oracle(case, combined_oracle):
    mine_out, mine_states, expected_out, expected_states = _combined_run(case, combined_oracle)
    combined_assert_out_close(mine_out, expected_out)
    combined_assert_states_close(mine_states, expected_states, len(case["seq_idx"]), case["cu_chunk_seqlens"])


def _combined_states_fp32(case, combined_oracle, round_operands):
    """Run the chain in fp32 with the pre-dot rounding on or off."""
    from mate import mamba

    x, dt, A, B, C = _combined_tensors(case)
    meta = {
        "cu_seqlens": torch.tensor(case["cu_seqlens"], dtype=torch.int32),
        "cu_chunk_seqlens": torch.tensor(case["cu_chunk_seqlens"], dtype=torch.int32),
        "last_chunk_indices": torch.tensor(case["last_chunk_indices"], dtype=torch.int32),
        "seq_idx": torch.tensor(case["seq_idx"], dtype=torch.int32),
    }
    kwargs = dict(
        chunk_size=case["chunk_size"],
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        state_dtype=torch.float32,
        return_intermediate_states=True,
    )
    saved = mamba._stage
    table = combined_reference_stage_table(round_operands=round_operands)
    mamba._stage = lambda name: table[name]
    mamba.reset_ssd_workspaces()
    try:
        mine = mamba.ssd_combined_fwd_varlen(
            x, dt, A, B, C, out=torch.zeros_like(x), **meta, **kwargs
        )
    finally:
        mamba._stage = saved
        mamba.reset_ssd_workspaces()
    expected = combined_oracle(x, dt, A, B, C, out=torch.zeros_like(x), **meta, **kwargs)
    return mine, expected


@pytest.mark.usefixtures("combined_reference_stages")
def test_fp32_chain_is_equivalent_to_the_oracle(combined_oracle):
    """Disabling the pre-dot rounding turns the chain into a pure fp32
    transcription of the same arithmetic, and it then reproduces the oracle to
    fp32 summation order. This is what makes the shipped path's residual
    difference attributable to the rounding convention rather than to a defect."""
    mine, expected = _combined_states_fp32(COMBINED_TINY, combined_oracle, round_operands=False)
    assert mine.dtype == torch.float32
    scale = max(1.0, float(expected.abs().max()))
    worst = float((mine - expected).abs().max())
    assert worst <= 1e-4 * scale, (
        f"pure fp32 states differ by {worst:.3e} (scale {scale:.3f}); the "
        f"orchestration is not equivalent to the oracle"
    )


@pytest.mark.usefixtures("combined_reference_stages")
def test_the_pre_dot_rounding_costs_about_one_bf16_ulp(combined_oracle):
    """The shipped path rounds the scaled operands to the activation dtype before
    every dot, so it is *by construction* less accurate than an fp32 oracle. The
    measurement, which bounds how much a state comparison may be relaxed:

        rounding on  : 1.35e-02 = 0.123% of the state scale
        rounding off : 1.43e-06 = 0.000013%

    bf16's own resolution is 2**-9 = 0.195%, so the convention costs about one
    ulp and nothing more. A structural error in this stage measured 3.8 (44%),
    forty times this bound, so the bound still fails loudly on real defects.
    """
    mine, expected = _combined_states_fp32(COMBINED_TINY, combined_oracle, round_operands=True)
    scale = float(expected.abs().max())
    worst = float((mine - expected).abs().max())
    assert 0 < worst <= 2 ** -8 * scale, (
        f"shipped states differ from the oracle by {worst:.4e} "
        f"({worst / scale:.4%} of scale {scale:.2f}); expected a difference "
        f"bounded by the bf16 operand rounding"
    )


@pytest.mark.usefixtures("combined_reference_stages")
def test_partial_chunks_do_not_leave_garbage_in_the_state_buffer(combined_oracle):
    """The first chunk of the 5-token sequence is partial: its padded rows must not
    contribute, so the state must match the oracle rather than explode."""
    _, mine_states, _, expected_states = _combined_run(COMBINED_TINY, combined_oracle)
    assert torch.isfinite(mine_states.float()).all()
    combined_assert_states_close(mine_states, expected_states, len(COMBINED_TINY["seq_idx"]), COMBINED_TINY)


@pytest.mark.usefixtures("combined_reference_stages")
def test_final_states_are_selected_by_last_chunk_indices(combined_oracle):
    from mate.mamba import ssd_combined_fwd_varlen

    case = COMBINED_TINY
    x, dt, A, B, C = _combined_tensors(case)
    meta = {
        "cu_seqlens": torch.tensor(case["cu_seqlens"], dtype=torch.int32),
        "cu_chunk_seqlens": torch.tensor(case["cu_chunk_seqlens"], dtype=torch.int32),
        "last_chunk_indices": torch.tensor(case["last_chunk_indices"], dtype=torch.int32),
        "seq_idx": torch.tensor(case["seq_idx"], dtype=torch.int32),
    }
    full = ssd_combined_fwd_varlen(
        x, dt, A, B, C, chunk_size=case["chunk_size"], return_intermediate_states=True, **meta
    )
    final = ssd_combined_fwd_varlen(
        x, dt, A, B, C, chunk_size=case["chunk_size"], return_intermediate_states=False, **meta
    )
    expected = combined_oracle(
        x, dt, A, B, C, chunk_size=case["chunk_size"], return_intermediate_states=False, **meta
    )
    assert final.shape == (2, case["heads"], case["dim"], case["dstate"])
    assert final.dtype == torch.bfloat16  # C.dtype when state_dtype is unset
    torch.testing.assert_close(final.float(), expected.float(), rtol=RTOL, atol=ATOL)
    # The selected rows are exactly the last chunk of each sequence.
    for b, chunk in enumerate(case["last_chunk_indices"]):
        assert torch.equal(final[b], full[chunk])


@pytest.mark.usefixtures("combined_reference_stages")
def test_state_dtype_overrides_the_default_and_is_honoured(combined_oracle):
    from mate.mamba import ssd_combined_fwd_varlen

    case = COMBINED_FULL_CHUNKS
    x, dt, A, B, C = _combined_tensors(case)
    meta = {
        "cu_seqlens": torch.tensor(case["cu_seqlens"], dtype=torch.int32),
        "cu_chunk_seqlens": torch.tensor(case["cu_chunk_seqlens"], dtype=torch.int32),
        "last_chunk_indices": torch.tensor(case["last_chunk_indices"], dtype=torch.int32),
        "seq_idx": torch.tensor(case["seq_idx"], dtype=torch.int32),
    }
    states = ssd_combined_fwd_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=case["chunk_size"],
        state_dtype=torch.float32,
        return_intermediate_states=True,
        **meta,
    )
    assert states.dtype == torch.float32


@pytest.mark.usefixtures("combined_reference_stages")
def test_checkpoint_arguments_raise_instead_of_being_ignored(combined_oracle):
    from mate.mamba import ssd_combined_fwd_varlen

    case = COMBINED_TINY
    x, dt, A, B, C = _combined_tensors(case)
    meta = {
        "cu_seqlens": torch.tensor(case["cu_seqlens"], dtype=torch.int32),
        "cu_chunk_seqlens": torch.tensor(case["cu_chunk_seqlens"], dtype=torch.int32),
        "last_chunk_indices": torch.tensor(case["last_chunk_indices"], dtype=torch.int32),
        "seq_idx": torch.tensor(case["seq_idx"], dtype=torch.int32),
    }
    with pytest.raises(NotImplementedError, match="checkpoint_"):
        ssd_combined_fwd_varlen(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=case["chunk_size"],
            checkpoint_token_indices=torch.zeros(1, dtype=torch.int32),
            **meta,
        )


@pytest.mark.usefixtures("combined_reference_stages")
def test_workspace_is_reused_across_calls(combined_oracle):
    """Steady state must not allocate the stage scratch again."""
    from mate import mamba

    _combined_run(COMBINED_FULL_CHUNKS, combined_oracle)
    first = dict(mamba._WORKSPACES)
    _combined_run(COMBINED_FULL_CHUNKS, combined_oracle)
    second = dict(mamba._WORKSPACES)
    assert first.keys() == second.keys()
    for key, space in first.items():
        assert space.dt_out is second[key].dt_out
        assert space.states is second[key].states
        assert space.CB is second[key].CB


def test_stage_factories_reuse_the_kernel_compiled_for_the_same_arguments():
    """A stage launcher asks its factory for the kernel on every call; the factory
    must build it once per argument set and hand the same kernel back after that."""
    from mate.mamba_kernels.tilelang._kernel_cache import reuse_compiled

    built = []

    def factory(*args, **kwargs):
        built.append(args)
        return object()

    factory.symbol = "tilelang_ssd_stage"
    module = types.SimpleNamespace(tilelang_ssd_stage=factory)
    reuse_compiled(module, "tilelang_ssd_stage")
    reuse_compiled(module, "tilelang_ssd_stage")

    first = module.tilelang_ssd_stage(64, 128, torch.bfloat16)
    assert module.tilelang_ssd_stage(64, 128, torch.bfloat16) is first
    assert module.tilelang_ssd_stage(64, 128, torch.float32) is not first
    assert built == [(64, 128, torch.bfloat16), (64, 128, torch.float32)]
    # Attributes of the factory stay reachable through the cache.
    assert module.tilelang_ssd_stage.symbol == "tilelang_ssd_stage"


# ---------------------------------------------------------------------------
# 7. Device: the SSD prefill BMM kernel's input-layout guarantee
#
# Sections 4-6 model the prefill stages on the CPU, and no CPU test can hold a statement
# about a kernel signature. The BMM is the stage that declares its read-only inputs
# strided, so it gets the one device test that keeps that declaration honest: the same
# values must come out of a strided view and its dense copy, and a dstate axis that is
# not dense must be refused instead of being read as if it were.
# ---------------------------------------------------------------------------


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_ssd_bmm_kernel_reads_strided_inputs():
    launcher = mate.mamba._stage("bmm")
    chunk, groups, dstate, nchunks = 4, 2, 16, 2
    tokens = chunk * nchunks
    device = torch.device("musa")
    torch.manual_seed(0)

    cu_chunk_seqlens = torch.tensor([0, chunk, tokens], dtype=torch.int32, device=device)
    wide = torch.randn(tokens, groups, 2 * dstate, dtype=torch.bfloat16, device=device)
    cmat = wide[:, :, :dstate]
    bmat = wide.flip(-1)[:, :, dstate:]
    assert cmat.stride(-1) == 1 and not cmat.is_contiguous()

    strided = launcher(cmat, bmat, cu_chunk_seqlens, chunk)
    dense = launcher(cmat.contiguous(), bmat.contiguous(), cu_chunk_seqlens, chunk)
    assert torch.equal(strided, dense)

    expected = ssd_bmm_reference(
        cmat.contiguous(), bmat.contiguous(), cu_chunk_seqlens, chunk
    )
    torch.testing.assert_close(strided, expected, rtol=2e-2, atol=2e-2)

    with pytest.raises(RuntimeError, match="dense dstate axis"):
        launcher(wide[:, :, ::2], bmat, cu_chunk_seqlens, chunk)
    # A token stride that is not a multiple of the 8 bf16 elements (16 bytes) the
    # operand loads vectorize is the other half of the declaration: the kernel's
    # ``T.assume`` would be false, so the view has to be refused rather than read.
    narrow = torch.randn(tokens, groups, dstate + 4, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="multiple of 8"):
        launcher(narrow[:, :, :dstate], bmat, cu_chunk_seqlens, chunk)


# ---------------------------------------------------------------------------
# 8. Device: the strided input contract of the rest of the TileLang family
#
# Section 7 holds the BMM to its input-layout guarantee. The other kernels declare
# the same thing -- every read-only input as a ``T.StridedTensor`` with dynamic outer
# strides, plus a ``T.assume`` alignment the launcher validates -- so each of them
# gets the same pair of statements here: a strided view and its dense copy produce
# identical results, and a view that would make the declaration false is refused
# instead of read. The buffers a stage writes (``dA_cumsum``, ``dt_out``, ``CB``,
# ``states``, ``out``, the state pool) and the metadata callers build stay dense, and
# their refusal is the older ``is_contiguous()`` gate the CPU sections already cover.
# ---------------------------------------------------------------------------


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_ssd_chunk_cumsum_kernel_reads_a_strided_dt():
    launcher = mate.mamba._stage("cumsum")
    heads, chunk, nchunks = 8, 8, 2
    tokens = chunk * nchunks
    device = torch.device("musa")
    torch.manual_seed(0)

    cu_chunk_seqlens = torch.tensor([0, chunk, tokens], dtype=torch.int32, device=device)
    A = -torch.rand(heads, dtype=torch.float32, device=device)
    dt_bias = torch.rand(heads, dtype=torch.float32, device=device)
    wide = torch.randn(tokens, 2 * heads, dtype=torch.bfloat16, device=device)
    dt = wide[:, :heads]
    assert dt.stride(-1) == 1 and not dt.is_contiguous()

    strided = launcher(dt, A, dt_bias, cu_chunk_seqlens, chunk, dt_softplus=True)
    dense = launcher(
        dt.contiguous(), A, dt_bias, cu_chunk_seqlens, chunk, dt_softplus=True
    )
    assert torch.equal(strided[0], dense[0]) and torch.equal(strided[1], dense[1])

    with pytest.raises(RuntimeError, match="dense head axis"):
        launcher(wide[:, ::2], A, dt_bias, cu_chunk_seqlens, chunk, dt_softplus=True)
    # The staged copy moves dt in 8-bf16 vectors, so a 12-element token stride is not
    # something the kernel may be told to assume.
    narrow = torch.randn(tokens, heads + 4, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="multiple of 8"):
        launcher(
            narrow[:, :heads], A, dt_bias, cu_chunk_seqlens, chunk, dt_softplus=True
        )


@supported_musa_compute_capability([31])
def test_ssd_chunk_cumsum_packs_heads_only_while_the_grid_stays_wide():
    from mate.mamba_kernels.tilelang import ssd_chunk_cumsum

    pick = ssd_chunk_cumsum._heads_per_block
    # 64 heads: one head per CTA up to four chunks, two at eight, four from sixteen.
    assert [pick(64, n) for n in (1, 4, 8, 16, 32)] == [1, 1, 2, 4, 4]
    assert pick(6, 1000) == 2
    assert pick(5, 1000) == 1
    assert pick(64, 32, requested=8) == 8
    with pytest.raises(RuntimeError, match="divisor"):
        pick(64, 4, requested=3)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_ssd_chunk_cumsum_output_does_not_depend_on_heads_per_block():
    launcher = mate.mamba._stage("cumsum")
    heads, chunk = 64, 128
    device = torch.device("musa")
    torch.manual_seed(0)
    # Full, partial, single-token, full and partial chunks.
    cu_chunk_seqlens = torch.tensor(
        [0, 128, 200, 201, 329, 400], dtype=torch.int32, device=device
    )
    dt = torch.randn(400, heads, dtype=torch.bfloat16, device=device)
    A = -torch.rand(heads, dtype=torch.float32, device=device)
    dt_bias = torch.rand(heads, dtype=torch.float32, device=device)

    args = (dt, A, dt_bias, cu_chunk_seqlens, chunk)
    dA, dt_out = launcher(*args, dt_softplus=True)
    for hb in (1, 2, 4, 8):
        other = launcher(*args, dt_softplus=True, heads_per_block=hb)
        assert torch.equal(other[0], dA) and torch.equal(other[1], dt_out), hb
    with pytest.raises(RuntimeError, match="divisor"):
        launcher(*args, dt_softplus=True, heads_per_block=3)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_ssd_chunk_state_kernel_reads_strided_operands():
    launcher = mate.mamba._stage("chunk_state")
    heads, groups, dim, dstate, chunk, nchunks = 4, 2, 8, 16, 8, 2
    tokens = chunk * nchunks
    device = torch.device("musa")
    torch.manual_seed(0)

    cu_chunk_seqlens = torch.tensor([0, chunk, tokens], dtype=torch.int32, device=device)
    x_wide = torch.randn(tokens, heads, 2 * dim, dtype=torch.bfloat16, device=device)
    b_wide = torch.randn(tokens, groups, 2 * dstate, dtype=torch.bfloat16, device=device)
    x, b = x_wide[:, :, :dim], b_wide[:, :, :dstate]
    assert x.stride(-1) == 1 and not x.is_contiguous() and not b.is_contiguous()
    dt_out = torch.rand(heads, nchunks, chunk, dtype=torch.float32, device=device)
    dA_cumsum = -torch.rand(heads, nchunks, chunk, dtype=torch.float32, device=device)

    strided = launcher(x, b, dt_out, dA_cumsum, cu_chunk_seqlens, chunk)
    dense = launcher(
        x.contiguous(), b.contiguous(), dt_out, dA_cumsum, cu_chunk_seqlens, chunk
    )
    assert torch.equal(strided, dense)

    with pytest.raises(RuntimeError, match="dense innermost axis"):
        launcher(x_wide[:, :, ::2], b, dt_out, dA_cumsum, cu_chunk_seqlens, chunk)
    # x's head stride is 12 elements here: dense along dim, but not a 16-byte row step.
    odd = torch.randn(tokens, heads, dim + 4, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="multiple of 8"):
        launcher(odd[:, :, :dim], b, dt_out, dA_cumsum, cu_chunk_seqlens, chunk)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_ssd_state_passing_kernel_reads_strided_initial_states():
    launcher = mate.mamba._stage("state_passing")
    heads, dim, dstate, chunk, nchunks = 2, 4, 16, 8, 2
    device = torch.device("musa")
    torch.manual_seed(0)

    states = torch.randn(nchunks, heads, dim, dstate, dtype=torch.float32, device=device)
    dA_cumsum = -torch.rand(heads, nchunks, chunk, dtype=torch.float32, device=device) - 0.1
    last_chunk_indices = torch.tensor([nchunks - 1], dtype=torch.int32, device=device)
    wide = torch.randn(1, heads, dim, 2 * dstate, dtype=torch.float32, device=device)
    initial = wide[:, :, :, :dstate]
    assert initial.stride(-1) == 1 and not initial.is_contiguous()

    # The kernel updates the pool in place, so each run gets its own copy of it.
    strided = launcher(
        states.clone(),
        dA_cumsum,
        initial,
        last_chunk_indices,
        state_dtype=torch.float32,
    )
    dense = launcher(
        states.clone(),
        dA_cumsum,
        initial.contiguous(),
        last_chunk_indices,
        state_dtype=torch.float32,
    )
    assert torch.equal(strided, dense)

    with pytest.raises(RuntimeError, match="dense dstate axis"):
        launcher(
            states.clone(),
            dA_cumsum,
            wide[:, :, :, ::2],
            last_chunk_indices,
            state_dtype=torch.float32,
        )
    # fp32 vectors 4 elements, so a 2-element dstate stride is the finest grain the
    # kernel cannot be told to assume away.
    odd = torch.randn(1, heads, dim, dstate + 2, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError, match="multiple of 4"):
        launcher(
            states.clone(),
            dA_cumsum,
            odd[:, :, :, :dstate],
            last_chunk_indices,
            state_dtype=torch.float32,
        )


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_ssd_chunk_scan_kernel_keeps_its_dense_contract():
    """``chunk_scan`` is the one stage that kept its dense input contract.

    Its strided declaration was tried on the device and dropped there: with the declaration
    and its hints in place a strided view returned different values than the same data read
    contiguously, while the identical pattern leaves the other stages bit-identical.

    The kernel is deliberately not executed here. Under the pinned TileLang its ``T.gemm``
    cannot be lowered at this test's shape - "m_warp * n_warp must equal num_warps, m_warp: 1,
    n_warp: 1, num_warps: 4" (gemm.cc:196) - and the dense baseline fails identically, so the
    shape is the limitation and not this contract. The production shapes compile and serve:
    the campaign's six rounds all ran this kernel. What this test holds is the contract, and
    the checks it asserts run before any kernel is built.
    """
    launcher = mate.mamba._stage("chunk_scan")
    heads, groups, dim, dstate, chunk = 4, 2, 8, 16, 8
    nchunks = 1
    tokens = chunk * nchunks
    device = torch.device("musa")
    torch.manual_seed(0)

    x_wide = torch.randn(tokens, heads, 2 * dim, dtype=torch.bfloat16, device=device)
    c_wide = torch.randn(tokens, groups, 2 * dstate, dtype=torch.bfloat16, device=device)
    z_wide = torch.randn(tokens, heads, 2 * dim, dtype=torch.bfloat16, device=device)
    d_wide = torch.randn(heads, 2 * dim, dtype=torch.float32, device=device)
    x, C = x_wide[:, :, :dim], c_wide[:, :, :dstate]
    z, D_param = z_wide[:, :, :dim], d_wide[:, :dim]
    assert x.stride(-1) == 1 and not x.is_contiguous()

    CB = torch.rand(nchunks, groups, chunk, chunk, dtype=torch.float32, device=device)
    dt_out = torch.rand(heads, nchunks, chunk, dtype=torch.float32, device=device)
    dA_cumsum = -torch.rand(heads, nchunks, chunk, dtype=torch.float32, device=device)
    states = torch.randn(nchunks, heads, dim, dstate, dtype=torch.bfloat16, device=device)
    seq_idx = torch.zeros(nchunks, dtype=torch.int32, device=device)
    cu_chunk_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device=device)

    def run(x_, C_, z_=None, D_=None):
        return launcher(
            x_,
            C_,
            CB,
            dt_out,
            dA_cumsum,
            states,
            None,
            seq_idx,
            cu_chunk_seqlens,
            torch.empty(tokens, heads, dim, dtype=torch.bfloat16, device=device),
            D_param=D_,
            z=z_,
            block_M=chunk,
        )

    # Every strided operand is refused rather than read with the wrong addressing.
    with pytest.raises(RuntimeError, match="must be contiguous"):
        run(x, C.contiguous())
    with pytest.raises(RuntimeError, match="must be contiguous"):
        run(x.contiguous(), C)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        run(x.contiguous(), C.contiguous(), z_=z)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        run(x.contiguous(), C.contiguous(), D_=D_param)
    # A dense call passes the contract checks and only then reaches the kernel, which this
    # shape cannot lower: the failure is the kernel's, and it is not a contract violation.
    with pytest.raises(Exception) as excinfo:
        run(x.contiguous(), C.contiguous())
    assert "must be contiguous" not in str(excinfo.value)


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_ssu_one_token_kernel_reads_strided_operands():
    """The pool operands ``B``/``C`` are read by lane-vectorized loads, so their
    strides are the alignment-bearing ones; ``x`` and ``z`` are read per element and
    only have to keep their dim axis dense."""
    case = _ssu_case()
    device = _ssu_device()
    torch.manual_seed(0)
    b_wide = torch.randn(
        SSU_BATCH, SSU_GROUPS, 2 * SSU_DSTATE, dtype=torch.bfloat16, device=device
    )
    c_wide = torch.randn_like(b_wide)
    b_view, c_view = b_wide[:, :, :SSU_DSTATE], c_wide[:, :, :SSU_DSTATE]
    assert b_view.stride(-1) == 1 and not b_view.is_contiguous()

    # Both arms read the same values; the view is the only difference. Sampling a second tensor
    # for the dense arm would compare different data and fail whatever the kernel did.
    strided_case = dict(case, state=case["state"].clone(), B=b_view, C=c_view)
    dense_case = dict(
        case,
        state=case["state"].clone(),
        B=b_view.contiguous(),
        C=c_view.contiguous(),
    )
    strided_out = _ssu_run_kernel(strided_case)
    dense_out = _ssu_run_kernel(dense_case)
    assert torch.equal(strided_out, dense_out)
    assert torch.equal(strided_case["state"], dense_case["state"])

    with pytest.raises(RuntimeError, match="dense dstate axis"):
        _ssu_run_kernel(dict(case, state=case["state"].clone(), B=b_wide[:, :, ::2]))
    odd = torch.randn(
        SSU_BATCH, SSU_GROUPS, SSU_DSTATE + 4, dtype=torch.bfloat16, device=device
    )
    with pytest.raises(RuntimeError, match="multiple of 8"):
        _ssu_run_kernel(
            dict(case, state=case["state"].clone(), C=odd[:, :, :SSU_DSTATE])
        )


@supported_musa_compute_capability([31])
@torch.inference_mode
def test_ssu_packed_kernel_reads_strided_operands():
    """The packed route accepts the same views as the one-token kernel; the slot
    tables and the sequence metadata stay dense."""
    device = _mtp_device()
    case = _mtp_case([3, 3], accepted=[2, 3])
    rows = case["x"].shape[0]
    torch.manual_seed(0)
    b_wide = torch.randn(
        rows, MTP_GROUPS, 2 * MTP_DSTATE, dtype=MTP_IO_DTYPE, device=device
    )
    c_wide = torch.randn_like(b_wide)
    x_wide = torch.randn(rows, MTP_HEADS, 2 * MTP_DIM, dtype=MTP_IO_DTYPE, device=device)
    x_view, b_view, c_view = (
        x_wide[:, :, :MTP_DIM],
        b_wide[:, :, :MTP_DSTATE],
        c_wide[:, :, :MTP_DSTATE],
    )
    assert x_view.stride(-1) == 1 and not x_view.is_contiguous()

    # Same values in both arms; the view is the only difference between them.
    strided_case = dict(case, state=case["state"].clone(), x=x_view, B=b_view, C=c_view)
    dense_case = dict(
        case,
        state=case["state"].clone(),
        x=x_view.contiguous(),
        B=b_view.contiguous(),
        C=c_view.contiguous(),
    )
    strided_out = _mtp_run_kernel(strided_case)
    dense_out = _mtp_run_kernel(dense_case)
    assert torch.equal(strided_out, dense_out)
    assert torch.equal(strided_case["state"], dense_case["state"])

    with pytest.raises(RuntimeError, match="dense dstate axis"):
        _mtp_run_kernel(dict(case, state=case["state"].clone(), B=b_wide[:, :, ::2]))
    odd = torch.randn(
        rows, MTP_GROUPS, MTP_DSTATE + 4, dtype=MTP_IO_DTYPE, device=device
    )
    with pytest.raises(RuntimeError, match="multiple of 8"):
        _mtp_run_kernel(
            dict(case, state=case["state"].clone(), C=odd[:, :, :MTP_DSTATE])
        )


# ---------------------------------------------------------------------------
# 9. Source: every name a ``T.assume`` hints is bound where the kernel is built
# ---------------------------------------------------------------------------
# A ``T.assume`` line is ordinary Python, evaluated while the kernel factory assembles the
# prim_func. A stride symbol that was only ever written inline inside a declaration -
# ``T.dynamic("x_stride_token")`` inside the ``T.StridedTensor`` strides tuple, never bound to
# a name - raises ``NameError`` at build time: after the launcher's shape checks have passed,
# before a single element is computed, and with no device-side diagnostic. In serving that is
# an engine that cannot start, which is how this one was found: the declarations were only
# exercised on a device once the hints had been added. The device tests above catch it too,
# but only where a MUSA device exists, so this checks the sources instead.
def test_every_assumed_stride_symbol_is_bound():
    """No ``T.assume`` may name an identifier its module never binds."""
    kernel_dir = pathlib.Path(mate.mamba.__file__).parent / "mamba_kernels" / "tilelang"
    assert kernel_dir.is_dir(), kernel_dir
    checked = 0
    for source in sorted(kernel_dir.glob("*.py")):
        text = source.read_text()
        hints = re.findall(r"T\.assume\(([^)]*)\)", text)
        if not hints:
            continue
        bound = set(re.findall(r"^\s*(\w+)\s*=", text, re.M))
        bound |= set(re.findall(r"\bfor\s+(\w+)\s+in\b", text))
        bound |= set(re.findall(r"^\s*import\s+(\w+)", text, re.M))
        for names in re.findall(r"^\s*from\s+\S+\s+import\s+([\w,\s]+)", text, re.M):
            bound |= {n.strip() for n in names.split(",") if n.strip()}
        for hint in hints:
            for name in re.findall(r"\b([A-Za-z_]\w*)\b", hint):
                if name in {"T", "if", "else", "and", "or", "not", "in", "is"}:
                    continue
                assert name in bound, (
                    f"{source.name}: T.assume names {name!r} in {hint.strip()!r}, "
                    "but that module never binds it"
                )
        checked += 1
    assert checked >= 6, f"only {checked} kernel modules declare T.assume hints"
