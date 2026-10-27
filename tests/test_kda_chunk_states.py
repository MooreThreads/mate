"""Chunk-boundary state snapshots for the fused KDA prefill API.

``chunk_kda(chunk_states=...)`` writes the recurrent state entering every chunk
into a caller-owned ``[chunk_rows, H, 128, 128]`` pool, which is what lets a
serving framework cache prefix states instead of recomputing a shared prefix.
``chunk_kda(checkpoint_every_n_tokens=C)`` writes the same state but only at
every ``C``-token boundary, packed per sequence.

Oracle notes (measured on MP31 at HEAD 9667670d + this feature):

* ``pool[0]`` is the incoming state bitwise, and ``pool[1]``/``pool[2]`` match an
  independent 16-token-chunk blockwise reference (below) to bf16 tolerance.
* An n-chunk run's ``final_state`` equals the (n+1)-chunk run's ``pool[n]``
  bitwise: the snapshot path and the validated final-state path agree.
* The "compare ``pool[i]`` against the final state of an i-chunk prefix run"
  oracle *is* usable on MP31 at this HEAD: ``pool[i]`` is bitwise identical to
  the final state of an i-chunk run for every scanned index up to 511
  (1, 2, 3, 4, 8, 16, 64, 128, 256, 511) at B=1/H=12 on T=8192 and T=2048, and
  for i=1,2,3 at the shape the tests below use.  An earlier revision of this
  file recorded the opposite (a 0.10-0.14 disagreement for i=1,2, blamed on the
  prepare workspace's ``k_restored`` depending on the total sequence length);
  that did not reproduce on MP31 after the prepare/recurrence rewrite.
* It does reproduce on MP32.  K1 writes total-length-dependent values for a
  fixed (chunk, head), so any oracle that relates a boundary in one run to a
  differently sized run fails there; see ``_cross_run_oracle_is_usable``.  The
  cross-run oracles below are therefore MP31-only, and every cadence test also
  carries a same-run check (the coarse cadence against the stride-1 cadence of
  the same total), which shares K1 and is exact on both arches.

The snapshot is activation-typed even when the recurrent state is fp32: the
kernel publishes the activation-typed state tile, so `chunk_states` still has to
match the state dtype while the checkpoint pool follows the io dtype.
"""

from __future__ import annotations

import pytest
import torch

from mate.jit import kda_ops
from mate.kda import _kda_checkpoint_rows, chunk_kda, kda_chunk_state_rows
from mate.testing import supported_musa_compute_capability

KDA_CHUNK = 16
HEAD = 128


def _get_runtime_device() -> torch.device:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device("musa")
    pytest.skip("MUSA is not available")


def _make_inputs(*, device, batch, tokens, hqk, hv, state_dtype, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    return dict(
        q=rn(batch, tokens, hqk, HEAD).to(device, torch.bfloat16),
        k=rn(batch, tokens, hqk, HEAD).to(device, torch.bfloat16),
        v=rn(batch, tokens, hv, HEAD).to(device, torch.bfloat16),
        g=rn(batch, tokens, hv, HEAD).to(device, torch.bfloat16),
        beta=rn(batch, tokens, hv).to(device, torch.bfloat16),
        state=(rn(batch, hv, HEAD, HEAD) * 0.05).to(device, state_dtype),
        a_log=(rn(hv) * 0.1).to(device),
        dt_bias=(rn(hv, HEAD) * 0.1).to(device),
    )


def _exp2_ftz_like(x):
    y = torch.exp2(torch.clamp(x, max=127.0))
    return torch.where(x < -126.0, torch.zeros_like(y), y)


def _normalize_last_dim(x):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1.0e-6)


def _cross_run_oracle_is_usable(device) -> bool:
    """Whether a boundary can be compared against a differently sized run.

    The cross-run oracles below are MP31-only.  MP32's K1 prepare kernel writes
    values that depend on the *total* token count for a fixed (chunk, head):
    measured on PH1S (MTT S6000, arch 3.2) at this commit and on pristine
    `origin/master` 5713c341, `workspace_k_restored[chunk]` for the same chunk
    and the same input prefix differs by up to 1.8e-1 between runs that differ
    only in their trailing chunk count, while `workspace_total` (the gate
    product) matches to 6e-13.  A NaN-prefilled workspace comes back fully
    written, so it is a K1 value defect rather than an uninitialised tail.  Any
    oracle that relates a boundary in one run to a *differently sized* run
    therefore cannot be applied on MP32; the MP32 paths below check cadences
    against each other within one run instead, which shares K1 and isolates K2.
    """
    # This release line renders only the MP31 arch tag, so a MUSA device here is
    # never MP32 and the cross-run oracles always apply.
    return True


@torch.inference_mode()
def reference_boundary_states(
    q, k, v, g, beta, initial_state, a_log, dt_bias, *, seq_lens, round_every_chunk=None
):
    """Independent blockwise reference at the kernel's chunk size.

    Returns, per sequence, the state entering every chunk in the pool's
    ``[H, V, K]`` layout.  The recurrence keeps the state in fp32 and only the
    consumed snapshot is rounded to the activation dtype, which is what the
    kernel does (its accumulator is fp32; commit_state publishes an
    activation-typed copy).

    ``round_every_chunk`` models the MP32 shared-state tile: MP32 carries the
    state in the activation-typed shared tile, so every chunk's update is
    rounded before the next Scale reads it, while MP31 keeps it in an fp32
    accumulator and rounds only the published snapshot.  It defaults to the arch
    of ``q`` so callers on MP32 hardware get the matching oracle.
    """
    if round_every_chunk is None:
        # MP31 carries the state in an fp32 accumulator and rounds only the
        # published snapshot, so there is no per-chunk rounding to model.
        round_every_chunk = False
    D = q.shape[-1]
    hv, hqk = v.shape[-2], q.shape[-2]
    grp = hv // hqk
    fq, fk = q.reshape(-1, hqk, D), k.reshape(-1, hqk, D)
    fv, fg, fb = v.reshape(-1, hv, D), g.reshape(-1, hv, D), beta.reshape(-1, hv)
    qn = _normalize_last_dim(fq.float()).to(q.dtype).float()
    kn = _normalize_last_dim(fk.float()).to(k.dtype).float()
    hidx = torch.arange(hv, device=q.device, dtype=torch.long) // grp
    log2e, lower_bound = 1.4426950408889634, -5.0

    def rw(x):
        return x.to(q.dtype).float()

    out, bos = [], 0
    for si, slen in enumerate(seq_lens):
        eos = bos + slen
        s = initial_state[si].float().transpose(-1, -2)
        acc = [s.transpose(-1, -2).clone()]
        for start in range(bos, eos, KDA_CHUNK):
            end = min(start + KDA_CHUNK, eos)
            actual = end - start
            qb = torch.zeros((hv, KDA_CHUNK, D), device=q.device, dtype=torch.float32)
            kb, vb, gb = (torch.zeros_like(qb) for _ in range(3))
            bb = torch.full(
                (hv, KDA_CHUNK), -80.0, device=q.device, dtype=torch.float32
            )
            qb[:, :actual] = qn[start:end].index_select(1, hidx).permute(1, 0, 2)
            kb[:, :actual] = kn[start:end].index_select(1, hidx).permute(1, 0, 2)
            vb[:, :actual] = fv[start:end].permute(1, 0, 2).float()
            gb[:, :actual] = fg[start:end].permute(1, 0, 2).float()
            bb[:, :actual] = fb[start:end].transpose(0, 1).float()
            gate = (
                lower_bound
                * log2e
                * torch.sigmoid(
                    torch.exp(a_log)[:, None, None] * (gb + dt_bias[:, None, :])
                )
            )
            if actual < KDA_CHUNK:
                gate[:, actual:] = 0.0
            gc = gate.cumsum(dim=1)
            gt = gc[:, -1]
            shift = 0.5 * (gc[:, 0] + gt)
            kd = rw(kb * _exp2_ftz_like(gc))
            kdt = rw(kb * _exp2_ftz_like(gc - shift[:, None]))
            ki = rw(kb * _exp2_ftz_like(shift[:, None] - gc))
            bs = torch.sigmoid(bb)
            lm = rw(torch.bmm(kdt, ki.transpose(1, 2)).tril(-1) * bs[:, :, None])
            a = torch.zeros(
                (hv, KDA_CHUNK, KDA_CHUNK), device=q.device, dtype=torch.float32
            )
            a[:, 0, 0] = bs[:, 0]
            for i in range(1, KDA_CHUNK):
                a[:, i, i] = bs[:, i]
                a[:, i, :i] = -torch.bmm(lm[:, i : i + 1, :i], a[:, :i, :i]).squeeze(1)
            a = rw(a)
            s_committed = rw(s)
            tmp = rw(vb - torch.bmm(kd, s_committed))
            u = rw(torch.bmm(a, tmp))
            kr = rw(kb * _exp2_ftz_like(gt[:, None] - gc))
            s = s * _exp2_ftz_like(gt)[:, None, :] + torch.bmm(kr.transpose(1, 2), u)
            if round_every_chunk:
                # MP32 stores the carried state in the activation-typed tile, so
                # the next chunk reads the rounded update back.
                s = rw(s)
            acc.append(s.transpose(-1, -2).clone())
        out.append(acc)
        bos = eos
    return out


def _run(t, *, tokens, pool=None, device):
    return chunk_kda(
        t["q"][:, :tokens],
        t["k"][:, :tokens],
        t["v"][:, :tokens],
        t["g"][:, :tokens],
        t["beta"][:, :tokens],
        initial_state=t["state"],
        output_final_state=True,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        chunk_states=pool,
    )


def test_chunk_states_match_reference_rows() -> None:
    device = _get_runtime_device()
    tokens, hv, hqk = 4 * KDA_CHUNK, 4, 2
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.bfloat16,
    )
    rows = kda_chunk_state_rows(batch=1, num_tokens=tokens)
    assert rows == 4
    # Over-provision: unwritten rows must stay untouched.
    pool = torch.full(
        (rows + 3, hv, HEAD, HEAD), float("nan"), device=device, dtype=torch.bfloat16
    )
    out, final = _run(t, tokens=tokens, pool=pool, device=device)

    # Row 0 is the state entering the first chunk, i.e. the incoming state.
    torch.testing.assert_close(pool[0], t["state"][0], rtol=0, atol=0)

    ref = reference_boundary_states(
        t["q"],
        t["k"],
        t["v"],
        t["g"],
        t["beta"],
        t["state"],
        t["a_log"],
        t["dt_bias"],
        seq_lens=[tokens],
    )[0]
    for i in (1, 2):
        torch.testing.assert_close(pool[i], ref[i].to(pool.dtype), rtol=2e-2, atol=2e-2)

    # The snapshot entering the last chunk is not the final state: it is the
    # final state of a run that stops one chunk earlier (checked bitwise).
    early = _run(t, tokens=(rows - 1) * KDA_CHUNK, device=device)[1]
    torch.testing.assert_close(pool[rows - 1], early[0], rtol=0, atol=0)

    assert torch.isnan(pool[rows:]).all(), (
        "snapshot wrote past the rows it was told to fill"
    )
    assert not torch.isnan(pool[:rows]).any(), (
        "a row it was told to fill stayed unwritten"
    )


def test_chunk_states_resume_from_snapshot() -> None:
    device = _get_runtime_device()
    tokens, hv, hqk = 4 * KDA_CHUNK, 4, 2
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.bfloat16,
    )
    rows = kda_chunk_state_rows(batch=1, num_tokens=tokens)
    pool = torch.full(
        (rows, hv, HEAD, HEAD), float("nan"), device=device, dtype=torch.bfloat16
    )
    _, final = _run(t, tokens=tokens, pool=pool, device=device)

    # Resuming from a snapshot must reproduce the full run's final state.
    for i in (rows - 2, rows - 1):
        resumed = chunk_kda(
            t["q"][:, i * KDA_CHUNK :],
            t["k"][:, i * KDA_CHUNK :],
            t["v"][:, i * KDA_CHUNK :],
            t["g"][:, i * KDA_CHUNK :],
            t["beta"][:, i * KDA_CHUNK :],
            initial_state=pool[i : i + 1].clone(),
            output_final_state=True,
            A_log=t["a_log"],
            dt_bias=t["dt_bias"],
        )[1]
        torch.testing.assert_close(resumed[0], final[0], rtol=2e-2, atol=2e-2)


def test_chunk_states_do_not_change_out_or_final_state() -> None:
    device = _get_runtime_device()
    tokens, hv, hqk = 3 * KDA_CHUNK, 4, 2
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.bfloat16,
    )
    rows = kda_chunk_state_rows(batch=1, num_tokens=tokens)
    pool = torch.full(
        (rows, hv, HEAD, HEAD), float("nan"), device=device, dtype=torch.bfloat16
    )
    out_with, final_with = _run(t, tokens=tokens, pool=pool, device=device)
    out_without, final_without = _run(t, tokens=tokens, device=device)
    # bitwise: the snapshot is a pure side effect
    assert torch.equal(out_with, out_without)
    assert torch.equal(final_with, final_without)


def test_chunk_states_deterministic() -> None:
    device = _get_runtime_device()
    tokens, hv, hqk = 4 * KDA_CHUNK, 4, 2
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.bfloat16,
    )
    rows = kda_chunk_state_rows(batch=1, num_tokens=tokens)
    pools = []
    for _ in range(3):
        pool = torch.full(
            (rows, hv, HEAD, HEAD), float("nan"), device=device, dtype=torch.bfloat16
        )
        _run(t, tokens=tokens, pool=pool, device=device)
        pools.append(pool.clone())
    assert torch.equal(pools[0], pools[1])
    assert torch.equal(pools[0], pools[2])


def test_chunk_states_varlen_rows_and_gaps() -> None:
    """Varlen slots each sequence at its own workspace row, not back to back."""
    device = _get_runtime_device()
    hv, hqk = 4, 2
    seq_lens = [2 * KDA_CHUNK, 3 * KDA_CHUNK]
    total = sum(seq_lens)
    t = _make_inputs(
        device=device, batch=1, tokens=total, hqk=hqk, hv=hv, state_dtype=torch.bfloat16
    )
    q, k, v, g, beta = (t[x].squeeze(0) for x in ("q", "k", "v", "g", "beta"))
    cu = torch.tensor([0, seq_lens[0], total], device=device, dtype=torch.int64)
    states = (
        torch.randn(len(seq_lens), hv, HEAD, HEAD, device=device, dtype=torch.bfloat16)
        * 0.05
    )

    rows = kda_chunk_state_rows(batch=1, num_tokens=total, cu_seqlens=cu)
    assert rows == 2 + 3 + len(seq_lens)
    pool = torch.full(
        (rows, hv, HEAD, HEAD), float("nan"), device=device, dtype=torch.bfloat16
    )
    chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=states.clone(),
        cu_seqlens=cu,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        chunk_states=pool,
    )

    # workspace_base = ceil(bos / kChunk) + seq_idx
    bases = [0, -(-seq_lens[0] // KDA_CHUNK) + 1]
    for seq_idx, base in enumerate(bases):
        torch.testing.assert_close(pool[base], states[seq_idx], rtol=0, atol=0)

    # A row that exists is fully written; rows the packed slot space skips are not.
    written = {bases[0], bases[0] + 1, bases[1], bases[1] + 1, bases[1] + 2}
    for r in written:
        assert not torch.isnan(pool[r]).any(), f"row {r} left unwritten"
    unused = [r for r in range(rows) if r not in written]
    assert unused, "expected the packed slot space to leave gaps"
    assert torch.isnan(pool[unused]).all()


def test_chunk_states_rejects_bad_pool() -> None:
    device = _get_runtime_device()
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=3 * KDA_CHUNK,
        hqk=2,
        hv=4,
        state_dtype=torch.bfloat16,
    )
    rows = kda_chunk_state_rows(batch=1, num_tokens=3 * KDA_CHUNK)

    with pytest.raises(ValueError, match="at least"):
        chunk_kda(
            t["q"],
            t["k"],
            t["v"],
            t["g"],
            t["beta"],
            initial_state=t["state"],
            A_log=t["a_log"],
            dt_bias=t["dt_bias"],
            chunk_states=torch.empty(
                (rows - 1, 4, HEAD, HEAD), device=device, dtype=torch.bfloat16
            ),
        )
    with pytest.raises(ValueError, match="contiguous"):
        chunk_kda(
            t["q"],
            t["k"],
            t["v"],
            t["g"],
            t["beta"],
            initial_state=t["state"],
            A_log=t["a_log"],
            dt_bias=t["dt_bias"],
            chunk_states=torch.empty(
                (rows, 4, 2 * HEAD, HEAD), device=device, dtype=torch.bfloat16
            )[:, :, :HEAD, :].transpose(2, 3),
        )
    with pytest.raises(ValueError, match="shape"):
        chunk_kda(
            t["q"],
            t["k"],
            t["v"],
            t["g"],
            t["beta"],
            initial_state=t["state"],
            A_log=t["a_log"],
            dt_bias=t["dt_bias"],
            chunk_states=torch.empty(
                (rows, 4, HEAD, HEAD // 2), device=device, dtype=torch.bfloat16
            ),
        )
    # fp32 pools are rejected: the published tile is activation-typed.
    with pytest.raises(NotImplementedError, match="fp32"):
        chunk_kda(
            t["q"],
            t["k"],
            t["v"],
            t["g"],
            t["beta"],
            initial_state=t["state"],
            A_log=t["a_log"],
            dt_bias=t["dt_bias"],
            chunk_states=torch.empty(
                (rows, 4, HEAD, HEAD), device=device, dtype=torch.float32
            ),
        )


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "seq_lens,is_varlen,has_state_in",
    [
        ([64], False, True),
        ([33, 33], False, True),
        ([17, 0, 33, 16], True, True),
        ([33], False, False),
    ],
    ids=["dense", "dense_batch_tail", "varlen_tail_gaps", "zero_initial"],
)
@torch.inference_mode()
def test_mp32_chunk_states_all_snapshots_controlled(
    dtype, seq_lens, is_varlen, has_state_in
) -> None:
    """Exercise every snapshot with exact, finite Scale + UK state updates.

    A hand-built K2 workspace isolates snapshot correctness from K1 rounding:
    decayed and P are zero, inverse is identity on active tokens, and each
    chunk adds one V vector into one logical key column while scaling S by 1/2.
    The reference rounds the state after each chunk, matching MP32's shared
    state representation. Both 64-value CTA slices contain distinct values.
    """
    from mate.jit.kda_ops import (
        get_kda_recurrence_ops_function_name,
        get_kda_recurrence_ops_module,
        make_kda_ops_config,
    )
    from mate.kda import _allocate_kda_workspace

    device = _get_runtime_device()
    heads, nseq = 3, len(seq_lens)
    batch = 1 if is_varlen else nseq
    tokens = sum(seq_lens) if is_varlen else seq_lens[0]
    chunks = -(-tokens // KDA_CHUNK)
    rows = chunks + nseq if is_varlen else batch * chunks
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device=device, dtype=torch.int64) if is_varlen else None
    workspace = _allocate_kda_workspace((rows, heads), device, dtype)
    host_workspace = [
        torch.zeros(tuple(field.shape), device="cpu", dtype=dtype)
        for field in workspace
    ]
    _, restored, inverse, _, total = host_workspace
    total.fill_(0.5)
    value_axis = torch.arange(HEAD, device="cpu", dtype=torch.float32)
    key_axis = torch.arange(HEAD, device="cpu", dtype=torch.float32)
    state = torch.empty((nseq, heads, HEAD, HEAD), device="cpu", dtype=dtype)
    for seq in range(nseq):
        for head in range(heads):
            state[seq, head] = (
                (17 * seq + 11 * head + 3 * value_axis[:, None] + 5 * key_axis) % 113
                - 56
            ) / 512
    values = torch.zeros((batch, tokens, heads, HEAD), device="cpu", dtype=dtype)
    expected_rows = {}
    expected_final = torch.empty_like(state)
    for seq, length in enumerate(seq_lens):
        current = state[seq].clone() if has_state_in else torch.zeros_like(state[seq])
        base = -(-offsets[seq] // KDA_CHUNK) + seq if is_varlen else seq * chunks
        for chunk in range(-(-length // KDA_CHUNK)):
            record = base + chunk
            expected_rows[record] = current.clone()
            actual = min(KDA_CHUNK, length - chunk * KDA_CHUNK)
            inverse[record, :, :actual, :actual] = torch.eye(
                actual, device="cpu", dtype=dtype
            )
            update = torch.zeros_like(current, dtype=torch.float32)
            token = (offsets[seq] if is_varlen else 0) + chunk * KDA_CHUNK
            for head in range(heads):
                key = (29 * chunk + 11 * head) % HEAD
                # LogicalToSlot from mp32_head_dim_slot_frame.hpp.
                slot = 32 * (key // 32) + 4 * (key % 8) + (key % 32) // 8
                restored[record, head, 0, slot] = 1
                vector = ((value_axis + 3 * seq + 7 * head) % 31 - 15) / 128
                values[0 if is_varlen else seq, token, head] = vector
                update[head, :, key] = vector
            current = (current.float() * 0.5 + update).to(dtype)
        expected_final[seq] = current
    for field, host_field in zip(workspace, host_workspace):
        field.copy_(host_field)

    config = make_kda_ops_config(
        dtype,
        state_dtype=dtype,
        cu_seqlens_dtype=torch.int64 if is_varlen else None,
        has_state_in=has_state_in,
        has_state_out=True,
        has_chunk_states_out=True,
        state_fp32=False,
        has_gate_params=False,
        is_varlen=is_varlen,
        normalize_qk=False,
    )
    module = get_kda_recurrence_ops_module(config)
    fn = module.get_function(get_kda_recurrence_ops_function_name(config))
    values = values.to(device)
    initial = state.to(device) if has_state_in else None
    final = torch.empty_like(state, device=device)
    out = torch.empty_like(values)
    indices = torch.arange(nseq, device=device, dtype=torch.int32)
    backing = torch.empty((rows + 2, heads, HEAD, HEAD), device=device, dtype=dtype)
    pool = backing[1:-1]
    untouched = [0, rows + 1] + [
        row + 1 for row in range(rows) if row not in expected_rows
    ]

    for _ in range(3):
        backing.fill_(float("nan"))
        final.fill_(float("nan"))
        out.fill_(float("nan"))
        fn(values, out, *workspace, indices, initial, final, cu, pool)
        observed = backing.cpu()
        for row, expected in expected_rows.items():
            torch.testing.assert_close(observed[row + 1], expected, rtol=0, atol=0)
        assert torch.isnan(observed[untouched]).all(), (
            "snapshot clobbered a guard or gap"
        )
        torch.testing.assert_close(final.cpu(), expected_final, rtol=0, atol=0)
        assert torch.count_nonzero(out).item() == 0
    print(
        f"MP32_SNAPSHOTS dtype={dtype} sequences={seq_lens} heads={heads} "
        f"active_rows={len(expected_rows)} repetitions=3 exact=PASS"
    )


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@torch.inference_mode()
def test_mp32_chunk_states_public_api(dtype) -> None:
    """The public wrapper dispatches both activation dtypes to MP32 snapshots."""
    device = _get_runtime_device()
    tokens, heads = 3 * KDA_CHUNK, 3
    zeros = torch.zeros((1, tokens, heads, HEAD), device=device, dtype=dtype)
    beta = torch.zeros((1, tokens, heads), device=device, dtype=dtype)
    pattern = torch.arange(heads * HEAD * HEAD, device="cpu", dtype=torch.float32)
    state = ((pattern % 113 - 56) / 512).reshape(1, heads, HEAD, HEAD).to(device, dtype)
    pool = torch.full((3, heads, HEAD, HEAD), float("nan"), device=device, dtype=dtype)
    out, final = chunk_kda(
        zeros,
        zeros,
        zeros,
        zeros,
        beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        chunk_states=pool,
    )
    torch.testing.assert_close(pool, state.expand_as(pool), rtol=0, atol=0)
    torch.testing.assert_close(final, state, rtol=0, atol=0)
    assert torch.count_nonzero(out).item() == 0


@supported_musa_compute_capability([32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("mismatch", ["pool", "state", "fp32_pool"])
def test_mp32_chunk_states_reject_mismatched_dtypes(dtype, mismatch) -> None:
    device = _get_runtime_device()
    other_dtype = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
    zeros = torch.zeros((1, KDA_CHUNK, 1, HEAD), device=device, dtype=dtype)
    beta = torch.zeros((1, KDA_CHUNK, 1), device=device, dtype=dtype)
    state = torch.zeros(
        (1, 1, HEAD, HEAD),
        device=device,
        dtype=torch.float32 if mismatch == "state" else dtype,
    )
    pool_dtype = (
        torch.float32
        if mismatch == "fp32_pool"
        else other_dtype
        if mismatch == "pool"
        else dtype
    )
    pool = torch.empty((1, 1, HEAD, HEAD), device=device, dtype=pool_dtype)
    with pytest.raises(
        (TypeError, NotImplementedError), match="state|dtype|activation|fp32"
    ):
        chunk_kda(
            zeros, zeros, zeros, zeros, beta, initial_state=state, chunk_states=pool
        )


@pytest.mark.parametrize("arch_list, expected_token", [("3.1", "mp31")])
def test_prepare_launch_policy_follows_the_arch_tag(
    monkeypatch, arch_list: str, expected_token: str
) -> None:
    """A cross build must size the metadata region for the arch it renders.

    ``MATE_MUSA_ARCH_LIST`` decides both the ArchTag the JIT renders (and
    with it the C++ ``SchedulePolicy::PrepareCtasPerMp``) and the host-side
    metadata capacity.  These were resolved separately once, so building for
    the other arch sized the region for the wrong CTA count.
    """
    monkeypatch.setenv("MATE_MUSA_ARCH_LIST", arch_list)
    assert kda_ops.resolve_kda_arch() == expected_token
    assert (
        kda_ops.resolve_kda_prepare_ctas_per_mp()
        == (kda_ops.KDA_PREPARE_CTAS_PER_MP[expected_token])
    )
    assert kda_ops._resolve_kda_arch_tag() == kda_ops.KDA_ARCH_TAGS[expected_token]


def test_jit_identity_is_arch_distinct(monkeypatch) -> None:
    """The generated symbol, source path and module cache carry the arch.

    The generated source path, the AOT path, the spec registry and the module
    cache are all derived from the function name, so a name without the arch in
    it let a cross build collide with -- or be served from -- the other arch.
    This release line only renders the MP31 arch tag, and asking for another
    arch has to fail loudly rather than build a collective it does not carry.
    """
    monkeypatch.setenv("MATE_MUSA_ARCH_LIST", "3.1")
    mp31_name = kda_ops.get_kda_recurrence_ops_function_name()
    mp31_key = kda_ops._module_cache_key(kda_ops._normalize_kda_ops_config(None))

    assert mp31_name.endswith("_arch_mp31")
    assert mp31_key[0] == ("arch", "mp31")
    # the extra key must not survive normalization into the rendered config
    assert "arch" not in kda_ops._normalize_kda_ops_config(dict(mp31_key))

    monkeypatch.setenv("MATE_MUSA_ARCH_LIST", "3.2")
    with pytest.raises(ValueError, match="mp31"):
        kda_ops.resolve_kda_arch()


def test_recurrence_identity_ignores_prepare_only_flags(monkeypatch) -> None:
    """K2 names and cache keys contain only recurrence-rendered axes."""
    monkeypatch.setenv("MATE_MUSA_ARCH_LIST", "3.1")
    base = kda_ops._normalize_kda_ops_config(None)
    variant = dict(base)
    variant.update(has_gate_params=False, normalize_qk=False, state_fp32=True)

    assert kda_ops.get_kda_recurrence_ops_function_name(
        base
    ) == kda_ops.get_kda_recurrence_ops_function_name(variant)
    assert kda_ops._recurrence_module_cache_key(
        base
    ) == kda_ops._recurrence_module_cache_key(variant)


def test_recurrence_state_mode_is_derived_from_state_dtype(monkeypatch) -> None:
    """The dropped ``state_fp32`` axis must not alter rendered K2 semantics."""
    monkeypatch.setenv("MATE_MUSA_ARCH_LIST", "3.1")
    bf16 = kda_ops._normalize_kda_ops_config(None)
    # A stale/direct caller flag cannot make the source differ from its name
    # and cache key: state_dtype_name is the authoritative state ABI.
    stale = dict(bf16, state_fp32=True)
    assert kda_ops._normalize_kda_ops_config(stale)["state_fp32"] is False

    fp32 = dict(
        bf16,
        state_element="float",
        state_dtype_name="fp32",
        state_fp32=False,
    )
    normalized_fp32 = kda_ops._normalize_kda_ops_config(fp32)
    assert normalized_fp32["state_fp32"] is True
    assert kda_ops.get_kda_recurrence_ops_function_name(
        bf16
    ) != kda_ops.get_kda_recurrence_ops_function_name(normalized_fp32)
    assert kda_ops._recurrence_module_cache_key(
        bf16
    ) != kda_ops._recurrence_module_cache_key(normalized_fp32)
    # Check the actual Jinja context consumed by the generated recurrence
    # source, rather than only the Python-side name/key helpers.
    assert kda_ops._render_config(bf16, "kda_bf16")["state_fp32"] == "false"
    assert kda_ops._render_config(normalized_fp32, "kda_fp32")["state_fp32"] == "true"


def test_recurrence_cadence_is_a_specialization_axis(monkeypatch) -> None:
    """The checkpoint cadence is compiled in, not passed at run time.

    The kernel folds the pool row mapping from
    ``Components::CheckpointEveryNTokens`` and ``static_assert``s that the cadence
    is a non-negative multiple of the chunk, so two cadences must not share a
    rendered source, a symbol or a cached module -- and the host has to refuse a
    cadence that cannot be rendered before anything is built.
    """
    monkeypatch.setenv("MATE_MUSA_ARCH_LIST", "3.1")
    packed = kda_ops._normalize_kda_ops_config(dict(has_chunk_states_out=True))
    c16 = kda_ops._normalize_kda_ops_config(
        dict(has_chunk_states_out=True, checkpoint_every_n_tokens=KDA_CHUNK)
    )
    c32 = kda_ops._normalize_kda_ops_config(
        dict(has_chunk_states_out=True, checkpoint_every_n_tokens=2 * KDA_CHUNK)
    )

    names = {
        kda_ops.get_kda_recurrence_ops_function_name(cfg) for cfg in (packed, c16, c32)
    }
    assert len(names) == 3
    keys = {kda_ops._recurrence_module_cache_key(cfg) for cfg in (packed, c16, c32)}
    assert len(keys) == 3
    rendered = kda_ops._render_config(c32, "kda_c32")
    assert rendered["checkpoint_every_n_tokens"] == 2 * KDA_CHUNK
    assert rendered["has_chunk_states_out"] == "true"
    # The kernel branches on this mode flag, not on the cadence value: a
    # non-positive cadence never reaches the renderer.
    assert rendered["has_checkpoint_cadence"] == "true"
    assert (
        kda_ops._render_config(packed, "kda_packed")["has_checkpoint_cadence"]
        == "false"
    )

    # A cadence without the pool is meaningless: normalization drops it, so the
    # feature-off identity (and its cached .mu) is unchanged.
    off = kda_ops._normalize_kda_ops_config(dict(checkpoint_every_n_tokens=KDA_CHUNK))
    assert off["checkpoint_every_n_tokens"] == 0
    assert kda_ops.get_kda_recurrence_ops_function_name(
        off
    ) == kda_ops.get_kda_recurrence_ops_function_name(
        kda_ops._normalize_kda_ops_config(None)
    )

    # The cadence must be renderable: non-negative and chunk-divisible.
    for bad in (1, -KDA_CHUNK, KDA_CHUNK + 1):
        with pytest.raises(ValueError, match="checkpoint_every_n_tokens"):
            kda_ops.make_kda_ops_config(
                torch.bfloat16,
                state_dtype=torch.bfloat16,
                has_state_in=True,
                has_state_out=True,
                has_chunk_states_out=True,
                checkpoint_every_n_tokens=bad,
                state_fp32=False,
                has_gate_params=True,
                is_varlen=False,
                normalize_qk=True,
            )


# ---------------------------------------------------------------------------
# checkpoint_every_n_tokens: cadence + compact per-sequence row layout
#
# Entry `j` of sequence `b` is the state entering token `j * C`, packed per
# sequence (base(b) = sum_{b' < b} ((len_b' - 1) // C + 1)), so the pool holds
# `sum_b ((len_b - 1) // C + 1)` valid rows in a `max(total // C + N, 1)`
# capacity.  cudnn-frontend's `checkpoint_every_n_tokens` has the same
# contract.
# ---------------------------------------------------------------------------


def test_checkpoint_rows_capacity() -> None:
    """The advertised capacity covers the exact packed row count."""
    cu = torch.tensor([0, 5, 16, 40], dtype=torch.int64)
    # Dense: one sequence per batch row.
    assert _kda_checkpoint_rows(batch=2, num_tokens=128, every_n_tokens=64) == (
        2 * 128 // 64 + 2
    )
    # Varlen: the packed token total plus one sequence per cu_seqlens interval.
    assert _kda_checkpoint_rows(
        batch=1, num_tokens=40, cu_seqlens=cu, every_n_tokens=16
    ) == (40 // 16 + 3)
    # Degenerate inputs still reserve at least one row.
    assert _kda_checkpoint_rows(batch=1, num_tokens=0, every_n_tokens=16) == 1
    with pytest.raises(ValueError):
        _kda_checkpoint_rows(batch=1, num_tokens=128, every_n_tokens=40)
    with pytest.raises(ValueError):
        _kda_checkpoint_rows(batch=1, num_tokens=128, every_n_tokens=0)
    with pytest.raises(ValueError):
        _kda_checkpoint_rows(batch=1, num_tokens=128, every_n_tokens=-16)


def _run_with_cadence(t, *, tokens, every_n_tokens, device, cu_seqlens=None):
    q, k, v, g, beta = (t[x][:, :tokens] for x in ("q", "k", "v", "g", "beta"))
    if cu_seqlens is not None:
        q, k, v, g, beta = (x.squeeze(0) for x in (q, k, v, g, beta))
    return chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=t["state"],
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        checkpoint_every_n_tokens=every_n_tokens,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    "every_n_tokens",
    [KDA_CHUNK, 2 * KDA_CHUNK, 4 * KDA_CHUNK],
    ids=["c16", "c32", "c64"],
)
def test_checkpoint_cadence_matches_reference(every_n_tokens: int) -> None:
    """Each entry equals the reference state at that token boundary.

    MP31-only: the oracle relates entries to separately sized prefix runs, which
    MP32's K1 defect rules out.  See `_cross_run_oracle_is_usable`.
    """
    device = _get_runtime_device()
    hv, hqk = 4, 2
    tokens = 8 * KDA_CHUNK
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.bfloat16,
    )
    rows = _kda_checkpoint_rows(
        batch=1, num_tokens=tokens, every_n_tokens=every_n_tokens
    )
    assert rows == tokens // every_n_tokens + 1

    out, final, checkpoints = _run_with_cadence(
        t, tokens=tokens, every_n_tokens=every_n_tokens, device=device
    )
    assert checkpoints.shape == (rows, hv, HEAD, HEAD)
    assert checkpoints.is_contiguous()
    assert checkpoints.dtype == t["q"].dtype

    # Entry 0 is the incoming state, bitwise.
    torch.testing.assert_close(checkpoints[0], t["state"][0], rtol=0, atol=0)

    stride = every_n_tokens // KDA_CHUNK
    valid = (tokens - 1) // every_n_tokens + 1

    # Same-total consistency, exact on every arch: the coarse cadence must be a
    # bitwise subset of the stride-1 cadence.  Both runs share K1, so this
    # isolates the row mapping and the snapshot store from K1's numerics.
    _, _, every_chunk = _run_with_cadence(
        t, tokens=tokens, every_n_tokens=KDA_CHUNK, device=device
    )
    for j in range(1, valid):
        torch.testing.assert_close(
            checkpoints[j], every_chunk[j * stride], rtol=0, atol=0
        )

    ref = reference_boundary_states(
        t["q"],
        t["k"],
        t["v"],
        t["g"],
        t["beta"],
        t["state"],
        t["a_log"],
        t["dt_bias"],
        seq_lens=[tokens],
    )[0]
    for j in range(1, valid):
        torch.testing.assert_close(
            checkpoints[j], ref[j * stride].to(checkpoints.dtype), rtol=2e-2, atol=2e-2
        )

    # Entries are bitwise the final state of the prefix run that ends there:
    # the cadence snapshot and the validated final-state path agree.  Checked at
    # the first and last valid entry; the reference covers the rest.
    for j in sorted({1, valid - 1}):
        prefix = _run(t, tokens=j * every_n_tokens, device=device)[1]
        torch.testing.assert_close(checkpoints[j], prefix[0], rtol=0, atol=0)

    # The end-of-sequence state is final_state, never a checkpoint.
    torch.testing.assert_close(final[0], _run(t, tokens=tokens, device=device)[1][0])


def test_checkpoint_cadence_does_not_change_out_or_final_state() -> None:
    """The checkpoint series is a pure side effect."""
    device = _get_runtime_device()
    tokens, hv, hqk = 6 * KDA_CHUNK, 4, 2
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.bfloat16,
    )
    out_plain, final_plain = _run(t, tokens=tokens, device=device)
    out_ckpt, final_ckpt, _ = _run_with_cadence(
        t, tokens=tokens, every_n_tokens=2 * KDA_CHUNK, device=device
    )
    assert torch.equal(out_ckpt, out_plain)
    assert torch.equal(final_ckpt, final_plain)


def test_checkpoint_cadence_without_final_state_returns_empty_final() -> None:
    """`output_final_state=False` still returns the series, with an empty final."""
    device = _get_runtime_device()
    tokens, hv, hqk = 4 * KDA_CHUNK, 4, 2
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.bfloat16,
    )
    out, final, checkpoints = chunk_kda(
        t["q"],
        t["k"],
        t["v"],
        t["g"],
        t["beta"],
        initial_state=t["state"],
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        checkpoint_every_n_tokens=2 * KDA_CHUNK,
    )
    assert final.numel() == 0
    assert checkpoints.shape[0] == _kda_checkpoint_rows(
        batch=1, num_tokens=tokens, every_n_tokens=2 * KDA_CHUNK
    )
    # Bitwise the same output as the plain call.
    plain = _run(t, tokens=tokens, device=device)[0]
    assert torch.equal(out, plain)


@supported_musa_compute_capability([31])
def test_checkpoint_cadence_varlen_prefix_rows() -> None:
    """Sequences pack back to back in cu_seqlens order.

    MP31-only: comparing the packed rows against the reference and against
    separately sized prefix runs needs the cross-run oracle that MP32's K1
    defect rules out.  See `_cross_run_oracle_is_usable`.
    """
    device = _get_runtime_device()
    hv, hqk = 4, 2
    every_n_tokens = 2 * KDA_CHUNK
    # The zero-length case lives in
    # `test_checkpoint_cadence_varlen_empty_sequence`; upstream d75e9cb2 made the
    # recurrence handle a zero-chunk work item, so it is covered separately.
    seq_lens = [3 * KDA_CHUNK, 5 * KDA_CHUNK]
    total = sum(seq_lens)
    t = _make_inputs(
        device=device, batch=1, tokens=total, hqk=hqk, hv=hv, state_dtype=torch.bfloat16
    )
    q, k, v, g, beta = (t[x].squeeze(0) for x in ("q", "k", "v", "g", "beta"))
    cu = torch.tensor([0, seq_lens[0], total], device=device, dtype=torch.int64)
    states = (
        torch.randn(len(seq_lens), hv, HEAD, HEAD, device=device, dtype=torch.bfloat16)
        * 0.05
    )

    rows = _kda_checkpoint_rows(
        batch=1, num_tokens=total, cu_seqlens=cu, every_n_tokens=every_n_tokens
    )
    assert rows == total // every_n_tokens + len(seq_lens)

    _, _, checkpoints = chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=states.clone(),
        output_final_state=True,
        cu_seqlens=cu,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        checkpoint_every_n_tokens=every_n_tokens,
    )
    assert checkpoints.shape[0] == rows

    counts = [(n - 1) // every_n_tokens + 1 for n in seq_lens]
    bases = [0, counts[0]]
    assert sum(counts) <= rows
    stride = every_n_tokens // KDA_CHUNK

    # Same-total consistency, exact on every arch: the coarse cadence must be a
    # bitwise subset of the stride-1 cadence of the *same* run.  The two
    # cadences pack sequences independently, so each has its own row bases.
    _, _, every_chunk = chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=states.clone(),
        output_final_state=True,
        cu_seqlens=cu,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        checkpoint_every_n_tokens=KDA_CHUNK,
    )
    counts16 = [-(-n // KDA_CHUNK) for n in seq_lens]
    bases16, running = [], 0
    for count in counts16:
        bases16.append(running)
        running += count
    for seq_idx, base in enumerate(bases):
        for j in range(1, counts[seq_idx]):
            torch.testing.assert_close(
                checkpoints[base + j],
                every_chunk[bases16[seq_idx] + j * stride],
                rtol=0,
                atol=0,
            )

    ref = reference_boundary_states(
        q, k, v, g, beta, states, t["a_log"], t["dt_bias"], seq_lens=seq_lens
    )
    for seq_idx, base in enumerate(bases):
        if counts[seq_idx] == 0:
            continue
        # Entry 0 of a sequence is its incoming state, bitwise.
        torch.testing.assert_close(checkpoints[base], states[seq_idx], rtol=0, atol=0)
        for j in range(1, counts[seq_idx]):
            torch.testing.assert_close(
                checkpoints[base + j],
                ref[seq_idx][j * stride].to(checkpoints.dtype),
                rtol=2e-2,
                atol=2e-2,
            )


def test_checkpoint_cadence_rejects_bad_arguments() -> None:
    device = _get_runtime_device()
    tokens, hv, hqk = 4 * KDA_CHUNK, 4, 2
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.bfloat16,
    )
    common = dict(
        initial_state=t["state"],
        output_final_state=True,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
    )
    args = (t["q"], t["k"], t["v"], t["g"], t["beta"])
    with pytest.raises(ValueError):
        chunk_kda(*args, checkpoint_every_n_tokens=40, **common)
    with pytest.raises(ValueError):
        chunk_kda(*args, checkpoint_every_n_tokens=-16, **common)
    # Two pools with different row mappings cannot both be written.
    with pytest.raises(ValueError):
        chunk_kda(
            *args,
            chunk_states=torch.empty(
                (kda_chunk_state_rows(batch=1, num_tokens=tokens), hv, HEAD, HEAD),
                device=device,
                dtype=torch.bfloat16,
            ),
            checkpoint_every_n_tokens=2 * KDA_CHUNK,
            **common,
        )
    # The pool-addressing mode is mutually exclusive with checkpoints, the same
    # way cudnn's `state_indices` is.
    pool_state = (
        torch.randn(1, hv, HEAD, HEAD, device=device, dtype=torch.bfloat16) * 0.05
    )
    with pytest.raises(ValueError):
        chunk_kda(
            *args,
            initial_state=pool_state,
            output_final_state=True,
            initial_state_indices=torch.zeros(1, device=device, dtype=torch.int32),
            A_log=t["a_log"],
            dt_bias=t["dt_bias"],
            checkpoint_every_n_tokens=2 * KDA_CHUNK,
        )
    # The legacy pool is caller-owned and must match the state dtype, so an fp32
    # state still cannot use `chunk_states` (the snapshot is activation-typed).
    fp32_state = t["state"].float()
    fp32_initial_pool = fp32_state.clone()
    with pytest.raises(NotImplementedError):
        chunk_kda(
            *args,
            initial_state=fp32_initial_pool,
            output_final_state=True,
            A_log=t["a_log"],
            dt_bias=t["dt_bias"],
            chunk_states=torch.empty(
                (kda_chunk_state_rows(batch=1, num_tokens=tokens), hv, HEAD, HEAD),
                device=device,
                dtype=torch.float32,
            ),
        )


def test_checkpoint_cadence_accepts_an_fp32_state() -> None:
    """`checkpoints` is io-dtype even when `initial_state` is fp32 (cudnn parity).

    cudnn's `state_checkpoints` is io-dtype regardless of the recurrent state
    dtype, and its FLA shim passes an fp32 `initial_state` together with a
    checkpoint cadence.
    """
    device = _get_runtime_device()
    every_n_tokens = 2 * KDA_CHUNK
    tokens, hv, hqk = 6 * KDA_CHUNK, 2, 1
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.float32,
    )
    fp32_state = t["state"].float()
    out, final, checkpoints = chunk_kda(
        t["q"],
        t["k"],
        t["v"],
        t["g"],
        t["beta"],
        initial_state=fp32_state,
        output_final_state=True,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        checkpoint_every_n_tokens=every_n_tokens,
    )
    assert final.dtype == torch.float32
    assert checkpoints.dtype == t["q"].dtype
    rows = _kda_checkpoint_rows(
        batch=1, num_tokens=tokens, every_n_tokens=every_n_tokens
    )
    assert checkpoints.shape[0] == rows
    # Entry 0 is the incoming state, rounded to the activation dtype.
    torch.testing.assert_close(
        checkpoints[0], fp32_state[0].to(checkpoints.dtype), rtol=0, atol=0
    )
    # Same-total consistency, exact on every arch (see `_cross_run_oracle_is_usable`).
    stride = every_n_tokens // KDA_CHUNK
    valid = (tokens - 1) // every_n_tokens + 1
    _, _, every_chunk = chunk_kda(
        t["q"],
        t["k"],
        t["v"],
        t["g"],
        t["beta"],
        initial_state=fp32_state,
        output_final_state=True,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        checkpoint_every_n_tokens=KDA_CHUNK,
    )
    for j in range(1, valid):
        torch.testing.assert_close(
            checkpoints[j], every_chunk[j * stride], rtol=0, atol=0
        )

    if _cross_run_oracle_is_usable(device):
        # Every valid entry matches the independent blockwise reference seeded
        # with the same fp32 state: the pool dtype is decoupled, the recurrence
        # is not.
        ref = reference_boundary_states(
            t["q"],
            t["k"],
            t["v"],
            t["g"],
            t["beta"],
            fp32_state,
            t["a_log"],
            t["dt_bias"],
            seq_lens=[tokens],
        )[0]
        for j in range(1, valid):
            torch.testing.assert_close(
                checkpoints[j],
                ref[j * stride].to(checkpoints.dtype),
                rtol=2e-2,
                atol=2e-2,
            )
    # An fp32 state that is exactly bf16-representable reproduces the
    # activation-typed run bitwise: only the container differs.
    bf16_state = fp32_state.to(torch.bfloat16)
    _, bf16_final, bf16_checkpoints = chunk_kda(
        t["q"],
        t["k"],
        t["v"],
        t["g"],
        t["beta"],
        initial_state=bf16_state.float(),
        output_final_state=True,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        checkpoint_every_n_tokens=every_n_tokens,
    )
    _, _, bf16_input_checkpoints = chunk_kda(
        t["q"],
        t["k"],
        t["v"],
        t["g"],
        t["beta"],
        initial_state=bf16_state,
        output_final_state=True,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        checkpoint_every_n_tokens=every_n_tokens,
    )
    # Only the valid prefix is defined: the capacity bound can leave tail rows
    # unwritten (cudnn pads its allocation the same way).
    torch.testing.assert_close(
        bf16_input_checkpoints[:valid], checkpoints[:valid], rtol=0, atol=0
    )
    torch.testing.assert_close(bf16_final[0], final[0], rtol=0, atol=0)


@supported_musa_compute_capability([32])
def test_mp32_checkpoint_cadence_matches_reference() -> None:
    """MP32 writes the compact cadence rows and they agree with its own run.

    A boundary in a six-chunk run cannot be compared against a separately sized
    prefix run on MP32: K1's per-chunk values depend on the total token count
    (see `_cross_run_oracle_is_usable`).  The state inside one run is still well
    defined, so this checks the cadence against the stride-1 cadence of the
    *same* run, which shares K1 and isolates the row mapping and the snapshot
    store.
    """
    device = _get_runtime_device()
    hv, hqk = 2, 1
    every_n_tokens = 2 * KDA_CHUNK
    tokens = 6 * KDA_CHUNK
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.bfloat16,
    )
    rows = _kda_checkpoint_rows(
        batch=1, num_tokens=tokens, every_n_tokens=every_n_tokens
    )
    _, final, checkpoints = _run_with_cadence(
        t, tokens=tokens, every_n_tokens=every_n_tokens, device=device
    )
    assert checkpoints.shape[0] == rows
    torch.testing.assert_close(checkpoints[0], t["state"][0], rtol=0, atol=0)
    stride = every_n_tokens // KDA_CHUNK
    valid = (tokens - 1) // every_n_tokens + 1
    _, _, every_chunk = _run_with_cadence(
        t, tokens=tokens, every_n_tokens=KDA_CHUNK, device=device
    )
    for j in range(1, valid):
        torch.testing.assert_close(
            checkpoints[j], every_chunk[j * stride], rtol=0, atol=0
        )
    # The last cadence row is the state entering the last cadence interval, not
    # the end-of-sequence state.
    assert not torch.equal(checkpoints[valid - 1], final[0])


def test_checkpoint_cadence_dense_batch_rows() -> None:
    """A dense batch packs one sequence's entries after another's."""
    device = _get_runtime_device()
    hv, hqk = 2, 1
    every_n_tokens = 2 * KDA_CHUNK
    tokens = 5 * KDA_CHUNK
    batch = 2
    t = _make_inputs(
        device=device,
        batch=batch,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.bfloat16,
    )
    rows = _kda_checkpoint_rows(
        batch=batch, num_tokens=tokens, every_n_tokens=every_n_tokens
    )
    # ceil(T / C) entries per sequence, in batch order.
    per_sequence = -(-tokens // every_n_tokens)
    assert rows >= batch * per_sequence

    _, _, checkpoints = _run_with_cadence(
        t, tokens=tokens, every_n_tokens=every_n_tokens, device=device
    )
    assert checkpoints.shape[0] == rows
    stride = every_n_tokens // KDA_CHUNK
    for b in range(batch):
        base = b * per_sequence
        torch.testing.assert_close(checkpoints[base], t["state"][b], rtol=0, atol=0)

    # Same-total consistency, exact on every arch (see `_cross_run_oracle_is_usable`):
    # each batch row's cadence entries are the stride-1 entries of that row.
    _, _, every_chunk = _run_with_cadence(
        t, tokens=tokens, every_n_tokens=KDA_CHUNK, device=device
    )
    for b in range(batch):
        base = b * per_sequence
        base16 = b * (-(-tokens // KDA_CHUNK))
        for j in range(1, per_sequence):
            torch.testing.assert_close(
                checkpoints[base + j], every_chunk[base16 + j * stride], rtol=0, atol=0
            )

    if not _cross_run_oracle_is_usable(device):
        return  # MP32's K1 defeats a cross-run oracle.

    for b in range(batch):
        base = b * per_sequence
        for j in range(1, per_sequence):
            prefix = chunk_kda(
                t["q"][b : b + 1, : j * every_n_tokens],
                t["k"][b : b + 1, : j * every_n_tokens],
                t["v"][b : b + 1, : j * every_n_tokens],
                t["g"][b : b + 1, : j * every_n_tokens],
                t["beta"][b : b + 1, : j * every_n_tokens],
                initial_state=t["state"][b : b + 1],
                output_final_state=True,
                A_log=t["a_log"],
                dt_bias=t["dt_bias"],
            )[1]
            torch.testing.assert_close(checkpoints[base + j], prefix[0], rtol=0, atol=0)


def test_checkpoint_cadence_exact_capacity_tail() -> None:
    """A trailing partial cadence fills the last row and leaves no padding.

    ``tokens = 5 * chunk`` with cadence ``2 * chunk`` has three valid entries
    (tokens 0, 32, 64) and a capacity of exactly three, so the pool has no
    unwritten tail: the last entry is the state entering token 64, not the
    end-of-sequence state at token 80.
    """
    device = _get_runtime_device()
    hv, hqk = 2, 1
    every_n_tokens = 2 * KDA_CHUNK
    tokens = 5 * KDA_CHUNK
    t = _make_inputs(
        device=device,
        batch=1,
        tokens=tokens,
        hqk=hqk,
        hv=hv,
        state_dtype=torch.bfloat16,
    )
    rows = _kda_checkpoint_rows(
        batch=1, num_tokens=tokens, every_n_tokens=every_n_tokens
    )
    valid = (tokens - 1) // every_n_tokens + 1
    assert rows == valid == 3

    _, final, checkpoints = _run_with_cadence(
        t, tokens=tokens, every_n_tokens=every_n_tokens, device=device
    )
    assert checkpoints.shape[0] == rows
    torch.testing.assert_close(checkpoints[0], t["state"][0], rtol=0, atol=0)
    # The last valid entry is a mid-sequence boundary, not the final state.
    prefix = _run(t, tokens=(valid - 1) * every_n_tokens, device=device)[1]
    torch.testing.assert_close(checkpoints[valid - 1], prefix[0], rtol=0, atol=0)
    assert not torch.equal(checkpoints[valid - 1], final[0])


def test_checkpoint_capacity_matches_cudnn_reference_formula() -> None:
    """Cross-check the pool contract against cudnn-frontend's own formula.

    ``emit_checkpoint_seq_descs`` (``frost/common/thd.py``) decides the valid
    rows with ``cnt = (len - 1) // every_n + 1`` (0 for an empty sequence) and
    prefix-sums it; ``run_kda_fwd`` allocates ``max(total // every_n + N, 1)``.
    Both must hold here, for dense-equivalent and ragged length sets.
    """

    def cudnn_valid_rows(lens: list[int], every_n: int) -> int:
        run = 0
        for length in lens:
            cnt = (length - 1) // every_n + 1
            run += cnt if length > 0 else 0
        return run

    cases = [
        [16],
        [15],
        [17],
        [16, 16],
        [0, 16],
        [16, 0, 16],
        [31, 33, 16],
        [100],
        [64, 1, 64],
        [0, 0, 0],
    ]
    for every_n in (16, 32, 64, 128):
        for lens in cases:
            offsets = [0]
            for length in lens:
                offsets.append(offsets[-1] + length)
            total = offsets[-1]
            cu = torch.tensor(offsets, dtype=torch.int64)
            rows = _kda_checkpoint_rows(
                batch=1, num_tokens=total, cu_seqlens=cu, every_n_tokens=every_n
            )
            # The allocated capacity is exactly cudnn's bound ...
            assert rows == max(total // every_n + len(lens), 1)
            # ... and it always covers the rows the kernel can actually write.
            assert rows >= cudnn_valid_rows(lens, every_n)
    # Dense is the same contract with one length per batch row.
    for every_n in (16, 64):
        for length in (1, 15, 16, 17, 64, 100):
            assert _kda_checkpoint_rows(
                batch=3, num_tokens=length, every_n_tokens=every_n
            ) == max(3 * length // every_n + 3, 1)


def test_checkpoint_cadence_varlen_empty_sequence() -> None:
    """An empty sequence contributes no rows and carries its state through.

    Upstream d75e9cb2 ("handle empty recurrence work before role dispatch") made
    the recurrence handle a zero-chunk work item; before that fix this input hung
    the MP31 kernel.  The compact base must skip the empty sequence, which is what
    the `if (length > 0)` guard in `chunk_kda_checkpoint_base` is for.
    """
    device = _get_runtime_device()
    hv, hqk = 4, 2
    every_n_tokens = 2 * KDA_CHUNK
    seq_lens = [3 * KDA_CHUNK, 0, 4 * KDA_CHUNK]
    total = sum(seq_lens)
    t = _make_inputs(
        device=device, batch=1, tokens=total, hqk=hqk, hv=hv, state_dtype=torch.bfloat16
    )
    q, k, v, g, beta = (t[x].squeeze(0) for x in ("q", "k", "v", "g", "beta"))
    cu = torch.tensor(
        [0, seq_lens[0], seq_lens[0], total], device=device, dtype=torch.int64
    )
    states = (
        torch.randn(len(seq_lens), hv, HEAD, HEAD, device=device, dtype=torch.bfloat16)
        * 0.05
    )

    rows = _kda_checkpoint_rows(
        batch=1, num_tokens=total, cu_seqlens=cu, every_n_tokens=every_n_tokens
    )
    assert rows == total // every_n_tokens + len(seq_lens)

    _, final, checkpoints = chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=states.clone(),
        output_final_state=True,
        cu_seqlens=cu,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        checkpoint_every_n_tokens=every_n_tokens,
    )

    counts = [0 if n == 0 else (n - 1) // every_n_tokens + 1 for n in seq_lens]
    assert counts[1] == 0
    bases = [0, counts[0], counts[0]]
    for seq_idx, base in enumerate(bases):
        if counts[seq_idx] == 0:
            continue
        # Entry 0 is the incoming state, bitwise.
        torch.testing.assert_close(checkpoints[base], states[seq_idx], rtol=0, atol=0)

    # Same-total consistency, exact on every arch (see `_cross_run_oracle_is_usable`).
    stride = every_n_tokens // KDA_CHUNK
    _, _, every_chunk = chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=states.clone(),
        output_final_state=True,
        cu_seqlens=cu,
        A_log=t["a_log"],
        dt_bias=t["dt_bias"],
        checkpoint_every_n_tokens=KDA_CHUNK,
    )
    bases16 = [0, -(-seq_lens[0] // KDA_CHUNK), -(-seq_lens[0] // KDA_CHUNK)]
    for seq_idx, base in enumerate(bases):
        for j in range(1, counts[seq_idx]):
            torch.testing.assert_close(
                checkpoints[base + j],
                every_chunk[bases16[seq_idx] + j * stride],
                rtol=0,
                atol=0,
            )

    if _cross_run_oracle_is_usable(device):
        ref = reference_boundary_states(
            q, k, v, g, beta, states, t["a_log"], t["dt_bias"], seq_lens=seq_lens
        )
        for seq_idx, base in enumerate(bases):
            for j in range(1, counts[seq_idx]):
                torch.testing.assert_close(
                    checkpoints[base + j],
                    ref[seq_idx][j * stride].to(checkpoints.dtype),
                    rtol=2e-2,
                    atol=2e-2,
                )

    # The empty sequence carries its incoming state through to final_state.
    torch.testing.assert_close(final[1], states[1], rtol=0, atol=0)
