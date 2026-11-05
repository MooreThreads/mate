"""Host-side contract tests for the dense W4A16 GEMM.

These exercise the parts that need no device: the packed-weight layout contract, the zero-point
algebra, the act-order gate, input validation, and the JIT source rendering. Everything below the
rendered source is covered by the on-device tests.
"""

import pathlib

import pytest
import torch

from mate.jit.gemm.dense_w4a16_gemm import (
    DENSE_W4A16_GEMM_NIBBLE_BIAS,
    MATE_W4A16_WEIGHT_LAYOUT,
    apply_gidx_permutation,
    DENSE_W4A16_GEMM_TILE,
    DENSE_W4A16_GEMM_TILE_NARROW,
    DENSE_W4A16_GEMM_WIDE_N_TILE,
    dense_w4a16_gemm_dispatch_name,
    dense_w4a16_gemm_mutlass,
    dense_w4a16_gemm_plan,
    dense_w4a16_gemm_tile_for,
    dense_w4a16_plan_cost,
    dense_w4a16_splitk_for,
    dense_w4a16_use_prefill,
    fold_zero_points,
    render_dense_w4a16_gemm_source,
)


def test_weight_layout_contract_is_explicit() -> None:
    assert MATE_W4A16_WEIGHT_LAYOUT.even_k_low_nibble
    assert MATE_W4A16_WEIGHT_LAYOUT.nibble_bias == DENSE_W4A16_GEMM_NIBBLE_BIAS
    assert MATE_W4A16_WEIGHT_LAYOUT.scale_group_size == 128


def test_dispatch_name_does_not_depend_on_m() -> None:
    # One compiled op serves every M, so the name is a function of the dtypes and the zero-point mode
    # only. A name that grew with M would mean a shape-independent trace had been frozen somewhere.
    # The trailing tag is the launcher ABI, which the split-K arguments bumped: it is part of the JIT
    # cache key, so a module compiled against the old signature is never loaded.
    assert dense_w4a16_gemm_dispatch_name(torch.float16, False) == (
        "mp31_dense_w4a16_gemm_fp16_32x256x256_s2_sym_sk2"
    )
    assert dense_w4a16_gemm_dispatch_name(torch.float16, True) == (
        "mp31_dense_w4a16_gemm_fp16_32x256x256_s2_zp_sk2"
    )
    assert dense_w4a16_gemm_dispatch_name(torch.bfloat16, True) != (
        dense_w4a16_gemm_dispatch_name(torch.float16, True)
    )


def test_wide_n_small_m_tile_takes_the_gemm_family() -> None:
    # The decode gate_up shape (M=8, N=17920) is too wide for the narrow rectangle and too small for the
    # prefill band, so it used to run the prefill kernel at a tile with no tile_m/tile_k freedom and
    # measured 43.98 us against 31.17 us on the wide GEMM tile. The two decisions must agree: the tile
    # selector returns the wide tile exactly where the family boundary routes away from prefill, because
    # the prefill spec generator hard-guards 128x?x128 and cannot express this tile.
    wide = DENSE_W4A16_GEMM_WIDE_N_TILE
    assert wide == (16, 128, 256, 3)
    for n in (8960, 17920):
        assert dense_w4a16_gemm_tile_for(8, n) == wide
        assert dense_w4a16_gemm_tile_for(16, n) == wide
        assert dense_w4a16_use_prefill(8, n) is False
        assert dense_w4a16_use_prefill(16, n) is False
    # M=17..32 stays on prefill: at M=32 the wide tile gives 280 CTAs over 6 waves and measures 21% worse.
    assert dense_w4a16_use_prefill(32, 17920) is True
    assert dense_w4a16_gemm_tile_for(32, 17920) != wide
    # N <= 4096 is the narrow rectangle, which keeps its own tile and its own prefill decision.
    assert dense_w4a16_gemm_tile_for(8, 1536) == DENSE_W4A16_GEMM_TILE_NARROW
    assert dense_w4a16_use_prefill(8, 1536) is False


def test_gemm_plan_prices_the_tile_against_the_k_axis() -> None:
    # The tile decides the tile count, the split turns that count into work items, and the product is
    # how many K blocks each CTA streams -- the number the per-CTA fixed cost multiplies. So the tile
    # cannot be a rectangle in (M, N): `down` at K=8960 has 24 tiles on the narrow tile, so filling 60
    # MPs makes 120 items and leaves 14 blocks per CTA, while the wide tile's 12 tiles take the same
    # 60-item grid at 7 blocks each. Measured on S5000 (60 MPs, cold cache, best split per tile):
    #   down    M=8 24.6 -> 17.3   M=16 24.5 -> 17.2   M=32 40.5 -> 26.0
    #   qkv     M=8 14.2 -> 12.2 (N=2048)   M=16 13.2 -> 11.7   M=32 19.5 -> 13.7
    #   gate_up M=8 42.7 -> 29.5   (N=17920, already on the wide tile)
    #   o       M=8 12.33 against 12.24 -- a tie, which the model resolves by its own prediction.
    wide = DENSE_W4A16_GEMM_WIDE_N_TILE
    narrow = DENSE_W4A16_GEMM_TILE_NARROW
    for m in (8, 16, 32):
        # 12 tiles x 5 = 60 work items, one per CTA, seven K blocks each.
        assert dense_w4a16_gemm_plan(m, 1536, 8960) == (wide, 5)
    assert dense_w4a16_gemm_plan(8, 2048, 1536)[0] == wide
    assert dense_w4a16_gemm_plan(8, 17920, 1536) == (wide, 1)
    # The narrow tile keeps the shapes whose tile count already fills the wave, and the model must not
    # widen a shape to no purpose: at M=16, N=1536 the pair measures 12.00 us against the wide tile's
    # 12.39, so the cheaper prediction and the cheaper measurement agree.
    assert dense_w4a16_gemm_plan(16, 1536, 1536) == (narrow, 2)
    assert dense_w4a16_gemm_plan(8, 1536, 1536) == (narrow, 2)
    # At M=32 the same shape is 48 narrow tiles and measures 13.07 against the wide tile's 12.50.
    assert dense_w4a16_gemm_plan(32, 1536, 1536) == (wide, 2)
    # A split is never wider than the K axis it splits.
    tile, split = dense_w4a16_gemm_plan(8, 1536, 256)
    assert split == 1 and tile == narrow


def test_splitk_policy_leaves_a_full_wave_alone(monkeypatch) -> None:
    # The split exists to fill the wave the launch gets. A launch that already has a wave of tiles
    # must not be touched, and a narrow one takes the largest split whose partials stay small.
    narrow = DENSE_W4A16_GEMM_TILE_NARROW
    assert dense_w4a16_splitk_for(8, 1536, narrow) == 2  # 24 tiles
    assert dense_w4a16_splitk_for(8, 3840, narrow) == 1  # exactly one wave
    assert dense_w4a16_splitk_for(8, 3968, narrow) == 1  # past one wave
    assert dense_w4a16_splitk_for(16, 128, narrow) == 8  # single tile, capped
    # A split pays only while every CTA keeps more than one K block. At K=1536 (6 blocks) a 2-way
    # split still measures 14.4 -> 12.2 us at M=8 and 13.6 -> 12.0 us at M=16, because 24 tiles leave
    # 36 of the 60 MPs without a CTA. Where the split would leave one block per CTA the reduction
    # dominates: [M=16, K=512, N=128] has 2 blocks and measures 10.5 -> 12.3 us, so it must not split.
    # 35 K blocks is the long-K regime, where the wave-aligned model takes over: see
    # test_splitk_wave_model_covers_the_long_k_shapes for the measurement behind the 5.
    assert dense_w4a16_splitk_for(8, 1536, narrow, k_blocks=35) == 5
    assert dense_w4a16_splitk_for(8, 1536, narrow, k_blocks=6) == 2
    assert dense_w4a16_splitk_for(16, 128, narrow, k_blocks=4) == 1
    assert dense_w4a16_splitk_for(16, 128, narrow, k_blocks=2) == 1
    # The floor sits between 3 and 4 blocks, and the batch dimension moves the answer the other way:
    # at M=32 the same shape is 48 tiles and a split measures 13.2 -> 15.7 us, so it must stay whole.
    assert dense_w4a16_splitk_for(8, 1536, narrow, k_blocks=3) == 1
    assert dense_w4a16_splitk_for(8, 1536, narrow, k_blocks=4) == 2
    assert dense_w4a16_splitk_for(16, 1536, narrow, k_blocks=6) == 2
    assert dense_w4a16_splitk_for(32, 1536, narrow, k_blocks=6) == 1
    # A wave is the machine the launch gets, not a constant. The same 24-tile shape is two units on
    # both shipping parts (48 and 60 MPs) but already a whole launch on a 24-MP one, where the unsplit
    # grid covers the machine and a split would only add the reduction.
    assert dense_w4a16_splitk_for(8, 1536, narrow, k_blocks=6, num_mps=60) == 2
    assert dense_w4a16_splitk_for(8, 1536, narrow, k_blocks=6, num_mps=48) == 2
    assert dense_w4a16_splitk_for(8, 1536, narrow, k_blocks=6, num_mps=24) == 1
    # 48 tiles is one wave on a 48-MP part and stays whole on 60, where the fill rule also declines.
    assert dense_w4a16_splitk_for(32, 1536, narrow, k_blocks=6, num_mps=48) == 1
    assert dense_w4a16_splitk_for(32, 1536, narrow, k_blocks=6, num_mps=60) == 1
    with pytest.raises(ValueError, match="num_mps must be a positive MP count"):
        dense_w4a16_splitk_for(8, 1536, narrow, num_mps=0)
    # The override is a benchmark seam, and it is the only way to force a split count.
    monkeypatch.setattr(
        "mate.jit.gemm.dense_w4a16_gemm.DENSE_W4A16_SPLITK_OVERRIDE", 3
    )
    assert dense_w4a16_splitk_for(8, 3840, narrow) == 3


def test_splitk_override_is_a_working_env_seam(monkeypatch) -> None:
    # An env var that silently does nothing is worse than no knob: the A/B has to be one export.
    import mate.jit.gemm.dense_w4a16_gemm as G

    monkeypatch.setenv("DENSE_W4A16_SPLITK_OVERRIDE", "3")
    assert G._splitk_override_from_env() == 3
    monkeypatch.setenv("DENSE_W4A16_SPLITK_OVERRIDE", "  ")
    assert G._splitk_override_from_env() is None
    monkeypatch.setenv("DENSE_W4A16_SPLITK_OVERRIDE", "two")
    with pytest.raises(ValueError, match="must be an integer split count"):
        G._splitk_override_from_env()


def test_splitk_policy_names_its_arguments() -> None:
    # The A/B recipe is easy to get wrong: 1536 lands in `tile` if the call is (m, k, n).
    with pytest.raises(TypeError, match="tile must be"):
        dense_w4a16_splitk_for(8, 8960, 1536)
    # (size_m, size_n, tile, k_blocks): 24 tiles over 60 MPs is two units each, and 35 K blocks is
    # enough that halving them keeps more than the floor of streaming blocks per CTA.
    assert dense_w4a16_splitk_for(8, 1536, DENSE_W4A16_GEMM_TILE_NARROW, 35) == 5
    # The same shape read with N and K swapped is a full wave and must not split at all.
    assert dense_w4a16_splitk_for(8, 8960, DENSE_W4A16_GEMM_TILE_NARROW, 35) == 1
    # The machine is keyword-only: a fifth positional argument is a mis-read of the signature, and
    # silently treating it as `num_mps` would size the split against the wrong machine.
    with pytest.raises(TypeError):
        dense_w4a16_splitk_for(8, 1536, DENSE_W4A16_GEMM_TILE_NARROW, 35, 48)


def test_rendered_source_instantiates_both_k_split_modes() -> None:
    # The whole-K path must be its own instantiation, not a runtime branch of the split one: that is
    # what keeps an unsplit launch at the cost it had before split-K existed.
    source = render_dense_w4a16_gemm_source(torch.float16, False)
    assert "using KernelSplit" in source
    assert "kZeroPoints, false>" in source.replace("\n", " ").replace("  ", " ")
    assert "kZeroPoints, true>" in source.replace("\n", " ").replace("  ", " ")
    # Exactly one host-side launch site, reached through one dispatch on the split count: a guard over
    # the partial-argument checks, then the launch itself.
    assert source.count("if (ks > 1)") == 2
    assert source.count("launch_kernel(KernelTag<") == 2
    # And no device-side split branch to pay for on the way through the unsplit kernel.
    header = (
        pathlib.Path(__file__).resolve().parents[1]
        / "include/mate/gemm/deep_gemm/mp31_dense_w4a16_gemm.hpp"
    ).read_text()
    assert "if (params.k_split" not in header
    assert "if (k_split" not in header


def test_dispatch_name_carries_the_launcher_abi() -> None:
    # The cache key has to move with the exported signature, or a module built for the old arguments
    # is loaded and called with the new ones.
    from mate.jit.gemm.dense_w4a16_gemm import DENSE_W4A16_GEMM_DISPATCH_ABI

    assert DENSE_W4A16_GEMM_DISPATCH_ABI == "sk2"
    assert dense_w4a16_gemm_dispatch_name(torch.float16, False, DENSE_W4A16_GEMM_TILE_NARROW)[
        -4:
    ] == "_sk2"


def test_symmetric_zero_points_produce_no_correction() -> None:
    # The packed nibble already carries the bias, so a zero point equal to the nibble bias is the
    # symmetric GPTQ (uint4b8) case and its rank-1 correction has to vanish exactly.
    scale = torch.rand(4, 3, dtype=torch.float16) + 0.5
    symmetric = torch.full_like(scale, float(DENSE_W4A16_GEMM_NIBBLE_BIAS))
    assert torch.equal(fold_zero_points(symmetric, scale), torch.zeros_like(scale))


def test_asymmetric_zero_points_fold_to_scaled_offset() -> None:
    scale = torch.tensor([[2.0, 0.5]], dtype=torch.float16)
    zero_points = torch.tensor([[6, 10]], dtype=torch.int32)
    folded = fold_zero_points(zero_points, scale)
    assert torch.equal(folded, torch.tensor([[4.0, -1.0]], dtype=torch.float16))


def test_padded_group_column_contributes_nothing() -> None:
    # The odd group column needed by the paired 16-bit scale words is filled with the nibble bias, so
    # the extra group folds to an exact zero instead of a spurious offset.
    padded_zero = torch.full(
        (2, 4), float(DENSE_W4A16_GEMM_NIBBLE_BIAS), dtype=torch.int32
    )
    padded_zero[:, :3] = 7
    folded = fold_zero_points(padded_zero, torch.ones(2, 4, dtype=torch.float16))
    assert folded[0, 3].item() == 0.0
    assert folded[0, 0].item() == 1.0


def test_naive_dequant_is_wrong_by_the_bias() -> None:
    # A reference that drops the rank-1 term is off by (bias - zero_point) * sum(scale) per group,
    # which is exactly the correction the kernel applies for a zero point of zero.
    scale = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    zero_points = torch.zeros((1, 2), dtype=torch.int32)
    correction = fold_zero_points(zero_points, scale)
    assert torch.equal(correction, float(DENSE_W4A16_GEMM_NIBBLE_BIAS) * scale)


def test_host_operand_is_rejected() -> None:
    # A host tensor would be dereferenced as device memory, so mixing placements must be refused
    # before the kernel is launched.
    a = torch.zeros((1, 256), dtype=torch.float16, device="meta")
    w = torch.zeros((64, 128), dtype=torch.int8, device="cpu")
    s = torch.zeros((64, 2), dtype=torch.float16, device="meta")
    out = torch.zeros((1, 64), dtype=torch.float16, device="meta")
    with pytest.raises(ValueError, match="device pointer"):
        dense_w4a16_gemm_mutlass(a, w, s, out)


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("zero_points", [False, True])
def test_rendered_source_selects_the_kernel(
    out_dtype: torch.dtype, zero_points: bool
) -> None:
    func_name = dense_w4a16_gemm_dispatch_name(out_dtype, zero_points)
    source = render_dense_w4a16_gemm_source(out_dtype, zero_points)
    assert f"void {func_name}(" in source
    assert "mate::gemm::dense_w4a16_gemm::Mp31DenseW4A16Gemm<" in source
    assert "using ElementB = mute::int4_t;" in source
    assert (
        f"constexpr bool kZeroPoints = {'true' if zero_points else 'false'};" in source
    )
    # Every variant must reach the dense scheduler, never a grouped one.
    assert "mate::deep_gemm::GemmType::Normal" in source
    assert "get_grid_shape_persistent(params)" in source


def test_act_order_is_rejected_until_the_layout_is_defined() -> None:
    packed = torch.zeros(2, 4, dtype=torch.int8)
    ident = torch.arange(8, dtype=torch.int32)
    assert apply_gidx_permutation(packed, None) is packed
    assert apply_gidx_permutation(packed, ident) is packed
    with pytest.raises(NotImplementedError):
        apply_gidx_permutation(packed, ident.flip(0))
    with pytest.raises(ValueError):
        apply_gidx_permutation(packed, torch.zeros(2, 4, dtype=torch.int32))


def test_input_validation_rejects_a_broken_contract() -> None:
    m, n, k = 4, 8, 256
    a = torch.zeros(m, k, dtype=torch.float16)
    b = torch.zeros(n, k // 2, dtype=torch.int8)
    scale = torch.ones(n, k // 128, dtype=torch.float16)
    out = torch.zeros(m, n, dtype=torch.float16)

    # A K that the K256 tile cannot cover would make the group count ambiguous.
    with pytest.raises(ValueError):
        dense_w4a16_gemm_mutlass(
            torch.zeros(m, k // 2, dtype=torch.float16), b, scale, out
        )
    with pytest.raises(ValueError):
        dense_w4a16_gemm_mutlass(a, b, scale[:, :1], out)
    with pytest.raises(ValueError):
        dense_w4a16_gemm_mutlass(
            a, b, scale, torch.zeros(m, n + 1, dtype=torch.float16)
        )
    with pytest.raises(ValueError):
        dense_w4a16_gemm_mutlass(a.to(torch.bfloat16), b, scale, out)
    with pytest.raises(ValueError):
        dense_w4a16_gemm_mutlass(a, b.to(torch.float16), scale, out)
    with pytest.raises(ValueError):
        dense_w4a16_gemm_mutlass(a, b, scale, torch.zeros(m, n, dtype=torch.float32))


def test_launcher_passes_the_resolved_mp_count_to_the_split_policy() -> None:
    # The policy sizes a split against a machine; the launcher owns the resolved count. A call site that
    # dropped the keyword would silently size every split against the module default again, which is the
    # bug this guards, so assert the plumbing and not just the arithmetic.
    import ast

    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "mate/jit/gemm/dense_w4a16_gemm.py"
    ).read_text()
    launcher = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "dense_w4a16_gemm_mutlass"
    )
    calls = [
        node
        for node in ast.walk(launcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dense_w4a16_gemm_plan"
    ]
    assert len(calls) == 1, "the launcher must size the split in exactly one place"
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(keywords.get("num_mps"), ast.Name)
    assert keywords["num_mps"].id == "launch_mps"
    # ...and `launch_mps` has to be the resolved count, not a literal that happens to look like one.
    assignments = [
        node
        for node in ast.walk(launcher)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "launch_mps"
    ]
    assert assignments, "launch_mps is never assigned in the launcher"
    assert all(
        isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "resolve_num_mps"
        for node in assignments
    ), "launch_mps must come from resolve_num_mps, not from a constant"


def test_splitk_wave_model_covers_the_long_k_shapes() -> None:
    # Every expectation below is the argmin of a brute-force split sweep on S5000 (60 MPs, cold cache,
    # grid 60). The model only runs at kb >= 16, so it can only ever raise what the fill rule chose.
    narrow, wide = DENSE_W4A16_GEMM_TILE_NARROW, DENSE_W4A16_GEMM_WIDE_N_TILE
    # down: K=8960 is 35 K blocks, 24 tiles at M=8/16 and 48 at M=32. Split 5 is two full waves of 120
    # work items at 7 blocks each; the fill rule's split 2 fills 48 of 60 MPs at 18 blocks. Measured
    # 27.72 -> 24.03 us (M=8), 27.20 -> 23.87 (M=16), 43.06 -> 39.36 (M=32).
    assert dense_w4a16_splitk_for(8, 1536, narrow, 35) == 5
    assert dense_w4a16_splitk_for(16, 1536, narrow, 35) == 5
    assert dense_w4a16_splitk_for(32, 1536, narrow, 35) == 5
    # Long K that the old rule already got right must not move: K=4096 is 16 blocks at 24 tiles.
    assert dense_w4a16_splitk_for(8, 1536, narrow, 16) == 2
    # Below the boundary the fill rule still owns the answer, and every measured shape keeps its split.
    assert dense_w4a16_splitk_for(8, 1536, narrow, 11) == 2
    assert dense_w4a16_splitk_for(8, 1536, narrow, 6) == 2  # o, 24 tiles
    assert dense_w4a16_splitk_for(32, 1536, narrow, 6) == 1  # o at M=32, 48 tiles
    assert dense_w4a16_splitk_for(8, 2048, narrow, 6) == 1  # qkv, 32 tiles
    assert dense_w4a16_splitk_for(8, 17920, wide, 6) == 1  # gate_up, 140 tiles
    # The prefill tile is bandwidth-bound and split-invariant -- `down` measures 71.8 / 71.9 / 72.3 us
    # at splits 2 / 5 / 1 for M=128 / 256 / 512 -- so the model stays out of it and the fill rule keeps
    # its answer instead of paying for partials that save nothing.
    assert dense_w4a16_splitk_for(128, 1536, DENSE_W4A16_GEMM_TILE, 35) == 2
    assert dense_w4a16_splitk_for(256, 1536, DENSE_W4A16_GEMM_TILE, 35) == 1


def test_gemm_plan_forwards_the_machine_it_was_given() -> None:
    # The plan adds the tile to the search; it must not add a second split heuristic. For whichever tile
    # it picks, the split has to be exactly what `dense_w4a16_splitk_for` answers for that tile and that
    # machine -- otherwise the two rules can drift apart and only the benchmark would notice.
    for num_mps in (24, 48, 60):
        tile, split = dense_w4a16_gemm_plan(8, 1536, 1536, num_mps=num_mps)
        k_blocks = -(-1536 // tile[2])
        assert split == dense_w4a16_splitk_for(
            8, 1536, tile, k_blocks, num_mps=num_mps
        )
    # ...and the machine does reach it: the 24-tile shape is already a whole wave on a 24-MP part, where
    # the narrow tile must not split, while on 60 MPs the same tile takes 2.
    narrow = DENSE_W4A16_GEMM_TILE_NARROW
    assert dense_w4a16_splitk_for(8, 1536, narrow, 6, num_mps=24) == 1
    assert dense_w4a16_splitk_for(8, 1536, narrow, 6, num_mps=60) == 2
