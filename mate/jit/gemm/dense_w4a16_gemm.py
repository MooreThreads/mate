"""Dense W4A16 GEMM: fp16 activations, signed INT4 weights, per-K128-group fp16 scales.

The kernel keeps every M-dependent decision inside one compiled op, so the dispatch name never
depends on M and the caller cannot freeze a shape-specific choice at trace time. One tile shape
covers the whole range; the persistent scheduler adapts the work distribution instead.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Optional

import torch
from jinja2 import Environment, FileSystemLoader

from ...mate_runtime import resolve_num_mps
from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec
from ..utils import EXPORT_FUNC, TVM_HEADER, dtype_torch2mutlass_map
from . import gemm_utils

# The K256 tile carries exactly two K128 weight-scale blocks, which is the group size W4A16
# checkpoints are quantized with; a different group size needs a different tile, not a flag.
#
# Two stages is the most this tile can stage. Each stage costs about 68 KB (the fp16 A tile, its copy
# in the MMA atom layout, the packed INT4 B tile and the paired group scales), and the MP31 per-block
# limit is 192 KB, so a third stage cannot fit for either the symmetric or the zero-point build.
DENSE_W4A16_GEMM_TILE = (32, 256, 256, 2)

# An M-narrow instantiation for small M. At M=1 the wide tile does useful work in 1 of its 32 rows;
# the narrow tile is faster wherever M <= 32 and N <= 4096 and slower past that, so it is selected
# only inside that rectangle. Rows the GEMV serves never reach either tile.
DENSE_W4A16_GEMM_TILE_NARROW = (16, 64, 256, 3)
DENSE_W4A16_GEMM_NARROW_MAX_M = 32
DENSE_W4A16_GEMM_NARROW_MAX_N = 4096

# A wide-N tile for a small-M launch -- the decode gate_up shape, M=8 K=1536 N=17920. It is wider than
# the narrow tile because N is large enough that the extra columns amortise the shorter K stage, and it
# runs on the SQMMA GEMM family rather than the prefill kernel: the prefill tile has no tile_m/tile_k
# freedom (the spec generator hard-guards 128x?x128), so this shape is only reachable by also moving the
# family boundary in dense_w4a16_use_prefill. Measured on S5000 at M=8 K=1536 N=17920: 43.98 us on the
# prefill path against 31.17 us here, and 43.98 -> 31.17 across 28 calls a decode step is -0.39 ms/step.
#
# The benefit is bounded by M: at M=32 this tile gives 280 CTAs over 6 waves and measures 21.0% WORSE
# than the prefill path, so the rectangle stops at 16 and M=17..32 keeps prefill.
DENSE_W4A16_GEMM_WIDE_N_TILE = (16, 128, 256, 3)
DENSE_W4A16_GEMM_WIDE_N_MAX_M = 16


def dense_w4a16_gemm_tile_for(size_m: int, size_n: int) -> tuple:
    """Pick the tile for one launch. Pure, so the choice is testable without a device.

    An INTERNAL runtime choice on the activation row count and the output width. It is not a kernel
    argument and must not become one: the ported suite forbids M-shaped parameters precisely so a
    caller cannot freeze a shape-specific decision at trace time, and keeping the choice internal is
    also what makes the op graph-capture safe.

    This is the answer when K is unknown -- the rectangle the tile was validated in. The launch path
    does not use it: `dense_w4a16_gemm_plan` prices both decode tiles against the actual K, because
    the tile count a tile produces decides the split, and the split decides the blocks per CTA.
    """
    if size_m <= DENSE_W4A16_GEMM_NARROW_MAX_M and size_n <= DENSE_W4A16_GEMM_NARROW_MAX_N:
        return DENSE_W4A16_GEMM_TILE_NARROW
    if size_m <= DENSE_W4A16_GEMM_WIDE_N_MAX_M and size_n > DENSE_W4A16_GEMM_NARROW_MAX_N:
        return DENSE_W4A16_GEMM_WIDE_N_TILE
    return DENSE_W4A16_GEMM_TILE
DENSE_W4A16_GEMM_GROUP_SIZE = 128

# A launch on the SQMMA family costs two things that a tile count alone does not show: a fixed
# prologue/epilogue per CTA and a serial per-K-block stream inside it. Measured on S5000 (60 MPs,
# cold cache) by holding the tile count and sweeping K, so the slope and the intercept separate:
# [M=8, N=1536, tile 16x64] measures 9.95 us at 2 K blocks and 33.41 us at 24, i.e.
#     time = 7.9 us + 1.07 us/block
# and the same fit on the wide tile gives 8.7 + 1.15, so the per-block price barely moves when the B
# tile doubles: the stream is latency-bound per CTA, not bandwidth-bound (an 8 KB and a 16 KB block
# cost the same). Two consequences drive the plan below.
#
# First, the intercept is per CTA, not per work item: at 24 tiles, K=1536, 8 blocks per CTA measures
# 18.15 us as one work item and 18.32 us as two items of 4 blocks, so there is nothing to hoist out
# of the item loop. What the intercept punishes is a launch whose CTAs have too few K blocks to
# amortise it -- at K=1536 the fixed part is ~55% of the run.
#
# Second, whose CTAs those are is decided by the tile, because the tile decides the tile count and
# the split then decides how many work items that count becomes. That is why the tile cannot be
# chosen from M and N alone: `down` at K=8960 is 24 tiles on the narrow tile, so the split that
# fills 60 MPs has to make 120 items and each CTA still streams 14 blocks, while the wide tile makes
# 12 tiles and the same 60-item grid leaves 7 blocks each: 24.5 -> 17.3 us at M=8, 24.5 -> 17.2 at
# M=16, 40.5 -> 26.0 at M=32. The K axis is what turns the wide tile from a N-driven special case
# into the better tile for every decode projection.
DENSE_W4A16_PLAN_OVERHEAD_US = 8.0
DENSE_W4A16_PLAN_BLOCK_US_NARROW = 1.05
DENSE_W4A16_PLAN_BLOCK_US_WIDE = 1.15


def dense_w4a16_plan_cost(
    size_m: int,
    size_n: int,
    tile: tuple,
    k_split: int,
    k_blocks: int,
    num_mps: int,
) -> float:
    """Price one (tile, split) pair in microseconds. Pure, so the choice is testable without a device.

    Every CTA that ends up holding work pays the prologue once and then streams its own K blocks
    serially, so a work item that lands on top of another one costs a whole extra block chain rather
    than eliding it: blocks per CTA is waves times blocks per item, not the total divided by the grid.
    """
    tile_m, tile_n, tile_k, _ = tile
    tiles = -(-size_m // tile_m) * -(-size_n // tile_n)
    items = tiles * k_split
    waves = -(-items // num_mps)
    blocks_per_cta = waves * -(-k_blocks // k_split)
    block_us = (
        DENSE_W4A16_PLAN_BLOCK_US_NARROW
        if tile_n <= DENSE_W4A16_GEMM_TILE_NARROW[1]
        else DENSE_W4A16_PLAN_BLOCK_US_WIDE
    )
    return (
        DENSE_W4A16_PLAN_OVERHEAD_US
        + blocks_per_cta * block_us
        + DENSE_W4A16_SPLITK_REDUCTION_BLOCKS * (k_split - 1)
    )


def dense_w4a16_gemm_plan(
    size_m: int, size_n: int, size_k: int, num_mps: Optional[int] = None
) -> tuple:
    """Pick (tile, k_split) for one launch by pricing every candidate. Pure and device-free.

    The candidates are the two decode instantiations. The prefill family is chosen earlier, before
    this runs, by `dense_w4a16_use_prefill`, and the GEMV keeps its own rectangle, so neither is
    priced here. The split for each tile comes from `dense_w4a16_splitk_for`, which is the rule the
    shipped path already used for a fixed tile -- the plan adds the tile to the search, not a second
    split heuristic.
    """
    if num_mps is None:
        # The S5000 figure, for a caller with no device in hand. The launch path passes the machine it
        # actually resolved, because a split sized to fill 60 MPs is wrong on a smaller part.
        num_mps = DENSE_W4A16_SPLITK_WAVE
    best = None
    for tile in (DENSE_W4A16_GEMM_TILE_NARROW, DENSE_W4A16_GEMM_WIDE_N_TILE):
        k_blocks = -(-size_k // tile[2])
        k_split = min(
            dense_w4a16_splitk_for(size_m, size_n, tile, k_blocks, num_mps=num_mps), k_blocks
        )
        cost = dense_w4a16_plan_cost(size_m, size_n, tile, k_split, k_blocks, num_mps)
        if best is None or cost < best[0]:
            best = (cost, tile, k_split)
    return best[1], best[2]

# Split-K. One CTA streams a whole column block, so its wall time is set by the bytes it reads and not
# by how many CTAs are resident: measured on S5000 (60 MPs) at [M=8, K=8960], time was flat at ~62 us
# from N=512 (8 tiles) to N=3840 (60 tiles) and only one wave of 60 tiles fills the machine. Below that
# the idle MPs are the whole problem, so splitting K spreads one tile's column block over several CTAs
# and divides its stream time by the split count.
#
# The rule is pure arithmetic on the tile count: a launch that already fills a wave must not be touched
# (a split would only add a reduction), and anything narrower takes the largest split that fits, capped
# so the extra partials stay small.
#
# "A wave" is the machine the launch actually gets, which is why the policy takes the resolved MP count
# as an argument: the launcher resolves it with resolve_num_mps(input_a.device, num_mps) -- an explicit
# caller override, a set_num_mps() limit, or the device's own count -- and the prefill tile chooser in
# this file already reads it the same way. The default below is only the S5000 figure for a caller that
# has no device in hand.
DENSE_W4A16_SPLITK_WAVE = 60
DENSE_W4A16_SPLITK_MAX = 8
# This guard is about occupancy, not about how many K blocks a split saves. Measured on S5000 (60 MPs,
# cold cache, tile 16x64x256, two independent timers agreeing): [M=8, K=1536, N=1536] has 24 tiles, so
# 36 of the 60 MPs hold no CTA, and a 2-way split takes 14.4 -> 12.2 us; at [M=16] the same shape
# measures 13.6 -> 12.0 us. Once the tile count covers most of the machine the split is a loss: at
# [M=32] that shape has 48 tiles and measures 13.2 -> 15.7 us, and [M=32, K=8960] has 48 tiles and
# measures 44.4 -> 46.9 us. And where the split leaves every CTA with a single K block the reduction
# dominates outright: [M=16, K=512, N=128] has 2 tiles and 2 K blocks and measures 10.5 -> 12.3 us.
#
# The previous rule counted saved blocks instead (a split had to remove >= 6 from every CTA). At
# K=1536 there are only 6 blocks, so it vetoed the very split that wins by 15% at M=8. The trade it
# was approximating is "streaming blocks per CTA versus the ~4.5 us reduction"; the floor below states
# the break-even for that trade empirically, at two blocks, rather than deriving it.
DENSE_W4A16_SPLITK_MIN_BLOCKS_PER_CTA = 2
# Long K is the one regime where the fill rule is provably leaving time on the table, because there the
# split can be aligned to whole waves instead of to a single one. Measured on S5000 (60 MPs, cold cache,
# grid 60): `down` at K=8960 has 35 K blocks and 24 tiles, so the fill rule picks 2 -- 48 CTAs, 12 MPs
# idle, 18 blocks each -- while 5 loads the machine with 120 work items, exactly two waves, at 7 blocks
# each: 28.11 -> 24.26 us at M=8, 27.43 -> 24.00 at M=16, and at M=32 (48 tiles) 43.31 -> 40.05 us.
# The curve is not monotonic in the split: 3 and 4 land on a fractional wave and measure 34.2 and 28.4 us
# at M=8, which is why this is a cost model and not a bigger constant.
#
# The model prices every split by `waves * blocks_per_cta` plus a per-extra-split reduction term, and is
# only consulted above this K. Below it the block count is too small for that trade: at K=1536 (6 blocks)
# the fill rule already matches the measured optimum on `qkv`, `o` and `gate_up` at every M.
DENSE_W4A16_SPLITK_LONG_K_BLOCKS = 16
# Model constants. A CTA that gets fewer than three K blocks never fills the 3-stage pipeline, and each
# extra split adds a partial to the final sum, priced here at a third of a K block. Fitted on the table
# above; every pick below is unchanged for any value in [0.2, 0.6].
DENSE_W4A16_SPLITK_PIPELINE_BLOCKS = 3
DENSE_W4A16_SPLITK_REDUCTION_BLOCKS = 0.3
# The model prices K blocks stepped per CTA, which is the binding cost only on the narrow tile: there the
# part is latency-bound and a CTA that stalls on its block stream is time nobody reclaims. On the prefill
# tile the same shape is bandwidth-bound and the split does not matter -- `down` at M=128/256/512 measures
# 71.8 / 71.9 / 72.3 us at splits 2, 5 and 1 alike (within +-0.3 %), so consulting the model there would
# only add partials traffic. Gate on the tile the policy was validated on.
DENSE_W4A16_SPLITK_WAVE_TILE_M = 16
# Test and benchmark seam: when set, every SQMMA launch runs with this split count. Never set by the
# serving path -- a traced/captured graph must read the same policy value on every replay. The env var
# is read once at import so an A/B is one `DENSE_W4A16_SPLITK_OVERRIDE=1` away; a malformed value is
# loud here rather than silently ignored.
def _splitk_override_from_env() -> Optional[int]:
    raw = os.environ.get("DENSE_W4A16_SPLITK_OVERRIDE")
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"DENSE_W4A16_SPLITK_OVERRIDE must be an integer split count, got {raw!r}"
        ) from exc


DENSE_W4A16_SPLITK_OVERRIDE: Optional[int] = _splitk_override_from_env()


@functools.lru_cache(maxsize=256)
def _dense_w4a16_splitk_wave_cost(tiles: int, k_blocks: int, num_mps: int) -> int:
    """The split that minimises modelled time when K is long enough to be worth re-aligning.

    Time is priced in K blocks stepped: every wave of CTAs steps its own blocks, so the streaming term is
    `waves * blocks_per_cta`, and each extra split adds a partial to the final sum. Ties keep the smaller
    split, so this never splits for free.

    Cached because it is pure over three integers and sits on a path called once per GEMM: the search
    measured 4.4 us per call against the 0.6 us the previous rule needed, which is invisible under graph
    capture but not for an eager caller.
    """
    best, best_cost = 1, None
    for split in range(1, min(DENSE_W4A16_SPLITK_MAX, k_blocks) + 1):
        blocks = -(-k_blocks // split)
        if blocks < DENSE_W4A16_SPLITK_PIPELINE_BLOCKS:
            continue
        waves = -(-tiles * split // num_mps)
        cost = waves * blocks + DENSE_W4A16_SPLITK_REDUCTION_BLOCKS * (split - 1)
        if best_cost is None or cost < best_cost:
            best, best_cost = split, cost
    return best


def dense_w4a16_splitk_for(
    size_m: int,
    size_n: int,
    tile=None,
    k_blocks: Optional[int] = None,
    *,
    num_mps: int = DENSE_W4A16_SPLITK_WAVE,
) -> int:
    """The K-split count for one launch. Pure, so the policy is testable without a device.

    Arguments, in order:

    ``size_m``    problem M (rows of the activation).
    ``size_n``    problem N (columns of the output).
    ``tile``      optional ``(tile_m, tile_n, tile_k, stages)``; defaults to DENSE_W4A16_GEMM_TILE.
                  A plain number here is a mistake, not a tile.
    ``k_blocks``  optional K axis in tile units. When it is given, a split that would leave every CTA
                  with a single K block is rejected: at that point the ~4.5 us reduction costs more
                  than the one block of streaming it removes.
    ``num_mps``   keyword-only: the machine this launch actually gets, as the launcher resolves it.
                  Defaults to the S5000 figure for callers with no device in hand.

    Returns 1 for a whole-K launch or the split count for a split launch.
    """
    if tile is not None and not isinstance(tile, (tuple, list)):
        raise TypeError(
            f"tile must be (tile_m, tile_n, tile_k, stages) or None, got {tile!r}; "
            f"call as dense_w4a16_splitk_for(size_m, size_n, tile, k_blocks)"
        )
    if DENSE_W4A16_SPLITK_OVERRIDE is not None:
        return max(1, int(DENSE_W4A16_SPLITK_OVERRIDE))
    if num_mps <= 0:
        raise ValueError(f"num_mps must be a positive MP count, got {num_mps!r}")
    tile_m, tile_n, _, _ = DENSE_W4A16_GEMM_TILE if tile is None else tile
    tiles = -(-size_m // tile_m) * -(-size_n // tile_n)
    if tiles <= 0 or tiles >= num_mps:
        return 1
    k_split = max(1, min(DENSE_W4A16_SPLITK_MAX, num_mps // tiles))
    if k_split > 1 and k_blocks is not None:
        if k_blocks // k_split < DENSE_W4A16_SPLITK_MIN_BLOCKS_PER_CTA:
            k_split = 1
    if (
        k_blocks is not None
        and k_blocks >= DENSE_W4A16_SPLITK_LONG_K_BLOCKS
        and tile[0] <= DENSE_W4A16_SPLITK_WAVE_TILE_M
    ):
        # Long K on the narrow tile: a wave-aligned split can beat the one-wave fill rule. `max` keeps
        # this strictly additive -- the model may raise the split, never lower the one the rules above
        # already chose.
        k_split = max(k_split, _dense_w4a16_splitk_wave_cost(tiles, k_blocks, num_mps))
    return k_split

# The MMA atom reads each 4-bit weight field as a signed two's-complement integer; measured on the
# K256 tile, nibble 0x0 yields 0, 0x3 yields +3 and 0x8 yields -8. A checkpoint whose unsigned nibble
# `u` decodes as `u - nibble_bias` must therefore be packed as `u ^ 0x8` (the packed byte XOR 0x88),
# which is the two's-complement encoding of `u - nibble_bias`. With that packing the atom's product
# already carries the bias, so a zero point equal to the bias needs no correction at all and every
# other zero point needs only the rank-1 term the kernel applies.
DENSE_W4A16_GEMM_NIBBLE_BIAS = 8

DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

DENSE_W4A16_GEMM_MCC_FLAGS = ["-Od3", "-O3", "-std=c++17"]
DENSE_W4A16_GEMM_CXX_FLAGS = ["-O3"]


@dataclass(frozen=True)
class MateW4A16WeightLayout:
    """The packed-weight contract this kernel consumes today.

    Contract: `b_packed[n, k // 2]` is one byte holding the pair of K-adjacent weights of output
    column `n`: the even K index occupies the low nibble, the odd K index the high nibble, and each
    nibble holds `unsigned_nibble ^ 0x8`, the two's-complement encoding of `unsigned_nibble -
    nibble_bias`. The atom multiplies those fields signed, so the bias is already inside the product
    and no per-element correction is needed. `scale_b` and the folded `zero_b` are `(N, K // 128)`.

    TODO: consuming an upstream checkpoint needs both halves of the same translation. Upstream's
    Marlin layout (`[K // 16, 2 * N]` int32 words) holds the same logical `(k, n) -> (word, nibble)`
    index map in a different physical order, and its nibbles are the raw unsigned values, so the
    packing has to permute the nibble addresses into this atom's fragment addresses and XOR each byte
    with 0x88 on the way. Nothing else in this module depends on the physical order or on the raw
    encoding, so that translation is the only change.
    """

    even_k_low_nibble: bool = True
    nibble_bias: int = DENSE_W4A16_GEMM_NIBBLE_BIAS
    scale_group_size: int = DENSE_W4A16_GEMM_GROUP_SIZE


MATE_W4A16_WEIGHT_LAYOUT = MateW4A16WeightLayout()

# The dispatch name is the JIT cache key, so it also has to carry the launcher ABI: the split-K
# arguments changed the exported function's signature, and a module compiled against the old one would
# be loaded and called with the new argument list. Bump this tag whenever that signature changes, or
# when the kernels behind an unchanged signature change enough that a cached .so must not be reused
# ("sk2" is the two-instantiation form, where the whole-K path is no longer a runtime-branched copy).
DENSE_W4A16_GEMM_DISPATCH_ABI = "sk2"


def dense_w4a16_gemm_dispatch_name(out_dtype: torch.dtype, zero_points: bool, tile=None) -> str:
    if out_dtype not in DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS:
        raise ValueError(f"out_dtype must be fp16 or bf16, got {out_dtype!r}")
    tile_m, tile_n, tile_k, stages = DENSE_W4A16_GEMM_TILE if tile is None else tile
    zp_tag = "zp" if zero_points else "sym"
    return (
        f"mp31_dense_w4a16_gemm_{DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS[out_dtype]}"
        f"_{tile_m}x{tile_n}x{tile_k}_s{stages}_{zp_tag}_{DENSE_W4A16_GEMM_DISPATCH_ABI}"
    )


def render_dense_w4a16_gemm_source(out_dtype: torch.dtype, zero_points: bool, tile=None) -> str:
    tile_m, tile_n, tile_k, stages = DENSE_W4A16_GEMM_TILE if tile is None else tile
    render_config = {
        "func_name": dense_w4a16_gemm_dispatch_name(out_dtype, zero_points, tile),
        "element_a": dtype_torch2mutlass_map[torch.float16],
        "element_scale_b": dtype_torch2mutlass_map[torch.float16],
        "element_zero_b": dtype_torch2mutlass_map[torch.float16],
        "element_d": dtype_torch2mutlass_map[out_dtype],
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "stages": stages,
        "zero_points": zero_points,
    }
    return (
        TVM_HEADER
        + Environment(
            loader=FileSystemLoader(
                (gemm_utils.gemm_template_dir / "dense_w4a16_gemm").as_posix()
            ),
            keep_trailing_newline=True,
        )
        .get_template("dense_w4a16_gemm_kern.j2")
        .render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


def gen_dense_w4a16_gemm_spec(out_dtype: torch.dtype, zero_points: bool, tile=None) -> JitSpec:
    dispatch_name = dense_w4a16_gemm_dispatch_name(out_dtype, zero_points, tile)
    source_path = jit_env.MATE_GEN_SRC_DIR / "dense_w4a16_gemm" / f"{dispatch_name}.mu"
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cflags=DENSE_W4A16_GEMM_CXX_FLAGS,
        extra_cuda_cflags=DENSE_W4A16_GEMM_MCC_FLAGS,
        extra_include_paths=[
            jit_env.MATE_INCLUDE_DIR,
            jit_env.MATE_CSRC_DIR,
            jit_env.MUTLASS_INCLUDE_DIR,
        ],
        generated_sources={
            source_path: render_dense_w4a16_gemm_source(out_dtype, zero_points, tile)
        },
    )


def gen_dense_w4a16_gemm_aot() -> list[JitSpec]:
    specs = [
        gen_dense_w4a16_gemm_spec(out_dtype, zero_points, tile)
        for out_dtype in DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS
        for zero_points in (False, True)
        for tile in (None, DENSE_W4A16_GEMM_TILE_NARROW)
    ]
    specs += [
        gen_dense_w4a16_prefill_spec(out_dtype, tile, zero_points)
        for out_dtype in DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS
        for zero_points in (False, True)
        for tile in DENSE_W4A16_PREFILL_TILES
    ]
    return specs


@dataclass(frozen=True)
class DenseW4A16GemvConfig:
    """One small-M instantiation: up to ``m_max`` activation rows. ``lanes_per_row`` lanes share
    ``rows_per_group`` output columns, ``k_split`` warps split K, each lane covers up to
    ``chunks_per_lane`` 32-wide K chunks, and a block has ``warps`` warps."""

    m_max: int
    lanes_per_row: int = 16
    rows_per_group: int = 1
    k_split: int = 1
    chunks_per_lane: int = 3
    warps: int = 8
    a_smem: bool = True  # stage the activation in shared memory; False reads it from L2 per chunk


def dense_w4a16_gemv_dispatch_name(
    out_dtype: torch.dtype, zero_points: bool, config: DenseW4A16GemvConfig
) -> str:
    if out_dtype not in DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS:
        raise ValueError(f"out_dtype must be fp16 or bf16, got {out_dtype!r}")
    zp_tag = "zp" if zero_points else "sym"
    return (
        f"mp31_dense_w4a16_gemv_{DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS[out_dtype]}_m{config.m_max}"
        f"_l{config.lanes_per_row}_r{config.rows_per_group}_s{config.k_split}_c{config.chunks_per_lane}"
        f"_w{config.warps}{'' if config.a_smem else '_ag'}_{zp_tag}"
    )


def gen_dense_w4a16_gemv_spec(
    out_dtype: torch.dtype, zero_points: bool, config: DenseW4A16GemvConfig
) -> JitSpec:
    dispatch_name = dense_w4a16_gemv_dispatch_name(out_dtype, zero_points, config)
    render_config = {
        "func_name": dispatch_name,
        "element_d": dtype_torch2mutlass_map[out_dtype],
        "m_max": config.m_max,
        "lanes_per_row": config.lanes_per_row,
        "rows_per_group": config.rows_per_group,
        "k_split": config.k_split,
        "chunks_per_lane": config.chunks_per_lane,
        "warps": config.warps,
        "a_smem": config.a_smem,
        "zero_points": zero_points,
    }
    source = (
        TVM_HEADER
        + Environment(
            loader=FileSystemLoader(
                (gemm_utils.gemm_template_dir / "dense_w4a16_gemm").as_posix()
            ),
            keep_trailing_newline=True,
        )
        .get_template("dense_w4a16_gemv_kern.j2")
        .render(render_config)
        + EXPORT_FUNC.render(render_config)
    )
    source_path = jit_env.MATE_GEN_SRC_DIR / "dense_w4a16_gemm" / f"{dispatch_name}.mu"
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cflags=DENSE_W4A16_GEMM_CXX_FLAGS,
        extra_cuda_cflags=DENSE_W4A16_GEMM_MCC_FLAGS,
        extra_include_paths=[
            jit_env.MATE_INCLUDE_DIR,
            jit_env.MATE_CSRC_DIR,
            jit_env.MUTLASS_INCLUDE_DIR,
        ],
        generated_sources={source_path: source},
    )


# Large M takes the prefill kernel: a 128-row tile whose K step is exactly one scale group, with one
# 128x64 atom per consumer squad and no restage of A. 0 disables it.
DENSE_W4A16_PREFILL_MIN_M = 128
DENSE_W4A16_PREFILL_TILE = (128, 256, 128, 3)
DENSE_W4A16_PREFILL_TILES = ((128, 256, 128, 3), (128, 128, 128, 4))
# Per-tile cost on S5000 in microseconds, by tile width: (per K128 step, epilogue).
_DENSE_W4A16_PREFILL_TILE_COST = {256: (1.25, 2.3), 128: (1.08, 1.15)}


def dense_w4a16_use_prefill(size_m: int, size_n: int) -> bool:
    """Whether the modelled prefill kernel should serve this launch.

    Pure, so the routing decision is testable without a device. Measured on S5000 (60 MPs) with real
    Qwen2.5-1.5B AWQ weights under captured replay, GB/s for MATE's own two kernels on the same shape:

        M         gate/up (K1536 N8960)      down (K8960 N1536)     q/o (K1536 N1536)
                  prefill   SQMMA tile       prefill  SQMMA tile    prefill  SQMMA tile
        8         144.1     113.1           55.5     83.0          26.1     44.2
        16        138.4     116.1           55.2     83.5          25.0     45.4
        32        134.7     119.9           54.6     83.8          24.3     44.8
        64        131.6      73.1           53.0     27.5          23.7     20.8

    Two regions follow. A wide-N launch above the GEMV ceiling (N > 4096) is better served by prefill from
    M=8 on, and M in 33..127 was previously no region at all: the M-narrow tile stops at 32 and the prefill
    entry started at 128, so that band ran the wide tile and lost up to 1.9x to this kernel.
    """
    if size_m >= DENSE_W4A16_PREFILL_MIN_M:
        return True
    if size_m > DENSE_W4A16_GEMM_NARROW_MAX_M:
        return True
    # This rectangle is served by the wide GEMM tile above, not by prefill.
    if size_m <= DENSE_W4A16_GEMM_WIDE_N_MAX_M and size_n > DENSE_W4A16_GEMM_NARROW_MAX_N:
        return False
    return size_m >= 8 and size_n > DENSE_W4A16_GEMM_NARROW_MAX_N

def dense_w4a16_prefill_tile_for(m: int, n: int, k: int, num_mps: int):
    """The prefill tile with the lowest modeled time: waves x (K steps x step cost + epilogue).

    The wide tile does more work per step; the narrow one wins whenever it fits in the same number
    of waves, which is where the wide tile leaves MPs idle."""
    best = None
    for tile in DENSE_W4A16_PREFILL_TILES:
        tile_m, tile_n, tile_k, _ = tile
        tiles = -(-m // tile_m) * -(-n // tile_n)
        waves = -(-tiles // max(num_mps, 1))
        step, epilogue = _DENSE_W4A16_PREFILL_TILE_COST[tile_n]
        cost = waves * (k // tile_k * step + epilogue)
        if best is None or cost < best[0]:
            best = (cost, tile)
    return best[1]


def dense_w4a16_prefill_dispatch_name(
    out_dtype: torch.dtype, tile=None, zero_points: bool = False
) -> str:
    if out_dtype not in DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS:
        raise ValueError(f"out_dtype must be fp16 or bf16, got {out_dtype!r}")
    tile_m, tile_n, tile_k, stages = DENSE_W4A16_PREFILL_TILE if tile is None else tile
    return (
        f"mp31_dense_w4a16_prefill_{DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS[out_dtype]}"
        f"_{tile_m}x{tile_n}x{tile_k}_s{stages}_{'zp' if zero_points else 'sym'}"
    )


def gen_dense_w4a16_prefill_spec(
    out_dtype: torch.dtype, tile=None, zero_points: bool = False
) -> JitSpec:
    tile_m, tile_n, tile_k, stages = DENSE_W4A16_PREFILL_TILE if tile is None else tile
    if (tile_m, tile_k) != (128, 128):
        raise ValueError(f"the prefill kernel is a 128x?x128 tile, got {(tile_m, tile_n, tile_k)}")
    dispatch_name = dense_w4a16_prefill_dispatch_name(out_dtype, tile, zero_points)
    render_config = {
        "func_name": dispatch_name,
        "element_d": dtype_torch2mutlass_map[out_dtype],
        "tile_n": tile_n,
        "stages": stages,
        "zero_points": "true" if zero_points else "false",
    }
    source = (
        TVM_HEADER
        + Environment(
            loader=FileSystemLoader(
                (gemm_utils.gemm_template_dir / "dense_w4a16_gemm").as_posix()
            ),
            keep_trailing_newline=True,
        )
        .get_template("dense_w4a16_prefill_kern.j2")
        .render(render_config)
        + EXPORT_FUNC.render(render_config)
    )
    source_path = jit_env.MATE_GEN_SRC_DIR / "dense_w4a16_gemm" / f"{dispatch_name}.mu"
    return gen_jit_spec(
        dispatch_name,
        [source_path],
        extra_cflags=DENSE_W4A16_GEMM_CXX_FLAGS,
        extra_cuda_cflags=DENSE_W4A16_GEMM_MCC_FLAGS,
        extra_include_paths=[
            jit_env.MATE_INCLUDE_DIR,
            jit_env.MATE_CSRC_DIR,
            jit_env.MUTLASS_INCLUDE_DIR,
        ],
        generated_sources={source_path: source},
    )


@functools.cache
def _load_dense_w4a16_prefill_module(out_dtype: torch.dtype, tile=None, zero_points: bool = False):
    return gen_dense_w4a16_prefill_spec(out_dtype, tile, zero_points).build_and_load()


def get_dense_w4a16_prefill_module(out_dtype: torch.dtype, tile=None, zero_points: bool = False):
    return (
        dense_w4a16_prefill_dispatch_name(out_dtype, tile, zero_points),
        _load_dense_w4a16_prefill_module(out_dtype, tile, zero_points),
    )


def prefill_row_sums(input_a: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """The zero-point operand that depends on the activation: ``(K // group_size, M')`` fp32 with
    ``M' >= M`` a multiple of 4 (16-byte TME rows), entry ``[g, m]`` the sum of ``a[m]`` over group
    ``g``. Rows past ``M`` are never read for a stored output.

    The kernel wants the group axis outer, so reducing over the group axis of an already transposed view
    produces that layout directly. The earlier form reduced to ``(M, G)`` and then copied the transpose into
    place, which cost a second launch for 1.5 KB of data -- on the prefill path this operand is rebuilt on
    every call, so that launch was ~5.6 us of a ~55 us op. ``M % 4 == 0`` keeps the output rows 16-byte
    aligned without a repack; when ``M`` is not a multiple of 4 the padded two-step form is still needed,
    because the padding has to be written somewhere and a strided scatter is not cheaper than the copy.
    """
    m, k = input_a.shape
    groups = k // group_size
    if m % 4 == 0:
        return input_a.reshape(m, groups, group_size).transpose(0, 1).sum(
            dim=-1, dtype=torch.float32
        )  # (G, M) M-contiguous in one reduction
    row_sum_t = torch.empty((groups, -(-m // 4) * 4), dtype=torch.float32, device=input_a.device)
    sums = input_a.reshape(m, groups, group_size).sum(dim=-1, dtype=torch.float32)  # (M, G)
    row_sum_t[:, :m].copy_(sums.t())
    return row_sum_t


def prepare_w4a16_prefill_operand(
    scale_b: torch.Tensor, zero_points: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """The prefill kernel's scale operand, N-contiguous so one K step reads contiguous rows.

    ``(K // 128, N)`` fp16 scales, or with ``zero_points`` (same ``(N, K // 128)`` shape as
    ``scale_b``) ``(2 * K // 128, N)`` whose rows ``2g`` and ``2g + 1`` hold ``s`` and
    ``(8 - zp) * s``. Weight-derived, so a prepared frame builds it once at load time."""
    if zero_points is None:
        return scale_b.t().contiguous().to(torch.float16)
    scale = scale_b.float()
    zero_bias = (DENSE_W4A16_GEMM_NIBBLE_BIAS - zero_points.float()) * scale
    return torch.stack([scale.t(), zero_bias.t()], dim=1).reshape(-1, scale_b.shape[0]).to(torch.float16)


# Small M takes the GEMV instantiation: it reads every weight byte once for all rows and streams the
# activation from L2. The rules below are the measured crossovers against the SQMMA kernel on the Qwen2
# per-rank shapes (MTT S5000, cold L2): up to 2 rows on very long K, up to 4 rows with zero points,
# and for symmetric weights 1 row on very wide outputs and 4 rows otherwise.
DENSE_W4A16_GEMV_WIDE_N = 8192
DENSE_W4A16_GEMV_LONG_K = 4096
DENSE_W4A16_GEMV_MAX_CHUNKS_PER_LANE = 12
def dense_w4a16_gemv_max_m(size_n: int, size_k: int, zero_points: bool) -> int:
    if size_k >= DENSE_W4A16_GEMV_LONG_K and size_n < DENSE_W4A16_GEMV_WIDE_N:
        return 2
    if zero_points:
        return 4
    if size_n >= DENSE_W4A16_GEMV_WIDE_N:
        return 1
    return 4


def dense_w4a16_gemv_config_for(size_m: int, size_n: int, size_k: int, zero_points: bool = False):
    """The GEMV config for one launch, or ``None`` when the SQMMA kernel serves it.

    Pure, like :func:`dense_w4a16_gemm_tile_for`, and decided inside the op so it is never frozen at
    trace time. Lanes per output column: 16 for very wide outputs (many columns already fill the
    machine), 64 for long K (each column needs more lanes to keep loads in flight), 32 otherwise.
    """
    if size_m < 1 or size_m > dense_w4a16_gemv_max_m(size_n, size_k, zero_points):
        return None
    if size_k % DENSE_W4A16_GEMM_GROUP_SIZE:
        return None
    if size_n >= DENSE_W4A16_GEMV_WIDE_N:
        lanes_per_row, k_split = 8, 2
    elif size_k >= DENSE_W4A16_GEMV_LONG_K:
        lanes_per_row, k_split = 16, 4
    else:
        lanes_per_row, k_split = 8, 4
    chunks = size_k // 32
    chunks_per_lane = -(-chunks // (lanes_per_row * k_split))
    if chunks_per_lane > DENSE_W4A16_GEMV_MAX_CHUNKS_PER_LANE:
        return None
    return DenseW4A16GemvConfig(
        m_max=1 << (size_m - 1).bit_length(),
        lanes_per_row=lanes_per_row,
        rows_per_group=1,
        k_split=k_split,
        chunks_per_lane=chunks_per_lane,
        warps=8,
        a_smem=False,
    )


@functools.cache
def _load_dense_w4a16_gemv_module(
    out_dtype: torch.dtype, zero_points: bool, config: DenseW4A16GemvConfig
):
    return gen_dense_w4a16_gemv_spec(out_dtype, zero_points, config).build_and_load()


def get_dense_w4a16_gemv_module(
    out_dtype: torch.dtype, zero_points: bool, config: DenseW4A16GemvConfig
):
    return (
        dense_w4a16_gemv_dispatch_name(out_dtype, zero_points, config),
        _load_dense_w4a16_gemv_module(out_dtype, zero_points, config),
    )


@functools.cache
def _load_dense_w4a16_gemm_module(out_dtype: torch.dtype, zero_points: bool, tile=None):
    return gen_dense_w4a16_gemm_spec(out_dtype, zero_points, tile).build_and_load()


def get_dense_w4a16_gemm_module(out_dtype: torch.dtype, zero_points: bool, tile=None):
    return (
        dense_w4a16_gemm_dispatch_name(out_dtype, zero_points, tile),
        _load_dense_w4a16_gemm_module(out_dtype, zero_points, tile),
    )


@functools.cache
def _dense_w4a16_gemm_entry(out_dtype: torch.dtype, zero_points: bool, tile=None, reduce_entry=False):
    dispatch_name, module = get_dense_w4a16_gemm_module(out_dtype, zero_points, tile)
    suffix = "_reduce" if reduce_entry else ""
    return module.get_function(f"{dispatch_name}{suffix}")


def get_dense_w4a16_gemm_functions(out_dtype: torch.dtype, zero_points: bool, tile=None):
    """``(launch, reduce)`` for one instantiation.

    ``launch`` also performs the cross-split reduction, because a split costs one extra kernel launch
    and the host-side dispatch of a second entry point was measured at ~6 us against a 32 us kernel.
    ``reduce`` is the same reduction on its own, for measuring it.
    """
    return (
        _dense_w4a16_gemm_entry(out_dtype, zero_points, tile),
        _dense_w4a16_gemm_entry(out_dtype, zero_points, tile, True),
    )


def apply_gidx_permutation(
    weight: torch.Tensor, g_idx: Optional[torch.Tensor]
) -> torch.Tensor:
    """Materialize the act-order (g_idx) permutation into the packed weight.

    The kernel copies K contiguously and asserts that it is given an identity ordering, so a
    permutation has to land here, in the packed operand, rather than inside the kernel. Only the
    identity ordering is accepted until the act-order layout itself is defined.
    """
    if g_idx is None:
        return weight
    if g_idx.dim() != 1:
        raise ValueError(f"g_idx must be 1-D, got shape {tuple(g_idx.shape)}")
    if not bool(
        torch.equal(
            g_idx, torch.arange(g_idx.numel(), device=g_idx.device, dtype=g_idx.dtype)
        )
    ):
        raise NotImplementedError(
            "act-order (desc_act) checkpoints are not supported yet: g_idx must be the identity"
        )
    return weight


def fold_zero_points(
    zero_points: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Fold per-group zero points into the single correction stream the kernel consumes.

    With the packed nibbles holding the two's-complement value `q`, the atom computes `reduce_k(a * q)`
    and the true weight is `w = scale * (q + nibble_bias - zero_point)`. The residual is the rank-1 term
    `(nibble_bias - zero_point) * scale * reduce_k(a)`, whose second factor this function returns in
    fp16. A reference that dequantizes the packed nibbles without that term is off by
    `(nibble_bias - zero_point) * sum(scale)` per group, so parity tests must use the folded reference.
    """
    if zero_points.shape != scale_b.shape:
        raise ValueError(
            f"zero_points shape {tuple(zero_points.shape)} must match scale_b shape {tuple(scale_b.shape)}"
        )
    correction = (
        float(MATE_W4A16_WEIGHT_LAYOUT.nibble_bias) - zero_points.to(torch.float32)
    ) * scale_b.to(torch.float32)
    return correction.to(scale_b.dtype).contiguous()


def _check_contiguous_last_dim(tensor: torch.Tensor, name: str) -> None:
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in its last dimension")


def _pad_last_dim_to_even(tensor: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Pad the group dimension to an even count, which the paired 16-bit scale words require."""
    if tensor.shape[-1] % 2 == 0:
        return tensor
    padded = torch.full(
        (*tensor.shape[:-1], tensor.shape[-1] + 1),
        fill,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    padded[..., :-1].copy_(tensor)
    return padded.contiguous()


def prepare_w4a16_call_operands(scale_b, zero_points):
    """Precompute the two weight-derived operands that ``dense_w4a16_gemm`` otherwise rebuilds on
    every call.

    Both depend only on the weights, so a consumer that holds a prepared frame calls this once per
    layer instead of once per forward step: ``fold_zero_points`` costs several device ops and a fresh
    allocation, which at decode shapes is the same order as the GEMM itself. Returns
    ``(scale_b_padded, correction)``, where ``correction`` is ``None`` for the symmetric case.
    """
    scale_b_padded = _pad_last_dim_to_even(scale_b)
    if zero_points is None:
        return scale_b_padded, None
    bias = float(MATE_W4A16_WEIGHT_LAYOUT.nibble_bias)
    return scale_b_padded, fold_zero_points(
        _pad_last_dim_to_even(zero_points, fill=bias), scale_b_padded
    )


def dense_w4a16_gemm_mutlass(
    input_a: torch.Tensor,
    weight_b: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    zero_points: Optional[torch.Tensor] = None,
    g_idx: Optional[torch.Tensor] = None,
    num_mps: Optional[int] = None,
    scale_b_padded: Optional[torch.Tensor] = None,
    correction_operand: Optional[torch.Tensor] = None,
    scale_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the dense W4A16 GEMM. See `MateW4A16WeightLayout` for the packed-weight contract.

    `input_a` is (M, K) fp16, `weight_b` is (N, K // 2) int8, `scale_b` is (N, K // 128) fp16,
    `zero_points` (optional, same shape as `scale_b`) holds the per-group AWQ zero points, and `out`
    is the preallocated (M, N) result.
    """
    tile_k = DENSE_W4A16_GEMM_TILE[2]
    group_size = MATE_W4A16_WEIGHT_LAYOUT.scale_group_size

    if input_a.dim() != 2 or weight_b.dim() != 2 or out.dim() != 2:
        raise ValueError("input_a, weight_b and out must all be 2-D")
    if input_a.dtype != torch.float16:
        raise ValueError(f"input_a must be fp16, got {input_a.dtype}")
    if weight_b.dtype != torch.int8:
        raise ValueError(
            f"weight_b must be int8 holding packed INT4 pairs, got {weight_b.dtype}"
        )
    if out.dtype not in DENSE_W4A16_GEMM_OUTPUT_DTYPE_TAGS:
        raise ValueError(f"out must be fp16 or bf16, got {out.dtype}")

    m, k = input_a.shape
    n = weight_b.shape[0]
    if k % tile_k:
        raise ValueError(f"K must be a multiple of the tile K ({tile_k}), got {k}")
    if weight_b.shape[1] * 2 != k:
        raise ValueError(
            f"weight_b holds {weight_b.shape[1] * 2} K values but input_a has K = {k}"
        )
    expected_scale_cols = k // group_size
    if scale_b.shape != (n, expected_scale_cols):
        raise ValueError(
            f"scale_b must have shape {(n, expected_scale_cols)}, got {tuple(scale_b.shape)}"
        )
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")

    for tensor, name in (
        (input_a, "input_a"),
        (weight_b, "weight_b"),
        (scale_b, "scale_b"),
        (out, "out"),
    ):
        _check_contiguous_last_dim(tensor, name)
        if tensor.device != input_a.device:
            raise ValueError(
                f"{name} is on {tensor.device} while input_a is on {input_a.device}; every operand "
                "is read by the kernel through a device pointer"
            )
    if zero_points is not None and zero_points.device != input_a.device:
        raise ValueError(
            f"zero_points is on {zero_points.device} while input_a is on {input_a.device}"
        )

    weight_b = apply_gidx_permutation(weight_b, g_idx)

    # The packed scale words pair two 16-bit group values, so the group dimension has to be even. A
    # padded zero-point column is filled with the nibble bias, which makes its folded correction
    # exactly zero, and a padded scale column is zero, which keeps the product zero as well.
    if zero_points is not None and zero_points.shape != scale_b.shape:
        raise ValueError(
            f"zero_points shape {tuple(zero_points.shape)} must match scale_b shape "
            f"{tuple(scale_b.shape)}"
        )
    if scale_b_padded is None or (zero_points is not None and correction_operand is None):
        # The slow path: rebuild the weight-derived operands. Callers that hold a prepared frame pass
        # them in and skip this entirely.
        scale_b_padded, correction = prepare_w4a16_call_operands(scale_b, zero_points)
        if correction_operand is None:
            correction_operand = correction
    correction = correction_operand

    correction_or_scale = correction if correction is not None else scale_b_padded
    if (
        DENSE_W4A16_PREFILL_MIN_M
        and dense_w4a16_use_prefill(m, n)
        and n % 8 == 0  # the N-contiguous scale rows must stay 16-byte aligned for the TME
        and out.stride(1) == 1
    ):
        launch_mps = resolve_num_mps(input_a.device, num_mps)
        tile = dense_w4a16_prefill_tile_for(m, n, k, launch_mps)
        dispatch_name, module = get_dense_w4a16_prefill_module(
            out.dtype, tile, zero_points is not None
        )
        scale_operand = (
            scale_t if scale_t is not None else prepare_w4a16_prefill_operand(scale_b, zero_points)
        )
        module.get_function(dispatch_name)(
            input_a,
            weight_b,
            scale_operand,
            prefill_row_sums(input_a) if zero_points is not None else scale_operand,
            out,
            launch_mps,
        )
        return out

    gemv_config = dense_w4a16_gemv_config_for(m, n, k, correction is not None)
    # The GEMV reads 16-byte activation vectors, so each row has to start 16-byte aligned.
    if gemv_config is not None and input_a.stride(0) % 8 == 0 and input_a.data_ptr() % 16 == 0:
        dispatch_name, module = get_dense_w4a16_gemv_module(
            out.dtype, correction is not None, gemv_config
        )
        module.get_function(dispatch_name)(
            input_a, weight_b, scale_b_padded, correction_or_scale, out
        )
        return out

    launch_mps = resolve_num_mps(input_a.device, num_mps)
    # Price both decode tiles against this launch's K and keep the cheaper (tile, split) pair. Choosing
    # one argmaxes nothing: the tile fixes the tile count, the split turns that count into work items,
    # and the blocks each CTA streams is the product the cost model prices.
    tile, k_split = dense_w4a16_gemm_plan(
        input_a.shape[0], out.shape[1], k, num_mps=launch_mps
    )
    # A split can never be wider than the K axis it splits.
    k_split = min(k_split, -(-k // tile[2]))
    launch = _dense_w4a16_gemm_entry(out.dtype, correction is not None, tile)
    if k_split <= 1:
        # One split is the whole-K launch: the kernel writes `out` directly. `out` is handed over as
        # the unused partial operand only to keep the argument list fixed across both modes.
        launch(input_a, weight_b, scale_b_padded, correction_or_scale, out, out, launch_mps, 1)
        return out
    # Each split accumulates its own fp32 partial over its K range, so the cross-split reduction is a
    # plain deterministic sum in a fixed order with no atomics: capture-safe, and the same inputs give
    # the same bits. The same launch reduces the partials into `out` once the GEMM is done.
    m_pad = -(-input_a.shape[0] // tile[0]) * tile[0]
    partials = torch.empty(
        (k_split, m_pad, out.shape[1]), dtype=torch.float32, device=input_a.device
    )
    launch(input_a, weight_b, scale_b_padded, correction_or_scale, out, partials, launch_mps, k_split)
    return out


# dynamo must not trace into the loader, and neither `torch._dynamo.disable` (rejected outright by
# the fullgraph capture the serving path uses) nor `allow_in_graph` (which still executes this entry
# with fake tensors, and the entry takes data pointers) is enough. PyTorch's own diagnosis for that
# runtime error is "wrap the custom kernel into an opaque custom op", which is also how kernels enter
# vLLM: a registered op with a fake implementation, not traced Python.
#
# Registration is best-effort. It runs at import time, and a registration that cannot complete -- a
# duplicate registration under a second import path, an API difference, a namespace clash -- used to
# take the whole module's import down, which the runtime reports as the module not existing at all.
def _register_opaque_op():
    """Register the opaque op, or hand back the raw entry so this module still imports."""
    if not hasattr(torch.library, "custom_op"):
        return dense_w4a16_gemm_mutlass
    try:

        @torch.library.custom_op("mate_w4a16::gemm", mutates_args={"out"})
        def _op(
            input_a: torch.Tensor,
            weight_b: torch.Tensor,
            scale_b: torch.Tensor,
            out: torch.Tensor,
            zero_points: Optional[torch.Tensor] = None,
            g_idx: Optional[torch.Tensor] = None,
            num_mps: Optional[int] = None,
            scale_b_padded: Optional[torch.Tensor] = None,
            correction_operand: Optional[torch.Tensor] = None,
            scale_t: Optional[torch.Tensor] = None,
        ) -> None:
            # Returns nothing: `out` is declared mutated, and a custom op may not return a tensor
            # that aliases one of its inputs.
            dense_w4a16_gemm_mutlass(
                input_a,
                weight_b,
                scale_b,
                out,
                zero_points=zero_points,
                g_idx=g_idx,
                num_mps=num_mps,
                scale_b_padded=scale_b_padded,
                correction_operand=correction_operand,
                scale_t=scale_t,
            )

        @_op.register_fake
        def _fake(
            input_a, weight_b, scale_b, out, zero_points=None, g_idx=None, num_mps=None,
            scale_b_padded=None, correction_operand=None, scale_t=None,
        ) -> None:
            return None

        return _op
    except Exception as err:  # pragma: no cover - fall back rather than break the import
        print(f"mate: opaque-op registration unavailable ({type(err).__name__}: {err})")
        return dense_w4a16_gemm_mutlass


dense_w4a16_gemm_op = _register_opaque_op()
