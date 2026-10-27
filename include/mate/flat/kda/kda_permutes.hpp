#pragma once

#include <mutlass/arch/arch.h>

#include <mute/layout.hpp>

namespace mate::flat::kda {

// The KDA axis permutations: the compile-time layout isomorphisms that the two
// kernels -- or a kernel and the workspace ABI -- have to interpret the same
// way.
//
// Each is a mute layout pinned by a bidirectional static_assert, because the
// alternative is a closed form on one side and a comment restating it on the
// other, and nothing makes the two move together.
//
// The names read as "permute <axis>": PermuteN permutes the GEMM's N axis, and
// its parameter is that axis's width (a value-axis PermuteV would sit beside it).

// The order of the headDim axis of the k_restored workspace field.  headDim is
// the UK GEMM's N axis, so each architecture's frame is a permutation of N.
//
// Declared but not defined: an architecture without a frame is a hard error
// rather than a silent fallback.
template <class ArchTag, int HeadDim>
struct PermuteN;

// ---------------------------------------------------------------------------
// MP31
//
// K1 (prepare) writes logical key h of k_restored at slot
//
//     slot = G^-1(h) = 8 * (h % 16) + h / 16        (h < 128)
//
// so that K2's state squad lane holds 16 CONSECUTIVE smem slots per row -- one
// 32 B vector store instead of 16 stride-8 scalar stores.  Writing h = 16*a + b
// with a < 8, b < 16 gives slot = a + 8*b: a transpose of the (a, b) 8x16 block.
//
// K2 does not evaluate this -- the permutation is baked into its UK operand
// layout -- which is why the frame has one caller rather than two.  It still
// belongs in a frame: it is the one place the order is written down, and K1
// used to spell it out inline.
template <int HeadDim>
struct PermuteN<mutlass::arch::Mp31, HeadDim> {
  static_assert(HeadDim == 128, "the MP31 PermuteN is the 8x16 decomposition of a 128-wide headDim");

  // mute's idx2crd is column-major (mode 0 varies fastest) and Step assigns
  // strides in increasing step order, so the domain is Shape<_16, _8>: h = 16*a
  // + b lands on (b, a) and the strides below give 8*b + a.
  using LogicalToSlot =
      decltype(mute::make_ordered_layout(mute::Shape<mute::_16, mute::_8>{}, mute::Step<mute::_8, mute::_1>{}));
  using SlotToLogical = decltype(mute::right_inverse(LogicalToSlot{}));

  // h = 19 -> (a, b) = (1, 3) -> 25;  h = 61 -> (a, b) = (3, 13) -> 107.
  static_assert(LogicalToSlot{}(19) == 25 && LogicalToSlot{}(61) == 107, "MP31 PermuteN moved");
  static_assert(SlotToLogical{}(25) == 19 && SlotToLogical{}(107) == 61, "MP31 PermuteN moved");
};

}  // namespace mate::flat::kda

// ---------------------------------------------------------------------------
// The value axis.
//
// Which tile row holds a given value column, and what an mma atom's C
// coordinate means on this axis.  The two architectures answer differently --
// see the specialisations -- and that difference is what makes each arch's
// packed 16 B store legal.
//
// A map is a bijection as soon as its extents multiply out to ValueTile:
// make_ordered_layout, filter and right_inverse all yield compact layouts.  So
// each frame asserts the coverage it is responsible for -- that its reshape is
// exactly the value axis -- and pins the concrete order with literal values.
//
// Declared but not defined, like PermuteN.
template <class ArchTag, int ValueTile>
struct PermuteV;

// ---------------------------------------------------------------------------
// MP31
//
// The SQMMA enumerates a lane's value columns in 8x8-transposed blocks, and the
// state tile stores the logical order.  An atom coordinate therefore reaches the
// tile row through that transpose, and a logical column IS its tile row.
template <int ValueTile>
struct PermuteV<mutlass::arch::Mp31, ValueTile> {
  static_assert(ValueTile == 32 || ValueTile % 64 == 0,
                "MP31's value permutation is a whole number of 64-column blocks, or 32");

  using Natural = mute::Layout<mute::Shape<mute::Int<ValueTile>>, mute::Stride<mute::_1>>;
  // The third mode is 1 for ValueTile == 64 and gets filtered away.  It is also 1
  // at 32, where the conditional below picks Natural instead.
  using Transposed = decltype(mute::filter(
      mute::make_ordered_layout(mute::Shape<mute::_8, mute::_8, mute::Int<(ValueTile + 63) / 64>>{},
                                mute::Step<mute::_2, mute::_1, mute::_3>{})));

  static constexpr bool kTransposed = ValueTile != 32;

  using TileRowFromInternal = std::conditional_t<kTransposed, Transposed, Natural>;
  using TileRowFromLogical  = Natural;
  using LogicalFromTileRow  = Natural;
  using LogicalFromInternal = TileRowFromInternal;

  // At 32 columns the frame is Natural, so only the transpose needs checking.
  static_assert(!kTransposed || mute::size(Transposed{}) == ValueTile,
                "MP31's value transpose must cover exactly ValueTile columns");
  // 64 columns: value 1 sits at row 8 and value 8 at row 1 (the 8x8 block transpose).
  static_assert(!kTransposed || (TileRowFromInternal{}(1) == 8 && TileRowFromInternal{}(8) == 1),
                "MP31's value transpose moved");
};
