#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>

#include "mate/flat/simd_helper.hpp"

namespace mate::flat::collective {

using namespace mute;

// Lower-triangular inverse of a Block x Block matrix, solved one column per
// thread, so Block must fit within a single warp.
//
// The solved columns live in shared_storage.solution - one padded row per
// column - and are drained by publish_solution().  This layout is fixed, it is
// not a tuning knob, and there are three reasons for it, in decreasing order of
// how easy they are to get wrong:
//
//  1. Keeping the working vector in registers is what makes this solve the
//     register peak of the KDA prepare kernel.  The substitution is sequential,
//     so a thread would hold Block floats of live state for the whole solve,
//     read back as float4 while it walks the rows of the matrix.  Measured on
//     MP32 with the vector in registers: 72 registers per thread, which is
//     131072 / (256 * 72) = 7 resident CTAs.  In shared memory the same solve
//     costs one padded row per column (1280 bytes) and lands at 62 registers -
//     131072 / (256 * 62) = 8 CTAs - without a spilled byte.  MP31 is capped at
//     six CTAs by shared memory either way, so the tile changes nothing there.
//
//  2. The unrolled loops below are what make the shared-memory form cheap: with
//    1280 extra bytes free, do NOT let a loop bound depend on col.  "The
//     solution of row < col is zero, so start the row walk at col" is correct
//     algebra and a 4.32x regression: a runtime bound stops the vector from
//     being promoted, the compiler demotes it to a stack frame
//     (private_memory_size 0 -> 64, plus an LSU.ST.B512 of all 16 floats per
//     row), and the kernel goes from 0.2198 ms to 0.9484 ms.  Skipping the
//     leading zeros buys nothing anyway - the warp issues the deepest lane's
//     trip count, and lane col = 0 still walks all Block rows.
//
//  3. publish_solution() is split out of solve_columns() on purpose.  Folding
//     the global store back into the solve keeps the 64-bit output pointer live
//     across the whole substitution, which measured two more registers than
//     computing the address afterwards.
template <class Element_, int Block_>
struct CollectiveLowerTriangularInverseNxN {
  using Element = Element_;

  static constexpr int Block   = Block_;
  static constexpr int VecSize = 4;

  using SmemLayout = decltype(make_layout(make_shape(Int<Block>{}, Int<Block>{}), make_stride(Int<Block>{}, _1{})));

  // One row of the solve tile per column, padded by a full vector: lane c then
  // reads its float4s from bank group (5 * c) mod 8, so the eight lanes of a
  // 256-byte phase never share a bank group.
  static constexpr int SolutionStride = Block + VecSize;
  static constexpr int SolutionWords  = Block * SolutionStride;

  struct SharedStorage {
    alignas(16) float matrix[cosize_v<SmemLayout>];
    // The solve's working vector: one padded row per column, i.e. lane c owns
    // shared_storage.solution[c * SolutionStride .. c * SolutionStride + Block].
    alignas(16) float solution[SolutionWords];
  };

  static_assert(Block > 0);
  static_assert(Block <= mutlass::NumThreadsPerWarp);
  static_assert(Block % VecSize == 0);

  // Forward substitution of one column: solution[row] = -sum_{k<row} A[row][k] *
  // solution[k], seeded with -beta at the diagonal and zero above it.  solution
  // is the column's row of the tile, so every access below is an immediate
  // offset - see note 2 in the header before touching either trip count.
  template <class TensorMatrix>
  MUTLASS_DEVICE static void solve_column(TensorMatrix const& matrix, float beta_col, int col, float* solution) {
    MUTE_UNROLL
    for (int row = 0; row < Block; ++row) {
      solution[row] = 0.0f;
    }

    float rhs = beta_col;

    MUTE_UNROLL
    for (int row = 0; row < Block; ++row) {
      float value = row == col ? -rhs : 0.0f;

      MUTE_UNROLL
      for (int k = 0; k < row; k += VecSize) {
        float4 coeffs = simd::load_float4(&matrix(row, k));
        float4 sol    = *reinterpret_cast<float4 const*>(solution + k);
        simd::dot4(coeffs, sol, value);
      }

      solution[row] = -value;
    }
  }

  // Solves every column into shared_storage.solution.  sync_barrier is the
  // caller's rendezvous with whatever produced the matrix, so it runs before
  // the substitution; the caller must rendezvous again before the matrix is
  // reused for the next chunk.  Only lanes [0, Block) participate, which is why
  // Block must fit in one warp.
  template <class SyncBarrier>
  MUTLASS_DEVICE void solve_columns(SharedStorage& shared_storage,
                                    float          beta_lane,
                                    SyncBarrier&&  sync_barrier,
                                    int            local_tid) const {
    auto matrix = make_tensor(make_smem_ptr(shared_storage.matrix), SmemLayout{});
    sync_barrier();

    if (local_tid < Block) {
      int    col  = local_tid;
      float* tile = shared_storage.solution + col * SolutionStride;
      solve_column(matrix, beta_lane, col, tile);
    }
  }

  // Drains the tile to the caller's output.  Deliberately a separate step from
  // solve_columns: the output address is only known here, so the 64-bit pointer
  // stays out of the solve's live range (see note 3 in the header).  The whole
  // Block x Block tile is written, including the zero triangle above the
  // diagonal, because the consumer reads all Block rows.
  MUTLASS_DEVICE static void publish_solution(SharedStorage const& shared_storage, Element* inv_output, int local_tid) {
    if (local_tid < Block) {
      int          col  = local_tid;
      float const* tile = shared_storage.solution + col * SolutionStride;
      using SolutionVec = mutlass::Array<float, VecSize>;
      using ElementVec  = mutlass::Array<Element, VecSize>;
      mutlass::NumericArrayConverter<Element, float, VecSize, mutlass::FloatRoundStyle::round_to_nearest>
          convert_solution;

      MUTE_UNROLL
      for (int row = 0; row < Block; row += VecSize) {
        SolutionVec values                  = *reinterpret_cast<SolutionVec const*>(tile + row);
        ElementVec  packed                  = convert_solution(values);
        inv_output[(row + 0) * Block + col] = packed[0];
        inv_output[(row + 1) * Block + col] = packed[1];
        inv_output[(row + 2) * Block + col] = packed[2];
        inv_output[(row + 3) * Block + col] = packed[3];
      }
    }
  }
};

}  // namespace mate::flat::collective
