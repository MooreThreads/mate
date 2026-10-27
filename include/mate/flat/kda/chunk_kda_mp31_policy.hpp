#pragma once

#include "mate/flat/kda/chunk_kda_problem.hpp"

namespace mate::flat::kda {

// MP31 occupancy/resource choices.  They are injected into the common work
// traversal instead of being treated as KDA semantics.
// prepare CTAs/MP=5, recurrence CTAs/MP=3, value tile=64,
// metadata threads=512, metadata dynamic smem limit=192 KiB.
//
// The recurrence's register demand is dominated by its 2048-bit SQMMA state
// accumulator (a 64x128 fp32 fragment, 64 dword per thread in the one state
// squad), so at recurrence CTAs/MP = 3 the kernel already uses 107-108 of the
// 113 dword/thread the launch bound allows.  A 5.1.0-line backend cannot
// allocate those accumulators at all ("TEMP_2048"); mate/jit/kda_ops.py defines
// MATE_KDA_MP31_REC_CTAS_PER_MP=2 for such compilers.  Grid and cap move
// together by construction: lowering only the cap measured no better at H=64/84
// and worse at H=168.
#if defined(MATE_KDA_MP31_REC_CTAS_PER_MP)
inline constexpr int kMp31RecurrenceCtasPerMp = MATE_KDA_MP31_REC_CTAS_PER_MP;
#else
inline constexpr int kMp31RecurrenceCtasPerMp = 3;
#endif

struct ChunkKdaMp31SchedulePolicy : ChunkKdaPrefillSchedulePolicy<5, kMp31RecurrenceCtasPerMp, 64, 512, 192 * 1024> {};

}  // namespace mate::flat::kda
