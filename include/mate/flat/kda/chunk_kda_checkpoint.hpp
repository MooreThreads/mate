#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>

namespace mate::flat::kda {

// Row mapping for the compact checkpoint pool.
//
// `chunk_kda(checkpoint_every_n_tokens = C)` writes the recurrent state that
// enters token `j * C` of sequence `b` into entry `j` of that sequence, for
// `j = 0 .. (len_b - 1) / C` (entry 0 is the incoming state; the state after
// the last token is `final_state`, never a checkpoint).  Entries are packed per
// sequence in `cu_seqlens` order, so the pool stays dense:
//
//   row(b, j) = base(b) + j
//   base(b)   = sum_{b' < b} cnt(b')
//   cnt(b)    = 0                     if len_b == 0
//             = (len_b - 1) / C + 1   otherwise
//
// which is the same prefix sum cudnn-frontend derives on device in
// `frost/common/thd.py` (`emit_checkpoint_seq_descs`).  Empty sequences
// contribute no rows.
//
// A dense batch has one length for every sequence, so the base collapses to
// `seq_idx * ceil(T / C)` and needs no loop.
//
// `EveryNTokens` is the compile-time cadence from
// `Components::CheckpointEveryNTokens`: positive and a multiple of the kernel
// chunk.  `make_recurrence_work` calls this once per CTA under
// `Components::HasCheckpointCadence`, whose assertion in the components
// establishes that, so there is no check here; the result is cached in the work
// descriptor instead of being recomputed for every stored row.  `IsVarlen`
// comes from the components so the dense build never touches `cu_seqlens`.
template <bool IsVarlen, int EveryNTokens, class ProblemSize, class WorkDesc>
MUTLASS_HOST_DEVICE int64_t chunk_kda_checkpoint_base(ProblemSize const& problem_size, WorkDesc const& work_desc) {
  if constexpr (!IsVarlen) {
    int64_t const per_sequence = mutlass::ceil_div(int64_t(problem_size.T), int64_t(EveryNTokens));
    return int64_t(work_desc.seq_idx) * per_sequence;
  } else {
    int64_t base = 0;
    // The recurrence is one CTA per (sequence, head, value tile) chain, and the
    // scan is O(seq_idx) once per checkpoint row: the sequences before this one
    // are the only ones that can contribute to its base.
    for (int seq = 0; seq < int(work_desc.seq_idx); ++seq) {
      int64_t const length = int64_t(problem_size.cu_seqlens[seq + 1]) - int64_t(problem_size.cu_seqlens[seq]);
      if (length > 0) {
        base += mutlass::ceil_div(length, int64_t(EveryNTokens));
      }
    }
    return base;
  }
}

}  // namespace mate::flat::kda
