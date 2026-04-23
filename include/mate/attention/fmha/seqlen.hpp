#pragma once

#include <cstdint>

#include "mutlass/mutlass.h"

namespace mate::attention::fmha {

template <bool HasCuSeqlens, bool HasSeqused>
struct SeqlenInfoAny {
  uint32_t const offset;
  uint32_t const seqlen;

  MUTLASS_DEVICE
  SeqlenInfoAny(uint32_t const        batch_idx,
                uint32_t const        seqlen_static,
                uint32_t const* const cu_seqlens,
                uint32_t const* const seqused)
      : offset(HasCuSeqlens ? cu_seqlens[batch_idx] : 0),
        seqlen(HasSeqused ? seqused[batch_idx]
                          : (HasCuSeqlens ? cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx] : seqlen_static)) {
  }
};

template <bool HasCuSeqlensQ,
          bool HasSequsedQ,
          bool HasCuSeqlensK,
          bool HasSequsedK,
          bool HasLeftpadK,
          bool AppendKV,
          bool HasCuSeqlensKNew,
          bool EnableCP = false>
struct SeqlenInfoQK {
  static constexpr bool Varlen = HasCuSeqlensQ || HasSequsedQ || HasCuSeqlensK || HasSequsedK || HasLeftpadK;
  static_assert(!(AppendKV && !Varlen), "AppendKV is only supported with Varlen");

  // uint32_t const leftpad_k;

  uint32_t const offset_q;
  uint32_t const offset_k;
  uint32_t const offset_k_new;
  uint32_t const seqlen_q;
  uint32_t const seqlen_k_og;
  uint32_t const seqlen_k_new;
  uint32_t const seqlen_k;
  // uint32_t const seqlen_rotary;

  // CP
  int const cp_world_size;
  int const cp_rank;
  int const tot_seqlen_k;

  MUTLASS_DEVICE
  SeqlenInfoQK(uint32_t const        batch_idx,
               uint32_t const        seqlen_q_static,
               uint32_t const        seqlen_k_static,
               uint32_t const        seqlen_k_new_static,
               uint32_t const* const cu_seqlens_q,
               uint32_t const* const cu_seqlens_k,
               uint32_t const* const cu_seqlens_k_new,
               uint32_t const* const seqused_q,
               uint32_t const* const seqused_k,
               //  uint32_t const* const ptr_leftpad_k,
               //  uint32_t const* const seqlens_rotary,
               int const             cp_world_size_   = 1,
               int const             cp_rank_         = 0,
               uint32_t const* const cp_tot_seqused_k = nullptr)
      // : leftpad_k(HasLeftpadK ? ptr_leftpad_k[batch_idx] : 0),
      : offset_q(HasCuSeqlensQ ? cu_seqlens_q[batch_idx] : 0),
        offset_k(HasCuSeqlensK ? cu_seqlens_k[batch_idx] : 0),
        offset_k_new(!AppendKV || !HasCuSeqlensKNew ? 0 : cu_seqlens_k_new[batch_idx]),
        seqlen_q(HasSequsedQ
                     ? seqused_q[batch_idx]
                     : (HasCuSeqlensQ ? cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx] : seqlen_q_static)),
        seqlen_k_og([&]() -> uint32_t {
          if constexpr (HasSequsedK) {
            return seqused_k[batch_idx];
          }

          if constexpr (HasCuSeqlensK) {
            return cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx];
          }

          return seqlen_k_static;
        }()),
        seqlen_k_new(!AppendKV ? 0
                               : (HasCuSeqlensKNew ? cu_seqlens_k_new[batch_idx + 1] - cu_seqlens_k_new[batch_idx]
                                                   : seqlen_k_new_static)),
        seqlen_k(!AppendKV ? seqlen_k_og : seqlen_k_og + seqlen_k_new),
        // seqlen_rotary(!AppendKV || !HasSeqlensRotary ? seqlen_k_og + leftpad_k : seqlens_rotary[batch_idx]),
        cp_world_size(cp_world_size_),
        cp_rank(cp_rank_),
        tot_seqlen_k(EnableCP ? static_cast<int>(cp_tot_seqused_k[batch_idx])
                              : (static_cast<int>(
                                    HasSequsedK ? seqused_k[batch_idx]
                                                : (HasCuSeqlensK ? cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx]
                                                                 : seqlen_k_static)))) {
  }
};

}  // namespace mate::attention::fmha
