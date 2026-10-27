#pragma once

#include <cstdint>

#include "mutlass/arch/barrier.hpp"

namespace mate::flat::kda {

using KdaNamedBarrier = uint32_t;

// The prepare kernel's barrier map: the one owner of the ids.  The norm ring is
// one stage deep, the decayed tile needs a publish side only, and the three raw
// TME rings are sized by RawRingSize.
template <int NormRingSize_, int RawRingSize_>
struct PrepareBarriers {
  static constexpr int NormRingSize = NormRingSize_;
  static constexpr int RawRingSize  = RawRingSize_;

  // One rendezvous for the staged P tile, shared by all four consumer warps.  It
  // also carries the cross-chunk ordering of the staging tile: the four warps
  // meet here at the end of every chunk, so the separate start-of-chunk
  // rendezvous -- and the sync inside the nxn inverse collective -- is no longer
  // needed.
  // The QK^T hand-off, both directions on one pair: the four consumer warps
  // arrive KktReady after writing their half of the tile (one pair writes the
  // KKT matrix, the other stages P), and the solve warp waits that flip before
  // reading the matrix.  KktFree is armed by the solve warp once it is done with
  // the matrix.
  static constexpr int KktReady = 0;
  static constexpr int KktFree  = KktReady + 1;
  // K1's hand-off rings, one full/empty pair per stage.  The arrival counts live
  // in the collective's init loop, next to the roles that arrive there.
  static constexpr int NormFull  = KktFree + 1;
  static constexpr int NormEmpty = NormFull + NormRingSize;
  // The decayed tile is single-staged; its reuse across chunks is ordered by
  // KktReady's rendezvous, so it needs the publish side only.
  static constexpr int DecayedReady = NormEmpty + NormRingSize;
  // One transaction barrier per raw tile, full side only: a raw tile IS the norm
  // tile, normalized in place, so the free side is NormEmpty.
  static constexpr int RawQFull    = DecayedReady + NormRingSize;
  static constexpr int RawKFull    = RawQFull + RawRingSize;
  static constexpr int RawGFull    = RawKFull + RawRingSize;
  static constexpr int NumBarriers = RawGFull + RawRingSize;
};

// The recurrence's barrier map: the one owner of the ids.
//
// Both architectures speak this protocol but do not use the same set -- MP31
// stages a V ring and reserves a state-snapshot barrier that MP32 has no use for
// -- so the set is a parameter and the id arithmetic is written once, here.
//
// A staged ring owns a full and an empty barrier of `TmeStages` ids each, and an
// unused barrier keeps its id reserved, so that an id never moves with a feature
// or an architecture.  Weights are in the collectives' init loops, next to the
// comments that justify them.
template <int TmeStages_, int NumConsumerSquads_, bool HasVRing_, bool HasStateSnapshot_>
struct RecurrenceBarriers {
  static constexpr int  TmeStages         = TmeStages_;
  static constexpr int  NumConsumerSquads = NumConsumerSquads_;
  static constexpr bool HasVRing          = HasVRing_;
  static constexpr bool HasStateSnapshot  = HasStateSnapshot_;

  static constexpr int DecayedReady  = 0;
  static constexpr int DecayedFree   = DecayedReady + TmeStages;
  static constexpr int RestoredReady = DecayedFree + TmeStages;
  static constexpr int RestoredFree  = RestoredReady + TmeStages;
  // MP31's alone.  -1 rather than a live id, so that using it on an architecture
  // without the ring is a wrong answer instead of silently aliasing inverse/P.
  static constexpr int VReady    = HasVRing ? RestoredFree + TmeStages : -1;
  static constexpr int VFree     = HasVRing ? VReady + TmeStages : -1;
  static constexpr int InvPReady = (HasVRing ? VFree : RestoredFree) + TmeStages;
  static constexpr int InvPFree  = InvPReady + TmeStages;
  // One rendezvous per consumer squad.
  static constexpr int Local = InvPFree + TmeStages;
  // The once-per-work-item CTA-boundary handshakes.
  static constexpr int Seeded  = Local + NumConsumerSquads;
  static constexpr int Drained = Seeded + 1;
  // The per-chunk cross-squad handshakes.
  static constexpr int UReady         = Drained + 1;
  static constexpr int StateCommitted = UReady + 1;
  // Reserved even when the feature is off, so the ids above never move with it.
  static constexpr int StateSnapshot = StateCommitted + 1;
  static constexpr int NumBarriers   = HasStateSnapshot ? StateSnapshot + 1 : StateCommitted + 1;
};

MUTLASS_DEVICE
static uint32_t named_barrier_id(uint32_t barrier_id_) {
  return barrier_id_ + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount;
}

template <typename Barrier>
MUTLASS_DEVICE static uint32_t named_barrier_id(Barrier barrier_id_) {
  return named_barrier_id(static_cast<KdaNamedBarrier>(barrier_id_));
}

MUTLASS_DEVICE
static void named_barrier_init(uint32_t barrier_id_, uint32_t arrive_count, uint32_t init_phase = 0) {
  mutlass::arch::AsyncBarrier::init(named_barrier_id(barrier_id_), arrive_count, init_phase);
}

template <typename Barrier>
MUTLASS_DEVICE static void named_barrier_init(Barrier barrier_id_, uint32_t arrive_count, uint32_t init_phase = 0) {
  named_barrier_init(static_cast<KdaNamedBarrier>(barrier_id_), arrive_count, init_phase);
}

MUTLASS_DEVICE
static void named_barrier_arrive(uint32_t barrier_id_) {
  mutlass::arch::AsyncBarrier::arrive(named_barrier_id(barrier_id_));
}

template <typename Barrier>
MUTLASS_DEVICE static void named_barrier_arrive(Barrier barrier_id_) {
  named_barrier_arrive(static_cast<KdaNamedBarrier>(barrier_id_));
}

MUTLASS_DEVICE
static uint32_t named_barrier_arrive_phase(uint32_t barrier_id_) {
  return mutlass::arch::AsyncBarrier::arrive<true>(named_barrier_id(barrier_id_));
}

template <typename Barrier>
MUTLASS_DEVICE static uint32_t named_barrier_arrive_phase(Barrier barrier_id_) {
  return named_barrier_arrive_phase(static_cast<KdaNamedBarrier>(barrier_id_));
}

MUTLASS_DEVICE
static void named_barrier_arrive_and_expect_tx(uint32_t barrier_id_, uint32_t transaction_bytes) {
  mutlass::arch::AsyncTransactionBarrier::arrive_and_expect_tx(named_barrier_id(barrier_id_), transaction_bytes);
}

template <typename Barrier>
MUTLASS_DEVICE static void named_barrier_arrive_and_expect_tx(Barrier barrier_id_, uint32_t transaction_bytes) {
  named_barrier_arrive_and_expect_tx(static_cast<KdaNamedBarrier>(barrier_id_), transaction_bytes);
}

MUTLASS_DEVICE
static void named_barrier_wait(uint32_t barrier_id_, uint32_t phase) {
  mutlass::arch::AsyncBarrier::wait(named_barrier_id(barrier_id_), phase);
}

template <typename Barrier>
MUTLASS_DEVICE static void named_barrier_wait(Barrier barrier_id_, uint32_t phase) {
  named_barrier_wait(static_cast<KdaNamedBarrier>(barrier_id_), phase);
}

MUTLASS_DEVICE
static void named_barrier_arrive_and_wait(uint32_t barrier_id_) {
  uint32_t barrier_id = named_barrier_id(barrier_id_);
  uint32_t phase      = mutlass::arch::AsyncBarrier::arrive<true>(barrier_id);
  mutlass::arch::AsyncBarrier::wait(barrier_id, phase);
}

template <typename Barrier>
MUTLASS_DEVICE static void named_barrier_arrive_and_wait(Barrier barrier_id_) {
  named_barrier_arrive_and_wait(static_cast<KdaNamedBarrier>(barrier_id_));
}

// ---- lane gating -----------------------------------------------------------
//
// Only lane 0 touches a barrier, so the warp has to rendezvous around it; the
// ordering comes from the barrier itself, which the stage_barrier_* helpers this
// replaced relied on the same way: __musa_async_wait is modelled as writing
// memory and __syncwarp as having unmodelled memory effects, so neither stores
// nor loads move across them.  An extra compiler fence would buy nothing.
MUTLASS_DEVICE static void arrive_warp(int id) {
  __syncwarp();
  if ((int(threadIdx.x) & 31) == 0) {
    named_barrier_arrive(id);
  }
}

// Arm one TME transaction per producer warp.  `arrive_and_expect_tx` is
// per-thread: run unguarded it arrives 32 times on these count-1/count-2 barriers
// and adds 32 transactions for the single copy that lane 0 issues.
MUTLASS_DEVICE static void expect_tx_warp(int id, uint32_t bytes) {
  __syncwarp();
  if ((int(threadIdx.x) & 31) == 0) {
    named_barrier_arrive_and_expect_tx(id, bytes);
  }
}

MUTLASS_DEVICE static void wait_warp(int id, int phase) {
  if ((int(threadIdx.x) & 31) == 0) {
    named_barrier_wait(id, uint32_t(phase) & 1u);
  }
  __syncwarp();
}

// A squad-local rendezvous, shared by both consumer roles: every one of the
// squad's four warps arrives on its own `RecurrenceBarriers::Local + squad`
// barrier.  `local` is the role's own id, so neither role carries a private copy
// of this.
MUTLASS_DEVICE static void sync_warp_squad(int local) {
  __syncwarp();
  if ((int(threadIdx.x) & 31) == 0) {
    named_barrier_arrive_and_wait(local);
  }
  __syncwarp();
}

// ---- a staged input ring ---------------------------------------------------
//
// A ring owns a full and an empty barrier of TmeStages ids each.  The stage index
// is ADDED to the base rather than looked up in a table: an array initialised with
// the base alone leaves every stage above 0 aliased to stage 0 for any ring deeper
// than one, which is a silently wrong barrier rather than a build error.
MUTLASS_DEVICE constexpr int ring_stage(int sequence, int tme_stages) {
  return sequence % tme_stages;
}

// The phase flips once per lap.  A producer's `empty` wait is the complement,
// because on the first lap the barrier is fresh and already sits at the phase the
// producer must not wait for -- the same convention as mutlass'
// make_producer_start_state.
MUTLASS_DEVICE constexpr int ring_phase(int sequence, int tme_stages) {
  return (sequence / tme_stages) & 1;
}

template <int TmeStages, int FullBase, int EmptyBase>
struct NamedBarrierRing {
  static_assert(TmeStages >= 1, "a ring is at least one stage deep");
  static_assert(FullBase >= 0 && EmptyBase >= 0, "a ring's bases come from RecurrenceBarriers");

  static MUTLASS_DEVICE constexpr int stage(int sequence) {
    return ring_stage(sequence, TmeStages);
  }
  static MUTLASS_DEVICE constexpr int phase(int sequence) {
    return ring_phase(sequence, TmeStages);
  }
  static MUTLASS_DEVICE constexpr int full(int sequence) {
    return FullBase + stage(sequence);
  }
  static MUTLASS_DEVICE constexpr int empty(int sequence) {
    return EmptyBase + stage(sequence);
  }

  // Consumer half.
  static MUTLASS_DEVICE void wait(int sequence) {
    wait_warp(full(sequence), phase(sequence));
  }
  static MUTLASS_DEVICE void release(int sequence) {
    arrive_warp(empty(sequence));
  }
  // Producer half.
  static MUTLASS_DEVICE void wait_empty(int sequence) {
    wait_warp(empty(sequence), phase(sequence) ^ 1);
  }
  static MUTLASS_DEVICE void publish(int sequence) {
    arrive_warp(full(sequence));
  }
};

}  // namespace mate::flat::kda
