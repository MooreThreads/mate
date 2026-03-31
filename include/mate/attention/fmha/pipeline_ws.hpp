/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "mutlass/arch/barrier.hpp"
#include "mutlass/detail/dependent_false.hpp"
#include "mutlass/mutlass.h"
#include "mutlass/pipeline/mp31_pipeline.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Mp31 TME load (producer) Async Pipeline class FIX
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <int Stages_>
class Mp31PipelineTmeAsyncWarpsepcialized {
 public:
  using FullBarrier                = mutlass::arch::AsyncTransactionBarrier;
  using EmptyBarrier               = mutlass::arch::AsyncBarrier;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState              = mutlass::PipelineState<Stages>;
  static_assert(FullBarrier::ReservedAsyncBarrierCount == EmptyBarrier::ReservedAsyncBarrierCount &&
                FullBarrier::ReservedAsyncBarrierCount == 1);

  static constexpr uint32_t NumBarriers = 2 * Stages;

  struct Params {
    uint32_t transaction_bytes = 0;
    uint32_t num_consumers     = 0;
    uint32_t num_producers     = 1;
  };

  // Constructor
  MUTLASS_DEVICE
  Mp31PipelineTmeAsyncWarpsepcialized(Params params, uint32_t barrier_base = 0, uint32_t init_warps = 1)
      : params_(params), barrier_base_(barrier_base + FullBarrier::ReservedAsyncBarrierCount) {
    int warp_idx = canonical_warp_idx();

    if (warp_idx < init_warps) {
      // Init full barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        FullBarrier::init(barrier_base_ + i, params_.num_producers, 0);
      }

      // Init empty barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        EmptyBarrier::init(barrier_base_ + i + Stages, params_.num_consumers + params_.num_producers, 0);
      }
    }
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  MUTLASS_DEVICE
  void producer_acquire(PipelineState state) {
    producer_acquire(state.index(), state.phase());
  }

  MUTLASS_DEVICE
  void producer_expect_transaction(PipelineState state, uint32_t transaction_bytes) {
    producer_expect_transaction(state.index(), transaction_bytes);
  }

  MUTLASS_DEVICE
  uint32_t producer_get_barrier_id(PipelineState state) {
    return producer_get_barrier_id(state.index());
  }

  ////////////////////
  // Consumers APIs
  ////////////////////
  MUTLASS_DEVICE
  void consumer_wait(PipelineState state) {
    consumer_wait(state.index(), state.phase());
  }

  MUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

 private:
  Params   params_;
  uint32_t barrier_base_;

  MUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase) {
    uint32_t empty_barrier_id = barrier_base_ + stage + Stages;
    EmptyBarrier::arrive(empty_barrier_id);
    EmptyBarrier::wait(empty_barrier_id, phase);

    uint32_t full_barrier_id = barrier_base_ + stage;
    FullBarrier::arrive_and_expect_tx(full_barrier_id, params_.transaction_bytes);
  }

  MUTLASS_DEVICE
  void producer_expect_transaction(uint32_t stage, uint32_t transaction_bytes) {
    uint32_t full_barrier_id = barrier_base_ + stage;
    FullBarrier::expect_transaction(full_barrier_id, transaction_bytes);
  }

  MUTLASS_DEVICE
  uint32_t producer_get_barrier_id(uint32_t stage) {
    return barrier_base_ + stage;
  }

  MUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    uint32_t full_barrier_id = barrier_base_ + stage;
    FullBarrier::wait(full_barrier_id, phase);
  }

  MUTLASS_DEVICE
  void consumer_release(uint32_t stage) {
    uint32_t empty_barrier_id = barrier_base_ + stage + Stages;
    EmptyBarrier::arrive(empty_barrier_id);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Simple producer-consumer async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages_>
class Mp31PipelineAsyncWarpsepcialized {
 public:
  using FullBarrier                = mutlass::arch::AsyncBarrier;
  using EmptyBarrier               = mutlass::arch::AsyncBarrier;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState              = mutlass::PipelineState<Stages>;
  static_assert(FullBarrier::ReservedAsyncBarrierCount == EmptyBarrier::ReservedAsyncBarrierCount &&
                FullBarrier::ReservedAsyncBarrierCount == 1);

  static constexpr uint32_t NumBarriers = 2 * Stages;

  struct Params {
    uint32_t producer_arv_count = 1;
    uint32_t consumer_arv_count = 1;
  };

  // Constructor
  MUTLASS_DEVICE
  Mp31PipelineAsyncWarpsepcialized(Params params, uint32_t barrier_base = 0)
      : params_(params), barrier_base_(barrier_base + FullBarrier::ReservedAsyncBarrierCount) {
    int warp_idx = canonical_warp_idx();

    if (warp_idx == 0) {
      // Init full barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        FullBarrier::init(barrier_base_ + i, params_.producer_arv_count, 0);
      }

      // Init empty barriers
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Stages; ++i) {
        EmptyBarrier::init(barrier_base_ + i + Stages, params_.consumer_arv_count + params_.producer_arv_count, 0);
      }
    }
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  MUTLASS_DEVICE
  void producer_acquire(PipelineState state) {
    producer_acquire(state.index(), state.phase());
  }

  MUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    producer_commit(state.index());
  }

  MUTLASS_DEVICE
  uint32_t producer_get_barrier_id(PipelineState state) {
    return producer_get_barrier_id(state.index());
  }

  ////////////////////
  // Consumers APIs
  ////////////////////
  MUTLASS_DEVICE
  void consumer_wait(PipelineState state) {
    consumer_wait(state.index(), state.phase());
  }

  MUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

 private:
  Params   params_;
  uint32_t barrier_base_;

  MUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase) {
    uint32_t empty_barrier_id = barrier_base_ + stage + Stages;
    EmptyBarrier::arrive(empty_barrier_id);
    EmptyBarrier::wait(empty_barrier_id, phase);
  }

  MUTLASS_DEVICE
  void producer_commit(uint32_t stage) {
    uint32_t full_barrier_id = barrier_base_ + stage;
    FullBarrier::arrive(full_barrier_id);
  }

  MUTLASS_DEVICE
  uint32_t producer_get_barrier_id(uint32_t stage) {
    return barrier_base_ + stage;
  }

  MUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    uint32_t full_barrier_id = barrier_base_ + stage;
    FullBarrier::wait(full_barrier_id, phase);
  }

  MUTLASS_DEVICE
  void consumer_release(uint32_t stage) {
    uint32_t empty_barrier_id = barrier_base_ + stage + Stages;
    EmptyBarrier::arrive(empty_barrier_id);
  }
};

}  // end namespace mutlass
