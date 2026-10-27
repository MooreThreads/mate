#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>
#include <type_traits>

namespace mate::flat {

template <auto TagValue, typename Default, typename... Options>
struct find_option;

template <auto TagValue, typename Default>
struct find_option<TagValue, Default> {
  using option_value = Default;
};

template <auto TagValue, typename Default, typename Option, typename... Options>
struct find_option<TagValue, Default, Option, Options...>
    : std::conditional_t<Option::tag == TagValue, Option, find_option<TagValue, Default, Options...>> {};

template <auto TagValue, typename Default, typename... Options>
using find_option_t = typename find_option<TagValue, Default, Options...>::option_value;

template <auto TagValue, class Value>
struct Option {
  static constexpr auto tag = TagValue;
  using option_value        = Value;
};

enum class Tag {
  HasStateIn,
  HasStateOut,
  HasGateParams,
  IsVarlen,
  NormalizeQK,
  // Emit the recurrent state at every chunk boundary into a caller-allocated
  // pool.  A serving framework uses the snapshot to cache prefix states and
  // skip recomputing a shared prefix on a later request.
  HasChunkStatesOut,
  // The pool above uses the compact per-sequence checkpoint row layout instead
  // of its packed chunk-slot rows.  This is a mode the host selects; the kernel
  // never derives it from the cadence value, because a non-positive cadence is a
  // host-side error that must never reach the renderer.
  HasCheckpointCadence,
  // Cadence of that layout in tokens: a positive multiple of the kernel chunk.
  // Rendered only as the divisor of the row arithmetic; the host validates it
  // before rendering (see `mate.jit.kda_ops`).
  CheckpointEveryNTokens,
};

}  // namespace mate::flat
