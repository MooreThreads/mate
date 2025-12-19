#pragma once

#include "mutlass/bfloat16.h"
#include "mutlass/half.h"

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

struct TileConfig {
  int kCTA_Q;
  int kCTA_K;
  int kHeadSizeQK;
  int kHeadSizeV;
};

template <int HEADDIM_QK, int HEADDIM_V, bool PAGED_KV>
constexpr TileConfig get_tile_config() {
  if constexpr (PAGED_KV) {
    if constexpr (HEADDIM_QK == 128 && HEADDIM_V == 128) {
      return {256, 64, 128, 128};
    }
  } else {
    if constexpr (HEADDIM_QK == 128 && HEADDIM_V == 128) {
      return {256, 128, 128, 128};
    } else if constexpr (HEADDIM_QK == 192 && HEADDIM_V == 128) {
      return {256, 64, 192, 128};
    }
  }
  return {0, 0, 0, 0};
}

#define HEADDIM_SWITCH(HEADDIM_QK, HEADDIM_V, ...)      \
  [&] {                                                 \
    if (HEADDIM_QK == 128 && HEADDIM_V == 128) {        \
      constexpr static int kHeadSizeQK = 128;           \
      constexpr static int kHeadSizeV  = 128;           \
      return __VA_ARGS__();                             \
    } else if (HEADDIM_QK == 192 && HEADDIM_V == 128) { \
      constexpr static int kHeadSizeQK = 192;           \
      constexpr static int kHeadSizeV  = 128;           \
      return __VA_ARGS__();                             \
    }                                                   \
  }()

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = mutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = mutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()
