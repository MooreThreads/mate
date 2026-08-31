#pragma once

#include <type_traits>

namespace mate::attention::msa {

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
  HasMetadata,
  HasSeqUsedK,
  HasCuseqlensQ,
  IsVarlen,
  IsPagedKV,
  IsSparseKV,
  IsCausal,
  IsPackGQA,
  IsSplit,
  UseLSULoadK,
  UseLSULoadV,
  PackQueryPair,
  PackQuery16,
  ParallelKTiles,
  SparseTopK,
  SparseQueryBlockSize,
  UseBlockQKScale,
  UseQueryQKScale,
  ElementPV,
  ElementOutput,
};

template <class Value>
using HasMetadata = Option<Tag::HasMetadata, Value>;

template <class Value>
using HasSeqUsedK = Option<Tag::HasSeqUsedK, Value>;

template <class Value>
using HasCuseqlensQ = Option<Tag::HasCuseqlensQ, Value>;

template <class Value>
using IsVarlen = Option<Tag::IsVarlen, Value>;

template <class Value>
using IsPagedKV = Option<Tag::IsPagedKV, Value>;

template <class Value>
using IsSparseKV = Option<Tag::IsSparseKV, Value>;

template <class Value>
using IsCausal = Option<Tag::IsCausal, Value>;

template <class Value>
using IsPackGQA = Option<Tag::IsPackGQA, Value>;

template <class Value>
using IsSplit = Option<Tag::IsSplit, Value>;

template <class Value>
using UseLSULoadK = Option<Tag::UseLSULoadK, Value>;

template <class Value>
using UseLSULoadV = Option<Tag::UseLSULoadV, Value>;

template <class Value>
using PackQueryPair = Option<Tag::PackQueryPair, Value>;

template <class Value>
using PackQuery16 = Option<Tag::PackQuery16, Value>;

template <class Value>
using ParallelKTiles = Option<Tag::ParallelKTiles, Value>;

template <class Value>
using SparseTopK = Option<Tag::SparseTopK, Value>;

template <class Value>
using SparseQueryBlockSize = Option<Tag::SparseQueryBlockSize, Value>;

template <class Value>
using UseBlockQKScale = Option<Tag::UseBlockQKScale, Value>;

template <class Value>
using UseQueryQKScale = Option<Tag::UseQueryQKScale, Value>;

template <class Value>
using ElementPV = Option<Tag::ElementPV, Value>;

template <class Value>
using ElementOutput = Option<Tag::ElementOutput, Value>;

}  // namespace mate::attention::msa
