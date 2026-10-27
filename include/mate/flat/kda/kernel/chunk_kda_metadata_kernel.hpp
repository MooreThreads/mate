#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstddef>
#include <cstdint>
#include <cub/block/block_scan.cuh>

#include "mate/flat/kda/chunk_kda_metadata.hpp"

namespace mate::flat::kda {

namespace detail {

template <int Threads>
using MetadataBlockScan = cub::BlockScan<int32_t, Threads>;

MUTLASS_HOST_DEVICE constexpr std::size_t metadata_scan_offset(int num_seqs) {
  constexpr std::size_t alignment    = 16;
  const std::size_t     prefix_bytes = std::size_t(num_seqs + 1) * sizeof(int32_t);
  return (prefix_bytes + alignment - 1) & ~(alignment - 1);
}

template <int Threads>
constexpr std::size_t metadata_shared_bytes(int num_seqs) {
  return metadata_scan_offset(num_seqs) + sizeof(typename MetadataBlockScan<Threads>::TempStorage);
}

template <int Chunk>
__device__ __forceinline__ int32_t metadata_chunk_count(int32_t length) {
  if constexpr (Chunk == 16) {
    return (length + 15) >> 4;
  } else {
    return (length + Chunk - 1) / Chunk;
  }
}

struct MetadataPrefixCallback {
  int32_t running;

  __device__ __forceinline__ int32_t operator()(int32_t aggregate) {
    int32_t old = running;
    running += aggregate;
    return old;
  }
};

}  // namespace detail

// Fallback for unusually large sequence-count arrays that do not fit in the
// metadata CTA's dynamic shared-memory prefix buffer.  All scheduler quantities
// and metadata fields are int32, so this path avoids MP31's software 64-bit
// divide implementation as well.
template <int Chunk, class CuSeqlensElement>
__global__ void chunk_kda_build_partition_metadata_kernel_serial(ChunkKdaPartitionMetadata* __restrict__ metadata,
                                                                 CuSeqlensElement const* __restrict__ cu_seqlens,
                                                                 int num_seqs,
                                                                 int num_heads,
                                                                 int num_partitions) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  using Index              = int32_t;
  const Index heads        = static_cast<Index>(num_heads);
  Index       total_chunks = 0;
  for (int seq = 0; seq < num_seqs; ++seq) {
    Index bos = static_cast<Index>(cu_seqlens[seq]);
    Index eos = static_cast<Index>(cu_seqlens[seq + 1]);
    total_chunks += detail::metadata_chunk_count<Chunk>(eos - bos);
  }
  Index total_works = total_chunks * heads;
  Index base_count  = total_works / static_cast<Index>(num_partitions);
  Index remainder   = total_works - base_count * static_cast<Index>(num_partitions);

  // Partition starts are monotonic.  Keep the sequence cursor and prefix work
  // count across partitions so this fallback remains O(N + P).
  int   seq        = 0;
  Index seq_prefix = 0;
  Index seq_works  = 0;
  if (num_seqs > 0) {
    Index bos = static_cast<Index>(cu_seqlens[0]);
    Index eos = static_cast<Index>(cu_seqlens[1]);
    seq_works = detail::metadata_chunk_count<Chunk>(eos - bos) * heads;
  }
  for (int partition = 0; partition < num_partitions; ++partition) {
    Index begin =
        static_cast<Index>(partition) * base_count + static_cast<Index>(partition < remainder ? partition : remainder);
    Index count = base_count + static_cast<Index>(partition < remainder);
    while (seq < num_seqs) {
      if (begin < seq_prefix + seq_works) {
        break;
      }
      seq_prefix += seq_works;
      ++seq;
      if (seq < num_seqs) {
        Index bos = static_cast<Index>(cu_seqlens[seq]);
        Index eos = static_cast<Index>(cu_seqlens[seq + 1]);
        seq_works = detail::metadata_chunk_count<Chunk>(eos - bos) * heads;
      }
    }
    begin -= seq_prefix;
    int chunk           = count > 0 ? int(begin / heads) : 0;
    int head            = count > 0 ? int(begin - static_cast<Index>(chunk) * heads) : 0;
    metadata[partition] = ChunkKdaPartitionMetadata{seq, chunk, head, count};
  }
}

// Build an exclusive sequence-work prefix in shared memory, then let one
// thread handle each partition.  The first phase is a tiled CUB BlockScan;
// the second phase binary-searches the prefix, so partition construction is
// O(N / 512 + P log N) rather than a lane-0 O(N + P) walk.
template <int Chunk, class CuSeqlensElement, int Threads>
__global__ __launch_bounds__(Threads) void chunk_kda_build_partition_metadata_kernel(
    ChunkKdaPartitionMetadata* __restrict__ metadata,
    CuSeqlensElement const* __restrict__ cu_seqlens,
    int num_seqs,
    int num_heads,
    int num_partitions) {
  using Scan  = detail::MetadataBlockScan<Threads>;
  using Index = int32_t;
  extern __shared__ unsigned char shared_raw[];
  Index*                          prefix = reinterpret_cast<Index*>(shared_raw);
  auto*                           scan_storage =
      reinterpret_cast<typename Scan::TempStorage*>(shared_raw + detail::metadata_scan_offset(num_seqs));

  int                            tid = int(threadIdx.x);
  detail::MetadataPrefixCallback prefix_op{0};
  for (int base = 0; base < num_seqs; base += Threads) {
    int   seq   = base + tid;
    Index value = 0;
    if (seq < num_seqs) {
      Index bos = static_cast<Index>(cu_seqlens[seq]);
      Index eos = static_cast<Index>(cu_seqlens[seq + 1]);
      value     = detail::metadata_chunk_count<Chunk>(eos - bos);
    }
    Index local_prefix = 0;
    Scan(*scan_storage).ExclusiveSum(value, local_prefix, prefix_op);
    if (seq < num_seqs) {
      prefix[seq] = local_prefix;
    }
    __syncthreads();
  }

  if (tid == 0) {
    prefix[num_seqs] = prefix_op.running;
  }
  __syncthreads();

  const Index heads       = static_cast<Index>(num_heads);
  Index       total_works = prefix[num_seqs] * heads;
  Index       base_count  = total_works / static_cast<Index>(num_partitions);
  Index       remainder   = total_works - base_count * static_cast<Index>(num_partitions);
  for (int partition = tid; partition < num_partitions; partition += Threads) {
    Index begin =
        static_cast<Index>(partition) * base_count + static_cast<Index>(partition < remainder ? partition : remainder);
    Index count = base_count + static_cast<Index>(partition < remainder);

    // Strict comparison intentionally skips zero-length sequences and matches
    // the old serial builder's upper-bound behavior.
    int lo = 0;
    int hi = num_seqs;
    while (lo < hi) {
      int   mid     = lo + ((hi - lo) >> 1);
      Index seq_end = prefix[mid + 1] * heads;
      if (begin < seq_end) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    int   seq           = lo;
    Index local         = begin - prefix[seq] * heads;
    int   chunk         = count > 0 ? int(local / heads) : 0;
    int   head          = count > 0 ? int(local - static_cast<Index>(chunk) * heads) : 0;
    metadata[partition] = ChunkKdaPartitionMetadata{seq, chunk, head, count};
  }
}

template <int Chunk, class CuSeqlensElement, class SchedulePolicy>
inline musaError_t launch_chunk_kda_metadata(ChunkKdaPartitionMetadata* metadata,
                                             CuSeqlensElement const*    cu_seqlens,
                                             int                        num_seqs,
                                             int                        num_heads,
                                             int                        num_partitions,
                                             musaStream_t               stream) {
  constexpr int     threads      = SchedulePolicy::MetadataThreads;
  const std::size_t shared_bytes = detail::metadata_shared_bytes<threads>(num_seqs);
  if (shared_bytes <= SchedulePolicy::MetadataSharedLimit) {
    chunk_kda_build_partition_metadata_kernel<Chunk, CuSeqlensElement, threads>
        <<<1, threads, shared_bytes, stream>>>(metadata, cu_seqlens, num_seqs, num_heads, num_partitions);
    return musaGetLastError();
  } else {
    chunk_kda_build_partition_metadata_kernel_serial<Chunk>
        <<<1, 32, 0, stream>>>(metadata, cu_seqlens, num_seqs, num_heads, num_partitions);
    return musaGetLastError();
  }
}

}  // namespace mate::flat::kda
