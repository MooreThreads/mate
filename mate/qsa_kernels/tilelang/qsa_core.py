"""QSA core attention: shared indexed math and regular/WS prefill pipelines."""

from functools import lru_cache
from typing import Any

from . import _pass_configs

from ...utils import cosize

try:
    import tilelang
    import tilelang.language as T
except ImportError:
    tilelang = None
    T = None


def _make_qsa_copy_vector_macro(page_size=0, robust=False):
    """Copy one lane's eight BF16/FP16 values with optional source bounds."""

    @T.macro
    def copy_vector(src, dst, physical_token, head, token_offset, dim, descriptor):
        # SW-91554: invalid ids are clamped to 0 by the producer, so these
        # bounds hold for every lane. Tell the compiler instead of emitting a
        # per-token branch: MTCC 5.3 requires each cross-BB commit to stay in
        # the same basic block as its async-copy producers.
        capacity = src.shape[0] * page_size if page_size else src.shape[0]
        T.assume(physical_token >= 0)
        T.assume(physical_token < capacity)
        # TODO: Use native T.copy once TileLang preserves async16 vectorization
        # for dynamic-stride/swizzled gathers, including robust source bounds.
        T.call_extern(
            "void",
            "tl::cp_async_gs_robust<16>" if robust else "tl::cp_async_gs<16>",
            T.address_of(dst[token_offset, dim]),
            T.address_of(
                src[physical_token // page_size, physical_token % page_size, head, dim]
            )
            if page_size
            else T.address_of(src[physical_token, head, dim]),
            *([descriptor] if robust else []),
        )

    return copy_vector


def _make_qsa_load_v_macro(
    block_n,
    structured_microblocks,
    predecode_physical_ids,
    direct_async_gather,
    page_size=0,
    stripe_major=False,
):
    """Share V gather addressing across regular and warp-specialized prefill."""
    copy_vector = _make_qsa_copy_vector_macro(page_size, not direct_async_gather)

    @T.macro
    def load_lane(v, dst, physical_local, head, tx, ty, descriptor, capacity, r, u):
        if structured_microblocks:
            token_offset = ty * 4 + r
        else:
            token_offset = ((r * 16 + ty) % 8) * (block_n // 8) + (r * 16 + ty) // 8
        physical = (
            physical_local[r]
            if predecode_physical_ids
            else T.if_then_else(physical_local[r] >= 0, physical_local[r], 0)
        )
        if direct_async_gather:
            T.assume((physical >= 0) & (physical < capacity))
        copy_vector(v, dst, physical, head, token_offset, u * 64 + tx * 8, descriptor)

    @T.macro
    def load_v(v, dst, physical_local, head, tx, ty, descriptor, capacity):
        if stripe_major:
            for u in T.unroll(4, explicit=True):
                for r in T.unroll(4, explicit=True):
                    load_lane(
                        v, dst, physical_local, head, tx, ty, descriptor, capacity, r, u
                    )
        else:
            for r in T.unroll(4, explicit=True):
                for u in T.unroll(4):
                    load_lane(
                        v, dst, physical_local, head, tx, ty, descriptor, capacity, r, u
                    )

    return load_v


def _make_qsa_load_k_macro(
    block_n, structured, predecode, direct, page_size=0, assume_full_blocks=False
):
    """Decode selected ids once and share the producer's K addressing."""
    copy_vector = _make_qsa_copy_vector_macro(page_size, not direct)

    @T.macro
    def load_k(
        dst,
        mask,
        tile,
        row,
        head,
        ty,
        tx,
        physical,
        base,
        selected,
        k,
        robust,
        descriptor,
    ):
        capacity = k.shape[0] * page_size if page_size else k.shape[0]
        if structured:
            full = tile < 2051 // block_n if assume_full_blocks else False
            low = T.if_then_else(
                full, 3, T.if_then_else(descriptor >= 0, descriptor & 3, 0)
            )
            block_valid = T.if_then_else(full, True, descriptor >= 0)
            base[0] = descriptor - low
        for r in T.unroll(4, explicit=True):
            if structured:
                token_offset = ty * 4 + r
                valid = block_valid & (r <= low)
                token = base[0] + r
            else:
                token_offset = ((r * 16 + ty) % 8) * (block_n // 8) + (r * 16 + ty) // 8
                slot = tile * block_n + token_offset
                token = T.if_then_else(
                    slot < selected.shape[1],
                    selected[row, T.min(slot, selected.shape[1] - 1)],
                    -1,
                )
                valid = (token >= 0) & (token < capacity)
            physical[r] = T.if_then_else(valid, token, 0 if predecode else -1)
            mask[token_offset] = T.if_then_else(valid, token, -1)
            safe = physical[r] if predecode else T.max(physical[r], 0)
            if direct:
                T.assume((safe >= 0) & (safe < capacity))
            for u in T.unroll(4):
                copy_vector(k, dst, safe, head, token_offset, u * 64 + tx * 8, robust)

    return load_k


def _make_qsa_zero_v_macro(block_n, structured):
    @T.macro
    def zero_invalid_v(dst, physical, ty, tx):
        # Same lanes as the copy; call only after its completion wait.
        for r in T.unroll(4, explicit=True):
            offset = (
                ty * 4 + r
                if structured
                else ((r * 16 + ty) % 8) * (block_n // 8) + (r * 16 + ty) // 8
            )
            if physical[offset] < 0:
                for u in T.unroll(4):
                    for vec in T.vectorized(8):
                        dst[offset, u * 64 + tx * 8 + vec] = 0

    return zero_invalid_v


def _make_qsa_attention_macro(
    block_n,
    block_h,
    group,
    dtype,
    scale,
    *,
    mode,
    num_splits=1,
    annotate_v_layout=False,
    flat_physical_metadata=False,
    compact_partial_output=False,
    direct_async_gather=False,
    page_size=0,
    return_lse=False,
):
    """One indexed QK/softmax/PV loop for contiguous, paged and SplitKV ABIs."""
    paged, split = mode != "indexed", mode == "splitkv"
    output_dtype = "float32" if split and not compact_partial_output else dtype
    copy_layout = T.Fragment((block_n, 32), lambda n, d: ((n % 4) * 32 + d, n // 4))
    copy_vector = _make_qsa_copy_vector_macro(page_size=page_size)

    @T.macro
    def attention(
        q,
        k,
        v,
        selected,
        pages,
        offsets,
        requests,
        positions,
        lengths,
        output,
        lse,
        row,
        qid,
        head,
        split_id,
    ):
        block_tokens = selected.shape[-1]
        num_tiles = T.ceildiv(block_tokens, block_n)
        tiles_per_split = T.ceildiv(num_tiles, num_splits)
        T.assume(k.shape[0] <= 1 or k.strides[0] % 8 == 0)
        T.assume(k.shape[1] <= 1 or k.strides[1] % 8 == 0)
        T.assume(k.shape[2] <= 1 or k.strides[2] % 8 == 0)
        T.assume(v.shape[0] <= 1 or v.strides[0] % 8 == 0)
        T.assume(v.shape[1] <= 1 or v.strides[1] % 8 == 0)
        T.assume(v.shape[2] <= 1 or v.strides[2] % 8 == 0)
        q_shared = T.alloc_shared((block_h, 256), dtype)
        k_shared = T.alloc_shared((block_n, 256), dtype)
        v_shared = T.alloc_shared((block_n, 256), dtype)
        p_shared = T.alloc_shared((block_h, block_n), dtype)
        physical = T.alloc_shared((block_n,), T.int32)
        output_shared = T.alloc_shared((block_h, 256), output_dtype)
        scores = T.alloc_fragment((block_h, block_n), T.float32)
        acc = T.alloc_fragment((block_h, 256), T.float32)
        maximum = T.alloc_fragment((block_h,), T.float32)
        previous = T.alloc_fragment((block_h,), T.float32)
        total = T.alloc_fragment((block_h,), T.float32)
        tile_sum = T.alloc_fragment((block_h,), T.float32)
        rescale = T.alloc_shared((block_h,), T.float32)
        normalizer = T.alloc_shared((block_h,), T.float32)
        maximum_shared = T.alloc_shared((block_h,), T.float32)
        if annotate_v_layout:
            T.annotate_layout(
                {
                    v_shared: tilelang.layout.make_sqmma_swizzled_layout(
                        v_shared, k_major=False
                    )
                },
                allow_reannotation=True,
            )
        T.clear(q_shared)
        if (
            paged
            and q.strides[2] == 1
            and q.strides[1] == 256
            and T.reinterpret(T.address_of(q[row, head * group, 0]), T.uint64) % 16 == 0
        ):
            dense_q = T.make_tensor_from_addr(
                T.address_of(q[row, head * group, 0]), (group, 256), dtype
            )
            T.copy(dense_q, q_shared[0:group, 0:256])
        else:
            T.copy(
                q[row, head * group : (head + 1) * group, 0:256]
                if paged
                else q[row, qid, head * group : (head + 1) * group, 0:256],
                q_shared[0:group, 0:256],
            )
        T.sync_threads()
        if paged:
            if not flat_physical_metadata:
                request = requests[row]
                position = positions[row]
                length = T.alloc_var(T.int32, init=0)
                if request >= 0 and request < lengths.shape[0]:
                    length = lengths[request]
        else:
            position = positions[row, qid]
            length = lengths[row]
        T.clear(acc)
        T.clear(total)
        # Finite FP32 minimum handles empty tiles without eclipsing valid logits.
        T.fill(maximum, -3.4028234663852886e38)
        for tile in T.Pipelined(tiles_per_split, num_stages=1):
            start = (split_id * tiles_per_split + tile) * block_n
            for n in T.Parallel(block_n):
                column = T.min(start + n, block_tokens - 1)
                if paged and flat_physical_metadata:
                    token = pages[row, column]
                    valid = (
                        start + n < block_tokens
                        and token >= 0
                        and T.int64(token) < T.int64(k.shape[0]) * page_size
                    )
                elif paged:
                    token_id = selected[row, column]
                    page = pages[row, column]
                    offset = offsets[row, column]
                    valid = (
                        start + n < block_tokens
                        and token_id >= 0
                        and token_id < length
                        and token_id <= position
                        and page >= 0
                        and page < k.shape[0]
                        and offset >= 0
                        and offset < page_size
                    )
                    token = T.int64(page) * page_size + offset
                else:
                    token = selected[row, qid, column]
                    valid = (
                        start + n < block_tokens
                        and token >= 0
                        and token < length
                        and token < k.shape[1]
                        and token <= position
                    )
                physical[n] = T.if_then_else(valid, token, -1)
            T.sync_threads()
            if direct_async_gather:
                for n, d in T.Parallel(block_n, 32, loop_layout=copy_layout):
                    safe = T.max(physical[n], 0)
                    T.assume(safe < T.int64(k.shape[0]) * page_size)
                    copy_vector(k, k_shared, safe, head, n, d * 8, 0)
                T.async_copy_commit_group(0)
                T.async_copy_wait_group(0)
                T.lma_wait()
                for n, d in T.Parallel(block_n, 32, loop_layout=copy_layout):
                    safe = T.max(physical[n], 0)
                    T.assume(safe < T.int64(v.shape[0]) * page_size)
                    copy_vector(v, v_shared, safe, head, n, d * 8, 0)
                T.async_copy_commit_group(1)
                T.async_copy_wait_group(1)
                T.lma_wait()
                # Keep invalid-V zeroing on the lanes that issued each copy.
                for n, d in T.Parallel(block_n, 32, loop_layout=copy_layout):
                    if physical[n] < 0:
                        for lane in T.vectorized(8):
                            v_shared[n, d * 8 + lane] = 0
            else:
                for n, d in T.Parallel(block_n, 256):
                    safe = T.max(physical[n], 0)
                    if paged:
                        k_shared[n, d] = T.if_then_else(
                            physical[n] >= 0,
                            k[safe // page_size, safe % page_size, head, d],
                            0,
                        )
                        v_shared[n, d] = T.if_then_else(
                            physical[n] >= 0,
                            v[safe // page_size, safe % page_size, head, d],
                            0,
                        )
                    else:
                        k_shared[n, d] = T.if_then_else(
                            physical[n] >= 0, k[row, safe, head, d], 0
                        )
                        v_shared[n, d] = T.if_then_else(
                            physical[n] >= 0, v[row, safe, head, d], 0
                        )
            T.sync_threads()
            T.gemm(
                q_shared,
                k_shared,
                scores,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, n in T.Parallel(block_h, block_n):
                scores[i, n] = T.if_then_else(
                    i < group and physical[n] >= 0, scores[i, n], -T.infinity(T.float32)
                )
            T.copy(maximum, previous)
            T.fill(maximum, -3.4028234663852886e38)
            T.reduce_max(scores, maximum, dim=1, clear=False)
            for i in T.Parallel(block_h):
                maximum[i] = T.max(maximum[i], previous[i])
                rescale[i] = T.exp2((previous[i] - maximum[i]) * scale)
            T.sync_threads()
            for i, n in T.Parallel(block_h, block_n):
                scores[i, n] = T.exp2((scores[i, n] - maximum[i]) * scale)
            T.reduce_sum(scores, tile_sum, dim=1, clear=True)
            for i in T.Parallel(block_h):
                total[i] = total[i] * rescale[i] + tile_sum[i]
            for i, d in T.Parallel(block_h, 256):
                acc[i, d] *= rescale[i]
            T.copy(scores, p_shared)
            T.sync_threads()
            T.gemm(p_shared, v_shared, acc, policy=T.GemmWarpPolicy.FullRow)
            for residual in T.serial(2):
                for i, n in T.Parallel(block_h, block_n):
                    scores[i, n] -= T.cast(T.cast(scores[i, n], dtype), T.float32)
                    p_shared[i, n] = T.cast(scores[i, n], dtype)
                T.sync_threads()
                T.gemm(p_shared, v_shared, acc, policy=T.GemmWarpPolicy.FullRow)
            T.sync_threads()
        for i in T.Parallel(block_h):
            normalizer[i] = total[i]
            maximum_shared[i] = maximum[i]
        T.sync_threads()
        for i, d in T.Parallel(block_h, 256):
            acc[i, d] = T.if_then_else(
                i < group and normalizer[i] > 0,
                acc[i, d] / T.max(normalizer[i], 1e-20),
                0,
            )
        T.copy(acc, output_shared)
        T.sync_threads()
        if split:
            for i in T.Parallel(block_h):
                if i < group:
                    lse[split_id, row, head * group + i] = T.if_then_else(
                        normalizer[i] > 0,
                        T.log2(normalizer[i]) + maximum_shared[i] * scale,
                        -T.infinity(T.float32),
                    )
            if (
                output.strides[3] == 1
                and output.strides[2] == 256
                and T.reinterpret(
                    T.address_of(output[split_id, row, head * group, 0]), T.uint64
                )
                % 16
                == 0
            ):
                dense_output = T.make_tensor_from_addr(
                    T.address_of(output[split_id, row, head * group, 0]),
                    (group, 256),
                    output_dtype,
                )
                T.copy(output_shared[0:group, 0:256], dense_output)
            else:
                for i, d in T.Parallel(block_h, 256):
                    if i < group:
                        output[split_id, row, head * group + i, d] = output_shared[i, d]
        else:
            if return_lse:
                for i in T.Parallel(block_h):
                    if i < group:
                        lse[row, qid, head * group + i] = T.if_then_else(
                            normalizer[i] > 0,
                            (T.log2(normalizer[i]) + maximum_shared[i] * scale)
                            * 0.6931471805599453,
                            -T.infinity(T.float32),
                        )
            T.copy(
                output_shared[0:group, 0:256],
                output[row, head * group : (head + 1) * group, 0:256]
                if paged
                else output[row, qid, head * group : (head + 1) * group, 0:256],
            )

    return attention


@lru_cache(maxsize=None)
def _qsa_core_kernel(
    block_n: int,
    scale: float,
    pv_block_h: int = 32,
    annotate_v_layout: bool = True,
    kv_heads: int = 2,
    gqa_group_size: int = 12,
    dtype_name: str = "bfloat16",
    *,
    page_size: int = 0,
    num_splits: int = 0,
    flat_physical_metadata: bool = False,
    compact_partial_output: bool = False,
    direct_async_gather: bool = False,
    narrow_address: bool = False,
    return_lse: bool = False,
):
    """Indexed GQA; page_size selects paged KV, splits > 0 return O/LSE partials."""
    pass_configs = _pass_configs(tilelang, disable_tma_lower=True)
    if block_n <= 0 or block_n % 8:
        raise ValueError("block_n must be positive and divisible by 8.")
    if page_size < 0 or num_splits < 0 or (num_splits and not page_size):
        raise ValueError("SplitKV requires positive page_size and num_splits.")
    block_h = pv_block_h
    if block_h not in (16, 32) or (block_h == 16 and annotate_v_layout):
        raise ValueError("PV requires M32, or M16 with annotate_v_layout=False.")
    if kv_heads <= 0 or not 0 < gqa_group_size <= block_h:
        raise ValueError("positive KV heads and a GQA group fitting block_h required.")
    if direct_async_gather and not (page_size and flat_physical_metadata):
        raise ValueError("direct_async_gather requires paged physical metadata.")

    paged, split = page_size > 0, num_splits > 0
    if return_lse and paged:
        raise ValueError("paged LSE uses the SplitKV statistics path.")
    rows, batch, query_len = T.dynamic("rows, batch, query_len")
    cache_len, num_tokens = T.dynamic("cache_len, num_tokens")
    num_pages, requests = T.dynamic("num_pages, num_requests")
    heads = kv_heads * gqa_group_size
    q_shape = (rows, heads, 256) if paged else (batch, query_len, heads, 256)
    kv_shape = (
        (num_pages, page_size, kv_heads, 256)
        if paged
        else (batch, cache_len, kv_heads, 256)
    )
    index_shape = (rows, num_tokens) if paged else (batch, query_len, num_tokens)
    output_shape = (num_splits, rows, heads, 256) if split else q_shape
    output_dtype = "float32" if split and not compact_partial_output else dtype_name
    attention = _make_qsa_attention_macro(
        block_n,
        block_h,
        gqa_group_size,
        dtype_name,
        scale,
        mode="splitkv" if split else "paged" if paged else "indexed",
        num_splits=max(num_splits, 1),
        annotate_v_layout=annotate_v_layout,
        flat_physical_metadata=flat_physical_metadata,
        compact_partial_output=compact_partial_output,
        direct_async_gather=direct_async_gather,
        page_size=page_size,
        return_lse=return_lse,
    )
    if direct_async_gather:
        pass_configs[tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY] = False
    if split:
        pass_configs.update(
            {
                tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: narrow_address,
                tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            }
        )

    @tilelang.jit(
        target="musa",
        out_idx=[-2, -1] if split or return_lse else [-1],
        pass_configs=pass_configs,
    )
    def kernel_factory():
        q_strides = tuple(
            T.dynamic(f"q_stride_{a}", "int64") for a in range(len(q_shape))
        )
        q_type: Any = T.StridedTensor(q_shape, q_strides, dtype_name)
        k_strides = T.dynamic("k_stride_0, k_stride_1, k_stride_2", "int64") + (1,)
        k_type: Any = T.StridedTensor(kv_shape, k_strides, dtype_name)
        v_strides = T.dynamic("v_stride_0, v_stride_1, v_stride_2", "int64") + (1,)
        v_type: Any = T.StridedTensor(kv_shape, v_strides, dtype_name)
        indices_strides = tuple(
            T.dynamic(f"token_indices_stride_{a}", "int64")
            for a in range(len(index_shape))
        )
        indices_type: Any = T.StridedTensor(index_shape, indices_strides, T.int32)
        positions_shape = (rows,) if paged else (batch, query_len)
        positions_strides = tuple(
            T.dynamic(f"query_positions_stride_{a}", "int64")
            for a in range(len(positions_shape))
        )
        positions_type: Any = T.StridedTensor(
            positions_shape, positions_strides, T.int32
        )
        lengths_strides = (T.dynamic("seq_lens_stride_0", "int64"),)
        lengths_type: Any = T.StridedTensor(
            (requests if paged else batch,), lengths_strides, T.int32
        )
        output_strides = tuple(
            T.dynamic(f"output_stride_{a}", "int64") for a in range(len(output_shape))
        )
        output_type: Any = T.StridedTensor(output_shape, output_strides, output_dtype)
        if paged:
            pages_strides = T.dynamic(
                "physical_pages_stride_0, physical_pages_stride_1", "int64"
            )
            pages_type: Any = T.StridedTensor(
                (rows, num_tokens), pages_strides, T.int32
            )
            offsets_strides = T.dynamic(
                "page_offsets_stride_0, page_offsets_stride_1", "int64"
            )
            offsets_type: Any = T.StridedTensor(
                (rows, num_tokens), offsets_strides, T.int32
            )
            requests_strides = (T.dynamic("token_to_req_stride_0", "int64"),)
            requests_type: Any = T.StridedTensor((rows,), requests_strides, T.int32)
        if split:
            lse_strides = T.dynamic(
                "partial_lse_stride_0, partial_lse_stride_1, partial_lse_stride_2",
                "int64",
            )
            lse_type: Any = T.StridedTensor(
                (num_splits, rows, heads), lse_strides, T.float32
            )

            @T.prim_func
            def qsa_core_paged_splitk(
                q: q_type,
                k: k_type,
                v: v_type,
                token_indices: indices_type,
                physical_pages: pages_type,
                page_offsets: offsets_type,
                token_to_req: requests_type,
                query_positions: positions_type,
                seq_lens: lengths_type,
                partial_output: output_type,
                partial_lse: lse_type,
            ):
                with T.Kernel(rows * num_splits, kv_heads, threads=128) as (
                    split_row,
                    head,
                ):
                    attention(
                        q,
                        k,
                        v,
                        token_indices,
                        physical_pages,
                        page_offsets,
                        token_to_req,
                        query_positions,
                        seq_lens,
                        partial_output,
                        partial_lse,
                        split_row % rows,
                        0,
                        head,
                        split_row // rows,
                    )

            return qsa_core_paged_splitk
        if paged:

            @T.prim_func
            def qsa_core_paged_gqa(
                q: q_type,
                k: k_type,
                v: v_type,
                token_indices: indices_type,
                physical_pages: pages_type,
                page_offsets: offsets_type,
                token_to_req: requests_type,
                query_positions: positions_type,
                seq_lens: lengths_type,
                output: output_type,
            ):
                with T.Kernel(rows, kv_heads, threads=128) as (row, head):
                    attention(
                        q,
                        k,
                        v,
                        token_indices,
                        physical_pages,
                        page_offsets,
                        token_to_req,
                        query_positions,
                        seq_lens,
                        output,
                        output,
                        row,
                        0,
                        head,
                        0,
                    )

            return qsa_core_paged_gqa

        @T.macro
        def launch_indexed(
            q, k, v, token_indices, query_positions, seq_lens, output, lse
        ):
            with T.Kernel(batch, query_len, kv_heads, threads=128) as (bid, qid, head):
                attention(
                    q,
                    k,
                    v,
                    token_indices,
                    token_indices,
                    token_indices,
                    seq_lens,
                    query_positions,
                    seq_lens,
                    output,
                    lse,
                    bid,
                    qid,
                    head,
                    0,
                )

        if return_lse:
            lse_strides = T.dynamic("lse_stride_0, lse_stride_1, lse_stride_2", "int64")
            lse_type: Any = T.StridedTensor(
                (batch, query_len, heads), lse_strides, T.float32
            )

            @T.prim_func
            def qsa_core_indexed_gqa_lse(
                q: q_type,
                k: k_type,
                v: v_type,
                token_indices: indices_type,
                query_positions: positions_type,
                seq_lens: lengths_type,
                output: output_type,
                lse: lse_type,
            ):
                launch_indexed(
                    q, k, v, token_indices, query_positions, seq_lens, output, lse
                )

            return qsa_core_indexed_gqa_lse

        @T.prim_func
        def qsa_core_indexed_gqa(
            q: q_type,
            k: k_type,
            v: v_type,
            token_indices: indices_type,
            query_positions: positions_type,
            seq_lens: lengths_type,
            output: output_type,
        ):
            launch_indexed(
                q, k, v, token_indices, query_positions, seq_lens, output, output
            )

        return qsa_core_indexed_gqa

    return kernel_factory()


@lru_cache(maxsize=None)
def _qsa_core_paged_prefill_kernel(
    block_n: int,
    scale: float,
    structured_microblocks: bool = False,
    predecode_physical_ids: bool = False,
    disable_index_type_promotion: bool = False,
    direct_async_gather: bool = False,
    use_tme_output: bool = True,
    assume_full_blocks: bool = False,
    compact_dynamic_tiles: bool = False,
    active_tile_marker: bool = False,
    prefetch_next_descriptor: bool = False,
    v_gather_stripe_major: bool = False,
    dtype_name: str = "bfloat16",
    kv_heads: int = 2,
    gqa_group_size: int = 12,
    warp_specialized: bool = False,
    kv_page_size: int = 0,
):
    """Regular/WS prefill; structured descriptors encode base + valid_tokens - 1."""

    if warp_specialized and (
        kv_heads != 2
        or gqa_group_size != 12
        or compact_dynamic_tiles
        or assume_full_blocks
    ):
        raise ValueError("WS prefill requires 24/2 heads and ordinary descriptors.")
    if kv_page_size < 0 or (kv_page_size and not warp_specialized):
        raise ValueError("unflattened paged K/V is only supported by WS prefill.")
    if block_n != 64:
        raise ValueError("prefill requires block_n=64 for its producer layout.")
    if direct_async_gather and not predecode_physical_ids:
        raise ValueError("direct_async_gather requires predecode_physical_ids=True")
    if prefetch_next_descriptor and (
        (not structured_microblocks)
        or (not predecode_physical_ids)
        or (not direct_async_gather)
        or (compact_dynamic_tiles)
    ):
        raise ValueError(
            "prefetch_next_descriptor requires the fixed direct-gather path"
        )
    if (assume_full_blocks or compact_dynamic_tiles) and (
        not structured_microblocks or not predecode_physical_ids
    ):
        raise ValueError(
            "assume_full_blocks/compact_dynamic_tiles require structured predecoded ids"
        )
    if active_tile_marker and not compact_dynamic_tiles:
        raise ValueError("active_tile_marker requires compact_dynamic_tiles=True")
    pass_configs = _pass_configs(tilelang, disable_tma_lower=True)
    metadata_width = T.dynamic("metadata_width")
    tile_count = T.ceildiv(
        metadata_width, block_n // 4 if structured_microblocks else block_n
    )
    rows, physical_tokens = T.dynamic("rows, physical_tokens")
    q_dim = 256
    block_h = 16
    if kv_heads <= 0 or not 0 < gqa_group_size <= block_h:
        raise ValueError("positive KV heads and a GQA group fitting M16 required.")
    kv_shape = (
        (T.dynamic("num_pages"), kv_page_size, kv_heads, q_dim)
        if kv_page_size
        else (physical_tokens, kv_heads, q_dim)
    )
    if kv_page_size:
        physical_tokens = kv_shape[0] * kv_page_size
    io_heads = gqa_group_size * kv_heads
    total_threads, num_stages = 384, 2
    qk_threads = pv_threads = 128
    qk_warp_groups, producer_group, pv_warp_groups = (0,), 1, (2,)
    dtype = dtype_name
    accum_dtype = T.float32
    v_gather_stripe_major = (
        (v_gather_stripe_major)
        and (structured_microblocks)
        and (predecode_physical_ids)
        and (direct_async_gather)
        and (prefetch_next_descriptor)
    )
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER] = False
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = True
    if disable_index_type_promotion:
        # The wrapper must prove that the allocation fits 32-bit addressing.
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION] = True
    pass_configs[
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED
    ] = not warp_specialized
    compile_flags = [] if warp_specialized else ["-mllvm", "-mtgpu-opt-level=1"]
    load_k = _make_qsa_load_k_macro(
        block_n,
        structured_microblocks,
        predecode_physical_ids,
        direct_async_gather,
        assume_full_blocks=assume_full_blocks,
        page_size=kv_page_size,
    )
    zero_invalid_v = _make_qsa_zero_v_macro(block_n, structured_microblocks)
    load_v_tile = _make_qsa_load_v_macro(
        block_n,
        structured_microblocks,
        predecode_physical_ids,
        direct_async_gather,
        stripe_major=v_gather_stripe_major,
        page_size=kv_page_size,
    )

    @tilelang.jit(
        target="musa",
        out_idx=[4],
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
    def kernel_factory():
        @T.macro
        def consume_tile(
            q_shared,
            k_shared,
            scores,
            scores_max,
            scores_max_prev,
            scores_sum,
            logsum,
            scores_scale_shared,
            scores_shared,
            v_shared,
            acc_o,
            physical_shared,
            barriers,
            barrier_slot,
            phase,
        ):
            """Run QK, online softmax, and PV for one ready tile."""

            T.gemm(
                q_shared,
                k_shared,
                scores,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, n in T.Parallel(block_h, block_n):
                valid = (i < gqa_group_size) and physical_shared[n] >= 0
                scores[i, n] = T.if_then_else(
                    valid,
                    scores[i, n],
                    -T.infinity(accum_dtype),
                )
            T.warpgroup_wait(0)
            T.mbarrier_arrive(barriers[3 + barrier_slot])
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -3.4028234663852886e38)
            T.reduce_max(scores, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_h):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                scores_scale_shared[i] = T.exp2(
                    (scores_max_prev[i] - scores_max[i]) * scale
                )
            T.mbarrier_arrive(barriers[9 + barrier_slot])
            T.mbarrier_wait_parity(barriers[9 + barrier_slot], phase)
            for i, n in T.Parallel(block_h, block_n):
                scores[i, n] = T.exp2((scores[i, n] - scores_max[i]) * scale)
            T.clear(scores_sum)
            T.reduce_sum(scores, scores_sum, dim=1, clear=True)
            for i in T.Parallel(block_h):
                logsum[i] = logsum[i] * scores_scale_shared[i] + scores_sum[i]
            for i, d in T.Parallel(block_h, q_dim):
                acc_o[i, d] = acc_o[i, d] * scores_scale_shared[i]
            T.copy(scores, scores_shared)
            T.mbarrier_arrive(barriers[11 + barrier_slot])
            T.mbarrier_wait_parity(barriers[11 + barrier_slot], phase)
            T.mbarrier_wait_parity(barriers[5 + barrier_slot], phase)
            T.gemm(
                scores_shared,
                v_shared,
                acc_o,
                policy=T.GemmWarpPolicy.FullRow,
            )
            if gqa_group_size == 3:
                # Probability residuals preserve precision and the outer barrier phase.
                for residual in T.serial(2):
                    T.warpgroup_wait(0)
                    for i, n in T.Parallel(block_h, block_n):
                        scores[i, n] = scores[i, n] - T.cast(
                            T.cast(scores[i, n], dtype), accum_dtype
                        )
                        scores_shared[i, n] = T.cast(scores[i, n], dtype)
                    T.mbarrier_arrive(barriers[11 + barrier_slot])
                    T.mbarrier_wait_parity(
                        barriers[11 + barrier_slot], (phase + residual + 1) % 2
                    )
                    T.gemm(
                        scores_shared, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                    )
            T.warpgroup_wait(0)
            T.mbarrier_arrive(barriers[7 + barrier_slot])

        q_strides = T.dynamic("q_stride_0, q_stride_1, q_stride_2", "int64")
        q_type: Any = T.StridedTensor((rows, io_heads, q_dim), q_strides, dtype)
        k_strides = tuple(
            T.dynamic(f"k_stride_{a}", "int64") for a in range(len(kv_shape) - 1)
        ) + (1,)
        k_type: Any = T.StridedTensor(kv_shape, k_strides, dtype)
        v_strides = tuple(
            T.dynamic(f"v_stride_{a}", "int64") for a in range(len(kv_shape) - 1)
        ) + (1,)
        v_type: Any = T.StridedTensor(kv_shape, v_strides, dtype)
        selected_physical_strides = T.dynamic(
            "selected_physical_stride_0, selected_physical_stride_1", "int64"
        )
        selected_physical_type: Any = T.StridedTensor(
            (rows, metadata_width), selected_physical_strides, T.int32
        )
        output_strides = T.dynamic(
            "output_stride_0, output_stride_1, output_stride_2", "int64"
        )
        output_type: Any = T.StridedTensor(
            (rows, io_heads, q_dim), output_strides, dtype
        )
        if warp_specialized:

            @T.prim_func
            def qsa_core_paged_prefill_ws(
                q: q_type,
                k: k_type,
                v: v_type,
                selected_physical: selected_physical_type,
                output: output_type,
            ):
                with T.Kernel(rows, kv_heads, threads=total_threads) as (row, kv_head):
                    T.assume(k.shape[0] <= 1 or k.strides[0] % 8 == 0)
                    T.assume(k.shape[1] <= 1 or k.strides[1] % 8 == 0)
                    T.assume(k.shape[-2] <= 1 or k.strides[-2] % 8 == 0)
                    T.assume(v.shape[0] <= 1 or v.strides[0] % 8 == 0)
                    T.assume(v.shape[1] <= 1 or v.strides[1] % 8 == 0)
                    T.assume(v.shape[-2] <= 1 or v.strides[-2] % 8 == 0)
                    q_shared = T.alloc_shared((block_h, q_dim), dtype)
                    # Separate rank-2 stages preserve SQMMA layouts on 0.1.12.
                    k_shared0 = T.alloc_shared((block_n, q_dim), dtype)
                    k_shared1 = T.alloc_shared((block_n, q_dim), dtype)
                    v_shared0 = T.alloc_shared((block_n, q_dim), dtype)
                    v_shared1 = T.alloc_shared((block_n, q_dim), dtype)
                    scores_shared0 = T.alloc_shared((block_h, block_n), dtype)
                    scores_shared1 = T.alloc_shared((block_h, block_n), dtype)
                    scores_scale0 = T.alloc_shared((block_h,), accum_dtype)
                    scores_scale1 = T.alloc_shared((block_h,), accum_dtype)
                    physical_shared0 = T.alloc_shared((block_n,), T.int32)
                    physical_shared1 = T.alloc_shared((block_n,), T.int32)
                    output_shared = T.alloc_shared((block_h, q_dim), dtype)
                    logsum_shared = T.alloc_shared((block_h,), accum_dtype)
                    scores = T.alloc_fragment((block_h, block_n), accum_dtype)
                    scores_max = T.alloc_fragment((block_h,), accum_dtype)
                    scores_max_prev = T.alloc_fragment((block_h,), accum_dtype)
                    scores_sum = T.alloc_fragment((block_h,), accum_dtype)
                    logsum = T.alloc_fragment((block_h,), accum_dtype)
                    acc_o = T.alloc_fragment((block_h, q_dim), accum_dtype)
                    # Dual IDs: q_ready, k/v_ready[2][2], qk_done[2][2],
                    # stage_free[2][2], final_ready, scores/scale_visible[2].
                    # Both consumers release each stage: 256 arrivals.
                    barriers = T.alloc_barrier(
                        [128] * 9
                        + [qk_threads] * 4
                        + [qk_threads + pv_threads] * 4
                        + [qk_threads] * 5
                    )
                    T.annotate_layout(
                        {
                            q_shared: tilelang.layout.make_sqmma_swizzled_layout(
                                q_shared, k_major=True
                            ),
                            k_shared0: tilelang.layout.make_sqmma_swizzled_layout(
                                k_shared0, k_major=True
                            ),
                            k_shared1: tilelang.layout.make_sqmma_swizzled_layout(
                                k_shared1, k_major=True
                            ),
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                    # Robust descriptors cover independent K/V storage spans.
                    k_bytes = cosize(k.shape, k.strides) * 2
                    v_bytes = cosize(v.shape, v.strides) * 2
                    k_robust_desc = T.make_robust_desc(
                        T.address_of(k[0, 0, 0, 0])
                        if kv_page_size
                        else T.address_of(k[0, 0, 0]),
                        k_bytes,
                    )
                    v_robust_desc = T.make_robust_desc(
                        T.address_of(v[0, 0, 0, 0])
                        if kv_page_size
                        else T.address_of(v[0, 0, 0]),
                        v_bytes,
                    )
                    q_head_start = kv_head * 12
                    # Publish K before V so QK overlaps the V gather.
                    with T.ws(producer_group):
                        # Manual WS retains CTA-global thread ids; normalize to 0..127.
                        producer_tid = T.get_thread_binding() - qk_threads
                        producer_tx = producer_tid % 8
                        producer_ty = producer_tid // 8
                        physical_local = T.alloc_local((4,), T.int32)
                        block_base_local = T.alloc_local((1,), T.int32)
                        T.clear(q_shared)
                        T.copy(
                            q[row, q_head_start : q_head_start + 12, 0:q_dim],
                            q_shared[0:12, 0:q_dim],
                            disable_tma=True,
                        )
                        T.sync_threads(15, 128)
                        T.mbarrier_arrive(barriers[0])
                        for tile_id in T.serial(tile_count):
                            stage = tile_id % num_stages
                            barrier_slot = (tile_id // num_stages) % 2
                            phase = (tile_id // (2 * num_stages)) % 2
                            if tile_id >= num_stages:
                                previous_tile = tile_id - num_stages
                                T.mbarrier_wait_parity(
                                    barriers[
                                        13
                                        + stage * 2
                                        + (previous_tile // num_stages) % 2
                                    ],
                                    (previous_tile // (2 * num_stages)) % 2,
                                )
                            descriptor_slot = tile_id * (block_n // 4) + producer_ty
                            descriptor = (
                                T.if_then_else(
                                    descriptor_slot < metadata_width,
                                    selected_physical[
                                        row, T.min(descriptor_slot, metadata_width - 1)
                                    ],
                                    -1,
                                )
                                if structured_microblocks
                                else -1
                            )
                            if stage == 0:
                                load_k(
                                    k_shared0,
                                    physical_shared0,
                                    tile_id,
                                    row,
                                    kv_head,
                                    producer_ty,
                                    producer_tx,
                                    physical_local,
                                    block_base_local,
                                    selected_physical,
                                    k,
                                    k_robust_desc,
                                    descriptor,
                                )
                            else:
                                load_k(
                                    k_shared1,
                                    physical_shared1,
                                    tile_id,
                                    row,
                                    kv_head,
                                    producer_ty,
                                    producer_tx,
                                    physical_local,
                                    block_base_local,
                                    selected_physical,
                                    k,
                                    k_robust_desc,
                                    descriptor,
                                )
                            # K group is drained before either stage is published.
                            T.async_copy_commit_group(0)
                            T.async_copy_wait_group(0)
                            T.lma_wait()
                            T.mbarrier_arrive(barriers[1 + 2 * stage + barrier_slot])
                            if stage == 0:
                                load_v_tile(
                                    v,
                                    v_shared0,
                                    physical_local,
                                    kv_head,
                                    producer_tx,
                                    producer_ty,
                                    v_robust_desc,
                                    physical_tokens,
                                )
                            else:
                                load_v_tile(
                                    v,
                                    v_shared1,
                                    physical_local,
                                    kv_head,
                                    producer_tx,
                                    producer_ty,
                                    v_robust_desc,
                                    physical_tokens,
                                )
                            # V group is drained before publication and ID reuse.
                            T.async_copy_commit_group(1)
                            T.async_copy_wait_group(1)
                            T.lma_wait()
                            if stage == 0:
                                zero_invalid_v(
                                    v_shared0,
                                    physical_shared0,
                                    producer_ty,
                                    producer_tx,
                                )
                                T.mbarrier_arrive(barriers[5 + barrier_slot])
                            else:
                                zero_invalid_v(
                                    v_shared1,
                                    physical_shared1,
                                    producer_ty,
                                    producer_tx,
                                )
                                T.mbarrier_arrive(barriers[7 + barrier_slot])
                    # QK/softmax consumer warp-group.
                    with T.ws(*qk_warp_groups):
                        T.clear(logsum)
                        T.fill(scores_max, -3.4028234663852886e38)
                        T.mbarrier_wait_parity(barriers[0], 0)
                        for tile_id in T.serial(tile_count):
                            stage = tile_id % num_stages
                            barrier_slot = (tile_id // num_stages) % 2
                            phase = (tile_id // (2 * num_stages)) % 2
                            T.mbarrier_wait_parity(
                                barriers[1 + 2 * stage + barrier_slot], phase
                            )
                            T.clear(scores)
                            if stage == 0:
                                T.gemm(
                                    q_shared,
                                    k_shared0,
                                    scores,
                                    transpose_B=True,
                                    clear_accum=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                )
                            else:
                                T.gemm(
                                    q_shared,
                                    k_shared1,
                                    scores,
                                    transpose_B=True,
                                    clear_accum=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                )
                            for i, n in T.Parallel(block_h, block_n):
                                physical_id = T.if_then_else(
                                    stage == 0, physical_shared0[n], physical_shared1[n]
                                )
                                valid = (i < 12) & (physical_id >= 0)
                                scores[i, n] = T.if_then_else(
                                    valid, scores[i, n], -T.infinity(accum_dtype)
                                )
                            T.warpgroup_wait(0)
                            T.copy(scores_max, scores_max_prev)
                            T.fill(scores_max, -3.4028234663852886e38)
                            T.reduce_max(scores, scores_max, dim=1, clear=False)
                            for i in T.Parallel(block_h):
                                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                                if stage == 0:
                                    scores_scale0[i] = T.exp2(
                                        (scores_max_prev[i] - scores_max[i]) * scale
                                    )
                                else:
                                    scores_scale1[i] = T.exp2(
                                        (scores_max_prev[i] - scores_max[i]) * scale
                                    )
                            T.mbarrier_arrive(barriers[20 + tile_id % 2])
                            T.mbarrier_wait_parity(
                                barriers[20 + tile_id % 2], (tile_id // 2) % 2
                            )
                            for i, n in T.Parallel(block_h, block_n):
                                scores[i, n] = T.exp2(
                                    (scores[i, n] - scores_max[i]) * scale
                                )
                            T.clear(scores_sum)
                            T.reduce_sum(scores, scores_sum, dim=1, clear=True)
                            for i in T.Parallel(block_h):
                                stage_scale = T.if_then_else(
                                    stage == 0, scores_scale0[i], scores_scale1[i]
                                )
                                logsum[i] = logsum[i] * stage_scale + scores_sum[i]
                            if stage == 0:
                                T.copy(scores, scores_shared0)
                            else:
                                T.copy(scores, scores_shared1)
                            # A dynamic sync_threads id cannot express a barrier ring.
                            T.mbarrier_arrive(barriers[18 + tile_id % 2])
                            T.mbarrier_wait_parity(
                                barriers[18 + tile_id % 2], (tile_id // 2) % 2
                            )
                            T.mbarrier_arrive(barriers[9 + 2 * stage + barrier_slot])
                            T.mbarrier_arrive(barriers[13 + 2 * stage + barrier_slot])
                        for i in T.Parallel(block_h):
                            logsum_shared[i] = logsum[i]
                        T.mbarrier_arrive(barriers[17])
                    # PV waits only for each stage's score/V publication.
                    with T.ws(*pv_warp_groups):
                        T.clear(acc_o)
                        for tile_id in T.serial(tile_count):
                            stage = tile_id % num_stages
                            barrier_slot = (tile_id // num_stages) % 2
                            phase = (tile_id // (2 * num_stages)) % 2
                            T.mbarrier_wait_parity(
                                barriers[5 + 2 * stage + barrier_slot], phase
                            )
                            T.mbarrier_wait_parity(
                                barriers[9 + 2 * stage + barrier_slot], phase
                            )
                            for i, d in T.Parallel(block_h, q_dim):
                                stage_scale = T.if_then_else(
                                    stage == 0, scores_scale0[i], scores_scale1[i]
                                )
                                acc_o[i, d] = acc_o[i, d] * stage_scale
                            if stage == 0:
                                T.gemm(
                                    scores_shared0,
                                    v_shared0,
                                    acc_o,
                                    clear_accum=False,
                                    policy=T.GemmWarpPolicy.FullRow,
                                )
                            else:
                                T.gemm(
                                    scores_shared1,
                                    v_shared1,
                                    acc_o,
                                    clear_accum=False,
                                    policy=T.GemmWarpPolicy.FullRow,
                                )
                            T.warpgroup_wait(0)
                            T.mbarrier_arrive(barriers[13 + 2 * stage + barrier_slot])
                        T.mbarrier_wait_parity(barriers[17], 0)
                        for i, d in T.Parallel(block_h, q_dim):
                            acc_o[i, d] = T.if_then_else(
                                i < 12,
                                acc_o[i, d] / T.max(logsum_shared[i], 1e-20),
                                0,
                            )
                        T.copy(acc_o, output_shared)
                        T.sync_threads(15, 128)
                        T.copy(
                            output_shared[0:12, 0:q_dim],
                            output[row, q_head_start : q_head_start + 12, 0:q_dim],
                            disable_tma=True,
                        )

            return qsa_core_paged_prefill_ws

        @T.prim_func
        def qsa_core_paged_prefill(
            q: q_type,
            k: k_type,
            v: v_type,
            selected_physical: selected_physical_type,
            output: output_type,
        ):
            with T.Kernel(rows, kv_heads, threads=256) as (row_id, kv_head):
                T.assume(k.shape[0] <= 1 or k.strides[0] % 8 == 0)
                T.assume(k.shape[1] <= 1 or k.strides[1] % 8 == 0)
                T.assume(k.shape[-2] <= 1 or k.strides[-2] % 8 == 0)
                T.assume(v.shape[0] <= 1 or v.strides[0] % 8 == 0)
                T.assume(v.shape[1] <= 1 or v.strides[1] % 8 == 0)
                T.assume(v.shape[-2] <= 1 or v.strides[-2] % 8 == 0)
                q_shared = T.alloc_shared((block_h, q_dim), dtype)
                k_shared = T.alloc_shared((block_n, q_dim), dtype)
                v_shared = T.alloc_shared((block_n, q_dim), dtype)
                output_shared = T.alloc_shared((block_h, q_dim), dtype)
                scores_shared = T.alloc_shared((block_h, block_n), dtype)
                physical_shared = T.alloc_shared((block_n,), T.int32)
                scores_scale_shared = T.alloc_shared((block_h,), accum_dtype)
                logsum_shared = T.alloc_shared((block_h,), accum_dtype)
                scores = T.alloc_fragment((block_h, block_n), accum_dtype)
                acc_o = T.alloc_fragment((block_h, q_dim), accum_dtype)
                scores_max = T.alloc_fragment((block_h,), accum_dtype)
                scores_max_prev = T.alloc_fragment((block_h,), accum_dtype)
                scores_sum = T.alloc_fragment((block_h,), accum_dtype)
                logsum = T.alloc_fragment((block_h,), accum_dtype)
                # One data stage, two notification IDs; Q-ready is published once.
                # q_ready, k_ready[2], k_free[2], v_ready[2], v_free[2],
                # scale_visible[2], scores_visible[2]. Repeated partial sync sites
                # also need distinct physical IDs to protect notification reuse.
                barriers = T.alloc_barrier([128] * 13)
                if compact_dynamic_tiles:
                    active_tiles_shared = T.alloc_shared((1,), T.int32)
                qk_layouts = {
                    q_shared: tilelang.layout.make_sqmma_swizzled_layout(
                        q_shared, k_major=True
                    )
                }
                qk_layouts[k_shared] = tilelang.layout.make_sqmma_swizzled_layout(
                    k_shared, k_major=True
                )
                T.annotate_layout(
                    qk_layouts,
                    allow_reannotation=True,
                    allow_buffer_region=True,
                )
                # Runtime int64 strides preserve independent K/V storage spans.
                k_bytes = cosize(k.shape, k.strides) * 2
                v_bytes = cosize(v.shape, v.strides) * 2
                k_robust_desc = T.make_robust_desc(T.address_of(k[0, 0, 0]), k_bytes)
                v_robust_desc = T.make_robust_desc(T.address_of(v[0, 0, 0]), v_bytes)
                q_head_start = kv_head * gqa_group_size
                tid = T.get_thread_binding()
                if tid >= 128:
                    producer_tid = tid - 128
                    producer_tx = producer_tid % 8
                    producer_ty = producer_tid // 8
                    physical_local = T.alloc_local((4,), T.int32)
                    block_base_local = T.alloc_local((1,), T.int32)
                    if prefetch_next_descriptor:
                        descriptor_prefetch_local = T.alloc_local((1,), T.int32)
                    if compact_dynamic_tiles:
                        # Fused metadata appends a tile-count marker to the valid prefix.
                        # Hand-built descriptors use the bounded probe instead.
                        if producer_tx == 0 and producer_ty == 0:
                            if active_tile_marker:
                                marker = selected_physical[row_id, metadata_width - 1]
                                active_tiles_shared[0] = T.min(
                                    T.max(-marker - 1000, 0), tile_count
                                )
                            else:
                                active_tiles_shared[0] = 0
                                for probe_tile in T.serial(tile_count):
                                    if (
                                        selected_physical[
                                            row_id, probe_tile * (block_n // 4)
                                        ]
                                        >= 0
                                    ):
                                        active_tiles_shared[0] = probe_tile + 1
                        T.sync_threads(14, 128)
                    # Keep dummy SQMMA rows local; global Q uses logical heads.
                    T.clear(q_shared)
                    T.copy(
                        q[
                            row_id,
                            q_head_start : q_head_start + gqa_group_size,
                            0:q_dim,
                        ],
                        q_shared[0:gqa_group_size, 0:q_dim],
                        disable_tma=True,
                    )
                    T.sync_threads(15, 128)
                    T.mbarrier_arrive(barriers[0])
                    producer_tile_bound = (
                        active_tiles_shared[0] if compact_dynamic_tiles else tile_count
                    )
                    if prefetch_next_descriptor:
                        descriptor_prefetch_local[0] = T.if_then_else(
                            producer_ty < metadata_width,
                            selected_physical[
                                row_id, T.min(producer_ty, metadata_width - 1)
                            ],
                            -1,
                        )
                    for tile_id in T.serial(producer_tile_bound):
                        barrier_slot = tile_id % 2
                        phase = (tile_id // 2) % 2
                        # Release follows buffer reuse, independently of barrier reuse.
                        if tile_id > 0:
                            T.mbarrier_wait_parity(
                                barriers[3 + (tile_id - 1) % 2],
                                ((tile_id - 1) // 2) % 2,
                            )
                        if structured_microblocks:
                            descriptor = (
                                descriptor_prefetch_local[0]
                                if prefetch_next_descriptor
                                else T.if_then_else(
                                    tile_id * (block_n // 4) + producer_ty
                                    < metadata_width,
                                    selected_physical[
                                        row_id,
                                        T.min(
                                            tile_id * (block_n // 4) + producer_ty,
                                            metadata_width - 1,
                                        ),
                                    ],
                                    -1,
                                )
                            )
                        else:
                            descriptor = -1
                        load_k(
                            k_shared,
                            physical_shared,
                            tile_id,
                            row_id,
                            kv_head,
                            producer_ty,
                            producer_tx,
                            physical_local,
                            block_base_local,
                            selected_physical,
                            k,
                            k_robust_desc,
                            descriptor,
                        )
                        # Group 0 owns K; drain it before publishing K-ready.
                        T.async_copy_commit_group(0)
                        if prefetch_next_descriptor:
                            next_descriptor_slot = (tile_id + 1) * (
                                block_n // 4
                            ) + producer_ty
                            descriptor_prefetch_local[0] = T.if_then_else(
                                next_descriptor_slot < metadata_width,
                                selected_physical[
                                    row_id,
                                    T.min(next_descriptor_slot, metadata_width - 1),
                                ],
                                -1,
                            )
                        T.async_copy_wait_group(0)
                        T.lma_wait()
                        T.mbarrier_arrive(barriers[1 + barrier_slot])
                        if tile_id > 0:
                            T.mbarrier_wait_parity(
                                barriers[7 + (tile_id - 1) % 2],
                                ((tile_id - 1) // 2) % 2,
                            )
                        load_v_tile(
                            v,
                            v_shared,
                            physical_local,
                            kv_head,
                            producer_tx,
                            producer_ty,
                            v_robust_desc,
                            physical_tokens,
                        )
                        # Group 1 owns V; drain it before publishing V-ready.
                        T.async_copy_commit_group(1)
                        T.async_copy_wait_group(1)
                        T.lma_wait()
                        zero_invalid_v(
                            v_shared, physical_shared, producer_ty, producer_tx
                        )
                        T.mbarrier_arrive(barriers[5 + barrier_slot])
                else:
                    T.clear(acc_o)
                    T.clear(logsum)
                    T.fill(scores_max, -3.4028234663852886e38)
                    T.mbarrier_wait_parity(barriers[0], 0)
                    consumer_tile_bound = (
                        active_tiles_shared[0] if compact_dynamic_tiles else tile_count
                    )
                    for tile_id in T.serial(consumer_tile_bound):
                        barrier_slot = tile_id % 2
                        phase = (tile_id // 2) % 2
                        T.mbarrier_wait_parity(barriers[1 + barrier_slot], phase)
                        consume_tile(
                            q_shared,
                            k_shared,
                            scores,
                            scores_max,
                            scores_max_prev,
                            scores_sum,
                            logsum,
                            scores_scale_shared,
                            scores_shared,
                            v_shared,
                            acc_o,
                            physical_shared,
                            barriers,
                            barrier_slot,
                            phase,
                        )
                    for i in T.Parallel(block_h):
                        logsum_shared[i] = logsum[i]
                    T.sync_threads(15, 128)
                    # Retain full M16 fragments until the logical-head global store.
                    for i, d in T.Parallel(block_h, q_dim):
                        acc_o[i, d] = T.if_then_else(
                            i < gqa_group_size,
                            acc_o[i, d] / T.max(logsum_shared[i], 1e-20),
                            0,
                        )
                    T.copy(acc_o, output_shared)
                    T.sync_threads(15, 128)
                    # 0.1.12 TME stores discard output strides; use them only for dense rows.
                    if (
                        use_tme_output
                        and output.strides[1] == q_dim
                        and output.strides[0] == io_heads * q_dim
                        and output.strides[2] == 1
                        and T.reinterpret(T.address_of(output[0, 0, 0]), T.uint64) % 16
                        == 0
                    ):
                        T.tma_copy(
                            output_shared[0:gqa_group_size, 0:q_dim],
                            output[
                                row_id,
                                q_head_start : q_head_start + gqa_group_size,
                                0:q_dim,
                            ],
                        )
                        T.tma_store_wait()
                    else:
                        # Plain LSU stores need an explicit completion barrier.
                        T.copy(
                            output_shared[0:gqa_group_size, 0:q_dim],
                            output[
                                row_id,
                                q_head_start : q_head_start + gqa_group_size,
                                0:q_dim,
                            ],
                            disable_tma=True,
                        )
                        T.sync_threads(15, 128)

        return qsa_core_paged_prefill

    return kernel_factory()


@lru_cache(maxsize=None)
def _qsa_core_splitk_merge_kernel(
    num_splits: int,
    partial_dtype_name: str,
    heads: int = 24,
    output_dtype_name: str = "bfloat16",
    *,
    gate_dtype_name: str | None = None,
    return_lse: bool = False,
):
    """Share merge math and layout dispatch; gate is a compile-time epilogue."""
    pass_configs = _pass_configs(tilelang, disable_tma_lower=True)

    if num_splits <= 0 or heads <= 0:
        raise ValueError("num_splits and heads must be positive.")
    rows = T.dynamic("rows")
    q_dim = 256
    accum_dtype = T.float32
    has_gate = gate_dtype_name is not None
    if has_gate and return_lse:
        raise ValueError("merged LSE is only available without the gate epilogue.")

    @tilelang.jit(
        target="musa",
        out_idx=[2, 3] if return_lse else [3] if has_gate else [2],
        pass_configs=pass_configs,
    )
    def kernel_factory():
        partial_output_strides = T.dynamic(
            "partial_output_stride_0, partial_output_stride_1, partial_output_stride_2, partial_output_stride_3",
            "int64",
        )
        partial_output_type: Any = T.StridedTensor(
            (num_splits, rows, heads, q_dim), partial_output_strides, partial_dtype_name
        )
        partial_lse_strides = T.dynamic(
            "partial_lse_stride_0, partial_lse_stride_1, partial_lse_stride_2",
            "int64",
        )
        partial_lse_type: Any = T.StridedTensor(
            (num_splits, rows, heads), partial_lse_strides, accum_dtype
        )
        output_strides = T.dynamic(
            "output_stride_0, output_stride_1, output_stride_2", "int64"
        )
        output_type: Any = T.StridedTensor(
            (rows, heads, q_dim), output_strides, output_dtype_name
        )
        epilogue_strides = T.dynamic("epilogue_stride_0, epilogue_stride_1", "int64")
        epilogue_type: Any = T.StridedTensor(
            (rows, heads),
            epilogue_strides,
            gate_dtype_name if has_gate else accum_dtype,
        )

        @T.macro
        def merge_values(
            partial_output, partial_lse, gate, output, output_lse, row, head
        ):
            weights = T.alloc_shared((num_splits,), accum_dtype)
            for split_id in T.Parallel(num_splits):
                max_lse = T.alloc_var(accum_dtype)
                max_lse = -T.infinity(accum_dtype)
                for candidate in T.serial(num_splits):
                    max_lse = T.max(max_lse, partial_lse[candidate, row, head])
                finite_max_lse = T.isfinite(max_lse)
                safe_max_lse = T.if_then_else(finite_max_lse, max_lse, 0.0)
                lse = partial_lse[split_id, row, head]
                weights[split_id] = T.if_then_else(
                    finite_max_lse and T.isfinite(lse),
                    T.exp2(lse - safe_max_lse),
                    T.cast(0, accum_dtype),
                )
            T.sync_threads()
            for d in T.Parallel(q_dim):
                denom = T.alloc_var(accum_dtype)
                accum = T.alloc_var(accum_dtype)
                denom = 0
                accum = 0
                for split_id in T.serial(num_splits):
                    weight = weights[split_id]
                    denom += weight
                    partial = T.cast(
                        partial_output[split_id, row, head, d], accum_dtype
                    )
                    accum += T.if_then_else(weight > 0, weight * partial, 0.0)
                merged = T.if_then_else(denom > 0, accum / T.max(denom, 1.0e-20), 0.0)
                gate_scale = (
                    T.sigmoid(T.cast(gate[row, head], accum_dtype)) if has_gate else 1.0
                )
                output[row, head, d] = T.cast(
                    T.if_then_else(denom > 0, merged * gate_scale, 0.0),
                    output_dtype_name,
                )
                if return_lse and d == 0:
                    max_lse = T.alloc_var(accum_dtype)
                    max_lse = -T.infinity(accum_dtype)
                    for candidate in T.serial(num_splits):
                        max_lse = T.max(max_lse, partial_lse[candidate, row, head])
                    output_lse[row, head] = T.if_then_else(
                        denom > 0,
                        (max_lse + T.log2(T.max(denom, 1.0e-20))) * 0.6931471805599453,
                        -T.infinity(accum_dtype),
                    )

        @T.macro
        def launch_merge(partial_output, partial_lse, gate, output, output_lse):
            with T.Kernel(rows, heads, threads=256) as (row, head):
                gate_is_dense = T.alloc_var(T.bool)
                if has_gate:
                    gate_is_dense = (
                        gate.strides[1] == 1
                        and gate.strides[0] == heads
                        and T.reinterpret(T.address_of(gate[0, 0]), T.uint64) % 16 == 0
                    )
                else:
                    gate_is_dense = True
                if (
                    partial_output.strides[3] == 1
                    and partial_output.strides[2] == q_dim
                    and partial_output.strides[1] == T.int64(heads) * q_dim
                    and partial_output.strides[0] == T.int64(rows) * heads * q_dim
                    and partial_lse.strides[2] == 1
                    and partial_lse.strides[1] == heads
                    and partial_lse.strides[0] == T.int64(rows) * heads
                    and output.strides[2] == 1
                    and output.strides[1] == q_dim
                    and output.strides[0] == T.int64(heads) * q_dim
                    and T.reinterpret(
                        T.address_of(partial_output[0, 0, 0, 0]), T.uint64
                    )
                    % 16
                    == 0
                    and T.reinterpret(T.address_of(partial_lse[0, 0, 0]), T.uint64) % 16
                    == 0
                    and T.reinterpret(T.address_of(output[0, 0, 0]), T.uint64) % 16 == 0
                    and gate_is_dense
                ):
                    dense_partial_output = T.make_tensor_from_addr(
                        T.address_of(partial_output[0, 0, 0, 0]),
                        (num_splits, rows, heads, q_dim),
                        partial_dtype_name,
                    )
                    dense_partial_lse = T.make_tensor_from_addr(
                        T.address_of(partial_lse[0, 0, 0]),
                        (num_splits, rows, heads),
                        accum_dtype,
                    )
                    dense_output = T.make_tensor_from_addr(
                        T.address_of(output[0, 0, 0]),
                        (rows, heads, q_dim),
                        output_dtype_name,
                    )
                    if has_gate:
                        dense_gate = T.make_tensor_from_addr(
                            T.address_of(gate[0, 0]),
                            (rows, heads),
                            gate_dtype_name,
                        )
                    merge_values(
                        dense_partial_output,
                        dense_partial_lse,
                        dense_gate if has_gate else gate,
                        dense_output,
                        output_lse,
                        row,
                        head,
                    )
                else:
                    merge_values(
                        partial_output, partial_lse, gate, output, output_lse, row, head
                    )

        if return_lse:

            @T.prim_func
            def qsa_core_splitk_merge_lse(
                partial_output: partial_output_type,
                partial_lse: partial_lse_type,
                output: output_type,
                output_lse: epilogue_type,
            ):
                launch_merge(
                    partial_output, partial_lse, partial_lse, output, output_lse
                )

            return qsa_core_splitk_merge_lse

        if has_gate:

            @T.prim_func
            def qsa_core_splitk_merge_gate(
                partial_output: partial_output_type,
                partial_lse: partial_lse_type,
                gate: epilogue_type,
                output: output_type,
            ):
                launch_merge(partial_output, partial_lse, gate, output, partial_lse)

            return qsa_core_splitk_merge_gate

        @T.prim_func
        def qsa_core_splitk_merge(
            partial_output: partial_output_type,
            partial_lse: partial_lse_type,
            output: output_type,
        ):
            # The static no-gate epilogue never reads this placeholder operand.
            launch_merge(partial_output, partial_lse, partial_lse, output, partial_lse)

        return qsa_core_splitk_merge

    return kernel_factory()


@lru_cache(maxsize=None)
def _qsa_prefill_metadata_kernel(
    page_size: int,
    sort_physical: bool = False,
    threads: int = 256,
    compact_tail: bool = False,
    active_tile_marker: bool = False,
    active_tile_width: int = 16,
    structured_microblocks: bool = True,
):
    pass_configs = _pass_configs(tilelang, disable_tma_lower=True)

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if threads not in (128, 256, 512):
        raise ValueError("threads must be 128, 256, or 512")
    if active_tile_marker and (not compact_tail or active_tile_width <= 0):
        raise ValueError(
            "active_tile_marker requires compact_tail and a positive tile width"
        )

    rows, requests, num_blocks = T.dynamic("rows, requests, num_blocks")
    max_pages_per_request, descriptor_groups = T.dynamic(
        "max_pages_per_request, descriptor_groups"
    )

    if sort_physical:
        # Every compare/swap pass explicitly publishes its shared writes.
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = True

    @tilelang.jit(
        target="musa",
        out_idx=[5],
        pass_configs=pass_configs,
    )
    def kernel_factory():
        block_indices_strides = T.dynamic(
            "block_indices_stride_0, block_indices_stride_1", "int64"
        )
        block_indices_type: Any = T.StridedTensor(
            (rows, num_blocks), block_indices_strides, T.int32
        )
        token_to_req_strides = (T.dynamic("token_to_req_stride_0", "int64"),)
        token_to_req_type: Any = T.StridedTensor((rows,), token_to_req_strides, T.int32)
        query_positions_strides = (T.dynamic("query_positions_stride_0", "int64"),)
        query_positions_type: Any = T.StridedTensor(
            (rows,), query_positions_strides, T.int32
        )
        seq_lens_strides = (T.dynamic("seq_lens_stride_0", "int64"),)
        seq_lens_type: Any = T.StridedTensor((requests,), seq_lens_strides, T.int32)
        block_table_strides = T.dynamic(
            "block_table_stride_0, block_table_stride_1", "int64"
        )
        block_table_type: Any = T.StridedTensor(
            (requests, max_pages_per_request), block_table_strides, T.int32
        )
        descriptors_strides = T.dynamic(
            "descriptors_stride_0, descriptors_stride_1", "int64"
        )
        descriptors_type: Any = T.StridedTensor(
            (rows, descriptor_groups), descriptors_strides, T.int32
        )

        @T.macro
        def decode_descriptor(block, count, request, visible, pages, capacity, result):
            result[0] = -1
            logical = T.int64(block) * (4 if structured_microblocks else 1)
            page = logical // page_size
            if (
                block >= 0
                and (block <= 0x1FFFFFFF if structured_microblocks else True)
                and logical + count - 1 < visible
                and page < pages.shape[1]
            ):
                physical = pages[request, page]
                if physical >= 0 and physical < capacity:
                    result[0] = physical * page_size + logical % page_size + count - 1

        @T.prim_func
        def qsa_prefill_metadata_fused(
            block_indices: block_indices_type,
            token_to_req: token_to_req_type,
            query_positions: query_positions_type,
            seq_lens: seq_lens_type,
            block_table: block_table_type,
            descriptors: descriptors_type,
            num_physical_pages: T.int32,
        ):
            with T.Kernel(rows, threads=threads) as row:
                tx = T.get_thread_binding(0)
                row_meta = T.alloc_shared((2,), T.int32)
                descriptor = T.alloc_local((1,), T.int32)
                visible = T.alloc_local((1,), T.int32)
                tail_len = T.alloc_local((1,), T.int32)
                if sort_physical:
                    sort_values = T.alloc_shared((512,), T.uint32)
                if tx == 0:
                    row_meta[0] = token_to_req[row]
                    row_meta[1] = 0
                    if row_meta[0] >= 0 and row_meta[0] < requests:
                        row_meta[1] = T.max(
                            T.min(
                                T.int64(query_positions[row]) + 1,
                                T.int64(seq_lens[row_meta[0]]),
                            ),
                            0,
                        )
                T.sync_threads()
                visible[0] = row_meta[1]
                tail_len[0] = visible[0] % 4
                if sort_physical:
                    for slot in T.Parallel(512):
                        decode_descriptor(
                            block_indices[row, slot],
                            4,
                            row_meta[0],
                            visible[0],
                            block_table,
                            num_physical_pages,
                            descriptor,
                        )
                        # Unsigned -1 sorts last without consuming a valid int32 id.
                        sort_values[slot] = T.cast(descriptor[0], T.uint32)
                    T.sync_threads()
                    for network_stage in T.serial(1, 10):
                        for network_pass in T.serial(network_stage):
                            network_size = 1 << network_stage
                            network_stride = 1 << (network_stage - network_pass - 1)
                            for slot in T.Parallel(512):
                                partner = slot ^ network_stride
                                if partner > slot:
                                    left = sort_values[slot]
                                    right = sort_values[partner]
                                    ascending = (slot & network_size) == 0
                                    if (ascending and left > right) or (
                                        not ascending and left < right
                                    ):
                                        sort_values[slot] = right
                                        sort_values[partner] = left
                            T.sync_threads()
                tail_slot = (
                    T.min(visible[0] // 4, num_blocks) if compact_tail else num_blocks
                )
                for slot in T.Parallel(descriptor_groups):
                    if not structured_microblocks:
                        decode_descriptor(
                            block_indices[row, slot],
                            1,
                            row_meta[0],
                            visible[0],
                            block_table,
                            num_physical_pages,
                            descriptor,
                        )
                        descriptors[row, slot] = descriptor[0]
                    elif active_tile_marker and slot == descriptor_groups - 1:
                        descriptors[row, slot] = -1000 - T.ceildiv(
                            tail_slot + T.if_then_else(tail_len[0] > 0, 1, 0),
                            active_tile_width,
                        )
                    else:
                        descriptors[row, slot] = -1
                        if slot == tail_slot and tail_len[0] > 0:
                            decode_descriptor(
                                visible[0] // 4,
                                tail_len[0],
                                row_meta[0],
                                visible[0],
                                block_table,
                                num_physical_pages,
                                descriptor,
                            )
                            descriptors[row, slot] = descriptor[0]
                        elif slot < num_blocks:
                            if sort_physical:
                                descriptors[row, slot] = T.int32(sort_values[slot])
                            else:
                                decode_descriptor(
                                    block_indices[row, slot],
                                    4,
                                    row_meta[0],
                                    visible[0],
                                    block_table,
                                    num_physical_pages,
                                    descriptor,
                                )
                                descriptors[row, slot] = descriptor[0]

        return qsa_prefill_metadata_fused

    return kernel_factory()


@lru_cache(maxsize=None)
def _qsa_gate_kernel(
    gate_dtype_name: str,
    pack_heads: bool,
    heads: int = 24,
    output_dtype_name: str = "bfloat16",
):
    pass_configs = _pass_configs(tilelang, disable_tma_lower=True)

    if heads <= 0:
        raise ValueError("heads must be positive.")
    rows = T.dynamic("rows")
    tile_heads = heads if pack_heads else 1

    @tilelang.jit(target="musa", out_idx=[], pass_configs=pass_configs)
    def kernel_factory():
        output_strides = T.dynamic(
            "output_stride_0, output_stride_1, output_stride_2", "int64"
        )
        output_type: Any = T.StridedTensor(
            (rows, heads, 256), output_strides, output_dtype_name
        )
        gate_strides = T.dynamic("gate_stride_0, gate_stride_1", "int64")
        gate_type: Any = T.StridedTensor((rows, heads), gate_strides, gate_dtype_name)
        position_type: Any = T.StridedTensor(
            (rows,), (T.dynamic("position_stride", "int64"),), T.int32
        )

        @T.prim_func
        def qsa_gate_apply(
            output: output_type, gate: gate_type, positions: position_type
        ):
            with T.Kernel(rows, heads // tile_heads, threads=256) as (row, group):
                gate_scale = T.alloc_shared((tile_heads,), T.float32)
                for h in T.Parallel(tile_heads):
                    gate_scale[h] = T.sigmoid(
                        T.cast(gate[row, group * tile_heads + h], T.float32)
                    )
                T.sync_threads()
                for h, d in T.Parallel(tile_heads, 256):
                    head = group * tile_heads + h
                    result = T.cast(output[row, head, d], T.float32) * gate_scale[h]
                    output[row, head, d] = T.cast(
                        T.if_then_else(positions[row] >= 0, result, 0.0),
                        output_dtype_name,
                    )

        return qsa_gate_apply

    return kernel_factory()
