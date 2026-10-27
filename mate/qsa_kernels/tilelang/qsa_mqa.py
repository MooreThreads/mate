"""TileLang-MUSA QSA MQA score kernels.

The long-prefill path prefetches query tiles with TME and uses supported
SQMMA and reduction operations. Bounded row partitions retain arbitrary query
lengths; large output strides use explicit INT64 addressing.
"""

from functools import lru_cache
from typing import Any

from . import _pass_configs

try:
    import tilelang
    import tilelang.language as T
except ImportError as error:
    _tilelang_import_error = error
    tilelang = None
    T = None


_LONG_CONTEXT_COMPILE_FLAGS = (
    "-mllvm -enable-cross-basicblock-fence-setting",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
)


@lru_cache(maxsize=None)
def _qsa_mqa_score_packed_kernel(
    block_n: int,
    block_m: int,
    group_n: int,
    scale: float,
    use_tme_q: bool = True,
    use_tme_k: bool = True,
    threads: int = 128,
    dtype_name: str = "bfloat16",
):
    """Build a batched PackGQA score kernel.

    Packing ``block_m`` query rows turns the four indexer heads into one
    ``block_m * 4`` GEMM dimension. It executes only the four semantic heads
    and reuses the staged query tile for
    ``group_n`` consecutive key tiles.
    """

    if tilelang is None:
        raise ModuleNotFoundError(
            "QSA kernels require a working TileLang-MUSA installation.", name="tilelang"
        ) from _tilelang_import_error

    if block_n <= 0 or block_n % 16:
        raise ValueError("block_n must be a positive multiple of 16.")
    if block_m <= 0 or (block_m * 4) % 32:
        raise ValueError("block_m * 4 must be a positive multiple of 32.")

    if group_n <= 0:
        raise ValueError("group_n must be positive.")
    if threads not in (128, 256, 512):
        raise ValueError("threads must be 128, 256, or 512.")

    batch = T.dynamic("batch")
    rows = T.dynamic("rows")
    num_blocks = T.dynamic("num_blocks")
    heads = 4
    dim = 128
    dtype = dtype_name
    accum_dtype = T.float32
    output_type: Any = T.StridedTensor(
        (batch, rows, num_blocks),
        tuple(T.dynamic(f"output_stride_{axis}", "int64") for axis in range(3)),
        accum_dtype,
    )

    @tilelang.jit(
        target="musa",
        out_idx=[3],
        pass_configs=_pass_configs(tilelang, enable_tme=use_tme_q or use_tme_k),
    )
    def kernel_factory():
        q_type: Any = T.StridedTensor(
            (batch, rows, heads, dim),
            (
                T.dynamic("q_stride_0", "int64"),
                T.dynamic("q_stride_1", "int64"),
                T.dynamic("q_stride_2", "int64"),
                1,
            ),
            dtype,
        )
        k_type: Any = T.StridedTensor(
            (batch, num_blocks, dim),
            (
                T.dynamic("k_stride_0", "int64"),
                T.dynamic("k_stride_1", "int64"),
                1,
            ),
            dtype,
        )
        visible_counts_type: Any = T.StridedTensor(
            (batch, rows),
            tuple(
                T.dynamic(f"visible_counts_stride_{axis}", "int64") for axis in range(2)
            ),
            T.int32,
        )

        @T.prim_func
        def qsa_mqa_score_packed(
            q: q_type,
            k: k_type,
            visible_counts: visible_counts_type,
            output: output_type,
        ):
            with T.Kernel(
                batch * (T.ceildiv(rows, block_m)),
                T.ceildiv(num_blocks, block_n * group_n),
                threads=threads,
            ) as (global_row_block, key_group):
                # Wrappers validate aligned vector rows; strides remain runtime values.
                T.assume(q.shape[0] <= 1 or q.strides[0] % 8 == 0)
                T.assume(q.shape[-2] <= 1 or q.strides[-2] % 8 == 0)
                T.assume(k.shape[0] <= 1 or k.strides[0] % 8 == 0)
                T.assume(k.shape[-2] <= 1 or k.strides[-2] % 8 == 0)
                q_shared = T.alloc_shared((block_m, heads, dim), dtype)
                q_matrix = T.reshape(q_shared, (block_m * heads, dim))
                k_shared = T.alloc_shared((block_n, dim), dtype)
                scores = T.alloc_fragment((block_n, block_m * heads), accum_dtype)
                scores_3d = T.reshape(scores, (block_n, block_m, heads))
                reduced = T.alloc_fragment((block_n, block_m), accum_dtype)

                blocks_per_batch = T.ceildiv(rows, block_m)
                bid = global_row_block // blocks_per_batch
                row_start = (global_row_block % blocks_per_batch) * block_m
                if use_tme_q:
                    q_barrier = T.alloc_barrier(128)
                    q_tme = T.make_tensor_from_addr(
                        q.data,
                        q.shape,
                        dtype,
                        strides=(
                            T.if_then_else(
                                batch <= 1, T.int64(rows) * heads * dim, q.strides[0]
                            ),
                            T.if_then_else(
                                rows <= 1, T.int64(heads * dim), q.strides[1]
                            ),
                            q.strides[2],
                            1,
                        ),
                    )
                    T.tma_copy(
                        q_tme[
                            bid,
                            row_start : row_start + block_m,
                            0:heads,
                            0:dim,
                        ],
                        q_shared,
                        barrier=q_barrier,
                    )
                    T.barrier_arrive(q_barrier)
                    T.mbarrier_wait_parity(q_barrier, 0)
                else:
                    T.copy(
                        q[
                            bid,
                            row_start : row_start + block_m,
                            0:heads,
                            0:dim,
                        ],
                        q_shared,
                        disable_tma=True,
                    )

                if use_tme_k:
                    # Alternate physical IDs before reusing a TME completion.
                    k_barrier = T.alloc_barrier([128, 128])
                for group_idx in T.serial(group_n):
                    key_start = (key_group * group_n + group_idx) * block_n
                    if use_tme_k:
                        k_tme = T.make_tensor_from_addr(
                            k.data,
                            k.shape,
                            dtype,
                            strides=(
                                T.if_then_else(
                                    batch <= 1, T.int64(num_blocks) * dim, k.strides[0]
                                ),
                                T.if_then_else(
                                    num_blocks <= 1, T.int64(dim), k.strides[1]
                                ),
                                1,
                            ),
                        )
                        T.tma_copy(
                            k_tme[bid, key_start : key_start + block_n, 0:dim],
                            k_shared,
                            barrier=k_barrier[group_idx % 2],
                        )
                        T.barrier_arrive(k_barrier[group_idx % 2])
                        T.mbarrier_wait_parity(
                            k_barrier[group_idx % 2], (group_idx // 2) % 2
                        )
                    else:
                        T.copy(
                            k[bid, key_start : key_start + block_n, 0:dim],
                            k_shared,
                            disable_tma=True,
                        )

                    T.clear(scores)
                    T.gemm(
                        k_shared,
                        q_matrix,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, m, h in T.Parallel(block_n, block_m, heads):
                        scores_3d[n, m, h] = T.max(scores_3d[n, m, h], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=2, clear=True)
                    for n, m in T.Parallel(block_n, block_m):
                        row = row_start + m
                        logical_block = key_start + n
                        if row < rows and logical_block < num_blocks:
                            output[bid, row, logical_block] = T.if_then_else(
                                logical_block < visible_counts[bid, row],
                                reduced[n, m] * scale,
                                -T.infinity(accum_dtype),
                            )

        return qsa_mqa_score_packed

    return kernel_factory()


@lru_cache(maxsize=None)
def _qsa_mqa_score_short_context_kernel(
    block_n: int,
    block_m: int,
    group_n: int,
    scale: float,
    skip_mask: bool = False,
    compile_profile: str = "default",
    dtype_name: str = "bfloat16",
):
    """Reuse resident K tiles across query subtiles with dynamic lengths.

    The regular PackGQA path uses a 128-column accumulator for BM32 (32 rows
    times four heads). The steady BM64 path uses 16-row, 64-column SQMMA
    tiles and overlaps the next Q copy with the current epilogue. Its explicit
    shared-lifetime sync avoids a redundant CTA barrier after the synchronous
    SQMMA call. BM128/BM256/BM512 use the same M16 loop and reuse each staged K
    tile across progressively more query rows. K remains resident across all
    row sub-tiles. This short-context specialization retains the validated
    stage-1 LSU schedule independently of the long-context TME mainloop.
    Mask-free ranges retain the M8 schedule for partial key/row tiles.

    Key ranges are runtime scalars. ``skip_mask`` requires a range fully
    visible for every real row. Logical lengths and tensor strides are dynamic.
    """

    if tilelang is None:
        raise ModuleNotFoundError(
            "QSA kernels require a working TileLang-MUSA installation.", name="tilelang"
        ) from _tilelang_import_error

    heads = 4
    optimized_mainloop = (
        # Reuse the resident K tile across all row subtiles with M16 SQMMA.
        block_m in (64, 128, 256, 512, 1024, 2048)
    )
    split_m = (
        8 if optimized_mainloop and (skip_mask or compile_profile == "long") else 16
    )
    mainloop_stages = 1 if optimized_mainloop else 0
    mainloop_manual_sync = optimized_mainloop
    mainloop_extra_wg_wait = mainloop_manual_sync
    if block_n <= 0 or block_n % 16:
        raise ValueError("block_n must be a positive multiple of 16.")
    if group_n <= 0:
        raise ValueError("group_n must be positive.")
    fast_visible_mask = (
        optimized_mainloop
        # Explicit BM1024/BM2048 tiles retain elementwise masks. The score
        # overlap path may enlarge a BM512 tile below; its serial visibility
        # scan avoids unrolling hundreds of loads per participating lane.
        and block_m <= 512
        and mainloop_stages == 1
        and mainloop_manual_sync
        and not skip_mask
    )
    if block_m <= 0 or block_m % split_m:
        raise ValueError("block_m must be a positive multiple of mainloop_split_m.")

    batch = T.dynamic("batch")
    rows_per_batch = T.dynamic("rows_per_batch")
    num_blocks = T.dynamic("num_blocks")
    dim = 128
    dtype = dtype_name
    accum_dtype = T.float32
    output_type: Any = T.StridedTensor(
        (batch, rows_per_batch, num_blocks),
        tuple(T.dynamic(f"output_stride_{axis}", "int64") for axis in range(3)),
        accum_dtype,
    )

    pass_configs = _pass_configs(tilelang, enable_tme=False)
    # Global strides and tile bases are INT64. Only the guarded complete-tile
    # alias below uses a bounded INT32 offset relative to that base.
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION] = True
    if mainloop_manual_sync:
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = True
    if compile_profile not in {"default", "long", "combine_fop"}:
        raise ValueError("unknown split-M compiler profile")
    compile_flags = (
        # The short LSU loop needs the ILP/FOP schedule, but cross-BB fence
        # scheduling crashes MCC 5.2's register allocator on this mainloop.
        [
            *_LONG_CONTEXT_COMPILE_FLAGS[1:],
            "-mllvm",
            "-enable-cross-basicblock-fence-setting=false",
        ]
        if compile_profile == "long"
        else ["-mllvm", "-mtgpu-combine-fop-instr=1"]
        if compile_profile == "combine_fop"
        else []
    )
    copy_kwargs: dict[str, Any] = {"disable_tma": True}

    @tilelang.jit(
        target="musa",
        out_idx=[3],
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
    def kernel_factory():
        q_type: Any = T.StridedTensor(
            (batch, rows_per_batch, heads, dim),
            (
                T.dynamic("q_stride_0", "int64"),
                T.dynamic("q_stride_1", "int64"),
                T.dynamic("q_stride_2", "int64"),
                1,
            ),
            dtype,
        )
        k_type: Any = T.StridedTensor(
            (batch, num_blocks, dim),
            (
                T.dynamic("k_stride_0", "int64"),
                T.dynamic("k_stride_1", "int64"),
                1,
            ),
            dtype,
        )
        visible_counts_type: Any = T.StridedTensor(
            (batch, rows_per_batch),
            tuple(
                T.dynamic(f"visible_counts_stride_{axis}", "int64") for axis in range(2)
            ),
            T.int32,
        )

        @T.macro
        def compute_scores(
            q,
            k,
            visible_counts,
            output,
            row_block,
            bid,
            key_group,
            q_shared,
            k_shared,
            score_shared,
            visible_min_shared,
            key_start_offset,
            key_end,
            q_batch_stride,
            q_row_stride,
            q_head_stride,
            k_batch_stride,
            k_row_stride,
            output_batch_stride,
            output_row_stride,
            output_column_stride,
            full_tiles,
        ):
            if fast_visible_mask:
                visible_min_local = T.alloc_var(T.int32)
                tid = T.get_thread_binding()
            scores = T.alloc_fragment((block_n, split_m * heads), accum_dtype)
            scores_3d = T.reshape(scores, (block_n, split_m, heads))
            reduced = T.alloc_fragment((block_n, split_m), accum_dtype)
            row_start = row_block * block_m
            # Keep tile aliases outside the pipelined subtile loop so the
            # compiler can overlap the next Q load with the score epilogue.
            query_tiles = T.make_tensor_from_addr(
                q.data,
                (batch, T.ceildiv(rows_per_batch, block_m), block_m * heads, dim),
                dtype,
                strides=(q_batch_stride, block_m * heads * dim, dim, 1),
            )
            key_tiles = T.make_tensor_from_addr(
                k.data,
                (batch, T.ceildiv(num_blocks, block_n), block_n, dim),
                dtype,
                strides=(k_batch_stride, block_n * dim, dim, 1),
            )
            query_tile = T.make_tensor_from_addr(
                T.reinterpret(
                    T.reinterpret(q.data, T.uint64)
                    + (
                        T.int64(bid) * q_batch_stride
                        + T.int64(row_start) * q_row_stride
                    )
                    * 2,
                    "handle",
                ),
                (block_m, heads, dim),
                dtype,
                strides=(q_row_stride, q_head_stride, 1),
            )
            for group_idx in T.serial(group_n):
                key_start = (
                    key_start_offset + (key_group * group_n + group_idx) * block_n
                )
                output_tile = T.make_tensor_from_addr(
                    T.reinterpret(
                        T.reinterpret(output.data, T.uint64)
                        + (
                            T.int64(bid) * output_batch_stride
                            + T.int64(row_start) * output_row_stride
                            + T.int64(key_start) * output_column_stride
                        )
                        * 4,
                        "handle",
                    ),
                    (block_m, block_n),
                    accum_dtype,
                    strides=(output_row_stride, output_column_stride),
                )
                if full_tiles:
                    T.copy(
                        key_tiles[bid, key_start // block_n, 0:block_n, 0:dim],
                        k_shared,
                        **copy_kwargs,
                    )
                elif key_start + block_n <= num_blocks:
                    key_tile = T.make_tensor_from_addr(
                        T.reinterpret(
                            T.reinterpret(k.data, T.uint64)
                            + (
                                T.int64(bid) * k_batch_stride
                                + T.int64(key_start) * k_row_stride
                            )
                            * 2,
                            "handle",
                        ),
                        (block_n, dim),
                        dtype,
                        strides=(k_row_stride, 1),
                    )
                    T.copy(key_tile, k_shared, **copy_kwargs)
                else:
                    T.copy(
                        k[bid, key_start : key_start + block_n, 0:dim],
                        k_shared,
                        **copy_kwargs,
                    )
                if fast_visible_mask:
                    if tid < 8:
                        if full_tiles:
                            visible_min_local = visible_counts[bid, row_start + tid]
                            for row_group in T.unroll(block_m // 8 - 1, explicit=True):
                                visible_min_local = T.min(
                                    visible_min_local,
                                    visible_counts[
                                        bid, row_start + tid + (row_group + 1) * 8
                                    ],
                                )
                        else:
                            visible_min_local = T.if_then_else(
                                row_start + tid < rows_per_batch,
                                visible_counts[
                                    bid, T.min(row_start + tid, rows_per_batch - 1)
                                ],
                                2147483647,
                            )
                            for row_group in T.unroll(block_m // 8 - 1, explicit=True):
                                visible_min_local = T.min(
                                    visible_min_local,
                                    T.if_then_else(
                                        row_start + tid + (row_group + 1) * 8
                                        < rows_per_batch,
                                        visible_counts[
                                            bid,
                                            T.min(
                                                row_start + tid + (row_group + 1) * 8,
                                                rows_per_batch - 1,
                                            ),
                                        ],
                                        2147483647,
                                    ),
                                )
                        visible_min_shared[tid] = visible_min_local
                if mainloop_manual_sync:
                    T.sync_threads()
                if fast_visible_mask:
                    if tid == 0:
                        visible_min_local = visible_min_shared[0]
                        for row_group in T.unroll(7, explicit=True):
                            visible_min_local = T.min(
                                visible_min_local,
                                visible_min_shared[row_group + 1],
                            )
                        visible_min_shared[0] = visible_min_local
                for sub in T.Pipelined(
                    block_m // split_m,
                    num_stages=mainloop_stages,
                ):
                    sub_row_start = row_start + sub * split_m
                    if full_tiles:
                        T.copy(
                            query_tiles[
                                bid,
                                row_block,
                                sub * split_m * heads : (sub + 1) * split_m * heads,
                                0:dim,
                            ],
                            q_shared,
                            **copy_kwargs,
                        )
                    elif row_start + block_m <= rows_per_batch:
                        T.copy(
                            query_tile[
                                sub * split_m : (sub + 1) * split_m, 0:heads, 0:dim
                            ],
                            T.reshape(q_shared, (split_m, heads, dim)),
                            **copy_kwargs,
                        )
                    else:
                        T.copy(
                            q[
                                bid,
                                sub_row_start : sub_row_start + split_m,
                                0:heads,
                                0:dim,
                            ],
                            T.reshape(q_shared, (split_m, heads, dim)),
                            **copy_kwargs,
                        )
                    if mainloop_manual_sync:
                        T.sync_threads()
                    T.clear(scores)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    if mainloop_extra_wg_wait:
                        T.warpgroup_wait(0)
                    for n, m, h in T.Parallel(block_n, split_m, heads):
                        scores_3d[n, m, h] = T.max(scores_3d[n, m, h], 0.0)
                    T.reduce_sum(
                        scores_3d,
                        reduced,
                        dim=2,
                        clear=True,
                        batch=4 if optimized_mainloop else 1,
                    )
                    if skip_mask:
                        for n, m in T.Parallel(block_n, split_m):
                            reduced[n, m] = reduced[n, m] * scale
                    elif fast_visible_mask:
                        if key_start + block_n <= visible_min_shared[0]:
                            for n, m in T.Parallel(block_n, split_m):
                                reduced[n, m] = reduced[n, m] * scale
                        else:
                            for n, m in T.Parallel(block_n, split_m):
                                row = sub_row_start + m
                                logical_block = key_start + n
                                reduced[n, m] = T.if_then_else(
                                    logical_block
                                    < visible_counts[
                                        bid, T.min(row, rows_per_batch - 1)
                                    ],
                                    reduced[n, m] * scale,
                                    -T.infinity(accum_dtype),
                                )
                    else:
                        for n, m in T.Parallel(block_n, split_m):
                            row = sub_row_start + m
                            logical_block = key_start + n
                            reduced[n, m] = T.if_then_else(
                                logical_block
                                < visible_counts[bid, T.min(row, rows_per_batch - 1)],
                                reduced[n, m] * scale,
                                -T.infinity(accum_dtype),
                            )
                    if full_tiles:
                        for n, m in T.Parallel(block_n, split_m):
                            score_shared[m, n] = reduced[n, m]
                        T.sync_threads()
                        T.assume(num_blocks % 4 == 0)
                        T.assume(key_start % 4 == 0)
                        output_vectors = T.make_tensor_from_addr(
                            output.data,
                            (batch, rows_per_batch, num_blocks // 4, 4),
                            accum_dtype,
                        )
                        for m, n in T.Parallel(split_m, block_n // 4):
                            for lane in T.vectorized(4):
                                output_vectors[
                                    bid, sub_row_start + m, key_start // 4 + n, lane
                                ] = score_shared[m, n * 4 + lane]
                    elif (
                        row_start + block_m <= rows_per_batch
                        and key_start + block_n <= key_end
                    ):
                        for n, m in T.Parallel(block_n, split_m):
                            output_tile[sub * split_m + m, n] = reduced[n, m]
                    else:
                        for n, m in T.Parallel(block_n, split_m):
                            if (
                                sub_row_start + m < rows_per_batch
                                and key_start + n < key_end
                            ):
                                output[bid, sub_row_start + m, key_start + n] = reduced[
                                    n, m
                                ]

        @T.prim_func
        def qsa_mqa_score_short_context(
            q: q_type,
            k: k_type,
            visible_counts: visible_counts_type,
            output: output_type,
            key_start_offset: T.int32,
            key_blocks: T.int32,
        ):
            with T.Kernel(
                T.ceildiv(rows_per_batch, block_m),
                T.ceildiv(key_blocks, block_n * group_n),
                batch,
                threads=128,
            ) as (row_block, key_group, bid):
                T.assume(q.shape[0] <= 1 or q.strides[0] % 8 == 0)
                T.assume(q.shape[1] <= 1 or q.strides[1] % 8 == 0)
                T.assume(q.shape[2] <= 1 or q.strides[2] % 8 == 0)
                T.assume(k.shape[0] <= 1 or k.strides[0] % 8 == 0)
                T.assume(k.shape[-2] <= 1 or k.strides[-2] % 8 == 0)
                q_shared = T.alloc_shared((split_m * heads, dim), dtype)
                k_shared = T.alloc_shared((block_n, dim), dtype)
                score_shared = T.alloc_shared((split_m, block_n), accum_dtype)
                T.annotate_layout(
                    {
                        score_shared: tilelang.Layout(
                            (split_m, block_n), lambda m, n: (m, n)
                        )
                    }
                )
                visible_min_shared = T.alloc_shared((8,), T.int32)
                if (
                    q.strides[2] == dim
                    and q.strides[1] == heads * dim
                    and q.strides[0] == T.int64(rows_per_batch) * heads * dim
                    and k.strides[1] == dim
                    and k.strides[0] == T.int64(num_blocks) * dim
                    and output.strides[2] == 1
                    and output.strides[1] == num_blocks
                    and num_blocks % 4 == 0
                    and output.strides[0] == T.int64(rows_per_batch) * num_blocks
                    and T.int64(batch) * rows_per_batch * num_blocks < 2147483648
                    and T.int64(batch) * rows_per_batch * heads * dim < 2147483648
                    and T.int64(batch) * num_blocks * dim < 2147483648
                    and key_start_offset >= 0
                    and key_start_offset % block_n == 0
                    and key_blocks > 0
                    and num_blocks >= block_n
                    and (row_block + 1) * block_m <= rows_per_batch
                    and key_start_offset + (key_group + 1) * block_n * group_n
                    <= T.min(num_blocks, key_start_offset + key_blocks)
                    and T.reinterpret(T.address_of(q[0, 0, 0, 0]), T.uint64) % 16 == 0
                    and T.reinterpret(T.address_of(k[0, 0, 0]), T.uint64) % 16 == 0
                    and T.reinterpret(T.address_of(output[0, 0, 0]), T.uint64) % 16 == 0
                ):
                    dense_q = T.make_tensor_from_addr(
                        q.data,
                        (batch, rows_per_batch, heads, dim),
                        dtype,
                    )
                    dense_k = T.make_tensor_from_addr(
                        k.data,
                        (batch, num_blocks, dim),
                        dtype,
                    )
                    dense_output = T.make_tensor_from_addr(
                        output.data,
                        (batch, rows_per_batch, num_blocks),
                        accum_dtype,
                    )
                    compute_scores(
                        dense_q,
                        dense_k,
                        visible_counts,
                        dense_output,
                        row_block,
                        bid,
                        key_group,
                        q_shared,
                        k_shared,
                        score_shared,
                        visible_min_shared,
                        key_start_offset,
                        T.min(num_blocks, key_start_offset + key_blocks),
                        T.int64(rows_per_batch) * heads * dim,
                        heads * dim,
                        dim,
                        T.int64(num_blocks) * dim,
                        dim,
                        output.strides[0],
                        T.int32(output.strides[1]),
                        1,
                        True,
                    )
                else:
                    compute_scores(
                        q,
                        k,
                        visible_counts,
                        output,
                        row_block,
                        bid,
                        key_group,
                        q_shared,
                        k_shared,
                        score_shared,
                        visible_min_shared,
                        key_start_offset,
                        T.min(num_blocks, key_start_offset + key_blocks),
                        q.strides[0],
                        q.strides[1],
                        q.strides[2],
                        k.strides[0],
                        k.strides[1],
                        output.strides[0],
                        output.strides[1],
                        output.strides[2],
                        False,
                    )

        return qsa_mqa_score_short_context

    return kernel_factory()


@lru_cache(maxsize=None)
def _qsa_mqa_score_split_m_kernel(
    block_n: int,
    block_m: int,
    group_n: int,
    scale: float,
    skip_mask: bool = False,
    bound_tail_rows: bool = False,
    score_epilogue_overlap: bool = False,
    partition_rows: bool = False,
    mainloop_visible_min: bool = True,
    compile_profile: str = "default",
    dtype_name: str = "bfloat16",
):
    """Score long-context tiles with resident K and a two-bank TME Q pipeline.

    Query/key lengths, strides, key ranges and scheduled row ranges are runtime
    values. Cached parameters select the SQMMA tile and compiler schedule only.
    The caller launches complete row tiles and an optional bounded tail without
    padding global inputs. Key ranges start on a block boundary and end on a
    block boundary or at the logical key length.
    """
    if tilelang is None:
        raise ModuleNotFoundError(
            "QSA kernels require a working TileLang-MUSA installation.", name="tilelang"
        ) from _tilelang_import_error
    if block_n != 128 or group_n != 1:
        raise ValueError("the long split-M mainloop requires block_n=128 and group_n=1")
    if block_m not in (64, 128, 256, 512, 1024, 2048):
        raise ValueError("unsupported long split-M query tile")
    if partition_rows and (bound_tail_rows or score_epilogue_overlap):
        raise ValueError("partition_rows requires complete non-overlapped tiles")
    if compile_profile not in {"default", "long", "combine_fop"}:
        raise ValueError("unknown split-M compiler profile")

    heads, dim, split_m = 4, 128, 8
    mainloop_stages = 1
    single_wg_tme_lookahead = True
    store_batch = 2
    fast_visible_mask = mainloop_visible_min and not skip_mask
    batch = T.dynamic("batch")
    rows_per_batch = T.dynamic("rows_per_batch")
    num_blocks = T.dynamic("num_blocks")
    dtype = dtype_name
    accum_dtype = T.float32
    output_type: Any = T.StridedTensor(
        (batch, rows_per_batch, num_blocks),
        tuple(T.dynamic(f"output_stride_{axis}", "int64") for axis in range(3)),
        accum_dtype,
    )
    pass_configs = _pass_configs(tilelang, enable_tme=True)
    # Preserve INT64 global bases; the guarded store alias alone uses a
    # bounded INT32 pitch, with an explicit four-float vector axis.
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION] = True
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = True
    compile_flags = (
        list(_LONG_CONTEXT_COMPILE_FLAGS[1:])
        if compile_profile == "long"
        else ["-mllvm", "-mtgpu-combine-fop-instr=1"]
        if compile_profile == "combine_fop"
        else []
    )
    compile_flags.extend(
        (
            "-mllvm",
            "-enable-pre=false",
            "-mllvm",
            "-enable-cross-basicblock-fence-setting=true",
        )
    )
    compile_flags.extend(("-mllvm", "-mtgpu-if-convert=1"))
    compile_flags.extend(("-mllvm", "-misched=mtgpu-max-ilp"))
    if compile_profile == "default":
        compile_flags.extend(("-mllvm", "-mtgpu-combine-fop-instr=1"))

    @tilelang.jit(
        target="musa",
        out_idx=[3],
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
    def kernel_factory():
        cta_row_parts = 2 if partition_rows else 1
        cta_block_m = block_m // cta_row_parts

        @T.macro
        def preload_q_lookahead(
            q, q_shared, barriers, barrier_slot, bid, row_start, sub
        ):
            # Compute uniform descriptor coordinates before the TME issuer predicate.
            # Promote each operand before multiplication; no loop-carried cursor.
            q_row_start = T.int64(row_start) + T.int64(sub) * split_m
            if T.get_thread_binding() == 0:
                q_tme = T.make_tensor_from_addr(
                    q.data,
                    q.shape,
                    dtype,
                    strides=(
                        T.if_then_else(
                            batch <= 1,
                            T.int64(rows_per_batch) * heads * dim,
                            q.strides[0],
                        ),
                        T.if_then_else(
                            rows_per_batch <= 1, T.int64(heads * dim), q.strides[1]
                        ),
                        q.strides[2],
                        1,
                    ),
                )
                T.tma_copy(
                    q_tme[bid, q_row_start : q_row_start + split_m, 0:heads, 0:dim],
                    q_shared[barrier_slot, 0:split_m, 0:heads, 0:dim],
                    barrier=barriers[barrier_slot],
                )
            # Every warp arrives, including the three non-issuer warps.
            T.barrier_arrive(barriers[barrier_slot])

        # One subtile per iteration keeps a common epilogue after the bank join.
        bank_pair_width = 1

        @T.macro
        def store_dense_scores(
            output,
            contiguous_store,
            batch_id,
            row_start,
            key_start,
            store_rows,
            base_row,
        ):
            tile_pitch = T.int32(output.strides[1])
            T.assume(tile_pitch > 0)
            T.assume(tile_pitch <= 2147483647 // block_m)
            output_vectors = T.make_tensor_from_addr(
                T.reinterpret(
                    T.reinterpret(output.data, T.uint64)
                    + (
                        T.int64(batch_id) * output.strides[0]
                        + T.int64(base_row) * output.strides[1]
                        + T.int64(key_start)
                    )
                    * 4,
                    "handle",
                ),
                (block_m, block_n // 4, 4),
                accum_dtype,
                strides=((tile_pitch // 4) * 4, 4, 1),
            )
            tid = T.get_thread_binding()
            for group in T.unroll(store_rows // 4):
                m = group * 4 + tid // 32
                n = tid % 32
                for lane in T.vectorized(4):
                    output_vectors[T.int32(row_start - base_row) + m, n, lane] = (
                        contiguous_store[m, n * 4 + lane] * scale
                    )

        @T.macro
        def store_scores(
            output, contiguous_store, batch_id, row_start, key_start, store_rows
        ):
            if (
                output.strides[2] == 1
                and output.strides[1] == num_blocks
                and output.strides[0] == T.int64(rows_per_batch) * num_blocks
                and T.reinterpret(T.address_of(output[0, 0, 0]), T.uint64) % 16 == 0
                and row_start + store_rows <= rows_per_batch
                and key_start + block_n <= num_blocks
                and num_blocks % 4 == 0
                and key_start % 4 == 0
                and num_blocks <= 2147483647 // block_m
            ):
                store_dense_scores(
                    output,
                    contiguous_store,
                    batch_id,
                    row_start,
                    key_start,
                    store_rows,
                    row_start,
                )
            else:
                for m, n in T.Parallel(store_rows, block_n):
                    if row_start + m < rows_per_batch and key_start + n < num_blocks:
                        output[batch_id, row_start + m, key_start + n] = (
                            contiguous_store[m, n] * scale
                        )

        @T.macro
        def issue_prefix_score(k_tile, q_tile, acc):
            T.gemm(
                k_tile,
                q_tile,
                acc,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullCol,
                wg_wait=-1,
            )

        @T.macro
        def reduce_visible_score(
            acc,
            dst,
            row_offset,
            visible_counts,
            minimum_visible,
            bid,
            row_start,
            key_start,
        ):
            # The fixed M128/N32 fragment layout gives 32 FP32 registers per lane.
            local_scores = T.make_tensor_from_addr(
                T.address_of(acc[0, 0]), (32,), accum_dtype, storage_scope="local"
            )
            shuffled = T.alloc_local((4,), accum_dtype)
            for group in T.unroll(8):
                index = T.Ramp(group * 4, 1, 4)
                local_scores[index] = T.max(local_scores[index], 0.0)
                for lane in T.unroll(4):
                    shuffled[lane] = T.shfl_xor(local_scores[group * 4 + lane], 2)
                local_scores[index] += shuffled[T.Ramp(0, 1, 4)]
                for lane in T.unroll(4):
                    shuffled[lane] = T.shfl_xor(local_scores[group * 4 + lane], 1)
                local_scores[index] += shuffled[T.Ramp(0, 1, 4)]
            tid = T.get_thread_binding()
            if skip_mask or key_start + block_n <= minimum_visible:
                if tid % 4 == 0:
                    for group in T.unroll(8):
                        for lane in T.unroll(4):
                            m = lane * 2 + (tid % 8) // 4
                            n = group * 16 + tid // 8
                            dst[row_offset + m, n] = local_scores[group * 4 + lane]
            else:
                if tid % 4 == 0:
                    for group in T.unroll(8):
                        for lane in T.unroll(4):
                            m = lane * 2 + (tid % 8) // 4
                            n = group * 16 + tid // 8
                            dst[row_offset + m, n] = T.if_then_else(
                                key_start + n
                                < visible_counts[
                                    bid, T.min(row_start + m, rows_per_batch - 1)
                                ],
                                local_scores[group * 4 + lane],
                                -T.infinity(accum_dtype),
                            )

        @T.macro
        def run_score_overlap(
            q,
            k_tile,
            q_tiles,
            q_load_tiles,
            barriers,
            first_acc,
            next_acc,
            scratch,
            output,
            visible_counts,
            minimum_visible,
            bid,
            row_start,
            key_start,
        ):
            pitch4 = T.int32(output.strides[1]) // 4
            output_vectors = T.make_tensor_from_addr(
                T.reinterpret(
                    T.reinterpret(output.data, T.uint64)
                    + (
                        T.int64(bid) * output.strides[0]
                        + T.int64(row_start) * output.strides[1]
                        + T.int64(key_start)
                    )
                    * 4,
                    "handle",
                ),
                ((block_m - 1) * pitch4 + block_n // 4, 4),
                accum_dtype,
                strides=(4, 1),
            )
            tid = T.get_thread_binding()
            # The initial bank was prefetched by the common producer. Each
            # issue overlaps only the other bank's completed-score epilogue.
            T.mbarrier_wait_parity(barriers[0], 0)
            T.sync_threads()
            preload_q_lookahead(
                q,
                q_load_tiles,
                barriers,
                1,
                bid,
                row_start,
                1,
            )
            issue_prefix_score(
                k_tile,
                q_tiles[0, 0 : split_m * heads, 0:dim],
                first_acc,
            )
            T.wait_sqmma(0)
            # Keep the paired mainloop compact when strides/lengths are runtime values.
            for pair in T.unroll(
                block_m // (2 * split_m) - 1, explicit=False, unroll_factor=2
            ):
                T.mbarrier_wait_parity(barriers[1], pair % 2)
                T.sync_threads()
                preload_q_lookahead(
                    q,
                    q_load_tiles,
                    barriers,
                    0,
                    bid,
                    row_start,
                    2 * pair + 2,
                )
                issue_prefix_score(
                    k_tile,
                    q_tiles[1, 0 : split_m * heads, 0:dim],
                    next_acc,
                )
                reduce_visible_score(
                    first_acc,
                    scratch,
                    0,
                    visible_counts,
                    minimum_visible,
                    bid,
                    row_start + 2 * pair * split_m,
                    key_start,
                )
                T.wait_sqmma(0)
                T.mbarrier_wait_parity(barriers[0], (pair + 1) % 2)
                T.sync_threads()
                preload_q_lookahead(
                    q,
                    q_load_tiles,
                    barriers,
                    1,
                    bid,
                    row_start,
                    2 * pair + 3,
                )
                issue_prefix_score(
                    k_tile,
                    q_tiles[0, 0 : split_m * heads, 0:dim],
                    first_acc,
                )
                reduce_visible_score(
                    next_acc,
                    scratch,
                    split_m,
                    visible_counts,
                    minimum_visible,
                    bid,
                    row_start + (2 * pair + 1) * split_m,
                    key_start,
                )
                T.sync_threads()
                for group in T.unroll(store_batch * split_m // 4):
                    m = group * 4 + tid // 32
                    dense_index = (2 * pair * split_m + m) * pitch4 + tid % 32
                    T.assume(dense_index >= 0)
                    T.assume(dense_index < (block_m - 1) * pitch4 + block_n // 4)
                    for lane in T.vectorized(4):
                        output_vectors[dense_index, lane] = (
                            scratch[m, (tid % 32) * 4 + lane] * scale
                        )
                # Complete the score before its Q bank is released for reuse.
                T.wait_sqmma(0)
            T.mbarrier_wait_parity(barriers[1], (block_m // (2 * split_m) - 1) % 2)
            T.sync_threads()
            issue_prefix_score(
                k_tile,
                q_tiles[1, 0 : split_m * heads, 0:dim],
                next_acc,
            )
            reduce_visible_score(
                first_acc,
                scratch,
                0,
                visible_counts,
                minimum_visible,
                bid,
                row_start + block_m - 2 * split_m,
                key_start,
            )
            T.wait_sqmma(0)
            reduce_visible_score(
                next_acc,
                scratch,
                split_m,
                visible_counts,
                minimum_visible,
                bid,
                row_start + block_m - split_m,
                key_start,
            )
            T.sync_threads()
            for group in T.unroll(store_batch * split_m // 4):
                m = group * 4 + tid // 32
                dense_index = (block_m - 2 * split_m + m) * pitch4 + tid % 32
                T.assume(dense_index >= 0)
                T.assume(dense_index < (block_m - 1) * pitch4 + block_n // 4)
                for lane in T.vectorized(4):
                    output_vectors[dense_index, lane] = (
                        scratch[m, (tid % 32) * 4 + lane] * scale
                    )

        q_type: Any = T.StridedTensor(
            (batch, rows_per_batch, heads, dim),
            (
                T.dynamic("q_stride_0", "int64"),
                T.dynamic("q_stride_1", "int64"),
                T.dynamic("q_stride_2", "int64"),
                1,
            ),
            dtype,
        )
        k_type: Any = T.StridedTensor(
            (batch, num_blocks, dim),
            (
                T.dynamic("k_stride_0", "int64"),
                T.dynamic("k_stride_1", "int64"),
                1,
            ),
            dtype,
        )
        visible_counts_type: Any = T.StridedTensor(
            (batch, rows_per_batch),
            tuple(
                T.dynamic(f"visible_counts_stride_{axis}", "int64") for axis in range(2)
            ),
            T.int32,
        )

        @T.prim_func
        def qsa_mqa_score_split_m(
            q: q_type,
            k: k_type,
            visible_counts: visible_counts_type,
            output: output_type,
            key_start_offset: T.int32,
            key_blocks: T.int32,
            row_start_offset: T.int32,
            row_count: T.int32,
        ):
            row_tiles = T.ceildiv(row_count, block_m)
            cta_row_tiles = row_tiles * cta_row_parts
            with T.Kernel(
                T.ceildiv(key_blocks, block_n * group_n)
                if score_epilogue_overlap
                else cta_row_tiles,
                cta_row_tiles
                if score_epilogue_overlap
                else T.ceildiv(key_blocks, block_n * group_n),
                batch,
                threads=128,
            ) as (grid_x, grid_y, bid):
                # Wrappers validate aligned vector rows; strides remain runtime values.
                T.assume(row_start_offset >= 0)
                T.assume(row_count > 0)
                T.assume(row_start_offset <= rows_per_batch - row_count)
                T.assume(key_start_offset >= 0)
                T.assume(key_blocks > 0)
                T.assume(key_start_offset <= num_blocks - key_blocks)
                T.assume(q.shape[0] <= 1 or q.strides[0] % 8 == 0)
                T.assume(q.shape[1] <= 1 or q.strides[1] % 8 == 0)
                T.assume(q.shape[2] <= 1 or q.strides[2] % 8 == 0)
                T.assume(k.shape[0] <= 1 or k.strides[0] % 8 == 0)
                T.assume(k.shape[-2] <= 1 or k.strides[-2] % 8 == 0)
                row_block = grid_y if score_epilogue_overlap else grid_x
                key_group = grid_x if score_epilogue_overlap else grid_y
                q_shared = T.alloc_shared((2, split_m, heads, dim), dtype)
                q_matrix = T.reshape(q_shared, (2, split_m * heads, dim))
                k_shared = T.alloc_shared((block_n, dim), dtype)
                T.annotate_layout(
                    {
                        q_matrix: tilelang.layout.make_sqmma_swizzled_layout(
                            q_matrix, k_major=True
                        ),
                        k_shared: tilelang.layout.make_sqmma_swizzled_layout(
                            k_shared, k_major=True
                        ),
                    },
                    allow_reannotation=True,
                )
                # Resident K is loaded once (group_n == 1). Q transfers
                # have an independent two-ID ring, not K's phase cursor.
                k_load_barrier = T.alloc_barrier(128)
                q_load_barrier = T.alloc_barrier([128, 128])
                if fast_visible_mask:
                    visible_min_shared = T.alloc_shared((8,), T.int32)
                scores = T.alloc_fragment((block_n, split_m * heads), accum_dtype)
                if score_epilogue_overlap:
                    scores_alt = T.alloc_fragment(
                        (block_n, split_m * heads), accum_dtype
                    )
                    T.annotate_layout(
                        {
                            scores_alt: tilelang.Fragment(
                                (block_n, split_m * heads),
                                forward_fn=lambda n, mh: (
                                    (n % 16) * 8 + mh % 8,
                                    (n // 16) * 4 + mh // 8,
                                ),
                            ),
                        },
                        allow_reannotation=True,
                    )
                T.annotate_layout(
                    {
                        scores: tilelang.Fragment(
                            (block_n, split_m * heads),
                            forward_fn=lambda n, mh: (
                                (n % 16) * 8 + mh % 8,
                                (n // 16) * 4 + mh // 8,
                            ),
                        ),
                    },
                    allow_reannotation=True,
                )
                contiguous_store = T.alloc_shared(
                    (store_batch * split_m, block_n), accum_dtype
                )
                T.annotate_layout(
                    {
                        contiguous_store: tilelang.Layout(
                            (store_batch * split_m, block_n), lambda m, n: (m, n)
                        )
                    }
                )
                row_start = row_start_offset + row_block * cta_block_m
                for group_idx in T.serial(group_n):
                    key_start = (
                        key_start_offset + (key_group * group_n + group_idx) * block_n
                    )
                    k_tme = T.make_tensor_from_addr(
                        k.data,
                        k.shape,
                        dtype,
                        strides=(
                            T.if_then_else(
                                batch <= 1, T.int64(num_blocks) * dim, k.strides[0]
                            ),
                            T.if_then_else(num_blocks <= 1, T.int64(dim), k.strides[1]),
                            1,
                        ),
                    )
                    T.tma_copy(
                        k_tme[bid, key_start : key_start + block_n, 0:dim],
                        k_shared,
                        barrier=k_load_barrier,
                    )
                    T.barrier_arrive(k_load_barrier)
                    T.mbarrier_wait_parity(k_load_barrier, 0)
                    if single_wg_tme_lookahead and fast_visible_mask:
                        # Converge the resident K wait before visibility loads.
                        # Keep the later sync that publishes visible_min_shared.
                        T.sync_threads()
                    if fast_visible_mask:
                        visible_rows = T.alloc_fragment((block_m,), T.int32)
                        minimum = T.alloc_fragment((1,), T.int32)
                        for r in T.Parallel(block_m):
                            visible_rows[r] = T.if_then_else(
                                row_start + r < rows_per_batch,
                                visible_counts[
                                    bid, T.min(row_start + r, rows_per_batch - 1)
                                ],
                                2147483647,
                            )
                        T.reduce_min(visible_rows, minimum, dim=0, clear=True)
                        T.copy(minimum, visible_min_shared[0:1])
                    T.sync_threads()
                    row_subtiles = (
                        T.min(
                            cta_block_m // split_m,
                            T.ceildiv(
                                row_start_offset + row_count - row_start, split_m
                            ),
                        )
                        if bound_tail_rows
                        else cta_block_m // split_m
                    )
                    if row_subtiles > 0:
                        preload_q_lookahead(
                            q,
                            q_shared,
                            q_load_barrier,
                            0,
                            bid,
                            row_start,
                            0,
                        )
                    # Only the proven dense store uses the overlapped epilogue.
                    # Sliced outputs keep the same executable and regular two-bank loop.
                    if score_epilogue_overlap and (
                        output.strides[2] == 1
                        and output.strides[1] == num_blocks
                        and output.strides[0] == T.int64(rows_per_batch) * num_blocks
                        and T.reinterpret(T.address_of(output[0, 0, 0]), T.uint64) % 16
                        == 0
                        and row_start + block_m <= rows_per_batch
                        and key_start + block_n <= num_blocks
                        and num_blocks % 4 == 0
                        and key_start % 4 == 0
                        and num_blocks <= 2147483647 // block_m
                    ):
                        run_score_overlap(
                            q,
                            k_shared,
                            q_matrix,
                            q_shared,
                            q_load_barrier,
                            scores,
                            scores_alt,
                            contiguous_store,
                            output,
                            visible_counts,
                            visible_min_shared[0] if fast_visible_mask else T.int32(-1),
                            bid,
                            row_start,
                            key_start,
                        )
                    else:
                        # A single expanded slot retains CTA-uniform alternating bank branches.
                        for sub_pair in T.Pipelined(
                            row_subtiles // bank_pair_width,
                            num_stages=0
                            if single_wg_tme_lookahead
                            else mainloop_stages,
                        ):
                            for bank_slot in T.unroll(bank_pair_width, explicit=True):
                                sub = sub_pair * bank_pair_width + bank_slot
                                current_bank = (
                                    bank_slot if bank_pair_width == 2 else sub % 2
                                )
                                T.mbarrier_wait_parity(
                                    q_load_barrier[current_bank], (sub // 2) % 2
                                )
                                T.sync_threads()
                                # Publication above consumes the current ready phase.
                                # The opposite bank's preceding SQMMA has completed
                                # before this slot; no active reader aliases its TME.
                                if sub + 1 < row_subtiles:
                                    preload_q_lookahead(
                                        q,
                                        q_shared,
                                        q_load_barrier,
                                        (sub + 1) % 2,
                                        bid,
                                        row_start,
                                        sub + 1,
                                    )
                                T.gemm(
                                    k_shared,
                                    q_matrix[
                                        sub_pair % 2,
                                        0 : split_m * heads,
                                        0:dim,
                                    ],
                                    scores,
                                    transpose_B=True,
                                    clear_accum=True,
                                    policy=T.GemmWarpPolicy.FullCol,
                                    wg_wait=-1,
                                )
                                T.wait_sqmma(0)
                                sub_row_start = row_start + sub * split_m
                                reduce_visible_score(
                                    scores,
                                    contiguous_store,
                                    (sub % store_batch) * split_m,
                                    visible_counts,
                                    visible_min_shared[0]
                                    if fast_visible_mask
                                    else T.int32(-1),
                                    bid,
                                    sub_row_start,
                                    key_start,
                                )
                                if (sub + 1) % store_batch == 0:
                                    T.sync_threads()
                                    store_scores(
                                        output,
                                        contiguous_store,
                                        bid,
                                        sub_row_start - (store_batch - 1) * split_m,
                                        key_start,
                                        store_batch * split_m,
                                    )
                                elif bound_tail_rows and sub + 1 == row_subtiles:
                                    # An odd final subtile flushes only its
                                    # initialized half; arbitrary row tails
                                    # must not read stale shared values.
                                    T.sync_threads()
                                    store_scores(
                                        output,
                                        contiguous_store,
                                        bid,
                                        sub_row_start,
                                        key_start,
                                        split_m,
                                    )

        return qsa_mqa_score_split_m

    return kernel_factory()


@lru_cache(maxsize=None)
def _qsa_mqa_prefill_kernel(
    block_n: int,
    block_m: int,
    scale: float,
    use_tme: bool = True,
    dtype_name: str = "bfloat16",
):
    """Build a packed-row MQA matrix kernel.

    Logical row/key lengths and outer strides stay dynamic. TME zero-fills
    incomplete tiles; output stores cover only the caller's logical shape.
    """

    if tilelang is None:
        raise ModuleNotFoundError(
            "QSA kernels require a working TileLang-MUSA installation.", name="tilelang"
        ) from _tilelang_import_error

    if block_n <= 0 or block_n % 16 or block_m <= 0 or block_m % 2:
        raise ValueError(
            "block_n must be a multiple of 16 and block_m a multiple of 2."
        )
    heads, dim = 4, 128
    dtype = dtype_name
    accum_dtype = T.float32
    rows = T.dynamic("rows")
    num_keys = T.dynamic("num_keys")

    @tilelang.jit(
        target="musa",
        out_idx=[4],
        pass_configs=_pass_configs(tilelang, enable_tme=use_tme),
    )
    def kernel_factory():
        q_type: Any = T.StridedTensor(
            (rows, heads, dim),
            (
                T.dynamic("q_stride_0", "int64"),
                T.dynamic("q_stride_1", "int64"),
                1,
            ),
            dtype,
        )
        k_type: Any = T.StridedTensor(
            (num_keys, dim),
            (
                T.dynamic("k_stride_0", "int64"),
                1,
            ),
            dtype,
        )
        row_starts_type: Any = T.StridedTensor(
            (rows,), (T.dynamic("row_starts_stride_0", "int64"),), T.int32
        )
        row_ends_type: Any = T.StridedTensor(
            (rows,), (T.dynamic("row_ends_stride_0", "int64"),), T.int32
        )
        output_type: Any = T.StridedTensor(
            (rows, num_keys),
            tuple(T.dynamic(f"output_stride_{axis}", "int64") for axis in range(2)),
            accum_dtype,
        )

        @T.prim_func
        def qsa_mqa_prefill(
            q: q_type,
            k: k_type,
            row_starts: row_starts_type,
            row_ends: row_ends_type,
            output: output_type,
        ):
            with T.Kernel(
                T.ceildiv(rows, block_m),
                T.ceildiv(num_keys, block_n),
                threads=128,
            ) as (row_block, tile_id):
                # Wrappers validate aligned vector rows; strides remain runtime values.
                T.assume(q.shape[0] <= 1 or q.strides[0] % 8 == 0)
                T.assume(q.shape[-2] <= 1 or q.strides[-2] % 8 == 0)
                T.assume(k.shape[0] <= 1 or k.strides[0] % 8 == 0)
                T.assume(k.shape[-2] <= 1 or k.strides[-2] % 8 == 0)
                q_shared = T.alloc_shared((block_m, heads, dim), dtype)
                q_matrix = T.reshape(q_shared, (block_m * heads, dim))
                k_shared = T.alloc_shared((block_n, dim), dtype)
                scores = T.alloc_fragment((block_n, block_m * heads), accum_dtype)
                scores_3d = T.reshape(scores, (block_n, block_m, heads))
                reduced = T.alloc_fragment((block_n, block_m), accum_dtype)

                row_start = row_block * block_m
                key_start = tile_id * block_n
                if use_tme:
                    load_barrier = T.alloc_barrier(128)
                    q_tme = T.make_tensor_from_addr(
                        q.data,
                        q.shape,
                        dtype,
                        strides=(
                            T.if_then_else(
                                rows <= 1, T.int64(heads * dim), q.strides[0]
                            ),
                            q.strides[1],
                            1,
                        ),
                    )
                    k_tme = T.make_tensor_from_addr(
                        k.data,
                        k.shape,
                        dtype,
                        strides=(
                            T.if_then_else(num_keys <= 1, T.int64(dim), k.strides[0]),
                            1,
                        ),
                    )
                    T.tma_copy(
                        q_tme[
                            row_start : row_start + block_m,
                            0:heads,
                            0:dim,
                        ],
                        q_shared,
                        barrier=load_barrier,
                    )
                    T.tma_copy(
                        k_tme[key_start : key_start + block_n, 0:dim],
                        k_shared,
                        barrier=load_barrier,
                    )
                    T.barrier_arrive(load_barrier)
                    T.mbarrier_wait_parity(load_barrier, 0)
                else:
                    T.copy(
                        q[
                            row_start : row_start + block_m,
                            0:heads,
                            0:dim,
                        ],
                        q_shared,
                        disable_tma=True,
                    )
                    T.copy(
                        k[key_start : key_start + block_n, 0:dim],
                        k_shared,
                        disable_tma=True,
                    )
                T.clear(scores)
                T.gemm(
                    k_shared,
                    q_matrix,
                    scores,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for n, m, h in T.Parallel(block_n, block_m, heads):
                    scores_3d[n, m, h] = T.max(scores_3d[n, m, h], 0.0)
                T.reduce_sum(scores_3d, reduced, dim=2, clear=True)
                for n, m in T.Parallel(block_n, block_m):
                    row = row_start + m
                    logical_key = key_start + n
                    if row < rows and logical_key < num_keys:
                        output[row, logical_key] = T.if_then_else(
                            (logical_key >= row_starts[row])
                            and (logical_key < row_ends[row]),
                            reduced[n, m] * scale,
                            -T.infinity(accum_dtype),
                        )

        return qsa_mqa_prefill

    return kernel_factory()


@lru_cache(maxsize=None)
def _qsa_mqa_decode_kernel(
    page_size: int,
    group_size: int,
    scale: float,
    dtype_name: str = "bfloat16",
    use_tme_q: bool = True,
):
    if tilelang is None:
        raise ModuleNotFoundError(
            "QSA kernels require a working TileLang-MUSA installation.", name="tilelang"
        ) from _tilelang_import_error

    if page_size not in (8, 16, 32, 64):
        raise ValueError("QSA MQA decode page_size must be 8/16/32/64.")
    if group_size <= 0 or group_size % page_size:
        raise ValueError("group_size must be positive and divisible by page_size.")
    dtype = dtype_name
    accum_dtype = T.float32
    batch = T.dynamic("batch")
    num_pages = T.dynamic("num_pages")
    max_pages = T.dynamic("max_pages")
    output_width = T.dynamic("output_width")
    # Keep paged decode on N16; the N8 short-cache specialization can cause
    # excessive compilation with the MP31 MCC 5.2 toolchain.
    heads = 16
    dim = 128
    pages_per_group = group_size // page_size

    @tilelang.jit(
        target="musa",
        out_idx=[4],
        # TME accepts aligned unit-feature-stride Q; sliced features use LSU.
        # Both layouts retain runtime outer strides and logical dimensions.
        pass_configs=_pass_configs(tilelang, enable_tme=use_tme_q),
    )
    def kernel_factory():
        q_type: Any = T.StridedTensor(
            (batch, 4, dim),
            (
                T.dynamic("q_stride_0", "int64"),
                T.dynamic("q_stride_1", "int64"),
                1 if use_tme_q else T.dynamic("q_stride_2", "int64"),
            ),
            dtype,
        )
        k_cache_type: Any = T.StridedTensor(
            (num_pages, page_size, 1, dim),
            tuple(T.dynamic(f"k_cache_stride_{axis}", "int64") for axis in range(4)),
            dtype,
        )
        page_table_type: Any = T.StridedTensor(
            (batch, max_pages),
            tuple(T.dynamic(f"page_table_stride_{axis}", "int64") for axis in range(2)),
            T.int32,
        )
        context_lens_type: Any = T.StridedTensor(
            (batch,), (T.dynamic("context_lens_stride_0", "int64"),), T.int32
        )
        output_type: Any = T.StridedTensor(
            (batch, output_width),
            tuple(T.dynamic(f"output_stride_{axis}", "int64") for axis in range(2)),
            accum_dtype,
        )

        @T.prim_func
        def qsa_mqa_decode(
            q: q_type,
            k_cache: k_cache_type,
            page_table: page_table_type,
            context_lens: context_lens_type,
            output: output_type,
        ):
            with T.Kernel(
                batch,
                T.ceildiv(output_width, group_size),
                threads=128,
            ) as (bid, group_id):
                q_shared = T.alloc_shared((heads, dim), dtype)
                k_shared = T.alloc_shared((group_size, dim), dtype)
                scores = T.alloc_fragment((group_size, heads), accum_dtype)
                reduced = T.alloc_fragment((group_size,), accum_dtype)
                T.clear(q_shared)
                T.sync_threads()
                if use_tme_q:
                    q_barrier = T.alloc_barrier(128)
                    q_tme = T.make_tensor_from_addr(
                        q.data,
                        q.shape,
                        dtype,
                        strides=(
                            T.if_then_else(batch <= 1, T.int64(4 * dim), q.strides[0]),
                            q.strides[1],
                            q.strides[2],
                        ),
                    )
                    T.tma_copy(
                        q_tme[bid, 0:4, 0:dim],
                        q_shared[0:4, 0:dim],
                        barrier=q_barrier,
                    )
                    T.barrier_arrive(q_barrier)
                    T.mbarrier_wait_parity(q_barrier, 0)
                else:
                    T.copy(q[bid, 0:4, 0:dim], q_shared[0:4, 0:dim], disable_tma=True)
                T.sync_threads()
                context_len = context_lens[bid]

                for sub_page in range(pages_per_group):
                    logical_page = group_id * pages_per_group + sub_page
                    page_valid = (logical_page < max_pages) and (
                        logical_page * page_size < context_len
                    )
                    safe_page_index = T.if_then_else(
                        logical_page < max_pages, logical_page, 0
                    )
                    table_page = page_table[bid, safe_page_index]
                    safe_page = T.if_then_else(
                        page_valid and (table_page >= 0) and (table_page < num_pages),
                        table_page,
                        0,
                    )
                    T.copy(
                        k_cache[safe_page, 0:page_size, 0:1, 0:dim],
                        k_shared[
                            sub_page * page_size : (sub_page + 1) * page_size,
                            0:dim,
                        ],
                        disable_tma=True,
                    )

                T.clear(scores)
                T.gemm(
                    k_shared,
                    q_shared,
                    scores,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                for n, h in T.Parallel(group_size, heads):
                    scores[n, h] = T.if_then_else(h < 4, T.max(scores[n, h], 0.0), 0.0)
                T.reduce_sum(scores, reduced, dim=1, clear=True)
                for n in T.Parallel(group_size):
                    logical_token = group_id * group_size + n
                    if logical_token < output_width:
                        logical_page = logical_token // page_size
                        table_page = page_table[bid, T.min(logical_page, max_pages - 1)]
                        output[bid, logical_token] = T.if_then_else(
                            (logical_token < context_len)
                            and (logical_page < max_pages)
                            and (table_page >= 0)
                            and (table_page < num_pages),
                            reduced[n] * scale,
                            -T.infinity(accum_dtype),
                        )

        return qsa_mqa_decode

    return kernel_factory()
