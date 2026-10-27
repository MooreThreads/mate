"""Fused Q preparation and O epilogue for the FlashMLA fused-module ABI."""

import torch
import tilelang
from tilelang import language as T

from mate.execution_context import skip_kernel_launch_if_dry_run


def output_scale_exponent(amax):
    """E8M0 exponent of ceil_pow2(max(amax, 1e-4) / 448).

    Rounding the positive float's mantissa upward is exact at powers of two.
    Its reciprocal has exponent 254 - e, avoiding per-element float division.
    """
    bits = T.reinterpret("int32", T.max(amax, 1e-4) * (1.0 / 448.0))
    return (bits + 0x7FFFFF) >> 23


def allocate_output_scales(rows, heads, groups, device):
    """Packed column-major E8M0 words and their byte-addressable storage."""
    aligned = (rows + 3) // 4 * 4
    storage = torch.empty(
        (groups, heads // groups * 4, aligned), dtype=torch.int32, device=device
    )
    return storage.permute(2, 0, 1)[:rows], storage.view(torch.uint8).flatten()


def make_output_epilogue(heads, groups, tile_heads, width=128):
    """Store a normalized FP32 SQMMA fragment directly in the Wv FP8 layout.

    Each group of 32 channels owns one scale byte. Byte stores give distinct
    heads exclusive addresses even when their scales share an int32 word.
    Neither intermediate BF16 O nor a cross-consumer shared buffer is needed.
    """
    group_heads = heads // groups

    @T.macro
    def store(acc, row, head_start, col_start, output, sf, positions, cos_sin):
        output_flat = T.view(output, [positions.shape[0] * heads * 512], "float8_e4m3")
        values = T.alloc_fragment((tile_heads, width // 32, 32), "float32")
        inv_scales = T.alloc_fragment((tile_heads, width // 32), "float32")
        for h, g, c in T.Parallel(tile_heads, width // 32, 32):
            values[h, g, c] = acc[h, g * 32 + c]
        if col_start + width > 448:
            for h, g, c in T.Parallel(tile_heads, width // 32, 32):
                col = col_start + g * 32 + c
                if col >= 448:
                    cosine = cos_sin[positions[row], (col - 448) // 2]
                    sine = cos_sin[positions[row], 32 + (col - 448) // 2]
                    values[h, g, c] = values[h, g, c] * cosine + T.shfl_xor(
                        values[h, g, c], 1
                    ) * T.if_then_else(c % 2 == 0, sine, -sine)
        T.reduce_absmax(values, inv_scales, dim=2)
        for h, g in T.Parallel(tile_heads, width // 32):
            exponent = output_scale_exponent(inv_scales[h, g])
            inv_scales[h, g] = T.reinterpret("float32", (254 - exponent) << 23)
            # FullRow fragments replicate the reduced scale over eight lanes.
            # Only their leader writes the byte; adjacent heads own other bytes.
            if head_start + h < heads and T.get_thread_binding() % 8 == 0:
                head = head_start + h
                chunk = (col_start // 32 + g) * group_heads + head % group_heads
                aligned_rows = T.ceildiv(positions.shape[0], 4) * 4
                byte = (
                    (head // group_heads * (group_heads * 4) + chunk // 4)
                    * aligned_rows
                    + row
                ) * 4 + chunk % 4
                sf[byte] = T.Cast("uint8", exponent)
        for h, g, c in T.Parallel(tile_heads, width // 32, 32):
            if head_start + h < heads:
                head = head_start + h
                chunk = (col_start // 32 + g) * group_heads + head % group_heads
                output_flat[
                    row * heads * 512
                    + head // group_heads * group_heads * 512
                    + chunk * 32
                    + c
                ] = T.Cast("float8_e4m3", values[h, g, c] * inv_scales[h, g])

    return store


@tilelang.jit(out_idx=[])
def _prepare_q(heads: int, norm: bool):
    rows, positions = T.dynamic("rows"), T.dynamic("positions")
    stride = T.dynamic("stride")

    @T.prim_func
    def prepare(
        q: T.StridedTensor[(rows, heads * 512), (stride, 1), T.bfloat16],
        pos: T.Tensor[(rows,), T.int32],
        cos_sin: T.Tensor[(positions, 64), T.float32],
        output: T.Tensor[(rows, heads, 512), T.bfloat16],
        eps: T.float32,
    ):
        with T.Kernel(rows, heads, threads=128) as (row, head):
            x = T.alloc_fragment((512,), "float32")
            square = T.alloc_fragment((512,), "float32")
            inv = T.alloc_fragment((1,), "float32")
            for c in T.Parallel(512):
                x[c] = T.Cast(
                    "float32", q[row, c // 16 * heads * 16 + head * 16 + c % 16]
                )
                square[c] = x[c] * x[c]
            if norm:
                T.reduce_sum(square, inv, dim=0)
                inv[0] = T.rsqrt(inv[0] / 512.0 + eps)
            else:
                inv[0] = 1.0
            for c in T.Parallel(512):
                if c < 448:
                    output[row, head, c] = T.Cast("bfloat16", x[c] * inv[0])
                else:
                    partner = q[
                        row, (c ^ 1) // 16 * heads * 16 + head * 16 + (c ^ 1) % 16
                    ]
                    cosine = cos_sin[pos[row], (c - 448) // 2]
                    sine = cos_sin[pos[row], 32 + (c - 448) // 2]
                    rotated = x[c] * cosine + T.Cast(
                        "float32", partner
                    ) * T.if_then_else(c % 2 == 0, -sine, sine)
                    output[row, head, c] = T.Cast("bfloat16", rotated * inv[0])

    return prepare


@tilelang.jit(out_idx=[])
def _finish_o(heads: int, groups: int):
    rows, positions = T.dynamic("rows"), T.dynamic("positions")
    aligned_rows = T.dynamic("aligned_rows")
    group_heads = heads // groups
    packed_cols = group_heads * 4

    @T.prim_func
    def finish(
        out: T.Tensor[(rows, heads, 512), T.bfloat16],
        pos: T.Tensor[(rows,), T.int32],
        cos_sin: T.Tensor[(positions, 64), T.float32],
        quant: T.Tensor[(rows, groups, group_heads * 512), T.float8_e4m3],
        sf: T.StridedTensor[
            (rows, groups, packed_cols),
            (1, aligned_rows * packed_cols, aligned_rows),
            T.int32,
        ],
    ):
        # Four adjacent output scale groups produce one packed int32 word.
        with T.Kernel(rows, groups, packed_cols, threads=128) as (row, group, word):
            x = T.alloc_fragment((4, 32), "float32")
            magnitude = T.alloc_fragment((4, 32), "float32")
            scale = T.alloc_fragment((4,), "float32")
            exponent = T.alloc_shared((4,), "int32")
            for g, c in T.Parallel(4, 32):
                chunk = word * 4 + g
                head = group * group_heads + chunk % group_heads
                col = chunk // group_heads * 32 + c
                value = T.alloc_var("float32")
                value = T.Cast("float32", out[row, head, col])
                if col >= 448:
                    partner = T.Cast("float32", out[row, head, col ^ 1])
                    cosine = cos_sin[pos[row], (col - 448) // 2]
                    sine = cos_sin[pos[row], 32 + (col - 448) // 2]
                    value = value * cosine + partner * T.if_then_else(
                        col % 2 == 0, sine, -sine
                    )
                x[g, c] = value
                magnitude[g, c] = T.abs(value)
            T.reduce_max(magnitude, scale, dim=1)
            for g in T.Parallel(4):
                scale[g] = T.exp2(T.ceil(T.log2((T.max(scale[g], 1e-4) / 448.0))))
                exponent[g] = (T.reinterpret("int32", scale[g]) >> 23) & 255
            for g, c in T.Parallel(4, 32):
                quant[row, group, word * 128 + g * 32 + c] = T.Cast(
                    "float8_e4m3", x[g, c] / scale[g]
                )
            T.sync_threads()
            if T.get_thread_binding() == 0:
                sf[row, group, word] = (
                    exponent[0]
                    | (exponent[1] << 8)
                    | (exponent[2] << 16)
                    | (exponent[3] << 24)
                )

    return finish


def prepare_q(q, positions, cos_sin, enable_norm, eps):
    rows, heads, _ = q.shape
    result = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=q.device)
    kernel = _prepare_q(heads, enable_norm)
    if not skip_kernel_launch_if_dry_run():
        kernel(q.view(rows, heads * 512), positions, cos_sin, result, float(eps))
    return result


def finish_o(out, positions, cos_sin, groups):
    rows, heads, _ = out.shape
    cols = heads // groups * 512
    quant = torch.empty(
        (rows, groups, cols), dtype=torch.float8_e4m3fn, device=out.device
    )
    padded = (rows + 3) // 4 * 4
    sf = torch.empty_strided(
        (rows, groups, cols // 128),
        (1, padded * (cols // 128), padded),
        dtype=torch.int32,
        device=out.device,
    )
    kernel = _finish_o(heads, groups)
    if not skip_kernel_launch_if_dry_run():
        kernel(out, positions, cos_sin, quant, sf)
    return quant, sf
