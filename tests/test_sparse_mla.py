import pdb  # noqa: F401
import time  # noqa: F401
import math  # noqa: F401
import torch
import pytest
import tilelang.testing
from typing import Optional, Tuple  # noqa: F401
import random
import enum
from typing import List
# torch.random.manual_seed(42)


class FP8KVCacheLayout(enum.Enum):
    V32_FP8Sparse = 1
    MODEL1_FP8Sparse = 2

    def get_meta(self) -> Tuple[int, int, int, int, int]:
        # Return: (d, d_nope, d_rope, tile_size, num_tiles)
        return {
            FP8KVCacheLayout.V32_FP8Sparse: (576, 512, 64, 128, 4),
            FP8KVCacheLayout.MODEL1_FP8Sparse: (512, 448, 64, 64, 7),
        }[self]


def _cast_scale_inv_to_ue8m0(
    scales_inv: torch.Tensor, out_dtype=torch.float32
) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(out_dtype)


def quantize_k_cache(
    input_k_cache: torch.Tensor,  # (num_blocks, block_size, h_k, d)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    Quantize the k-cache
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)  # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        bytes_per_token = d_nope + num_tiles * 4 + input_elem_size * d_rope
        result = torch.empty(
            (num_blocks, block_size + 1, bytes_per_token),
            dtype=torch.float8_e4m3fn,
            device=input_k_cache.device,
        )[:, :block_size, :]
        result_k_nope_part = result[..., :d_nope]
        result_k_scale_factor = result[..., d_nope : d_nope + num_tiles * 4].view(
            torch.float32
        )
        result_k_rope_part = result[..., d_nope + num_tiles * 4 :].view(
            input_k_cache.dtype
        )
        result_k_rope_part[:] = input_k_cache[..., d_nope:]

        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = (
                torch.abs(
                    input_k_cache[
                        ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                    ]
                )
                .max(dim=-1)
                .values.float()
                / 448.0
            )  # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

            cur_scale_factors_inv.unsqueeze_(-1)  # [num_blocks, block_size, 1]
            cur_quantized_nope = (
                input_k_cache[
                    ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                / cur_scale_factors_inv.float()
            ).to(torch.float8_e4m3fn)
            result_k_nope_part[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ] = cur_quantized_nope

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        bytes_per_token = d_nope + 2 * d_rope + num_tiles + 1
        size_per_block_padded = (block_size * bytes_per_token + 576 - 1) // 576 * 576
        result = torch.empty(
            (num_blocks, size_per_block_padded),
            dtype=torch.float8_e4m3fn,
            device=input_k_cache.device,
        )[:, : block_size * bytes_per_token]
        result_k_nope_rope_part = result[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        result_k_nope = result_k_nope_rope_part[
            :, :, :d_nope
        ]  # [num_blocks, block_size, d_nope]
        result_k_rope = result_k_nope_rope_part[:, :, d_nope:].view(
            input_k_cache.dtype
        )  # [num_blocks, block_size, d_rope]
        result_k_scale_factor = (
            result[:, block_size * (d_nope + 2 * d_rope) :]
            .view(num_blocks, block_size, 8)[:, :, :7]
            .view(torch.float8_e8m0fnu)
        )  # [num_blocks, block_size, num_tiles]

        result_k_rope[:] = input_k_cache[..., d_nope:]
        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = (
                torch.abs(
                    input_k_cache[
                        ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                    ]
                )
                .max(dim=-1)
                .values.float()
                / 448.0
            )  # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv.to(
                torch.float8_e8m0fnu
            )

            cur_scale_factors_inv = cur_scale_factors_inv.view(
                num_blocks, block_size, 1
            )
            cur_quantized_nope = (
                input_k_cache[
                    ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                / cur_scale_factors_inv.float()
            ).to(torch.float8_e4m3fn)
            result_k_nope[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_quantized_nope
            )

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,  # (num_blocks, block_size, 1, bytes_per_token)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device
    )

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

        input_nope = quant_k_cache[..., :d_nope]
        input_scale = quant_k_cache[..., d_nope : d_nope + num_tiles * 4].view(
            torch.float32
        )
        input_rope = quant_k_cache[..., d_nope + num_tiles * 4 :].view(torch.bfloat16)
        result[..., d_nope:] = input_rope

        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.float32)
            cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
            result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            )

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, -1)  # [num_blocks, ...]
        input_nope_rope = quant_k_cache[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        input_nope = input_nope_rope[:, :, :d_nope]
        input_rope = input_nope_rope[:, :, d_nope:].view(torch.bfloat16)
        input_scale = (
            quant_k_cache[:, block_size * (d_nope + 2 * d_rope) :]
            .view(num_blocks, block_size, 8)[:, :, :7]
            .view(torch.float8_e8m0fnu)
        )  # [num_blocks, block_size, num_tiles]

        result[..., d_nope:] = input_rope
        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.bfloat16)
            cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
            result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            )

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")

    result = result.view(num_blocks, block_size, 1, d)
    return result


def abs_indices2indices_in_kvcache(
    abs_indices: torch.Tensor,  # [b, s_q, topk]
    block_table: torch.Tensor,  # [b, /]
    block_size: int,
) -> torch.Tensor:
    """
    Convert abs_indices (logical index, ranging from 0 to s_k-1) to index expected by the sparse attn kernel
    Equivalent to:

    b, s_q, topk = abs_indices.shape
    indices_in_kvcache = torch.empty_like(abs_indices)
    for i in range(b):
        cur_abs_indices = abs_indices[i, :, :].clone()  # [s_q, topk]
        invalid_mask = cur_abs_indices == -1
        cur_abs_indices[invalid_mask] = 0
        cur_indices_in_kvcache = block_table[i].index_select(0, cur_abs_indices.flatten()//block_size).view(s_q, topk)*block_size + cur_abs_indices%block_size
        cur_indices_in_kvcache[invalid_mask] = -1
        indices_in_kvcache[i] = cur_indices_in_kvcache
    return indices_in_kvcache

    """
    b, s_q, topk = abs_indices.shape
    _, max_blocks_per_seq = block_table.shape

    abs_indices = abs_indices.clone()
    invalid_mask = abs_indices == -1
    abs_indices[invalid_mask] = 0

    real_block_idxs = block_table.view(-1).index_select(
        0,
        (
            abs_indices // block_size
            + torch.arange(0, b).view(b, 1, 1) * max_blocks_per_seq
        ).view(-1),
    )
    indices_in_kvcache = (
        real_block_idxs.view(b, s_q, topk) * block_size + abs_indices % block_size
    )
    indices_in_kvcache[invalid_mask] = -1

    return indices_in_kvcache


def check_is_bitwise_equal_comparator(
    ans: torch.Tensor, ref: torch.Tensor, result: torch.Tensor
):
    """
    Return if two tensors are bitwise equal
    Return a bool if avoid_sync is False, else return a tensor
    """
    assert ans.shape == ref.shape, "Shape mismatch"
    torch.all(torch.eq(ans, ref), out=result)


def check_is_bitwise_equal(
    name: str, ans: torch.Tensor, ref: torch.Tensor, quiet: bool = False
) -> bool:
    is_bitwise_equal = torch.equal(ans, ref)
    if not quiet and not is_bitwise_equal:
        print(
            f"`{name}` mismatch: not bitwise equal. Mismatch count: {(ans != ref).sum().item()} out of {ans.numel()}"
        )
    return is_bitwise_equal


def get_cos_diff(ans: torch.Tensor, ref: torch.Tensor) -> float:
    """
    Calculate the cosine diff between two tensors
    Return a float if avoid_sync is False, else return a tensor
    """
    ans, ref = ans.double(), ref.double()
    if (ref * ref).sum().item() < 1e-12:
        return 0
    denominator = (ans * ans + ref * ref).sum().item()
    sim = 2 * (ans * ref).sum().item() / denominator
    return 1 - sim


def check_is_allclose(
    name: str,
    ans: torch.Tensor,
    ref: torch.Tensor,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-2,
    cos_diff_tol: float = 1e-7,
    quiet: bool = False,
) -> bool:
    """
    Check if two tensors are close enough
    Return a bool if avoid_sync is False, else return a tensor
    """
    assert ans.shape == ref.shape, (
        f"`{name}` Shape mismatch: {ans.shape} vs {ref.shape}"
    )
    assert ans.dtype == ref.dtype, (
        f"`{name}` Dtype mismatch: {ans.dtype} vs {ref.dtype}"
    )

    ans = ans.clone().to(torch.float)
    ref = ref.clone().to(torch.float)

    def report_err(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    # Deal with anomalies
    def deal_with_anomalies(val: float):
        ref_mask = (ref == val) if (val == val) else (ref != ref)
        ans_mask = (ans == val) if (val == val) else (ans != ans)
        ref[ref_mask] = 0.0
        ans[ans_mask] = 0.0
        if not torch.equal(ref_mask, ans_mask):
            report_err(
                f"`{name}` Anomaly number `{val}` mismatch: {ans_mask.sum().item()} in ans but {ref_mask.sum().item()} in ref"
            )
            return False
        return True

    anomalies_check_passed = True
    anomalies_check_passed &= deal_with_anomalies(float("inf"))
    anomalies_check_passed &= deal_with_anomalies(float("-inf"))
    anomalies_check_passed &= deal_with_anomalies(float("nan"))

    cos_diff = get_cos_diff(ans, ref)
    raw_abs_err = torch.abs(ans - ref)
    raw_rel_err = raw_abs_err / (torch.abs(ref) + (1e-6))
    rel_err = raw_rel_err.masked_fill(raw_abs_err < abs_tol, 0)
    abs_err = raw_abs_err.masked_fill(raw_rel_err < rel_tol, 0)
    pass_mask = (abs_err < abs_tol) | (rel_err < rel_tol)

    if not anomalies_check_passed:
        return False

    if not pass_mask.all():
        report_err(f"`{name}` mismatch")
        max_abs_err_pos: int = torch.argmax(abs_err, keepdim=True).item()
        max_rel_err_pos: int = torch.argmax(rel_err, keepdim=True).item()

        def get_pos_in_tensor(t: torch.Tensor, pos: int) -> List[int]:
            result = []
            for size in t.shape[::-1]:
                result.append(pos % size)
                pos = pos // size
            assert pos == 0
            return result[::-1]

        report_err(
            f"max abs err: {torch.max(abs_err).item()}: pos {get_pos_in_tensor(ans, max_abs_err_pos)}, {ans.reshape(-1)[max_abs_err_pos].item()} vs {ref.reshape(-1)[max_abs_err_pos].item()}"
        )
        report_err(
            f"max rel err: {torch.max(rel_err).item()}: pos {get_pos_in_tensor(ans, max_rel_err_pos)}, {ans.reshape(-1)[max_rel_err_pos].item()} vs {ref.reshape(-1)[max_rel_err_pos].item()}"
        )
        report_err(
            f"{pass_mask.sum()} out of {pass_mask.numel()} passed ({pass_mask.sum() / pass_mask.numel() * 100.0:.2f}%)"
        )
        report_err(f"Cosine diff: {cos_diff} (threshold: {cos_diff_tol})")
        return False
    else:
        if abs(cos_diff) > cos_diff_tol:
            report_err(
                f"`{name}` mismatch: Cosine diff too large: {cos_diff} vs {cos_diff_tol})"
            )
            return False
        return True


def check_is_allclose_comparator(
    name: str,
    ans: torch.Tensor,
    ref: torch.Tensor,
    out: torch.Tensor,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-2,
    cos_diff_tol: float = 1e-7,
):
    out.fill_(check_is_allclose(name, ans, ref, abs_tol, rel_tol, cos_diff_tol))


def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither MUSA nor CUDA is available")


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(0, 1).long()
    sq, h, dim_q = q.shape
    sk, g, _ = kv.shape

    # assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    v = kv[..., :dim]

    _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    valid_mask = (indices >= 0) & (indices < sk)
    indices_clamped = torch.where(valid_mask, indices, indices.new_full((), sk))

    kv_by_group = kv.permute(1, 0, 2).contiguous()
    v_by_group = v.permute(1, 0, 2).contiguous()
    kv_with_padding = torch.cat(
        [kv_by_group, kv_by_group.new_zeros(g_index, 1, kv.shape[-1])], dim=1
    )
    v_with_padding = torch.cat(
        [v_by_group, v_by_group.new_zeros(g_index, 1, dim_v)], dim=1
    )

    group_idx = torch.arange(g_index, device=q.device)[:, None, None]
    k_selected = kv_with_padding[group_idx, indices_clamped]
    v_selected = v_with_padding[group_idx, indices_clamped]

    q = q.view(sq, g, -1, dim_q).permute(1, 2, 0, 3).contiguous()
    score = torch.einsum("ghmd,gmnd->ghmn", q, k_selected)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.mul(sm_scale)
    score = score.masked_fill(~valid_mask[:, None, :, :], float("-inf"))
    p = score.softmax(dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    p = p.view(g_index, h_index, sq, indices.shape[-1])
    p = p.view(g, -1, sq, indices.shape[-1])
    o = torch.einsum("ghmn,gmnd->mghd", p.type(v_selected.dtype), v_selected)
    o = o.reshape(sq, h, dim_v)
    return o.to(torch.bfloat16), score


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("sq", [1, 2, 3, 8])
@pytest.mark.parametrize("skv", [65536, 1024])
@pytest.mark.parametrize(
    "heads",
    [
        128,
        64,
    ],
)
@pytest.mark.parametrize(
    "hkv",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "dqk",
    [
        576,
    ],
)
@pytest.mark.parametrize(
    "dv",
    [
        512,
    ],
)
@pytest.mark.parametrize(
    "topk",
    [
        2048,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "sm_scale",
    [
        0.1352337788608801,
        0.0625,
    ],
)
def test_dsa_decode(batch, sq, skv, heads, hkv, dqk, dv, topk, dtype, sm_scale):
    device = get_test_device()
    q = (
        torch.randn((batch, sq, heads, dqk), dtype=dtype, device=device) / 10
        + (random.random() - 0.5) / 10
    )
    pagesize = 64
    pagenum = (skv + pagesize - 1) // pagesize

    kv = (
        torch.randn((pagenum, pagesize, hkv, dqk), dtype=dtype, device=device) / 10
        + (random.random() - 0.5) / 10
    )

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((batch, sq, hkv, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        for t in range(sq):
            if random.random() < 0.8:
                for h in range(hkv):
                    i_i = torch.randperm(skv, device=device)[:topk]
                    indices[b, t, h, : len(i_i)] = i_i
    kcache = quantize_k_cache(kv, FP8KVCacheLayout.V32_FP8Sparse).contiguous()
    kv_dequant = dequantize_k_cache(kcache, FP8KVCacheLayout.V32_FP8Sparse).contiguous()

    import mate

    tile_scheduler_metadata, num_splits = mate.flashmla.get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=sq * heads // 1,
        num_heads_k=1,
        num_heads_q=heads,
        topk=topk,
        is_fp8_kvcache=True,
        q=q,
    )

    # import pdb

    # pdb.set_trace()

    tl_out, lse = mate.flashmla.flash_mla_with_kvcache(
        q=q,
        k_cache=kcache.view(pagenum, pagesize, hkv, 656),
        block_table=None,
        cache_seqlens=None,
        head_dim_v=dv,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=sm_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
    )
    torch.musa.synchronize()

    ref_out, _ = ref_sparse_mla_fwd_interface(
        q.view(batch * sq, heads, dqk),
        kv_dequant.view(pagenum * pagesize, hkv, -1),
        indices.view(batch * sq, hkv, topk),
        sm_scale=sm_scale,
    )
    is_out_correct = check_is_allclose(
        "output",
        tl_out.view(-1),
        ref_out.to(device).view(-1),
        abs_tol=1e-3,
        rel_tol=2.01 / 128,
        cos_diff_tol=5e-6,
    )
    assert is_out_correct


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("sq", [128, 256, 255])
@pytest.mark.parametrize("skv", [65536, 32768, 1024])
@pytest.mark.parametrize(
    "heads",
    [
        128,
        64,
    ],
)
@pytest.mark.parametrize(
    "hkv",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "dqk",
    [
        576,
    ],
)
@pytest.mark.parametrize(
    "dv",
    [
        512,
    ],
)
@pytest.mark.parametrize(
    "topk",
    [
        2048,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "sm_scale",
    [
        0.1352337788608801,
        0.0625,
    ],
)
def test_dsa_prefill(sq, skv, heads, hkv, dqk, dv, topk, dtype, sm_scale):
    device = get_test_device()
    q = (
        torch.randn((sq, heads, dqk), dtype=dtype, device=device) / 10
        + (random.random() - 0.5) / 10
    )
    kv = (
        torch.randn((skv, hkv, dqk), dtype=dtype, device=device) / 10
        + (random.random() - 0.5) / 10
    )

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((sq, hkv, topk), -1, dtype=torch.int32, device=device)
    for t in range(sq):
        if random.random() < 0.8:
            for h in range(hkv):
                i_i = torch.randperm(skv, device=device)[:topk]
                indices[t, h, : len(i_i)] = i_i

    import mate

    tl_out, _, _ = mate.flashmla.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=dv,
        attn_sink=None,
        topk_length=None,
    )
    torch.musa.synchronize()

    ref_out, _ = ref_sparse_mla_fwd_interface(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
    )
    is_out_correct = check_is_allclose(
        "output",
        tl_out.view(-1),
        ref_out.to(device).view(-1),
        abs_tol=1e-3,
        rel_tol=2.01 / 128,
        cos_diff_tol=5e-6,
    )
    assert is_out_correct
