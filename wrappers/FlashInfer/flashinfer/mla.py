"""Native sparse MLA compatibility on MUSA, including GLM-5.3 NoPE."""

from dataclasses import dataclass

import torch

from mate.sparse_mla.glm_nope import sparse_mla_nope_decode


@dataclass(frozen=True)
class SparseMLASm120DecodeConfig:
    d_qk: int = 512
    page_block_size: int = 64
    max_num_tokens: int = 64
    topks: frozenset = frozenset((2048, 2112, 2176))
    min_topk: int = 1
    max_num_heads: int = 128
    kv_cache_format: str = "fp8"
    bytes_per_token: int = 528
    head_counts: object = None
    topk_is_runtime: bool = True
    extra_page_block_sizes: frozenset = frozenset()

    def supported_num_heads(self):
        return tuple(range(1, self.max_num_heads + 1))

    def supported_topk(self, num_heads=None):
        return (
            tuple(sorted(self.topks))
            if num_heads is None or 1 <= num_heads <= self.max_num_heads
            else ()
        )

    def supports_decode(self, num_heads, topk, *, num_tokens=1, page_block_size=None):
        return (
            1 <= num_heads <= self.max_num_heads
            and topk >= self.min_topk
            and num_tokens <= self.max_num_tokens
            and page_block_size in (None, self.page_block_size)
        )


def supported_sparse_mla_sm120_configs(*, kv_cache_format="fp8"):
    return (
        {"glm53_nope": SparseMLASm120DecodeConfig()} if kv_cache_format == "fp8" else {}
    )


class SparseMLASm120Wrapper:
    """FlashInfer's native API, backed by MATE's PH1 scheduled NoPE kernel.

    ``arbitrary_fp32`` explicitly selects GLM inline scales. V4.1 page-footer
    caches belong to the FlashMLA API and are never inferred from 528 bytes here.
    """

    def __init__(
        self,
        max_num_tokens=None,
        max_num_heads=None,
        *,
        d_v=512,
        kv_scale_format="auto",
        kv_cache_format="fp8",
        device=None,
    ):
        if (
            d_v != 512
            or kv_cache_format != "fp8"
            or kv_scale_format != "arbitrary_fp32"
        ):
            raise ValueError(
                "MATE native runner currently requires GLM NoPE arbitrary_fp32 FP8 cache"
            )
        if (max_num_tokens is None) != (max_num_heads is None):
            raise ValueError(
                "max_num_tokens and max_num_heads must be provided together"
            )
        self.max_num_tokens, self.max_num_heads = max_num_tokens, max_num_heads
        self._out_lse = None
        if max_num_tokens is not None:
            if max_num_tokens <= 0 or not 1 <= max_num_heads <= 128:
                raise ValueError("invalid maximum token/head count")
            self._out_lse = torch.empty(
                (max_num_tokens, max_num_heads),
                dtype=torch.float32,
                device=device or "musa",
            )

    def run(
        self,
        q,
        kv_cache,
        indices,
        output,
        sm_scale,
        *,
        topk_length=None,
        attn_sink=None,
        extra_kv_cache=None,
        extra_indices=None,
        extra_topk_length=None,
        out_lse=None,
        mid_out=None,
        mid_lse=None,
        prefill_impl=None,
        return_lse=False,
    ):
        if any(
            x is not None for x in (extra_kv_cache, extra_indices, extra_topk_length)
        ):
            raise ValueError("GLM NoPE uses a single cache")
        if prefill_impl not in (None, "auto", "mg"):
            raise ValueError("unsupported prefill implementation")
        if q.ndim == 4:
            assert q.shape[1] == 1
            q = q.squeeze(1)
        if output.ndim == 4:
            assert output.shape[1] == 1
            output = output.squeeze(1)
        tokens, heads, dim = q.shape
        assert output.shape == (tokens, heads, 512) and output.dtype == torch.bfloat16
        assert dim == 512 and 1 <= heads <= 128
        assert output.device == kv_cache.device == indices.device == q.device
        if self.max_num_tokens is not None:
            assert tokens <= self.max_num_tokens and heads <= self.max_num_heads
        if indices.ndim == 3:
            assert indices.shape[1] == 1
            indices = indices.squeeze(1)
        assert indices.ndim == 2 and indices.shape[0] == tokens
        assert (mid_out is None) == (mid_lse is None)
        # Caller scratch is an implementation aid, not the output contract.
        # MATE owns its scheduled split workspace during each stream-ordered call.
        if kv_cache.ndim == 3:
            kv_cache = kv_cache.unsqueeze(2)
        if out_lse is None:
            if (
                self._out_lse is None
                or self._out_lse.shape[0] < tokens
                or self._out_lse.shape[1] < heads
            ):
                self._out_lse = torch.empty(
                    (tokens, heads), dtype=torch.float32, device=q.device
                )
            out_lse = self._out_lse
        assert (
            out_lse.dtype == torch.float32
            and out_lse.shape[0] >= tokens
            and out_lse.shape[1] >= heads
        )
        out_lse = out_lse[:tokens, :heads]
        if tokens == 0:
            return out_lse if return_lse else None
        width = indices.shape[-1]
        padded_width = max(64, (width + 63) // 64 * 64)
        if padded_width != width:
            indices = torch.nn.functional.pad(
                indices, (0, padded_width - width), value=-1
            )
        result, lse = sparse_mla_nope_decode(
            q.unsqueeze(1),
            kv_cache,
            indices[:, None, :],
            softmax_scale=float(sm_scale),
            topk_length=topk_length,
            attn_sink=attn_sink,
        )
        output.copy_(result.squeeze(1))
        out_lse.copy_(lse.squeeze(-1))
        return out_lse if return_lse else None
