"""FlashMLA fused-module API backed by MATE's MUSA implementation."""

from mate.sparse_mla.fused_norm_rope_attn_rope_cast import (
    decode as decode,
    permute_q_b_proj as permute_q_b_proj,
    permute_wv_proj as permute_wv_proj,
    prefill as prefill,
)
