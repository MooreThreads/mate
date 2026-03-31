import torch
import mate._C  # noqa: F401
from typing import Optional
from mate.api_logging import mate_api


@mate_api
def mla(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    page_table: torch.Tensor,  # [batch]
    kv_len: torch.Tensor,  # [batch]
    sm_scale: float,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_causal: bool = True,
):
    return torch.ops.mate.mla(
        q_nope,
        q_pe,
        ckv,
        kpe,
        page_table,
        kv_len,
        sm_scale,
        out,
        lse,
        is_causal,
    )


@mate_api
def get_mla_metadata(
    seqlens_k: torch.Tensor,
    num_q_tokens_per_head_k: int,
    h_k: int,
    h_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
):
    return torch.ops.mate.get_mla_decoding_metadata(
        seqlens_k,
        num_q_tokens_per_head_k,
        h_k,
        h_q,
        is_fp8_kvcache,
        topk,
    )
