import torch
import mate._C  # noqa: F401
from mate.testing.utils import ceil_div
from typing import Tuple, Optional, Literal


def ragged_moe_gemm_8bit(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    alignment_m: Optional[int] = None,
):
    """
    Perform 8-bit GEMM operation for MoE (Mixture of Experts) with ragged tensor inputs.

    This function computes matrix multiplication between 8-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens assigned to them.

    Parameters
    ----------
    input_a : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input A.
        **fp8_tensor** has shape ``(total_tokens, hidden_size)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(total_tokens, hidden_size // scale_granularity_m)`` and should be of fp32 type.
    input_b : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input B.
        **fp8_tensor** has shape ``(num_expert, out_hidden_size, hidden_size)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(num_expert, out_hidden_size // scale_granularity_n, hidden_size // scale_granularity_k)`` and should be of fp32 type.
    ragged_tokens_info : Tensor
        Tensor indicating which expert each token belongs to, with shape ``(total_tokens,)``.
        Values represent expert indices, with -1 for unused positions.
    out : Tensor
        Output tensor with shape ``(total_tokens, out_hidden_size)``.
        Should be of fp16 or bf16 type.
    scale_granularity_mnk : Optional[Tuple[int, int, int]]
        Quantization granularity for total_tokens, out_hidden_size, hidden_size (m, n, k) dimensions respectively.
        Default is ``(1, 128, 128)``.
    alignment_m : Optional[int]
        Alignment requirement for total_tokens (m) dimension. Must be 128 or 256.
        Default is 128.

    Returns
    -------
    Tensor
        Result tensor with shape ``(total_tokens, out_hidden_size)`` containing the GEMM output in fp16 or bf16 data type.

    """

    if scale_granularity_mnk is None:
        scale_granularity_mnk = (1, 128, 128)

    if alignment_m is None:
        alignment_m = 128

    torch.ops.mate.ragged_moe_gemm_8bit(
        input_a, input_b, ragged_tokens_info, scale_granularity_mnk, out, alignment_m
    )

    return out


def masked_moe_gemm_8bit(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    masked_tokens_info: torch.Tensor,
    out: torch.Tensor,
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    expect_tokens: Optional[int] = None,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
):
    """
    Perform 8-bit GEMM operation for MoE (Mixture of Experts) with masked tensor inputs.

    This function computes matrix multiplication between 8-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens, using a mask to indicate
    the actual number of tokens per expert.

    Parameters
    ----------
    input_a : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input A.
        **fp8_tensor** has shape ``(num_expert, max_tokens, hidden_size)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(num_expert, max_tokens, hidden_size // scale_granularity_k)`` and should be of fp32 type.
    input_b : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input B.
        **fp8_tensor** has shape ``(num_expert, out_hidden_size, hidden_size)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(num_expert, out_hidden_size // scale_granularity_n, hidden_size // scale_granularity_k)`` and should be of fp32 type.
    masked_tokens_info : Tensor
        Tensor indicating the actual number of tokens for each expert, with shape ``(num_expert,)``.
        Values represent token counts for each expert.
    out : Tensor
        Output tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        Should be of fp16 or bf16 type. If None, a new tensor will be created.
    scale_granularity_mnk : Optional[Tuple[int, int, int]]
        Quantization granularity for max_tokens, out_hidden_size, hidden_size (m, n, k) dimensions respectively.
        Default is ``(1, 128, 128)``.
    expect_tokens : Optional[int]
        Expected number of tokens. If None, defaults to 0.
    enable_overlap : Optional[bool]
        Whether to enable Single-Batch Overlap (SBO). Default is False.
    signal : Optional[Tensor]
        Signal tensor with shape ``(num_expert * ceil_div(max_m, 64))``for SBO. Required if enable_overlap is True. If None, a new tensor will be created if needed.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor, int, int]]
        If ``enable_overlap`` is ``False``, returns result tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        If ``enable_overlap`` is ``True``, returns a tuple containing:

            - result tensor with shape ``(num_expert, max_tokens, out_hidden_size)``
            - signal tensor
            - block_m int
            - threshold int

    """
    if scale_granularity_mnk is None:
        scale_granularity_mnk = (1, 128, 128)

    if expect_tokens is None:
        expect_tokens = 0

    if not enable_overlap:
        signal = None

    if enable_overlap and signal is None:
        tile_signal = 64
        a, _ = input_a
        expert_sz = a.size(0)
        max_m = a.size(1)

        # zero init is required
        signal = torch.zeros(
            expert_sz * ceil_div(max_m, tile_signal), dtype=torch.int32, device=a.device
        )

    res = torch.ops.mate.masked_moe_gemm_8bit(
        input_a,
        input_b,
        masked_tokens_info,
        scale_granularity_mnk,
        out,
        expect_tokens,
        signal,
    )

    return (out, signal, res[0], res[1]) if enable_overlap else out


def bmm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["mudnn", "auto"] = "auto",
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
):
    """
    Perform batched matrix multiplication with FP8 quantized tensors.

    This function computes the batched matrix multiplication of two FP8 quantized tensors,
    applying scaling factors to produce a result in the specified output data type.

    Parameters
    ----------
    a : Tensor
        Input tensor A with shape ``(batch, m, k)`` in FP8 format (e4m3/e5m2).
        The **`k`** dimension must be contiguous.
    b : Tensor
        Input tensor B with shape ``(batch, k, n)`` in FP8 format (e4m3/e5m2).
        The **`k`** dimension must be contiguous.
    a_scale : Tensor
        Scaling factors for tensor A with shape depending on scale_granularity.
        Should be of fp32 type.
    b_scale : Tensor
        Scaling factors for tensor B with shape depending on scale_granularity.
        Should be of fp32 type.
    out_dtype : torch.dtype
        Data type for the output tensor. Only torch.bfloat16 and torch.float16 are supported.
    out : Optional[Tensor]
        Pre-allocated output tensor with shape ``(batch, m, n)``.
        Default is None.
        If None, a new tensor will be allocated.
    backend : str
        Backend to use for the operation.
        Current support backends are "mudnn" and "auto".
        Default is "auto".
    scale_granularity_mnk : Optional[Tuple[int, int, int]]
        Granularity of scaling for batch, m, and n dimensions respectively.
        Only ``(-1, -1, -1)`` and ``(1, -1, -1)`` are supported.
        If None, defaults to ``(-1, -1, -1)``.

    Returns
    -------
    Tensor
        Result tensor with shape ``(batch, m, n)`` in the specified output data type.
    """
    if backend not in ["mudnn", "auto"]:
        raise ValueError("backend must be one of ['mudnn', 'auto']")

    if scale_granularity_mnk is None:
        scale_granularity_mnk = (-1, -1, -1)

    if out is None:
        batch = a.size(0)
        m = a.size(1)
        n = b.size(2)

        if out_dtype not in [torch.bfloat16, torch.float16]:
            raise ValueError("Only bf16 and fp16 are supported for out_type!")

        out = torch.empty((batch, m, n), dtype=out_dtype, device=a.device)

    return torch.ops.mate.bmm_fp8(
        a, b, a_scale, b_scale, out, scale_granularity_mnk, backend
    )


def gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[Literal["MN", "K"]] = None,
    mma_sm: Optional[int] = None,
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    backend: Literal["mudnn"] = "mudnn",
):
    """
    Perform groupwise FP8 GEMM operation with scaling.

    This function computes the matrix multiplication of two FP8 quantized tensors, applying scaling factors to produce a result
    in the specified output data type. It supports groupwise quantization with configurable
    scale granularity.

    Parameters
    ----------
    a : Tensor
        Input tensor A with shape ``(m, k)`` in FP8 format (e4m3/e5m2).
        Tensor must be contiguous.
    b : Tensor
        Input tensor B with shape ``(n, k)`` in FP8 format (e4m3/e5m2).
        Tensor must be contiguous.
    a_scale : Tensor
        Scaling factors for tensor A. Shape depends on scale_granularity_mnk parameter.
        Should be of fp32 type. Must be contiguous.
    b_scale : Tensor
        Scaling factors for tensor B. Shape depends on scale_granularity_mnk parameter.
        Should be of fp32 type. Must be contiguous.
    scale_major_mode : str
        Scale major mode "MN" or "K" for groupwise operations. Default is "K".
    mma_sm : Optional[int]
        MMA SM configuration. Currently only supports 1. Default is 1.
    scale_granularity_mnk : Optional[Tuple[int, int, int]]
        Granularity of scaling for m, n, and k dimensions respectively.
        Default is ``(1, 128, 128)``.
    out : Optional[Tensor]
        Pre-allocated output tensor with shape ``(m, n)``.
        Should be of bf16 type. If None, a new tensor will be allocated.
    out_dtype : Optional[torch.dtype]
        Data type for the output tensor. Only torch.bfloat16 is supported.
        If None, defaults to torch.bfloat16.
    backend : str
        Backend to use for the operation. Currently only supports "mudnn".

    Returns
    -------
    Tensor
        Result tensor with shape ``(m, n)`` in the specified output data type (bf16).
    """
    if scale_major_mode is None:
        scale_major_mode = "K"

    if mma_sm is None:
        mma_sm = 1

    if mma_sm != 1:
        mma_sm = 1
        print("Warning: only mma_sm=1 is supported now, set mma_sm=1")

    if scale_granularity_mnk is None:
        scale_granularity_mnk = (1, 128, 128)

    if out is None:
        m = a.size(0)
        n = b.size(0)

        if out_dtype is None:
            out_dtype = torch.bfloat16

        if out_dtype not in [torch.bfloat16]:
            raise ValueError("Only bf16 and fp16 are supported for out_type!")

        out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    return torch.ops.mate.gemm_fp8_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        scale_major_mode,
        mma_sm,
        scale_granularity_mnk,
        out,
        backend,
    )
