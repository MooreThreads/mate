import functools
import torch
from mate.api_logging import mate_api
from mate._backend import resolve_backend
from mate.jit.gemm_ops import get_gemm_ops_module
from mate.testing.utils import ceil_div
from typing import Tuple, Optional, Literal


@functools.cache
def _get_module():
    return get_gemm_ops_module()


@mate_api
def ragged_m_moe_gemm_16bit(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    gemm_mode: Optional[
        Literal["per_token", "psum_expert", "per_expert"]
    ] = "per_token",
    major_a_mode: Optional[Literal["M", "K"]] = "K",
    major_b_mode: Optional[Literal["N", "K"]] = "K",
    num_mp: Optional[int] = None,
    alignment_m: Optional[int] = None,
):
    """
    Perform 16-bit GEMM operation for MoE (Mixture of Experts) with ragged tensor inputs.

    This function computes matrix multiplication between 16-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens assigned to them.

    Parameters
    ----------
    input_a : Tensor
        Input tensor A with shape ``(total_tokens, hidden_size)`` in fp16/bf16 format.
    input_b : Tensor
        Input tensor B with shape ``(num_expert, out_hidden_size, hidden_size)`` in fp16/bf16 format.
    ragged_tokens_info : Tensor
        If gemm_mode is `per_token`:
            Tensor indicating which expert each token belongs to, with shape ``(total_tokens,)``.
            Values represent expert indices, with -1 for unused positions.
        If gemm_mode is `psum_expert`
            Tensor with shape `(num_expert, )`, indicating how many tokens that first few experts have.
        If gemm_mode is `per_expert`
            Tensor with shape `(num_expert, )`, indicating how many tokens that every expert has.
    out : Tensor
        Output tensor with shape ``(total_tokens, out_hidden_size)``.
    major_a_mode : Optional[str]
        Indicating major stride of A.
        Default to `K`.
    major_b_mode : Optional[str]
        Indicating major stride of B.
        Default to `K`.
    gemm_mode : Optional[str],
        Indicating different meaning of ragged_tokens_info.
    alignment_m : Optional[int]
        Alignment requirement for total_tokens (m) dimension. Must be 128 or 256.
        Default is 128.
    num_mp : Optional[int]
        Suggest mp number.
        If None, will be get from device info.

    Returns
    -------
    Tensor
        Result tensor with shape ``(total_tokens, out_hidden_size)`` containing the GEMM output in fp16 or bf16 data type.

    """

    if alignment_m is None:
        alignment_m = 128

    if gemm_mode == "per_token":
        _get_module().get_function("ragged_moe_gemm_16bit")(
            input_a,
            input_b,
            ragged_tokens_info,
            out,
            False,
            None,
            alignment_m,
        )
    elif gemm_mode == "per_expert":
        _get_module().get_function("m_grouped_contig_gemm_16bit")(
            input_a,
            input_b,
            ragged_tokens_info,
            out,
            major_a_mode,
            major_b_mode,
            num_mp,
        )
    else:
        assert False, "Not supported gemm mode."

    return out


@mate_api
def masked_moe_gemm_16bit(
    a: torch.Tensor,
    b: torch.Tensor,
    masked_tokens_info: torch.Tensor,
    out: torch.Tensor,
    expect_tokens: Optional[int] = None,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
):
    """
    Perform 16-bit GEMM operation for MoE (Mixture of Experts) with masked tensor inputs.

    This function computes matrix multiplication between 16-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens, using a mask to indicate
    the actual number of tokens per expert.

    Parameters
    ----------
    a : Tensor
        Input tensor A with shape ``(num_expert, max_tokens, hidden_size)`` in fp16/bf16 format.
    b : Tensor
        Input tensor B with shape ``(num_expert, out_hidden_size, hidden_size)`` in fp16/bf16 format.
    masked_tokens_info : Tensor
        Tensor indicating the actual number of tokens for each expert, with shape ``(num_expert,)``.
        Values represent token counts for each expert.
    out : Tensor
        Output tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        Should be of fp16 or bf16 type. If None, a new tensor will be created.
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

    if expect_tokens is None:
        expect_tokens = 0

    if not enable_overlap:
        signal = None

    if enable_overlap and signal is None:
        tile_signal = 64
        expert_sz = a.size(0)
        max_m = a.size(1)

        # zero init is required
        signal = torch.zeros(
            expert_sz * ceil_div(max_m, tile_signal), dtype=torch.int32, device=a.device
        )

    res = _get_module().get_function("masked_moe_gemm_16bit")(
        a,
        b,
        masked_tokens_info,
        out,
        expect_tokens,
        signal,
    )

    return (out, signal, res[0], res[1]) if enable_overlap else out


@mate_api
def ragged_m_moe_gemm_8bit(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    gemm_mode: Optional[
        Literal["per_token", "psum_expert", "per_expert"]
    ] = "per_token",
    major_a_mode: Optional[Literal["M", "K"]] = "K",
    major_b_mode: Optional[Literal["N", "K"]] = "K",
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    num_mp: Optional[int] = None,
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
            If gemm_mode is `per_token`:
            Tensor indicating which expert each token belongs to, with shape ``(total_tokens,)``.
            Values represent expert indices, with -1 for unused positions.
        If gemm_mode is `psum_expert`
            Tensor with shape `(num_expert, )`, indicating how many tokens that first few experts have.
        If gemm_mode is `per_expert`
            Tensor with shape `(num_expert, )`, indicating how many tokens that every expert has.
    out : Tensor
        Output tensor with shape ``(total_tokens, out_hidden_size)``.
    major_a_mode : Optional[str]
        Indicating major stride of A.
        Default to `K`.
    major_b_mode : Optional[str]
        Indicating major stride of B.
        Default to `K`.
    gemm_mode : Optional[str],
        Indicating different meaning of ragged_tokens_info.
    scale_granularity_mnk : Optional[Tuple[int, int, int]]
        Quantization granularity for total_tokens, out_hidden_size, hidden_size (m, n, k) dimensions respectively.
        Default is ``(1, 128, 128)``.
    alignment_m : Optional[int]
        Alignment requirement for total_tokens (m) dimension. Must be 128 or 256.
        Default is 128.
    num_mp : Optional[int]
        Suggest mp number.
        If None, will be get from device info.

    Returns
    -------
    Tensor
        Result tensor with shape ``(total_tokens, out_hidden_size)`` containing the GEMM output in fp16 or bf16 data type.

    """

    if scale_granularity_mnk is None:
        scale_granularity_mnk = (1, 128, 128)

    if alignment_m is None:
        alignment_m = 128

    if gemm_mode == "per_token":
        _get_module().get_function("ragged_moe_gemm_8bit")(
            input_a,
            input_b,
            ragged_tokens_info,
            scale_granularity_mnk,
            out,
            alignment_m,
        )
    elif gemm_mode == "per_expert":
        _get_module().get_function("m_grouped_contig_gemm_8bit")(
            input_a,
            input_b,
            ragged_tokens_info,
            scale_granularity_mnk,
            out,
            major_a_mode,
            major_b_mode,
            num_mp,
        )
    else:
        assert False, "Not supported gemm mode"

    return out


@mate_api
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

    res = _get_module().get_function("masked_moe_gemm_8bit")(
        input_a,
        input_b,
        masked_tokens_info,
        scale_granularity_mnk,
        out,
        expect_tokens,
        signal,
    )

    return (out, signal, res[0], res[1]) if enable_overlap else out


@mate_api
def ragged_k_moe_gemm_8bit(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    gemm_mode: Optional[Literal["per_expert"]] = "per_expert",
    major_a_mode: Optional[Literal["M", "K"]] = "M",
    major_b_mode: Optional[Literal["N", "K"]] = "N",
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    num_mp: Optional[int] = None,
):
    """
    Perform 8-bit GEMM operation for MoE (Mixture of Experts) with token of each expert.

    This function computes matrix multiplication between 8-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens.

    Parameters
    ----------
    input_a : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input A.
        **fp8_tensor** has shape ``(k, m)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(k // scale_granularity_k, m)`` and should be of fp32 type.
    input_b : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input B.
        **fp8_tensor** has shape ``(k, n)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(k // scale_granularity_k, n)`` and should be of fp32 type.
    ragged_tokens_info : Tensor
        Tensor indicating the actual number of tokens for each expert, with shape ``(num_expert,)``.
        Values represent token counts for each expert.
    out : Tensor
        Output tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        Should be of float type. Should not be None.
    gemm_mode : Optional[str],
        Indicating different meaning of ragged_tokens_info.
    major_a_mode : Optional[str]
        Major mode of A, defult to `M`.
        Only support TN m_grouped_gemm on MP31.
    major_b_mode : Optional[str]
        Major mode of B, defult to `N`.
    scale_granularity_mnk : Optional[Tuple[int, int, int]]
        Quantization granularity for max_tokens, out_hidden_size, hidden_size (m, n, k) dimensions respectively.
        Kgroupgemm only support 1D1D scale, should be ``(1, 1, 128)``.
    num_mp : Optional[int]
        Suggest mp number.
        If None, will be get from device info.

    Returns
    -------
    Result tensor with shape ``(num_experts, total_tokens, out_hidden_size)`` containing the GEMM output in float data type,
    Representing D = D + A * B for each expert

    """
    if scale_granularity_mnk is None:
        scale_granularity_mnk = (1, 1, 128)
    else:
        assert scale_granularity_mnk == (1, 1, 128), (
            "k_grouped_contig_gemm_8bit only support 1D1D gemm"
        )

    if major_a_mode is None:
        major_a_mode = "M"
    if major_b_mode is None:
        major_b_mode = "N"

    assert major_a_mode == "M" and major_b_mode == "N", (
        "k_grouped_contig_gemm_8bit only support TN layout"
    )

    _get_module().get_function("k_grouped_contig_gemm_8bit")(
        input_a,
        input_b,
        ragged_tokens_info,
        scale_granularity_mnk,
        out,
        num_mp,
    )

    return out


@mate_api
def ragged_k_moe_gemm_16bit(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    gemm_mode: Optional[Literal["per_expert"]] = "per_expert",
    major_a_mode: Optional[Literal["M", "K"]] = "M",
    major_b_mode: Optional[Literal["N", "K"]] = "N",
    num_mp: Optional[int] = None,
):
    """
    Perform 16-bit GEMM operation for MoE (Mixture of Experts) with token of each expert.

    This function computes matrix multiplication between 16-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens.

    Parameters
    ----------
    input_a : Tensor
        Input tensor A with shape ``(total_tokens, hidden_size)`` in fp16/bf16 format.
    input_b : Tensor
        Input tensor B with shape ``(num_expert, out_hidden_size, hidden_size)`` in fp16/bf16 format.
    ragged_tokens_info : Tensor
        Tensor indicating the actual number of tokens for each expert, with shape ``(num_expert,)``.
        Values represent token counts for each expert.
    out : Tensor
        Output tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        Should be of float type. Should not be None.
    gemm_mode : Optional[str],
        Indicating different meaning of ragged_tokens_info.
    major_a_mode : Optional[str]
        Major mode of A, defult to `M`.
        Only support TN m_grouped_gemm on MP31.
    major_b_mode : Optional[str]
        Major mode of B, defult to `N`.
    num_mp : Optional[int]
        Suggest mp number.
        If None, will be get from device info.

    Returns
    -------
    Result tensor with shape ``(num_experts, total_tokens, out_hidden_size)`` containing the GEMM output in float data type,
    Representing D = D + A * B for each expert

    """

    if major_a_mode is None:
        major_a_mode = "M"
    if major_b_mode is None:
        major_b_mode = "N"

    assert major_a_mode == "M" and major_b_mode == "N", (
        "k_grouped_contig_gemm_8bit only support TN layout"
    )

    _get_module().get_function("k_grouped_contig_gemm_16bit")(
        input_a,
        input_b,
        ragged_tokens_info,
        out,
        num_mp,
    )

    return out


@mate_api
def bmm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: str = "auto",
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
    backend = resolve_backend(
        backend, supported=("mudnn",), allow_auto=True, default="auto"
    )

    if scale_granularity_mnk is None:
        scale_granularity_mnk = (-1, -1, -1)

    if out is None:
        batch = a.size(0)
        m = a.size(1)
        n = b.size(2)

        if out_dtype not in [torch.bfloat16, torch.float16]:
            raise ValueError("Only bf16 and fp16 are supported for out_type!")

        out = torch.empty((batch, m, n), dtype=out_dtype, device=a.device)

    _get_module().get_function("bmm_fp8")(
        a, b, a_scale, b_scale, out, scale_granularity_mnk, backend
    )
    return out


@mate_api
def bmm_fp16(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: str = "auto",
):
    backend = resolve_backend(
        backend, supported=("mudnn",), allow_auto=True, default="auto"
    )
    if out is None:
        batch = a.size(0)
        m = a.size(1)
        n = b.size(2)

        if out_dtype not in [torch.bfloat16, torch.float16]:
            raise ValueError("Only bf16 and fp16 are supported for out_type!")

        out = torch.empty((batch, m, n), dtype=out_dtype, device=a.device)
    _get_module().get_function("bmm_fp16")(a, b, out, backend)
    return out


@mate_api
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
    backend: str = "auto",
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
    backend = resolve_backend(
        backend, supported=("mudnn",), allow_auto=True, default="auto"
    )

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

    _get_module().get_function("gemm_fp8_nt_groupwise")(
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
    return out
