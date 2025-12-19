.. _apigemm:

GEMM
=========

MoE GEMM
-------------------------

.. currentmodule:: mate.gemm
.. autofunction:: ragged_moe_gemm_8bit
.. autofunction:: masked_moe_gemm_8bit

FP8 GEMM
-------------------------

.. autofunction:: bmm_fp8
.. autofunction:: gemm_fp8_nt_groupwise

DeepGemm Lighting Indexer
-------------------------

.. currentmodule:: mate.deep_gemm
.. autofunction:: get_paged_mqa_logits_metadata
.. autofunction:: fp8_paged_mqa_logits
