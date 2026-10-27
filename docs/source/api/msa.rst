.. _apimsa:

MSA
===

.. currentmodule:: mate.msa_interface

Use the ``fmha_sm100`` wrapper first when your project already targets the MSA
package surface. Use the direct MATE APIs below when you need the native plan
and runtime contract.

MSA (MiniMax Sparse Attention) covers dense, paged, and sparse attention on
MUSA.

MSA improves indexer performance and skips metadata computation on short
queries.

The ``mate`` package also exports the primary planning and execution
entrypoints at the top level.

Dense and paged paths
---------------------

.. autofunction:: msa_plan
.. autofunction:: msa

.. autoclass:: MsaPlan
.. autoclass:: MsaPrefillPlan
.. autoclass:: MsaDecodePlan
.. autoclass:: MsaRuntimeMetadata
.. autodata:: MsaPlanInfo

Use ``build_page_table_from_flat_kv_indices`` to turn flat page indices into a
fixed-shape page table for paged MSA.

Sparse paths
------------

.. autofunction:: sparse_msa_plan
.. autofunction:: sparse_msa
.. autofunction:: sparse_topk_select
.. autofunction:: sparse_decode_atten_func

``sparse_topk_select`` always runs the exact selection kernel, which ranks
scores in FP32 with deterministic ordering for tied indices and keeps forced
sink and local slots separate from the scored candidates. Earlier releases
copied the identity page order when no ``query_positions`` were given and the
valid page count was within the selected block count; 0.2.8 removed that
shortcut, so results no longer depend on that input combination.
