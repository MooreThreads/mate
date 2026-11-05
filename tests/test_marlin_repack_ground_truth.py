# SPDX-License-Identifier: Apache-2.0
"""Independent ground truth for the Marlin permute/pack axis (ticket MUSA-100077).

Why this file exists. ``tests/test_w4a16_marlin.py::test_repack_byte_parity_oracle_vs_repack_musa``
compares MATE's shipping repack (``mate.marlin.layout``) against ``mate.marlin.reference``, and both
were written from the same reading of ``gptq_marlin_repack.cu``. Agreement between them proves MATE
is self-consistent and nothing more: if the reading is wrong, both are wrong in the same way and the
gate passes forever. This file pins the axis to signals that were *not* produced by that reading.

Signal A -- upstream vLLM's own oracle, executed rather than transcribed. ``get_weight_perm``,
``marlin_permute_weights`` and ``marlin_weights`` are extracted from the live upstream file
``vllm/model_executor/layers/quantization/utils/marlin_utils_test.py`` with ``ast`` and ``exec``'d
here (with a small numpy shim: the extracting host has no torch). Upstream's own Marlin tests use
exactly those three functions as the oracle for the quantized layout, and their frame is a *third*
one -- a single 1024-entry permutation of a tile's flattened ``(k_local, n_local)`` slots, applied
per tile -- so agreement is not a restatement of MATE's ``(k_local, n_local)`` decode. When no
upstream checkout is reachable, the vendored copy of the same source text is executed instead; it is
byte-pinned by ``VENDORED_SHA256`` and
``test_vendored_source_text_still_matches_the_upstream_file`` re-derives it from the file whenever
the file is there.

Signal B -- a simulation of ``gptq_marlin_repack_kernel`` written from the CUDA control flow rather
than from an index equation: per-tile shared-memory stage, per-thread register file, ``pack_idx``
write, in the kernel's own order. It shares no code with ``mate.marlin``.

Signal C -- a mutation gate. ``test_the_gate_bites_when_the_tile_map_is_perturbed`` perturbs the
convention (transpositions of ``PACK_IDX``, ``TC_OFFSETS`` and the k-gather) and asserts that each
perturbation *is* detected, so a passing gate is evidence that the comparison can see a wrong
convention rather than a check that cannot fail.

Not covered here, deliberately: MATE's own ``weight_b`` frame (``(N, K//2)`` int8, even K low, plus
the ``^ 0x8`` presentation) is a *hardware* convention, measured on a leased mp_31 S5000, and it is
not an upstream layout at all. Nothing on a CPU-only host can re-derive it, and no test in this file
claims to; the gates for it live in ``tests/test_w4a16_marlin.py``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest


# =================================================================================================
# Loading the code under test: same by-path pattern as tests/test_w4a16_marlin.py, so these gates
# need no installed package and no torch.
# =================================================================================================
def _load_mate_marlin():
    try:
        import mate.marlin as marlin_module

        return marlin_module
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    package = root / "mate" / "marlin"
    if not (package / "__init__.py").is_file():
        return None
    if "mate" not in sys.modules:
        parent = types.ModuleType("mate")
        parent.__path__ = [str(root / "mate")]
        sys.modules["mate"] = parent
    spec = importlib.util.spec_from_file_location(
        "mate.marlin", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    assert spec is not None and spec.loader is not None, f"cannot load {package}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["mate.marlin"] = module
    spec.loader.exec_module(module)
    return module


_MATE = _load_mate_marlin()
LAYOUT = _MATE.layout
REFERENCE = _MATE.reference
GUARDS = _MATE.guards
COMPAT = _MATE.compat
QUANT = _MATE.quant
LINEAR = _MATE.linear

# =================================================================================================
# Locating upstream vLLM. The pinned revision is the one this port declares in
# mate/marlin/reference.py:4 (vLLM 2cf0a6915c, v0.28.0-dev); its repack .cu sources and
# marlin_utils_test.py are byte-identical to the newer workspace checkout, so the derivation does not
# depend on which of the two is found. Set VLLM_UPSTREAM_ROOT to point at a root explicitly.
# =================================================================================================
UPSTREAM_COMMIT = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
_UPSTREAM_RELATIVE = {
    "marlin_utils_test.py": "vllm/model_executor/layers/quantization/utils/marlin_utils_test.py",
    "quant_utils.py": "vllm/model_executor/layers/quantization/utils/quant_utils.py",
}


def _candidate_roots() -> list[Path]:
    here = Path(__file__).resolve()
    roots = []
    env = os.environ.get("VLLM_UPSTREAM_ROOT")
    if env:
        roots.append(Path(env))
    roots += [
        here.parents[1].parent / "vllm-workspace" / "vllm",
        here.parents[1].parent / "vllm-musa100015-vllm-verify",  # the declared pin
        here.parents[1].parent / "vllm",
        here.parents[1] / ".." / "vllm",
    ]
    return roots


def _find_upstream(name: str) -> Path | None:
    for root in _candidate_roots():
        candidate = (root / _UPSTREAM_RELATIVE[name]).resolve()
        if candidate.is_file():
            return candidate
    return None


def _extract_functions(path: Path, names: set[str]) -> dict[str, str]:
    """Pull whole function sources out of a file by AST, so the text is the file's, not ours."""
    text = path.read_text()
    tree = ast.parse(text)
    found = {
        node.name: ast.get_source_segment(text, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = names - set(found)
    assert not missing, f"{path} does not define {sorted(missing)}"
    return {name: found[name] for name in sorted(found)}


# =================================================================================================
# The shim that lets upstream's source run on numpy. It ADDS only what upstream's code calls:
# Tensor.permute(dims) (== numpy transpose, torch semantics), .numel(), .cpu(), .numpy(), .to(),
# .device, and torch.from_numpy. No upstream arithmetic is reimplemented.
# =================================================================================================
class _Array(np.ndarray):
    def permute(self, *dims):
        dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else dims
        return np.transpose(self, dims).view(_Array)

    def numel(self) -> int:
        return int(self.size)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def to(self, *args, **kwargs):
        return self

    @property
    def device(self):
        return "cpu"


class _FakeTorch:
    Tensor = _Array

    @staticmethod
    def from_numpy(array):
        return np.asarray(array).view(_Array)


def _build_oracle(sources: dict[str, str]):
    """``exec`` upstream's functions and return ``(oracle, perm)``.

    ``oracle(codes, size_k, size_n)`` reproduces ``marlin_utils_test.marlin_weights``: the tile
    permutation applied to the flat 1024-slot tile, then 8 flat elements packed per word, LSB first.
    The result is the same buffer ``gptq_marlin_repack`` must produce, in unsigned 32-bit words.
    """
    namespace: dict = {
        "np": np,
        "numpy": np,
        "torch": _FakeTorch(),
        # marlin_permute_weights' default argument, upstream marlin_utils.py:31.
        "GPTQ_MARLIN_TILE": 16,
    }
    order = ["get_pack_factor", "get_weight_perm", "marlin_permute_weights", "marlin_weights"]
    for name in order:
        exec(compile(sources[name], f"<upstream:{name}>", "exec"), namespace)  # noqa: S102
    perm = namespace["get_weight_perm"](4)

    def oracle(codes: np.ndarray, size_k: int, size_n: int) -> np.ndarray:
        """Upstream's layout for an int4 weight: ``(size_k, size_n)`` codes -> Marlin int32 words."""
        words = namespace["marlin_weights"](
            np.asarray(codes, dtype=np.int64).view(_Array), size_k, size_n, 4, perm
        )
        return np.asarray(words).astype(np.int64) & 0xFFFFFFFF

    return oracle, perm


# =================================================================================================
# VENDORED ground truth: upstream vLLM's own oracle, copied verbatim (no edits, no reformatting)
# from the checkout at
#   /home/xiaodongye/ws/vllm-musa100015-vllm-verify
# revision 2cf0a6915ce544dc493a0990f2ea38d81601128a (the revision mate/marlin/reference.py:4
# declares), by `ast.get_source_segment` -- mechanically, so no hand transcription can drift.
# VENDORED_SHA256 pins each segment, and
# test_vendored_source_text_still_matches_the_upstream_file re-derives them from the file
# whenever a checkout is reachable.
# =================================================================================================
VENDORED_SOURCES = {
# vllm/model_executor/layers/quantization/utils/quant_utils.py:679-681
    "get_pack_factor": r"""
def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits
    """,
# vllm/model_executor/layers/quantization/utils/marlin_utils_test.py:73-125
    "get_weight_perm": r"""
def get_weight_perm(num_bits: int, is_a_8bit: bool = False):
    perm_list: list[int] = []
    if is_a_8bit:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    4 * (i % 4),
                    4 * (i % 4) + 1,
                    4 * (i % 4) + 2,
                    4 * (i % 4) + 3,
                    4 * (i % 4 + 4),
                    4 * (i % 4 + 4) + 1,
                    4 * (i % 4 + 4) + 2,
                    4 * (i % 4 + 4) + 3,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(2):
                perm_list.extend([p + 512 * j for p in perm1])
    else:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)

    if num_bits == 4:
        if is_a_8bit:  # noqa: SIM108
            interleave = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        else:
            interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        if is_a_8bit:  # noqa: SIM108
            interleave = np.array([0, 1, 2, 3])
        else:
            interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm
    """,
# vllm/model_executor/layers/quantization/utils/marlin_utils_test.py:33-51
    "marlin_permute_weights": r"""
def marlin_permute_weights(
    q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE, is_a_8bit=False
):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    if is_a_8bit:
        # Permute weights to 32x32 marlin tiles
        q_w = q_w.reshape((size_k // (tile * 2), tile * 2, size_n // tile, tile))
    else:
        # Permute weights to 16x64 marlin tiles
        q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w
    """,
# vllm/model_executor/layers/quantization/utils/marlin_utils_test.py:54-70
    "marlin_weights": r"""
def marlin_weights(q_w, size_k, size_n, num_bits, perm, is_a_8bit=False):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm, is_a_8bit=is_a_8bit)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_packed
    """,
}

VENDORED_SHA256 = {
    "get_pack_factor": "32b5e103f076071fd37e001e7fc7b134ba070b43d9ee49b1413662fdfe332cd3",
    "get_weight_perm": "f1edf8814f7b4de969a8cbfe5ff914a74f3a2f8909c6f7e4b4fe09124b65856b",
    "marlin_permute_weights": "cf64e92069e5f531131650f27025a432cc555508998fdd8078df456d300b4313",
    "marlin_weights": "3f5f7622e3cc8df3ae6bf3568da87569a68dbf22d5e9e9962b7d4433fb59f8f3",
}


GROUND_TRUTH, UPSTREAM_TILE_PERM = _build_oracle(VENDORED_SOURCES)


def _upstream_sources_from_disk() -> dict[str, str] | None:
    mut = _find_upstream("marlin_utils_test.py")
    qu = _find_upstream("quant_utils.py")
    if mut is None or qu is None:
        return None
    sources = _extract_functions(mut, {"get_weight_perm", "marlin_permute_weights", "marlin_weights"})
    sources.update(_extract_functions(qu, {"get_pack_factor"}))
    return sources


needs_upstream_checkout = pytest.mark.skipif(
    _upstream_sources_from_disk() is None,
    reason=(
        "no upstream vLLM checkout reachable (set VLLM_UPSTREAM_ROOT); the ground-truth gates "
        "themselves still run against the sha256-pinned vendored source, only the provenance "
        "re-check and the live-execution comparison are skipped"
    ),
)


# =================================================================================================
# Signal B: the CUDA kernel's control flow, simulated. Written from gptq_marlin_repack.cu /
# awq_marlin_repack.cu only; shares no code with mate.marlin.
# =================================================================================================
PACK_IDX_UPSTREAM = (0, 2, 4, 6, 1, 3, 5, 7)  # gptq_marlin_repack.cu:201
TC_OFFSETS_UPSTREAM = (0, 1, 8, 9)  # gptq_marlin_repack.cu:132
AWQ_UNDO_PACK = (0, 4, 1, 5, 2, 6, 3, 7)  # awq_marlin_repack.cu:107


def simulate_gptq_kernel(b_q_weight: np.ndarray, size_k: int, size_n: int, perm=None) -> np.ndarray:
    """Step-through of ``gptq_marlin_repack_kernel<*, 4, has_perm, false>``.

    The kernel stages, for shared row ``k_id``, the packed word of source k
    ``perm[k_tile*16 + k_id]`` when act order is on and ``k_tile*16 + k_id`` otherwise, then extracts
    nibble ``src_k % 8``; that is exactly ``w[src_k][n]`` for the tile's 64 columns. From there it is
    ``repack_tile``: thread (``th_id``, ``warp_id``) of each of 4 warps writes word
    ``th_id*4 + warp_id``, nibble ``i`` holding ``vals[pack_idx[i]]`` with ``vals[i]`` =
    (``tc_row + tc_offsets[i]``, ``cur_n``) and ``vals[4+i]`` the same k at ``cur_n + 8``.
    """
    assert size_k % 16 == 0 and size_n % 64 == 0, (size_k, size_n)
    k_tiles, n_tiles = size_k // 16, size_n // 64
    packed = np.asarray(b_q_weight)
    out = np.zeros((k_tiles, 2 * size_n), dtype=np.int64)
    for k_tile in range(k_tiles):
        for n_tile in range(n_tiles):
            for th_id in range(32):  # :127
                tc_col = th_id // 4  # :129
                tc_row = (th_id % 4) * 2  # :130
                for warp_id in range(4):  # :125
                    cur_n = warp_id * 16 + tc_col  # :134
                    vals = [0] * 8
                    for i in range(4):
                        k_idx = tc_row + TC_OFFSETS_UPSTREAM[i]
                        k_dst = k_tile * 16 + k_idx
                        src_k = k_dst if perm is None else int(perm[k_dst])  # :181-183 / :151
                        # the staged word's nibble for src_k is the logical value at (src_k, n)
                        vals[i] = (int(packed[src_k // 8, n_tile * 64 + cur_n]) >> (4 * (src_k % 8))) & 0xF
                        vals[4 + i] = (
                            int(packed[src_k // 8, n_tile * 64 + cur_n + 8]) >> (4 * (src_k % 8))
                        ) & 0xF
                    res = 0
                    for i in range(8):
                        res |= vals[PACK_IDX_UPSTREAM[i]] << (4 * i)  # :205-207
                    out[k_tile, n_tile * 128 + th_id * 4 + warp_id] = res  # :196,209
    return out


def simulate_awq_kernel(b_q_weight: np.ndarray, size_k: int, size_n: int) -> np.ndarray:
    """Step-through of ``awq_marlin_repack_kernel<*, 4, false>``: same write pattern, AWQ input."""
    assert size_k % 16 == 0 and size_n % 64 == 0, (size_k, size_n)
    k_tiles, n_tiles = size_k // 16, size_n // 64
    out = np.zeros((k_tiles, 2 * size_n), dtype=np.int64)
    for k_tile in range(k_tiles):
        for n_tile in range(n_tiles):
            first_k = k_tile * 16
            first_n_packed = n_tile * 8  # tile_n_ints = 64 / 8
            for th_id in range(32):
                tc_col = th_id // 4
                tc_row = (th_id % 4) * 2
                for warp_id in range(4):
                    cur_n = warp_id * 16 + tc_col
                    cur_n_packed = cur_n // 8
                    pos = AWQ_UNDO_PACK[cur_n % 8]  # :104-108
                    vals = [0] * 8
                    for i in range(4):
                        k = first_k + tc_row + TC_OFFSETS_UPSTREAM[i]
                        lo = int(b_q_weight[k, first_n_packed + cur_n_packed])
                        hi = int(b_q_weight[k, first_n_packed + cur_n_packed + 1])
                        vals[i] = (lo >> (4 * pos)) & 0xF  # :137
                        vals[4 + i] = (hi >> (4 * pos)) & 0xF  # :138
                    res = 0
                    for i in range(8):
                        res |= vals[PACK_IDX_UPSTREAM[i]] << (4 * i)  # :154
                    out[k_tile, n_tile * 128 + th_id * 4 + warp_id] = res
    return out


# =================================================================================================
# The shape lattice: one tile, partial tiles, several tiles, the checkpoint's own K and N, and the
# N < 64-free cases are refused by design (see the guard tests at the bottom).
# =================================================================================================
SHAPES = [(16, 64), (16, 128), (32, 64), (64, 64), (128, 64), (128, 128), (256, 128), (512, 256),
          (1536, 64), (1536, 2048), (8960, 1536)]


def _shape_id(shape) -> str:
    return f"k{shape[0]}_n{shape[1]}"


def codes_for(size_k: int, size_n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 16, size=(size_k, size_n), dtype=np.int64)


def _mate_repack(codes: np.ndarray, size_k: int, size_n: int, perm=None) -> np.ndarray:
    """MATE's shipped repack: codes -> GPTQ words -> Marlin words, as the loader does it."""
    words = REFERENCE.pack_rows(codes)
    return np.asarray(LAYOUT.repack_gptq(words, size_k, size_n, perm), dtype=np.int64)


# =================================================================================================
# Ground-truth gates
# =================================================================================================
@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_repack_matches_upstreams_own_oracle_bits(shape):
    """MATE's Marlin words must equal upstream's ``marlin_weights``, bit for bit.

    Independent signal: upstream's ``get_weight_perm``/``marlin_permute_weights``/``marlin_weights``,
    executed from source. Any wrong convention on the permute/pack axis -- a transposed ``pack_idx``,
    a wrong ``tc_offsets`` pairing, the ``+8`` half swapped, the tile flattened in the other order --
    moves at least one of the 32 bits of some word here.
    """
    size_k, size_n = shape
    codes = codes_for(size_k, size_n, seed=size_k * 7 + size_n)
    expected = GROUND_TRUTH(codes, size_k, size_n)
    got = _mate_repack(codes, size_k, size_n)
    assert got.shape == expected.shape == (size_k // 16, 2 * size_n)
    assert np.array_equal(got, expected), (
        f"MATE repack != upstream marlin_weights at {shape}: "
        f"{int(np.abs(got - expected).max())} max word difference over {got.size} words"
    )


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_repack_matches_a_simulation_of_the_cuda_kernel(shape):
    """The same comparison against the control-flow simulation (a different derivation path)."""
    size_k, size_n = shape
    codes = codes_for(size_k, size_n, seed=size_k * 13 + size_n)
    words = REFERENCE.pack_rows(codes)
    got = np.asarray(LAYOUT.repack_gptq(words, size_k, size_n), dtype=np.int64)
    expected = simulate_gptq_kernel(words, size_k, size_n)
    assert np.array_equal(got, expected), (
        f"MATE repack != CUDA-control-flow simulation at {shape}: "
        f"{int(np.abs(got - expected).max())} max word difference"
    )


@pytest.mark.parametrize("shape", SHAPES[:6], ids=_shape_id)
def test_act_order_gather_matches_upstreams_sorted_weight_path(shape):
    """The perm arm must equal upstream's *other* act-order route: sort the weight, then tile-permute.

    Independent signal: upstream has two implementations of act order -- the kernel's gather
    (``gptq_marlin_repack(..., perm=argsort(g_idx))``) and ``marlin_quantize``, which applies
    ``sort_weights`` to the codes and then the *fixed* 1024-entry tile permutation
    (``marlin_utils_test.py:151-161``). They must agree, so the oracle can check ``perm`` without
    ever taking a perm argument: ``repack(codes, perm) == oracle(codes[perm])``.
    """
    size_k, size_n = shape
    codes = codes_for(size_k, size_n, seed=size_k + size_n)
    perm = np.random.default_rng(size_k).permutation(size_k)
    got = _mate_repack(codes, size_k, size_n, perm)
    expected = GROUND_TRUTH(codes[perm], size_k, size_n)
    assert np.array_equal(got, expected), (
        f"act-order gather disagrees with sort_weights + fixed tile permutation at {shape}: "
        f"{int(np.abs(got - expected).max())} max word difference"
    )


@pytest.mark.parametrize("shape", SHAPES[:6], ids=_shape_id)
def test_awq_arm_matches_the_same_logical_frame(shape):
    """AWQ repack == AWQ kernel simulation == upstream oracle on the *logical* codes.

    Independent signal: the AWQ input words are re-interleaved before packing, and upstream's oracle
    knows nothing about that interleave, so this compares three things that each handle it
    differently: MATE's ``awq_marlin_repack``, the AWQ kernel simulation (``undo_pack``), and
    upstream's ``marlin_weights`` on the plain codes.
    """
    size_k, size_n = shape
    codes = codes_for(size_k, size_n, seed=2 * size_k + size_n)
    packed = REFERENCE.awq_pack(codes)  # quant_utils.awq_pack: interleave along N, then pack
    sim = simulate_awq_kernel(packed, size_k, size_n)
    mate = np.asarray(LAYOUT.awq_marlin_repack(packed, size_k, size_n), dtype=np.int64)
    oracle = GROUND_TRUTH(codes, size_k, size_n)
    assert np.array_equal(mate, sim), f"AWQ: MATE != AWQ kernel simulation ({size_k}, {size_n})"
    assert np.array_equal(mate, oracle), f"AWQ: MATE != upstream logical-frame oracle ({size_k}, {size_n})"


def test_upstreams_tile_permutation_is_a_permutation_of_1024_slots():
    """Pin the *frame* of the ground truth, so a future editor cannot silently swap frames.

    Measured: ``get_weight_perm(4)`` returns 1024 entries (one 16x64 tile flattened), not 16 and not
    ``size_k``. ``mate.marlin.layout`` accepts a 16-entry frame of its own as a test shorthand, and
    the 1024-entry frame is *not* accepted by it; both facts are pinned here because the docs in
    ``layout.resolve_perm`` used to call the 16-entry frame "the test frame from get_weight_perm",
    which is not what that function returns.
    """
    assert UPSTREAM_TILE_PERM.size == 16 * 64 == 1024
    assert sorted(int(v) for v in UPSTREAM_TILE_PERM) == list(range(1024))


@needs_upstream_checkout
@pytest.mark.parametrize("shape", SHAPES[:6], ids=_shape_id)
def test_repack_matches_the_oracle_extracted_from_the_live_checkout(shape):
    """Same comparison, but against upstream's source re-extracted from the checkout on this host."""
    sources = _upstream_sources_from_disk()
    assert sources is not None
    live_oracle, _ = _build_oracle(sources)
    size_k, size_n = shape
    codes = codes_for(size_k, size_n, seed=size_k * 3 + size_n)
    got = _mate_repack(codes, size_k, size_n)
    assert np.array_equal(got, live_oracle(codes, size_k, size_n))


@needs_upstream_checkout
def test_vendored_source_text_still_matches_the_upstream_file():
    """Provenance: the vendored ground truth is the upstream file's own text, sha256-pinned.

    This is the one gate that can be skipped on a host without an upstream checkout. It cannot
    weaken the others: they run against the vendored text either way, and a mismatch here means the
    vendored copy is stale, which this test reports as a failure rather than skipping.
    """
    sources = _upstream_sources_from_disk()
    assert sources is not None
    assert set(sources) == set(VENDORED_SHA256)
    drifted = {
        name: (hashlib.sha256(sources[name].encode()).hexdigest(), VENDORED_SHA256[name])
        for name in sources
        if hashlib.sha256(sources[name].encode()).hexdigest() != VENDORED_SHA256[name]
    }
    assert not drifted, (
        f"vendored upstream source is stale against the checkout (got, expected): {drifted}; "
        f"re-vendor from {_find_upstream('marlin_utils_test.py')} and update VENDORED_SHA256"
    )


# =================================================================================================
# Signal C: the gate bites. A comparison that cannot fail proves nothing; these perturbations are
# applied to MATE's own index map and must each be detected by the same comparison as above.
# =================================================================================================
def _perturbed_repack(codes, size_k, size_n, pack_idx=None, tc_offsets=None, swap_gather=False):
    """MATE's repack with one element of the convention changed, expressed on upstream's terms."""
    pack_idx = PACK_IDX_UPSTREAM if pack_idx is None else pack_idx
    tc_offsets = TC_OFFSETS_UPSTREAM if tc_offsets is None else tc_offsets
    words = REFERENCE.pack_rows(codes)
    w = REFERENCE.unpack_rows(words, size_k, size_n)
    out = np.zeros((size_k // 16, 2 * size_n), dtype=np.int64)
    for k_tile in range(size_k // 16):
        for n_tile in range(size_n // 64):
            for th_id in range(32):
                tc_col, tc_row = th_id // 4, (th_id % 4) * 2
                for warp_id in range(4):
                    vals = []
                    for half in (0, 1):
                        for i in range(4):
                            k = k_tile * 16 + tc_row + tc_offsets[i]
                            if swap_gather:
                                k = k_tile * 16 + tc_row + tc_offsets[(i + 1) % 4]
                            n = n_tile * 64 + warp_id * 16 + tc_col + 8 * half
                            vals.append(int(w[k, n]))
                    res = 0
                    for i in range(8):
                        res |= vals[pack_idx[i]] << (4 * i)
                    out[k_tile, n_tile * 128 + th_id * 4 + warp_id] = res
    return out


@pytest.mark.parametrize(
    "label,pack_idx,tc_offsets,swap_gather",
    [
        ("pack_idx 0<->1", (2, 0, 4, 6, 1, 3, 5, 7), None, False),
        ("pack_idx 1<->2", (0, 4, 2, 6, 1, 3, 5, 7), None, False),
        ("pack_idx halves swapped", (1, 3, 5, 7, 0, 2, 4, 6), None, False),
        ("tc_offsets 0<->1", None, (1, 0, 8, 9), False),
        ("tc_offsets 1<->2 (the +8 row group)", None, (0, 8, 1, 9), False),
        ("k pairs swapped in the register file", None, None, True),
    ],
)
def test_the_gate_bites_when_the_tile_map_is_perturbed(label, pack_idx, tc_offsets, swap_gather):
    """Each perturbation must be *detected*: the oracle comparison has to fail for it.

    This is the anti-tautology gate. It asserts ``not array_equal`` for a deliberately wrong map, so
    the passing comparisons above are known to be capable of failing.
    """
    size_k, size_n = 128, 64
    codes = codes_for(size_k, size_n, seed=99)
    wrong = _perturbed_repack(codes, size_k, size_n, pack_idx, tc_offsets, swap_gather)
    right = _perturbed_repack(codes, size_k, size_n)
    oracle = GROUND_TRUTH(codes, size_k, size_n)
    assert np.array_equal(right, oracle), "the perturbation harness itself drifted from the oracle"
    assert not np.array_equal(wrong, oracle), f"the gate did not detect: {label}"


# =================================================================================================
# Shape guards on the permute/pack axis. Each of these was silent before the guard: measured, the
# transposed-input case returned a shape-identical buffer with a 4.2e9 word difference.
# =================================================================================================
def test_guard_refuses_a_transposed_gptq_word_buffer():
    """Upstream checks the two extents of ``b_q_weight`` separately (gptq_marlin_repack.cu:300-305).

    ``(size_n, size_k/8)`` holds the same number of words as ``(size_k/8, size_n)``, so a
    product-only check -- or none -- unpacks the wrong axis and returns the right shape.
    """
    codes = codes_for(128, 64, seed=1)
    words = REFERENCE.pack_rows(codes)
    with pytest.raises(ValueError, match=r"GPTQ packed weight must be"):
        LAYOUT.repack_gptq(words.T, 128, 64)


def test_guard_refuses_a_marlin_buffer_in_the_wrong_frame():
    """Upstream allocates exactly ``(size_k/16, size_n*16/8)`` (gptq_marlin_repack.cu:324-326)."""
    codes = codes_for(512, 128, seed=2)
    repacked = LAYOUT.repack_gptq(REFERENCE.pack_rows(codes), 512, 128)
    with pytest.raises(ValueError, match=r"Marlin repacked weight must be"):
        _MATE.marlin_weight_to_musa(repacked, 256, 128)  # buffer is for K=512
    with pytest.raises(ValueError, match=r"Marlin repacked weight must be"):
        LAYOUT.unpack_gptq(REFERENCE.pack_rows(codes), 512, 128)  # GPTQ words, not Marlin words


def test_guard_refuses_an_awq_buffer_that_is_not_word_packed():
    """AWQ packs along N (awq_marlin_repack.cu:242-247), so the input is ``(size_k, size_n/8)``."""
    codes = codes_for(128, 64, seed=3)
    with pytest.raises(ValueError, match=r"AWQ packed weight must be"):
        LAYOUT.awq_marlin_repack(codes, 128, 64)  # logical nibbles, not packed words


def test_guard_refuses_a_scale_frame_with_the_wrong_group_count():
    """Scales are ``(size_k/group_size, size_n)`` (marlin_utils.py:470-480, :198-202).

    The 64-wide reshape succeeds for any row count whose product is a multiple of 64, so scales from
    another K used to permute the wrong groups and come back with a plausible shape: measured, this
    call returned a ``(2, 256)`` array where ``(8, 256)`` was meant.
    """
    scales = np.zeros((2, 256), dtype=np.float32)
    with pytest.raises(ValueError, match=r"scales must be \(num_groups, size_n\)"):
        REFERENCE.marlin_permute_scales(scales, 1024, 256, 128)
    with pytest.raises(ValueError, match=r"size_k=1024 is not divisible by group_size"):
        REFERENCE.marlin_permute_scales(np.zeros((8, 256), dtype=np.float32), 1024, 256, 100)
    with pytest.raises(ValueError, match=r"scales must be \(num_groups, size_n\)"):
        REFERENCE.marlin_unpermute_scales(np.zeros((2, 256), dtype=np.float32), 1024, 256, 128)
    # linear._unpermute is the serving path's inverse and is numpy-compatible, so its guard is
    # exercised here as well (its torch wrapper marlin_scales_to_musa is not).
    with pytest.raises(ValueError, match=r"scales must be \(num_groups, size_n\)"):
        LINEAR._unpermute(np.zeros((2, 256), dtype=np.float32), 1024, 256, 128)


def test_torch_entry_points_check_the_frame_before_touching_tensors():
    """The torch wrappers need torch, so pin the *order* of their guard statically.

    A frame check placed after the first reshape would be worthless -- the reshape is what silently
    accepts the wrong frame. Keeping the check first is what makes these paths fail closed on a torch
    host, which is the only host where they can be executed (see the report's coverage note).
    """
    import inspect

    tensor_ops = (".reshape(", ".t()", ".contiguous()", "unpack_marlin_zero_points(")
    for func, guard_name in [
        (QUANT.marlin_permute_scales, "require_scale_frame"),
        (QUANT.marlin_zero_points, "require_zero_point_frame"),
        (LINEAR.marlin_zero_points_to_musa, "require_packed_zero_point_frame"),
    ]:
        lines = inspect.getsource(func).splitlines()
        guard_at = next(i for i, line in enumerate(lines) if guard_name in line)
        first_op = next(i for i, line in enumerate(lines) if any(op in line for op in tensor_ops))
        assert guard_at < first_op, f"{func.__name__}: {guard_name} must precede {lines[first_op].strip()}"


def test_guard_refuses_a_zero_point_frame_that_cannot_be_permuted():
    """Zero points are permuted in 64-wide blocks and packed along N (marlin_utils.py:512-533)."""
    with pytest.raises(ValueError, match=r"zero points are permuted in 64-wide blocks"):
        REFERENCE.marlin_zero_points(np.zeros((8, 24), dtype=np.int64), 1024, 24, 4)
    with pytest.raises(ValueError, match=r"packed zero points must be"):
        REFERENCE.unpack_marlin_zero_points(np.zeros((8, 12), dtype=np.int64), 512, 64, 4)


def test_guard_refuses_the_wrong_nibble_axis():
    """The packed axis is K for GPTQ and N for AWQ (quant_utils.py:871 and :895)."""
    with pytest.raises(ValueError, match=r"the GPTQ word axis packs 8 k-nibbles per word"):
        REFERENCE.pack_rows(np.zeros((12, 64), dtype=np.int64))
    with pytest.raises(ValueError, match=r"the AWQ word axis packs 8 n-nibbles per word"):
        REFERENCE.pack_cols(np.zeros((16, 12), dtype=np.int64))


def test_guard_refuses_a_perm_that_is_not_a_k_frame():
    """Act order is one entry per k, values in ``[0, size_k)`` (gptq_marlin_repack.cu:62,151)."""
    codes = codes_for(128, 64, seed=4)
    words = REFERENCE.pack_rows(codes)
    # A length matching no upstream frame is a frame error, not a value error (the op frame is one
    # entry per k; the python oracle's single frame is 1024 entries).
    with pytest.raises(ValueError, match=r"perm has 4 entries; expected 128"):
        LAYOUT.repack_gptq(words, 128, 64, np.array([0, 1, 2, 3]))
    # A right-length frame whose *values* leave [0, size_k) reads a k row from another tile.
    with pytest.raises(ValueError, match=r"act-order perm entries are source k rows"):
        LAYOUT.repack_gptq(words, 128, 64, np.arange(128) + 1)
    with pytest.raises(ValueError, match=r"act-order perm must have one entry per k"):
        GUARDS.require_act_order_perm(np.zeros(4, dtype=np.int64), 128)
    # A perm that stays inside [0, size_k) but leaves the tile: legal for the op frame (upstream's
    # gather is global, :151), and the guard must still accept it -- this one is a k-tile shift.
    tile_shift = np.concatenate([np.arange(112, 128), np.arange(112)])
    assert LAYOUT.repack_gptq(words, 128, 64, tile_shift).shape == (8, 128)
    # upstream's "no act order" is an empty perm, not an error (gptq_marlin_repack.cu:329)
    assert np.array_equal(
        LAYOUT.repack_gptq(words, 128, 64, np.zeros(0, dtype=np.int64)),
        LAYOUT.repack_gptq(words, 128, 64),
    )


def test_guard_refuses_tile_unaligned_shapes():
    """K % 16 and N % 64 are upstream's own host checks (gptq_marlin_repack.cu:290-293)."""
    words = REFERENCE.pack_rows(codes_for(64, 64, seed=5))
    with pytest.raises(ValueError, match=r"size_k % 16 == 0"):
        LAYOUT.repack_gptq(REFERENCE.pack_rows(codes_for(64, 64, seed=5))[:5], 40, 64)
    with pytest.raises(ValueError, match=r"size_n % 64 == 0"):
        LAYOUT.repack_gptq(words[:, :32], 64, 32)


def test_port_k_multiple_guard_is_enforced_at_the_op_boundary():
    """``K % 256 == 0`` is MATE's port requirement (guard docstring; compat.py), not upstream's.

    Upstream tile-pads K to ``lcm(64 or 128, group_size)`` (marlin_utils.py:221-243) and its repack
    has no K % 256 rule, so the guard lives at the op boundary that MATE's kernel actually needs --
    and the guard's own message is what the caller reports.
    """
    ok, reason = COMPAT.check_musa_marlin_supports_shape(4480, 1536, 128)  # TP=2 down_proj
    assert not ok and reason is not None
    assert "multiple of 256" in reason and "4480" in reason
    assert COMPAT.check_musa_marlin_supports_shape(1536, 1536, 128) == (True, None)
    with pytest.raises(ValueError, match=r"multiple of 256"):
        GUARDS.require_musa_k_multiple(4480)
    # the upstream-mirroring repack must NOT enforce it: K=384 is a legal repack shape upstream.
    assert LAYOUT.repack_gptq(REFERENCE.pack_rows(codes_for(384, 64, seed=6)), 384, 64).shape == (24, 128)
