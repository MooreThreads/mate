from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Mapping, SupportsInt, cast

import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from packaging.version import Version

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .cpp_ext import get_mcc_version
from .utils import dtype_torch2mutlass_map


CXX_FLAGS = ["-O3", "-Wno-switch-bool"]
CUDA_FLAGS = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]

# The MP31 recurrence keeps five 2048-bit SQMMA accumulators alive -- its state
# fragment alone is a 64x128 fp32 fragment, i.e. 64 dword per thread in the one
# state squad -- which needs more than the 113 dword/thread that recurrence
# CTAs/MP = 3 allows.  Backends of the 5.1.0 line cannot allocate them at all
# and fail with "TEMP_2048"; halving the CTAs/MP halves the demand.  Keyed on
# the compiler version exactly like the FMHA BST overlap workaround
# (mate/jit/attention/fmha/fmha_fwd.py), so a 5.3.0-line compiler keeps the fast
# configuration.
_MCC_MP31_KDA_REGPRESSURE_CUTOFF = Version("5.3.0")


def _get_kda_extra_cuda_cflags() -> list[str]:
    flags = list(CUDA_FLAGS)
    mcc_version = get_mcc_version()
    if mcc_version is not None and mcc_version < _MCC_MP31_KDA_REGPRESSURE_CUTOFF:
        flags.append("-DMATE_KDA_MP31_REC_CTAS_PER_MP=2")
    return flags


INCLUDE_PATHS = [
    jit_env.MATE_INCLUDE_DIR,
    jit_env.MATE_CSRC_DIR,
    jit_env.MUTLASS_INCLUDE_DIR,
]

KDA_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}
KDA_DTYPE_FFI = {
    torch.float16: "dl_float16",
    torch.bfloat16: "dl_bfloat16",
}
# The recurrent state pool may also be fp32, so it needs its own FFI map: the
# activation whitelist above must stay fp16/bf16-only.
KDA_STATE_DTYPE_FFI = {
    torch.float16: "dl_float16",
    torch.bfloat16: "dl_bfloat16",
    torch.float32: "dl_float32",
}
KDA_CU_SEQLENS_ELEMENT_NAMES = {torch.int32: "int32_t", torch.int64: "int64_t"}
KDA_CU_SEQLENS_DTYPE_NAMES = {torch.int32: "i32", torch.int64: "i64"}
KDA_CU_SEQLENS_DTYPE_FFI = {torch.int32: "dl_int32", torch.int64: "dl_int64"}

# Logical KDA prefill ABI shared by architecture backends.  SQMMA layouts and
# launch occupancy remain backend-specific.
KDA_PREFILL_CHUNK_SIZE = 16
KDA_PREFILL_HEAD_DIM = 128
KDA_MP31_PREPARE_CTAS_PER_MP = 5

# The prepare collective is templated on ArchTag; resolve the tag from the
# same architecture list that drives the mcc --offload-arch flags.  This
# release line carries the MP31 (PH1/QY2) collective only.
KDA_ARCH_TAGS = {
    "mp31": "mutlass::arch::Mp31",
}

# The prepare launch policy belongs to the arch the kernel was *built* for
# (ChunkKdaMp31SchedulePolicy in chunk_kda_mp31_policy.hpp), not to the
# device it happens to run on.  The metadata region is sized from it, so it
# has to come from the same resolver that renders KDA_ARCH_TAGS: otherwise a
# cross build sizes the region for one arch and schedules for the other.
KDA_PREPARE_CTAS_PER_MP = {
    "mp31": KDA_MP31_PREPARE_CTAS_PER_MP,
}

KDA_BOOL_KEYS = (
    "has_state_in",
    "has_state_out",
    "has_chunk_states_out",
    "state_fp32",
    "has_gate_params",
    "is_varlen",
    "normalize_qk",
)
KDA_CONFIG_KEYS = (
    "element",
    "element_dtype",
    "dtype_name",
    "state_element",
    "state_dtype_name",
    "cu_seqlens_element",
    "cu_seqlens_dtype",
    "cu_seqlens_dtype_name",
    *KDA_BOOL_KEYS,
    # Compile-time checkpoint cadence; see `Tag::CheckpointEveryNTokens`.  It is
    # a recurrence specialization axis (0 = the packed `chunk_states` rows), so
    # it participates in the recurrence name and cache key.  The prepare template
    # does not read it and `gen_kda_prepare_ops_spec` zeroes it back out.
    "checkpoint_every_n_tokens",
)
KDA_DEFAULT_CONFIG = {
    "element": dtype_torch2mutlass_map[torch.bfloat16],
    "element_dtype": KDA_DTYPE_FFI[torch.bfloat16],
    "dtype_name": KDA_DTYPE_NAMES[torch.bfloat16],
    "state_element": dtype_torch2mutlass_map[torch.bfloat16],
    "state_dtype_name": KDA_DTYPE_NAMES[torch.bfloat16],
    "cu_seqlens_element": KDA_CU_SEQLENS_ELEMENT_NAMES[torch.int64],
    "cu_seqlens_dtype": KDA_CU_SEQLENS_DTYPE_FFI[torch.int64],
    "cu_seqlens_dtype_name": KDA_CU_SEQLENS_DTYPE_NAMES[torch.int64],
    "has_state_in": True,
    "has_state_out": True,
    "has_chunk_states_out": False,
    "state_fp32": False,
    "has_gate_params": True,
    "is_varlen": False,
    "normalize_qk": True,
    "checkpoint_every_n_tokens": 0,
}


@functools.lru_cache
def _get_kda_template_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _get_kda_template(template_name: str):
    template_dir = (jit_env.MATE_TEMPLATE_DIR / "flat" / "kda").as_posix()
    return _get_kda_template_env(template_dir).get_template(template_name)


KDA_PREPARE_TEMPLATE = _get_kda_template("chunk_kda_prepare_kernel.j2")
KDA_RECURRENCE_TEMPLATE = _get_kda_template("chunk_kda_recurrence_kernel.j2")


def _normalize_checkpoint_cadence(value: object) -> int:
    """Coerce and validate a config's checkpoint cadence.

    The value arrives as ``object`` from a plain config mapping, so the cast is
    what keeps ``int()`` mypy-clean while the call itself keeps the runtime
    coercion (the same shape ``mate/aot.py`` uses for its int config fields).
    This is the host-side gate: a cadence that is negative or not a multiple of
    the kernel chunk never reaches the renderer.
    """
    cadence = int(cast(SupportsInt, value))
    if cadence < 0 or cadence % KDA_PREFILL_CHUNK_SIZE != 0:
        raise ValueError(
            "checkpoint_every_n_tokens must be a non-negative multiple of the "
            f"kernel chunk ({KDA_PREFILL_CHUNK_SIZE}), got {cadence}"
        )
    return cadence


def make_kda_ops_config(
    dtype: torch.dtype,
    *,
    state_dtype: torch.dtype,
    cu_seqlens_dtype: torch.dtype | None = None,
    has_state_in: bool,
    has_state_out: bool,
    has_chunk_states_out: bool = False,
    checkpoint_every_n_tokens: int = 0,
    state_fp32: bool,
    has_gate_params: bool,
    is_varlen: bool,
    normalize_qk: bool,
) -> dict[str, object]:
    if dtype not in KDA_DTYPE_FFI:
        raise TypeError("chunk_kda supports torch.float16 and torch.bfloat16 inputs")
    if state_dtype not in KDA_DTYPE_NAMES:
        raise TypeError("chunk_kda state must be fp16, bf16, or fp32")
    cu_seqlens_dtype = cu_seqlens_dtype or torch.int64
    if cu_seqlens_dtype not in KDA_CU_SEQLENS_DTYPE_FFI:
        raise TypeError("chunk_kda cu_seqlens must be int32 or int64")
    cadence = _normalize_checkpoint_cadence(checkpoint_every_n_tokens)
    return {
        "element": dtype_torch2mutlass_map[dtype],
        "element_dtype": KDA_DTYPE_FFI[dtype],
        "dtype_name": KDA_DTYPE_NAMES[dtype],
        "state_element": dtype_torch2mutlass_map[state_dtype],
        "state_dtype_name": KDA_DTYPE_NAMES[state_dtype],
        "cu_seqlens_element": KDA_CU_SEQLENS_ELEMENT_NAMES[cu_seqlens_dtype],
        "cu_seqlens_dtype": KDA_CU_SEQLENS_DTYPE_FFI[cu_seqlens_dtype],
        "cu_seqlens_dtype_name": KDA_CU_SEQLENS_DTYPE_NAMES[cu_seqlens_dtype],
        "has_state_in": bool(has_state_in),
        "has_state_out": bool(has_state_out),
        "has_chunk_states_out": bool(has_chunk_states_out),
        "checkpoint_every_n_tokens": cadence,
        "state_fp32": bool(state_fp32),
        "has_gate_params": bool(has_gate_params),
        "is_varlen": bool(is_varlen),
        "normalize_qk": bool(normalize_qk),
    }


def _normalize_kda_ops_config(
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized = dict(KDA_DEFAULT_CONFIG)
    if config is not None:
        normalized.update(config)
    for key in KDA_CONFIG_KEYS:
        if key not in normalized:
            raise ValueError(f"missing KDA JIT config key: {key}")
    for key in KDA_BOOL_KEYS:
        normalized[key] = bool(normalized[key])
    # The recurrence template only needs this value for its chunk-state ABI
    # guard.  It is not an independent specialization axis: all state tensors
    # are required to have one dtype, and ``state_dtype_name`` is the source of
    # truth for that dtype.  Derive it here so callers that construct a config
    # directly cannot make the rendered source disagree with the recurrence
    # name/cache key (which intentionally omit ``state_fp32``).
    normalized["state_fp32"] = normalized["state_dtype_name"] == "fp32"
    # The cadence is a compile-time specialization axis, so it is validated here
    # once for every path that renders a kernel (the Python entry point rejects
    # user input as well; this is the backstop that also covers direct config
    # construction).  It is meaningless without the pool, so normalize it away
    # there to keep the feature-off name and cache key unchanged.
    cadence = _normalize_checkpoint_cadence(normalized["checkpoint_every_n_tokens"])
    normalized["checkpoint_every_n_tokens"] = (
        cadence if normalized["has_chunk_states_out"] else 0
    )
    if not normalized["is_varlen"]:
        for key in (
            "cu_seqlens_element",
            "cu_seqlens_dtype",
            "cu_seqlens_dtype_name",
        ):
            normalized[key] = KDA_DEFAULT_CONFIG[key]
    return {key: normalized[key] for key in KDA_CONFIG_KEYS}


def _visible_device_arch() -> str | None:
    try:
        if torch.musa.is_available() and torch.musa.device_count() > 0:
            props = torch.musa.get_device_properties(torch.musa.current_device())
            return f"mp{int(props.major)}{int(props.minor)}"
    except Exception:
        return None
    return None


def resolve_kda_arch() -> str:
    """The one KDA architecture decision: ``"mp31"``.

    Everything that depends on the architecture -- the ArchTag rendered into
    the source, the JIT cache key, and the launch policy the host sizes
    buffers from -- must come from here, so that a cross build cannot make
    two of them disagree.
    """
    # ArchTag describes the GPU the kernel will run on, and it must agree with
    # the mcc --offload-arch flags: an Mp31-tagged source compiled for an mp_32
    # target selects the MP31 SQMMA, whose instruction bodies are guarded by
    # MUTE_ARCH_SQMMA_MP31_ENABLED and therefore compiled out under
    # __MUSA_ARCH__ == 320. The gemm then emits nothing and MTGPU-DEPENDENCY-GRAPH
    # rejects the resulting empty TCE commit group.
    #
    # An explicit MATE_MUSA_ARCH_LIST wins, because that is the only way to
    # build for a target other than the locally visible device - cross
    # building, or validating an MP32 build from an MP31 host. It is also the
    # same variable that drives the --offload-arch flags, so honouring it keeps
    # the two in step by construction.
    if os.environ.get("MATE_MUSA_ARCH_LIST", "").strip():
        arch_list = jit_env.resolve_musa_arch_list()
    else:
        # Otherwise the visible device decides. Note that an MP32 device also
        # executes mp31-coded binaries in compatibility mode, so there is no
        # case for preferring a lower arch tag than the device reports.
        device_arch = _visible_device_arch()
        if device_arch in KDA_ARCH_TAGS:
            return device_arch
        arch_list = jit_env.resolve_musa_arch_list()
    for arch in reversed(sorted(KDA_ARCH_TAGS)):
        if arch in arch_list:
            return arch
    raise ValueError(
        f"chunk_kda supports {sorted(KDA_ARCH_TAGS)}, got resolved arch list {arch_list}; "
        "this branch carries the MP31 implementation only, so build it for the "
        "MP31 target explicitly (MATE_MUSA_ARCH_LIST=3.1)."
    )


def resolve_kda_prepare_ctas_per_mp() -> int:
    """Prepare CTAs per MP for the resolved arch's launch policy."""
    return KDA_PREPARE_CTAS_PER_MP[resolve_kda_arch()]


def _resolve_kda_arch_tag() -> str:
    return KDA_ARCH_TAGS[resolve_kda_arch()]


def _render_config(config: Mapping[str, object], func_name: str) -> dict[str, object]:
    render_config = dict(config)
    render_config["func_name"] = func_name
    render_config["arch_tag"] = _resolve_kda_arch_tag()
    # The chunk-boundary pool is always activation-typed: the kernel publishes the
    # activation-typed state tile, so the pool's element type follows the
    # activation, not the recurrent state.  The checkpoint cadence therefore allows
    # an fp32 recurrent state (`checkpoint_every_n_tokens` matches cudnn's
    # io-dtype `state_checkpoints`), while the caller-owned legacy `chunk_states`
    # pool keeps its "dtype == state dtype" rule in `mate.kda.chunk_kda`.  The
    # rendered kernel enforces the pool element type with
    # `CHECK_INPUT_TYPE(cs, <element>)` and refuses the legacy pool under an fp32
    # state at run time.
    render_config["chunk_size"] = KDA_PREFILL_CHUNK_SIZE
    render_config["head_dim"] = KDA_PREFILL_HEAD_DIM
    # Derived, not a config key: the template needs the FFI tag to validate the
    # caller-owned chunk-state pool against the state dtype.
    render_config["state_dtype_ffi"] = {
        "fp16": KDA_STATE_DTYPE_FFI[torch.float16],
        "bf16": KDA_STATE_DTYPE_FFI[torch.bfloat16],
        "fp32": KDA_STATE_DTYPE_FFI[torch.float32],
    }[str(config["state_dtype_name"])]
    for key in KDA_BOOL_KEYS:
        render_config[key] = "true" if bool(config[key]) else "false"
    # Rendered mode flag, derived from the (already host-validated) cadence: the
    # kernel branches on this instead of comparing the cadence against zero, so a
    # non-positive cadence is purely a host-side error.  `_normalize_kda_ops_config`
    # zeroes the cadence without the pool, so this cannot disagree with it.
    render_config["has_checkpoint_cadence"] = (
        "true" if cast(int, config["checkpoint_every_n_tokens"]) > 0 else "false"
    )
    return render_config


def _common_suffix(config: Mapping[str, object]) -> str:
    seqlens = (
        f"_seqlens_{config['cu_seqlens_dtype_name']}"
        if bool(config["is_varlen"])
        else ""
    )
    return (
        f"dtype_{config['dtype_name']}"
        f"_gate_{int(bool(config['has_gate_params']))}"
        f"_varlen_{int(bool(config['is_varlen']))}"
        f"_normqk_{int(bool(config['normalize_qk']))}{seqlens}"
    )


def _recurrence_suffix(config: Mapping[str, object]) -> str:
    """Name only the axes rendered by the recurrence template.

    Gate parameters and Q/K normalization belong to prepare.  They are not
    referenced by ``chunk_kda_recurrence_kernel.j2`` and must not create four
    duplicate recurrence modules for the same state/workspace specialization.
    ``state_fp32`` is likewise implied by ``state_dtype_name``.
    """
    seqlens = (
        f"_seqlens_{config['cu_seqlens_dtype_name']}"
        if bool(config["is_varlen"])
        else ""
    )
    # The checkpoint cadence only exists alongside the pool, and it changes the
    # rendered row mapping, so it is part of the recurrence name.  A cadence of 0
    # (the packed `chunk_states` rows) keeps the historical name.
    cadence = cast(int, config["checkpoint_every_n_tokens"])
    cadence_token = f"_ck{cadence}" if cadence > 0 else ""
    return (
        f"dtype_{config['dtype_name']}"
        f"_varlen_{int(bool(config['is_varlen']))}{seqlens}"
        f"_state_{config['state_dtype_name']}"
        f"_in_{int(bool(config['has_state_in']))}"
        f"_out_{int(bool(config['has_state_out']))}"
        f"_cs_{int(bool(config['has_chunk_states_out']))}"
        f"{cadence_token}"
    )


# The arch is part of the name, not just of the rendered source: the name is
# what the generated source path, the AOT path, the spec registry key and the
# module cache all derive from.  Without it a process that resolves two arches
# shares one generated .mu and hands back one arch's module for the other.
def get_kda_prepare_ops_function_name(
    config: Mapping[str, object] | None = None,
) -> str:
    config = _normalize_kda_ops_config(config)
    return f"prepare_kda_{_common_suffix(config)}_arch_{resolve_kda_arch()}"


def get_kda_recurrence_ops_function_name(
    config: Mapping[str, object] | None = None,
) -> str:
    config = _normalize_kda_ops_config(config)
    return f"kda_recurrence_{_recurrence_suffix(config)}_arch_{resolve_kda_arch()}"


def _gen_spec(
    config: Mapping[str, object],
    *,
    func_name: str,
    template,
) -> JitSpec:
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / "kda" / f"{func_name}.mu")
    return gen_jit_spec(
        func_name,
        [source_file],
        generated_sources={
            source_file: template.render(_render_config(config, func_name))
        },
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=_get_kda_extra_cuda_cflags(),
        extra_include_paths=list(INCLUDE_PATHS),
    )


def gen_kda_prepare_ops_spec(config: Mapping[str, object] | None = None) -> JitSpec:
    config = _normalize_kda_ops_config(config)
    config.update(
        state_element=config["element"],
        state_dtype_name=config["dtype_name"],
        has_state_in=False,
        has_state_out=False,
        has_chunk_states_out=False,
        state_fp32=False,
        # Recurrence-only axis: the prepare template never reads it.
        checkpoint_every_n_tokens=0,
    )
    return _gen_spec(
        config,
        func_name=get_kda_prepare_ops_function_name(config),
        template=KDA_PREPARE_TEMPLATE,
    )


def gen_kda_recurrence_ops_spec(config: Mapping[str, object] | None = None) -> JitSpec:
    config = _normalize_kda_ops_config(config)
    return _gen_spec(
        config,
        func_name=get_kda_recurrence_ops_function_name(config),
        template=KDA_RECURRENCE_TEMPLATE,
    )


def gen_kda_ops_aot() -> list[JitSpec]:
    return [gen_kda_prepare_ops_spec(), gen_kda_recurrence_ops_spec()]


def _module_cache_key(
    config: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """The per-process JIT module cache key.

    The arch leads it because one logical config builds a different kernel
    per arch and this cache lives for the whole process.  The loaders drop the
    extra entry again in ``_normalize_kda_ops_config``.
    """
    return (("arch", resolve_kda_arch()), *sorted(config.items()))


def _recurrence_module_cache_key(
    config: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """Cache key for the recurrence's rendered inputs only.

    Keep the full normalized config for the prepare key, but drop controls that
    do not occur in the recurrence template.  This mirrors the function name
    and prevents a process-local cache from retaining four equivalent K2
    modules.
    """
    recurrence_only = {
        key: value
        for key, value in config.items()
        if key not in {"has_gate_params", "normalize_qk", "state_fp32"}
    }
    return _module_cache_key(recurrence_only)


@functools.cache
def _load_prepare(frozen_config: tuple[tuple[str, object], ...]):
    return gen_kda_prepare_ops_spec(dict(frozen_config)).build_and_load()


@functools.cache
def _load_recurrence(frozen_config: tuple[tuple[str, object], ...]):
    return gen_kda_recurrence_ops_spec(dict(frozen_config)).build_and_load()


def get_kda_prepare_ops_module(config: Mapping[str, object] | None = None):
    normalized = _normalize_kda_ops_config(config)
    normalized.update(
        state_element=normalized["element"],
        state_dtype_name=normalized["dtype_name"],
        has_state_in=False,
        has_state_out=False,
        has_chunk_states_out=False,
        state_fp32=False,
        checkpoint_every_n_tokens=0,
    )
    return _load_prepare(_module_cache_key(normalized))


def get_kda_recurrence_ops_module(config: Mapping[str, object] | None = None):
    normalized = _normalize_kda_ops_config(config)
    return _load_recurrence(_recurrence_module_cache_key(normalized))
