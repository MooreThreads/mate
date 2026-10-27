from dataclasses import asdict
import functools
import hashlib
import json
from pathlib import Path

from mate.artifacts import ensure_mubin_kernel_artifact, load_kernel_map

from ..common import MP31_ARCH, get_asm_dtype_from_torch_dtype
from .types import FlashAttentionBwdMubinId


@functools.cache
def get_flash_attention_bwd_mubin_id_hash(asm_id):
    payload = asdict(asm_id)
    # Preserve the original packed/non-persistent dispatch hashes.
    if not payload["is_persistence"]:
        payload.pop("is_persistence")
    payload["dtype"] = get_asm_dtype_from_torch_dtype(asm_id.dtype)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class FlashAttentionBwdMubinDispatcher:
    def __init__(self, kernel_map_path: Path):
        self.module_dir = Path(kernel_map_path).parent
        self.entries = {
            entry.dispatch_hash: entry for entry in load_kernel_map(kernel_map_path)
        }

    def get_asm_id(self, dtype, is_causal, headdim_qk, is_gqa, *, is_persistence=False):
        return FlashAttentionBwdMubinId(
            MP31_ARCH,
            dtype,
            is_causal,
            128 if headdim_qk <= 128 else headdim_qk,
            is_gqa,
            is_persistence,
        )

    def resolve_kernel_entry(self, asm_id):
        entry = self.entries.get(get_flash_attention_bwd_mubin_id_hash(asm_id))
        if entry is None:
            raise ValueError(f"No FlashAttention backward MUBIN kernel for {asm_id}")
        return entry

    def resolve_kernel_path(self, asm_id):
        return ensure_mubin_kernel_artifact(
            "flash_attention_bwd", self.module_dir, self.resolve_kernel_entry(asm_id)
        )
