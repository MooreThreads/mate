from .fmha_get_metadata import (
    _fmha_get_metadata,  # noqa: F401
    gen_fmha_metadata_aot,
)
from .fmha_fwd import (
    _fmha_fwd,  # noqa: F401
    gen_fmha_fwd_aot,
)

from .fmha_combine import (
    _fmha_fwd_combine,  # noqa: F401
    gen_fmha_fwd_combine_aot,
)


def gen_fmha_aot(dry_run: bool = False) -> int:
    num_metadata = gen_fmha_metadata_aot(dry_run=dry_run)
    num_combine = gen_fmha_fwd_combine_aot(dry_run=dry_run)
    num_fwd = gen_fmha_fwd_aot(dry_run=dry_run)
    return num_metadata + num_combine + num_fwd


__all__ = [
    "_fmha_get_metadata",
    "_fmha_fwd",
    "_fmha_fwd_combine",
    "gen_fmha_aot",
]
