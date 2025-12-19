# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

KERNEL_BATCH = namedtuple("KERNEL_BATCH", ["template", "filename"])

DTYPE_MAP = {
    "fp16": "mutlass::half_t",
    "bf16": "mutlass::bfloat16_t",
}

MP = [31]  # Mp kernels support up to
TILE_SHAPES = [
    [256, 128, 128, 128],
    [256, 64, 192, 128],
]

TILE_SHAPES_PAGED = [
    [256, 64, 128, 128],
]

SPLIT = [False]
CAUSAL = [True, False]
VERLEN = [True, False]
# PAGEDKV = [True, False]

KERNEL_IMPL_TEMPLATE_FWD_MP31 = """#include "../flash_fwd_launch_template.hpp"

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}_{HEAD_DIM_V}
template void dispatch_fmha_kernel<{ARCH}, {DTYPE}, {DTYPE}, {CAUSAL}, {VERLEN}, {CTA_Q}, {CTA_KV}, {HEAD_DIM}, {HEAD_DIM_V}, {SPLIT}, {PAGEDKV}>(Flash_fwd_params params);
#endif
"""


@dataclass
class Kernel:
    mp: int
    dtype: str
    cta_q: int
    cta_kv: int
    head_dim: int
    head_dim_v: int
    split: bool
    pagedkv: bool
    verlen: bool
    causal: bool
    direction: str

    @property
    def template(self) -> str:
        # Always enable PackGQA for PagedKV or Split to reduce compilation
        return KERNEL_IMPL_TEMPLATE_FWD_MP31.format(
            ARCH=str(self.mp),
            DTYPE=DTYPE_MAP[self.dtype],
            CTA_Q=self.cta_q,
            CTA_KV=self.cta_kv,
            HEAD_DIM=self.head_dim,
            HEAD_DIM_V=self.head_dim_v,
            SPLIT=str(self.split).lower(),
            PAGEDKV=str(self.pagedkv).lower(),
            VERLEN=str(self.verlen).lower(),
            CAUSAL=str(self.causal).lower(),
        )

    @property
    def filename(self) -> str:
        return f"flash_{self.direction}_hdim{self.head_dim}{f'_{self.head_dim_v}' if self.head_dim_v != self.head_dim else ''}_{self.dtype}{'_verlen' if self.verlen else ''}{'_split' if self.split else ''}{'_pagedkv' if self.pagedkv else ''}{'_causal' if self.causal else ''}_mp{self.mp}.mu"


def get_all_kernels() -> Iterator[Kernel]:
    for dtype, (
        cta_q,
        cta_kv,
        head_dim,
        head_dim_v,
    ), split, verlen, causal, mp in itertools.product(
        DTYPE_MAP.keys(), TILE_SHAPES, SPLIT, VERLEN, CAUSAL, MP
    ):
        yield Kernel(
            mp=mp,
            dtype=dtype,
            cta_q=cta_q,
            cta_kv=cta_kv,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            split=split,
            pagedkv=False,
            verlen=verlen,
            causal=causal,
            direction="fwd",
        )

    for dtype, (
        cta_q,
        cta_kv,
        head_dim,
        head_dim_v,
    ), split, verlen, causal, mp in itertools.product(
        DTYPE_MAP.keys(), TILE_SHAPES_PAGED, SPLIT, VERLEN, CAUSAL, MP
    ):
        yield Kernel(
            mp=mp,
            dtype=dtype,
            cta_q=cta_q,
            cta_kv=cta_kv,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            split=split,
            pagedkv=True,
            verlen=verlen,
            causal=causal,
            direction="fwd",
        )


def batch_hdim(kernels_all) -> Iterator[KERNEL_BATCH]:
    if len(kernels_all) > 0:
        filename = "flash_fwd_mp31.mu"
        template = "\n".join([f'#include "{k.filename}"' for k in kernels_all])
        yield KERNEL_BATCH(template, filename)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir: Optional[str]) -> None:
    outdir: Path = Path(output_dir) if output_dir is not None else Path(__file__).parent
    outdir.mkdir(parents=True, exist_ok=True)

    kernels_all = list(get_all_kernels())
    for kernel in kernels_all:
        write_kernel(kernel, outdir)
    # for kernel in batch_hdim(kernels_all):
    #     write_kernel(kernel, output_dir)
    # for kernel in batch_softcap(kernels_all):
    #     write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        default="instantiations",
        required=False,
        help="Where to generate the kernels  will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
