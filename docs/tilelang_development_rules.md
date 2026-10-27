# TileLang Development Rules

These rules apply when writing or modifying MATE TileLang kernels and their
tensor interfaces. Review both the kernel declaration and the caller that
provides its tensors. Examples use `import tilelang.language as T` and the
TileLang MUSA APIs supported by MATE.

## 1. Describe tensor strides explicitly

Use `T.StridedTensor` by default for tensor arguments, including outputs.
Declare the actual strides of the view passed to the kernel; do not infer a
dense layout from its shape.

For example, declare a general two-dimensional input inside the kernel factory:

```python
import tilelang.language as T

m, n = T.dynamic("m, n")
stride_m, stride_n = T.dynamic("stride_m, stride_n", "int64")
Input = T.StridedTensor((m, n), (stride_m, stride_n), "float16")
```

Use `Input` as the argument annotation in the factory's `@T.prim_func`.

`T.Tensor` is allowed only when the allocation or an enforced interface contract
guarantees dense, contiguous storage in the declared dimension order. For
example, an output allocated by the caller with `torch.empty((m, n), ...)` can
use `T.Tensor((m, n), "float16")` if it is passed without changing that layout.
Document the guarantee and validate caller-provided tensors when needed.

A last-dimension stride of one does not prove that the whole tensor is
contiguous. Row padding, sliced views, and transposes must retain their actual
strides. Do not introduce a `.contiguous()` copy just to avoid describing a
layout the kernel is intended to support.

### Expose guaranteed layout properties with `T.assume`

Use `T.assume` to make known stride relationships and divisibility visible to
the compiler so it can enable vectorized memory accesses while retaining
dynamic strides. For a path whose caller guarantees unit inner stride and a
row stride divisible by eight elements, add these assumptions inside the
`@T.prim_func` before the relevant accesses:

```python
# stride_m and stride_n are the dynamic symbols declared above.
# The caller must guarantee both conditions before launching this path.
T.assume(stride_n == 1)
T.assume(stride_m % 8 == 0)
```

For float16, eight elements correspond to 16 bytes. The multiple must match the
dtype and intended access width; it is not a universal requirement. The view's
base address and access offsets must also satisfy the required alignment.

Every assumption must follow from an enforced input contract, an allocation
guarantee, or a validated dispatch condition. `T.assume` supplies facts to the
compiler; it does not perform a runtime check or fix an incompatible layout.
Reject inputs that violate the contract or route them to an implementation
that supports their layout. Do not invent alignment or divisibility assumptions
to force vectorization, and retain the masks needed for tails.

The row stride can still vary between calls and can include padding. These
assumptions do not establish whole-tensor contiguity or guarantee a particular
generated instruction; verify generated code when assessing vectorization.

## 2. Keep variable dimensions and strides dynamic

Every dimension or stride that can vary between calls must be represented by
`T.dynamic`, or an expression derived from dynamic values and guaranteed layout
relationships. Do not capture one invocation's shape or strides as constants in
a kernel factory to produce an entirely constant input signature.

For an interface whose shape and row pitch vary, this declaration is incorrect:

```python
# Incorrect: freezes one invocation's dimensions and row pitch.
Input = T.StridedTensor((128, 256), (512, 1), "float16")
```

Use dynamic symbols as in the previous example. If the interface guarantees
unit stride in the last dimension, the following declaration is valid:

```python
m, n = T.dynamic("m, n")
stride_m = T.dynamic("stride_m", "int64")
Input = T.StridedTensor((m, n), (stride_m, 1), "float16")
```

Genuinely fixed interface properties may remain constant, including the length
of a fixed-size scale input and an enforced unit stride. Tile sizes, thread
counts, and other compile-time algorithm parameters may also remain constant.
A value being constant in a benchmark does not establish an interface
guarantee. State and enforce the guarantee for each fixed tensor property.

Choose an integer width that covers the supported address range. In particular,
use wide arithmetic for strides and storage-span calculations that can exceed
32-bit limits; casting an already overflowed result is too late.

## 3. Use the TileLang language surface

Import the language through its standard namespace:

```python
import tilelang.language as T
```

Do not import `tir`, `tirx`, or their members directly, including imports from
`tvm`, `tilelang.tvm`, and their submodules. Aliases and qualified access through
`tvm.tir` or `tvm.tirx` must not be used to bypass this rule.

```python
# Forbidden in MATE TileLang kernel code:
from tvm import tir
from tvm import tirx as tir
from tvm.tirx import call_extern
```

Prefer operations exposed by `T`, such as `T.cast`, `T.ceildiv`, `T.Ramp`, and
`T.Broadcast`. If a required operation is unavailable through the supported
language surface, report the missing API to the TileLang compiler team instead
of coupling the kernel to compiler internals.

## 4. Do not inject handwritten source

Do not use `T.import_source`, an `import_source` argument, or an equivalent
source-injection attribute to insert handwritten device code into a TileLang
kernel. Moving the source string into a helper or another file does not change
this rule.

Express the computation through supported TileLang operations. If the generated
code has a performance problem, summarize it for the TileLang compiler team
with:

- A minimal reproducer and the input shapes, strides, and dtypes.
- The target architecture and MATE, TileLang MUSA, and MCC versions or commits.
- Measured latency or throughput, the comparison baseline, and timing conditions.
- Relevant generated IR, device code, or assembly that illustrates the issue.

The resolution should improve the supported compiler or language path. Source
injection is not an accepted performance workaround in MATE TileLang code.

## 5. Size robust descriptors with the actual storage span

Construct robust descriptor bounds with
[`mate.utils.cosize`](../mate/utils.py), passing both the logical shape and the
actual strides. `cosize` returns a span in **elements**; the size argument to
`T.make_robust_desc` is in **bytes**.

For a nonempty view with nonnegative strides, the span is:

```text
span_elements = 1 + sum((shape[i] - 1) * stride[i])
span_bytes = span_elements * element_size_bytes
```

For example, a float16 view with shape `(2, 3)` and strides `(8, 1)` spans
11 elements, or 22 bytes. Its six logical elements occupy only 12 bytes, which
is not a sufficient descriptor bound.

The following declarations belong in a kernel factory:

```python
from mate.utils import cosize

m, n = T.dynamic("m, n", "int64")
stride_m, stride_n = T.dynamic("stride_m, stride_n", "int64")
shape = (m, n)
strides = (stride_m, stride_n)
Input = T.StridedTensor(shape, strides, "float16")
span_bytes = cosize(shape, strides) * 2  # float16 element size
```

Inside the corresponding `@T.prim_func`, whose argument `A` has type `Input`:

```python
source_desc = T.make_robust_desc(T.address_of(A[0, 0]), span_bytes)
```

The descriptor base and span must describe the same view. Use the address of
that view's first element and its own shape and strides; do not combine a parent
allocation's base with a subview's span or add the view's storage offset twice.
Do not replace `cosize(shape, strides)` with `numel`, a shape product, or
`cosize(shape)` for a view whose dense layout has not been established.

Handle empty inputs in the caller before forming an element address or applying
the nonempty span formula. Reject unsupported layouts, including negative
strides, rather than applying this formula to them.

A robust descriptor bounds an address interval, not the set of logical tensor
elements. Padding and gaps inside that interval can contain arbitrary values,
including NaN or Inf. Preserve logical masks and tail handling; do not assume
that a descriptor makes gaps zero or that multiplying unwanted values by zero
removes them.

## Code review checklist

- [ ] Tensor arguments use `T.StridedTensor`, or have a documented and enforced
  dense-contiguous guarantee for `T.Tensor`; any `T.assume` layout facts are
  guaranteed by the caller.
- [ ] All variable dimensions and strides remain dynamic; fixed tensor
  properties have an explicit interface guarantee.
- [ ] Kernel code uses the `T` language surface without direct `tir`/`tirx`
  imports or equivalent access.
- [ ] No handwritten source is injected; compiler performance issues have a
  reproducible report for the TileLang compiler team.
- [ ] Robust descriptor bounds use `cosize` with actual strides, the correct
  byte size and view base, and separate logical boundary handling.
