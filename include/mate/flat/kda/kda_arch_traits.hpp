#pragma once

namespace mate::flat::kda {

// Keep architecture selection in one place.  The ArchTag is the architecture
// compiled into the kernel; it must not be inferred from the device at a call
// site because KDA supports cross-building for the other MUSA target.
template <class ArchTag>
struct arch_traits {
  static constexpr bool is_mp32 = ArchTag::kMinComputeCapability == 32;
};

}  // namespace mate::flat::kda
