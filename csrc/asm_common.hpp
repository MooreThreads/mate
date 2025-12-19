
#pragma once

#include <mute/arch/copy_mp31_desc.hpp>

#include "mate_utils.muh"
#include "musa.h"

struct MateAsmKernel {
 public:
  MUmodule   asm_module;
  MUfunction asm_func;

 public:
  MateAsmKernel() = default;
  // TODO: optimize this to load
  MateAsmKernel(const unsigned char* path, const std::string& func_name) {
    MATE_MUSA_DRIVER_CHECK(muModuleLoadData(&asm_module, path));
    MATE_MUSA_DRIVER_CHECK(muModuleGetFunction(&asm_func, asm_module, func_name.c_str()));
  }

  void load(const std::string& path, const std::string& func_name) {
    MATE_MUSA_DRIVER_CHECK(muModuleLoad(&asm_module, path.c_str()));
    MATE_MUSA_DRIVER_CHECK(muModuleGetFunction(&asm_func, asm_module, func_name.c_str()));
  }

  void loadData(const void* path, const std::string& func_name) {
    MATE_MUSA_DRIVER_CHECK(muModuleLoadData(&asm_module, path));
    MATE_MUSA_DRIVER_CHECK(muModuleGetFunction(&asm_func, asm_module, func_name.c_str()));
  }

  ~MateAsmKernel() {
    MATE_MUSA_DRIVER_CHECK(muModuleUnload(asm_module));
  }

};  // struct MateAsmKernel

template <class Args>
static void launch_asm(const MUfunction& kernel, const MUlaunchConfig& config, Args& args) {
  size_t kernel_param_size = sizeof(Args);
  void*  kernel_args[]     = {
      MU_LAUNCH_PARAM_BUFFER_POINTER, &args, MU_LAUNCH_PARAM_BUFFER_SIZE, &kernel_param_size, MU_LAUNCH_PARAM_END};

  MATE_MUSA_DRIVER_CHECK(muLaunchKernelEx(&config, kernel, nullptr, kernel_args));
}
