cmake_minimum_required(VERSION 3.26)

# When building directly using CMake, make sure you run the install step
# (it places the .so files in the correct location).
#
# Example:
# mkdir build && cd build
# cmake -G Ninja -DVLLM_PYTHON_EXECUTABLE=`which python3` -DCMAKE_INSTALL_PREFIX=.. ..
# cmake --build . --target install
#
# If you want to only build one target, make sure to install it manually:
# cmake --build . --target _C
# cmake --install . --component _C
project(moe_kernels LANGUAGES CXX)

# CUDA by default, can be overridden by using -DVLLM_TARGET_DEVICE=... (used by setup.py)
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# Prevent installation of dependencies (cutlass) by default.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11" "3.12")

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.9;9.0")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from Dockerfile.rocm
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.4.0")
set(TORCH_SUPPORTED_VERSION_ROCM "2.5.0")

#
# Try to find python package with an executable that exactly matches
# `VLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# Ensure the 'nvcc' command is in the PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

#
message(STATUS "Enabling core extension.")

#
# Forward the non-CUDA device extensions to external CMake scripts.
#
if (NOT VLLM_TARGET_DEVICE STREQUAL "cuda" AND
    NOT VLLM_TARGET_DEVICE STREQUAL "rocm")
    if (VLLM_TARGET_DEVICE STREQUAL "cpu")
        include(${CMAKE_CURRENT_LIST_DIR}/cmake/cpu_extension.cmake)
    else()
        return()
    endif()
    return()
endif()

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (NOT HIP_FOUND AND CUDA_FOUND)
  set(VLLM_GPU_LANG "CUDA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
elseif(HIP_FOUND)
  set(VLLM_GPU_LANG "HIP")

  # Importing torch recognizes and sets up some HIP/ROCm configuration but does
  # not let cmake recognize .hip files. In order to get cmake to understand the
  # .hip extension automatically, HIP must be enabled explicitly.
  enable_language(HIP)

  # ROCm 5.X and 6.X
  if (ROCM_VERSION_DEV_MAJOR GREATER_EQUAL 5 AND
      NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM})
    message(WARNING "Pytorch version >= ${TORCH_SUPPORTED_VERSION_ROCM} "
      "expected for ROCm build, saw ${Torch_VERSION} instead.")
  endif()
else()
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()


#
# For cuda we want to be able to control which architectures we compile for on 
# a per-file basis in order to cut down on compile time. So here we extract
# the set of architectures we want to compile for and remove the from the 
# CMAKE_CUDA_FLAGS so that they are not applied globally.
#
if(VLLM_GPU_LANG STREQUAL "CUDA")
  clear_cuda_arches(CUDA_ARCH_FLAGS)
  extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
  message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")
endif()

#
# Override the GPU architectures detected by cmake/torch and filter them by
# the supported versions for the current language.
# The final set of arches is stored in `VLLM_GPU_ARCHES`.
#
override_gpu_arches(VLLM_GPU_ARCHES
  ${VLLM_GPU_LANG}
  "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()


#
# Use FetchContent for C++ dependencies that are compiled as part of vLLM's build process.
# Configure it to place files in vllm/.deps, in order to play nicely with sccache.
#
include(FetchContent)
get_filename_component(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
file(MAKE_DIRECTORY "${FETCHCONTENT_BASE_DIR}")
set(FETCHCONTENT_BASE_DIR "${PROJECT_ROOT_DIR}/.deps")
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

#
# Define other extension targets
#

#
# _C extension

set(VLLM_EXT_SRC
  "csrc/activation_kernels.cu"
  "csrc/quantization/fp8/common.cu"
  "csrc/moe/torch_bindings.cpp"
  "csrc/moe/moe_align_sum_kernels.cu"
  "csrc/moe/topk_softmax_kernels.cu")

set_gencode_flags_for_srcs(
  SRCS "${VLLM_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0;8.6;8.9;9.0" "${CUDA_ARCHS}")
  if (MARLIN_MOE_ARCHS)
    set(MARLIN_MOE_SRC
        "csrc/moe/marlin_kernels/marlin_moe_kernel.h"
        "csrc/moe/marlin_kernels/marlin_moe_kernel_ku4b8.h"
        "csrc/moe/marlin_kernels/marlin_moe_kernel_ku4b8.cu"
        "csrc/moe/marlin_kernels/marlin_moe_kernel_ku8b128.h"
        "csrc/moe/marlin_kernels/marlin_moe_kernel_ku8b128.cu"
        "csrc/moe/marlin_kernels/marlin_moe_kernel_ku4.h"
        "csrc/moe/marlin_kernels/marlin_moe_kernel_ku4.cu"
        "csrc/moe/marlin_moe_ops.cu")

    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_MOE_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_ARCHS}")

    list(APPEND VLLM_EXT_SRC "${MARLIN_MOE_SRC}")
    message(STATUS "Building Marlin MOE kernels for archs: ${MARLIN_MOE_ARCHS}")
  else()
    message(STATUS "Not building Marlin MOE kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()
endif()

message(STATUS "Enabling C extension.")
define_gpu_extension_target(
  _moe_kernels
  DESTINATION moe_kernels
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

# If CUTLASS is compiled on NVCC >= 12.5, it by default uses 
# cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the 
# driver API. This causes problems when linking with earlier versions of CUDA.
# Setting this variable sidesteps the issue by calling the driver directly.
target_compile_definitions(_moe_kernels PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
