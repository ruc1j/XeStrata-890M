# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
# tools/vision/own - Strata's own Qwen vision tower on the CPU (upstream's own-stack phase 3).  No ggml, no llama.cpp: plain C++17.
# Upstream's GPU towers (CUDA / HIP, SYCL, Metal) are not carried: the GPU reader here is mtmd's through Vulkan (tools/vision).
#
# Standalone (quick; the reference and the unit tests):
#   cmake -S tools/vision/own -B build-own -DCMAKE_BUILD_TYPE=Release && cmake --build build-own -j
#   build-own/strata-vision-own --mmproj <mmproj.gguf>      build-own/strata-vision-own-test [mmproj.gguf ...]      build-own/strata-vision-own-cpu-test
# From tools/vision/CMakeLists.txt it is added as a subdirectory (the programs below are then built only when named, in <build>/own), and strata-vision
# links strata_vit_own.
cmake_minimum_required(VERSION 3.21)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(strata_vision_own C CXX)
  set(STRATA_VIT_OWN_STANDALONE ON)
endif()

set(STB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/main/stb" CACHE PATH "stb_image.h (third_party/main/stb)")

add_library(strata_vit_own STATIC gguf_lite.cpp vit_geometry.cpp image_prep.cpp vit_ref.cpp own_reader.cpp
            vit_cpu.cpp cpu_ops.cpp cpu_kernels_scalar.cpp cpu_kernels_sse41.cpp cpu_kernels_avx.cpp cpu_kernels_avx2.cpp cpu_kernels_avx512.cpp cpu_kernels_neon.cpp)
find_package(Threads REQUIRED)
target_link_libraries(strata_vit_own PUBLIC Threads::Threads)
if(NOT MSVC)
  # the reference's loops are written for the compiler's vectoriser: -O3, and this machine's instruction set when asked (the standalone unit-test and
  # reference builds); the helper's portable build keeps the AVX2 baseline its other code has
  option(STRATA_VIT_OWN_NATIVE "build the reference for this CPU (-march=native)" ${STRATA_VIT_OWN_STANDALONE})
  set_source_files_properties(vit_ref.cpp vit_cpu.cpp cpu_kernels_scalar.cpp cpu_kernels_neon.cpp PROPERTIES COMPILE_OPTIONS "-O3")
  # The reference (VitRef, a test oracle) is the only generic file compiled above the baseline, and only where the whole program has that floor anyway: an
  # instruction-set flag on a file that uses the standard library also reaches its inline / template copies (COMDAT), which the linker may keep for the WHOLE
  # program, so a wider flag here would put that code on the path of a CPU without it.
  #   STRATA_VIT_OWN_NATIVE (standalone test builds) or a native helper build (STRATA_PORTABLE=OFF): -march=native, this CPU;
  #   the portable helper (STRATA_PORTABLE=ON, an AVX2 baseline for the whole program, as setup and the packages build it): -mavx2 -mfma.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(STRATA_VIT_OWN_NATIVE OR (DEFINED STRATA_PORTABLE AND NOT STRATA_PORTABLE))
      set_source_files_properties(vit_ref.cpp PROPERTIES COMPILE_OPTIONS "-O3;-march=native")
    elseif(STRATA_PORTABLE)
      set_source_files_properties(vit_ref.cpp PROPERTIES COMPILE_OPTIONS "-O3;-mavx2;-mfma")
    endif()
  elseif(STRATA_VIT_OWN_NATIVE)
    set_source_files_properties(vit_ref.cpp PROPERTIES COMPILE_OPTIONS "-O3;-march=native")
  endif()
  # the CPU tower: the generic part (vit_cpu.cpp) keeps the baseline, each kernel tier is compiled for its own instruction set and only called when the CPU
  # has it (cpu_caps.h); the tiers' files include no standard-library code (cpu_tier.h)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set_source_files_properties(cpu_kernels_sse41.cpp PROPERTIES COMPILE_OPTIONS "-O3;-msse4.1")
    set_source_files_properties(cpu_kernels_avx.cpp PROPERTIES COMPILE_OPTIONS "-O3;-mavx")
    set_source_files_properties(cpu_kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-O3;-mavx2;-mfma;-mf16c")
    set_source_files_properties(cpu_kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "-O3;-mavx2;-mfma;-mf16c;-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl")
  else()
    set_source_files_properties(cpu_kernels_sse41.cpp cpu_kernels_avx.cpp cpu_kernels_avx2.cpp cpu_kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "-O3")
  endif()
else()
  set_source_files_properties(cpu_kernels_avx.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")      # (the SSE4.1 tier needs no flag on x64)
  set_source_files_properties(cpu_kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(cpu_kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
endif()
target_include_directories(strata_vit_own PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(strata_vit_own SYSTEM PRIVATE ${STB_DIR})   # a third-party header: its unused static functions are not ours to warn about
target_compile_features(strata_vit_own PUBLIC cxx_std_17)
if(WIN32)
  # the C runtime's deprecation of getenv and fopen off, as for strata-vision (tools/vision/CMakeLists.txt)
  target_compile_definitions(strata_vit_own PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()

# the own-tower-only reader, the unit tests, and with a picture list the comparison with mtmd's preparation (oracle_prep.cpp is built in
# tools/vision's mtmd build) and the embedding comparison
add_executable(strata-vision-own own_main.cpp)
target_link_libraries(strata-vision-own PRIVATE strata_vit_own)
add_executable(strata-vision-own-test test_gguf_geometry.cpp)
target_link_libraries(strata-vision-own-test PRIVATE strata_vit_own)
add_executable(strata-vision-own-cpu-test test_cpu_kernels.cpp)
target_link_libraries(strata-vision-own-cpu-test PRIVATE strata_vit_own)
add_executable(strata-vision-own-prep-check test_image_prep.cpp)
target_link_libraries(strata-vision-own-prep-check PRIVATE strata_vit_own)
add_executable(strata-vision-own-compare compare_sve.cpp)
target_compile_features(strata-vision-own-compare PRIVATE cxx_std_17)
if(NOT STRATA_VIT_OWN_STANDALONE)
  set_target_properties(strata-vision-own strata-vision-own-test strata-vision-own-cpu-test strata-vision-own-prep-check strata-vision-own-compare PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
