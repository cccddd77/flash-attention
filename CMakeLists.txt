cmake_minimum_required(VERSION 3.18)

project(flash_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(CUDA REQUIRED)

set(CUTLASS_3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass)
set(FLASH_ATTN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn)

file(GLOB FLASH_ATTN_SOURCE_FILES ${FLASH_ATTN_DIR}/src/*.cu)

add_library(flash_attention SHARED ${FLASH_ATTN_SOURCE_FILES})

set_property(TARGET flash_attention PROPERTY CUDA_ARCHITECTURES OFF)

target_include_directories(flash_attention PRIVATE 
  ${CUTLASS_3_DIR}/include
  ${FLASH_ATTN_DIR}/include)

target_compile_options(flash_attention PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
-w
-Xcompiler="-fPIC"
-Xcompiler="-O3"
-std=c++17
-U__CUDA_NO_HALF_OPERATORS__
-U__CUDA_NO_HALF_CONVERSIONS__
-U__CUDA_NO_HALF2_OPERATORS__
-U__CUDA_NO_BFLOAT16_CONVERSIONS__
--expt-relaxed-constexpr
--expt-extended-lambda
--use_fast_math
-arch=native
# "${FA_GENCODE_OPTION}"
>)

include(GNUInstallDirs)

install(DIRECTORY ${FLASH_ATTN_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message(STATUS ${CMAKE_INSTALL_INCLUDEDIR})

install(
  TARGETS flash_attention
  EXPORT flash_attention
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})