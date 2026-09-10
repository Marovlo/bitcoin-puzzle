cmake_minimum_required(VERSION 3.18)
project(puzzle_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform detection
if(APPLE)
    message(STATUS "Platform: macOS (Metal)")
    set(USE_METAL ON)
elseif(WIN32 OR UNIX)
    message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
endif()

# CUDA support (optional)
option(USE_CUDA "Enable NVIDIA CUDA support" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
endif()

# HIP/ROCm support (optional)
option(USE_HIP "Enable AMD ROCm/HIP support" OFF)
if(USE_HIP)
    enable_language(HIP)
    find_package(hip REQUIRED)
    message(STATUS "HIP found")
endif()

# OpenCL support (optional). This is the portable GPU path: it needs no vendor
# SDK because the kernel is compiled at runtime by the installed driver, and on
# Windows/AMD it is the only option for pre-RDNA3 cards (AMD's HIP SDK for
# Windows does not support RDNA2 and older).
option(USE_OPENCL "Enable OpenCL support" OFF)

# Sources
set(WORKER_SOURCES main.cpp)

if(USE_CUDA)
    list(APPEND WORKER_SOURCES
        kernels/cuda/puzzle_kernel.cu
        kernels/cuda/cuda_solver.cu
    )
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "86;89;90")
    endif()
endif()

if(USE_HIP)
    list(APPEND WORKER_SOURCES
        kernels/hip/puzzle_kernel.hip
        kernels/hip/hip_solver.hip
    )
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx1030") # RX 6800 XT default
    endif()
endif()

if(USE_OPENCL)
    list(APPEND WORKER_SOURCES kernels/opencl/opencl_solver.cpp)
    # Embed the kernel source as a raw string literal, mirroring the Metal path.
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/kernels/opencl/puzzle_opencl_source.inc
        COMMAND ${CMAKE_COMMAND}
                -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/kernels/opencl/puzzle_kernel.cl
                -DOUTPUT=${CMAKE_CURRENT_SOURCE_DIR}/kernels/opencl/puzzle_opencl_source.inc
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_opencl_kernel.cmake
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/opencl/puzzle_kernel.cl
              ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_opencl_kernel.cmake
        COMMENT "Embedding OpenCL kernel source"
    )
    list(APPEND WORKER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/opencl/puzzle_opencl_source.inc)
endif()

if(USE_METAL)
    list(APPEND WORKER_SOURCES kernels/metal_solver.mm)
    # Generate Metal shader include
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/kernels/puzzle_metal_source.inc
        COMMAND ${CMAKE_COMMAND} -E echo "R\"METAL(" > ${CMAKE_CURRENT_SOURCE_DIR}/kernels/puzzle_metal_source.inc
        COMMAND ${CMAKE_COMMAND} -E cat ${CMAKE_CURRENT_SOURCE_DIR}/kernels/puzzle.metal >> ${CMAKE_CURRENT_SOURCE_DIR}/kernels/puzzle_metal_source.inc
        COMMAND ${CMAKE_COMMAND} -E echo ")METAL\"" >> ${CMAKE_CURRENT_SOURCE_DIR}/kernels/puzzle_metal_source.inc
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/puzzle.metal
        COMMENT "Generating Metal shader include"
    )
    list(APPEND WORKER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/puzzle_metal_source.inc)
endif()

# Executable
add_executable(puzzle_worker ${WORKER_SOURCES})

target_include_directories(puzzle_worker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

# Platform-specific flags
if(APPLE)
    target_compile_options(puzzle_worker PRIVATE -fobjc-arc)
    target_link_libraries(puzzle_worker
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
    )
endif()

if(WIN32)
    target_link_libraries(puzzle_worker ws2_32) # Winsock for HTTP
endif()

target_link_libraries(puzzle_worker Threads::Threads)
find_package(Threads REQUIRED)

if(USE_CUDA)
    target_compile_definitions(puzzle_worker PRIVATE USE_CUDA)
    target_link_libraries(puzzle_worker CUDA::cudart)
endif()

if(USE_HIP)
    target_compile_definitions(puzzle_worker PRIVATE USE_HIP)
    target_link_libraries(puzzle_worker hip::device)
endif()

if(USE_OPENCL)
    target_compile_definitions(puzzle_worker PRIVATE USE_OPENCL)
    # The vendored Khronos headers keep the build SDK-free; the ICD loader itself
    # is whatever the graphics driver installed.
    target_include_directories(puzzle_worker PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        target_link_libraries(puzzle_worker OpenCL::OpenCL)
    elseif(WIN32)
        # No import library on a stock Windows install: link the ICD loader DLL
        # directly, which ld resolves from its export table.
        target_link_libraries(puzzle_worker "$ENV{SystemRoot}/System32/OpenCL.dll")
    else()
        target_link_libraries(puzzle_worker OpenCL)
    endif()
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    if(MSVC)
        target_compile_options(puzzle_worker PRIVATE /O2 /DNDEBUG)
    else()
        target_compile_options(puzzle_worker PRIVATE -O3 -march=native -DNDEBUG)
    endif()
endif()
