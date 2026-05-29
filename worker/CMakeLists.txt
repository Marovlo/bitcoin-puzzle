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
    add_definitions(-DUSE_CUDA)
endif()

# HIP/ROCm support (optional)
option(USE_HIP "Enable AMD ROCm/HIP support" OFF)
if(USE_HIP)
    enable_language(HIP)
    find_package(hip REQUIRED)
    message(STATUS "HIP found")
    add_definitions(-DUSE_HIP)
endif()

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
    target_link_libraries(puzzle_worker CUDA::cudart)
endif()

if(USE_HIP)
    target_link_libraries(puzzle_worker hip::device)
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    if(MSVC)
        target_compile_options(puzzle_worker PRIVATE /O2 /DNDEBUG)
    else()
        target_compile_options(puzzle_worker PRIVATE -O3 -march=native -DNDEBUG)
    endif()
endif()
