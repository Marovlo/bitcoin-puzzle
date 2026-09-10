# Wraps an OpenCL kernel source file in a C++ raw string literal so it can be
# embedded in the binary. Same idea as the Metal backend's generated .inc.
#
# Invoked at build time via:
#   cmake -DINPUT=<kernel.cl> -DOUTPUT=<kernel.inc> -P embed_opencl_kernel.cmake
#
# The delimiter OPENCL is safe because ")OPENCL\"" cannot occur in kernel source.
# The file is written as raw bytes (no BOM), which matters: a leading BOM would
# end up inside the literal and break compilation.

if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
    message(FATAL_ERROR "embed_opencl_kernel.cmake: INPUT and OUTPUT are required")
endif()

file(READ "${INPUT}" contents)
file(WRITE "${OUTPUT}" "R\"OPENCL(${contents})OPENCL\"")
