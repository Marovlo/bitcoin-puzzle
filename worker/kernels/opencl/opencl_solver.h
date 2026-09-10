#pragma once
#ifdef USE_OPENCL

#include <cstdint>
#include <array>
#include <string>

// OpenCL driver wrapper for the puzzle_search kernel.
//
// Deliberately SDK-free: it links against the OpenCL ICD loader that ships with
// Windows (System32/OpenCL.dll) and compiles the kernel at runtime with whatever
// driver is installed. On AMD that means the normal Adrenalin package, no ROCm.
class OpenCLSolver {
public:
    OpenCLSolver();
    ~OpenCLSolver();

    bool init(int device_id = 0);
    std::string device_name() const;
    bool set_target(const std::array<uint8_t, 20>& hash160);
    void set_batch_size(uint64_t bs);
    uint64_t batch_size() const;
    bool search_batch(uint64_t start_lo, uint64_t start_hi, uint64_t batch_size,
                      uint64_t& found_lo, uint64_t& found_hi);
    double benchmark(uint64_t num_keys);

    // Human-readable listing of every OpenCL GPU found, for diagnostics.
    static std::string list_devices();

private:
    struct Impl;
    Impl* impl_;
};

#endif // USE_OPENCL
