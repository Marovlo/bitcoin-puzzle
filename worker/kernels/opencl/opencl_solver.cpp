#ifdef USE_OPENCL
#include "opencl_solver.h"
#include "../secp256k1.h"

// Pin the headers to the OpenCL 1.2 core profile: every function used here is
// 1.2, so the loader never needs a newer symbol and the build works against any
// vendor ICD.
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Embedded kernel source, expanded to a raw string literal by the build system
// (mirrors how the Metal backend embeds puzzle.metal).
static const char* kOpenCLKernelSource =
#include "puzzle_opencl_source.inc"
;

namespace {

constexpr uint64_t kDefaultBatchSize = 4'000'000ull;
constexpr size_t kMaxPlatforms = 8;
constexpr size_t kMaxDevices = 16;

void print_build_log(cl_program program, cl_device_id device, const char* stage) {
    size_t log_size = 0;
    if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size) != CL_SUCCESS)
        return;
    if (log_size == 0) return;
    std::vector<char> log(log_size + 1, '\0');
    if (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr) != CL_SUCCESS)
        return;
    fprintf(stderr, "[OpenCL] %s log:\n%s\n", stage, log.data());
}

// Pick the platform that actually exposes a GPU: prefer AMD, fall back to the
// first platform with a usable device so the backend also works on NVIDIA/Intel.
bool select_platform(cl_platform_id* out_platform) {
    cl_platform_id platforms[kMaxPlatforms];
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(kMaxPlatforms, platforms, &num_platforms) != CL_SUCCESS || num_platforms == 0)
        return false;

    cl_platform_id first_with_gpu = nullptr;
    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != CL_SUCCESS ||
            num_devices == 0)
            continue;

        if (first_with_gpu == nullptr) first_with_gpu = platforms[i];

        char vendor[256] = {0};
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor) - 1, vendor, nullptr);
        if (strstr(vendor, "AMD") || strstr(vendor, "Advanced Micro Devices")) {
            *out_platform = platforms[i];
            return true;
        }
    }
    if (first_with_gpu) { *out_platform = first_with_gpu; return true; }
    return false;
}

std::string device_string(cl_device_id device, cl_device_info param) {
    size_t size = 0;
    if (clGetDeviceInfo(device, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) return {};
    std::string out(size, '\0');
    if (clGetDeviceInfo(device, param, size, &out[0], nullptr) != CL_SUCCESS) return {};
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

} // namespace

struct OpenCLSolver::Impl {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel_group = nullptr;   // group search (preferred)
    cl_kernel kernel_naive = nullptr;   // one scalar mul per key

    cl_mem d_gtable = nullptr;
    cl_mem d_grp_table = nullptr;
    cl_mem d_target = nullptr;
    cl_mem d_match_lo = nullptr;
    cl_mem d_match_hi = nullptr;
    cl_mem d_match_found = nullptr;

    std::string device_name_str;
    uint64_t batch = kDefaultBatchSize;
    size_t local_size = 256;

    // Persistent source for the non-blocking match-slot resets. A non-blocking
    // enqueue may copy the source later, so it must not point at a stack local
    // that dies when search_batch() returns. These are always zero, so whenever
    // the driver reads them the value is correct.
    uint64_t reset64 = 0;
    cl_uint reset32 = 0;

    // Group geometry, read back from the kernel itself (kernel_params) so the
    // host launch shape can never drift from the compiled constants.
    uint32_t group_h = 0;
    uint32_t group_size = 0;
    uint32_t groups_per_item = 1;

    bool use_group = true;

    cl_kernel active_kernel() const {
        return (use_group && kernel_group) ? kernel_group : kernel_naive;
    }
};

OpenCLSolver::OpenCLSolver() : impl_(new Impl()) {}

OpenCLSolver::~OpenCLSolver() {
    if (!impl_) return;
    if (impl_->kernel_group) clReleaseKernel(impl_->kernel_group);
    if (impl_->kernel_naive) clReleaseKernel(impl_->kernel_naive);
    if (impl_->program) clReleaseProgram(impl_->program);
    if (impl_->d_gtable) clReleaseMemObject(impl_->d_gtable);
    if (impl_->d_grp_table) clReleaseMemObject(impl_->d_grp_table);
    if (impl_->d_target) clReleaseMemObject(impl_->d_target);
    if (impl_->d_match_lo) clReleaseMemObject(impl_->d_match_lo);
    if (impl_->d_match_hi) clReleaseMemObject(impl_->d_match_hi);
    if (impl_->d_match_found) clReleaseMemObject(impl_->d_match_found);
    if (impl_->queue) clReleaseCommandQueue(impl_->queue);
    if (impl_->context) clReleaseContext(impl_->context);
    delete impl_;
}

std::string OpenCLSolver::list_devices() {
    std::string out;
    cl_platform_id platform = nullptr;
    if (!select_platform(&platform)) return "  (no OpenCL GPU platforms found)\n";

    cl_device_id devices[kMaxDevices];
    cl_uint count = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, kMaxDevices, devices, &count) != CL_SUCCESS)
        return "  (no OpenCL GPU devices found)\n";

    char pname[256] = {0};
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pname) - 1, pname, nullptr);
    out += std::string("  Platform: ") + pname + "\n";
    for (cl_uint i = 0; i < count; i++) {
        cl_uint cus = 0;
        cl_ulong mem = 0;
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
        char line[512];
        snprintf(line, sizeof(line), "  [%u] %s (%u CUs, %d MB) %s\n", i,
                 device_string(devices[i], CL_DEVICE_NAME).c_str(), cus,
                 (int)(mem / 1048576),
                 device_string(devices[i], CL_DEVICE_VERSION).c_str());
        out += line;
    }
    return out;
}

bool OpenCLSolver::init(int device_id) {
    if (!select_platform(&impl_->platform)) {
        fprintf(stderr, "[OpenCL] No GPU OpenCL platform found\n");
        return false;
    }

    cl_device_id devices[kMaxDevices];
    cl_uint count = 0;
    if (clGetDeviceIDs(impl_->platform, CL_DEVICE_TYPE_GPU, kMaxDevices, devices, &count) != CL_SUCCESS ||
        count == 0 || device_id < 0 || (cl_uint)device_id >= count) {
        fprintf(stderr, "[OpenCL] Device %d not available\n", device_id);
        return false;
    }
    impl_->device = devices[device_id];
    impl_->device_name_str = device_string(impl_->device, CL_DEVICE_NAME);

    cl_uint cus = 0;
    cl_ulong global_mem = 0;
    size_t max_wg = 0;
    clGetDeviceInfo(impl_->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
    clGetDeviceInfo(impl_->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, nullptr);
    clGetDeviceInfo(impl_->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);

    printf("[OpenCL] Device: %s (%u CUs, %d MB, max WG %d) %s\n",
           impl_->device_name_str.c_str(), cus, (int)(global_mem / 1048576), (int)max_wg,
           device_string(impl_->device, CL_DEVICE_VERSION).c_str());

    impl_->context = clCreateContext(nullptr, 1, &impl_->device, nullptr, nullptr, nullptr);
    if (!impl_->context) { fprintf(stderr, "[OpenCL] clCreateContext failed\n"); return false; }

    impl_->queue = clCreateCommandQueue(impl_->context, impl_->device, 0, nullptr);
    if (!impl_->queue) { fprintf(stderr, "[OpenCL] clCreateCommandQueue failed\n"); return false; }

    impl_->program = clCreateProgramWithSource(impl_->context, 1, &kOpenCLKernelSource, nullptr, nullptr);
    if (!impl_->program) { fprintf(stderr, "[OpenCL] clCreateProgramWithSource failed\n"); return false; }

    const cl_int build_status = clBuildProgram(impl_->program, 1, &impl_->device, nullptr, nullptr, nullptr);
    if (build_status != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clBuildProgram failed (err %d)\n", build_status);
        print_build_log(impl_->program, impl_->device, "build");
        return false;
    }

    impl_->kernel_group = clCreateKernel(impl_->program, "puzzle_search_group", nullptr);
    impl_->kernel_naive = clCreateKernel(impl_->program, "puzzle_search_naive", nullptr);
    if (!impl_->kernel_group && !impl_->kernel_naive) {
        fprintf(stderr, "[OpenCL] neither puzzle_search_group nor puzzle_search_naive is present\n");
        return false;
    }

    // Read back the group geometry the kernel was compiled with, so the launch
    // shape below is derived from the kernel rather than duplicated here.
    if (impl_->kernel_group) {
        cl_kernel kp = clCreateKernel(impl_->program, "kernel_params", nullptr);
        cl_mem d_params = kp ? clCreateBuffer(impl_->context, CL_MEM_WRITE_ONLY, 3 * sizeof(cl_uint),
                                              nullptr, nullptr) : nullptr;
        if (kp && d_params) {
            cl_uint params[3] = {0, 0, 0};
            clSetKernelArg(kp, 0, sizeof(cl_mem), &d_params);
            const size_t one = 1;
            if (clEnqueueNDRangeKernel(impl_->queue, kp, 1, nullptr, &one, nullptr, 0, nullptr, nullptr) == CL_SUCCESS &&
                clEnqueueReadBuffer(impl_->queue, d_params, CL_TRUE, 0, sizeof(params), params,
                                    0, nullptr, nullptr) == CL_SUCCESS) {
                impl_->group_h = params[0];
                impl_->group_size = params[1];
                impl_->groups_per_item = params[2] > 0 ? params[2] : 1;
            }
        }
        if (d_params) clReleaseMemObject(d_params);
        if (kp) clReleaseKernel(kp);

        if (impl_->group_h == 0 || impl_->group_size == 0) {
            fprintf(stderr, "[OpenCL] kernel_params unavailable, falling back to the naive kernel\n");
            clReleaseKernel(impl_->kernel_group);
            impl_->kernel_group = nullptr;
        }
    }

    // Escape hatch for A/B measurements: force the one-scalar-mul-per-key kernel.
    if (std::getenv("PUZZLE_OCL_NAIVE")) impl_->use_group = false;

    // Work-group size: the device maximum is authoritative, clamped so a large
    // dispatch still splits into plenty of groups.
    size_t preferred_multiple = 0;
    cl_kernel active = impl_->active_kernel();
    if (active &&
        clGetKernelWorkGroupInfo(active, impl_->device,
                                 CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                 sizeof(preferred_multiple), &preferred_multiple, nullptr) == CL_SUCCESS &&
        preferred_multiple > 0) {
        impl_->local_size = preferred_multiple * (256 / preferred_multiple);
    }
    if (impl_->local_size == 0) impl_->local_size = 64;
    if (impl_->local_size > max_wg && max_wg > 0) impl_->local_size = max_wg;

    // Batch sizing. Dispatch cost is per-dispatch, not per-key, so the batch is
    // sized for tens of milliseconds of GPU work rather than a few: the group
    // kernel retires hundreds of millions of keys per second, and a 1M-key batch
    // would be dominated by the surrounding driver calls.
    impl_->batch = 16'000'000ull;
    if (const char* env = std::getenv("PUZZLE_OCL_BATCH")) {
        const long long v = atoll(env);
        if (v > 0) impl_->batch = (uint64_t)v;
    }
    // Round up to a whole number of groups: a partial group would only waste
    // hashes on keys past the end of the batch.
    if (impl_->group_size > 0) {
        const uint64_t g = impl_->group_size;
        impl_->batch = ((impl_->batch + g - 1) / g) * g;
    }

    // Device buffers
    const size_t gtable_bytes = secp256k1::G_TABLE_ULONGS * sizeof(uint64_t);
    const size_t grp_bytes = impl_->kernel_group ? (size_t)(impl_->group_h + 1) * 8 * sizeof(uint64_t) : 0;
    impl_->d_gtable = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY, gtable_bytes, nullptr, nullptr);
    impl_->d_target = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY, 32, nullptr, nullptr);
    impl_->d_match_lo = clCreateBuffer(impl_->context, CL_MEM_READ_WRITE, sizeof(uint64_t), nullptr, nullptr);
    impl_->d_match_hi = clCreateBuffer(impl_->context, CL_MEM_READ_WRITE, sizeof(uint64_t), nullptr, nullptr);
    impl_->d_match_found = clCreateBuffer(impl_->context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, nullptr);
    if (grp_bytes) impl_->d_grp_table = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY, grp_bytes, nullptr, nullptr);
    if (!impl_->d_gtable || !impl_->d_target || !impl_->d_match_lo ||
        !impl_->d_match_hi || !impl_->d_match_found || (grp_bytes && !impl_->d_grp_table)) {
        fprintf(stderr, "[OpenCL] buffer allocation failed\n");
        return false;
    }

    std::vector<uint64_t> gtable_host(secp256k1::G_TABLE_ULONGS);
    secp256k1::build_g_table(gtable_host.data());
    if (clEnqueueWriteBuffer(impl_->queue, impl_->d_gtable, CL_TRUE, 0, gtable_bytes,
                             gtable_host.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] G table upload failed\n");
        return false;
    }

    if (grp_bytes) {
        std::vector<uint64_t> grp_host((size_t)(impl_->group_h + 1) * 8);
        secp256k1::build_group_table(grp_host.data(), (int)impl_->group_h, gtable_host.data());
        if (clEnqueueWriteBuffer(impl_->queue, impl_->d_grp_table, CL_TRUE, 0, grp_bytes,
                                 grp_host.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
            fprintf(stderr, "[OpenCL] group table upload failed\n");
            return false;
        }
    }

    printf("[OpenCL] G table %zu KB, local size %d, batch %llu, kernel %s (group %u+1 keys, %u groups/item)\n",
           gtable_bytes / 1024, (int)impl_->local_size, (unsigned long long)impl_->batch,
           impl_->active_kernel() == impl_->kernel_group ? "group" : "naive",
           impl_->group_h, impl_->groups_per_item);
    return true;
}

std::string OpenCLSolver::device_name() const { return impl_->device_name_str; }
uint64_t OpenCLSolver::batch_size() const { return impl_->batch; }
void OpenCLSolver::set_batch_size(uint64_t bs) { if (bs > 0) impl_->batch = bs; }

bool OpenCLSolver::set_target(const std::array<uint8_t, 20>& hash160) {
    uint8_t padded[32] = {0};
    memcpy(padded, hash160.data(), 20);
    // Blocking on purpose: padded[] is a stack local, so a deferred copy could
    // read freed memory. This runs once per task, not once per dispatch, so the
    // synchronisation is free in practice.
    return clEnqueueWriteBuffer(impl_->queue, impl_->d_target, CL_TRUE, 0, sizeof(padded),
                                padded, 0, nullptr, nullptr) == CL_SUCCESS;
}

bool OpenCLSolver::search_batch(uint64_t start_lo, uint64_t start_hi, uint64_t batch_size,
                                uint64_t& found_lo, uint64_t& found_hi) {
    if (batch_size == 0) batch_size = impl_->batch;

    // Reset the match slots. These are non-blocking on purpose: the queue is
    // in-order, so they are guaranteed to complete before the kernel starts.
    // Synchronising here instead would add a host round-trip to every dispatch,
    // which is a measurable share of the cost for a kernel that only runs for a
    // few milliseconds.
    clEnqueueWriteBuffer(impl_->queue, impl_->d_match_lo, CL_FALSE, 0, sizeof(impl_->reset64),
                         &impl_->reset64, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(impl_->queue, impl_->d_match_hi, CL_FALSE, 0, sizeof(impl_->reset64),
                         &impl_->reset64, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(impl_->queue, impl_->d_match_found, CL_FALSE, 0, sizeof(impl_->reset32),
                         &impl_->reset32, 0, nullptr, nullptr);

    const cl_ulong arg_start_lo = (cl_ulong)start_lo;
    const cl_ulong arg_start_hi = (cl_ulong)start_hi;
    const cl_ulong arg_total = (cl_ulong)batch_size;

    cl_kernel kernel = impl_->active_kernel();
    if (!kernel) return false;

    const size_t local = impl_->local_size;
    cl_int err = CL_SUCCESS;
    size_t global = 0;

    if (kernel == impl_->kernel_group) {
        // One work-item covers GROUPS_PER_ITEM consecutive groups of GROUP_SIZE
        // keys. Both factors come from the kernel itself (kernel_params), so the
        // grid is derived from the compiled geometry rather than duplicated.
        uint64_t items = (batch_size + impl_->group_size - 1) / impl_->group_size;
        items = (items + impl_->groups_per_item - 1) / impl_->groups_per_item;
        if (items == 0) return false;
        const uint64_t kMaxItems = 1ull << 31;   // sanity cap for absurd batches
        if (items > kMaxItems) items = kMaxItems;
        global = (size_t)((items + local - 1) / local) * local;

        err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &impl_->d_gtable);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &impl_->d_grp_table);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &impl_->d_target);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_ulong), &arg_start_lo);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_ulong), &arg_start_hi);
        err |= clSetKernelArg(kernel, 5, sizeof(cl_ulong), &arg_total);
        err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &impl_->d_match_lo);
        err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &impl_->d_match_hi);
        err |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &impl_->d_match_found);
    } else {
        // One work-item per key; the kernel guards the tail with `gid >= total_keys`.
        global = (size_t)((batch_size + local - 1) / local) * local;

        err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &impl_->d_gtable);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &impl_->d_target);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_ulong), &arg_start_lo);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_ulong), &arg_start_hi);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_ulong), &arg_total);
        err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &impl_->d_match_lo);
        err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &impl_->d_match_hi);
        err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &impl_->d_match_found);
    }

    if (err != CL_SUCCESS || global == 0) {
        fprintf(stderr, "[OpenCL] kernel argument setup failed (err %d)\n", err);
        return false;
    }

    err = clEnqueueNDRangeKernel(impl_->queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed (err %d)\n", err);
        return false;
    }

    // The only synchronisation point. A blocking read waits for everything
    // queued before it in this in-order queue, so no separate clFinish is needed.
    cl_uint found = 0;
    if (clEnqueueReadBuffer(impl_->queue, impl_->d_match_found, CL_TRUE, 0, sizeof(found),
                            &found, 0, nullptr, nullptr) != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] reading the match flag failed\n");
        return false;
    }
    if (found) {
        clEnqueueReadBuffer(impl_->queue, impl_->d_match_lo, CL_TRUE, 0, sizeof(found_lo), &found_lo, 0, nullptr, nullptr);
        clEnqueueReadBuffer(impl_->queue, impl_->d_match_hi, CL_TRUE, 0, sizeof(found_hi), &found_hi, 0, nullptr, nullptr);
        return true;
    }
    return false;
}

double OpenCLSolver::benchmark(uint64_t num_keys) {
    std::array<uint8_t, 20> dummy;
    memset(dummy.data(), 0xFF, 20);
    set_target(dummy);

    const auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t remaining = num_keys;
    uint64_t offset = 1;
    while (remaining > 0) {
        const uint64_t bs = (remaining < impl_->batch) ? remaining : impl_->batch;
        uint64_t fl = 0, fh = 0;
        search_batch(offset, 0, bs, fl, fh);
        offset += bs;
        remaining -= bs;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    return elapsed > 0 ? (double)num_keys / elapsed : 0.0;
}

#endif // USE_OPENCL
