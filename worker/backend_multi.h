#pragma once
#include "worker.h"
#include "backend_cpu.h"
#ifdef __APPLE__
#include "backend_metal.h"
#endif
#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include <chrono>
#include <cstdio>

// Multi-device backend: uses ALL available compute on the machine.
// On macOS: Metal GPU + CPU threads (the GPU doesn't use all CPU, so both run)
// Strategy: split each task proportionally to measured device speeds.
class MultiBackend : public ComputeBackend {
public:
    bool init() override {
        // Try Metal first (fastest on Apple Silicon)
#ifdef __APPLE__
        auto metal = std::make_unique<MetalBackend>();
        if (metal->init()) {
            devices_.push_back({std::move(metal), 0, "metal"});
            printf("  [+] Metal GPU: available\n");
        }
#endif
        // Always add CPU (uses threads not used by GPU)
        // On M-series: GPU uses efficiency cores mostly, so perf cores are free
        int cpu_threads = (int)std::thread::hardware_concurrency();
#ifdef __APPLE__
        // If Metal is available, use fewer CPU threads (leave some headroom)
        if (!devices_.empty()) cpu_threads = std::max(2, cpu_threads / 2);
#endif
        auto cpu = std::make_unique<CPUBackend>(cpu_threads);
        if (cpu->init()) {
            devices_.push_back({std::move(cpu), 0, "cpu"});
            printf("  [+] CPU (%d threads): available\n", cpu_threads);
        }

        if (devices_.empty()) return false;

        // Benchmark each device to determine split ratio
        printf("  [*] Benchmarking devices...\n");
        for (auto& dev : devices_) {
            // Warmup run (Metal needs shader compile + first dispatch)
            dev.backend->benchmark(5000);
            // Real benchmark with larger sample
            dev.rate = dev.backend->benchmark(200000);
            printf("      %s: %llu Keys/s (%.2f MK/s)\n",
                   dev.label.c_str(), (unsigned long long)dev.rate, dev.rate / 1e6);
        }
        return true;
    }

    std::string name() const override {
        std::string n = "multi[";
        for (size_t i = 0; i < devices_.size(); i++) {
            if (i > 0) n += "+";
            n += devices_[i].label;
        }
        n += "]";
        return n;
    }

    bool search(uint64_t start_lo, uint64_t start_hi, uint64_t size,
                const uint8_t target_h160[20],
                uint64_t& found_lo, uint64_t& found_hi) override {
        if (devices_.size() == 1) {
            return devices_[0].backend->search(start_lo, start_hi, size,
                                               target_h160, found_lo, found_hi);
        }

        // Split proportionally to measured rates
        uint64_t total_rate = 0;
        for (auto& d : devices_) total_rate += d.rate;

        std::atomic<bool> found_flag{false};
        std::atomic<uint64_t> res_lo{0}, res_hi{0};

        std::vector<std::thread> threads;
        uint64_t offset = 0;

        for (size_t i = 0; i < devices_.size(); i++) {
            uint64_t dev_size;
            if (i == devices_.size() - 1) {
                dev_size = size - offset; // last device gets remainder
            } else {
                dev_size = (uint64_t)((double)size * devices_[i].rate / total_rate);
                // Align to 1024
                dev_size = (dev_size + 1023) & ~(uint64_t)1023;
            }

            uint64_t dev_start_lo = start_lo + offset;
            uint64_t dev_start_hi = start_hi;
            if (dev_start_lo < start_lo) dev_start_hi++; // carry

            auto* backend = devices_[i].backend.get();
            uint8_t target_copy[20];
            memcpy(target_copy, target_h160, 20);

            threads.emplace_back([backend, dev_start_lo, dev_start_hi, dev_size,
                                  target_copy, &found_flag, &res_lo, &res_hi]() {
                uint64_t fl = 0, fh = 0;
                if (backend->search(dev_start_lo, dev_start_hi, dev_size,
                                    target_copy, fl, fh)) {
                    found_flag.store(true);
                    res_lo.store(fl);
                    res_hi.store(fh);
                }
            });
            offset += dev_size;
        }

        for (auto& t : threads) t.join();

        if (found_flag.load()) {
            found_lo = res_lo.load();
            found_hi = res_hi.load();
            return true;
        }
        return false;
    }

    uint64_t benchmark(uint64_t sample_size) override {
        // Combined rate
        uint64_t total = 0;
        for (auto& d : devices_) total += d.rate;
        return total;
    }

private:
    struct DeviceSlot {
        std::unique_ptr<ComputeBackend> backend;
        uint64_t rate;
        std::string label;
    };
    std::vector<DeviceSlot> devices_;
};
