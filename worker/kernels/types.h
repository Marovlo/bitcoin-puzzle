#pragma once
#include <cstdint>
#include <array>
#include <string>

// 256-bit unsigned integer (4 x 64-bit limbs, little-endian by limb)
struct uint256_t {
    uint64_t d[4] = {0, 0, 0, 0};
};

// Search result
struct SearchResult {
    bool found = false;
    uint256_t private_key;
};

// Puzzle target definition
struct PuzzleTarget {
    int bit_range;                      // e.g., 71 means key in [2^70, 2^71)
    std::array<uint8_t, 20> hash160;    // target RIPEMD160(SHA256(compressed_pubkey))
    std::string address;                // human-readable Bitcoin address
};

// Benchmark result
struct BenchmarkResult {
    double keys_per_second;
    double elapsed_seconds;
    uint64_t keys_checked;
    std::string device_name;
};
