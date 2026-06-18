# Bitcoin Puzzle Pool Worker
CXX      := clang++

UNAME := $(shell uname -s)
ARCH  := $(shell uname -m)
TARGET := puzzle_worker

# On ARM64 (Apple Silicon `arm64`, Linux/Android `aarch64`), -march=native does
# NOT enable the ARMv8 crypto (SHA-256) feature macros that hash.h needs;
# -mcpu=native does, and keeps all the same tuning. Use it on either arm64 name,
# plain -march=native elsewhere.
ifeq ($(ARCH), arm64)
    ARCHFLAG := -mcpu=native
else ifeq ($(ARCH), aarch64)
    ARCHFLAG := -mcpu=native
else
    ARCHFLAG := -march=native
endif

CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -DNDEBUG $(ARCHFLAG) -I. -Ikernels

ifeq ($(UNAME), Darwin)
    FRAMEWORKS := -framework Metal -framework Foundation -framework CoreGraphics
    LDFLAGS := $(FRAMEWORKS) -lpthread
    METAL_OBJ := metal_solver.o
    OBJS := main.o $(METAL_OBJ)
else
    LDFLAGS := -lpthread
    OBJS := main.o
endif

all: $(TARGET)

kernels/puzzle_metal_source.inc: kernels/puzzle.metal
	@echo 'R"METAL(' > $@
	@cat $< >> $@
	@echo ')METAL"' >> $@

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.cpp worker.h http_client.h json_helpers.h backend_cpu.h backend_multi.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

ifeq ($(UNAME), Darwin)
metal_solver.o: kernels/metal_solver.mm kernels/metal_solver.h kernels/puzzle_metal_source.inc
	$(CXX) $(CXXFLAGS) -fobjc-arc -c -o $@ $<
endif

# ===== Tests =====
# CPU-only harnesses (run anywhere):
#   test-correct   hash160 of known puzzle keys vs ground truth (test_vectors.txt)
#   test-coverage  exhaustive per-offset coverage of the CPU group search
# Metal harness (Darwin only):
#   test-metal     range-search: known keys at non-zero offsets must be found
test-correct: test_correctness
	./test_correctness
test_correctness: test_correctness.cpp kernels/secp256k1.h kernels/hash.h
	$(CXX) $(CXXFLAGS) -o $@ $<

test-coverage: test_coverage
	./test_coverage
test_coverage: test_coverage.cpp backend_cpu.h kernels/secp256k1.h kernels/hash.h
	$(CXX) $(CXXFLAGS) -o $@ $<

ifeq ($(UNAME), Darwin)
test-metal: test_metal_correctness
	./test_metal_correctness
test_metal_correctness: test_metal_correctness.cpp metal_solver.o kernels/metal_solver.h
	$(CXX) $(CXXFLAGS) -fobjc-arc -o $@ $< metal_solver.o $(LDFLAGS)

# Run every available test (CPU + Metal).
test: test-correct test-coverage test-metal
else
test: test-correct test-coverage
endif

clean:
	rm -f $(OBJS) $(TARGET) kernels/puzzle_metal_source.inc \
	      test_correctness test_coverage test_metal_correctness

.PHONY: all clean test test-correct test-coverage test-metal
