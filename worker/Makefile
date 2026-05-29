# Bitcoin Puzzle Pool Worker
CXX      := clang++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -DNDEBUG -march=native -I. -Ikernels

UNAME := $(shell uname -s)
TARGET := puzzle_worker

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

clean:
	rm -f $(OBJS) $(TARGET) kernels/puzzle_metal_source.inc

.PHONY: all clean
