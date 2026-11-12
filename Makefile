# DSA5208 Project 3 — Poisson + CG
#
#   make serial          build/poisson_serial   (CXX, no MPI, no OpenMP)
#   make mpi             build/poisson_mpi      (MPICXX, MPI + OpenMP)
#                        build/pingpong         (network latency / bandwidth, E4)
#   make cuda            build/poisson_mpi_cuda (NVCC for src/gpu.cu, MPICXX for the host code)
#   make clean
#
# Override compilers/flags on the command line, e.g.
#   make serial CXX=g++ OPT="-O3 -march=native"
#   make mpi MPICXX=mpicxx OMPFLAGS=-fopenmp
# On Atlas and Vanda both targets are built with the same GCC (module
# foss/2023b), so serial and parallel timings are comparable.
#
# CUDA on Vanda (module nvhpc-hpcx-cuda12/26.1: HPC-X Open MPI with CUDA
# support, mpicxx = nvc++). The A40 driver supports CUDA <= 12.9, so the
# CUDA 12.9 nvcc of the NVHPC installation is used, with g++ as its host
# compiler (see scripts/pbs/vanda_gpu_*.pbs):
#   make cuda NVCC=$NVHPC_ROOT/cuda/12.9/bin/nvcc

CXX      ?= g++
MPICXX   ?= mpicxx
OPT      ?= -O3 -march=native
OMPFLAGS ?= -fopenmp
CXXFLAGS ?= -std=c++17 $(OPT) -Wall -Wextra -Wno-unknown-pragmas
INC      := -Iinclude

BUILD    := build
CORE_SRC := src/grid.cpp src/kernels.cpp src/options.cpp
MPI_SRC  := $(CORE_SRC) src/decomp.cpp src/halo.cpp
HEADERS  := $(wildcard include/*.hpp)

SERIAL_OBJ := $(patsubst src/%.cpp,$(BUILD)/serial/%.o,$(CORE_SRC))
MPI_OBJ    := $(patsubst src/%.cpp,$(BUILD)/mpi/%.o,$(MPI_SRC))

# ---- CUDA ----
NVCC        ?= nvcc
CUDA_ARCH   ?= sm_86
NVCCFLAGS   ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -ccbin g++ -Xcompiler -Wall
CUDA_HOSTCXXFLAGS ?= -std=c++17 -O3
CUDA_LIBDIR ?= $(abspath $(dir $(shell which $(NVCC) 2>/dev/null))../lib64)
CUDA_SRC    := src/grid.cpp src/kernels.cpp src/options.cpp src/decomp.cpp
CUDA_OBJ    := $(patsubst src/%.cpp,$(BUILD)/cuda/%.o,$(CUDA_SRC)) $(BUILD)/cuda/gpu.o

.PHONY: all serial mpi cuda clean

all: serial mpi

serial: $(BUILD)/poisson_serial
mpi: $(BUILD)/poisson_mpi $(BUILD)/pingpong

$(BUILD)/serial/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(BUILD)/poisson_serial: executables/poisson_serial.cpp $(SERIAL_OBJ) $(HEADERS)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $< $(SERIAL_OBJ) -o $@

$(BUILD)/mpi/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -DUSE_MPI $(INC) -c $< -o $@

$(BUILD)/poisson_mpi: executables/poisson_mpi.cpp $(MPI_OBJ) $(HEADERS)
	@mkdir -p $(BUILD)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -DUSE_MPI $(INC) $< $(MPI_OBJ) -o $@

$(BUILD)/pingpong: executables/pingpong.cpp
	@mkdir -p $(BUILD)
	$(MPICXX) $(CXXFLAGS) $< -o $@

cuda: $(BUILD)/poisson_mpi_cuda

$(BUILD)/cuda/gpu.o: src/gpu.cu include/gpu.hpp include/grid.hpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INC) -c $< -o $@

$(BUILD)/cuda/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(MPICXX) $(CUDA_HOSTCXXFLAGS) -DUSE_MPI $(INC) -c $< -o $@

$(BUILD)/poisson_mpi_cuda: executables/poisson_mpi_cuda.cpp $(CUDA_OBJ) $(HEADERS)
	@mkdir -p $(BUILD)
	$(MPICXX) $(CUDA_HOSTCXXFLAGS) -DUSE_MPI $(INC) $< $(CUDA_OBJ) \
	    -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart -o $@

clean:
	rm -rf $(BUILD)
