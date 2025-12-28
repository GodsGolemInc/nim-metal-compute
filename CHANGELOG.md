# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.10](https://github.com/GodsGolemInc/nim-metal-compute/compare/v0.0.9...v0.0.10) (2025-12-28)

### Features

* **metal_api:** add unified ComputeContext with GPU/CPU fallback
* **metal_api:** add vectorAdd, vectorMul, matmul, transpose operations
* **logging:** integrate production logging with std/logging
* **stats:** add operation statistics tracking

## [0.0.9](https://github.com/GodsGolemInc/nim-metal-compute/compare/v0.0.8...v0.0.9) (2025-12-28)

### Features

* **metal_stress:** add comprehensive stress testing module
* **metal_stress:** add buffer allocation, vector/matrix compute stress tests
* **metal_stress:** add memory pressure and async operations tests
* **metal_optimize:** add shader optimization utilities
* **metal_optimize:** add thread group size optimization for 1D/2D/matmul

## [0.0.8](https://github.com/GodsGolemInc/nim-metal-compute/compare/v0.0.7...v0.0.8) (2025-12-28)

### Features

* **metal_async:** add async command buffer execution
* **metal_async:** add completion handlers with callbacks
* **metal_async:** add GPU timing queries (gpuStartTime, gpuEndTime)
* **metal_async:** add SharedEvent for cross-command buffer sync
* **metal_async:** add DoubleBuffer generic class for pipelining

## [0.0.7](https://github.com/GodsGolemInc/nim-metal-compute/compare/v0.0.6...v0.0.7) (2025-12-28)

### Features

* **metal_nn:** add NeuralNetworkGPU class for GPU inference
* **metal_nn:** add dense layer shader
* **metal_nn:** add ReLU, Sigmoid, Tanh, Softmax activation functions
* **metal_nn:** add multi-layer inference pipeline

### Performance

* neural network inference: 3319 inferences/sec on Apple M2

## [0.0.6](https://github.com/GodsGolemInc/nim-metal-compute/compare/v0.0.5...v0.0.6) (2025-12-28)

### Features

* **metal_pool:** add buffer pooling with size-based bucketing
* **metal_matrix:** add GPU matrix multiplication
* **metal_matrix:** add GPU matrix transpose

### Performance

* 512x512 matmul: 1398x speedup vs CPU

## [0.0.5](https://github.com/GodsGolemInc/nim-metal-compute/compare/v0.0.4...v0.0.5) (2025-12-28)

### Features

* **metal_shader:** add MTLLibrary compilation from source
* **metal_shader:** add MTLFunction extraction
* **metal_shader:** add MTLComputePipelineState creation
* **metal_compute:** add compute dispatch with buffer binding
* **metal_compute:** add vector addition/multiply shaders

## [0.0.4](https://github.com/GodsGolemInc/nim-metal-compute/compare/v0.0.3...v0.0.4) (2025-12-28)

### Features

* **metal_wrapper:** add C wrapper for Metal API (metal_wrapper.m)
* **metal_device:** add full MTLDevice property access
* **metal_buffer:** add actual buffer allocation/read/write
* **metal_command:** add command queue, buffer, encoder

### Bug Fixes

* replace problematic objc_msgSend with C wrapper

## [0.0.3](https://github.com/GodsGolemInc/nim-metal-compute/compare/v0.0.2...v0.0.3) (2025-12-28)

### Features

* **metal_device:** add MTLDevice bindings (stub)
* **metal_buffer:** add MTLBuffer management (stub)
* **metal_command:** add command queue/buffer/encoder (stub)
* **objc_runtime:** add Objective-C runtime bindings

## [0.0.2](https://github.com/GodsGolemInc/nim-metal-compute/compare/v0.0.1...v0.0.2) (2025-12-28)

### Features

* **errors:** add Result type (NMCResult) for error handling
* **errors:** add NMCErrorKind enumeration
* **errors:** add validation helpers

## [0.0.1](https://github.com/GodsGolemInc/nim-metal-compute/releases/tag/v0.0.1) (2025-12-28)

### Features

* **network_spec:** add NetworkSpec DSL for neural network definition
* **weights:** add Tensor storage with Xavier/Kaiming initialization
* **codegen:** add Metal shader and Nim CPU code generation
* **inference:** add CPU inference engines (SIMD, UltraFast, Extreme, Parallel, Actor)
