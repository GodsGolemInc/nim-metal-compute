# nim-metal-compute

GPU-accelerated compute library for Nim using Apple Metal framework.

## Features

- **Metal GPU Compute**: Vector and matrix operations on GPU
- **Neural Network Inference**: GPU-accelerated neural network forward pass
- **Automatic Fallback**: GPU→CPU fallback when Metal is unavailable
- **Async Execution**: Non-blocking GPU operations with completion handlers
- **Buffer Pooling**: Efficient memory management with buffer reuse
- **Stress Testing**: Comprehensive GPU stress testing utilities

## Requirements

- macOS 10.15+ (Catalina or later)
- Nim 2.0.0+
- Xcode Command Line Tools

## Installation

```bash
nimble install nim_metal_compute
```

## Quick Start

```nim
import nim_metal_compute/metal_api

# Create compute context with automatic backend selection
let ctxResult = newComputeContext(cbAuto)
if ctxResult.isOk:
  var ctx = ctxResult.get

  # Vector addition
  let a = @[1.0f, 2.0f, 3.0f, 4.0f]
  let b = @[5.0f, 6.0f, 7.0f, 8.0f]
  let result = ctx.vectorAdd(a, b)

  if result.success:
    echo "Result: ", result.data
    echo "Backend: ", result.backend

  ctx.destroy()
```

## Modules

| Module | Description |
|--------|-------------|
| `metal_device` | Metal device enumeration and management |
| `metal_buffer` | GPU buffer allocation and data transfer |
| `metal_command` | Command queue and command buffer management |
| `metal_shader` | Shader compilation and pipeline creation |
| `metal_compute` | Vector compute operations |
| `metal_matrix` | Matrix multiplication and transpose |
| `metal_nn` | Neural network GPU inference |
| `metal_async` | Async execution and double buffering |
| `metal_pool` | Buffer pooling for memory efficiency |
| `metal_stress` | Stress testing utilities |
| `metal_optimize` | Thread group optimization |
| `metal_api` | Unified API with GPU/CPU fallback |

## Performance

Benchmarks on Apple M2:

| Operation | Size | GPU Time | CPU Time | Speedup |
|-----------|------|----------|----------|---------|
| Matrix Multiply | 64x64 | 0.21ms | 1.01ms | 4.8x |
| Matrix Multiply | 128x128 | 0.07ms | 9.80ms | 140x |
| Matrix Multiply | 256x256 | 0.17ms | 80.4ms | 473x |
| Matrix Multiply | 512x512 | 0.46ms | 643ms | 1398x |

## CPU Inference Engines

For CPU-only environments, multiple inference engines are available:

| Engine | Throughput | Description |
|--------|------------|-------------|
| SIMDInference | 500K/s | SIMD-optimized inference |
| UltraFastInference | 1M/s | Cache-optimized inference |
| ExtremeInference | 2M+/s | Extreme optimization |
| ParallelInference | 10M+/s | Multi-threaded inference |
| ActorInference | 5M+/s | Actor-based parallelism |

## Documentation

- [Getting Started](docs/guides/getting-started.md)
- [API Reference](docs/api-reference/)
- [Architecture](docs/architecture/)
- [Performance Guide](docs/performance/)

## License

Apache-2.0
