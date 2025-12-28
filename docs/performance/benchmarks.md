# Performance Benchmarks

Benchmark results for nim-metal-compute inference engines.

## Test Environment

- **CPU**: Apple M2 (8-core)
- **Memory**: 16GB
- **OS**: macOS 14.x
- **Nim**: 2.0.0+
- **Build**: `nim c -d:release -d:danger`

## Network Configuration

```
Input: 64 neurons
Hidden: 64 neurons (ReLU)
Output: 23 neurons (Softmax)
Total Parameters: 5,911
```

## Single-Thread Performance

| Engine | Throughput | Latency | Notes |
|--------|------------|---------|-------|
| SIMDInference | 500K/s | 2.0 μs | Baseline SIMD |
| UltraFast | 1.0M/s | 1.0 μs | Loop unrolling |
| Extreme | 2.5M/s | 0.4 μs | Fast exp, category only |
| Extreme+Conf | 2.0M/s | 0.5 μs | With confidence |

## Multi-Thread Performance

| Threads | Throughput | Speedup | Efficiency |
|---------|------------|---------|------------|
| 1 | 2.5M/s | 1.0x | 100% |
| 2 | 4.8M/s | 1.9x | 96% |
| 4 | 9.2M/s | 3.7x | 92% |
| 8 | 17M/s | 6.8x | 85% |

## Batch Processing

| Batch Size | Throughput | 100K Samples |
|------------|------------|--------------|
| 1 | 2.5M/s | 40 ms |
| 100 | 8M/s | 12.5 ms |
| 10,000 | 15M/s | 6.7 ms |
| 1,000,000 | 20M/s | 5.0 ms |

## Memory Usage

| Component | Memory |
|-----------|--------|
| ExtremeEngine | 40 KB |
| ParallelEngine (8 threads) | 350 KB |
| 1M sample buffer | 256 MB |

## Comparison with Other Frameworks

| Framework | Throughput | Relative |
|-----------|------------|----------|
| nim-metal-compute | 20M/s | 1.0x |
| PyTorch (CPU) | 100K/s | 0.005x |
| TensorFlow Lite | 500K/s | 0.025x |
| ONNX Runtime | 1M/s | 0.05x |

*Note: Comparisons are approximate and depend on specific configurations.*

## Running Benchmarks

```bash
# Build with optimizations
nim c -d:release -d:danger src/nim_metal_compute/extreme_inference.nim
./bin/extreme_inference

# Parallel benchmark
nim c -d:release -d:danger src/nim_metal_compute/parallel_inference.nim
./bin/parallel_inference

# Threaded benchmark (wall-clock timing)
nim c -d:release -d:danger src/nim_metal_compute/threaded_inference.nim
./bin/threaded_inference

# All benchmarks
nimble bench
```

## Benchmark Modules

| Module | Purpose |
|--------|---------|
| `extreme_inference` | Single-thread performance |
| `parallel_inference` | Multi-thread batch processing |
| `threaded_inference` | Wall-clock parallel scaling |
| `actor_inference` | Actor model throughput |
