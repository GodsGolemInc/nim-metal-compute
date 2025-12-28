# Optimization Guide

Techniques to maximize inference performance.

## Compiler Flags

### Release Build
```bash
nim c -d:release -d:danger --opt:speed src/main.nim
```

### CPU-Specific
```bash
nim c -d:release --passC:"-march=native" src/main.nim
```

### LTO (Link-Time Optimization)
```bash
nim c -d:release --passC:"-flto" --passL:"-flto" src/main.nim
```

## Code-Level Optimizations

### 1. Disable Runtime Checks

```nim
{.push checks:off, boundChecks:off, overflowChecks:off.}
proc inferFast*(engine: var Engine, input: openArray[float32]): int =
  # Performance-critical code
{.pop.}
```

### 2. Use Fixed-Size Arrays

```nim
# Slower: dynamic allocation
var hidden = newSeq[float32](64)

# Faster: stack allocation
var hidden: array[64, float32]
```

### 3. Pointer Arithmetic

```nim
# Slower: bounds-checked access
result += input[i] * weights[i * hiddenSize + j]

# Faster: unchecked pointer access
let wPtr = cast[ptr UncheckedArray[float32]](addr weights[0])
result += input[i] * wPtr[i * hiddenSize + j]
```

### 4. Loop Unrolling

```nim
# Unroll 8x for modern CPUs
var i = 0
while i < 64:
  sum += input[i] * w[i] +
         input[i+1] * w[i+1] +
         input[i+2] * w[i+2] +
         input[i+3] * w[i+3] +
         input[i+4] * w[i+4] +
         input[i+5] * w[i+5] +
         input[i+6] * w[i+6] +
         input[i+7] * w[i+7]
  i += 8
```

### 5. Fast Exp Approximation

```nim
template fastExp(x: float32): float32 =
  let t = 1.0f32 + x * 0.00390625f32  # x/256
  let t2 = t * t
  let t4 = t2 * t2
  let t8 = t4 * t4
  let t16 = t8 * t8
  let t32 = t16 * t16
  let t64 = t32 * t32
  let t128 = t64 * t64
  t128 * t128  # (1 + x/256)^256
```

## Memory Optimizations

### 1. Transpose Weight Matrices

```nim
# Original: weights[input][hidden]
# Transposed: weights_T[hidden][input]

# Enables sequential memory access per neuron
for j in 0..<HiddenSize:
  for i in 0..<InputSize:
    sum += input[i] * weights_T[j][i]  # Sequential!
```

### 2. Pre-allocate Buffers

```nim
type
  Engine = object
    hidden: array[64, float32]  # Reused every inference
    output: array[23, float32]  # No allocation needed
```

### 3. Cache Line Padding

```nim
const CacheLinePadding = 128

type
  PaddedWorker = object
    engine: ExtremeEngine
    data: WorkerData
    pad: array[CacheLinePadding, byte]  # Prevent false sharing
```

## Threading Optimizations

### 1. Work Partitioning

```nim
let batchPerThread = (count + numThreads - 1) div numThreads
for i in 0..<numThreads:
  workers[i].startIdx = i * batchPerThread
  workers[i].endIdx = min((i + 1) * batchPerThread, count)
```

### 2. Avoid Contention

- Each thread owns its own engine instance
- No shared mutable state during inference
- Results written to separate memory regions

### 3. Thread Pool Reuse

```nim
# Initialize once
var engine: ParallelInferenceEngine
engine.initParallelEngine(numThreads = 8)

# Reuse for all batches
for batch in batches:
  discard engine.inferBatchFast(batch)
```

## Choosing the Right Engine

| Scenario | Recommended Engine |
|----------|-------------------|
| Single sample, lowest latency | ExtremeEngine |
| Batch processing, high throughput | ParallelInferenceEngine |
| Concurrent requests | InferenceActorSystem |
| Simple integration | UnifiedAPI (NeuralNet) |

## Profiling

```nim
import std/times

let start = cpuTime()
for _ in 0..<iterations:
  discard engine.inferExtreme(addr input)
let elapsed = cpuTime() - start

echo "Throughput: ", iterations.float / elapsed, " samples/sec"
echo "Latency: ", elapsed / iterations.float * 1e6, " μs"
```
