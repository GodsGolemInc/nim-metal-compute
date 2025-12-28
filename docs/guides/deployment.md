# Deployment Guide

Deploy nim-metal-compute in production environments.

## Build Configuration

### Release Build

```bash
nim c -d:release -d:danger --opt:speed src/main.nim
```

Flags explanation:
- `-d:release`: Enable optimizations
- `-d:danger`: Disable runtime checks
- `--opt:speed`: Optimize for speed over size

### Static Binary

```bash
nim c -d:release --passL:-static src/main.nim
```

## Deployment Patterns

### 1. Embedded Library

```nim
# inference_lib.nim
proc inferenceAPI*(input: ptr float32, inputLen: int): int {.exportc, dynlib.} =
  var engine: ExtremeEngine
  engine.initWeights(42)

  var arr: array[64, float32]
  copyMem(addr arr[0], input, 64 * sizeof(float32))

  result = engine.inferExtreme(addr arr)
```

Build shared library:
```bash
nim c -d:release --app:lib -o:libinference.so inference_lib.nim
```

### 2. Service Architecture

```nim
# inference_server.nim
import asynchttpserver, asyncdispatch, json

var engine: ParallelInferenceEngine
engine.initParallelEngine(numThreads = 8)

proc handler(req: Request) {.async.} =
  let body = parseJson(req.body)
  let inputs = body["inputs"].to(seq[array[64, float32]])
  let result = engine.inferBatchFast(inputs)
  await req.respond(Http200, $(%*{
    "categories": result.categories,
    "throughput": result.throughput
  }))

waitFor newAsyncHttpServer().serve(Port(8080), handler)
```

### 3. Actor-Based Concurrent Service

```nim
var system: InferenceActorSystem
system.initActorSystem()

# Handle requests concurrently
proc processRequest(id: int, input: array[64, float32]) =
  let req = InferenceRequest(id: id, input: input)
  discard system.routeRequest(req)

# Background processing loop
proc processLoop() {.async.} =
  while true:
    discard system.tick()
    await sleepAsync(1)

# Collect results
proc collectLoop() {.async.} =
  var results: seq[InferenceResponse]
  while true:
    discard system.collectResults(results)
    for r in results:
      sendResponse(r.id, r.category, r.confidence)
    results.setLen(0)
    await sleepAsync(10)
```

## Resource Management

### Memory

Pre-allocate buffers for predictable memory usage:

```nim
engine.initParallelEngine(
  numThreads = 8,
  bufferSize = 1_000_000  # Pre-allocate for 1M samples
)
```

### Threads

Match thread count to available cores:

```nim
import cpuinfo
let numCores = countProcessors()
engine.initParallelEngine(numThreads = numCores)
```

## Monitoring

### Throughput Metrics

```nim
let result = engine.inferBatchFast(inputs)
echo "Samples/sec: ", result.throughput
echo "Latency: ", result.elapsedNs / result.count, " ns/sample"
```

### Health Check

```nim
proc healthCheck(): bool =
  var testInput: array[64, float32]
  try:
    discard engine.inferExtreme(addr testInput)
    return true
  except:
    return false
```

## Platform-Specific Notes

### macOS (Apple Silicon)

- Use `-d:release` for best performance
- Metal shaders require macOS 10.13+

### Linux (x86_64)

- Ensure AVX/SSE support for SIMD
- Use `--passC:-march=native` for CPU-specific optimizations

### Cross-Compilation

```bash
# macOS ARM64
nim c --os:macosx --cpu:arm64 -d:release src/main.nim

# Linux x86_64
nim c --os:linux --cpu:amd64 -d:release src/main.nim
```
