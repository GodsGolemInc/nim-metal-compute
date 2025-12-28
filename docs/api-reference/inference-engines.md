# Inference Engines API Reference

Specialized high-performance inference engines.

## Engine Comparison

| Engine | Throughput | Features | Use Case |
|--------|------------|----------|----------|
| SIMDInference | 500K/s | SIMD optimized | Single-thread baseline |
| UltraFast | 1M/s | Loop unrolling | Low-latency single |
| Extreme | 2M+/s | Fast exp, no alloc | Maximum single-thread |
| Parallel | 10M+/s | Multi-thread | High-throughput batch |
| Actor | 5M+/s | Message passing | Concurrent systems |

---

## SIMDInferenceEngine

SIMD-optimized inference for Apple Silicon / x86.

### Type
```nim
type
  SIMDInferenceEngine* = object
    inputSize*, hiddenSize*, outputSize*: int
    weightsIH*, weightsHO*: seq[float32]
    biasH*, biasO*: seq[float32]
```

### Procedures
```nim
proc newSIMDInferenceEngine*(inputSize, hiddenSize, outputSize: int): SIMDInferenceEngine
proc initKaiming*(engine: var SIMDInferenceEngine, seed: int = 42)
proc inferFast*(engine: var SIMDInferenceEngine, input: openArray[float32]): tuple[category: int, confidence: float32]
proc inferBatchFast*(engine: var SIMDInferenceEngine, inputs: openArray[seq[float32]], categories: var seq[int], confidences: var seq[float32])
proc inferBatch4*(engine: var SIMDInferenceEngine, i0, i1, i2, i3: openArray[float32]): array[4, tuple[category: int, confidence: float32]]
```

---

## UltraFastEngine

Ultra-optimized with loop unrolling and cache optimization.

### Type
```nim
type
  UltraFastEngine* = object
    weights*: AlignedWeights
    hidden*: array[64, float32]
    output*: array[23, float32]
```

### Procedures
```nim
proc initWeights*(engine: var UltraFastEngine, seed: uint32 = 42)
proc inferUltraArray*(engine: var UltraFastEngine, input: array[64, float32]): tuple[cat: int, conf: float32]
proc inferBatch8*(engine: var UltraFastEngine, inputs: ptr UncheckedArray[array[64, float32]], results: ptr UncheckedArray[tuple[cat: int, conf: float32]])
```

---

## ExtremeEngine

Maximum performance with fast exp approximation.

### Constants
```nim
const
  InputSize* = 64
  HiddenSize* = 64
  OutputSize* = 23
```

### Type
```nim
type
  ExtremeEngine* = object
    wIH_T*: array[64, array[64, float32]]  # Transposed
    wHO_T*: array[23, array[64, float32]]  # Transposed
    bH*, bO*: array[64/23, float32]
```

### Procedures
```nim
proc initWeights*(engine: var ExtremeEngine, seed: uint32 = 42)
proc inferExtreme*(engine: var ExtremeEngine, input: ptr array[64, float32]): int
proc inferExtremeWithConf*(engine: var ExtremeEngine, input: ptr array[64, float32]): tuple[cat: int, conf: float32]
```

---

## ParallelInferenceEngine

Multi-threaded parallel inference.

### Type
```nim
type
  ParallelInferenceEngine* = object
    numThreads*: int
    bufferSize*: int
    outputBuffer*: seq[int]
    confBuffer*: seq[float32]

  BatchInferenceResult* = object
    categories*: seq[int]
    confidences*: seq[float32]
    count*: int
    throughput*: float
```

### Procedures
```nim
proc initParallelEngine*(engine: var ParallelInferenceEngine, numThreads: int = 0, bufferSize: int = 1_000_000)
proc inferBatchFast*(engine: var ParallelInferenceEngine, inputs: seq[array[64, float32]]): BatchInferenceResult
proc inferBatch*(engine: var ParallelInferenceEngine, inputs: seq[array[64, float32]]): BatchInferenceResult
```

---

## InferenceActorSystem

Actor-based concurrent inference.

### Types
```nim
type
  InferenceRequest* = object
    id*: int
    input*: array[64, float32]

  InferenceResponse* = object
    id*: int
    category*: int
    confidence*: float32

  WorkerState* = enum
    wsIdle, wsProcessing, wsStopped

  InferenceWorker* = object
    id*: int
    state*: WorkerState
    processedCount*: int

  InferenceActorSystem* = object
    workers*: array[8, InferenceWorker]
    isRunning*: bool
```

### Procedures
```nim
proc initActorSystem*(system: var InferenceActorSystem)
proc routeRequest*(system: var InferenceActorSystem, req: InferenceRequest): bool
proc tick*(system: var InferenceActorSystem): int
proc collectResults*(system: var InferenceActorSystem, output: var seq[InferenceResponse]): int
proc shutdown*(system: var InferenceActorSystem)
```

---

## Usage Examples

### High-Throughput Batch Processing
```nim
var engine: ParallelInferenceEngine
engine.initParallelEngine(numThreads = 8)

var inputs = newSeq[array[64, float32]](1_000_000)
# ... fill inputs ...

let result = engine.inferBatchFast(inputs)
echo "Throughput: ", result.throughput, " samples/sec"
```

### Low-Latency Single Inference
```nim
var engine: ExtremeEngine
engine.initWeights(42)

var input: array[64, float32]
# ... fill input ...

let category = engine.inferExtreme(addr input)
```

### Concurrent Actor System
```nim
var system: InferenceActorSystem
system.initActorSystem()

# Submit requests
for i in 0..<1000:
  let req = InferenceRequest(id: i, input: input)
  discard system.routeRequest(req)

# Process
while true:
  let processed = system.tick()
  if processed == 0: break

# Collect results
var results: seq[InferenceResponse]
discard system.collectResults(results)
```
