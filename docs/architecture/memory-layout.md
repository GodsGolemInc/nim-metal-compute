# Memory Layout

## Optimization Strategies

### 1. Cache Line Alignment

Modern CPUs load 64 bytes at a time. Align data structures:

```nim
const CacheLineSize = 64

type
  PaddedWorker = object
    engine: ExtremeEngine
    data: array[64, float32]
    pad: array[CacheLinePadding, byte]  # Prevent false sharing
```

### 2. Transposed Weight Matrices

Standard layout (row-major):
```
weights[i * hiddenSize + j]  # Non-sequential j access
```

Optimized layout (transposed):
```nim
wIH_T: array[HiddenSize, array[InputSize, float32]]
# Access: wIH_T[j][i] - sequential i access for each neuron j
```

### 3. Contiguous Buffers

Pre-allocated buffers eliminate runtime allocation:

```nim
type
  ExtremeEngine = object
    hidden: array[HiddenSize, float32]  # Reusable
    output: array[OutputSize, float32]  # Reusable
```

## Memory Access Patterns

### Dense Layer Forward Pass

```nim
# Optimized: Sequential read of weights for each neuron
for j in 0..<HiddenSize:
  var sum = bias[j]
  for i in 0..<InputSize:
    sum += input[i] * wIH_T[j][i]  # Sequential access
  hidden[j] = relu(sum)
```

### Loop Unrolling (8x)

```nim
var i = 0
while i < InputSize:
  sum += input[i] * wIH_T[j][i] +
         input[i+1] * wIH_T[j][i+1] +
         input[i+2] * wIH_T[j][i+2] +
         input[i+3] * wIH_T[j][i+3] +
         input[i+4] * wIH_T[j][i+4] +
         input[i+5] * wIH_T[j][i+5] +
         input[i+6] * wIH_T[j][i+6] +
         input[i+7] * wIH_T[j][i+7]
  i += 8
```

## Multi-Threading Considerations

### False Sharing Prevention

Each worker has its own cache-line-padded data:

```nim
type
  PaddedWorker = object
    engine: ExtremeEngine
    inputBuf: ptr UncheckedArray[array[64, float32]]
    outputBuf: ptr UncheckedArray[int]
    startIdx, endIdx: int
    pad: array[128, byte]  # Separate cache lines
```

### Work Distribution

```nim
let batchPerThread = (count + numThreads - 1) div numThreads
for i in 0..<numThreads:
  workers[i].startIdx = i * batchPerThread
  workers[i].endIdx = min((i + 1) * batchPerThread, count)
```

## Binary Format (NMW)

```
Header (16 bytes):
  magic: "NMW1" (4 bytes)
  version: uint32
  numTensors: uint32
  reserved: uint32

For each tensor:
  nameLen: uint32
  name: char[nameLen]
  numDims: uint32
  dims: int32[numDims]
  dataSize: uint32
  data: float32[dataSize]
```
