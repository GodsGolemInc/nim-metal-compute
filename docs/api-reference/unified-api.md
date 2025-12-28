# UnifiedAPI Reference

High-level API for neural network inference.

## Types

### NeuralNet
```nim
type
  NeuralNet* = ref object
    spec*: NetworkSpec
    weights*: NetworkWeights
    initialized*: bool
```

### BatchResult
```nim
type
  BatchResult* = object
    batchSize*: int
    categories*: seq[int]
    confidences*: seq[float32]
    inferenceTimeMs*: float
```

## Procedures

### newNeuralNet
```nim
proc newNeuralNet*(spec: NetworkSpec): NeuralNet
```
Creates a new neural network instance.

### initWeights
```nim
proc initWeights*(nn: NeuralNet, method: string = "xavier", seed: int = 42)
```
Initializes network weights.

**Methods:**
- `"xavier"` - Xavier/Glorot initialization
- `"kaiming"` - Kaiming/He initialization

### infer
```nim
proc infer*(nn: NeuralNet, input: seq[float32]): tuple[category: int, confidence: float32]
```
Single sample inference.

**Parameters:**
- `input`: Input vector matching network input size

**Returns:** Predicted category and confidence score.

### inferBatch
```nim
proc inferBatch*(nn: NeuralNet, inputs: seq[seq[float32]]): BatchResult
```
Batch inference for multiple samples.

### benchmark
```nim
proc benchmark*(nn: NeuralNet, batchSize: int, iterations: int): tuple[avgTime: float, throughput: float]
```
Runs performance benchmark.

**Returns:**
- `avgTime`: Average inference time (ms)
- `throughput`: Samples per second

## Example

```nim
# Create and initialize
let spec = koanClassifierSpec()
let nn = newNeuralNet(spec)
nn.initWeights("xavier", 42)

# Single inference
var input = newSeq[float32](64)
for i in 0..<64: input[i] = i.float32 / 64.0
let (category, confidence) = nn.infer(input)
echo "Category: ", category, " Confidence: ", confidence

# Batch inference
var batch = newSeq[seq[float32]](100)
for i in 0..<100:
  batch[i] = newSeq[float32](64)
let result = nn.inferBatch(batch)
echo "Processed: ", result.batchSize, " in ", result.inferenceTimeMs, "ms"

# Benchmark
let (avgTime, throughput) = nn.benchmark(32, 1000)
echo "Throughput: ", throughput, " samples/sec"
```
