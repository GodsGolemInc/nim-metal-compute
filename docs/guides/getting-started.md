# Getting Started

Quick start guide for nim-metal-compute.

## Installation

```bash
# Clone repository
git clone https://github.com/jasagiri/nim-metal-compute.git
cd nim-metal-compute

# Install dependencies
nimble install

# Run tests
nimble test
```

## Basic Usage

### 1. Define a Network

```nim
import nim_metal_compute

# Using DSL
var spec = newNetwork("MyClassifier", 64)
spec
  .addDense("hidden", 64, 32, actReLU)
  .addDense("output", 32, 10, actSoftmax)

# Or use a preset
let koanSpec = koanClassifierSpec()
```

### 2. Create and Initialize

```nim
let nn = newNeuralNet(spec)
nn.initWeights("kaiming", 42)  # ReLU networks use Kaiming
```

### 3. Run Inference

```nim
# Prepare input
var input = newSeq[float32](64)
for i in 0..<64:
  input[i] = yourData[i]

# Single inference
let (category, confidence) = nn.infer(input)
echo "Predicted: ", category, " (", confidence * 100, "%)"
```

### 4. Batch Processing

```nim
var batch = newSeq[seq[float32]](100)
for i in 0..<100:
  batch[i] = loadSample(i)

let result = nn.inferBatch(batch)
echo "Processed ", result.batchSize, " samples"
echo "Time: ", result.inferenceTimeMs, " ms"
```

## High-Performance Mode

For maximum throughput, use specialized engines:

```nim
import nim_metal_compute/parallel_inference

var engine: ParallelInferenceEngine
engine.initParallelEngine(numThreads = 8)

var inputs = newSeq[array[64, float32]](1_000_000)
# ... fill inputs ...

let result = engine.inferBatchFast(inputs)
echo "Throughput: ", result.throughput, " samples/sec"
```

## Save and Load Weights

```nim
# Save trained weights
nn.weights.saveNMW("model.nmw")

# Load weights
let loaded = loadNMW("model.nmw")
```

## Next Steps

- [Custom Networks](custom-networks.md) - Design your own architectures
- [Performance Guide](../performance/optimization-guide.md) - Optimize for your use case
- [API Reference](../api-reference/) - Complete API documentation
