# Custom Networks Guide

Design and implement custom neural network architectures.

## Network Design

### Basic MLP

```nim
var spec = newNetwork("SimpleMLP", inputSize = 128)
spec
  .addDense("hidden1", 128, 64, actReLU)
  .addDense("hidden2", 64, 32, actReLU)
  .addDense("output", 32, 10, actSoftmax)
```

### Deep Network

```nim
var spec = newNetwork("DeepMLP", inputSize = 256)
spec
  .addDense("h1", 256, 128, actReLU)
  .addDense("h2", 128, 64, actReLU)
  .addDense("h3", 64, 32, actReLU)
  .addDense("h4", 32, 16, actReLU)
  .addDense("out", 16, 5, actSoftmax)
```

### Binary Classifier

```nim
var spec = newNetwork("BinaryClassifier", inputSize = 100)
spec
  .addDense("hidden", 100, 50, actReLU)
  .addDense("output", 50, 2, actSoftmax)
```

## Validation

Always validate before use:

```nim
if not spec.validate():
  echo "Error: Layer dimensions don't match!"
  quit(1)

echo "Total parameters: ", spec.totalParams()
```

## Choosing Activation Functions

| Layer Type | Recommended Activation |
|------------|----------------------|
| Hidden layers | ReLU (actReLU) |
| Output (multi-class) | Softmax (actSoftmax) |
| Output (binary) | Sigmoid (actSigmoid) |
| Output (regression) | None (actNone) |

## Weight Initialization

| Activation | Initialization |
|------------|---------------|
| ReLU | Kaiming/He |
| Sigmoid/Tanh | Xavier/Glorot |

```nim
let nn = newNeuralNet(spec)

# For ReLU networks
nn.initWeights("kaiming", seed = 42)

# For Sigmoid/Tanh networks
nn.initWeights("xavier", seed = 42)
```

## Serialization

### JSON Export/Import

```nim
# Export architecture
let json = spec.toJson()
writeFile("architecture.json", json)

# Import architecture
let jsonStr = readFile("architecture.json")
let loadedSpec = fromJson(jsonStr)
```

### Binary Weights

```nim
# Save weights
nn.weights.saveNMW("weights.nmw")

# Load weights
let weights = loadNMW("weights.nmw")
```

## Code Generation

Generate optimized inference code:

```nim
var opts = defaultOptions()
opts.outputDir = "generated/"
opts.generateMetal = true
opts.generateNimCPU = true

spec.generate(opts)
# Creates:
#   generated/mynetwork.metal
#   generated/mynetwork_cpu.nim
```

## Performance Tips

1. **Fixed Sizes**: Use power-of-2 dimensions (32, 64, 128)
2. **Shallow > Deep**: Fewer wider layers often faster
3. **Batch Processing**: Process multiple inputs together
4. **Use Specialized Engines**: ParallelInferenceEngine for high throughput
