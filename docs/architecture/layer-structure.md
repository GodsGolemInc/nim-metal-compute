# Layer Structure

## Network Layer Types

### Dense (Fully-Connected) Layer

```
Input[n] → Weights[n×m] + Bias[m] → Activation → Output[m]
```

**Computation:**
```nim
for j in 0..<outputSize:
  output[j] = bias[j]
  for i in 0..<inputSize:
    output[j] += input[i] * weights[i, j]
  output[j] = activation(output[j])
```

### Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| ReLU | max(0, x) | Hidden layers |
| Softmax | exp(x)/Σexp | Output (classification) |
| Sigmoid | 1/(1+exp(-x)) | Binary output |
| Tanh | (exp(x)-exp(-x))/(exp(x)+exp(-x)) | Normalized output |

## Memory Layout

### Row-Major (Standard)
```
weights[inputSize * hiddenSize]
  [i0h0, i0h1, i0h2, ..., i1h0, i1h1, ...]
```

### Column-Major (Transposed)
```
weights_T[hiddenSize][inputSize]
  weights_T[h][i] for cache-friendly hidden neuron computation
```

## Fixed Network Architecture

For maximum performance, engines use fixed sizes:

```nim
const
  InputSize = 64    # Input vector dimension
  HiddenSize = 64   # Hidden layer neurons
  OutputSize = 23   # Output categories
```

## Layer Connection Validation

Networks must have matching layer dimensions:

```nim
proc validate*(spec: NetworkSpec): bool =
  for i in 1..<spec.layers.len:
    if spec.layers[i-1].outputSize != spec.layers[i].inputSize:
      return false
  return true
```

## Weight Initialization

### Xavier (Glorot)
For tanh/sigmoid activations:
```nim
scale = sqrt(2.0 / (inputSize + outputSize))
weight = random(-1, 1) * scale
```

### Kaiming (He)
For ReLU activations:
```nim
scale = sqrt(2.0 / inputSize)
weight = random(-1, 1) * scale
```
