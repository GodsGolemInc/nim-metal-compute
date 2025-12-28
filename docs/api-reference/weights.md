# NetworkWeights API Reference

Weight management for neural networks.

## Types

### NetworkWeights
```nim
type
  NetworkWeights* = object
    spec*: NetworkSpec
    tensors*: Table[string, Tensor]
```

### Tensor
```nim
type
  Tensor* = object
    shape*: seq[int]
    data*: seq[float32]
```

## Procedures

### newNetworkWeights
```nim
proc newNetworkWeights*(spec: NetworkSpec): NetworkWeights
```
Creates weight container from network specification.

### Initialization

#### initXavier
```nim
proc initXavier*(weights: var NetworkWeights, seed: int = 42)
```
Xavier/Glorot initialization for tanh/sigmoid activations.

#### initKaiming
```nim
proc initKaiming*(weights: var NetworkWeights, seed: int = 42)
```
Kaiming/He initialization for ReLU activations.

### Serialization

#### saveNMW / loadNMW
```nim
proc saveNMW*(weights: NetworkWeights, path: string)
proc loadNMW*(path: string): NetworkWeights
```
Save/load in NMW (Nim Metal Weights) binary format.

### Conversion

#### toFlatArray
```nim
proc toFlatArray*(weights: NetworkWeights): seq[float32]
```
Flattens all weights to single array.

#### fromFlatArray
```nim
proc fromFlatArray*(weights: var NetworkWeights, data: seq[float32])
```
Restores weights from flat array.

## Example

```nim
let spec = koanClassifierSpec()
var weights = newNetworkWeights(spec)
weights.initXavier(42)

# Access tensors
echo weights.tensors["hidden.weight"].shape  # @[64, 64]

# Save/load
weights.saveNMW("model.nmw")
let loaded = loadNMW("model.nmw")
```
