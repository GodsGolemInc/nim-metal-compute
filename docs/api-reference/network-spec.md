# NetworkSpec API Reference

Network specification DSL for defining neural network architectures.

## Types

### NetworkSpec
```nim
type
  NetworkSpec* = object
    name*: string
    layers*: seq[LayerSpec]
```

### LayerSpec
```nim
type
  LayerSpec* = object
    name*: string
    inputSize*: int
    outputSize*: int
    activation*: ActivationType
```

### ActivationType
```nim
type
  ActivationType* = enum
    actNone
    actReLU
    actSoftmax
    actSigmoid
    actTanh
```

## Procedures

### newNetwork
```nim
proc newNetwork*(name: string, inputSize: int): NetworkSpec
```
Creates a new network specification.

**Parameters:**
- `name`: Network identifier
- `inputSize`: Input layer size

**Returns:** `NetworkSpec`

### addDense
```nim
proc addDense*(spec: var NetworkSpec, name: string,
               inputSize, outputSize: int,
               activation: ActivationType): var NetworkSpec
```
Adds a dense (fully-connected) layer.

**Parameters:**
- `name`: Layer name
- `inputSize`: Input dimension
- `outputSize`: Output dimension
- `activation`: Activation function

**Returns:** `var NetworkSpec` (chainable)

### validate
```nim
proc validate*(spec: NetworkSpec): bool
```
Validates network layer connections.

### inputSize / outputSize
```nim
proc inputSize*(spec: NetworkSpec): int
proc outputSize*(spec: NetworkSpec): int
```
Returns network input/output dimensions.

### totalParams
```nim
proc totalParams*(spec: NetworkSpec): int
```
Returns total parameter count.

### toJson / fromJson
```nim
proc toJson*(spec: NetworkSpec): string
proc fromJson*(json: string): NetworkSpec
```
JSON serialization/deserialization.

## Presets

### koanClassifierSpec
```nim
proc koanClassifierSpec*(): NetworkSpec
```
Returns predefined Koan classifier (64 -> 64 -> 23).

## Example

```nim
var spec = newNetwork("MyNetwork", 128)
spec
  .addDense("hidden1", 128, 64, actReLU)
  .addDense("hidden2", 64, 32, actReLU)
  .addDense("output", 32, 10, actSoftmax)

assert spec.validate() == true
echo "Total parameters: ", spec.totalParams()
```
