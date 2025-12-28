# CodeGen API Reference

Code generation for Metal shaders and Nim CPU inference.

## Types

### CodeGenOptions
```nim
type
  CodeGenOptions* = object
    outputDir*: string
    generateMetal*: bool
    generateNimCPU*: bool
    optimizationLevel*: int
```

## Procedures

### defaultOptions
```nim
proc defaultOptions*(): CodeGenOptions
```
Returns default code generation options.

### generateMetalKernel
```nim
proc generateMetalKernel*(spec: NetworkSpec): string
```
Generates Metal compute shader source code.

**Returns:** Metal shader source as string.

### generateNimCPU
```nim
proc generateNimCPU*(spec: NetworkSpec): string
```
Generates optimized Nim CPU inference code.

**Returns:** Nim source code as string.

### generate
```nim
proc generate*(spec: NetworkSpec, opts: CodeGenOptions)
```
Generates and writes code files to disk.

**Output files:**
- `{name}.metal` - Metal compute shader
- `{name}_cpu.nim` - Nim CPU implementation

## Generated Code Features

### Metal Shader
- Parallel batch inference kernel
- Optimized memory access patterns
- ReLU/Softmax activation functions

### Nim CPU
- SIMD-optimized operations
- Loop unrolling
- Cache-friendly memory layout

## Example

```nim
let spec = koanClassifierSpec()

# Generate Metal shader
let metalCode = generateMetalKernel(spec)
writeFile("inference.metal", metalCode)

# Generate both to directory
var opts = defaultOptions()
opts.outputDir = "generated/"
spec.generate(opts)
# Creates: generated/koanclassifier.metal
#          generated/koanclassifier_cpu.nim
```
