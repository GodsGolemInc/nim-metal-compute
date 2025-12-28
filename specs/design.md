# Technical Design Specification

Version: 0.0.x Series → 0.1.0 (Production)

## 1. Architecture Overview

### Current Architecture (v0.0.1)

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ UnifiedAPI  │  │   Batch     │  │  Actor System   │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
└─────────┼────────────────┼──────────────────┼───────────┘
          │                │                  │
┌─────────▼────────────────▼──────────────────▼───────────┐
│                   CPU Inference Engines                  │
│  ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │
│  │   SIMD     │ │  Extreme   │ │     Parallel       │   │
│  │  (500K/s)  │ │  (2M+/s)   │ │     (10M+/s)       │   │
│  └────────────┘ └────────────┘ └────────────────────┘   │
└─────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────┐
│                   Code Generation                        │
│  ┌────────────┐ ┌────────────┐                          │
│  │ MSL Output │ │ Nim Output │                          │
│  │ (Shader)   │ │ (CPU)      │                          │
│  └────────────┘ └────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

### Target Architecture (v0.0.5+)

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ UnifiedAPI  │  │   Batch     │  │  Actor System   │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
└─────────┼────────────────┼──────────────────┼───────────┘
          │                │                  │
┌─────────▼────────────────▼──────────────────▼───────────┐
│                   Backend Abstraction                    │
│  ┌────────────────────┐ ┌────────────────────────────┐  │
│  │     CPU Backend    │ │       Metal Backend        │  │
│  │  ┌──────┐ ┌──────┐ │ │  ┌──────┐ ┌────────────┐  │  │
│  │  │ SIMD │ │Paral │ │ │  │Device│ │ComputePipe │  │  │
│  │  └──────┘ └──────┘ │ │  └──────┘ └────────────┘  │  │
│  └────────────────────┘ └────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Module Design

### 2.1 Core Modules (v0.0.1) ✅

```nim
# network_spec.nim
type
  ActivationType* = enum
    actNone, actReLU, actSoftmax, actSigmoid, actTanh

  LayerSpec* = object
    name*: string
    layerType*: LayerType
    inputSize*, outputSize*: int
    activation*: ActivationType

  NetworkSpec* = object
    name*: string
    layers*: seq[LayerSpec]
```

### 2.2 Backend Abstraction (v0.0.3+)

```nim
# backend.nim (proposed)
type
  BackendType* = enum
    btCPU      # Current SIMD/Extreme engines
    btMetal    # Metal compute shaders

  Backend* = ref object of RootObj
    backendType*: BackendType

  CPUBackend* = ref object of Backend
  MetalBackend* = ref object of Backend

proc newBackend*(bt: BackendType): Backend
proc infer*(b: Backend, input: Tensor): Tensor
proc inferBatch*(b: Backend, inputs: seq[Tensor]): seq[Tensor]
```

### 2.3 Metal Bindings (v0.0.3)

```nim
# metal_device.nim (proposed)
type
  MTLDevice* = distinct pointer

  DeviceInfo* = object
    name*: string
    isAppleSilicon*: bool
    hasUnifiedMemory*: bool
    maxThreadsPerGroup*: int

proc getDefaultDevice*(): MTLDevice
proc getDeviceInfo*(device: MTLDevice): DeviceInfo
proc isMetalAvailable*(): bool
```

### 2.4 Buffer Management (v0.0.3)

```nim
# metal_buffer.nim (proposed)
type
  MTLBuffer* = distinct pointer

  BufferMode* = enum
    bmShared     # CPU and GPU can access (unified memory)
    bmPrivate    # GPU only
    bmManaged    # Explicit sync required

proc newBuffer*(device: MTLDevice, size: int, mode: BufferMode): MTLBuffer
proc write*(buffer: MTLBuffer, data: pointer, size: int)
proc read*(buffer: MTLBuffer, dest: pointer, size: int)
proc release*(buffer: MTLBuffer)
```

### 2.5 Compute Pipeline (v0.0.4)

```nim
# metal_pipeline.nim (proposed)
type
  MTLComputePipeline* = distinct pointer
  MTLLibrary* = distinct pointer
  MTLFunction* = distinct pointer

proc compileShader*(device: MTLDevice, source: string): MTLLibrary
proc getFunction*(library: MTLLibrary, name: string): MTLFunction
proc createPipeline*(device: MTLDevice, function: MTLFunction): MTLComputePipeline
```

### 2.6 Command Execution (v0.0.5)

```nim
# metal_command.nim (proposed)
type
  MTLCommandQueue* = distinct pointer
  MTLCommandBuffer* = distinct pointer
  MTLComputeEncoder* = distinct pointer

proc newCommandQueue*(device: MTLDevice): MTLCommandQueue
proc newCommandBuffer*(queue: MTLCommandQueue): MTLCommandBuffer
proc newComputeEncoder*(buffer: MTLCommandBuffer): MTLComputeEncoder

proc setBuffer*(encoder: MTLComputeEncoder, buffer: MTLBuffer, index: int)
proc setPipeline*(encoder: MTLComputeEncoder, pipeline: MTLComputePipeline)
proc dispatch*(encoder: MTLComputeEncoder, gridSize, groupSize: tuple[x,y,z: int])
proc commit*(buffer: MTLCommandBuffer)
proc waitUntilCompleted*(buffer: MTLCommandBuffer)
```

---

## 3. Data Flow

### 3.1 Inference Pipeline

```
Input (seq[float32])
    │
    ▼
┌──────────────┐
│ Validate     │ Check dimensions
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Backend      │ Select: CPU / Metal
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Forward Pass │ Layer-by-layer computation
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Post-process │ Argmax, confidence
└──────┬───────┘
       │
       ▼
Output (category, confidence)
```

### 3.2 Memory Layout

```
CPU Backend:
  - Row-major / Transposed for cache efficiency
  - Stack allocation for fixed sizes
  - Pre-allocated buffers

Metal Backend:
  - MTLBuffer for GPU memory
  - Shared memory mode for unified memory (Apple Silicon)
  - Private memory for GPU-only operations
  - Double buffering for async
```

### 3.3 Metal Execution Flow (v0.0.5)

```
┌──────────────┐
│ Input Data   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ MTLBuffer    │ Allocate & copy input
│ (Shared)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Compute      │ Bind buffers, dispatch
│ Encoder      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ GPU Execute  │ Run shader
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Read Result  │ Copy from GPU buffer
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Output       │
└──────────────┘
```

---

## 4. API Design

### 4.1 Unified API (v0.0.1) ✅

```nim
# Simple high-level API
let nn = newNeuralNet(spec)
nn.initWeights("kaiming", 42)
let (category, confidence) = nn.infer(input)
```

### 4.2 Backend Selection (v0.0.5)

```nim
# Explicit backend selection
let nn = newNeuralNet(spec, backend = btMetal)
nn.initWeights("kaiming", 42)
let result = nn.infer(input)  # Uses Metal GPU
```

### 4.3 Auto Backend (v0.1.0)

```nim
# Automatic best backend selection
let nn = newNeuralNet(spec, backend = btAuto)
# Selects: Metal (if available) > CPU
# Automatic fallback on error
```

---

## 5. Error Handling

### 5.1 Current (v0.0.1)

```nim
# Simple exception-based
proc validate*(spec: NetworkSpec): bool
# Returns false on invalid configuration
```

### 5.2 Proposed (v0.0.2)

```nim
type
  NMCError* = object of CatchableError
  ValidationError* = object of NMCError
  MetalError* = object of NMCError
  BufferError* = object of NMCError

proc validate*(spec: NetworkSpec): Result[void, ValidationError]
```

### 5.3 Metal-Specific Errors (v0.0.3)

```nim
type
  MetalNotAvailable* = object of MetalError
  ShaderCompilationError* = object of MetalError
  BufferAllocationError* = object of MetalError
  PipelineCreationError* = object of MetalError
```

---

## 6. Configuration

### 6.1 Runtime Configuration

```nim
type
  NMCConfig* = object
    defaultBackend*: BackendType
    numThreads*: int
    enableProfiling*: bool
    logLevel*: LogLevel
    metalValidation*: bool  # Enable Metal validation layer

var globalConfig* = NMCConfig(
  defaultBackend: btAuto,
  numThreads: 0,  # Auto-detect
  enableProfiling: false,
  logLevel: llWarn,
  metalValidation: false
)
```

### 6.2 Compile-Time Configuration

```nim
# nim.cfg or config.nims
when defined(nmcMetal):
  const HasMetal* = true
when defined(nmcCPUOnly):
  const HasMetal* = false
```

---

## 7. Out of Scope

以下はnim-metal-computeのスコープ外です:

| Feature | Reason | Alternative |
|---------|--------|-------------|
| MLX統合 | 高レベルフレームワーク | nim-ml |
| Training | ML専用機能 | nim-ml |
| ONNX | モデル形式 | nim-ml |
| Conv2D/Pooling | 高レベル層 | nim-ml |

