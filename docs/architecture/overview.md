# Architecture Overview

nim-metal-compute is designed for high-performance neural network inference on Apple Silicon and x86 platforms.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ UnifiedAPI  │  │   Batch     │  │  Actor System   │  │
│  │  (Simple)   │  │ Processing  │  │  (Concurrent)   │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
└─────────┼────────────────┼──────────────────┼───────────┘
          │                │                  │
┌─────────▼────────────────▼──────────────────▼───────────┐
│                   Inference Engines                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │
│  │   SIMD     │ │  Extreme   │ │     Parallel       │   │
│  │  Engine    │ │  Engine    │ │     Engine         │   │
│  └────────────┘ └────────────┘ └────────────────────┘   │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                   Core Components                        │
│  ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │
│  │ NetworkSpec│ │  Weights   │ │     CodeGen        │   │
│  │   (DSL)    │ │ (Storage)  │ │ (Metal/Nim CPU)    │   │
│  └────────────┘ └────────────┘ └────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Core Layer

| Component | Responsibility |
|-----------|----------------|
| NetworkSpec | Network architecture definition (DSL) |
| Weights | Weight storage, initialization, serialization |
| CodeGen | Metal shader and Nim CPU code generation |

### Inference Layer

| Engine | Optimization Focus |
|--------|-------------------|
| SIMD | Vectorized operations |
| UltraFast | Loop unrolling, cache optimization |
| Extreme | Fast approximations, zero allocation |
| Parallel | Multi-threaded batch processing |
| Actor | Message-passing concurrency |

### Application Layer

| Interface | Use Case |
|-----------|----------|
| UnifiedAPI | Simple high-level API |
| BatchProcessing | High-throughput workloads |
| ActorSystem | Concurrent request handling |

## Data Flow

```
1. Define Network
   NetworkSpec DSL → LayerSpec[]

2. Initialize Weights
   NetworkSpec → NetworkWeights → Tensor[]

3. Generate Code (optional)
   NetworkSpec → Metal Shader / Nim CPU

4. Create Engine
   NetworkSpec + Weights → InferenceEngine

5. Run Inference
   Input[float32] → Engine → (category, confidence)
```

## Design Principles

1. **Zero-Copy Operations** - Minimize memory allocations during inference
2. **Cache-Friendly Layout** - Transposed weight matrices for sequential access
3. **Compile-Time Optimization** - Fixed sizes enable loop unrolling
4. **Separation of Concerns** - Spec, Weights, and Engine are independent
5. **Platform Abstraction** - Same API for CPU and GPU execution
