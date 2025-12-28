# Roadmap

## Version Overview

```
0.0.1  Complete   CPU推論エンジン (SIMD/並列)        ✅
0.0.2  Complete   エラーハンドリング改善             ✅
0.0.3  Current    Metal APIバインディング            ✅
0.0.4  Next       Compute Pipeline実装
0.0.5  Planned    シェーダーランタイム実行
0.0.6  Planned    バッファ最適化
0.0.7  Planned    非同期実行
0.0.8  Planned    プロファイリング
0.0.9  Planned    安定化・最適化
0.1.0  Milestone  Production ready
```

---

## v0.0.1 - Initial Release ✅

**Status:** Complete
**Release Date:** 2025-12-28

### Features
- [x] NetworkSpec DSL
- [x] Dense layer support
- [x] Activation functions (ReLU, Softmax, Sigmoid, Tanh)
- [x] Xavier/Kaiming weight initialization
- [x] NMW binary format
- [x] Metal shader code generation (MSL出力)
- [x] Nim CPU code generation

### Inference Engines (CPU)
- [x] UnifiedAPI (high-level)
- [x] SIMDInference (500K/s)
- [x] UltraFastInference (1M/s)
- [x] ExtremeInference (2M+/s)
- [x] ParallelInference (10M+/s)
- [x] ActorInference (5M+/s)

### Quality
- [x] 46 tests (100% pass rate)
- [x] MECE documentation
- [x] API reference complete

---

## v0.0.2 - Stabilization ✅

**Status:** Complete
**Release Date:** 2025-12-28

### Error Handling
- [x] Result type (NMCResult) for operations
- [x] Detailed error messages with error kinds
- [x] Input validation (validateResult, validateLayer)
- [x] Edge case handling

### Improvements
- [x] Validation helpers (validatePositive, validateRange, validateNonEmpty)
- [x] Result-based file I/O (saveNMWResult, loadNMWResult)
- [x] Result-based code generation (generateResult)

### Quality
- [x] 59 tests (100% pass rate)
- [x] 13 new error handling tests

---

## v0.0.3 - Metal API Bindings ✅

**Status:** Complete
**Release Date:** 2025-12-28

### Metal Core
- [x] MTLDevice bindings (metal_device.nim)
- [x] Device capability detection (metal_capabilities.nim)
- [x] Apple Silicon / Intel GPU detection
- [x] Error handling for Metal unavailable

### Basic Types
- [x] MTLBuffer Nim wrapper (metal_buffer.nim)
- [x] MTLCommandQueue wrapper (metal_command.nim)
- [x] MTLCommandBuffer wrapper
- [x] MTLComputeCommandEncoder wrapper

### Memory Management
- [x] Shared memory mode (unified memory)
- [x] Private memory mode
- [x] Managed memory mode
- [x] Buffer allocation/deallocation

### Capabilities Detection
- [x] GPU family detection (Apple1-9, Mac1-2, Common1-3)
- [x] Compute capabilities query
- [x] Memory capabilities query
- [x] Feature support table
- [x] Threadgroup size recommendations

### Quality
- [x] 37 new Metal tests (96 total)
- [x] Platform-conditional tests (macOS)

---

## v0.0.4 - Compute Pipeline

**Status:** Planned

### Pipeline Creation
- [ ] MTLComputePipelineState
- [ ] MTLLibrary (shader library)
- [ ] MTLFunction (shader function)
- [ ] Pipeline caching

### Shader Compilation
- [ ] Runtime MSL compilation
- [ ] Pre-compiled metallib support
- [ ] Compilation error handling
- [ ] Shader function discovery

### Thread Configuration
- [ ] Threadgroup size calculation
- [ ] Grid size management
- [ ] Optimal dispatch configuration

---

## v0.0.5 - Shader Runtime Execution

**Status:** Planned

### Execution
- [ ] Synchronous compute dispatch
- [ ] Buffer binding to shaders
- [ ] Result retrieval
- [ ] Basic matrix operations

### Generated Shader Execution
- [ ] Execute generated MSL from v0.0.1
- [ ] Dense layer GPU execution
- [ ] Activation functions on GPU
- [ ] Forward pass on GPU

### Validation
- [ ] CPU vs GPU result comparison
- [ ] Numerical accuracy verification
- [ ] Performance benchmarks

---

## v0.0.6 - Buffer Optimization

**Status:** Planned

### Memory Efficiency
- [ ] Buffer pooling
- [ ] Buffer reuse strategies
- [ ] Memory alignment optimization
- [ ] Zero-copy where possible

### Data Transfer
- [ ] Efficient CPU→GPU transfer
- [ ] GPU→CPU result retrieval
- [ ] Batch buffer operations
- [ ] Streaming data support

---

## v0.0.7 - Async Execution

**Status:** Planned

### Asynchronous Operations
- [ ] Async command submission
- [ ] Completion handlers
- [ ] Multiple command buffers
- [ ] Pipeline parallelism

### Double Buffering
- [ ] Frame overlap
- [ ] Resource synchronization
- [ ] Fence/Event support

### Integration
- [ ] Nim async/await integration
- [ ] Actor model integration
- [ ] Callback-based API

---

## v0.0.8 - Profiling & Debugging

**Status:** Planned

### Performance Tools
- [ ] GPU timing queries
- [ ] Bandwidth measurement
- [ ] Occupancy analysis
- [ ] Bottleneck detection

### Debug Support
- [ ] Shader debugging info
- [ ] Buffer content inspection
- [ ] Error diagnostics
- [ ] Metal validation layer integration

---

## v0.0.9 - Stabilization & Optimization

**Status:** Planned

### Performance
- [ ] Shader optimization
- [ ] Memory access patterns
- [ ] Cache-friendly layouts
- [ ] Benchmark suite

### Robustness
- [ ] Edge case handling
- [ ] Resource cleanup
- [ ] Error recovery
- [ ] Stress testing

---

## v0.1.0 - Production Ready

**Status:** Planned

### Stability
- [ ] API freeze
- [ ] Backward compatibility guarantee
- [ ] Deprecation policy
- [ ] Semantic versioning

### Production Features
- [ ] Production logging
- [ ] Resource monitoring
- [ ] Graceful degradation (GPU→CPU fallback)
- [ ] Health checks

### Quality
- [ ] Security audit
- [ ] Performance benchmarks
- [ ] Memory leak testing
- [ ] Cross-device testing

### Documentation
- [ ] Complete API docs
- [ ] Migration guides
- [ ] Best practices
- [ ] Performance tuning guide

### Ecosystem
- [ ] Nimble package published
- [ ] Examples repository
- [ ] Community guidelines

---

## Out of Scope (nim-ml別プロジェクト推奨)

以下の機能は nim-metal-compute のスコープ外です:

| Feature | Reason | Recommended Project |
|---------|--------|---------------------|
| MLX統合 | 高レベルMLフレームワーク | nim-ml |
| Training/Backprop | ML専用機能 | nim-ml |
| ONNX import/export | モデル形式 | nim-ml |
| Transformer blocks | 高レベルアーキテクチャ | nim-ml |
| Quantization | ML最適化 | nim-ml |
| Learning rate schedulers | 訓練機能 | nim-ml |

---

## Feature Priority Matrix

| Feature | Version | Impact | Effort |
|---------|---------|--------|--------|
| Error handling | 0.0.2 | Medium | Low |
| Metal bindings | 0.0.3 | High | High |
| Compute pipeline | 0.0.4 | High | Medium |
| Shader execution | 0.0.5 | High | Medium |
| Buffer optimization | 0.0.6 | Medium | Medium |
| Async execution | 0.0.7 | Medium | Medium |
| Profiling | 0.0.8 | Low | Low |

---

## Dependencies

### v0.0.3 Dependencies
- Metal framework (macOS/iOS)
- Nim >= 2.0.0
- Apple Silicon or Intel Mac with Metal support

### System Requirements
- macOS 10.14+ (Mojave) for Metal 2
- macOS 11+ (Big Sur) for Apple Silicon optimization

