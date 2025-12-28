# Requirements Specification

Version: 0.0.x Series → 0.1.0 (Production)

## 1. Functional Requirements

### FR-1: Network Definition

| ID | Requirement | Priority | Version |
|----|-------------|----------|---------|
| FR-1.1 | Define MLP networks via DSL | Must | 0.0.1 ✅ |
| FR-1.2 | Support Dense layers | Must | 0.0.1 ✅ |
| FR-1.3 | Network validation | Must | 0.0.1 ✅ |
| FR-1.4 | JSON serialization | Should | 0.0.2 |

### FR-2: Activation Functions

| ID | Requirement | Priority | Version |
|----|-------------|----------|---------|
| FR-2.1 | ReLU activation | Must | 0.0.1 ✅ |
| FR-2.2 | Softmax activation | Must | 0.0.1 ✅ |
| FR-2.3 | Sigmoid activation | Must | 0.0.1 ✅ |
| FR-2.4 | Tanh activation | Must | 0.0.1 ✅ |
| FR-2.5 | LeakyReLU activation | Could | 0.0.5 |

### FR-3: Weight Management

| ID | Requirement | Priority | Version |
|----|-------------|----------|---------|
| FR-3.1 | Xavier/Glorot initialization | Must | 0.0.1 ✅ |
| FR-3.2 | Kaiming/He initialization | Must | 0.0.1 ✅ |
| FR-3.3 | Binary save/load (NMW) | Must | 0.0.1 ✅ |
| FR-3.4 | Flat array conversion | Must | 0.0.1 ✅ |

### FR-4: CPU Inference

| ID | Requirement | Priority | Version |
|----|-------------|----------|---------|
| FR-4.1 | CPU inference (Nim) | Must | 0.0.1 ✅ |
| FR-4.2 | Batch inference | Must | 0.0.1 ✅ |
| FR-4.3 | Multi-threaded inference | Must | 0.0.1 ✅ |
| FR-4.4 | SIMD optimization | Must | 0.0.1 ✅ |
| FR-4.5 | Actor-based inference | Should | 0.0.1 ✅ |

### FR-5: Code Generation

| ID | Requirement | Priority | Version |
|----|-------------|----------|---------|
| FR-5.1 | Metal shader generation (MSL) | Must | 0.0.1 ✅ |
| FR-5.2 | Nim CPU code generation | Must | 0.0.1 ✅ |
| FR-5.3 | File output | Must | 0.0.1 ✅ |

### FR-6: Metal API

| ID | Requirement | Priority | Version |
|----|-------------|----------|---------|
| FR-6.1 | MTLDevice bindings | Must | 0.0.3 |
| FR-6.2 | MTLBuffer management | Must | 0.0.3 |
| FR-6.3 | MTLCommandQueue bindings | Must | 0.0.3 |
| FR-6.4 | Device capability detection | Should | 0.0.3 |
| FR-6.5 | Unified memory support | Should | 0.0.3 |

### FR-7: Compute Pipeline

| ID | Requirement | Priority | Version |
|----|-------------|----------|---------|
| FR-7.1 | MTLComputePipelineState | Must | 0.0.4 |
| FR-7.2 | Runtime shader compilation | Must | 0.0.4 |
| FR-7.3 | Pre-compiled metallib support | Should | 0.0.4 |
| FR-7.4 | Thread configuration | Must | 0.0.4 |

### FR-8: GPU Execution

| ID | Requirement | Priority | Version |
|----|-------------|----------|---------|
| FR-8.1 | Compute dispatch | Must | 0.0.5 |
| FR-8.2 | Buffer binding | Must | 0.0.5 |
| FR-8.3 | Result retrieval | Must | 0.0.5 |
| FR-8.4 | Generated shader execution | Must | 0.0.5 |

### FR-9: Async Operations

| ID | Requirement | Priority | Version |
|----|-------------|----------|---------|
| FR-9.1 | Async command submission | Should | 0.0.7 |
| FR-9.2 | Completion handlers | Should | 0.0.7 |
| FR-9.3 | Double buffering | Could | 0.0.7 |

---

## 2. Non-Functional Requirements

### NFR-1: Performance

| ID | Requirement | Target | Current |
|----|-------------|--------|---------|
| NFR-1.1 | Single inference latency (CPU) | < 1μs | ✅ 0.4μs |
| NFR-1.2 | Batch throughput (CPU) | > 10M/s | ✅ 20M/s |
| NFR-1.3 | Memory per engine | < 100KB | ✅ 40KB |
| NFR-1.4 | Startup time | < 10ms | ✅ <1ms |
| NFR-1.5 | GPU throughput | > 100M/s | 0.0.5 |
| NFR-1.6 | GPU latency | < 100μs | 0.0.5 |

### NFR-2: Compatibility

| ID | Requirement | Target | Current |
|----|-------------|--------|---------|
| NFR-2.1 | Nim version | >= 2.0.0 | ✅ |
| NFR-2.2 | macOS (Apple Silicon) | Full GPU | 0.0.5 |
| NFR-2.3 | macOS (Intel) | Metal GPU | 0.0.5 |
| NFR-2.4 | macOS (CPU fallback) | Full | ✅ |
| NFR-2.5 | Linux (x86_64) | CPU only | ✅ |
| NFR-2.6 | Windows | CPU only | Untested |

### NFR-3: Quality

| ID | Requirement | Target | Current |
|----|-------------|--------|---------|
| NFR-3.1 | Test coverage | 100% | ✅ 100% |
| NFR-3.2 | Test pass rate | 100% | ✅ 100% |
| NFR-3.3 | Documentation | Complete | ✅ |
| NFR-3.4 | API stability | Stable | 0.1.0 |

### NFR-4: Usability

| ID | Requirement | Target | Current |
|----|-------------|--------|---------|
| NFR-4.1 | Single import usage | Yes | ✅ |
| NFR-4.2 | Fluent DSL API | Yes | ✅ |
| NFR-4.3 | Type safety | Full | ✅ |
| NFR-4.4 | Error messages | Clear | 0.0.2 |
| NFR-4.5 | GPU fallback to CPU | Automatic | 0.1.0 |

---

## 3. Version Mapping

| Version | Primary Focus |
|---------|---------------|
| 0.0.1 | CPU推論エンジン ✅ |
| 0.0.2 | エラーハンドリング |
| 0.0.3 | Metal APIバインディング |
| 0.0.4 | Compute Pipeline |
| 0.0.5 | シェーダー実行 |
| 0.0.6 | バッファ最適化 |
| 0.0.7 | 非同期実行 |
| 0.0.8 | プロファイリング |
| 0.0.9 | 安定化・最適化 |
| 0.1.0 | Production |

---

## 4. Out of Scope

以下の機能はnim-metal-computeのスコープ外です（nim-ml推奨）:

| Feature | Reason |
|---------|--------|
| MLX統合 | 高レベルMLフレームワーク |
| Training/Backpropagation | ML専用機能 |
| ONNX import/export | モデル形式 |
| Quantization (INT8/FP16) | ML最適化 |
| Transformer blocks | 高レベルアーキテクチャ |
| Conv2D/Pooling layers | 高レベルML層 |
| Learning rate schedulers | 訓練機能 |

