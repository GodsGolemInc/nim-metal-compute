# nim-metal-compute Specifications

## Version: 0.0.x Series (Development) → 0.1.0 (Production)

This directory contains specifications for nim-metal-compute development.

## Project Scope

**nim-metal-compute** is a low-level Metal compute library for Nim:

| In Scope | Out of Scope (→ nim-ml) |
|----------|-------------------------|
| Metal API bindings | MLX統合 |
| GPU計算シェーダー | Training/Backprop |
| バッファ管理 | ONNX format |
| Compute Pipeline | Quantization |
| CPU推論エンジン | Transformer |

## Documents

| Document | Description |
|----------|-------------|
| [requirements.md](requirements.md) | Functional & non-functional requirements |
| [design.md](design.md) | Technical design & architecture |
| [roadmap.md](roadmap.md) | Version roadmap & milestones |
| [implementation-status.md](implementation-status.md) | Current implementation status |

## Version Scheme

```
0.0.1  CPU推論エンジン (SIMD/並列)      ✅ Current
0.0.2  エラーハンドリング改善
0.0.3  Metal APIバインディング
0.0.4  Compute Pipeline実装
0.0.5  シェーダーランタイム実行
0.0.6  バッファ最適化
0.0.7  非同期実行
0.0.8  プロファイリング
0.0.9  安定化・最適化
0.1.0  Production ready                  🎯 Milestone
```

## Current Status

- **Version:** 0.0.1
- **Focus:** CPU Inference Engines
- **Tests:** 46 (100% pass rate)
- **Documentation:** Complete
- **Next:** v0.0.2 (Stabilization)

## Architecture

```
v0.0.1 (Current)              v0.0.5+ (Target)
┌─────────────────┐           ┌─────────────────┐
│   UnifiedAPI    │           │   UnifiedAPI    │
└────────┬────────┘           └────────┬────────┘
         │                             │
┌────────▼────────┐           ┌────────▼────────┐
│  CPU Engines    │           │ Backend Layer   │
│ SIMD/Parallel   │           │ CPU │ Metal     │
└─────────────────┘           └─────────────────┘
```

