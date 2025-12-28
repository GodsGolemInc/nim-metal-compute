# nim-metal-compute Documentation

Metal Compute Shader bindings for Nim - GPU accelerated neural networks.

## Documentation Structure (MECE)

```
docs/
├── api-reference/     # API specifications (What)
│   ├── network-spec.md
│   ├── weights.md
│   ├── codegen.md
│   ├── unified-api.md
│   └── inference-engines.md
├── architecture/      # System design (How it works)
│   ├── overview.md
│   ├── layer-structure.md
│   └── memory-layout.md
├── guides/            # Usage instructions (How to use)
│   ├── getting-started.md
│   ├── custom-networks.md
│   └── deployment.md
└── performance/       # Benchmarks & optimization (How fast)
    ├── benchmarks.md
    └── optimization-guide.md
```

## Quick Links

- [Getting Started](guides/getting-started.md)
- [API Reference](api-reference/)
- [Architecture Overview](architecture/overview.md)
- [Performance Benchmarks](performance/benchmarks.md)

## MECE Categories

| Category | Scope | Contents |
|----------|-------|----------|
| **API Reference** | What functions exist | Types, procs, parameters |
| **Architecture** | How the system works | Design decisions, data flow |
| **Guides** | How to use it | Tutorials, examples |
| **Performance** | How fast it runs | Benchmarks, optimization |
