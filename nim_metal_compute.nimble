# Package

version = "0.1.0"
author = "GodsGolemInc"
description = "Metal Compute Shader bindings for Nim - GPU accelerated neural networks"
license = "Apache-2.0"
srcDir = "src"

# Dependencies

requires "nim >= 2.0.0"

# Tasks

task test, "Run tests":
  exec "nim c -r tests/test_metal.nim"

task bench, "Run benchmarks":
  exec "nim c -d:release -r tests/benchmark.nim"

task gencode, "Generate Metal/CPU code from spec":
  exec "nim c -r src/codegen/generate.nim"
