# Package

version = "0.0.10"
author = "GodsGolemInc"
description = "Metal Compute Shader bindings for Nim - GPU accelerated neural networks with CPU fallback"
license = "Apache-2.0"
srcDir = "src"

# Dependencies

requires "nim >= 2.0.0"

# Tasks

task test, "Run all tests":
  exec "nim c -r tests/test_all.nim"

task test_metal, "Run Metal device test":
  exec "nim c -r src/nim_metal_compute/metal_device.nim"

task test_buffer, "Run buffer test":
  exec "nim c -r src/nim_metal_compute/metal_buffer.nim"

task test_command, "Run command test":
  exec "nim c -r src/nim_metal_compute/metal_command.nim"

task test_compute, "Run compute test":
  exec "nim c -r src/nim_metal_compute/metal_compute.nim"

task test_matrix, "Run matrix test":
  exec "nim c -r src/nim_metal_compute/metal_matrix.nim"

task test_nn, "Run neural network test":
  exec "nim c -r src/nim_metal_compute/metal_nn.nim"

task test_async, "Run async test":
  exec "nim c -r src/nim_metal_compute/metal_async.nim"

task test_api, "Run unified API test":
  exec "nim c -r src/nim_metal_compute/metal_api.nim"

task stress, "Run stress tests":
  exec "nim c -r src/nim_metal_compute/metal_stress.nim"

task bench, "Run benchmarks":
  exec "nim c -d:release -r src/nim_metal_compute/metal_matrix.nim"

task optimize, "Run optimization tests":
  exec "nim c -r src/nim_metal_compute/metal_optimize.nim"

task all_tests, "Run all module tests":
  exec "nim c -r src/nim_metal_compute/metal_device.nim"
  exec "nim c -r src/nim_metal_compute/metal_buffer.nim"
  exec "nim c -r src/nim_metal_compute/metal_command.nim"
  exec "nim c -r src/nim_metal_compute/metal_compute.nim"
  exec "nim c -r src/nim_metal_compute/metal_matrix.nim"
  exec "nim c -r src/nim_metal_compute/metal_nn.nim"
  exec "nim c -r src/nim_metal_compute/metal_async.nim"
  exec "nim c -r src/nim_metal_compute/metal_stress.nim"
  exec "nim c -r src/nim_metal_compute/metal_optimize.nim"
  exec "nim c -r src/nim_metal_compute/metal_api.nim"
