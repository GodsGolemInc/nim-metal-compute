## Metal Matrix Operations
## v0.0.6: GPU-accelerated matrix operations
##
## This module provides:
## - Matrix multiplication (GEMM)
## - Element-wise operations
## - Transpose operations

import std/[strformat, times]
import ./errors
import ./metal_device
import ./metal_buffer
import ./metal_command
import ./metal_shader
import ./metal_wrapper

# ========== Matrix Types ==========

type
  Matrix*[T] = object
    ## Row-major matrix
    data*: seq[T]
    rows*: int
    cols*: int

# ========== Matrix Shaders ==========

const MatrixMultiplyShaderF32* = """
#include <metal_stdlib>
using namespace metal;

kernel void matrix_multiply_f32(device const float* A [[buffer(0)]],
                                device const float* B [[buffer(1)]],
                                device float* C [[buffer(2)]],
                                constant uint& M [[buffer(3)]],
                                constant uint& N [[buffer(4)]],
                                constant uint& K [[buffer(5)]],
                                uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
"""

const MatrixTransposeShaderF32* = """
#include <metal_stdlib>
using namespace metal;

kernel void matrix_transpose_f32(device const float* input [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& rows [[buffer(2)]],
                                 constant uint& cols [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= rows || col >= cols) return;

    output[col * rows + row] = input[row * cols + col];
}
"""

const MatrixAddShaderF32* = """
#include <metal_stdlib>
using namespace metal;

kernel void matrix_add_f32(device const float* A [[buffer(0)]],
                           device const float* B [[buffer(1)]],
                           device float* C [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    C[id] = A[id] + B[id];
}
"""

const MatrixScaleShaderF32* = """
#include <metal_stdlib>
using namespace metal;

kernel void matrix_scale_f32(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant float& scalar [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    output[id] = input[id] * scalar;
}
"""

# ========== Matrix Construction ==========

proc newMatrix*[T](rows, cols: int): Matrix[T] =
  ## Create a new zero-initialized matrix
  result.rows = rows
  result.cols = cols
  result.data = newSeq[T](rows * cols)

proc newMatrix*[T](rows, cols: int, data: seq[T]): Matrix[T] =
  ## Create a matrix from existing data
  assert data.len == rows * cols, "Data length must match rows * cols"
  result.rows = rows
  result.cols = cols
  result.data = data

proc `[]`*[T](m: Matrix[T], row, col: int): T =
  ## Access element at (row, col)
  m.data[row * m.cols + col]

proc `[]=`*[T](m: var Matrix[T], row, col: int, value: T) =
  ## Set element at (row, col)
  m.data[row * m.cols + col] = value

proc `$`*[T](m: Matrix[T]): string =
  result = fmt"Matrix[{m.rows}x{m.cols}]"

# ========== CPU Reference Implementations ==========

proc matmulCPU*(a, b: Matrix[float32]): Matrix[float32] =
  ## CPU matrix multiplication for verification
  assert a.cols == b.rows, "Matrix dimensions must match for multiplication"

  result = newMatrix[float32](a.rows, b.cols)

  for i in 0..<a.rows:
    for j in 0..<b.cols:
      var sum: float32 = 0.0
      for k in 0..<a.cols:
        sum += a[i, k] * b[k, j]
      result[i, j] = sum

proc transposeCPU*(m: Matrix[float32]): Matrix[float32] =
  ## CPU matrix transpose
  result = newMatrix[float32](m.cols, m.rows)
  for i in 0..<m.rows:
    for j in 0..<m.cols:
      result[j, i] = m[i, j]

# ========== GPU Matrix Operations ==========

proc matmulGPU*(device: MetalDevice, a, b: Matrix[float32]): NMCResult[Matrix[float32]] =
  ## GPU-accelerated matrix multiplication: C = A * B
  when defined(macosx):
    if a.cols != b.rows:
      return err[Matrix[float32]](ekDimensionMismatch,
        fmt"Matrix dimensions don't match: {a.rows}x{a.cols} * {b.rows}x{b.cols}")

    let M = a.rows.uint32
    let N = b.cols.uint32
    let K = a.cols.uint32

    # Compile shader
    let pipelineResult = device.compileAndCreatePipeline(MatrixMultiplyShaderF32, "matrix_multiply_f32")
    if not pipelineResult.isOk:
      return err[Matrix[float32]](pipelineResult.error)
    var pipeline = pipelineResult.get

    # Create buffers
    let bufA = device.newBuffer(a.data)
    if not bufA.isOk:
      pipeline.release()
      return err[Matrix[float32]](bufA.error)
    var bA = bufA.get

    let bufB = device.newBuffer(b.data)
    if not bufB.isOk:
      bA.release()
      pipeline.release()
      return err[Matrix[float32]](bufB.error)
    var bB = bufB.get

    let resultSize = M.int * N.int * sizeof(float32)
    let bufC = device.newBuffer(resultSize, smShared)
    if not bufC.isOk:
      bB.release()
      bA.release()
      pipeline.release()
      return err[Matrix[float32]](bufC.error)
    var bC = bufC.get

    # Create command infrastructure
    let queueResult = device.newCommandQueue()
    if not queueResult.isOk:
      bC.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[Matrix[float32]](queueResult.error)
    var queue = queueResult.get

    let cmdBufferResult = queue.newCommandBuffer()
    if not cmdBufferResult.isOk:
      queue.release()
      bC.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[Matrix[float32]](cmdBufferResult.error)
    var cmdBuffer = cmdBufferResult.get

    let encoderResult = cmdBuffer.newComputeEncoder()
    if not encoderResult.isOk:
      cmdBuffer.release()
      queue.release()
      bC.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[Matrix[float32]](encoderResult.error)
    var encoder = encoderResult.get

    # Set pipeline and buffers
    nmc_encoder_set_pipeline_state(encoder.handle.pointer, pipeline.handle.pointer)
    nmc_encoder_set_buffer(encoder.handle.pointer, bA.handle.pointer, 0, 0)
    nmc_encoder_set_buffer(encoder.handle.pointer, bB.handle.pointer, 0, 1)
    nmc_encoder_set_buffer(encoder.handle.pointer, bC.handle.pointer, 0, 2)

    # Set dimension constants
    var mVal = M
    var nVal = N
    var kVal = K
    nmc_encoder_set_bytes(encoder.handle.pointer, addr mVal, sizeof(uint32).uint64, 3)
    nmc_encoder_set_bytes(encoder.handle.pointer, addr nVal, sizeof(uint32).uint64, 4)
    nmc_encoder_set_bytes(encoder.handle.pointer, addr kVal, sizeof(uint32).uint64, 5)

    # Calculate thread configuration
    # Use 2D grid for matrix operations
    let threadGroupWidth = min(16, N.int)
    let threadGroupHeight = min(16, M.int)
    let gridWidth = (N.int + threadGroupWidth - 1) div threadGroupWidth
    let gridHeight = (M.int + threadGroupHeight - 1) div threadGroupHeight

    nmc_encoder_dispatch_threadgroups(
      encoder.handle.pointer,
      gridWidth.uint64, gridHeight.uint64, 1,
      threadGroupWidth.uint64, threadGroupHeight.uint64, 1
    )

    # Execute
    discard encoder.endEncoding()
    discard cmdBuffer.commit()
    discard cmdBuffer.waitUntilCompleted()

    # Read result
    var resultData = newSeq[float32](M.int * N.int)
    discard bC.read(resultData)

    # Cleanup
    cmdBuffer.release()
    queue.release()
    bC.release()
    bB.release()
    bA.release()
    pipeline.release()

    result = ok(newMatrix[float32](M.int, N.int, resultData))
  else:
    result = err[Matrix[float32]](ekPlatform, "Metal not available")

proc transposeGPU*(device: MetalDevice, m: Matrix[float32]): NMCResult[Matrix[float32]] =
  ## GPU-accelerated matrix transpose
  when defined(macosx):
    let rows = m.rows.uint32
    let cols = m.cols.uint32

    # Compile shader
    let pipelineResult = device.compileAndCreatePipeline(MatrixTransposeShaderF32, "matrix_transpose_f32")
    if not pipelineResult.isOk:
      return err[Matrix[float32]](pipelineResult.error)
    var pipeline = pipelineResult.get

    # Create buffers
    let bufIn = device.newBuffer(m.data)
    if not bufIn.isOk:
      pipeline.release()
      return err[Matrix[float32]](bufIn.error)
    var bIn = bufIn.get

    let resultSize = m.rows * m.cols * sizeof(float32)
    let bufOut = device.newBuffer(resultSize, smShared)
    if not bufOut.isOk:
      bIn.release()
      pipeline.release()
      return err[Matrix[float32]](bufOut.error)
    var bOut = bufOut.get

    # Create command infrastructure
    let queueResult = device.newCommandQueue()
    if not queueResult.isOk:
      bOut.release()
      bIn.release()
      pipeline.release()
      return err[Matrix[float32]](queueResult.error)
    var queue = queueResult.get

    let cmdBufferResult = queue.newCommandBuffer()
    if not cmdBufferResult.isOk:
      queue.release()
      bOut.release()
      bIn.release()
      pipeline.release()
      return err[Matrix[float32]](cmdBufferResult.error)
    var cmdBuffer = cmdBufferResult.get

    let encoderResult = cmdBuffer.newComputeEncoder()
    if not encoderResult.isOk:
      cmdBuffer.release()
      queue.release()
      bOut.release()
      bIn.release()
      pipeline.release()
      return err[Matrix[float32]](encoderResult.error)
    var encoder = encoderResult.get

    # Set pipeline and buffers
    nmc_encoder_set_pipeline_state(encoder.handle.pointer, pipeline.handle.pointer)
    nmc_encoder_set_buffer(encoder.handle.pointer, bIn.handle.pointer, 0, 0)
    nmc_encoder_set_buffer(encoder.handle.pointer, bOut.handle.pointer, 0, 1)

    var rowsVal = rows
    var colsVal = cols
    nmc_encoder_set_bytes(encoder.handle.pointer, addr rowsVal, sizeof(uint32).uint64, 2)
    nmc_encoder_set_bytes(encoder.handle.pointer, addr colsVal, sizeof(uint32).uint64, 3)

    # Thread configuration
    let threadGroupWidth = min(16, cols.int)
    let threadGroupHeight = min(16, rows.int)
    let gridWidth = (cols.int + threadGroupWidth - 1) div threadGroupWidth
    let gridHeight = (rows.int + threadGroupHeight - 1) div threadGroupHeight

    nmc_encoder_dispatch_threadgroups(
      encoder.handle.pointer,
      gridWidth.uint64, gridHeight.uint64, 1,
      threadGroupWidth.uint64, threadGroupHeight.uint64, 1
    )

    # Execute
    discard encoder.endEncoding()
    discard cmdBuffer.commit()
    discard cmdBuffer.waitUntilCompleted()

    # Read result
    var resultData = newSeq[float32](m.rows * m.cols)
    discard bOut.read(resultData)

    # Cleanup
    cmdBuffer.release()
    queue.release()
    bOut.release()
    bIn.release()
    pipeline.release()

    result = ok(newMatrix[float32](m.cols, m.rows, resultData))
  else:
    result = err[Matrix[float32]](ekPlatform, "Metal not available")

# ========== Verification ==========

proc verify*(expected, actual: Matrix[float32], tolerance: float32 = 1e-4): bool =
  ## Verify matrix equality within tolerance
  if expected.rows != actual.rows or expected.cols != actual.cols:
    return false

  for i in 0..<expected.data.len:
    if abs(expected.data[i] - actual.data[i]) > tolerance:
      return false

  result = true

# ========== Test ==========

when isMainModule:
  echo "=== Metal Matrix Operations Test ==="
  echo ""

  let deviceResult = getDefaultDevice()
  if not deviceResult.isOk:
    echo "Error: ", deviceResult.error
    quit(1)

  let device = deviceResult.get
  echo "Device: ", device.info.name
  echo ""

  # Test matrix multiplication
  echo "--- Matrix Multiplication Test ---"

  let sizes = @[(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)]

  for (m, n, k) in sizes:
    echo fmt"Matrix: {m}x{k} * {k}x{n}"

    # Create test matrices
    var a = newMatrix[float32](m, k)
    var b = newMatrix[float32](k, n)
    for i in 0..<a.data.len:
      a.data[i] = float32(i mod 10) / 10.0
    for i in 0..<b.data.len:
      b.data[i] = float32(i mod 10) / 10.0

    # GPU computation
    let gpuStart = cpuTime()
    let gpuResult = device.matmulGPU(a, b)
    let gpuTime = cpuTime() - gpuStart

    if gpuResult.isOk:
      let gpuMatrix = gpuResult.get

      # CPU computation for verification
      let cpuStart = cpuTime()
      let cpuMatrix = matmulCPU(a, b)
      let cpuTime = cpuTime() - cpuStart

      # Verify
      let correct = verify(cpuMatrix, gpuMatrix)

      echo fmt"  GPU time: {gpuTime * 1000:.2f} ms"
      echo fmt"  CPU time: {cpuTime * 1000:.2f} ms"
      echo fmt"  Speedup:  {cpuTime / gpuTime:.2f}x"
      echo fmt"  Correct:  {correct}"
    else:
      echo "  GPU Error: ", gpuResult.error

    echo ""

  # Test transpose
  echo "--- Matrix Transpose Test ---"
  let rows = 256
  let cols = 512

  var m = newMatrix[float32](rows, cols)
  for i in 0..<m.data.len:
    m.data[i] = float32(i)

  let gpuTransStart = cpuTime()
  let gpuTransResult = device.transposeGPU(m)
  let gpuTransTime = cpuTime() - gpuTransStart

  if gpuTransResult.isOk:
    let gpuTrans = gpuTransResult.get

    let cpuTransStart = cpuTime()
    let cpuTrans = transposeCPU(m)
    let cpuTransTime = cpuTime() - cpuTransStart

    let transCorrect = verify(cpuTrans, gpuTrans)

    echo fmt"Matrix: {rows}x{cols} -> {cols}x{rows}"
    echo fmt"  GPU time: {gpuTransTime * 1000:.2f} ms"
    echo fmt"  CPU time: {cpuTransTime * 1000:.2f} ms"
    echo fmt"  Correct:  {transCorrect}"
  else:
    echo "Transpose error: ", gpuTransResult.error

  echo ""
  echo "✅ Metal matrix operations test complete"
