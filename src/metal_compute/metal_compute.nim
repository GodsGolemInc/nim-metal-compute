## Metal Compute - Full GPU Computation Example
## v0.0.5: Complete GPU compute workflow
##
## This module demonstrates:
## - Shader compilation
## - Buffer creation
## - Command queue/buffer/encoder
## - GPU execution
## - Result retrieval

import std/[strformat, math]
import ./errors
import ./metal_device
import ./metal_buffer
import ./metal_command
import ./metal_shader
import ./metal_wrapper

# ========== High-level Compute Functions ==========

proc vectorAdd*(device: MetalDevice, a: seq[float32], b: seq[float32]): NMCResult[seq[float32]] =
  ## Perform vector addition on GPU: result = a + b
  when defined(macosx):
    if a.len != b.len:
      return err[seq[float32]](ekDimensionMismatch,
        fmt"Vector lengths must match: {a.len} vs {b.len}")

    let count = a.len
    if count == 0:
      return ok(newSeq[float32]())

    # Compile shader
    let pipelineResult = device.compileAndCreatePipeline(VectorAddShader, "vector_add")
    if not pipelineResult.isOk:
      return err[seq[float32]](pipelineResult.error)
    var pipeline = pipelineResult.get

    # Create buffers
    let bufferA = device.newBuffer(a)
    if not bufferA.isOk:
      pipeline.release()
      return err[seq[float32]](bufferA.error)
    var bA = bufferA.get

    let bufferB = device.newBuffer(b)
    if not bufferB.isOk:
      bA.release()
      pipeline.release()
      return err[seq[float32]](bufferB.error)
    var bB = bufferB.get

    let bufferResult = device.newBuffer(count * sizeof(float32), smShared)
    if not bufferResult.isOk:
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](bufferResult.error)
    var bResult = bufferResult.get

    # Create command queue
    let queueResult = device.newCommandQueue()
    if not queueResult.isOk:
      bResult.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](queueResult.error)
    var queue = queueResult.get

    # Create command buffer
    let cmdBufferResult = queue.newCommandBuffer()
    if not cmdBufferResult.isOk:
      queue.release()
      bResult.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](cmdBufferResult.error)
    var cmdBuffer = cmdBufferResult.get

    # Create compute encoder
    let encoderResult = cmdBuffer.newComputeEncoder()
    if not encoderResult.isOk:
      cmdBuffer.release()
      queue.release()
      bResult.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](encoderResult.error)
    var encoder = encoderResult.get

    # Set pipeline and buffers
    nmc_encoder_set_pipeline_state(encoder.handle.pointer, pipeline.handle.pointer)
    nmc_encoder_set_buffer(encoder.handle.pointer, bA.handle.pointer, 0, 0)
    nmc_encoder_set_buffer(encoder.handle.pointer, bB.handle.pointer, 0, 1)
    nmc_encoder_set_buffer(encoder.handle.pointer, bResult.handle.pointer, 0, 2)

    # Calculate thread configuration
    let threadGroupSize = min(pipeline.maxThreadsPerThreadgroup.int, count)
    let threadGroups = (count + threadGroupSize - 1) div threadGroupSize

    # Dispatch compute
    nmc_encoder_dispatch_threadgroups(
      encoder.handle.pointer,
      threadGroups.uint64, 1, 1,
      threadGroupSize.uint64, 1, 1
    )

    # End encoding
    let endResult = encoder.endEncoding()
    if not endResult.isOk:
      encoder.release()
      cmdBuffer.release()
      queue.release()
      bResult.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](endResult.error)

    # Commit and wait
    let commitResult = cmdBuffer.commit()
    if not commitResult.isOk:
      cmdBuffer.release()
      queue.release()
      bResult.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](commitResult.error)

    let waitResult = cmdBuffer.waitUntilCompleted()
    if not waitResult.isOk:
      cmdBuffer.release()
      queue.release()
      bResult.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](waitResult.error)

    # Read result
    var resultData = newSeq[float32](count)
    let readResult = bResult.read(resultData)
    if not readResult.isOk:
      cmdBuffer.release()
      queue.release()
      bResult.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](readResult.error)

    # Cleanup
    cmdBuffer.release()
    queue.release()
    bResult.release()
    bB.release()
    bA.release()
    pipeline.release()

    result = ok(resultData)
  else:
    result = err[seq[float32]](ekPlatform, "Metal not available on this platform")

proc vectorMultiply*(device: MetalDevice, a: seq[float32], b: seq[float32]): NMCResult[seq[float32]] =
  ## Perform element-wise vector multiplication on GPU: result = a * b
  when defined(macosx):
    if a.len != b.len:
      return err[seq[float32]](ekDimensionMismatch,
        fmt"Vector lengths must match: {a.len} vs {b.len}")

    let count = a.len
    if count == 0:
      return ok(newSeq[float32]())

    # Compile shader
    let pipelineResult = device.compileAndCreatePipeline(VectorMultiplyShader, "vector_multiply")
    if not pipelineResult.isOk:
      return err[seq[float32]](pipelineResult.error)
    var pipeline = pipelineResult.get

    # Create buffers
    let bufferA = device.newBuffer(a)
    if not bufferA.isOk:
      pipeline.release()
      return err[seq[float32]](bufferA.error)
    var bA = bufferA.get

    let bufferB = device.newBuffer(b)
    if not bufferB.isOk:
      bA.release()
      pipeline.release()
      return err[seq[float32]](bufferB.error)
    var bB = bufferB.get

    let bufferResult = device.newBuffer(count * sizeof(float32), smShared)
    if not bufferResult.isOk:
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](bufferResult.error)
    var bResult = bufferResult.get

    # Create command infrastructure
    let queueResult = device.newCommandQueue()
    if not queueResult.isOk:
      bResult.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](queueResult.error)
    var queue = queueResult.get

    let cmdBufferResult = queue.newCommandBuffer()
    if not cmdBufferResult.isOk:
      queue.release()
      bResult.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](cmdBufferResult.error)
    var cmdBuffer = cmdBufferResult.get

    let encoderResult = cmdBuffer.newComputeEncoder()
    if not encoderResult.isOk:
      cmdBuffer.release()
      queue.release()
      bResult.release()
      bB.release()
      bA.release()
      pipeline.release()
      return err[seq[float32]](encoderResult.error)
    var encoder = encoderResult.get

    # Set pipeline and buffers
    nmc_encoder_set_pipeline_state(encoder.handle.pointer, pipeline.handle.pointer)
    nmc_encoder_set_buffer(encoder.handle.pointer, bA.handle.pointer, 0, 0)
    nmc_encoder_set_buffer(encoder.handle.pointer, bB.handle.pointer, 0, 1)
    nmc_encoder_set_buffer(encoder.handle.pointer, bResult.handle.pointer, 0, 2)

    # Calculate thread configuration
    let threadGroupSize = min(pipeline.maxThreadsPerThreadgroup.int, count)
    let threadGroups = (count + threadGroupSize - 1) div threadGroupSize

    # Dispatch
    nmc_encoder_dispatch_threadgroups(
      encoder.handle.pointer,
      threadGroups.uint64, 1, 1,
      threadGroupSize.uint64, 1, 1
    )

    discard encoder.endEncoding()
    discard cmdBuffer.commit()
    discard cmdBuffer.waitUntilCompleted()

    # Read result
    var resultData = newSeq[float32](count)
    discard bResult.read(resultData)

    # Cleanup
    cmdBuffer.release()
    queue.release()
    bResult.release()
    bB.release()
    bA.release()
    pipeline.release()

    result = ok(resultData)
  else:
    result = err[seq[float32]](ekPlatform, "Metal not available on this platform")

# ========== CPU Reference Implementations ==========

proc vectorAddCPU*(a, b: seq[float32]): seq[float32] =
  ## CPU reference implementation
  result = newSeq[float32](a.len)
  for i in 0..<a.len:
    result[i] = a[i] + b[i]

proc vectorMultiplyCPU*(a, b: seq[float32]): seq[float32] =
  ## CPU reference implementation
  result = newSeq[float32](a.len)
  for i in 0..<a.len:
    result[i] = a[i] * b[i]

proc verifyResults*(expected, actual: seq[float32], tolerance: float32 = 1e-5): bool =
  ## Verify GPU results against expected values
  if expected.len != actual.len:
    return false
  for i in 0..<expected.len:
    if abs(expected[i] - actual[i]) > tolerance:
      return false
  result = true

# ========== Test ==========

when isMainModule:
  import std/times

  echo "=== Metal GPU Compute Test ==="
  echo ""

  let deviceResult = getDefaultDevice()
  if not deviceResult.isOk:
    echo "Error: ", deviceResult.error
    quit(1)

  let device = deviceResult.get
  echo "Device: ", device.info.name
  echo ""

  # Test vector sizes
  let sizes = @[1000, 10000, 100000, 1000000]

  for size in sizes:
    echo fmt"--- Vector size: {size} ---"

    # Create test data
    var a = newSeq[float32](size)
    var b = newSeq[float32](size)
    for i in 0..<size:
      a[i] = float32(i)
      b[i] = float32(i * 2)

    # GPU computation
    let gpuStart = cpuTime()
    let gpuResult = device.vectorAdd(a, b)
    let gpuTime = cpuTime() - gpuStart

    if gpuResult.isOk:
      let gpuData = gpuResult.get

      # CPU computation for verification
      let cpuStart = cpuTime()
      let cpuData = vectorAddCPU(a, b)
      let cpuTime = cpuTime() - cpuStart

      # Verify results
      let correct = verifyResults(cpuData, gpuData)

      echo fmt"  GPU time: {gpuTime * 1000:.2f} ms"
      echo fmt"  CPU time: {cpuTime * 1000:.2f} ms"
      echo fmt"  Speedup:  {cpuTime / gpuTime:.2f}x"
      echo fmt"  Correct:  {correct}"

      # Show first few results
      if size <= 10:
        echo fmt"  First results: {gpuData[0..min(4, size-1)]}"
    else:
      echo "  GPU Error: ", gpuResult.error

    echo ""

  echo "✅ Metal GPU compute test complete"
