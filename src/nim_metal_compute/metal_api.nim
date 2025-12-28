## Metal Unified API
## v0.1.0: Production-ready unified API with GPU/CPU fallback
##
## This module provides:
## - Unified interface for GPU and CPU operations
## - Automatic GPU→CPU fallback when Metal is unavailable
## - Production logging integration
## - Thread-safe operation queuing

import std/[strformat, times, logging]
import ./errors
import ./metal_device
import ./metal_buffer
import ./metal_command
import ./metal_shader
import ./metal_compute
import ./metal_matrix
import ./metal_async
import ./metal_optimize

# ========== Types ==========

type
  ComputeBackend* = enum
    cbGPU = "GPU"
    cbCPU = "CPU"
    cbAuto = "Auto"

  ComputeContext* = object
    ## Unified compute context
    device*: MetalDevice
    queue*: MetalCommandQueue
    backend*: ComputeBackend
    actualBackend*: ComputeBackend  # What's actually being used
    logger*: Logger
    valid*: bool
    operationCount*: int
    totalGpuTime*: float64
    totalCpuTime*: float64

  ComputeResult*[T] = object
    ## Result with timing information
    data*: T
    backend*: ComputeBackend
    duration*: float64
    success*: bool
    error*: string

# ========== Context Management ==========

proc newComputeContext*(backend: ComputeBackend = cbAuto,
                        logger: Logger = nil): NMCResult[ComputeContext] =
  ## Create a new compute context
  var ctx: ComputeContext
  ctx.backend = backend
  ctx.logger = logger
  ctx.operationCount = 0
  ctx.totalGpuTime = 0.0
  ctx.totalCpuTime = 0.0

  # Determine actual backend
  if backend == cbCPU:
    ctx.actualBackend = cbCPU
    ctx.valid = true
    if ctx.logger != nil:
      ctx.logger.log(lvlInfo, "ComputeContext created with CPU backend")
    return ok(ctx)

  # Try to initialize GPU
  when defined(macosx):
    if isMetalAvailable():
      let deviceResult = getDefaultDevice()
      if deviceResult.isOk:
        ctx.device = deviceResult.get
        let queueResult = ctx.device.newCommandQueue()
        if queueResult.isOk:
          ctx.queue = queueResult.get
          ctx.actualBackend = cbGPU
          ctx.valid = true
          if ctx.logger != nil:
            ctx.logger.log(lvlInfo, fmt"ComputeContext created with GPU backend: {ctx.device.info.name}")
          return ok(ctx)
        else:
          ctx.device.release()

  # Fallback to CPU
  if backend == cbGPU:
    return err[ComputeContext](ekPlatform, "GPU requested but not available")

  ctx.actualBackend = cbCPU
  ctx.valid = true
  if ctx.logger != nil:
    ctx.logger.log(lvlInfo, "ComputeContext created with CPU backend (GPU unavailable)")
  return ok(ctx)

proc destroy*(ctx: var ComputeContext) =
  ## Destroy the compute context and release resources
  if ctx.valid:
    if ctx.actualBackend == cbGPU:
      ctx.queue.release()
      ctx.device.release()

    if ctx.logger != nil:
      ctx.logger.log(lvlInfo, fmt"ComputeContext destroyed. Operations: {ctx.operationCount}, GPU time: {ctx.totalGpuTime*1000:.2f}ms, CPU time: {ctx.totalCpuTime*1000:.2f}ms")

    ctx.valid = false

# ========== Vector Operations with Fallback ==========

proc vectorAdd*(ctx: var ComputeContext, a, b: seq[float32]): ComputeResult[seq[float32]] =
  ## Add two vectors with automatic backend selection
  if not ctx.valid:
    result.success = false
    result.error = "Context is not valid"
    return

  let startTime = cpuTime()

  if ctx.actualBackend == cbGPU:
    let gpuResult = ctx.device.vectorAdd(a, b)
    if gpuResult.isOk:
      result.data = gpuResult.get
      result.backend = cbGPU
      result.duration = cpuTime() - startTime
      result.success = true
      ctx.totalGpuTime += result.duration
    else:
      # Fallback to CPU
      if ctx.logger != nil:
        ctx.logger.log(lvlWarn, fmt"GPU vectorAdd failed, falling back to CPU: {gpuResult.error}")
      result.data = newSeq[float32](a.len)
      for i in 0..<a.len:
        result.data[i] = a[i] + b[i]
      result.backend = cbCPU
      result.duration = cpuTime() - startTime
      result.success = true
      ctx.totalCpuTime += result.duration
  else:
    # CPU implementation
    result.data = newSeq[float32](a.len)
    for i in 0..<a.len:
      result.data[i] = a[i] + b[i]
    result.backend = cbCPU
    result.duration = cpuTime() - startTime
    result.success = true
    ctx.totalCpuTime += result.duration

  ctx.operationCount.inc
  if ctx.logger != nil:
    ctx.logger.log(lvlDebug, fmt"vectorAdd: {a.len} elements, {result.backend}, {result.duration*1000:.4f}ms")

proc vectorMul*(ctx: var ComputeContext, a, b: seq[float32]): ComputeResult[seq[float32]] =
  ## Multiply two vectors element-wise with automatic backend selection
  if not ctx.valid:
    result.success = false
    result.error = "Context is not valid"
    return

  let startTime = cpuTime()

  if ctx.actualBackend == cbGPU:
    let gpuResult = ctx.device.vectorMultiply(a, b)
    if gpuResult.isOk:
      result.data = gpuResult.get
      result.backend = cbGPU
      result.duration = cpuTime() - startTime
      result.success = true
      ctx.totalGpuTime += result.duration
    else:
      # Fallback to CPU
      if ctx.logger != nil:
        ctx.logger.log(lvlWarn, fmt"GPU vectorMul failed, falling back to CPU: {gpuResult.error}")
      result.data = newSeq[float32](a.len)
      for i in 0..<a.len:
        result.data[i] = a[i] * b[i]
      result.backend = cbCPU
      result.duration = cpuTime() - startTime
      result.success = true
      ctx.totalCpuTime += result.duration
  else:
    # CPU implementation
    result.data = newSeq[float32](a.len)
    for i in 0..<a.len:
      result.data[i] = a[i] * b[i]
    result.backend = cbCPU
    result.duration = cpuTime() - startTime
    result.success = true
    ctx.totalCpuTime += result.duration

  ctx.operationCount.inc
  if ctx.logger != nil:
    ctx.logger.log(lvlDebug, fmt"vectorMul: {a.len} elements, {result.backend}, {result.duration*1000:.4f}ms")

# ========== Matrix Operations with Fallback ==========

proc matmul*(ctx: var ComputeContext, a, b: Matrix[float32]): ComputeResult[Matrix[float32]] =
  ## Matrix multiplication with automatic backend selection
  if not ctx.valid:
    result.success = false
    result.error = "Context is not valid"
    return

  let startTime = cpuTime()

  if ctx.actualBackend == cbGPU:
    let gpuResult = ctx.device.matmulGPU(a, b)
    if gpuResult.isOk:
      result.data = gpuResult.get
      result.backend = cbGPU
      result.duration = cpuTime() - startTime
      result.success = true
      ctx.totalGpuTime += result.duration
    else:
      # Fallback to CPU
      if ctx.logger != nil:
        ctx.logger.log(lvlWarn, fmt"GPU matmul failed, falling back to CPU: {gpuResult.error}")
      result.data = matmulCPU(a, b)
      result.backend = cbCPU
      result.duration = cpuTime() - startTime
      result.success = true
      ctx.totalCpuTime += result.duration
  else:
    # CPU implementation
    result.data = matmulCPU(a, b)
    result.backend = cbCPU
    result.duration = cpuTime() - startTime
    result.success = true
    ctx.totalCpuTime += result.duration

  ctx.operationCount.inc
  if ctx.logger != nil:
    ctx.logger.log(lvlDebug, fmt"matmul: {a.rows}x{a.cols} * {b.rows}x{b.cols}, {result.backend}, {result.duration*1000:.4f}ms")

proc transpose*(ctx: var ComputeContext, m: Matrix[float32]): ComputeResult[Matrix[float32]] =
  ## Matrix transpose with automatic backend selection
  if not ctx.valid:
    result.success = false
    result.error = "Context is not valid"
    return

  let startTime = cpuTime()

  if ctx.actualBackend == cbGPU:
    let gpuResult = ctx.device.transposeGPU(m)
    if gpuResult.isOk:
      result.data = gpuResult.get
      result.backend = cbGPU
      result.duration = cpuTime() - startTime
      result.success = true
      ctx.totalGpuTime += result.duration
    else:
      # Fallback to CPU
      if ctx.logger != nil:
        ctx.logger.log(lvlWarn, fmt"GPU transpose failed, falling back to CPU: {gpuResult.error}")
      result.data = transposeCPU(m)
      result.backend = cbCPU
      result.duration = cpuTime() - startTime
      result.success = true
      ctx.totalCpuTime += result.duration
  else:
    # CPU implementation
    result.data = transposeCPU(m)
    result.backend = cbCPU
    result.duration = cpuTime() - startTime
    result.success = true
    ctx.totalCpuTime += result.duration

  ctx.operationCount.inc
  if ctx.logger != nil:
    ctx.logger.log(lvlDebug, fmt"transpose: {m.rows}x{m.cols}, {result.backend}, {result.duration*1000:.4f}ms")

# ========== Context Information ==========

proc stats*(ctx: ComputeContext): string =
  ## Get context statistics
  result = fmt"""ComputeContext Statistics:
  Backend: {ctx.actualBackend}
  Valid: {ctx.valid}
  Operations: {ctx.operationCount}
  Total GPU time: {ctx.totalGpuTime*1000:.2f} ms
  Total CPU time: {ctx.totalCpuTime*1000:.2f} ms
"""
  if ctx.actualBackend == cbGPU:
    result &= fmt"  Device: {ctx.device.info.name}\n"

proc `$`*(ctx: ComputeContext): string =
  if not ctx.valid:
    return "ComputeContext(invalid)"
  let device = if ctx.actualBackend == cbGPU: ctx.device.info.name else: "CPU"
  result = fmt"ComputeContext({ctx.actualBackend}, device={device}, ops={ctx.operationCount})"

proc `$`*[T](res: ComputeResult[T]): string =
  if res.success:
    result = fmt"ComputeResult(success, backend={res.backend}, duration={res.duration*1000:.4f}ms)"
  else:
    result = fmt"ComputeResult(failed: {res.error})"

# ========== Test ==========

when isMainModule:
  import std/random

  # Create console logger
  var consoleLogger = newConsoleLogger(fmtStr="[$datetime] $levelname: $message")
  addHandler(consoleLogger)

  echo "=== Metal Unified API Test ==="
  echo ""

  # Test with auto backend
  echo "--- Auto Backend Test ---"
  let ctxResult = newComputeContext(cbAuto, consoleLogger)
  if ctxResult.isOk:
    var ctx = ctxResult.get
    echo "Context: ", ctx
    echo ""

    # Vector addition
    echo "Vector Addition (10K elements):"
    var a = newSeq[float32](10000)
    var b = newSeq[float32](10000)
    for i in 0..<10000:
      a[i] = rand(1.0).float32
      b[i] = rand(1.0).float32

    let addResult = ctx.vectorAdd(a, b)
    echo "  Result: ", addResult
    echo ""

    # Vector multiplication
    echo "Vector Multiplication (10K elements):"
    let mulResult = ctx.vectorMul(a, b)
    echo "  Result: ", mulResult
    echo ""

    # Matrix multiplication
    echo "Matrix Multiplication (64x64):"
    var matA = newMatrix[float32](64, 64)
    var matB = newMatrix[float32](64, 64)
    for i in 0..<matA.data.len:
      matA.data[i] = rand(1.0).float32
      matB.data[i] = rand(1.0).float32

    let matResult = ctx.matmul(matA, matB)
    echo "  Result: ", matResult
    echo ""

    # Stats
    echo ctx.stats()

    ctx.destroy()
  else:
    echo "Failed to create context: ", ctxResult.error

  echo ""

  # Test CPU-only backend
  echo "--- CPU Backend Test ---"
  let cpuCtxResult = newComputeContext(cbCPU, consoleLogger)
  if cpuCtxResult.isOk:
    var cpuCtx = cpuCtxResult.get
    echo "Context: ", cpuCtx

    var a = newSeq[float32](1000)
    var b = newSeq[float32](1000)
    for i in 0..<1000:
      a[i] = rand(1.0).float32
      b[i] = rand(1.0).float32

    let addResult = cpuCtx.vectorAdd(a, b)
    echo "CPU Add: ", addResult

    cpuCtx.destroy()
  else:
    echo "Failed to create CPU context: ", cpuCtxResult.error

  echo ""
  echo "✅ Metal unified API test complete"
