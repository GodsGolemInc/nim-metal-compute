## Metal Stress Testing
## v0.0.9: Stress testing and validation utilities
##
## This module provides:
## - Stress testing for GPU compute operations
## - Memory pressure testing
## - Concurrent operation testing
## - Validation and verification utilities

import std/[times, strformat, random, sequtils]
import ./errors
import ./metal_device
import ./metal_buffer
import ./metal_command
import ./metal_shader
import ./metal_compute
import ./metal_matrix
import ./metal_async
import ./metal_wrapper

type
  StressTestResult* = object
    ## Results from a stress test
    testName*: string
    passed*: bool
    iterations*: int
    totalTime*: float64
    avgTime*: float64
    minTime*: float64
    maxTime*: float64
    errorMessage*: string
    memoryAllocated*: int64
    operationsPerSecond*: float64

  StressTestConfig* = object
    ## Configuration for stress tests
    iterations*: int
    warmupIterations*: int
    vectorSize*: int
    matrixSize*: int
    bufferCount*: int
    timeout*: float64  # seconds

# ========== Default Config ==========

proc defaultStressConfig*(): StressTestConfig =
  StressTestConfig(
    iterations: 100,
    warmupIterations: 10,
    vectorSize: 1024 * 1024,  # 1M elements
    matrixSize: 256,          # 256x256
    bufferCount: 100,
    timeout: 60.0
  )

# ========== Stress Test Utilities ==========

proc formatResult*(res: StressTestResult): string =
  if res.passed:
    result = fmt"""
✅ {res.testName}
   Iterations: {res.iterations}
   Total time: {res.totalTime * 1000:.2f} ms
   Avg time: {res.avgTime * 1000:.4f} ms
   Min time: {res.minTime * 1000:.4f} ms
   Max time: {res.maxTime * 1000:.4f} ms
   Ops/sec: {res.operationsPerSecond:.2f}
"""
  else:
    result = fmt"""
❌ {res.testName}
   Error: {res.errorMessage}
"""

# ========== Buffer Allocation Stress Test ==========

proc stressTestBufferAllocation*(device: MetalDevice,
                                  config: StressTestConfig): StressTestResult =
  ## Stress test buffer allocation and deallocation
  result.testName = "Buffer Allocation Stress"
  result.iterations = config.iterations

  var times: seq[float64] = @[]
  var totalMem: int64 = 0

  let startTotal = cpuTime()

  for i in 0..<config.iterations:
    let startIter = cpuTime()

    # Allocate multiple buffers
    var buffers: seq[MetalBuffer] = @[]
    for j in 0..<config.bufferCount:
      let size = (j + 1) * 1024  # Varying sizes
      let bufResult = device.newBuffer(size)
      if bufResult.isOk:
        buffers.add(bufResult.get)
        totalMem += size
      else:
        result.passed = false
        result.errorMessage = $bufResult.error
        return

    # Release all buffers
    for buffer in buffers.mitems:
      buffer.release()

    times.add(cpuTime() - startIter)

  result.totalTime = cpuTime() - startTotal
  result.passed = true
  result.memoryAllocated = totalMem
  result.avgTime = result.totalTime / config.iterations.float
  result.minTime = times.min
  result.maxTime = times.max
  result.operationsPerSecond = config.iterations.float / result.totalTime

# ========== Vector Compute Stress Test ==========

proc stressTestVectorCompute*(device: MetalDevice,
                               config: StressTestConfig): StressTestResult =
  ## Stress test vector compute operations
  result.testName = "Vector Compute Stress"
  result.iterations = config.iterations

  # Create test data
  var a = newSeq[float32](config.vectorSize)
  var b = newSeq[float32](config.vectorSize)
  for i in 0..<config.vectorSize:
    a[i] = rand(1.0).float32
    b[i] = rand(1.0).float32

  var times: seq[float64] = @[]

  # Warmup
  for i in 0..<config.warmupIterations:
    let addResult = device.vectorAdd(a, b)
    if not addResult.isOk:
      result.passed = false
      result.errorMessage = $addResult.error
      return

  let startTotal = cpuTime()

  for i in 0..<config.iterations:
    let startIter = cpuTime()

    let addResult = device.vectorAdd(a, b)
    if not addResult.isOk:
      result.passed = false
      result.errorMessage = $addResult.error
      return

    times.add(cpuTime() - startIter)

  result.totalTime = cpuTime() - startTotal
  result.passed = true
  result.avgTime = result.totalTime / config.iterations.float
  result.minTime = times.min
  result.maxTime = times.max
  result.operationsPerSecond = config.iterations.float / result.totalTime

# ========== Matrix Compute Stress Test ==========

proc stressTestMatrixCompute*(device: MetalDevice,
                               config: StressTestConfig): StressTestResult =
  ## Stress test matrix multiplication
  result.testName = "Matrix Compute Stress"
  result.iterations = config.iterations

  # Create test matrices
  var a = newMatrix[float32](config.matrixSize, config.matrixSize)
  var b = newMatrix[float32](config.matrixSize, config.matrixSize)
  for i in 0..<a.data.len:
    a.data[i] = rand(1.0).float32
    b.data[i] = rand(1.0).float32

  var times: seq[float64] = @[]

  # Warmup
  for i in 0..<config.warmupIterations:
    let mulResult = device.matmulGPU(a, b)
    if not mulResult.isOk:
      result.passed = false
      result.errorMessage = $mulResult.error
      return

  let startTotal = cpuTime()

  for i in 0..<config.iterations:
    let startIter = cpuTime()

    let mulResult = device.matmulGPU(a, b)
    if not mulResult.isOk:
      result.passed = false
      result.errorMessage = $mulResult.error
      return

    times.add(cpuTime() - startIter)

  result.totalTime = cpuTime() - startTotal
  result.passed = true
  result.avgTime = result.totalTime / config.iterations.float
  result.minTime = times.min
  result.maxTime = times.max
  result.operationsPerSecond = config.iterations.float / result.totalTime

# ========== Memory Pressure Test ==========

proc stressTestMemoryPressure*(device: MetalDevice,
                                config: StressTestConfig): StressTestResult =
  ## Test behavior under memory pressure
  result.testName = "Memory Pressure Stress"

  # Try to allocate increasingly large buffers
  var maxSize = 1024 * 1024  # Start at 1MB
  var totalAllocated: int64 = 0
  var buffers: seq[MetalBuffer] = @[]

  let startTime = cpuTime()

  while cpuTime() - startTime < config.timeout:
    let bufResult = device.newBuffer(maxSize)
    if bufResult.isOk:
      buffers.add(bufResult.get)
      totalAllocated += maxSize
      maxSize = (maxSize.float * 1.5).int  # Increase by 50%
    else:
      # Reached memory limit
      break

  result.iterations = buffers.len
  result.memoryAllocated = totalAllocated
  result.totalTime = cpuTime() - startTime
  result.passed = buffers.len > 0

  # Cleanup
  for buffer in buffers.mitems:
    buffer.release()

  if not result.passed:
    result.errorMessage = "Could not allocate any buffers"

# ========== Async Operation Stress Test ==========

proc stressTestAsyncOperations*(device: MetalDevice,
                                 config: StressTestConfig): StressTestResult =
  ## Stress test async command buffer operations
  result.testName = "Async Operations Stress"
  result.iterations = config.iterations

  let queueResult = device.newCommandQueue()
  if not queueResult.isOk:
    result.passed = false
    result.errorMessage = $queueResult.error
    return
  var queue = queueResult.get

  var times: seq[float64] = @[]
  let startTotal = cpuTime()

  for i in 0..<config.iterations:
    let startIter = cpuTime()

    let asyncResult = newAsyncCommandBuffer(queue)
    if not asyncResult.isOk:
      result.passed = false
      result.errorMessage = $asyncResult.error
      queue.release()
      return

    var asyncCmd = asyncResult.get

    let commitResult = asyncCmd.commitAsync()
    if not commitResult.isOk:
      result.passed = false
      result.errorMessage = $commitResult.error
      asyncCmd.release()
      queue.release()
      return

    let waitResult = asyncCmd.waitForCompletion(5000)
    if not waitResult.isOk:
      result.passed = false
      result.errorMessage = $waitResult.error
      asyncCmd.release()
      queue.release()
      return

    asyncCmd.release()
    times.add(cpuTime() - startIter)

  queue.release()

  result.totalTime = cpuTime() - startTotal
  result.passed = true
  result.avgTime = result.totalTime / config.iterations.float
  result.minTime = times.min
  result.maxTime = times.max
  result.operationsPerSecond = config.iterations.float / result.totalTime

# ========== Full Stress Test Suite ==========

proc runFullStressTest*(device: MetalDevice,
                        config: StressTestConfig = defaultStressConfig()): seq[StressTestResult] =
  ## Run all stress tests
  randomize()

  result = @[]

  echo "Running stress tests..."
  echo ""

  # Buffer allocation
  echo "  Testing buffer allocation..."
  result.add(stressTestBufferAllocation(device, config))

  # Vector compute
  echo "  Testing vector compute..."
  result.add(stressTestVectorCompute(device, config))

  # Matrix compute
  echo "  Testing matrix compute..."
  result.add(stressTestMatrixCompute(device, config))

  # Memory pressure
  echo "  Testing memory pressure..."
  result.add(stressTestMemoryPressure(device, config))

  # Async operations
  echo "  Testing async operations..."
  result.add(stressTestAsyncOperations(device, config))

  echo ""

# ========== Test ==========

when isMainModule:
  echo "=== Metal Stress Testing ==="
  echo ""

  let deviceResult = getDefaultDevice()
  if not deviceResult.isOk:
    echo "Error: ", deviceResult.error
    quit(1)

  var device = deviceResult.get
  echo "Device: ", device.info.name
  echo "Max buffer length: ", device.info.maxBufferLength div (1024 * 1024), " MB"
  echo ""

  # Run with smaller config for faster testing
  var config = defaultStressConfig()
  config.iterations = 10
  config.warmupIterations = 2
  config.vectorSize = 100_000
  config.matrixSize = 128
  config.bufferCount = 20
  config.timeout = 10.0

  let results = runFullStressTest(device, config)

  echo "=== Results ==="
  var allPassed = true
  for result in results:
    echo result.formatResult()
    if not result.passed:
      allPassed = false

  device.release()

  if allPassed:
    echo "✅ All stress tests passed"
  else:
    echo "❌ Some stress tests failed"
    quit(1)
