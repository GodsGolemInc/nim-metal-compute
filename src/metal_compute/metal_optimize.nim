## Metal Shader Optimization
## v0.0.9: Shader optimization utilities and recommendations
##
## This module provides:
## - Optimal thread group size calculation
## - Memory coalescing recommendations
## - Shader performance hints
## - Device-specific optimizations

import std/[strformat, algorithm, math]
import ./errors
import ./metal_device
import ./metal_shader
import ./metal_wrapper

type
  ThreadGroupConfig* = object
    ## Recommended thread group configuration
    threadsPerThreadgroup*: tuple[x, y, z: int]
    threadgroupsPerGrid*: tuple[x, y, z: int]
    totalThreads*: int
    efficiency*: float  # 0.0 to 1.0

  OptimizationHint* = object
    ## Shader optimization hint
    category*: string
    description*: string
    impact*: string  # "high", "medium", "low"

  ShaderProfile* = object
    ## Profile information for a shader
    name*: string
    maxThreadsPerThreadgroup*: int
    threadExecutionWidth*: int
    staticThreadgroupMemory*: int
    hints*: seq[OptimizationHint]

# ========== Thread Group Size Optimization ==========

proc optimizeThreadGroupSize1D*(workSize: int,
                                 maxThreadsPerThreadgroup: int,
                                 threadExecutionWidth: int = 32): ThreadGroupConfig =
  ## Calculate optimal 1D thread group configuration
  # Use thread execution width as base for efficiency
  var threadGroupSize = threadExecutionWidth

  # Scale up if work size is large
  while threadGroupSize * 2 <= maxThreadsPerThreadgroup and
        threadGroupSize * 2 <= workSize:
    threadGroupSize *= 2

  let threadgroups = (workSize + threadGroupSize - 1) div threadGroupSize

  result = ThreadGroupConfig(
    threadsPerThreadgroup: (threadGroupSize, 1, 1),
    threadgroupsPerGrid: (threadgroups, 1, 1),
    totalThreads: threadGroupSize * threadgroups,
    efficiency: workSize.float / (threadGroupSize * threadgroups).float
  )

proc optimizeThreadGroupSize2D*(width, height: int,
                                 maxThreadsPerThreadgroup: int,
                                 threadExecutionWidth: int = 32): ThreadGroupConfig =
  ## Calculate optimal 2D thread group configuration
  # Start with 16x16 or smaller
  var groupWidth = min(16, width)
  var groupHeight = min(16, height)

  # Adjust to fit within max threads
  while groupWidth * groupHeight > maxThreadsPerThreadgroup:
    if groupWidth > groupHeight:
      groupWidth = groupWidth div 2
    else:
      groupHeight = groupHeight div 2

  # Ensure we use at least threadExecutionWidth worth of threads
  if groupWidth * groupHeight < threadExecutionWidth:
    groupWidth = min(width, threadExecutionWidth)
    groupHeight = 1

  let gridWidth = (width + groupWidth - 1) div groupWidth
  let gridHeight = (height + groupHeight - 1) div groupHeight

  result = ThreadGroupConfig(
    threadsPerThreadgroup: (groupWidth, groupHeight, 1),
    threadgroupsPerGrid: (gridWidth, gridHeight, 1),
    totalThreads: groupWidth * groupHeight * gridWidth * gridHeight,
    efficiency: (width * height).float / (groupWidth * groupHeight * gridWidth * gridHeight).float
  )

proc optimizeForMatmul*(M, N, K: int,
                        maxThreadsPerThreadgroup: int): ThreadGroupConfig =
  ## Optimized thread group configuration for matrix multiplication
  # Use 16x16 tiles for good cache utilization
  var tileSize = 16

  while tileSize * tileSize > maxThreadsPerThreadgroup:
    tileSize = tileSize div 2

  let gridM = (M + tileSize - 1) div tileSize
  let gridN = (N + tileSize - 1) div tileSize

  result = ThreadGroupConfig(
    threadsPerThreadgroup: (tileSize, tileSize, 1),
    threadgroupsPerGrid: (gridN, gridM, 1),  # Note: column-major dispatch
    totalThreads: tileSize * tileSize * gridM * gridN,
    efficiency: (M * N).float / (tileSize * tileSize * gridM * gridN).float
  )

# ========== Shader Profiling ==========

proc profilePipeline*(device: MetalDevice,
                      pipeline: MetalComputePipeline): ShaderProfile =
  ## Get profile information for a compute pipeline
  result.name = pipeline.function.name
  result.maxThreadsPerThreadgroup = pipeline.maxThreadsPerThreadgroup.int
  result.threadExecutionWidth = pipeline.threadExecutionWidth.int
  result.staticThreadgroupMemory = pipeline.staticThreadgroupMemoryLength.int
  result.hints = @[]

  # Generate hints based on profile
  if result.threadExecutionWidth < 32:
    result.hints.add(OptimizationHint(
      category: "Thread Width",
      description: "Thread execution width is less than 32. Consider restructuring shader.",
      impact: "medium"
    ))

  if result.staticThreadgroupMemory > 16384:
    result.hints.add(OptimizationHint(
      category: "Threadgroup Memory",
      description: "High threadgroup memory usage may limit occupancy.",
      impact: "high"
    ))

  if result.maxThreadsPerThreadgroup < 256:
    result.hints.add(OptimizationHint(
      category: "Thread Limit",
      description: "Low max threads per threadgroup. Review register usage.",
      impact: "medium"
    ))

# ========== Optimization Recommendations ==========

proc getOptimizationHints*(device: MetalDevice): seq[OptimizationHint] =
  ## Get general optimization hints for the device
  result = @[]

  # Unified memory hints
  if device.info.hasUnifiedMemory:
    result.add(OptimizationHint(
      category: "Memory",
      description: "Device has unified memory. Use shared storage mode for CPU/GPU data.",
      impact: "high"
    ))
  else:
    result.add(OptimizationHint(
      category: "Memory",
      description: "Device has discrete memory. Minimize CPU/GPU data transfers.",
      impact: "high"
    ))

  # Buffer size hints
  result.add(OptimizationHint(
    category: "Buffers",
    description: fmt"Max buffer size: {device.info.maxBufferLength div (1024*1024)} MB. Keep buffers within limit.",
    impact: "medium"
  ))

  # Apple Silicon hint (unified memory indicates Apple Silicon)
  if device.info.hasUnifiedMemory:
    result.add(OptimizationHint(
      category: "GPU Features",
      description: "Apple Silicon GPU detected. Advanced features available.",
      impact: "low"
    ))

# ========== Memory Coalescing Utilities ==========

proc alignedSize*(size: int, alignment: int = 16): int =
  ## Calculate aligned size for optimal memory access
  result = (size + alignment - 1) and (not (alignment - 1))

proc isCoalescedAccess*(stride: int, elementSize: int): bool =
  ## Check if memory access pattern is coalesced
  # Coalesced if stride equals element size (sequential access)
  # or stride is a multiple of 128 bytes (cache line aligned)
  result = stride == elementSize or stride mod 128 == 0

proc suggestVectorization*(elementSize: int): int =
  ## Suggest vectorization factor for optimal performance
  # Metal prefers float4 (16 byte) operations
  case elementSize
  of 1: result = 16  # char -> char16
  of 2: result = 8   # short -> short8
  of 4: result = 4   # float -> float4
  of 8: result = 2   # double -> double2
  else: result = 1

# ========== String representations ==========

proc `$`*(config: ThreadGroupConfig): string =
  result = fmt"ThreadGroup({config.threadsPerThreadgroup.x}x{config.threadsPerThreadgroup.y}x{config.threadsPerThreadgroup.z})"
  result &= fmt" x Grid({config.threadgroupsPerGrid.x}x{config.threadgroupsPerGrid.y}x{config.threadgroupsPerGrid.z})"
  result &= fmt" = {config.totalThreads} threads (eff: {config.efficiency*100:.1f}%)"

proc `$`*(hint: OptimizationHint): string =
  result = fmt"[{hint.impact}] {hint.category}: {hint.description}"

proc `$`*(profile: ShaderProfile): string =
  result = fmt"""ShaderProfile({profile.name})
  Max threads/threadgroup: {profile.maxThreadsPerThreadgroup}
  Thread execution width: {profile.threadExecutionWidth}
  Threadgroup memory: {profile.staticThreadgroupMemory} bytes
"""
  if profile.hints.len > 0:
    result &= "  Hints:\n"
    for hint in profile.hints:
      result &= fmt"    - {hint}\n"

# ========== Test ==========

when isMainModule:
  echo "=== Metal Shader Optimization Test ==="
  echo ""

  let deviceResult = getDefaultDevice()
  if not deviceResult.isOk:
    echo "Error: ", deviceResult.error
    quit(1)

  var device = deviceResult.get
  echo "Device: ", device.info.name
  echo ""

  # Test thread group optimization
  echo "--- Thread Group Optimization ---"

  let config1D = optimizeThreadGroupSize1D(1000000, 1024, 32)
  echo "1D work (1M elements): ", config1D

  let config2D = optimizeThreadGroupSize2D(1920, 1080, 1024, 32)
  echo "2D work (1920x1080): ", config2D

  let configMat = optimizeForMatmul(512, 512, 512, 1024)
  echo "Matrix 512x512: ", configMat

  echo ""

  # Test device hints
  echo "--- Device Optimization Hints ---"
  let hints = getOptimizationHints(device)
  for hint in hints:
    echo "  ", hint

  echo ""

  # Test utility functions
  echo "--- Utility Functions ---"
  echo fmt"Aligned size (100 -> 16): {alignedSize(100, 16)}"
  echo fmt"Aligned size (100 -> 64): {alignedSize(100, 64)}"
  echo fmt"Coalesced (stride=4, elem=4): {isCoalescedAccess(4, 4)}"
  echo fmt"Coalesced (stride=8, elem=4): {isCoalescedAccess(8, 4)}"
  echo fmt"Vectorization for float (4 bytes): {suggestVectorization(4)}"

  echo ""

  device.release()
  echo "✅ Metal optimization test complete"
