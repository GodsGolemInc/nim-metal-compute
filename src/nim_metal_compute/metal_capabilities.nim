## Metal Device Capabilities Detection
## Comprehensive GPU feature detection for Metal
##
## v0.0.3: Device capability queries
##
## Features:
##   - GPU family detection (Apple, Mac, Common)
##   - Feature set detection
##   - Compute limits query
##   - Memory constraints

import std/[strformat, strutils, tables]
import errors
import metal_device

when defined(macosx):
  {.passL: "-framework Metal".}
  {.passL: "-framework Foundation".}

type
  # GPU family for capability detection
  MTLGPUFamily* = enum
    gfUnknown = 0
    # Apple GPU families (Apple Silicon)
    gfApple1 = 1001    # A7 GPU
    gfApple2 = 1002    # A8 GPU
    gfApple3 = 1003    # A9/A10 GPU
    gfApple4 = 1004    # A11 GPU
    gfApple5 = 1005    # A12 GPU
    gfApple6 = 1006    # A13 GPU
    gfApple7 = 1007    # A14/M1 GPU
    gfApple8 = 1008    # A15/M2 GPU
    gfApple9 = 1009    # A17/M3 GPU
    # Mac GPU families
    gfMac1 = 2001      # Mac family 1
    gfMac2 = 2002      # Mac family 2
    # Common GPU families
    gfCommon1 = 3001   # Common tier 1
    gfCommon2 = 3002   # Common tier 2
    gfCommon3 = 3003   # Common tier 3

  # Compute capabilities
  ComputeCapabilities* = object
    maxThreadsPerThreadgroup*: int
    maxThreadgroupMemoryLength*: int
    maxTotalThreadgroupsPerMeshGrid*: int
    simdWidth*: int
    supportsNonUniformThreadgroups*: bool
    supports32BitFloatFiltering*: bool
    supports32BitMSAA*: bool
    supportsBCTextureCompression*: bool
    supportsQueryTextureLOD*: bool
    supportsPrimitiveMotionBlur*: bool
    supportsRayTracing*: bool
    supports3DAstcTextures*: bool

  # Memory capabilities
  MemoryCapabilities* = object
    recommendedMaxWorkingSetSize*: uint64
    maxBufferLength*: uint64
    hasUnifiedMemory*: bool
    currentAllocatedSize*: uint64
    maxTransferRate*: uint64  # Estimated

  # Full device capabilities
  DeviceCapabilities* = object
    device*: MetalDevice
    gpuFamily*: MTLGPUFamily
    compute*: ComputeCapabilities
    memory*: MemoryCapabilities
    featureSupport*: Table[string, bool]

# ========== C/Objective-C bindings ==========
# Note: v0.0.3 uses stub implementations due to objc_msgSend issues

when defined(macosx):
  # Stub versions for v0.0.3
  proc supportsFamily(device: MTLDeviceRef, family: int): bool =
    ## Stub - always returns false for v0.0.3
    false

  proc supportsFeatureSet(device: MTLDeviceRef, featureSet: int): bool =
    ## Stub - always returns false for v0.0.3
    false

# ========== GPU Family Detection ==========

proc detectGPUFamily*(device: MetalDevice): MTLGPUFamily =
  ## Detect the highest supported GPU family
  ## Note: v0.0.3 returns assumed Apple7+ for Apple Silicon Macs
  when defined(macosx):
    if not device.valid:
      return gfUnknown

    # v0.0.3: Stub - assume Apple7 (M1+) for Apple Silicon
    # Actual detection will be implemented in v0.0.4
    result = gfApple7
  else:
    result = gfUnknown

proc familyName*(family: MTLGPUFamily): string =
  ## Get human-readable name for GPU family
  case family
  of gfUnknown: "Unknown"
  of gfApple1: "Apple GPU Family 1 (A7)"
  of gfApple2: "Apple GPU Family 2 (A8)"
  of gfApple3: "Apple GPU Family 3 (A9/A10)"
  of gfApple4: "Apple GPU Family 4 (A11)"
  of gfApple5: "Apple GPU Family 5 (A12)"
  of gfApple6: "Apple GPU Family 6 (A13)"
  of gfApple7: "Apple GPU Family 7 (A14/M1)"
  of gfApple8: "Apple GPU Family 8 (A15/M2)"
  of gfApple9: "Apple GPU Family 9 (A17/M3)"
  of gfMac1: "Mac GPU Family 1"
  of gfMac2: "Mac GPU Family 2"
  of gfCommon1: "Common GPU Family 1"
  of gfCommon2: "Common GPU Family 2"
  of gfCommon3: "Common GPU Family 3"

proc isAppleSiliconFamily*(family: MTLGPUFamily): bool =
  ## Check if GPU family is Apple Silicon
  family in {gfApple7, gfApple8, gfApple9}

# ========== Compute Capabilities ==========

proc getComputeCapabilities*(device: MetalDevice): ComputeCapabilities =
  ## Get compute-related capabilities
  when defined(macosx):
    if not device.valid or device.handle.pointer == nil:
      return ComputeCapabilities()

    let family = detectGPUFamily(device)

    # Default values based on GPU family
    result.maxThreadsPerThreadgroup = 1024  # Most Metal devices
    result.maxThreadgroupMemoryLength = 32768  # 32KB typical
    result.simdWidth = 32  # 32 for most GPUs

    # Family-specific capabilities
    case family
    of gfApple7, gfApple8, gfApple9:
      result.supportsNonUniformThreadgroups = true
      result.supports32BitFloatFiltering = true
      result.supports32BitMSAA = true
      result.supportsBCTextureCompression = true
      result.supportsQueryTextureLOD = true
      result.supportsPrimitiveMotionBlur = true
      result.supportsRayTracing = family >= gfApple6
      result.supports3DAstcTextures = true
    of gfApple4, gfApple5, gfApple6:
      result.supportsNonUniformThreadgroups = family >= gfApple4
      result.supports32BitFloatFiltering = true
      result.supports32BitMSAA = family >= gfApple5
      result.supportsBCTextureCompression = false
      result.supportsQueryTextureLOD = family >= gfApple5
      result.supportsPrimitiveMotionBlur = false
      result.supportsRayTracing = family >= gfApple6
    of gfMac1, gfMac2:
      result.supportsNonUniformThreadgroups = true
      result.supports32BitFloatFiltering = true
      result.supports32BitMSAA = true
      result.supportsBCTextureCompression = true
      result.supportsQueryTextureLOD = true
    else:
      result.supportsNonUniformThreadgroups = false
      result.supports32BitFloatFiltering = false
      result.supports32BitMSAA = false
  else:
    result = ComputeCapabilities()

# ========== Memory Capabilities ==========

proc getMemoryCapabilities*(device: MetalDevice): MemoryCapabilities =
  ## Get memory-related capabilities
  when defined(macosx):
    if not device.valid:
      return MemoryCapabilities()

    result.recommendedMaxWorkingSetSize = device.info.recommendedMaxWorkingSetSize
    result.maxBufferLength = device.info.maxBufferLength
    result.hasUnifiedMemory = device.info.hasUnifiedMemory

    # Estimate transfer rate based on device type
    if result.hasUnifiedMemory:
      # Apple Silicon: unified memory, very fast
      result.maxTransferRate = 400_000_000_000'u64  # ~400 GB/s
    else:
      # Discrete GPU: PCIe bandwidth
      result.maxTransferRate = 32_000_000_000'u64  # ~32 GB/s PCIe 4.0 x16
  else:
    result = MemoryCapabilities()

# ========== Full Capabilities ==========

proc getCapabilities*(device: MetalDevice): NMCResult[DeviceCapabilities] =
  ## Get complete device capabilities
  when defined(macosx):
    if not device.valid or device.handle.pointer == nil:
      return err[DeviceCapabilities](newError(ekDeviceNotFound,
        "Invalid Metal device",
        "Cannot get capabilities of invalid device"))

    var caps = DeviceCapabilities(
      device: device,
      gpuFamily: detectGPUFamily(device),
      compute: getComputeCapabilities(device),
      memory: getMemoryCapabilities(device)
    )

    # Build feature support table
    caps.featureSupport = initTable[string, bool]()
    caps.featureSupport["nonUniformThreadgroups"] = caps.compute.supportsNonUniformThreadgroups
    caps.featureSupport["32BitFloatFiltering"] = caps.compute.supports32BitFloatFiltering
    caps.featureSupport["32BitMSAA"] = caps.compute.supports32BitMSAA
    caps.featureSupport["bcTextureCompression"] = caps.compute.supportsBCTextureCompression
    caps.featureSupport["queryTextureLOD"] = caps.compute.supportsQueryTextureLOD
    caps.featureSupport["primitiveMotionBlur"] = caps.compute.supportsPrimitiveMotionBlur
    caps.featureSupport["rayTracing"] = caps.compute.supportsRayTracing
    caps.featureSupport["3DAstcTextures"] = caps.compute.supports3DAstcTextures
    caps.featureSupport["unifiedMemory"] = caps.memory.hasUnifiedMemory
    caps.featureSupport["appleSilicon"] = isAppleSiliconFamily(caps.gpuFamily)

    result = ok(caps)
  else:
    result = err[DeviceCapabilities](newError(ekMetalNotAvailable,
      "Metal is only available on macOS"))

# ========== String representations ==========

proc `$`*(caps: ComputeCapabilities): string =
  result = fmt"""Compute Capabilities:
  Max threads/threadgroup: {caps.maxThreadsPerThreadgroup}
  Max threadgroup memory:  {caps.maxThreadgroupMemoryLength} bytes
  SIMD width:              {caps.simdWidth}
  Non-uniform dispatch:    {caps.supportsNonUniformThreadgroups}
  Ray tracing:             {caps.supportsRayTracing}"""

proc `$`*(caps: MemoryCapabilities): string =
  result = fmt"""Memory Capabilities:
  Recommended max working set: {caps.recommendedMaxWorkingSetSize div (1024*1024)} MB
  Max buffer length:           {caps.maxBufferLength div (1024*1024*1024)} GB
  Unified memory:              {caps.hasUnifiedMemory}
  Estimated transfer rate:     {caps.maxTransferRate div 1_000_000_000} GB/s"""

proc summary*(caps: DeviceCapabilities): string =
  ## Get detailed capabilities summary
  if not caps.device.valid:
    return "Invalid device capabilities"

  result = fmt"""
Device Capabilities Summary
{'='.repeat(50)}
Device:     {caps.device.info.name}
GPU Family: {familyName(caps.gpuFamily)}
Apple Silicon: {isAppleSiliconFamily(caps.gpuFamily)}

{caps.compute}

{caps.memory}

Feature Support:
"""
  for feature, supported in caps.featureSupport:
    result.add fmt"  {feature}: {supported}\n"

# ========== Compute Recommendations ==========

proc recommendedThreadgroupSize*(caps: DeviceCapabilities, totalThreads: int): tuple[threads, groups: int] =
  ## Recommend optimal threadgroup configuration
  let maxThreads = caps.compute.maxThreadsPerThreadgroup
  let simd = caps.compute.simdWidth

  # Align to SIMD width for efficiency
  var threadsPerGroup = min(maxThreads, 256)  # Common optimal size
  threadsPerGroup = (threadsPerGroup div simd) * simd  # Align to SIMD

  let numGroups = (totalThreads + threadsPerGroup - 1) div threadsPerGroup
  result = (threadsPerGroup, numGroups)

proc isCapableFor*(caps: DeviceCapabilities, feature: string): bool =
  ## Check if device is capable of a specific feature
  if caps.featureSupport.hasKey(feature):
    return caps.featureSupport[feature]
  return false

# ========== Test ==========

when isMainModule:
  echo "=== Metal Capabilities Test ==="
  echo ""

  let deviceResult = getDefaultDevice()
  if not deviceResult.isOk:
    echo "Error: ", deviceResult.error
    quit(1)

  let device = deviceResult.get
  echo "Device: ", device.info.name
  echo ""

  let capsResult = getCapabilities(device)
  if capsResult.isOk:
    let caps = capsResult.get
    echo caps.summary()

    # Test recommendations
    let (threads, groups) = recommendedThreadgroupSize(caps, 1000000)
    echo fmt"Recommended for 1M threads: {threads} threads/group, {groups} groups"
    echo ""

    # Test feature check
    echo "Feature checks:"
    echo "  Ray tracing capable: ", isCapableFor(caps, "rayTracing")
    echo "  Apple Silicon: ", isCapableFor(caps, "appleSilicon")
    echo "  Unified memory: ", isCapableFor(caps, "unifiedMemory")
  else:
    echo "Error getting capabilities: ", capsResult.error

  echo ""
  echo "✅ Metal capabilities test complete"
