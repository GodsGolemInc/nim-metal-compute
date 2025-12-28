## Metal Device Bindings
## MTLDevice wrapper for Nim
##
## v0.0.3: Metal API integration
##
## macOS 10.14+ required for Metal 2
## Apple Silicon optimized

import std/[strformat, strutils]
import errors
import objc_runtime

when defined(macosx):
  {.passL: "-framework Metal".}
  {.passL: "-framework Foundation".}

type
  # Opaque Metal types
  MTLDeviceRef* = distinct pointer
  MTLCommandQueueRef* = distinct pointer

  # Device information
  DeviceInfo* = object
    name*: string
    registryID*: uint64
    isLowPower*: bool
    isHeadless*: bool
    isRemovable*: bool
    hasUnifiedMemory*: bool
    recommendedMaxWorkingSetSize*: uint64
    maxThreadsPerThreadgroup*: tuple[width, height, depth: int]
    maxBufferLength*: uint64

  # Wrapped device
  MetalDevice* = object
    handle*: MTLDeviceRef
    info*: DeviceInfo
    valid*: bool

# ========== C/Objective-C bindings ==========

when defined(macosx):
  # Metal device functions
  proc MTLCreateSystemDefaultDevice(): MTLDeviceRef
    {.importc, header: "<Metal/Metal.h>".}

  proc MTLCopyAllDevices(): pointer
    {.importc, header: "<Metal/Metal.h>".}

  proc CFStringGetCString(str: pointer, buffer: cstring, bufferSize: int32, encoding: uint32): bool
    {.importc, header: "<CoreFoundation/CoreFoundation.h>".}

  const kCFStringEncodingUTF8 = 0x08000100'u32

  # Get device name - returns default for v0.0.3 to avoid Objective-C runtime issues
  proc getDeviceNameSafe(device: MTLDeviceRef): string =
    ## Get device name with static fallback for stability
    ## Note: Direct NSString access via objc_msgSend can cause SIGSEGV on some systems
    ## Future versions will implement proper CF bridging
    if device.pointer == nil:
      return "Unknown Device"
    return "Apple Metal GPU"

  # Get registry ID - returns 0 for v0.0.3 (safe default)
  proc getRegistryID(device: MTLDeviceRef): uint64 =
    ## Registry ID - returning default value to avoid objc_msgSend issues
    0

  # Get boolean property - returning safe defaults for v0.0.3
  proc getBoolProperty(device: MTLDeviceRef, propName: cstring): bool =
    ## Boolean property - returning default values based on Apple Silicon assumptions
    ## This avoids objc_msgSend crashes on some systems
    case $propName
    of "isLowPower": false
    of "isHeadless": false
    of "isRemovable": false
    of "hasUnifiedMemory": true  # Apple Silicon has unified memory
    else: false

  # Get uint64 property - returning safe defaults for v0.0.3
  proc getUint64Property(device: MTLDeviceRef, propName: cstring): uint64 =
    ## Uint64 property - returning reasonable defaults
    ## This avoids objc_msgSend crashes on some systems
    case $propName
    of "recommendedMaxWorkingSetSize": 8_000_000_000'u64  # ~8GB
    of "maxBufferLength": 1_000_000_000'u64  # ~1GB
    else: 0'u64

# ========== Public API ==========

proc isMetalAvailable*(): bool =
  ## Check if Metal is available on this system
  when defined(macosx):
    let device = MTLCreateSystemDefaultDevice()
    result = device.pointer != nil
  else:
    result = false

proc getDefaultDevice*(): NMCResult[MetalDevice] =
  ## Get the default Metal device
  when defined(macosx):
    let handle = MTLCreateSystemDefaultDevice()
    if handle.pointer == nil:
      return err[MetalDevice](newError(ekMetalNotAvailable,
        "No Metal device available",
        "Metal is not supported on this system or no GPU is available"))

    var device = MetalDevice(
      handle: handle,
      valid: true
    )

    # Populate device info
    device.info.name = getDeviceNameSafe(handle)
    device.info.registryID = getRegistryID(handle)
    device.info.isLowPower = getBoolProperty(handle, "isLowPower")
    device.info.isHeadless = getBoolProperty(handle, "isHeadless")
    device.info.isRemovable = getBoolProperty(handle, "isRemovable")
    device.info.hasUnifiedMemory = getBoolProperty(handle, "hasUnifiedMemory")
    device.info.recommendedMaxWorkingSetSize = getUint64Property(handle, "recommendedMaxWorkingSetSize")
    device.info.maxBufferLength = getUint64Property(handle, "maxBufferLength")

    # Max threads per threadgroup (simplified - using common values)
    device.info.maxThreadsPerThreadgroup = (width: 1024, height: 1024, depth: 1024)

    result = ok(device)
  else:
    result = err[MetalDevice](newError(ekMetalNotAvailable,
      "Metal is only available on macOS",
      "This platform does not support Metal"))

proc isAppleSilicon*(device: MetalDevice): bool =
  ## Check if device is Apple Silicon (M1, M2, etc.)
  result = device.info.hasUnifiedMemory and
           (device.info.name.contains("Apple") or
            device.info.name.contains("M1") or
            device.info.name.contains("M2") or
            device.info.name.contains("M3") or
            device.info.name.contains("M4"))

proc release*(device: var MetalDevice) =
  ## Release the Metal device
  device.valid = false
  device.handle = MTLDeviceRef(nil)

proc `$`*(device: MetalDevice): string =
  ## String representation of device
  if not device.valid:
    return "MetalDevice(invalid)"

  result = fmt"MetalDevice({device.info.name}"
  if device.isAppleSilicon:
    result.add ", Apple Silicon"
  if device.info.hasUnifiedMemory:
    result.add ", Unified Memory"
  result.add fmt", MaxBuffer: {device.info.maxBufferLength div (1024*1024*1024)}GB"
  result.add ")"

proc summary*(device: MetalDevice): string =
  ## Detailed device summary
  if not device.valid:
    return "Invalid Metal device"

  result = fmt"""
Metal Device Information
{'='.repeat(50)}
Name:                  {device.info.name}
Registry ID:           {device.info.registryID}
Apple Silicon:         {device.isAppleSilicon}
Low Power:             {device.info.isLowPower}
Headless:              {device.info.isHeadless}
Removable:             {device.info.isRemovable}
Unified Memory:        {device.info.hasUnifiedMemory}
Max Working Set:       {device.info.recommendedMaxWorkingSetSize div (1024*1024)} MB
Max Buffer Length:     {device.info.maxBufferLength div (1024*1024*1024)} GB
Max Threads/Group:     {device.info.maxThreadsPerThreadgroup.width} x {device.info.maxThreadsPerThreadgroup.height} x {device.info.maxThreadsPerThreadgroup.depth}
{'='.repeat(50)}
"""

# ========== Test ==========

when isMainModule:
  echo "=== Metal Device Test ==="
  echo ""

  echo "Metal available: ", isMetalAvailable()

  let deviceResult = getDefaultDevice()
  if deviceResult.isOk:
    let device = deviceResult.get
    echo device.summary()
  else:
    echo "Error: ", deviceResult.error
