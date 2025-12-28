## Metal Device Bindings
## MTLDevice wrapper for Nim
##
## v0.0.4: Metal API integration via C wrapper
##
## macOS 10.14+ required for Metal 2
## Apple Silicon optimized

import std/[strformat, strutils]
import errors
import metal_wrapper

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

# ========== Public API ==========

proc isMetalAvailable*(): bool =
  ## Check if Metal is available on this system
  when defined(macosx):
    result = nmc_metal_available() != 0
  else:
    result = false

proc getDefaultDevice*(): NMCResult[MetalDevice] =
  ## Get the default Metal device
  when defined(macosx):
    let handle = nmc_create_default_device()
    if handle == nil:
      return err[MetalDevice](newError(ekMetalNotAvailable,
        "No Metal device available",
        "Metal is not supported on this system or no GPU is available"))

    var device = MetalDevice(
      handle: MTLDeviceRef(handle),
      valid: true
    )

    # Populate device info using C wrapper
    let namePtr = nmc_device_name(handle)
    if namePtr != nil:
      device.info.name = $namePtr
      nmc_free_string(namePtr)
    else:
      device.info.name = "Unknown Metal GPU"

    device.info.registryID = nmc_device_registry_id(handle)
    device.info.isLowPower = nmc_device_is_low_power(handle) != 0
    device.info.isHeadless = nmc_device_is_headless(handle) != 0
    device.info.isRemovable = nmc_device_is_removable(handle) != 0
    device.info.hasUnifiedMemory = nmc_device_has_unified_memory(handle) != 0
    device.info.recommendedMaxWorkingSetSize = nmc_device_recommended_max_working_set_size(handle)
    device.info.maxBufferLength = nmc_device_max_buffer_length(handle)

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
  when defined(macosx):
    if device.valid and device.handle.pointer != nil:
      nmc_release_device(device.handle.pointer)
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
    var device = deviceResult.get
    echo device.summary()
    device.release()
  else:
    echo "Error: ", deviceResult.error
