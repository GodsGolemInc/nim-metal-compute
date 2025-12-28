## Metal Buffer Management
## MTLBuffer wrapper for Nim
##
## v0.0.4: Buffer allocation and data transfer via C wrapper
##
## Memory modes:
##   - Shared: CPU/GPU shared access (unified memory)
##   - Private: GPU only
##   - Managed: Explicit synchronization

import std/[strformat]
import errors
import metal_device
import metal_wrapper

type
  # Opaque Metal buffer type
  MTLBufferRef* = distinct pointer

  # Storage mode for buffers
  MTLStorageMode* = enum
    smShared = 0    # CPU & GPU can access
    smManaged = 1   # Explicit sync needed
    smPrivate = 2   # GPU only
    smMemoryless = 3 # Tile memory only

  # Resource options
  MTLResourceOptions* = distinct uint32

  # Wrapped buffer
  MetalBuffer* = object
    handle*: MTLBufferRef
    device*: MetalDevice
    length*: int
    storageMode*: MTLStorageMode
    valid*: bool

# ========== Resource options ==========

const
  MTLResourceStorageModeShared* = MTLResourceOptions(0 shl 4)
  MTLResourceStorageModeManaged* = MTLResourceOptions(1 shl 4)
  MTLResourceStorageModePrivate* = MTLResourceOptions(2 shl 4)
  MTLResourceCPUCacheModeDefaultCache* = MTLResourceOptions(0 shl 0)
  MTLResourceCPUCacheModeWriteCombined* = MTLResourceOptions(1 shl 0)
  MTLResourceHazardTrackingModeDefault* = MTLResourceOptions(0 shl 8)
  MTLResourceHazardTrackingModeUntracked* = MTLResourceOptions(1 shl 8)

proc `or`(a, b: MTLResourceOptions): MTLResourceOptions =
  MTLResourceOptions(uint32(a) or uint32(b))

proc resourceOptionsFromStorageMode(mode: MTLStorageMode): uint32 =
  case mode
  of smShared: uint32(MTLResourceStorageModeShared)
  of smManaged: uint32(MTLResourceStorageModeManaged)
  of smPrivate: uint32(MTLResourceStorageModePrivate)
  of smMemoryless: uint32(MTLResourceStorageModePrivate)  # Fallback

# ========== Public API ==========

proc newBuffer*(device: MetalDevice, length: int,
                mode: MTLStorageMode = smShared): NMCResult[MetalBuffer] =
  ## Create a new buffer with specified length and storage mode
  when defined(macosx):
    if not device.valid or device.handle.pointer == nil:
      return err[MetalBuffer](newError(ekDeviceNotFound,
        "Invalid Metal device",
        "Cannot create buffer on invalid device"))

    if length <= 0:
      return err[MetalBuffer](newError(ekInvalidInputSize,
        fmt"Invalid buffer length: {length}",
        "Buffer length must be positive"))

    if length.uint64 > device.info.maxBufferLength:
      return err[MetalBuffer](newError(ekBufferAllocationError,
        fmt"Buffer length {length} exceeds device max {device.info.maxBufferLength}",
        "Requested buffer size is too large for this device"))

    let options = resourceOptionsFromStorageMode(mode)
    let handle = nmc_create_buffer(device.handle.pointer, length.uint64, options)

    if handle == nil:
      return err[MetalBuffer](newError(ekBufferAllocationError,
        "Failed to allocate Metal buffer",
        fmt"Could not allocate {length} bytes"))

    result = ok(MetalBuffer(
      handle: MTLBufferRef(handle),
      device: device,
      length: length,
      storageMode: mode,
      valid: true
    ))
  else:
    result = err[MetalBuffer](newError(ekMetalNotAvailable,
      "Metal is only available on macOS"))

proc newBufferWithData*[T](device: MetalDevice, data: openArray[T],
                           mode: MTLStorageMode = smShared): NMCResult[MetalBuffer] =
  ## Create a new buffer initialized with data
  when defined(macosx):
    if not device.valid or device.handle.pointer == nil:
      return err[MetalBuffer](newError(ekDeviceNotFound,
        "Invalid Metal device"))

    if data.len == 0:
      return err[MetalBuffer](newError(ekInvalidInputSize,
        "Cannot create buffer from empty data"))

    let length = data.len * sizeof(T)
    let options = resourceOptionsFromStorageMode(mode)
    let handle = nmc_create_buffer_with_data(device.handle.pointer,
                                               unsafeAddr data[0],
                                               length.uint64, options)

    if handle == nil:
      return err[MetalBuffer](newError(ekBufferAllocationError,
        "Failed to allocate Metal buffer with data",
        fmt"Could not allocate {length} bytes"))

    result = ok(MetalBuffer(
      handle: MTLBufferRef(handle),
      device: device,
      length: length,
      storageMode: mode,
      valid: true
    ))
  else:
    result = err[MetalBuffer](newError(ekMetalNotAvailable,
      "Metal is only available on macOS"))

proc contents*(buffer: MetalBuffer): pointer =
  ## Get pointer to buffer contents (for shared/managed only)
  when defined(macosx):
    if not buffer.valid or buffer.handle.pointer == nil:
      return nil

    if buffer.storageMode == smPrivate:
      return nil  # Private storage cannot be accessed from CPU

    result = nmc_buffer_contents(buffer.handle.pointer)
  else:
    result = nil

proc write*[T](buffer: var MetalBuffer, data: openArray[T], offset: int = 0): VoidResult =
  ## Write data to buffer at specified offset
  if not buffer.valid:
    return errVoid(ekBufferAllocationError, "Buffer is not valid")

  if buffer.storageMode == smPrivate:
    return errVoid(ekBufferAllocationError,
      "Cannot write to private storage buffer from CPU")

  let dataSize = data.len * sizeof(T)
  if offset + dataSize > buffer.length:
    return errVoid(ekBufferAllocationError,
      fmt"Data overflow: offset({offset}) + size({dataSize}) > buffer length({buffer.length})")

  let bufPtr = buffer.contents()
  if bufPtr == nil:
    return errVoid(ekBufferAllocationError, "Could not get buffer contents")

  let destPtr = cast[pointer](cast[uint](bufPtr) + offset.uint)
  copyMem(destPtr, unsafeAddr data[0], dataSize)

  result = okVoid()

proc read*[T](buffer: MetalBuffer, dest: var openArray[T], offset: int = 0): VoidResult =
  ## Read data from buffer at specified offset
  if not buffer.valid:
    return errVoid(ekBufferAllocationError, "Buffer is not valid")

  if buffer.storageMode == smPrivate:
    return errVoid(ekBufferAllocationError,
      "Cannot read from private storage buffer to CPU")

  let dataSize = dest.len * sizeof(T)
  if offset + dataSize > buffer.length:
    return errVoid(ekBufferAllocationError,
      fmt"Read overflow: offset({offset}) + size({dataSize}) > buffer length({buffer.length})")

  let bufPtr = buffer.contents()
  if bufPtr == nil:
    return errVoid(ekBufferAllocationError, "Could not get buffer contents")

  let srcPtr = cast[pointer](cast[uint](bufPtr) + offset.uint)
  copyMem(addr dest[0], srcPtr, dataSize)

  result = okVoid()

proc didModifyRange*(buffer: MetalBuffer, offset, length: int) =
  ## Notify that buffer was modified in range (for managed storage)
  when defined(macosx):
    if not buffer.valid or buffer.storageMode != smManaged:
      return
    if buffer.handle.pointer != nil:
      nmc_buffer_did_modify_range(buffer.handle.pointer, offset.uint64, length.uint64)

proc synchronize*(buffer: MetalBuffer) =
  ## Synchronize buffer for reading from CPU (managed storage)
  when defined(macosx):
    if not buffer.valid or buffer.storageMode != smManaged:
      return

    # Note: This requires a blit command encoder in a command buffer
    # For now, this is a placeholder. Full implementation in metal_command.nim
    discard

proc release*(buffer: var MetalBuffer) =
  ## Release the buffer
  when defined(macosx):
    if buffer.valid and buffer.handle.pointer != nil:
      nmc_release_buffer(buffer.handle.pointer)

  buffer.valid = false
  buffer.handle = MTLBufferRef(nil)

proc `$`*(buffer: MetalBuffer): string =
  ## String representation
  if not buffer.valid:
    return "MetalBuffer(invalid)"

  result = fmt"MetalBuffer(length: {buffer.length}, mode: {buffer.storageMode})"

proc summary*(buffer: MetalBuffer): string =
  ## Detailed buffer summary
  if not buffer.valid:
    return "Invalid Metal buffer"

  result = fmt"""
Metal Buffer
  Length:       {buffer.length} bytes ({buffer.length / 1024:.2f} KB)
  Storage Mode: {buffer.storageMode}
  Device:       {buffer.device.info.name}
  Valid:        {buffer.valid}
"""

# ========== Test ==========

when isMainModule:
  echo "=== Metal Buffer Test ==="
  echo ""

  let deviceResult = getDefaultDevice()
  if not deviceResult.isOk:
    echo "Error: ", deviceResult.error
    quit(1)

  var device = deviceResult.get
  echo "Device: ", device.info.name
  echo ""

  # Test buffer creation
  let bufferResult = newBuffer(device, 1024 * sizeof(float32))
  if bufferResult.isOk:
    var buffer = bufferResult.get
    echo buffer.summary()

    # Test write
    var data = newSeq[float32](256)
    for i in 0..<256:
      data[i] = i.float32

    let writeResult = buffer.write(data)
    if writeResult.isOk:
      echo "Write successful"

      # Test read
      var readData = newSeq[float32](256)
      let readResult = buffer.read(readData)
      if readResult.isOk:
        echo "Read successful"
        echo "First few values: ", readData[0..4]
      else:
        echo "Read error: ", readResult.error

    else:
      echo "Write error: ", writeResult.error

    buffer.release()
    echo "Buffer released"

  else:
    echo "Buffer creation error: ", bufferResult.error

  device.release()
  echo ""
  echo "Metal buffer test complete"
