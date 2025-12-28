## Metal Command Queue and Encoder
## MTLCommandQueue, MTLCommandBuffer, MTLComputeCommandEncoder wrappers
##
## v0.0.3: Command submission infrastructure
##
## Usage:
##   let queue = device.newCommandQueue()
##   let cmdBuffer = queue.newCommandBuffer()
##   let encoder = cmdBuffer.newComputeEncoder()
##   encoder.setBuffer(buffer, 0)
##   encoder.dispatch(...)
##   encoder.endEncoding()
##   cmdBuffer.commit()
##   cmdBuffer.waitUntilCompleted()

import std/[strformat]
import errors
import metal_device
import metal_buffer
import objc_runtime

when defined(macosx):
  {.passL: "-framework Metal".}
  {.passL: "-framework Foundation".}

type
  # Opaque Metal types
  MTLCommandBufferRef* = distinct pointer
  MTLComputeCommandEncoderRef* = distinct pointer
  MTLComputePipelineStateRef* = distinct pointer

  # Command buffer status
  MTLCommandBufferStatus* = enum
    cbsNotEnqueued = 0
    cbsEnqueued = 1
    cbsCommitted = 2
    cbsScheduled = 3
    cbsCompleted = 4
    cbsError = 5

  # Wrapped command queue
  MetalCommandQueue* = object
    handle*: MTLCommandQueueRef
    device*: MetalDevice
    valid*: bool

  # Wrapped command buffer
  MetalCommandBuffer* = object
    handle*: MTLCommandBufferRef
    queue*: MetalCommandQueue
    valid*: bool

  # Wrapped compute encoder
  MetalComputeEncoder* = object
    handle*: MTLComputeCommandEncoderRef
    commandBuffer*: MetalCommandBuffer
    valid*: bool

  # Thread configuration for dispatch
  MTLSize* = object
    width*: uint
    height*: uint
    depth*: uint

# ========== Helper functions ==========

proc mtlSize*(width, height, depth: int): MTLSize =
  ## Create MTLSize from integers
  MTLSize(width: width.uint, height: height.uint, depth: depth.uint)

proc mtlSize1D*(width: int): MTLSize =
  ## Create 1D MTLSize
  mtlSize(width, 1, 1)

proc mtlSize2D*(width, height: int): MTLSize =
  ## Create 2D MTLSize
  mtlSize(width, height, 1)

# ========== Command Queue ==========

proc newCommandQueue*(device: MetalDevice): NMCResult[MetalCommandQueue] =
  ## Create a new command queue for the device
  ## Note: v0.0.3 provides stub implementation - actual queue creation planned for v0.0.4
  when defined(macosx):
    if not device.valid or device.handle.pointer == nil:
      return err[MetalCommandQueue](newError(ekDeviceNotFound,
        "Invalid Metal device",
        "Cannot create command queue on invalid device"))

    # v0.0.3: Stub implementation - objc_msgSend integration pending
    result = ok(MetalCommandQueue(
      handle: MTLCommandQueueRef(nil),  # Stub handle
      device: device,
      valid: true  # Marked valid for API testing
    ))
  else:
    result = err[MetalCommandQueue](newError(ekMetalNotAvailable,
      "Metal is only available on macOS"))

proc release*(queue: var MetalCommandQueue) =
  ## Release the command queue
  ## Note: v0.0.3 stub - no actual release needed
  queue.valid = false
  queue.handle = MTLCommandQueueRef(nil)

# ========== Command Buffer ==========

proc newCommandBuffer*(queue: MetalCommandQueue): NMCResult[MetalCommandBuffer] =
  ## Create a new command buffer from the queue
  ## Note: v0.0.3 provides stub implementation
  when defined(macosx):
    if not queue.valid:
      return err[MetalCommandBuffer](newError(ekPipelineError,
        "Invalid command queue",
        "Cannot create command buffer from invalid queue"))

    # v0.0.3: Stub implementation
    result = ok(MetalCommandBuffer(
      handle: MTLCommandBufferRef(nil),  # Stub handle
      queue: queue,
      valid: true
    ))
  else:
    result = err[MetalCommandBuffer](newError(ekMetalNotAvailable,
      "Metal is only available on macOS"))

proc status*(cmdBuffer: MetalCommandBuffer): MTLCommandBufferStatus =
  ## Get the current status of the command buffer
  ## Note: v0.0.3 stub - returns completed for testing
  when defined(macosx):
    if not cmdBuffer.valid:
      return cbsError
    # v0.0.3: Stub - return completed for testing
    result = cbsCompleted
  else:
    result = cbsError

proc commit*(cmdBuffer: MetalCommandBuffer): VoidResult =
  ## Commit the command buffer for execution
  ## Note: v0.0.3 stub - no actual execution
  when defined(macosx):
    if not cmdBuffer.valid:
      return errVoid(ekPipelineError, "Command buffer is not valid")
    # v0.0.3: Stub - no-op
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc waitUntilCompleted*(cmdBuffer: MetalCommandBuffer): VoidResult =
  ## Wait for the command buffer to complete execution
  ## Note: v0.0.3 stub - returns immediately
  when defined(macosx):
    if not cmdBuffer.valid:
      return errVoid(ekPipelineError, "Command buffer is not valid")
    # v0.0.3: Stub - return immediately
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc waitUntilScheduled*(cmdBuffer: MetalCommandBuffer): VoidResult =
  ## Wait for the command buffer to be scheduled
  ## Note: v0.0.3 stub - returns immediately
  when defined(macosx):
    if not cmdBuffer.valid:
      return errVoid(ekPipelineError, "Command buffer is not valid")
    # v0.0.3: Stub - return immediately
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc release*(cmdBuffer: var MetalCommandBuffer) =
  ## Release the command buffer
  ## Note: v0.0.3 stub - no actual release needed
  cmdBuffer.valid = false
  cmdBuffer.handle = MTLCommandBufferRef(nil)

# ========== Compute Command Encoder ==========

proc newComputeEncoder*(cmdBuffer: MetalCommandBuffer): NMCResult[MetalComputeEncoder] =
  ## Create a new compute command encoder
  ## Note: v0.0.3 provides stub implementation
  when defined(macosx):
    if not cmdBuffer.valid:
      return err[MetalComputeEncoder](newError(ekPipelineError,
        "Invalid command buffer",
        "Cannot create compute encoder from invalid command buffer"))

    # v0.0.3: Stub implementation
    result = ok(MetalComputeEncoder(
      handle: MTLComputeCommandEncoderRef(nil),  # Stub handle
      commandBuffer: cmdBuffer,
      valid: true
    ))
  else:
    result = err[MetalComputeEncoder](newError(ekMetalNotAvailable,
      "Metal is only available on macOS"))

proc setBuffer*(encoder: MetalComputeEncoder, buffer: MetalBuffer,
                offset: int, index: int): VoidResult =
  ## Set a buffer at the specified index
  ## Note: v0.0.3 stub - no actual encoding
  when defined(macosx):
    if not encoder.valid:
      return errVoid(ekPipelineError, "Encoder is not valid")
    if not buffer.valid:
      return errVoid(ekBufferAllocationError, "Buffer is not valid")
    # v0.0.3: Stub - no-op
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc setBytes*(encoder: MetalComputeEncoder, data: pointer,
               length: int, index: int): VoidResult =
  ## Set bytes directly at the specified index
  ## Note: v0.0.3 stub - no actual encoding
  when defined(macosx):
    if not encoder.valid:
      return errVoid(ekPipelineError, "Encoder is not valid")
    # v0.0.3: Stub - no-op
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc setComputePipelineState*(encoder: MetalComputeEncoder,
                              pipeline: MTLComputePipelineStateRef): VoidResult =
  ## Set the compute pipeline state
  ## Note: v0.0.3 stub - no actual encoding
  when defined(macosx):
    if not encoder.valid:
      return errVoid(ekPipelineError, "Encoder is not valid")
    if pipeline.pointer == nil:
      return errVoid(ekPipelineError, "Pipeline state is nil")
    # v0.0.3: Stub - no-op
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc dispatchThreadgroups*(encoder: MetalComputeEncoder,
                           threadgroupsPerGrid: MTLSize,
                           threadsPerThreadgroup: MTLSize): VoidResult =
  ## Dispatch compute work using threadgroups
  ## Note: v0.0.3 stub - no actual dispatch
  when defined(macosx):
    if not encoder.valid:
      return errVoid(ekPipelineError, "Encoder is not valid")
    # v0.0.3: Stub - no-op
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc dispatchThreads*(encoder: MetalComputeEncoder,
                      threadsPerGrid: MTLSize,
                      threadsPerThreadgroup: MTLSize): VoidResult =
  ## Dispatch compute work using non-uniform threadgroups (Metal 2+)
  ## Note: v0.0.3 stub - no actual dispatch
  when defined(macosx):
    if not encoder.valid:
      return errVoid(ekPipelineError, "Encoder is not valid")
    # v0.0.3: Stub - no-op
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc endEncoding*(encoder: var MetalComputeEncoder): VoidResult =
  ## End encoding commands
  ## Note: v0.0.3 stub - just marks encoder as invalid
  when defined(macosx):
    if not encoder.valid:
      return errVoid(ekPipelineError, "Encoder is not valid")
    encoder.valid = false
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

# ========== String representations ==========

proc `$`*(queue: MetalCommandQueue): string =
  if not queue.valid:
    return "MetalCommandQueue(invalid)"
  result = fmt"MetalCommandQueue(device: {queue.device.info.name})"

proc `$`*(cmdBuffer: MetalCommandBuffer): string =
  if not cmdBuffer.valid:
    return "MetalCommandBuffer(invalid)"
  result = fmt"MetalCommandBuffer(status: {cmdBuffer.status})"

proc `$`*(encoder: MetalComputeEncoder): string =
  if not encoder.valid:
    return "MetalComputeEncoder(invalid)"
  result = "MetalComputeEncoder(active)"

proc `$`*(size: MTLSize): string =
  result = fmt"MTLSize({size.width} x {size.height} x {size.depth})"

# ========== Test ==========

when isMainModule:
  echo "=== Metal Command Test ==="
  echo ""

  let deviceResult = getDefaultDevice()
  if not deviceResult.isOk:
    echo "Error: ", deviceResult.error
    quit(1)

  let device = deviceResult.get
  echo "Device: ", device.info.name
  echo ""

  # Test command queue creation
  let queueResult = newCommandQueue(device)
  if queueResult.isOk:
    var queue = queueResult.get
    echo "Command queue: ", queue
    echo ""

    # Test command buffer creation
    let cmdBufferResult = newCommandBuffer(queue)
    if cmdBufferResult.isOk:
      var cmdBuffer = cmdBufferResult.get
      echo "Command buffer: ", cmdBuffer
      echo "Initial status: ", cmdBuffer.status
      echo ""

      # Test compute encoder creation
      let encoderResult = newComputeEncoder(cmdBuffer)
      if encoderResult.isOk:
        var encoder = encoderResult.get
        echo "Compute encoder: ", encoder

        # End encoding (no actual compute work without pipeline)
        let endResult = encoder.endEncoding()
        if endResult.isOk:
          echo "Encoding ended successfully"
        else:
          echo "End encoding error: ", endResult.error
      else:
        echo "Encoder creation error: ", encoderResult.error

      # Commit and wait (no-op since no work encoded)
      let commitResult = cmdBuffer.commit()
      if commitResult.isOk:
        echo "Command buffer committed"
        let waitResult = cmdBuffer.waitUntilCompleted()
        if waitResult.isOk:
          echo "Command buffer completed"
          echo "Final status: ", cmdBuffer.status
        else:
          echo "Wait error: ", waitResult.error
      else:
        echo "Commit error: ", commitResult.error

      cmdBuffer.release()
    else:
      echo "Command buffer error: ", cmdBufferResult.error

    queue.release()
  else:
    echo "Queue creation error: ", queueResult.error

  echo ""
  echo "✅ Metal command test complete"
