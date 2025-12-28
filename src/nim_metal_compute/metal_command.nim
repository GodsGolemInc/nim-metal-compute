## Metal Command Queue and Encoder
## MTLCommandQueue, MTLCommandBuffer, MTLComputeCommandEncoder wrappers
##
## v0.0.4: Command submission via C wrapper
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
import metal_wrapper

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
  when defined(macosx):
    if not device.valid or device.handle.pointer == nil:
      return err[MetalCommandQueue](newError(ekDeviceNotFound,
        "Invalid Metal device",
        "Cannot create command queue on invalid device"))

    let handle = nmc_create_command_queue(device.handle.pointer)
    if handle == nil:
      return err[MetalCommandQueue](newError(ekPipelineError,
        "Failed to create command queue",
        "Could not create Metal command queue for device"))

    result = ok(MetalCommandQueue(
      handle: MTLCommandQueueRef(handle),
      device: device,
      valid: true
    ))
  else:
    result = err[MetalCommandQueue](newError(ekMetalNotAvailable,
      "Metal is only available on macOS"))

proc release*(queue: var MetalCommandQueue) =
  ## Release the command queue
  when defined(macosx):
    if queue.valid and queue.handle.pointer != nil:
      nmc_release_command_queue(queue.handle.pointer)
  queue.valid = false
  queue.handle = MTLCommandQueueRef(nil)

# ========== Command Buffer ==========

proc newCommandBuffer*(queue: MetalCommandQueue): NMCResult[MetalCommandBuffer] =
  ## Create a new command buffer from the queue
  when defined(macosx):
    if not queue.valid or queue.handle.pointer == nil:
      return err[MetalCommandBuffer](newError(ekPipelineError,
        "Invalid command queue",
        "Cannot create command buffer from invalid queue"))

    let handle = nmc_create_command_buffer(queue.handle.pointer)
    if handle == nil:
      return err[MetalCommandBuffer](newError(ekPipelineError,
        "Failed to create command buffer",
        "Could not create Metal command buffer"))

    result = ok(MetalCommandBuffer(
      handle: MTLCommandBufferRef(handle),
      queue: queue,
      valid: true
    ))
  else:
    result = err[MetalCommandBuffer](newError(ekMetalNotAvailable,
      "Metal is only available on macOS"))

proc status*(cmdBuffer: MetalCommandBuffer): MTLCommandBufferStatus =
  ## Get the current status of the command buffer
  when defined(macosx):
    if not cmdBuffer.valid or cmdBuffer.handle.pointer == nil:
      return cbsError
    let s = nmc_command_buffer_status(cmdBuffer.handle.pointer)
    result = MTLCommandBufferStatus(s)
  else:
    result = cbsError

proc commit*(cmdBuffer: MetalCommandBuffer): VoidResult =
  ## Commit the command buffer for execution
  when defined(macosx):
    if not cmdBuffer.valid or cmdBuffer.handle.pointer == nil:
      return errVoid(ekPipelineError, "Command buffer is not valid")
    nmc_command_buffer_commit(cmdBuffer.handle.pointer)
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc waitUntilCompleted*(cmdBuffer: MetalCommandBuffer): VoidResult =
  ## Wait for the command buffer to complete execution
  when defined(macosx):
    if not cmdBuffer.valid or cmdBuffer.handle.pointer == nil:
      return errVoid(ekPipelineError, "Command buffer is not valid")
    nmc_command_buffer_wait_until_completed(cmdBuffer.handle.pointer)
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc waitUntilScheduled*(cmdBuffer: MetalCommandBuffer): VoidResult =
  ## Wait for the command buffer to be scheduled
  when defined(macosx):
    if not cmdBuffer.valid or cmdBuffer.handle.pointer == nil:
      return errVoid(ekPipelineError, "Command buffer is not valid")
    nmc_command_buffer_wait_until_scheduled(cmdBuffer.handle.pointer)
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc release*(cmdBuffer: var MetalCommandBuffer) =
  ## Release the command buffer
  when defined(macosx):
    if cmdBuffer.valid and cmdBuffer.handle.pointer != nil:
      nmc_release_command_buffer(cmdBuffer.handle.pointer)
  cmdBuffer.valid = false
  cmdBuffer.handle = MTLCommandBufferRef(nil)

# ========== Compute Command Encoder ==========

proc newComputeEncoder*(cmdBuffer: MetalCommandBuffer): NMCResult[MetalComputeEncoder] =
  ## Create a new compute command encoder
  when defined(macosx):
    if not cmdBuffer.valid or cmdBuffer.handle.pointer == nil:
      return err[MetalComputeEncoder](newError(ekPipelineError,
        "Invalid command buffer",
        "Cannot create compute encoder from invalid command buffer"))

    let handle = nmc_create_compute_encoder(cmdBuffer.handle.pointer)
    if handle == nil:
      return err[MetalComputeEncoder](newError(ekPipelineError,
        "Failed to create compute encoder",
        "Could not create Metal compute encoder"))

    result = ok(MetalComputeEncoder(
      handle: MTLComputeCommandEncoderRef(handle),
      commandBuffer: cmdBuffer,
      valid: true
    ))
  else:
    result = err[MetalComputeEncoder](newError(ekMetalNotAvailable,
      "Metal is only available on macOS"))

proc setBuffer*(encoder: MetalComputeEncoder, buffer: MetalBuffer,
                offset: int, index: int): VoidResult =
  ## Set a buffer at the specified index
  when defined(macosx):
    if not encoder.valid or encoder.handle.pointer == nil:
      return errVoid(ekPipelineError, "Encoder is not valid")
    if not buffer.valid or buffer.handle.pointer == nil:
      return errVoid(ekBufferAllocationError, "Buffer is not valid")
    nmc_encoder_set_buffer(encoder.handle.pointer, buffer.handle.pointer,
                           offset.uint64, index.uint32)
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc setBytes*(encoder: MetalComputeEncoder, data: pointer,
               length: int, index: int): VoidResult =
  ## Set bytes directly at the specified index
  when defined(macosx):
    if not encoder.valid or encoder.handle.pointer == nil:
      return errVoid(ekPipelineError, "Encoder is not valid")
    if data == nil:
      return errVoid(ekInvalidInputSize, "Data pointer is nil")
    nmc_encoder_set_bytes(encoder.handle.pointer, data,
                          length.uint64, index.uint32)
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc setComputePipelineState*(encoder: MetalComputeEncoder,
                              pipeline: MTLComputePipelineStateRef): VoidResult =
  ## Set the compute pipeline state
  when defined(macosx):
    if not encoder.valid or encoder.handle.pointer == nil:
      return errVoid(ekPipelineError, "Encoder is not valid")
    if pipeline.pointer == nil:
      return errVoid(ekPipelineError, "Pipeline state is nil")
    nmc_encoder_set_pipeline_state(encoder.handle.pointer, pipeline.pointer)
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc dispatchThreadgroups*(encoder: MetalComputeEncoder,
                           threadgroupsPerGrid: MTLSize,
                           threadsPerThreadgroup: MTLSize): VoidResult =
  ## Dispatch compute work using threadgroups
  when defined(macosx):
    if not encoder.valid or encoder.handle.pointer == nil:
      return errVoid(ekPipelineError, "Encoder is not valid")
    nmc_encoder_dispatch_threadgroups(encoder.handle.pointer,
                                       threadgroupsPerGrid.width.uint64,
                                       threadgroupsPerGrid.height.uint64,
                                       threadgroupsPerGrid.depth.uint64,
                                       threadsPerThreadgroup.width.uint64,
                                       threadsPerThreadgroup.height.uint64,
                                       threadsPerThreadgroup.depth.uint64)
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc dispatchThreads*(encoder: MetalComputeEncoder,
                      threadsPerGrid: MTLSize,
                      threadsPerThreadgroup: MTLSize): VoidResult =
  ## Dispatch compute work using non-uniform threadgroups (Metal 2+)
  when defined(macosx):
    if not encoder.valid or encoder.handle.pointer == nil:
      return errVoid(ekPipelineError, "Encoder is not valid")
    nmc_encoder_dispatch_threads(encoder.handle.pointer,
                                  threadsPerGrid.width.uint64,
                                  threadsPerGrid.height.uint64,
                                  threadsPerGrid.depth.uint64,
                                  threadsPerThreadgroup.width.uint64,
                                  threadsPerThreadgroup.height.uint64,
                                  threadsPerThreadgroup.depth.uint64)
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc endEncoding*(encoder: var MetalComputeEncoder): VoidResult =
  ## End encoding commands
  when defined(macosx):
    if not encoder.valid or encoder.handle.pointer == nil:
      return errVoid(ekPipelineError, "Encoder is not valid")
    nmc_encoder_end_encoding(encoder.handle.pointer)
    encoder.valid = false
    result = okVoid()
  else:
    result = errVoid(ekMetalNotAvailable, "Metal is only available on macOS")

proc release*(encoder: var MetalComputeEncoder) =
  ## Release the compute encoder
  when defined(macosx):
    if encoder.handle.pointer != nil:
      nmc_release_encoder(encoder.handle.pointer)
  encoder.valid = false
  encoder.handle = MTLComputeCommandEncoderRef(nil)

# ========== String representations ==========

proc `$`*(queue: MetalCommandQueue): string =
  if not queue.valid:
    return "MetalCommandQueue(invalid)"
  result = fmt"MetalCommandQueue(device: {queue.device.info.name})"

proc `$`*(cmdBuffer: MetalCommandBuffer): string =
  if not cmdBuffer.valid:
    return "MetalCommandBuffer(invalid)"
  if cmdBuffer.handle.pointer == nil:
    return "MetalCommandBuffer(nil handle)"
  result = fmt"MetalCommandBuffer(valid, handle: {cast[uint](cmdBuffer.handle.pointer):#X})"

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
      # Note: status() may not work correctly before commit
      # echo "Initial status: ", cmdBuffer.status
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
          # Note: status() temporarily disabled due to memory issue
          # echo "Final status: ", cmdBuffer.status
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
