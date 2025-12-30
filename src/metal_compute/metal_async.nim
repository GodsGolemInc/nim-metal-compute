## Metal Async Execution
## v0.0.8: Async command submission, completion handlers, and double buffering
##
## This module provides:
## - Async command buffer submission
## - Completion handlers with callbacks
## - Double buffering for pipelined execution
## - GPU timing queries for profiling
## - Event-based synchronization

import std/[locks, times, strformat, atomics, os]
import ./errors
import ./metal_device
import ./metal_buffer
import ./metal_command
import ./metal_wrapper

# ========== Types ==========

type
  CommandBufferStatus* = enum
    cbsNotEnqueued = 0
    cbsEnqueued = 1
    cbsCommitted = 2
    cbsScheduled = 3
    cbsCompleted = 4
    cbsError = 5

  AsyncCompletion* = object
    ## Completion state for async operations
    completed*: Atomic[bool]
    status*: Atomic[int]
    error*: string
    gpuStartTime*: float64
    gpuEndTime*: float64

  AsyncCommandBuffer* = object
    ## Command buffer with async execution support
    handle*: MTLCommandBufferRef
    queue*: MetalCommandQueue
    completion*: ptr AsyncCompletion
    valid*: bool

  SharedEvent* = object
    ## Shared event for cross-command buffer synchronization
    handle*: pointer
    device*: MetalDevice
    valid*: bool

  DoubleBuffer*[T] = object
    ## Double buffer for pipelined GPU operations
    bufferA*: MetalBuffer
    bufferB*: MetalBuffer
    cmdBufferA*: AsyncCommandBuffer
    cmdBufferB*: AsyncCommandBuffer
    currentIdx*: int
    device*: MetalDevice
    queue*: MetalCommandQueue
    valid*: bool

  GPUTimingInfo* = object
    ## GPU execution timing information
    gpuStartTime*: float64
    gpuEndTime*: float64
    kernelStartTime*: float64
    kernelEndTime*: float64
    gpuDuration*: float64
    kernelDuration*: float64

# ========== Completion Callback ==========

# Global completion callback that gets called from C
proc asyncCompletionCallback(context: pointer, status: cint) {.cdecl.} =
  if context != nil:
    let completion = cast[ptr AsyncCompletion](context)
    completion.status.store(status.int)
    completion.completed.store(true)

# ========== Async Command Buffer Functions ==========

proc newAsyncCommandBuffer*(queue: MetalCommandQueue): NMCResult[AsyncCommandBuffer] =
  ## Create a new async command buffer
  when defined(macosx):
    if not queue.valid or queue.handle.pointer == nil:
      return err[AsyncCommandBuffer](ekCommand, "Invalid command queue")

    let handle = nmc_create_command_buffer(queue.handle.pointer)
    if handle == nil:
      return err[AsyncCommandBuffer](ekCommand, "Failed to create command buffer")

    # Allocate completion state on heap
    let completion = cast[ptr AsyncCompletion](alloc0(sizeof(AsyncCompletion)))
    completion.completed.store(false)
    completion.status.store(0)

    result = ok(AsyncCommandBuffer(
      handle: MTLCommandBufferRef(handle),
      queue: queue,
      completion: completion,
      valid: true
    ))
  else:
    result = err[AsyncCommandBuffer](ekPlatform, "Metal not available")

proc commitAsync*(cmdBuffer: var AsyncCommandBuffer): VoidResult =
  ## Commit the command buffer asynchronously with a completion handler
  when defined(macosx):
    if not cmdBuffer.valid or cmdBuffer.handle.pointer == nil:
      return errVoid(ekCommand, "Invalid command buffer")

    # Add completion handler
    let added = nmc_command_buffer_add_completion_handler(
      cmdBuffer.handle.pointer,
      asyncCompletionCallback,
      cmdBuffer.completion
    )

    if added == 0:
      return errVoid(ekCommand, "Failed to add completion handler")

    # Commit async
    let committed = nmc_command_buffer_commit_async(cmdBuffer.handle.pointer)
    if committed == 0:
      return errVoid(ekCommand, "Failed to commit command buffer")

    result = okVoid()
  else:
    result = errVoid(ekPlatform, "Metal not available")

proc isCompleted*(cmdBuffer: AsyncCommandBuffer): bool =
  ## Check if the async command buffer has completed
  when defined(macosx):
    if cmdBuffer.completion != nil:
      result = cmdBuffer.completion.completed.load()
    else:
      result = nmc_command_buffer_is_completed(cmdBuffer.handle.pointer) != 0
  else:
    result = true

proc waitForCompletion*(cmdBuffer: AsyncCommandBuffer, timeoutMs: int = 10000): VoidResult =
  ## Wait for the async command buffer to complete
  when defined(macosx):
    if not cmdBuffer.valid:
      return errVoid(ekCommand, "Invalid command buffer")

    let startTime = cpuTime()
    let timeoutSec = timeoutMs.float / 1000.0

    while not cmdBuffer.isCompleted():
      if cpuTime() - startTime > timeoutSec:
        return errVoid(ekCommand, "Timeout waiting for command buffer")
      # Small sleep to avoid busy waiting
      sleep(1)

    # Check for errors
    if cmdBuffer.completion != nil and cmdBuffer.completion.status.load() == cbsError.int:
      let errorMsg = nmc_command_buffer_error_message(cmdBuffer.handle.pointer)
      if errorMsg != nil:
        let msg = $errorMsg
        nmc_free_string(errorMsg)
        return errVoid(ekCommand, msg)
      else:
        return errVoid(ekCommand, "Command buffer execution error")

    result = okVoid()
  else:
    result = errVoid(ekPlatform, "Metal not available")

proc getGPUTiming*(cmdBuffer: AsyncCommandBuffer): GPUTimingInfo =
  ## Get GPU timing information from a completed command buffer
  when defined(macosx):
    if cmdBuffer.valid and cmdBuffer.handle.pointer != nil:
      result.gpuStartTime = nmc_command_buffer_gpu_start_time(cmdBuffer.handle.pointer)
      result.gpuEndTime = nmc_command_buffer_gpu_end_time(cmdBuffer.handle.pointer)
      result.kernelStartTime = nmc_command_buffer_kernel_start_time(cmdBuffer.handle.pointer)
      result.kernelEndTime = nmc_command_buffer_kernel_end_time(cmdBuffer.handle.pointer)
      result.gpuDuration = result.gpuEndTime - result.gpuStartTime
      result.kernelDuration = result.kernelEndTime - result.kernelStartTime

proc status*(cmdBuffer: AsyncCommandBuffer): CommandBufferStatus =
  ## Get the current status of the command buffer
  when defined(macosx):
    if cmdBuffer.valid and cmdBuffer.handle.pointer != nil:
      result = CommandBufferStatus(nmc_command_buffer_status(cmdBuffer.handle.pointer))
    else:
      result = cbsError
  else:
    result = cbsCompleted

proc release*(cmdBuffer: var AsyncCommandBuffer) =
  ## Release the async command buffer
  when defined(macosx):
    if cmdBuffer.valid and cmdBuffer.handle.pointer != nil:
      nmc_release_command_buffer(cmdBuffer.handle.pointer)

    if cmdBuffer.completion != nil:
      dealloc(cmdBuffer.completion)
      cmdBuffer.completion = nil

  cmdBuffer.valid = false
  cmdBuffer.handle = MTLCommandBufferRef(nil)

# ========== Shared Event Functions ==========

proc newSharedEvent*(device: MetalDevice): NMCResult[SharedEvent] =
  ## Create a shared event for synchronization
  when defined(macosx):
    if not device.valid or device.handle.pointer == nil:
      return err[SharedEvent](ekDevice, "Invalid device")

    let handle = nmc_create_shared_event(device.handle.pointer)
    if handle == nil:
      return err[SharedEvent](ekDevice, "Failed to create shared event")

    result = ok(SharedEvent(
      handle: handle,
      device: device,
      valid: true
    ))
  else:
    result = err[SharedEvent](ekPlatform, "Metal not available")

proc value*(event: SharedEvent): uint64 =
  ## Get the current signaled value of the event
  when defined(macosx):
    if event.valid and event.handle != nil:
      result = nmc_shared_event_value(event.handle)

proc encodeWait*(cmdBuffer: AsyncCommandBuffer, event: SharedEvent, value: uint64) =
  ## Encode a wait for the event to reach a specific value
  when defined(macosx):
    if cmdBuffer.valid and event.valid:
      nmc_command_buffer_encode_wait_for_event(
        cmdBuffer.handle.pointer,
        event.handle,
        value
      )

proc encodeSignal*(cmdBuffer: AsyncCommandBuffer, event: SharedEvent, value: uint64) =
  ## Encode a signal to set the event to a specific value
  when defined(macosx):
    if cmdBuffer.valid and event.valid:
      nmc_command_buffer_encode_signal_event(
        cmdBuffer.handle.pointer,
        event.handle,
        value
      )

proc release*(event: var SharedEvent) =
  ## Release the shared event
  when defined(macosx):
    if event.valid and event.handle != nil:
      nmc_release_shared_event(event.handle)

  event.valid = false
  event.handle = nil

# ========== Double Buffering ==========

proc newDoubleBuffer*[T](device: MetalDevice, size: int,
                         mode: MTLStorageMode = smShared): NMCResult[DoubleBuffer[T]] =
  ## Create a double buffer for pipelined GPU operations
  when defined(macosx):
    if not device.valid:
      return err[DoubleBuffer[T]](ekDevice, "Invalid device")

    let bufferSize = size * sizeof(T)

    # Create buffer A
    let bufAResult = device.newBuffer(bufferSize, mode)
    if not bufAResult.isOk:
      return err[DoubleBuffer[T]](bufAResult.error)
    var bufA = bufAResult.get

    # Create buffer B
    let bufBResult = device.newBuffer(bufferSize, mode)
    if not bufBResult.isOk:
      bufA.release()
      return err[DoubleBuffer[T]](bufBResult.error)
    var bufB = bufBResult.get

    # Create command queue
    let queueResult = device.newCommandQueue()
    if not queueResult.isOk:
      bufA.release()
      bufB.release()
      return err[DoubleBuffer[T]](queueResult.error)
    let queue = queueResult.get

    result = ok(DoubleBuffer[T](
      bufferA: bufA,
      bufferB: bufB,
      currentIdx: 0,
      device: device,
      queue: queue,
      valid: true
    ))
  else:
    result = err[DoubleBuffer[T]](ekPlatform, "Metal not available")

proc currentBuffer*[T](db: DoubleBuffer[T]): MetalBuffer =
  ## Get the current buffer (for writing)
  if db.currentIdx == 0:
    result = db.bufferA
  else:
    result = db.bufferB

proc previousBuffer*[T](db: DoubleBuffer[T]): MetalBuffer =
  ## Get the previous buffer (for reading results)
  if db.currentIdx == 0:
    result = db.bufferB
  else:
    result = db.bufferA

proc swap*[T](db: var DoubleBuffer[T]) =
  ## Swap the buffers
  db.currentIdx = 1 - db.currentIdx

proc destroy*[T](db: var DoubleBuffer[T]) =
  ## Destroy the double buffer and release resources
  if db.valid:
    db.bufferA.release()
    db.bufferB.release()
    db.queue.release()
    db.valid = false

# ========== Test ==========

when isMainModule:
  echo "=== Metal Async Execution Test ==="
  echo ""

  let deviceResult = getDefaultDevice()
  if not deviceResult.isOk:
    echo "Error: ", deviceResult.error
    quit(1)

  var device = deviceResult.get
  echo "Device: ", device.info.name
  echo ""

  # Test async command buffer
  echo "--- Async Command Buffer Test ---"

  let queueResult = device.newCommandQueue()
  if not queueResult.isOk:
    echo "Queue error: ", queueResult.error
    quit(1)
  var queue = queueResult.get

  let asyncCmdResult = newAsyncCommandBuffer(queue)
  if asyncCmdResult.isOk:
    var asyncCmd = asyncCmdResult.get
    echo "Async command buffer created"
    echo "Status: ", asyncCmd.status()

    # Create a simple compute operation
    let bufResult = device.newBuffer(1024 * sizeof(float32))
    if bufResult.isOk:
      var buffer = bufResult.get

      echo "Command buffer status before commit: ", asyncCmd.status()

      # Commit async (empty command buffer, will complete immediately)
      let commitResult = asyncCmd.commitAsync()
      if commitResult.isOk:
        echo "Async commit successful"

        # Wait for completion
        let waitResult = asyncCmd.waitForCompletion(5000)
        if waitResult.isOk:
          echo "Completion wait successful"
          echo "Final status: ", asyncCmd.status()

          # Get timing info
          let timing = asyncCmd.getGPUTiming()
          echo fmt"GPU duration: {timing.gpuDuration * 1000:.4f} ms"
        else:
          echo "Wait error: ", waitResult.error
      else:
        echo "Commit error: ", commitResult.error

      buffer.release()
    else:
      echo "Buffer error: ", bufResult.error

    asyncCmd.release()
  else:
    echo "Async command buffer error: ", asyncCmdResult.error

  echo ""

  # Test shared event
  echo "--- Shared Event Test ---"
  let eventResult = newSharedEvent(device)
  if eventResult.isOk:
    var event = eventResult.get
    echo "Shared event created"
    echo "Initial value: ", event.value()
    event.release()
    echo "Event released"
  else:
    echo "Event error: ", eventResult.error

  echo ""

  # Test double buffering
  echo "--- Double Buffer Test ---"
  let dbResult = newDoubleBuffer[float32](device, 1024)
  if dbResult.isOk:
    var db = dbResult.get
    echo "Double buffer created"
    echo "Current buffer length: ", db.currentBuffer().length
    echo "Previous buffer length: ", db.previousBuffer().length

    db.swap()
    echo "Buffers swapped"
    echo "Current buffer length after swap: ", db.currentBuffer().length

    db.destroy()
    echo "Double buffer destroyed"
  else:
    echo "Double buffer error: ", dbResult.error

  echo ""

  queue.release()
  device.release()

  echo "✅ Metal async execution test complete"
