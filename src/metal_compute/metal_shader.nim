## Metal Shader and Pipeline Management
## v0.0.5: Shader compilation and compute pipeline support
##
## This module provides:
## - MTLLibrary management (shader compilation)
## - MTLFunction extraction
## - MTLComputePipelineState creation

import std/[strutils, strformat, sequtils]
import ./errors
import ./metal_device
import ./metal_wrapper

# ========== Type Definitions ==========

type
  MTLLibraryRef* = distinct pointer
    ## Reference to a Metal library (compiled shaders)

  MTLFunctionRef* = distinct pointer
    ## Reference to a Metal function (kernel)

  MTLComputePipelineStateRef* = distinct pointer
    ## Reference to a compute pipeline state

  MetalLibrary* = object
    ## Represents a compiled Metal shader library
    handle*: MTLLibraryRef
    device*: MetalDevice
    valid*: bool

  MetalFunction* = object
    ## Represents a Metal kernel function
    handle*: MTLFunctionRef
    name*: string
    library*: MetalLibrary
    valid*: bool

  MetalComputePipeline* = object
    ## Represents a compute pipeline state
    handle*: MTLComputePipelineStateRef
    function*: MetalFunction
    device*: MetalDevice
    maxThreadsPerThreadgroup*: uint64
    threadExecutionWidth*: uint64
    staticThreadgroupMemoryLength*: uint64
    valid*: bool

# ========== Library Management ==========

proc compileShader*(device: MetalDevice, source: string): NMCResult[MetalLibrary] =
  ## Compile Metal shader source code into a library
  when defined(macosx):
    if not device.valid or device.handle.pointer == nil:
      return err[MetalLibrary](ekDevice, "Invalid device")

    var errorMsg: cstring = nil
    let handle = nmc_compile_library(device.handle.pointer, source.cstring, addr errorMsg)

    if handle == nil:
      let msg = if errorMsg != nil:
        let s = $errorMsg
        nmc_free_string(errorMsg)
        s
      else:
        "Unknown shader compilation error"
      return err[MetalLibrary](ekShader, "Shader compilation failed: " & msg)

    result = ok(MetalLibrary(
      handle: MTLLibraryRef(handle),
      device: device,
      valid: true
    ))
  else:
    result = err[MetalLibrary](ekPlatform, "Metal not available on this platform")

proc loadLibrary*(device: MetalDevice, path: string): NMCResult[MetalLibrary] =
  ## Load a precompiled Metal library from a file
  when defined(macosx):
    if not device.valid or device.handle.pointer == nil:
      return err[MetalLibrary](ekDevice, "Invalid device")

    var errorMsg: cstring = nil
    let handle = nmc_load_library(device.handle.pointer, path.cstring, addr errorMsg)

    if handle == nil:
      let msg = if errorMsg != nil:
        let s = $errorMsg
        nmc_free_string(errorMsg)
        s
      else:
        "Failed to load library"
      return err[MetalLibrary](ekShader, "Library load failed: " & msg)

    result = ok(MetalLibrary(
      handle: MTLLibraryRef(handle),
      device: device,
      valid: true
    ))
  else:
    result = err[MetalLibrary](ekPlatform, "Metal not available on this platform")

proc getDefaultLibrary*(device: MetalDevice): NMCResult[MetalLibrary] =
  ## Get the default library (compiled into app bundle)
  when defined(macosx):
    if not device.valid or device.handle.pointer == nil:
      return err[MetalLibrary](ekDevice, "Invalid device")

    let handle = nmc_get_default_library(device.handle.pointer)
    if handle == nil:
      return err[MetalLibrary](ekShader, "No default library available")

    result = ok(MetalLibrary(
      handle: MTLLibraryRef(handle),
      device: device,
      valid: true
    ))
  else:
    result = err[MetalLibrary](ekPlatform, "Metal not available on this platform")

proc functionNames*(library: MetalLibrary): seq[string] =
  ## Get all function names from a library
  when defined(macosx):
    if not library.valid or library.handle.pointer == nil:
      return @[]

    let names = nmc_get_function_names(library.handle.pointer)
    if names == nil or names.len == 0:
      return @[]

    let namesStr = $names
    nmc_free_string(names)

    result = namesStr.splitLines().filterIt(it.len > 0)
  else:
    result = @[]

proc release*(library: var MetalLibrary) =
  ## Release a Metal library
  when defined(macosx):
    if library.handle.pointer != nil:
      nmc_release_library(library.handle.pointer)
  library.valid = false
  library.handle = MTLLibraryRef(nil)

# ========== Function Management ==========

proc getFunction*(library: MetalLibrary, name: string): NMCResult[MetalFunction] =
  ## Get a function (kernel) from a library by name
  when defined(macosx):
    if not library.valid or library.handle.pointer == nil:
      return err[MetalFunction](ekShader, "Invalid library")

    let handle = nmc_get_function(library.handle.pointer, name.cstring)
    if handle == nil:
      return err[MetalFunction](ekShader, fmt"Function '{name}' not found in library")

    result = ok(MetalFunction(
      handle: MTLFunctionRef(handle),
      name: name,
      library: library,
      valid: true
    ))
  else:
    result = err[MetalFunction](ekPlatform, "Metal not available on this platform")

proc release*(function: var MetalFunction) =
  ## Release a Metal function
  when defined(macosx):
    if function.handle.pointer != nil:
      nmc_release_function(function.handle.pointer)
  function.valid = false
  function.handle = MTLFunctionRef(nil)

# ========== Pipeline Management ==========

proc newComputePipeline*(device: MetalDevice, function: MetalFunction): NMCResult[MetalComputePipeline] =
  ## Create a compute pipeline state from a function
  when defined(macosx):
    if not device.valid or device.handle.pointer == nil:
      return err[MetalComputePipeline](ekDevice, "Invalid device")
    if not function.valid or function.handle.pointer == nil:
      return err[MetalComputePipeline](ekShader, "Invalid function")

    var errorMsg: cstring = nil
    let handle = nmc_create_pipeline_state(device.handle.pointer,
                                            function.handle.pointer,
                                            addr errorMsg)

    if handle == nil:
      let msg = if errorMsg != nil:
        let s = $errorMsg
        nmc_free_string(errorMsg)
        s
      else:
        "Failed to create pipeline"
      return err[MetalComputePipeline](ekPipeline, "Pipeline creation failed: " & msg)

    let maxThreads = nmc_pipeline_max_threads_per_threadgroup(handle)
    let threadWidth = nmc_pipeline_thread_execution_width(handle)
    let staticMem = nmc_pipeline_static_threadgroup_memory_length(handle)

    result = ok(MetalComputePipeline(
      handle: MTLComputePipelineStateRef(handle),
      function: function,
      device: device,
      maxThreadsPerThreadgroup: maxThreads,
      threadExecutionWidth: threadWidth,
      staticThreadgroupMemoryLength: staticMem,
      valid: true
    ))
  else:
    result = err[MetalComputePipeline](ekPlatform, "Metal not available on this platform")

proc release*(pipeline: var MetalComputePipeline) =
  ## Release a compute pipeline state
  when defined(macosx):
    if pipeline.handle.pointer != nil:
      nmc_release_pipeline_state(pipeline.handle.pointer)
  pipeline.valid = false
  pipeline.handle = MTLComputePipelineStateRef(nil)

# ========== Convenience Functions ==========

proc compileAndCreatePipeline*(device: MetalDevice, source: string,
                                functionName: string): NMCResult[MetalComputePipeline] =
  ## Compile shader source and create a pipeline in one call
  ## Note: This does not manage library/function lifecycle - they will be released
  ## when the pipeline is released
  when defined(macosx):
    let libraryResult = device.compileShader(source)
    if not libraryResult.isOk:
      return err[MetalComputePipeline](libraryResult.error)

    var library = libraryResult.get

    let functionResult = library.getFunction(functionName)
    if not functionResult.isOk:
      library.release()
      return err[MetalComputePipeline](functionResult.error)

    var function = functionResult.get

    let pipelineResult = device.newComputePipeline(function)
    if not pipelineResult.isOk:
      function.release()
      library.release()
      return err[MetalComputePipeline](pipelineResult.error)

    # Note: library and function are referenced by pipeline
    # They should ideally be managed separately for proper cleanup
    result = pipelineResult
  else:
    result = err[MetalComputePipeline](ekPlatform, "Metal not available on this platform")

# ========== String Representations ==========

proc `$`*(library: MetalLibrary): string =
  if not library.valid:
    return "MetalLibrary(invalid)"
  let functions = library.functionNames()
  result = fmt"MetalLibrary(functions: {functions.len})"

proc `$`*(function: MetalFunction): string =
  if not function.valid:
    return "MetalFunction(invalid)"
  result = fmt"MetalFunction(name: {function.name})"

proc `$`*(pipeline: MetalComputePipeline): string =
  if not pipeline.valid:
    return "MetalComputePipeline(invalid)"
  result = fmt"MetalComputePipeline(maxThreads: {pipeline.maxThreadsPerThreadgroup}, " &
           fmt"executionWidth: {pipeline.threadExecutionWidth})"

# ========== Common Shader Source ==========

const VectorAddShader* = """
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    result[id] = a[id] + b[id];
}
"""

const VectorMultiplyShader* = """
#include <metal_stdlib>
using namespace metal;

kernel void vector_multiply(device const float* a [[buffer(0)]],
                            device const float* b [[buffer(1)]],
                            device float* result [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    result[id] = a[id] * b[id];
}
"""

const ScalarMultiplyShader* = """
#include <metal_stdlib>
using namespace metal;

kernel void scalar_multiply(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant float& scalar [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    output[id] = input[id] * scalar;
}
"""

const MatrixMultiplyShader* = """
#include <metal_stdlib>
using namespace metal;

kernel void matrix_multiply(device const float* A [[buffer(0)]],
                            device const float* B [[buffer(1)]],
                            device float* C [[buffer(2)]],
                            constant uint& M [[buffer(3)]],
                            constant uint& N [[buffer(4)]],
                            constant uint& K [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
"""

# ========== Test ==========

when isMainModule:
  echo "=== Metal Shader Test ==="
  echo ""

  let deviceResult = getDefaultDevice()
  if not deviceResult.isOk:
    echo "Error: ", deviceResult.error
    quit(1)

  let device = deviceResult.get
  echo "Device: ", device.info.name
  echo ""

  # Test shader compilation
  echo "Compiling vector_add shader..."
  let libraryResult = device.compileShader(VectorAddShader)
  if libraryResult.isOk:
    var library = libraryResult.get
    echo "Library: ", library
    echo "Function names: ", library.functionNames()
    echo ""

    # Get the function
    let functionResult = library.getFunction("vector_add")
    if functionResult.isOk:
      var function = functionResult.get
      echo "Function: ", function
      echo ""

      # Create pipeline
      let pipelineResult = device.newComputePipeline(function)
      if pipelineResult.isOk:
        var pipeline = pipelineResult.get
        echo "Pipeline: ", pipeline
        echo "  Max threads per threadgroup: ", pipeline.maxThreadsPerThreadgroup
        echo "  Thread execution width: ", pipeline.threadExecutionWidth
        echo "  Static threadgroup memory: ", pipeline.staticThreadgroupMemoryLength
        pipeline.release()
      else:
        echo "Pipeline error: ", pipelineResult.error

      function.release()
    else:
      echo "Function error: ", functionResult.error

    library.release()
  else:
    echo "Compilation error: ", libraryResult.error

  echo ""
  echo "✅ Metal shader test complete"
