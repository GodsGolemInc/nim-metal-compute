## Metal Neural Network GPU Inference
## v0.0.7: GPU-accelerated neural network operations
##
## This module provides:
## - Dense layer (fully connected)
## - Activation functions (ReLU, Sigmoid, Tanh, Softmax)
## - High-level inference API

import std/[strformat, times, math, sequtils]
import ./errors
import ./metal_device
import ./metal_buffer
import ./metal_command
import ./metal_shader
import ./metal_wrapper

# ========== Neural Network Shaders ==========

const DenseLayerShader* = """
#include <metal_stdlib>
using namespace metal;

// Dense layer: output = input * weights + bias
// input: [batch, inputSize]
// weights: [inputSize, outputSize]
// bias: [outputSize]
// output: [batch, outputSize]
kernel void dense_layer(device const float* input [[buffer(0)]],
                        device const float* weights [[buffer(1)]],
                        device const float* bias [[buffer(2)]],
                        device float* output [[buffer(3)]],
                        constant uint& batchSize [[buffer(4)]],
                        constant uint& inputSize [[buffer(5)]],
                        constant uint& outputSize [[buffer(6)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint batch = gid.y;
    uint outIdx = gid.x;

    if (batch >= batchSize || outIdx >= outputSize) return;

    float sum = bias[outIdx];
    for (uint i = 0; i < inputSize; i++) {
        sum += input[batch * inputSize + i] * weights[i * outputSize + outIdx];
    }
    output[batch * outputSize + outIdx] = sum;
}
"""

const ReLUShader* = """
#include <metal_stdlib>
using namespace metal;

kernel void relu(device const float* input [[buffer(0)]],
                 device float* output [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    output[id] = max(0.0f, input[id]);
}

kernel void relu_inplace(device float* data [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
    data[id] = max(0.0f, data[id]);
}
"""

const SigmoidShader* = """
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(device const float* input [[buffer(0)]],
                    device float* output [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {
    output[id] = 1.0f / (1.0f + exp(-input[id]));
}

kernel void sigmoid_inplace(device float* data [[buffer(0)]],
                            uint id [[thread_position_in_grid]]) {
    data[id] = 1.0f / (1.0f + exp(-data[id]));
}
"""

const TanhShader* = """
#include <metal_stdlib>
using namespace metal;

kernel void tanh_activation(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    output[id] = tanh(input[id]);
}

kernel void tanh_inplace(device float* data [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
    data[id] = tanh(data[id]);
}
"""

const SoftmaxShader* = """
#include <metal_stdlib>
using namespace metal;

// Softmax for a single row (batch element)
// Each thread handles one batch element
kernel void softmax(device const float* input [[buffer(0)]],
                    device float* output [[buffer(1)]],
                    constant uint& size [[buffer(2)]],
                    uint batchIdx [[thread_position_in_grid]]) {
    uint offset = batchIdx * size;

    // Find max for numerical stability
    float maxVal = input[offset];
    for (uint i = 1; i < size; i++) {
        maxVal = max(maxVal, input[offset + i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        float expVal = exp(input[offset + i] - maxVal);
        output[offset + i] = expVal;
        sum += expVal;
    }

    // Normalize
    for (uint i = 0; i < size; i++) {
        output[offset + i] /= sum;
    }
}
"""

# ========== Layer Types ==========

type
  ActivationType* = enum
    atNone = "none"
    atReLU = "relu"
    atSigmoid = "sigmoid"
    atTanh = "tanh"
    atSoftmax = "softmax"

  DenseLayerGPU* = object
    ## GPU-backed dense layer
    inputSize*: int
    outputSize*: int
    activation*: ActivationType
    weightsBuffer*: MetalBuffer
    biasBuffer*: MetalBuffer
    valid*: bool

  NeuralNetworkGPU* = object
    ## GPU-backed neural network
    device*: MetalDevice
    layers*: seq[DenseLayerGPU]
    densePipeline*: MetalComputePipeline
    reluPipeline*: MetalComputePipeline
    sigmoidPipeline*: MetalComputePipeline
    tanhPipeline*: MetalComputePipeline
    softmaxPipeline*: MetalComputePipeline
    valid*: bool

# ========== Layer Creation ==========

proc newDenseLayerGPU*(device: MetalDevice, inputSize, outputSize: int,
                       weights: seq[float32], bias: seq[float32],
                       activation: ActivationType = atNone): NMCResult[DenseLayerGPU] =
  ## Create a GPU-backed dense layer with provided weights
  when defined(macosx):
    if weights.len != inputSize * outputSize:
      return err[DenseLayerGPU](ekDimensionMismatch,
        fmt"Weights size mismatch: expected {inputSize * outputSize}, got {weights.len}")

    if bias.len != outputSize:
      return err[DenseLayerGPU](ekDimensionMismatch,
        fmt"Bias size mismatch: expected {outputSize}, got {bias.len}")

    let wBuf = device.newBuffer(weights)
    if not wBuf.isOk:
      return err[DenseLayerGPU](wBuf.error)
    var weightsBuffer = wBuf.get

    let bBuf = device.newBuffer(bias)
    if not bBuf.isOk:
      weightsBuffer.release()
      return err[DenseLayerGPU](bBuf.error)
    var biasBuffer = bBuf.get

    result = ok(DenseLayerGPU(
      inputSize: inputSize,
      outputSize: outputSize,
      activation: activation,
      weightsBuffer: weightsBuffer,
      biasBuffer: biasBuffer,
      valid: true
    ))
  else:
    result = err[DenseLayerGPU](ekPlatform, "Metal not available")

proc release*(layer: var DenseLayerGPU) =
  ## Release GPU resources
  if layer.valid:
    layer.weightsBuffer.release()
    layer.biasBuffer.release()
    layer.valid = false

# ========== Neural Network Creation ==========

proc newNeuralNetworkGPU*(device: MetalDevice): NMCResult[NeuralNetworkGPU] =
  ## Create a new GPU-backed neural network
  when defined(macosx):
    # Compile all necessary shaders
    let densePipe = device.compileAndCreatePipeline(DenseLayerShader, "dense_layer")
    if not densePipe.isOk:
      return err[NeuralNetworkGPU](densePipe.error)

    let reluPipe = device.compileAndCreatePipeline(ReLUShader, "relu")
    if not reluPipe.isOk:
      return err[NeuralNetworkGPU](reluPipe.error)

    let sigmoidPipe = device.compileAndCreatePipeline(SigmoidShader, "sigmoid")
    if not sigmoidPipe.isOk:
      return err[NeuralNetworkGPU](sigmoidPipe.error)

    let tanhPipe = device.compileAndCreatePipeline(TanhShader, "tanh_activation")
    if not tanhPipe.isOk:
      return err[NeuralNetworkGPU](tanhPipe.error)

    let softmaxPipe = device.compileAndCreatePipeline(SoftmaxShader, "softmax")
    if not softmaxPipe.isOk:
      return err[NeuralNetworkGPU](softmaxPipe.error)

    result = ok(NeuralNetworkGPU(
      device: device,
      layers: @[],
      densePipeline: densePipe.get,
      reluPipeline: reluPipe.get,
      sigmoidPipeline: sigmoidPipe.get,
      tanhPipeline: tanhPipe.get,
      softmaxPipeline: softmaxPipe.get,
      valid: true
    ))
  else:
    result = err[NeuralNetworkGPU](ekPlatform, "Metal not available")

proc addLayer*(nn: var NeuralNetworkGPU, layer: DenseLayerGPU) =
  ## Add a layer to the network
  nn.layers.add(layer)

proc release*(nn: var NeuralNetworkGPU) =
  ## Release all GPU resources
  for layer in nn.layers.mitems:
    layer.release()
  nn.layers.setLen(0)
  nn.densePipeline.release()
  nn.reluPipeline.release()
  nn.sigmoidPipeline.release()
  nn.tanhPipeline.release()
  nn.softmaxPipeline.release()
  nn.valid = false

# ========== Inference ==========

proc forward*(nn: NeuralNetworkGPU, input: seq[float32],
              batchSize: int = 1): NMCResult[seq[float32]] =
  ## Run forward pass through the network
  when defined(macosx):
    if nn.layers.len == 0:
      return err[seq[float32]](ekNotInitialized, "Network has no layers")

    let inputSize = nn.layers[0].inputSize
    if input.len != inputSize * batchSize:
      return err[seq[float32]](ekDimensionMismatch,
        fmt"Input size mismatch: expected {inputSize * batchSize}, got {input.len}")

    # Create command queue
    let queueResult = nn.device.newCommandQueue()
    if not queueResult.isOk:
      return err[seq[float32]](queueResult.error)
    var queue = queueResult.get

    # Create input buffer
    let inputBufResult = nn.device.newBuffer(input)
    if not inputBufResult.isOk:
      queue.release()
      return err[seq[float32]](inputBufResult.error)
    var currentBuffer = inputBufResult.get
    var currentSize = inputSize

    # Process each layer
    for layerIdx, layer in nn.layers:
      # Create output buffer
      let outputSize = layer.outputSize * batchSize * sizeof(float32)
      let outputBufResult = nn.device.newBuffer(outputSize, smShared)
      if not outputBufResult.isOk:
        currentBuffer.release()
        queue.release()
        return err[seq[float32]](outputBufResult.error)
      var outputBuffer = outputBufResult.get

      # Create command buffer
      let cmdBufResult = queue.newCommandBuffer()
      if not cmdBufResult.isOk:
        outputBuffer.release()
        currentBuffer.release()
        queue.release()
        return err[seq[float32]](cmdBufResult.error)
      var cmdBuffer = cmdBufResult.get

      # Dense layer
      block:
        let encoderResult = cmdBuffer.newComputeEncoder()
        if not encoderResult.isOk:
          cmdBuffer.release()
          outputBuffer.release()
          currentBuffer.release()
          queue.release()
          return err[seq[float32]](encoderResult.error)
        var encoder = encoderResult.get

        nmc_encoder_set_pipeline_state(encoder.handle.pointer, nn.densePipeline.handle.pointer)
        nmc_encoder_set_buffer(encoder.handle.pointer, currentBuffer.handle.pointer, 0, 0)
        nmc_encoder_set_buffer(encoder.handle.pointer, layer.weightsBuffer.handle.pointer, 0, 1)
        nmc_encoder_set_buffer(encoder.handle.pointer, layer.biasBuffer.handle.pointer, 0, 2)
        nmc_encoder_set_buffer(encoder.handle.pointer, outputBuffer.handle.pointer, 0, 3)

        var batchVal = batchSize.uint32
        var inSizeVal = layer.inputSize.uint32
        var outSizeVal = layer.outputSize.uint32
        nmc_encoder_set_bytes(encoder.handle.pointer, addr batchVal, sizeof(uint32).uint64, 4)
        nmc_encoder_set_bytes(encoder.handle.pointer, addr inSizeVal, sizeof(uint32).uint64, 5)
        nmc_encoder_set_bytes(encoder.handle.pointer, addr outSizeVal, sizeof(uint32).uint64, 6)

        # Thread configuration
        let threadGroupWidth = min(16, layer.outputSize)
        let threadGroupHeight = min(16, batchSize)
        let gridWidth = (layer.outputSize + threadGroupWidth - 1) div threadGroupWidth
        let gridHeight = (batchSize + threadGroupHeight - 1) div threadGroupHeight

        nmc_encoder_dispatch_threadgroups(
          encoder.handle.pointer,
          gridWidth.uint64, gridHeight.uint64, 1,
          threadGroupWidth.uint64, threadGroupHeight.uint64, 1
        )

        discard encoder.endEncoding()

      # Apply activation (if any)
      if layer.activation != atNone:
        let activationEncoderResult = cmdBuffer.newComputeEncoder()
        if not activationEncoderResult.isOk:
          cmdBuffer.release()
          outputBuffer.release()
          currentBuffer.release()
          queue.release()
          return err[seq[float32]](activationEncoderResult.error)
        var actEncoder = activationEncoderResult.get

        let totalElements = layer.outputSize * batchSize
        let pipeline = case layer.activation
          of atReLU: nn.reluPipeline
          of atSigmoid: nn.sigmoidPipeline
          of atTanh: nn.tanhPipeline
          of atSoftmax: nn.softmaxPipeline
          else: nn.reluPipeline  # Fallback

        nmc_encoder_set_pipeline_state(actEncoder.handle.pointer, pipeline.handle.pointer)

        if layer.activation == atSoftmax:
          # Softmax processes per batch
          nmc_encoder_set_buffer(actEncoder.handle.pointer, outputBuffer.handle.pointer, 0, 0)
          nmc_encoder_set_buffer(actEncoder.handle.pointer, outputBuffer.handle.pointer, 0, 1)
          var sizeVal = layer.outputSize.uint32
          nmc_encoder_set_bytes(actEncoder.handle.pointer, addr sizeVal, sizeof(uint32).uint64, 2)

          let threadGroups = batchSize
          nmc_encoder_dispatch_threadgroups(
            actEncoder.handle.pointer,
            threadGroups.uint64, 1, 1,
            1, 1, 1
          )
        else:
          # Element-wise activations
          nmc_encoder_set_buffer(actEncoder.handle.pointer, outputBuffer.handle.pointer, 0, 0)
          nmc_encoder_set_buffer(actEncoder.handle.pointer, outputBuffer.handle.pointer, 0, 1)

          let threadGroupSize = min(256, totalElements)
          let threadGroups = (totalElements + threadGroupSize - 1) div threadGroupSize

          nmc_encoder_dispatch_threadgroups(
            actEncoder.handle.pointer,
            threadGroups.uint64, 1, 1,
            threadGroupSize.uint64, 1, 1
          )

        discard actEncoder.endEncoding()

      # Execute
      discard cmdBuffer.commit()
      discard cmdBuffer.waitUntilCompleted()
      cmdBuffer.release()

      # Swap buffers
      currentBuffer.release()
      currentBuffer = outputBuffer
      currentSize = layer.outputSize

    # Read final output
    let finalSize = nn.layers[^1].outputSize * batchSize
    var resultData = newSeq[float32](finalSize)
    discard currentBuffer.read(resultData)

    currentBuffer.release()
    queue.release()

    result = ok(resultData)
  else:
    result = err[seq[float32]](ekPlatform, "Metal not available")

# ========== CPU Reference ==========

proc reluCPU*(x: float32): float32 = max(0.0f, x)
proc sigmoidCPU*(x: float32): float32 = 1.0f / (1.0f + exp(-x))
proc tanhCPU*(x: float32): float32 = tanh(x)

proc softmaxCPU*(input: seq[float32]): seq[float32] =
  let maxVal = input.max
  var expSum: float32 = 0.0
  result = newSeq[float32](input.len)
  for i, x in input:
    result[i] = exp(x - maxVal)
    expSum += result[i]
  for i in 0..<result.len:
    result[i] /= expSum

proc forwardCPU*(weights: seq[float32], bias: seq[float32],
                 input: seq[float32], inputSize, outputSize: int,
                 activation: ActivationType = atNone): seq[float32] =
  ## CPU dense layer forward pass
  result = newSeq[float32](outputSize)
  for j in 0..<outputSize:
    var sum = bias[j]
    for i in 0..<inputSize:
      sum += input[i] * weights[i * outputSize + j]

    case activation
    of atReLU: result[j] = reluCPU(sum)
    of atSigmoid: result[j] = sigmoidCPU(sum)
    of atTanh: result[j] = tanhCPU(sum)
    else: result[j] = sum

  if activation == atSoftmax:
    result = softmaxCPU(result)

# ========== Utilities ==========

proc `$`*(layer: DenseLayerGPU): string =
  fmt"DenseLayerGPU({layer.inputSize} -> {layer.outputSize}, {layer.activation})"

proc `$`*(nn: NeuralNetworkGPU): string =
  if nn.layers.len == 0:
    return "NeuralNetworkGPU(empty)"
  result = fmt"NeuralNetworkGPU({nn.layers.len} layers)"

proc verify*(expected, actual: seq[float32], tolerance: float32 = 1e-4): bool =
  if expected.len != actual.len:
    return false
  for i in 0..<expected.len:
    if abs(expected[i] - actual[i]) > tolerance:
      return false
  result = true

# ========== Test ==========

when isMainModule:
  import std/random
  randomize()

  echo "=== Metal Neural Network GPU Test ==="
  echo ""

  let deviceResult = getDefaultDevice()
  if not deviceResult.isOk:
    echo "Error: ", deviceResult.error
    quit(1)

  let device = deviceResult.get
  echo "Device: ", device.info.name
  echo ""

  # Create a simple network: 784 -> 256 -> 128 -> 10
  echo "--- Creating Network: 784 -> 256 (ReLU) -> 128 (ReLU) -> 10 (Softmax) ---"
  echo ""

  let nnResult = device.newNeuralNetworkGPU()
  if not nnResult.isOk:
    echo "Error creating network: ", nnResult.error
    quit(1)

  var nn = nnResult.get

  # Create random weights (Xavier initialization)
  proc xavierInit(inSize, outSize: int): seq[float32] =
    let scale = sqrt(2.0 / float(inSize + outSize))
    result = newSeq[float32](inSize * outSize)
    for i in 0..<result.len:
      result[i] = (rand(2.0) - 1.0).float32 * scale.float32

  proc zeroInit(size: int): seq[float32] =
    newSeq[float32](size)

  # Layer 1: 784 -> 256 (ReLU)
  let layer1Weights = xavierInit(784, 256)
  let layer1Bias = zeroInit(256)
  let layer1Result = device.newDenseLayerGPU(784, 256, layer1Weights, layer1Bias, atReLU)
  if layer1Result.isOk:
    nn.addLayer(layer1Result.get)
    echo "Layer 1: 784 -> 256 (ReLU)"
  else:
    echo "Layer 1 error: ", layer1Result.error
    quit(1)

  # Layer 2: 256 -> 128 (ReLU)
  let layer2Weights = xavierInit(256, 128)
  let layer2Bias = zeroInit(128)
  let layer2Result = device.newDenseLayerGPU(256, 128, layer2Weights, layer2Bias, atReLU)
  if layer2Result.isOk:
    nn.addLayer(layer2Result.get)
    echo "Layer 2: 256 -> 128 (ReLU)"
  else:
    echo "Layer 2 error: ", layer2Result.error
    quit(1)

  # Layer 3: 128 -> 10 (Softmax)
  let layer3Weights = xavierInit(128, 10)
  let layer3Bias = zeroInit(10)
  let layer3Result = device.newDenseLayerGPU(128, 10, layer3Weights, layer3Bias, atSoftmax)
  if layer3Result.isOk:
    nn.addLayer(layer3Result.get)
    echo "Layer 3: 128 -> 10 (Softmax)"
  else:
    echo "Layer 3 error: ", layer3Result.error
    quit(1)

  echo ""

  # Test inference
  echo "--- Inference Test ---"

  # Create random input (simulating MNIST-like input)
  var input = newSeq[float32](784)
  for i in 0..<784:
    input[i] = rand(1.0).float32

  # GPU inference
  let gpuStart = cpuTime()
  let gpuResult = nn.forward(input)
  let gpuTime = cpuTime() - gpuStart

  if gpuResult.isOk:
    let output = gpuResult.get
    echo fmt"GPU inference time: {gpuTime * 1000:.2f} ms"
    echo fmt"Output shape: {output.len}"
    echo fmt"Output sum: {output.foldl(a + b, 0.0f):.4f} (should be ~1.0 for softmax)"

    # Find predicted class
    var maxIdx = 0
    var maxVal = output[0]
    for i in 1..<output.len:
      if output[i] > maxVal:
        maxVal = output[i]
        maxIdx = i
    echo fmt"Predicted class: {maxIdx} (confidence: {maxVal * 100:.1f}%)"
  else:
    echo "GPU inference error: ", gpuResult.error

  echo ""

  # Benchmark multiple inferences
  echo "--- Benchmark (100 inferences) ---"
  let benchStart = cpuTime()
  for i in 0..<100:
    discard nn.forward(input)
  let benchTime = cpuTime() - benchStart

  echo fmt"Total time: {benchTime * 1000:.2f} ms"
  echo fmt"Average per inference: {benchTime * 10:.2f} ms"
  echo fmt"Throughput: {100.0 / benchTime:.0f} inferences/sec"

  # Cleanup
  nn.release()

  echo ""
  echo "✅ Metal neural network GPU test complete"
