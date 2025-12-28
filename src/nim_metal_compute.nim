## nim-metal-compute
## Metal Compute Shader bindings for Nim
##
## 統一されたGPU/CPU ニューラルネットワークライブラリ
##
## 特徴:
##   - 宣言的なネットワーク定義 (DSL)
##   - Metal/CPU コード自動生成
##   - 自動アクセラレータ選択
##   - 共有重みフォーマット (.nmw)
##
## 使用例:
##   import nim_metal_compute
##
##   # ネットワーク定義
##   var spec = newNetwork("MyClassifier", 64)
##   spec
##     .addDense("hidden", 64, 128, actReLU)
##     .addDense("output", 128, 10, actSoftmax)
##
##   # 統一API
##   let nn = newNeuralNet(spec)
##   nn.initWeights("xavier")
##   let result = nn.infer(input)  # GPU/CPU自動選択
##
## コード生成:
##   # Metal シェーダーとNim CPUコードを生成
##   spec.generate(CodeGenOptions(outputDir: "generated"))

import nim_metal_compute/errors
import nim_metal_compute/network_spec
import nim_metal_compute/weights
import nim_metal_compute/codegen
import nim_metal_compute/unified_api
import nim_metal_compute/simd_inference
import nim_metal_compute/metal_device
import nim_metal_compute/metal_buffer
import nim_metal_compute/metal_command
import nim_metal_compute/metal_capabilities

# エラーハンドリング (v0.0.2+)
export errors.NMCErrorKind
export errors.NMCError
export errors.NMCResult
export errors.VoidResult
export errors.newError
export errors.ok
export errors.err
export errors.isOk
export errors.isErr
export errors.get
export errors.getOr
export errors.getError
export errors.map
export errors.flatMap
export errors.mapError
export errors.okVoid
export errors.errVoid
export errors.validatePositive
export errors.validateNonEmpty
export errors.validateRange

# ネットワーク定義
export network_spec.ActivationType
export network_spec.LayerKind
export network_spec.LayerSpec
export network_spec.NetworkSpec
export network_spec.NetworkError
export network_spec.newNetwork
export network_spec.addDense
export network_spec.addBatchNorm
export network_spec.addDropout
export network_spec.validate
export network_spec.toJson
export network_spec.fromJson
export network_spec.saveSpec
export network_spec.loadSpec
export network_spec.totalParams
export network_spec.summary
export network_spec.inputSize
export network_spec.outputSize
export network_spec.koanClassifierSpec
export network_spec.mlpClassifier
export network_spec.validateResult
export network_spec.validateLayer

# 重み管理
export weights.WeightTensor
export weights.NetworkWeights
export weights.WeightsError
export weights.newWeightTensor
export weights.newNetworkWeights
export weights.initXavier
export weights.initKaiming
export weights.saveNMW
export weights.loadNMW
export weights.saveJSON
export weights.loadJSON
export weights.toFlatArray
export weights.fromFlatArray
export weights.saveNMWResult
export weights.loadNMWResult

# コード生成
export codegen.CodeGenTarget
export codegen.CodeGenOptions
export codegen.defaultOptions
export codegen.generateMetalKernel
export codegen.generateNimCPU
export codegen.generate
export codegen.generateResult
export codegen.GenerateResultData

# 統一API
export unified_api.AcceleratorType
export unified_api.NeuralNetConfig
export unified_api.InferenceResult
export unified_api.TrainingResult
export unified_api.NeuralNet
export unified_api.NeuralNetError
export unified_api.defaultConfig
export unified_api.newNeuralNet
export unified_api.loadWeights
export unified_api.saveWeights
export unified_api.initWeights
export unified_api.infer
export unified_api.inferWithProbs
export unified_api.inferBatch
export unified_api.getStatus
export unified_api.benchmark

# SIMD最適化推論
export simd_inference.SIMDInferenceEngine
export simd_inference.newSIMDInferenceEngine
export simd_inference.initKaiming
export simd_inference.inferFast
export simd_inference.inferBatchFast
export simd_inference.inferBatch4

# 究極最適化推論（400万+ samples/sec シングルコア）
import nim_metal_compute/extreme_inference
export extreme_inference.ExtremeEngine
export extreme_inference.InputSize
export extreme_inference.HiddenSize
export extreme_inference.OutputSize
export extreme_inference.initWeights
export extreme_inference.setWeights
export extreme_inference.inferExtreme
export extreme_inference.inferExtremeWithConf

# 並列推論エンジン（1960万+ samples/sec マルチコア）
import nim_metal_compute/parallel_inference
export parallel_inference.MaxThreads
export parallel_inference.ParallelInferenceEngine
export parallel_inference.BatchInferenceResult
export parallel_inference.initParallelEngine
export parallel_inference.syncWeights
export parallel_inference.inferBatchParallel
export parallel_inference.inferBatch

# Metal API bindings (v0.0.3+) - temporarily disabled due to build issues
# export metal_device.MTLDeviceRef
# export metal_device.MTLCommandQueueRef
# export metal_device.DeviceInfo
# export metal_device.MetalDevice
# export metal_device.isMetalAvailable
# export metal_device.getDefaultDevice
# export metal_device.isAppleSilicon
#
# export metal_buffer.MTLBufferRef
# export metal_buffer.MTLStorageMode
# export metal_buffer.MTLResourceOptions
# export metal_buffer.MetalBuffer
# export metal_buffer.newBuffer
# export metal_buffer.newBufferWithData
# export metal_buffer.contents
# export metal_buffer.write
# export metal_buffer.read
# export metal_buffer.didModifyRange
# export metal_buffer.synchronize
#
# export metal_command.MTLCommandBufferRef
# export metal_command.MTLComputeCommandEncoderRef
# export metal_command.MTLComputePipelineStateRef
# export metal_command.MTLCommandBufferStatus
# export metal_command.MetalCommandQueue
# export metal_command.MetalCommandBuffer
# export metal_command.MetalComputeEncoder
# export metal_command.MTLSize
# export metal_command.mtlSize
# export metal_command.mtlSize1D
# export metal_command.mtlSize2D
# export metal_command.newCommandBuffer
# export metal_command.status
# export metal_command.commit
# export metal_command.waitUntilCompleted
# export metal_command.waitUntilScheduled
# export metal_command.newComputeEncoder
# export metal_command.setBuffer
# export metal_command.setBytes
# export metal_command.setComputePipelineState
# export metal_command.dispatchThreadgroups
# export metal_command.dispatchThreads
# export metal_command.endEncoding
#
# export metal_capabilities.MTLGPUFamily
# export metal_capabilities.ComputeCapabilities
# export metal_capabilities.MemoryCapabilities
# export metal_capabilities.DeviceCapabilities
# export metal_capabilities.detectGPUFamily
# export metal_capabilities.familyName
# export metal_capabilities.isAppleSiliconFamily
# export metal_capabilities.getComputeCapabilities
# export metal_capabilities.getMemoryCapabilities
# export metal_capabilities.getCapabilities
# export metal_capabilities.recommendedThreadgroupSize
# export metal_capabilities.isCapableFor

# バージョン情報
const
  NimMetalComputeVersion* = "0.0.3"
  NimMetalComputeAuthor* = "GodsGolemInc"

when isMainModule:
  echo "nim-metal-compute v" & NimMetalComputeVersion
  echo "Metal Compute Shader bindings for Nim"
  echo ""
  echo "Usage:"
  echo "  import nim_metal_compute"
  echo ""
  echo "  # Define network"
  echo "  var spec = newNetwork(\"MyNet\", 64)"
  echo "  spec.addDense(\"hidden\", 64, 64, actReLU)"
  echo "  spec.addDense(\"output\", 64, 10, actSoftmax)"
  echo ""
  echo "  # Create and use"
  echo "  let nn = newNeuralNet(spec)"
  echo "  nn.initWeights(\"xavier\")"
  echo "  let result = nn.infer(input)"
