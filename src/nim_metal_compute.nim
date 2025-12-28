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

import nim_metal_compute/network_spec
import nim_metal_compute/weights
import nim_metal_compute/codegen
import nim_metal_compute/unified_api
import nim_metal_compute/simd_inference

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

# コード生成
export codegen.CodeGenTarget
export codegen.CodeGenOptions
export codegen.defaultOptions
export codegen.generateMetalKernel
export codegen.generateNimCPU
export codegen.generate

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
export extreme_inference.inferExtreme
export extreme_inference.inferExtremeWithConf

# 並列推論エンジン（1960万+ samples/sec マルチコア）
import nim_metal_compute/parallel_inference
export parallel_inference.MaxThreads
export parallel_inference.ParallelInferenceEngine
export parallel_inference.BatchInferenceResult
export parallel_inference.initParallelEngine
export parallel_inference.inferBatchParallel
export parallel_inference.inferBatch

# バージョン情報
const
  NimMetalComputeVersion* = "0.1.0"
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
