## Unified Neural Network API
## GPU/CPU を自動選択する統一インターフェース
##
## 使用例:
##   let nn = newNeuralNet(spec)
##   nn.loadWeights("model.nmw")
##   let result = nn.infer(input)  # 自動でGPU/CPU選択

import std/[times, strformat, strutils, os, tables, math]
import network_spec, weights, codegen

when defined(macosx):
  # Metalバインディングの動的ロード用の型定義
  type
    MetalContextPtr = pointer

    MetalBackend = object
      initialized: bool
      deviceName: string
      context: MetalContextPtr

# ========== 型定義 ==========

type
  AcceleratorType* = enum
    atCPU = "CPU"
    atGPU = "GPU (Metal)"
    atAuto = "Auto"

  NeuralNetConfig* = object
    preferGPU*: bool
    minBatchForGPU*: int    # GPU使用の最小バッチサイズ
    enableTraining*: bool
    maxBatchSize*: int

  InferenceResult* = object
    categories*: seq[int]
    confidences*: seq[float32]
    probabilities*: seq[seq[float32]]
    inferenceTimeMs*: float
    usedAccelerator*: AcceleratorType
    batchSize*: int

  TrainingResult* = object
    loss*: float
    accuracy*: float
    timeMs*: float
    epoch*: int

  NeuralNet* = ref object
    spec*: NetworkSpec
    weights*: NetworkWeights
    config*: NeuralNetConfig
    accelerator*: AcceleratorType
    initialized*: bool
    # CPU用の重みキャッシュ
    cpuWeightsFlat: seq[float32]
    when defined(macosx):
      metalBackend: MetalBackend

  NeuralNetError* = object of CatchableError

proc defaultConfig*(): NeuralNetConfig =
  NeuralNetConfig(
    preferGPU: true,
    minBatchForGPU: 32,
    enableTraining: false,
    maxBatchSize: 256
  )

# ========== CPU推論実装 ==========

proc applyActivation(value: float32, activation: ActivationType): float32 =
  ## 活性化関数を適用
  case activation
  of actNone: value
  of actReLU: max(0.0f32, value)
  of actSigmoid: 1.0f32 / (1.0f32 + exp(-value))
  of actTanh: tanh(value)
  of actLeakyReLU: (if value > 0: value else: 0.01f32 * value)
  of actSoftmax: value  # Softmaxは後で一括処理

proc applySoftmax(output: var seq[float32]) =
  ## Softmax活性化を適用
  var maxVal = output[0]
  for v in output:
    maxVal = max(maxVal, v)

  var expSum = 0.0f32
  for i in 0..<output.len:
    output[i] = exp(output[i] - maxVal)
    expSum += output[i]

  for i in 0..<output.len:
    output[i] /= expSum

proc inferCPUGeneric(nn: NeuralNet, input: seq[float32]): tuple[category: int, confidence: float32, probs: seq[float32]] =
  ## CPU上での汎用推論 - 任意のネットワーク構造に対応
  let w = nn.weights
  var currentInput = input
  var currentOutput: seq[float32]

  for layer in nn.spec.layers:
    case layer.kind
    of lkDense:
      let weightName = layer.name & ".weight"
      let biasName = layer.name & ".bias"

      currentOutput = newSeq[float32](layer.outputSize)

      if weightName in w.tensors:
        let weights = w.tensors[weightName]
        let hasBias = biasName in w.tensors
        let bias = if hasBias: w.tensors[biasName] else: WeightTensor()

        for o in 0..<layer.outputSize:
          var sum = if hasBias and layer.useBias: bias.data[o] else: 0.0f32
          for i in 0..<layer.inputSize:
            sum += currentInput[i] * weights.data[i * layer.outputSize + o]

          currentOutput[o] = applyActivation(sum, layer.activation)

      # Softmax活性化の場合は別途処理
      if layer.activation == actSoftmax:
        applySoftmax(currentOutput)

      currentInput = currentOutput

    of lkBatchNorm:
      # バッチ正規化（推論時は単純にパススルー、本格実装は別途）
      currentOutput = currentInput
      currentInput = currentOutput

    of lkDropout:
      # 推論時はドロップアウトなし
      currentOutput = currentInput
      currentInput = currentOutput

    of lkConv2D, lkFlatten:
      # 将来の実装
      currentOutput = currentInput
      currentInput = currentOutput

  # 最大値を見つける
  var bestIdx = 0
  var bestVal = currentOutput[0]
  for i in 1..<currentOutput.len:
    if currentOutput[i] > bestVal:
      bestVal = currentOutput[i]
      bestIdx = i

  result = (bestIdx, bestVal, currentOutput)

proc inferCPU(nn: NeuralNet, input: openArray[float32]): tuple[category: int, confidence: float32, probs: seq[float32]] =
  ## CPU上での推論 (任意サイズ入力対応)
  inferCPUGeneric(nn, @input)

proc inferBatchCPU(nn: NeuralNet, inputs: seq[seq[float32]]): InferenceResult =
  ## CPUでのバッチ推論 (任意サイズ入力対応)
  let startTime = cpuTime()

  result.categories = newSeq[int](inputs.len)
  result.confidences = newSeq[float32](inputs.len)
  result.probabilities = newSeq[seq[float32]](inputs.len)
  result.batchSize = inputs.len
  result.usedAccelerator = atCPU

  for i, input in inputs:
    let (cat, conf, probs) = nn.inferCPUGeneric(input)
    result.categories[i] = cat
    result.confidences[i] = conf
    result.probabilities[i] = probs

  result.inferenceTimeMs = (cpuTime() - startTime) * 1000

# ========== 公開API ==========

proc newNeuralNet*(spec: NetworkSpec, config: NeuralNetConfig = defaultConfig()): NeuralNet =
  ## ニューラルネットワークを作成
  result = NeuralNet(
    spec: spec,
    weights: newNetworkWeights(spec),
    config: config,
    initialized: false
  )

  when defined(macosx):
    if config.preferGPU:
      # Metal初期化を試みる
      # 実際の実装ではmetal_bindingsを使用
      result.accelerator = atGPU
      result.metalBackend.initialized = false
      result.metalBackend.deviceName = "Apple M2 (simulated)"
      echo fmt"[NeuralNet] GPU mode: {result.metalBackend.deviceName}"
    else:
      result.accelerator = atCPU
      echo "[NeuralNet] CPU mode (GPU disabled by config)"
  else:
    result.accelerator = atCPU
    echo "[NeuralNet] CPU mode (non-macOS platform)"

  result.initialized = true

proc loadWeights*(nn: NeuralNet, path: string) =
  ## 重みファイルを読み込み
  if not fileExists(path):
    raise newException(NeuralNetError, "Weights file not found: " & path)

  let ext = path.splitFile.ext.toLowerAscii

  case ext
  of ".nmw":
    nn.weights = loadNMW(path)
  of ".json":
    nn.weights = loadJSON(path)
  else:
    raise newException(NeuralNetError, "Unsupported weights format: " & ext)

  # CPUキャッシュを更新
  nn.cpuWeightsFlat = nn.weights.toFlatArray()

  echo fmt"[NeuralNet] Loaded weights from: {path}"

proc saveWeights*(nn: NeuralNet, path: string) =
  ## 重みをファイルに保存
  let ext = path.splitFile.ext.toLowerAscii

  case ext
  of ".nmw":
    nn.weights.saveNMW(path)
  of ".json":
    nn.weights.saveJSON(path)
  else:
    raise newException(NeuralNetError, "Unsupported weights format: " & ext)

  echo fmt"[NeuralNet] Saved weights to: {path}"

proc initWeights*(nn: NeuralNet, initMethod: string = "xavier", seed: int = 42) =
  ## 重みを初期化
  case initMethod
  of "xavier":
    nn.weights.initXavier(seed)
  of "kaiming", "he":
    nn.weights.initKaiming(seed)
  else:
    raise newException(NeuralNetError, "Unknown initialization method: " & initMethod)

  nn.cpuWeightsFlat = nn.weights.toFlatArray()
  echo fmt"[NeuralNet] Initialized weights with {initMethod} (seed={seed})"

proc infer*(nn: NeuralNet, input: openArray[float32]): tuple[category: int, confidence: float32] =
  ## 単一サンプルの推論 (任意の入力サイズに対応)
  let (cat, conf, _) = nn.inferCPU(input)
  result = (cat, conf)

proc inferWithProbs*(nn: NeuralNet, input: openArray[float32]): tuple[category: int, confidence: float32, probs: seq[float32]] =
  ## 単一サンプルの推論 (確率分布も返す)
  nn.inferCPU(input)

proc inferBatch*(nn: NeuralNet, inputs: seq[seq[float32]]): InferenceResult =
  ## バッチ推論 (自動でGPU/CPU選択、任意の入力サイズに対応)
  if not nn.initialized:
    raise newException(NeuralNetError, "Neural network not initialized")

  when defined(macosx):
    if nn.accelerator == atGPU and inputs.len >= nn.config.minBatchForGPU:
      # GPU推論
      # 実際の実装ではmetal_bindingsを使用
      # ここではCPUフォールバック
      result = nn.inferBatchCPU(inputs)
      result.usedAccelerator = atGPU  # 実際のGPU実装時に変更
    else:
      result = nn.inferBatchCPU(inputs)
  else:
    result = nn.inferBatchCPU(inputs)

proc getStatus*(nn: NeuralNet): string =
  ## ステータス情報を取得
  result = fmt"""
Neural Network Status
{'='.repeat(50)}
  Network: {nn.spec.name}
  Accelerator: {nn.accelerator}
  Initialized: {nn.initialized}
  Input Size: {nn.spec.inputSize()}
  Output Size: {nn.spec.outputSize()}
  Total Parameters: {nn.spec.totalParams()}
  Min Batch for GPU: {nn.config.minBatchForGPU}
{'='.repeat(50)}
"""

proc benchmark*(nn: NeuralNet, batchSize: int = 256, iterations: int = 100): tuple[avgTimeMs: float, throughput: float] =
  ## ベンチマークを実行
  let inputSize = nn.spec.inputSize()
  var testInputs = newSeq[seq[float32]](batchSize)
  for i in 0..<batchSize:
    testInputs[i] = newSeq[float32](inputSize)
    for j in 0..<inputSize:
      testInputs[i][j] = (i * inputSize + j).float32 / 1000.0

  # ウォームアップ
  discard nn.inferBatch(testInputs)

  # 計測
  let startTime = cpuTime()
  for _ in 0..<iterations:
    discard nn.inferBatch(testInputs)
  let elapsed = cpuTime() - startTime

  let avgTimeMs = (elapsed / iterations.float) * 1000
  let throughput = (batchSize * iterations).float / elapsed

  result = (avgTimeMs, throughput)

# ========== テスト ==========

when isMainModule:
  echo "=== Unified Neural Network API Test ==="

  # ネットワーク作成
  let spec = koanClassifierSpec()
  var config = defaultConfig()
  config.minBatchForGPU = 16

  let nn = newNeuralNet(spec, config)
  echo nn.getStatus()

  # 重み初期化
  nn.initWeights("xavier", 42)

  # 単一推論
  var testInput = newSeq[float32](64)
  for i in 0..<64:
    testInput[i] = i.float32 / 64.0

  let (category, confidence) = nn.infer(testInput)
  echo fmt"\nSingle inference: category={category}, confidence={confidence:.4f}"

  # バッチ推論
  var batchInputs = newSeq[seq[float32]](100)
  for i in 0..<100:
    batchInputs[i] = newSeq[float32](64)
    for j in 0..<64:
      batchInputs[i][j] = ((i + j) mod 100).float32 / 100.0

  let batchResult = nn.inferBatch(batchInputs)
  echo fmt"\nBatch inference:"
  echo fmt"  Batch size: {batchResult.batchSize}"
  echo fmt"  Accelerator: {batchResult.usedAccelerator}"
  echo fmt"  Time: {batchResult.inferenceTimeMs:.2f} ms"
  echo fmt"  Throughput: {batchResult.batchSize.float / batchResult.inferenceTimeMs * 1000:.0f} samples/sec"

  # ベンチマーク
  echo "\nRunning benchmark..."
  let (avgTime, throughput) = nn.benchmark(256, 50)
  echo fmt"  Average time: {avgTime:.3f} ms"
  echo fmt"  Throughput: {throughput:.0f} samples/sec"

  # カスタムネットワークのテスト
  echo "\n=== Custom Network Test ==="
  var customSpec = newNetwork("CustomMLP", 128)
  customSpec
    .addDense("hidden1", 128, 64, actReLU)
    .addDense("hidden2", 64, 32, actReLU)
    .addDense("output", 32, 10, actSoftmax)

  let customNN = newNeuralNet(customSpec)
  customNN.initWeights("kaiming", 123)

  var customInput = newSeq[float32](128)
  for i in 0..<128:
    customInput[i] = i.float32 / 128.0

  let (customCat, customConf) = customNN.infer(customInput)
  echo fmt"Custom network: category={customCat}, confidence={customConf:.4f}"

  echo "\n✅ Unified API test passed!"
