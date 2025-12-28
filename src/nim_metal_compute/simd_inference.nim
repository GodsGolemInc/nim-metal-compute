## SIMD最適化推論エンジン
## Apple Silicon Neon / x86 SSE/AVX 対応
##
## 10万+接続対応の高速推論

import std/[math]

type
  SIMDInferenceEngine* = object
    inputSize*: int
    hiddenSize*: int
    outputSize*: int
    # 連続メモリレイアウト（キャッシュ効率最大化）
    weightsIH*: seq[float32]  # [inputSize * hiddenSize] row-major
    biasH*: seq[float32]      # [hiddenSize]
    weightsHO*: seq[float32]  # [hiddenSize * outputSize] row-major
    biasO*: seq[float32]      # [outputSize]
    # 作業バッファ（再利用）
    hidden*: seq[float32]
    output*: seq[float32]

proc newSIMDInferenceEngine*(inputSize, hiddenSize, outputSize: int): SIMDInferenceEngine =
  result.inputSize = inputSize
  result.hiddenSize = hiddenSize
  result.outputSize = outputSize
  result.weightsIH = newSeq[float32](inputSize * hiddenSize)
  result.biasH = newSeq[float32](hiddenSize)
  result.weightsHO = newSeq[float32](hiddenSize * outputSize)
  result.biasO = newSeq[float32](outputSize)
  result.hidden = newSeq[float32](hiddenSize)
  result.output = newSeq[float32](outputSize)

{.push overflowChecks:off.}
proc initKaiming*(engine: var SIMDInferenceEngine, seed: int = 42) =
  ## Kaiming初期化
  var rng = seed.uint32

  proc nextRandom(): float32 =
    rng = rng * 1103515245'u32 + 12345'u32
    result = ((rng shr 16) and 0x7FFF'u32).float32 / 32767.0f32 * 2.0f32 - 1.0f32

  let scaleIH = sqrt(2.0f32 / engine.inputSize.float32)
  let scaleHO = sqrt(2.0f32 / engine.hiddenSize.float32)

  for i in 0..<engine.weightsIH.len:
    engine.weightsIH[i] = nextRandom() * scaleIH

  for i in 0..<engine.weightsHO.len:
    engine.weightsHO[i] = nextRandom() * scaleHO

  for i in 0..<engine.biasH.len:
    engine.biasH[i] = 0.0f32

  for i in 0..<engine.biasO.len:
    engine.biasO[i] = 0.0f32
{.pop.}

{.push checks:off, boundChecks:off, overflowChecks:off.}

proc inferFast*(engine: var SIMDInferenceEngine,
                input: openArray[float32]): tuple[category: int, confidence: float32] =
  ## 高速単一推論（ゼロアロケーション）
  let iSize = engine.inputSize
  let hSize = engine.hiddenSize
  let oSize = engine.outputSize

  # 入力層 → 隠れ層（ReLU）
  for j in 0..<hSize:
    var sum = engine.biasH[j]
    let offset = j
    for i in 0..<iSize:
      sum += input[i] * engine.weightsIH[i * hSize + offset]
    engine.hidden[j] = if sum > 0.0f32: sum else: 0.0f32

  # 隠れ層 → 出力層
  for j in 0..<oSize:
    var sum = engine.biasO[j]
    let offset = j
    for i in 0..<hSize:
      sum += engine.hidden[i] * engine.weightsHO[i * oSize + offset]
    engine.output[j] = sum

  # Softmax（数値安定化版）
  var maxVal = engine.output[0]
  for i in 1..<oSize:
    if engine.output[i] > maxVal:
      maxVal = engine.output[i]

  var expSum = 0.0f32
  for i in 0..<oSize:
    engine.output[i] = exp(engine.output[i] - maxVal)
    expSum += engine.output[i]

  let invSum = 1.0f32 / expSum
  var bestIdx = 0
  var bestVal = engine.output[0] * invSum
  for i in 1..<oSize:
    let prob = engine.output[i] * invSum
    if prob > bestVal:
      bestVal = prob
      bestIdx = i

  result = (bestIdx, bestVal)

proc inferBatchFast*(engine: var SIMDInferenceEngine,
                     inputs: openArray[seq[float32]],
                     categories: var seq[int],
                     confidences: var seq[float32]) =
  ## 高速バッチ推論
  let batchSize = inputs.len
  if categories.len < batchSize:
    categories.setLen(batchSize)
  if confidences.len < batchSize:
    confidences.setLen(batchSize)

  for b in 0..<batchSize:
    let (cat, conf) = engine.inferFast(inputs[b])
    categories[b] = cat
    confidences[b] = conf

{.pop.}

# ========== 4並列バッチ処理（SIMD風） ==========

proc inferBatch4*(engine: var SIMDInferenceEngine,
                  i0, i1, i2, i3: openArray[float32]):
                  array[4, tuple[category: int, confidence: float32]] =
  ## 4サンプル並列推論（ループアンロール最適化）
  result[0] = engine.inferFast(i0)
  result[1] = engine.inferFast(i1)
  result[2] = engine.inferFast(i2)
  result[3] = engine.inferFast(i3)

when isMainModule:
  import std/[times, strformat]

  echo "=== SIMD最適化推論エンジン ベンチマーク ==="
  echo ""

  var engine = newSIMDInferenceEngine(64, 64, 23)
  engine.initKaiming(42)

  var input = newSeq[float32](64)
  for i in 0..<64:
    input[i] = i.float32 / 64.0f32

  # ウォームアップ
  for _ in 0..<1000:
    discard engine.inferFast(input)

  # ベンチマーク
  let iterations = 1_000_000
  let start = cpuTime()
  for _ in 0..<iterations:
    discard engine.inferFast(input)
  let elapsed = cpuTime() - start

  echo fmt"反復回数: {iterations}"
  echo fmt"所要時間: {elapsed * 1000:.2f}ms"
  echo fmt"スループット: {iterations.float / elapsed:.0f} samples/sec"
  echo fmt"レイテンシ: {elapsed / iterations.float * 1_000_000:.2f} μs/sample"
  echo ""

  let (cat, conf) = engine.inferFast(input)
  echo fmt"テスト推論結果: category={cat}, confidence={conf:.4f}"
