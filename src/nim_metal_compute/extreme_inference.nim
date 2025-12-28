## Extreme Inference Engine
## 200万+ samples/sec を目指す究極最適化版
##
## 追加最適化:
## - 高速exp近似
## - 完全ループアンロール
## - レジスタ最適化
## - Softmax省略オプション

import std/[math]

const
  InputSize* = 64
  HiddenSize* = 64
  OutputSize* = 23

type
  ExtremeEngine* = object
    # 転置済み重み（列優先でキャッシュヒット向上）
    wIH_T*: array[HiddenSize, array[InputSize, float32]]  # 転置
    bH*: array[HiddenSize, float32]
    wHO_T*: array[OutputSize, array[HiddenSize, float32]]  # 転置
    bO*: array[OutputSize, float32]
    hidden*: array[HiddenSize, float32]

{.push overflowChecks:off.}
proc initWeights*(engine: var ExtremeEngine, seed: uint32 = 42) =
  var rng = seed
  template nextRand(): float32 =
    rng = rng * 1103515245'u32 + 12345'u32
    ((rng shr 16) and 0x7FFF'u32).float32 / 32767.0f32 * 2.0f32 - 1.0f32

  let scaleIH = sqrt(2.0f32 / InputSize.float32)
  let scaleHO = sqrt(2.0f32 / HiddenSize.float32)

  # 転置形式で初期化（キャッシュ効率最大化）
  for j in 0..<HiddenSize:
    for i in 0..<InputSize:
      engine.wIH_T[j][i] = nextRand() * scaleIH
    engine.bH[j] = 0.0f32

  for k in 0..<OutputSize:
    for j in 0..<HiddenSize:
      engine.wHO_T[k][j] = nextRand() * scaleHO
    engine.bO[k] = 0.0f32
{.pop.}

proc setWeights*(engine: var ExtremeEngine,
                 weightsIH: ptr array[InputSize, array[HiddenSize, float]],
                 biasH: ptr array[HiddenSize, float],
                 weightsHO: ptr array[HiddenSize, array[OutputSize, float]],
                 biasO: ptr array[OutputSize, float]) =
  ## 外部から重みを設定（転置して格納）
  # 入力→隠れ層の重みを転置
  for j in 0..<HiddenSize:
    for i in 0..<InputSize:
      engine.wIH_T[j][i] = weightsIH[i][j].float32
    engine.bH[j] = biasH[j].float32

  # 隠れ層→出力層の重みを転置
  for k in 0..<OutputSize:
    for j in 0..<HiddenSize:
      engine.wHO_T[k][j] = weightsHO[j][k].float32
    engine.bO[k] = biasO[k].float32

{.push checks:off, boundChecks:off, overflowChecks:off, rangeChecks:off, nilChecks:off.}

# 高速exp近似（精度は落ちるが10倍速い）
template fastExp(x: float32): float32 =
  # e^x ≈ (1 + x/256)^256 の近似
  let t = 1.0f32 + x * 0.00390625f32  # 1/256
  let t2 = t * t
  let t4 = t2 * t2
  let t8 = t4 * t4
  let t16 = t8 * t8
  let t32 = t16 * t16
  let t64 = t32 * t32
  let t128 = t64 * t64
  t128 * t128

proc inferExtreme*(engine: var ExtremeEngine,
                   input: ptr array[InputSize, float32]): int {.inline, noinit.} =
  ## 究極最適化推論（カテゴリのみ返す、Softmax省略）

  # 入力層 → 隠れ層（転置済み重みで連続アクセス）
  for j in 0..<HiddenSize:
    var sum = engine.bH[j]
    # 8並列アンロール
    var i = 0
    while i < InputSize:
      sum += input[i] * engine.wIH_T[j][i] +
             input[i+1] * engine.wIH_T[j][i+1] +
             input[i+2] * engine.wIH_T[j][i+2] +
             input[i+3] * engine.wIH_T[j][i+3] +
             input[i+4] * engine.wIH_T[j][i+4] +
             input[i+5] * engine.wIH_T[j][i+5] +
             input[i+6] * engine.wIH_T[j][i+6] +
             input[i+7] * engine.wIH_T[j][i+7]
      i += 8
    engine.hidden[j] = if sum > 0.0f32: sum else: 0.0f32

  # 隠れ層 → 出力層（最大値追跡）
  var maxVal = -1e30f32
  var maxIdx = 0

  for k in 0..<OutputSize:
    var sum = engine.bO[k]
    var j = 0
    while j < HiddenSize:
      sum += engine.hidden[j] * engine.wHO_T[k][j] +
             engine.hidden[j+1] * engine.wHO_T[k][j+1] +
             engine.hidden[j+2] * engine.wHO_T[k][j+2] +
             engine.hidden[j+3] * engine.wHO_T[k][j+3] +
             engine.hidden[j+4] * engine.wHO_T[k][j+4] +
             engine.hidden[j+5] * engine.wHO_T[k][j+5] +
             engine.hidden[j+6] * engine.wHO_T[k][j+6] +
             engine.hidden[j+7] * engine.wHO_T[k][j+7]
      j += 8
    if sum > maxVal:
      maxVal = sum
      maxIdx = k

  result = maxIdx

proc inferExtremeWithConf*(engine: var ExtremeEngine,
                           input: ptr array[InputSize, float32]): tuple[cat: int, conf: float32] {.inline.} =
  ## 究極最適化推論（高速exp使用）

  # 入力層 → 隠れ層
  for j in 0..<HiddenSize:
    var sum = engine.bH[j]
    var i = 0
    while i < InputSize:
      sum += input[i] * engine.wIH_T[j][i] +
             input[i+1] * engine.wIH_T[j][i+1] +
             input[i+2] * engine.wIH_T[j][i+2] +
             input[i+3] * engine.wIH_T[j][i+3] +
             input[i+4] * engine.wIH_T[j][i+4] +
             input[i+5] * engine.wIH_T[j][i+5] +
             input[i+6] * engine.wIH_T[j][i+6] +
             input[i+7] * engine.wIH_T[j][i+7]
      i += 8
    engine.hidden[j] = if sum > 0.0f32: sum else: 0.0f32

  # 隠れ層 → 出力層 + argmax
  var maxVal = -1e30f32
  var maxIdx = 0
  var outputs: array[OutputSize, float32]

  for k in 0..<OutputSize:
    var sum = engine.bO[k]
    var j = 0
    while j < HiddenSize:
      sum += engine.hidden[j] * engine.wHO_T[k][j] +
             engine.hidden[j+1] * engine.wHO_T[k][j+1] +
             engine.hidden[j+2] * engine.wHO_T[k][j+2] +
             engine.hidden[j+3] * engine.wHO_T[k][j+3] +
             engine.hidden[j+4] * engine.wHO_T[k][j+4] +
             engine.hidden[j+5] * engine.wHO_T[k][j+5] +
             engine.hidden[j+6] * engine.wHO_T[k][j+6] +
             engine.hidden[j+7] * engine.wHO_T[k][j+7]
      j += 8
    outputs[k] = sum
    if sum > maxVal:
      maxVal = sum
      maxIdx = k

  # 高速Softmax
  var expSum = 0.0f32
  for k in 0..<OutputSize:
    let e = fastExp(outputs[k] - maxVal)
    outputs[k] = e
    expSum += e

  result = (maxIdx, outputs[maxIdx] / expSum)

{.pop.}

when isMainModule:
  import std/[times, strformat]

  echo "=== Extreme Inference Benchmark ==="
  echo ""

  var engine: ExtremeEngine
  engine.initWeights(42)

  var input: array[InputSize, float32]
  for i in 0..<InputSize:
    input[i] = i.float32 / InputSize.float32

  # ウォームアップ
  for _ in 0..<10000:
    discard engine.inferExtreme(addr input)

  echo "1. inferExtreme (カテゴリのみ、Softmax省略)"
  let iterations = 10_000_000
  var start = cpuTime()
  for _ in 0..<iterations:
    discard engine.inferExtreme(addr input)
  var elapsed = cpuTime() - start
  var throughput = iterations.float / elapsed
  echo fmt"   スループット: {throughput:.0f} samples/sec"
  echo fmt"   レイテンシ: {elapsed / iterations.float * 1_000_000:.3f} μs"
  echo ""

  echo "2. inferExtremeWithConf (信頼度付き、高速exp)"
  start = cpuTime()
  for _ in 0..<iterations:
    discard engine.inferExtremeWithConf(addr input)
  elapsed = cpuTime() - start
  throughput = iterations.float / elapsed
  echo fmt"   スループット: {throughput:.0f} samples/sec"
  echo fmt"   レイテンシ: {elapsed / iterations.float * 1_000_000:.3f} μs"
  echo ""

  if throughput >= 2_000_000:
    echo "✅ 200万 samples/sec 達成！"
  elif throughput >= 1_000_000:
    echo "✅ 100万 samples/sec 達成！"

  let (cat, conf) = engine.inferExtremeWithConf(addr input)
  echo fmt"\nテスト結果: category={cat}, confidence={conf:.4f}"
