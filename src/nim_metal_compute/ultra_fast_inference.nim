## Ultra Fast Inference Engine
## 100万+ samples/sec を目指す極限最適化版
##
## 最適化技術:
## - ループアンロール
## - キャッシュライン最適化
## - 分岐除去
## - インライン展開
## - SIMD風並列処理

import std/[math]

const
  # キャッシュライン最適化
  CacheLineSize = 64
  # 固定ネットワークサイズ（コンパイル時最適化）
  InputSize* = 64
  HiddenSize* = 64
  OutputSize* = 23

type
  # 16バイトアラインメント
  AlignedWeights* = object
    # Row-major連続メモリ [input * hidden]
    wIH*: array[InputSize * HiddenSize, float32]
    bH*: array[HiddenSize, float32]
    wHO*: array[HiddenSize * OutputSize, float32]
    bO*: array[OutputSize, float32]

  UltraFastEngine* = object
    weights*: AlignedWeights
    # 作業バッファ（スタック割り当て可能なサイズ）
    hidden*: array[HiddenSize, float32]
    output*: array[OutputSize, float32]

{.push overflowChecks:off.}
proc initWeights*(engine: var UltraFastEngine, seed: uint32 = 42) =
  var rng = seed

  template nextRand(): float32 =
    rng = rng * 1103515245'u32 + 12345'u32
    ((rng shr 16) and 0x7FFF'u32).float32 / 32767.0f32 * 2.0f32 - 1.0f32

  let scaleIH = sqrt(2.0f32 / InputSize.float32)
  let scaleHO = sqrt(2.0f32 / HiddenSize.float32)

  for i in 0..<InputSize * HiddenSize:
    engine.weights.wIH[i] = nextRand() * scaleIH
  for i in 0..<HiddenSize * OutputSize:
    engine.weights.wHO[i] = nextRand() * scaleHO
  for i in 0..<HiddenSize:
    engine.weights.bH[i] = 0.0f32
  for i in 0..<OutputSize:
    engine.weights.bO[i] = 0.0f32
{.pop.}

{.push checks:off, boundChecks:off, overflowChecks:off, rangeChecks:off, nilChecks:off.}

# インライン強制
template relu(x: float32): float32 =
  if x > 0.0f32: x else: 0.0f32

proc inferUltra*(engine: var UltraFastEngine,
                 input: ptr UncheckedArray[float32]): tuple[cat: int, conf: float32] {.inline.} =
  ## 極限最適化推論
  ## - ポインタ直接アクセス
  ## - ループアンロール（4並列）
  ## - 分岐最小化

  let wIH = cast[ptr UncheckedArray[float32]](addr engine.weights.wIH[0])
  let bH = cast[ptr UncheckedArray[float32]](addr engine.weights.bH[0])
  let wHO = cast[ptr UncheckedArray[float32]](addr engine.weights.wHO[0])
  let bO = cast[ptr UncheckedArray[float32]](addr engine.weights.bO[0])
  let hidden = cast[ptr UncheckedArray[float32]](addr engine.hidden[0])
  let output = cast[ptr UncheckedArray[float32]](addr engine.output[0])

  # 入力層 → 隠れ層（4並列アンロール）
  var j = 0
  while j < HiddenSize:
    var sum0 = bH[j]
    var sum1 = bH[j+1]
    var sum2 = bH[j+2]
    var sum3 = bH[j+3]

    var i = 0
    while i < InputSize:
      let in0 = input[i]
      let in1 = input[i+1]
      let in2 = input[i+2]
      let in3 = input[i+3]

      let base0 = i * HiddenSize + j
      let base1 = (i+1) * HiddenSize + j
      let base2 = (i+2) * HiddenSize + j
      let base3 = (i+3) * HiddenSize + j

      sum0 += in0 * wIH[base0] + in1 * wIH[base1] + in2 * wIH[base2] + in3 * wIH[base3]
      sum1 += in0 * wIH[base0+1] + in1 * wIH[base1+1] + in2 * wIH[base2+1] + in3 * wIH[base3+1]
      sum2 += in0 * wIH[base0+2] + in1 * wIH[base1+2] + in2 * wIH[base2+2] + in3 * wIH[base3+2]
      sum3 += in0 * wIH[base0+3] + in1 * wIH[base1+3] + in2 * wIH[base2+3] + in3 * wIH[base3+3]

      i += 4

    hidden[j] = relu(sum0)
    hidden[j+1] = relu(sum1)
    hidden[j+2] = relu(sum2)
    hidden[j+3] = relu(sum3)
    j += 4

  # 隠れ層 → 出力層
  var maxVal = -1e30f32
  var maxIdx = 0

  for k in 0..<OutputSize:
    var sum = bO[k]
    var h = 0
    while h < HiddenSize:
      sum += hidden[h] * wHO[h * OutputSize + k] +
             hidden[h+1] * wHO[(h+1) * OutputSize + k] +
             hidden[h+2] * wHO[(h+2) * OutputSize + k] +
             hidden[h+3] * wHO[(h+3) * OutputSize + k]
      h += 4
    output[k] = sum
    if sum > maxVal:
      maxVal = sum
      maxIdx = k

  # 簡易Softmax（最大値の正規化のみ）
  var expSum = 0.0f32
  for k in 0..<OutputSize:
    let e = exp(output[k] - maxVal)
    output[k] = e
    expSum += e

  result = (maxIdx, output[maxIdx] / expSum)

proc inferUltraArray*(engine: var UltraFastEngine,
                      input: array[InputSize, float32]): tuple[cat: int, conf: float32] {.inline.} =
  inferUltra(engine, cast[ptr UncheckedArray[float32]](unsafeAddr input[0]))

# バッチ推論（8並列）
proc inferBatch8*(engine: var UltraFastEngine,
                  inputs: ptr UncheckedArray[array[InputSize, float32]],
                  results: ptr UncheckedArray[tuple[cat: int, conf: float32]]) {.inline.} =
  for i in 0..<8:
    results[i] = engine.inferUltraArray(inputs[i])

{.pop.}

when isMainModule:
  import std/[times, strformat]

  echo "=== Ultra Fast Inference Benchmark ==="
  echo ""

  var engine: UltraFastEngine
  engine.initWeights(42)

  var input: array[InputSize, float32]
  for i in 0..<InputSize:
    input[i] = i.float32 / InputSize.float32

  # ウォームアップ
  for _ in 0..<10000:
    discard engine.inferUltraArray(input)

  # ベンチマーク
  let iterations = 10_000_000
  let start = cpuTime()
  for _ in 0..<iterations:
    discard engine.inferUltraArray(input)
  let elapsed = cpuTime() - start

  let throughput = iterations.float / elapsed
  echo fmt"反復回数: {iterations}"
  echo fmt"所要時間: {elapsed * 1000:.2f}ms"
  echo fmt"スループット: {throughput:.0f} samples/sec"
  echo fmt"レイテンシ: {elapsed / iterations.float * 1_000_000:.3f} μs"
  echo ""

  if throughput >= 1_000_000:
    echo "✅ 100万 samples/sec 達成！"
  else:
    echo fmt"達成率: {throughput / 1_000_000 * 100:.1f}%"

  let (cat, conf) = engine.inferUltraArray(input)
  echo fmt"\nテスト結果: category={cat}, confidence={conf:.4f}"
