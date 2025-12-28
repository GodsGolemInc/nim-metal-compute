## Neural Network Specification DSL
## ニューラルネットワーク構造を宣言的に定義
##
## 使用例:
##   let spec = newNetwork("MyClassifier")
##     .addDense("hidden1", 64, 128, actReLU)
##     .addDense("hidden2", 128, 64, actReLU)
##     .addDense("output", 64, 10, actSoftmax)

import std/[strformat, strutils, json, tables]

type
  ActivationType* = enum
    actNone = "none"
    actReLU = "relu"
    actSigmoid = "sigmoid"
    actTanh = "tanh"
    actSoftmax = "softmax"
    actLeakyReLU = "leaky_relu"

  LayerKind* = enum
    lkDense = "dense"
    lkConv2D = "conv2d"
    lkBatchNorm = "batch_norm"
    lkDropout = "dropout"
    lkFlatten = "flatten"

  LayerSpec* = object
    name*: string
    kind*: LayerKind
    inputSize*: int
    outputSize*: int
    activation*: ActivationType
    useBias*: bool
    # Conv2D用
    kernelSize*: int
    stride*: int
    padding*: int
    # Dropout用
    dropoutRate*: float

  NetworkSpec* = object
    name*: string
    layers*: seq[LayerSpec]
    inputShape*: seq[int]  # 入力形状 [batch, features] or [batch, channels, height, width]

  NetworkError* = object of CatchableError

# ========== ビルダーAPI ==========

proc newNetwork*(name: string, inputSize: int = 0): NetworkSpec =
  ## 新しいネットワーク定義を作成
  result = NetworkSpec(
    name: name,
    layers: @[],
    inputShape: if inputSize > 0: @[inputSize] else: @[]
  )

proc addDense*(net: var NetworkSpec, name: string,
               inputSize, outputSize: int,
               activation: ActivationType = actReLU,
               useBias: bool = true): var NetworkSpec {.discardable.} =
  ## 全結合層を追加
  net.layers.add(LayerSpec(
    name: name,
    kind: lkDense,
    inputSize: inputSize,
    outputSize: outputSize,
    activation: activation,
    useBias: useBias
  ))
  result = net

proc addBatchNorm*(net: var NetworkSpec, name: string,
                   size: int): var NetworkSpec {.discardable.} =
  ## バッチ正規化層を追加
  net.layers.add(LayerSpec(
    name: name,
    kind: lkBatchNorm,
    inputSize: size,
    outputSize: size,
    activation: actNone,
    useBias: false
  ))
  result = net

proc addDropout*(net: var NetworkSpec, name: string,
                 size: int, rate: float = 0.5): var NetworkSpec {.discardable.} =
  ## ドロップアウト層を追加
  net.layers.add(LayerSpec(
    name: name,
    kind: lkDropout,
    inputSize: size,
    outputSize: size,
    activation: actNone,
    useBias: false,
    dropoutRate: rate
  ))
  result = net

# ========== バリデーション ==========

proc validate*(spec: NetworkSpec): bool =
  ## ネットワーク定義の妥当性を検証
  if spec.layers.len == 0:
    raise newException(NetworkError, "Network has no layers")

  for i in 1..<spec.layers.len:
    let prev = spec.layers[i - 1]
    let curr = spec.layers[i]
    if prev.outputSize != curr.inputSize:
      raise newException(NetworkError,
        fmt"Layer size mismatch: {prev.name} output ({prev.outputSize}) != {curr.name} input ({curr.inputSize})")

  result = true

# ========== シリアライズ ==========

proc toJson*(spec: NetworkSpec): JsonNode =
  ## ネットワーク定義をJSONに変換
  result = %*{
    "name": spec.name,
    "input_shape": spec.inputShape,
    "layers": []
  }

  for layer in spec.layers:
    var layerJson = %*{
      "name": layer.name,
      "kind": $layer.kind,
      "input_size": layer.inputSize,
      "output_size": layer.outputSize,
      "activation": $layer.activation,
      "use_bias": layer.useBias
    }
    if layer.kind == lkDropout:
      layerJson["dropout_rate"] = %layer.dropoutRate
    result["layers"].add(layerJson)

proc fromJson*(json: JsonNode): NetworkSpec =
  ## JSONからネットワーク定義を復元
  result.name = json["name"].getStr()

  if json.hasKey("input_shape"):
    for s in json["input_shape"]:
      result.inputShape.add(s.getInt())

  for layerJson in json["layers"]:
    var layer = LayerSpec(
      name: layerJson["name"].getStr(),
      inputSize: layerJson["input_size"].getInt(),
      outputSize: layerJson["output_size"].getInt(),
      useBias: layerJson["use_bias"].getBool(true)
    )

    # Kind
    case layerJson["kind"].getStr()
    of "dense": layer.kind = lkDense
    of "batch_norm": layer.kind = lkBatchNorm
    of "dropout": layer.kind = lkDropout
    else: layer.kind = lkDense

    # Activation
    case layerJson["activation"].getStr()
    of "relu": layer.activation = actReLU
    of "sigmoid": layer.activation = actSigmoid
    of "tanh": layer.activation = actTanh
    of "softmax": layer.activation = actSoftmax
    of "leaky_relu": layer.activation = actLeakyReLU
    else: layer.activation = actNone

    if layer.kind == lkDropout:
      layer.dropoutRate = layerJson["dropout_rate"].getFloat(0.5)

    result.layers.add(layer)

proc saveSpec*(spec: NetworkSpec, path: string) =
  ## ネットワーク定義をファイルに保存
  writeFile(path, $spec.toJson())

proc loadSpec*(path: string): NetworkSpec =
  ## ファイルからネットワーク定義を読み込み
  let json = parseJson(readFile(path))
  result = fromJson(json)

# ========== ユーティリティ ==========

proc totalParams*(spec: NetworkSpec): int =
  ## 総パラメータ数を計算
  for layer in spec.layers:
    if layer.kind == lkDense:
      result += layer.inputSize * layer.outputSize
      if layer.useBias:
        result += layer.outputSize

proc summary*(spec: NetworkSpec): string =
  ## ネットワークのサマリーを表示
  result = fmt"""
Network: {spec.name}
{'='.repeat(60)}
"""

  var totalParams = 0
  for i, layer in spec.layers:
    let params = if layer.kind == lkDense:
      layer.inputSize * layer.outputSize + (if layer.useBias: layer.outputSize else: 0)
    else: 0

    totalParams += params

    result.add fmt"{i:3}: {layer.name:<15} {$layer.kind:<12} "
    result.add fmt"({layer.inputSize:5} -> {layer.outputSize:5}) "
    result.add fmt"{$layer.activation:<10} "
    result.add fmt"params: {params:>8}"
    result.add "\n"

  result.add "=".repeat(60) & "\n"
  result.add fmt"Total parameters: {totalParams}" & "\n"

proc inputSize*(spec: NetworkSpec): int =
  ## 入力サイズを取得
  if spec.layers.len > 0:
    result = spec.layers[0].inputSize

proc outputSize*(spec: NetworkSpec): int =
  ## 出力サイズを取得
  if spec.layers.len > 0:
    result = spec.layers[^1].outputSize

# ========== プリセット定義 ==========

proc koanClassifierSpec*(): NetworkSpec =
  ## Koan分類器のネットワーク定義
  var net = newNetwork("KoanClassifier", 64)
  net
    .addDense("hidden", 64, 64, actReLU)
    .addDense("output", 64, 23, actSoftmax)
  discard net.validate()
  result = net

proc mlpClassifier*(inputSize, hiddenSize, outputSize: int,
                    numHidden: int = 1): NetworkSpec =
  ## 汎用MLP分類器
  var net = newNetwork("MLPClassifier", inputSize)
  var prevSize = inputSize

  for i in 0..<numHidden:
    net.addDense(fmt"hidden{i}", prevSize, hiddenSize, actReLU)
    prevSize = hiddenSize

  net.addDense("output", prevSize, outputSize, actSoftmax)
  discard net.validate()
  result = net

# ========== テスト ==========

when isMainModule:
  echo "=== Network Specification Test ==="

  # Koan分類器
  let koan = koanClassifierSpec()
  echo koan.summary()

  # カスタムネットワーク
  var custom = newNetwork("CustomNet", 784)
  custom
    .addDense("hidden1", 784, 256, actReLU)
    .addBatchNorm("bn1", 256)
    .addDropout("dropout1", 256, 0.3)
    .addDense("hidden2", 256, 128, actReLU)
    .addDense("output", 128, 10, actSoftmax)

  echo custom.summary()

  # JSON変換テスト
  let json = custom.toJson()
  echo "\nJSON:"
  echo json.pretty()

  # 復元テスト
  let restored = fromJson(json)
  echo "\nRestored:"
  echo restored.summary()
