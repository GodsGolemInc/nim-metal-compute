## Neural Network Weights Management
## GPU/CPU 共通の重みフォーマット
##
## 対応フォーマット:
##   - .nmw (Nim Metal Weights) - バイナリ形式
##   - .json - JSON形式 (デバッグ用)
##   - .safetensors - SafeTensors形式 (将来)

import std/[streams, json, tables, strformat, strutils, random, math, os]
import network_spec

type
  WeightTensor* = object
    name*: string
    shape*: seq[int]
    data*: seq[float32]

  NetworkWeights* = ref object
    spec*: NetworkSpec
    tensors*: OrderedTable[string, WeightTensor]
    metadata*: Table[string, string]

  WeightsError* = object of CatchableError

const
  NMW_MAGIC* = 0x4E4D5701'u32  # "NMW\x01"
  NMW_VERSION* = 1'u16

# ========== テンソル操作 ==========

proc newWeightTensor*(name: string, shape: seq[int]): WeightTensor =
  ## 新しいテンソルを作成
  var size = 1
  for s in shape: size *= s

  result = WeightTensor(
    name: name,
    shape: shape,
    data: newSeq[float32](size)
  )

proc size*(t: WeightTensor): int =
  ## テンソルの要素数
  result = 1
  for s in t.shape: result *= s

proc `[]`*(t: WeightTensor, indices: varargs[int]): float32 =
  ## テンソルの要素にアクセス
  var idx = 0
  var stride = 1
  for i in countdown(t.shape.len - 1, 0):
    idx += indices[i] * stride
    stride *= t.shape[i]
  result = t.data[idx]

proc `[]=`*(t: var WeightTensor, indices: varargs[int], value: float32) =
  ## テンソルの要素を設定
  var idx = 0
  var stride = 1
  for i in countdown(t.shape.len - 1, 0):
    idx += indices[i] * stride
    stride *= t.shape[i]
  t.data[idx] = value

# ========== 重み管理 ==========

proc newNetworkWeights*(spec: NetworkSpec): NetworkWeights =
  ## ネットワーク定義から重みを初期化
  result = NetworkWeights(
    spec: spec,
    tensors: initOrderedTable[string, WeightTensor](),
    metadata: initTable[string, string]()
  )

  result.metadata["network_name"] = spec.name
  result.metadata["total_params"] = $spec.totalParams()

  for layer in spec.layers:
    case layer.kind
    of lkDense:
      # 重み行列
      let weightName = layer.name & ".weight"
      result.tensors[weightName] = newWeightTensor(
        weightName,
        @[layer.inputSize, layer.outputSize]
      )

      # バイアス
      if layer.useBias:
        let biasName = layer.name & ".bias"
        result.tensors[biasName] = newWeightTensor(
          biasName,
          @[layer.outputSize]
        )

    of lkBatchNorm:
      # gamma, beta, running_mean, running_var
      for suffix in ["gamma", "beta", "running_mean", "running_var"]:
        let name = layer.name & "." & suffix
        result.tensors[name] = newWeightTensor(name, @[layer.inputSize])

    else:
      discard  # Dropout等はパラメータなし

proc initXavier*(weights: NetworkWeights, seed: int = 42) =
  ## Xavier初期化
  var rng = initRand(seed)

  for layer in weights.spec.layers:
    if layer.kind == lkDense:
      let weightName = layer.name & ".weight"
      let scale = sqrt(2.0 / (layer.inputSize + layer.outputSize).float)

      for i in 0..<weights.tensors[weightName].size:
        weights.tensors[weightName].data[i] = (rng.rand(2.0) - 1.0).float32 * scale.float32

      if layer.useBias:
        let biasName = layer.name & ".bias"
        for i in 0..<weights.tensors[biasName].size:
          weights.tensors[biasName].data[i] = 0.0f32

proc initKaiming*(weights: NetworkWeights, seed: int = 42) =
  ## Kaiming (He) 初期化 - ReLU向け
  var rng = initRand(seed)

  for layer in weights.spec.layers:
    if layer.kind == lkDense:
      let weightName = layer.name & ".weight"
      let scale = sqrt(2.0 / layer.inputSize.float)

      for i in 0..<weights.tensors[weightName].size:
        weights.tensors[weightName].data[i] = (rng.rand(2.0) - 1.0).float32 * scale.float32

      if layer.useBias:
        let biasName = layer.name & ".bias"
        for i in 0..<weights.tensors[biasName].size:
          weights.tensors[biasName].data[i] = 0.0f32

# ========== バイナリ形式 (.nmw) ==========

proc saveNMW*(weights: NetworkWeights, path: string) =
  ## バイナリ形式で保存
  var s = newFileStream(path, fmWrite)
  if s == nil:
    raise newException(WeightsError, "Failed to open file: " & path)

  defer: s.close()

  # ヘッダー
  s.write(NMW_MAGIC)
  s.write(NMW_VERSION)

  # ネットワーク定義 (JSON)
  let specJson = $weights.spec.toJson()
  s.write(specJson.len.int32)
  s.write(specJson)

  # テンソル数
  s.write(weights.tensors.len.int32)

  # 各テンソル
  for name, tensor in weights.tensors:
    # テンソル名
    s.write(name.len.int32)
    s.write(name)

    # 形状
    s.write(tensor.shape.len.int32)
    for dim in tensor.shape:
      s.write(dim.int32)

    # データ
    s.write(tensor.data.len.int32)
    for val in tensor.data:
      s.write(val)

proc loadNMW*(path: string): NetworkWeights =
  ## バイナリ形式から読み込み
  var s = newFileStream(path, fmRead)
  if s == nil:
    raise newException(WeightsError, "Failed to open file: " & path)

  defer: s.close()

  # ヘッダー検証
  var magic: uint32
  s.read(magic)
  if magic != NMW_MAGIC:
    raise newException(WeightsError, "Invalid NMW file format")

  var version: uint16
  s.read(version)
  if version != NMW_VERSION:
    raise newException(WeightsError, fmt"Unsupported NMW version: {version}")

  # ネットワーク定義
  var specLen: int32
  s.read(specLen)
  var specJson = newString(specLen)
  discard s.readData(addr specJson[0], specLen)

  let spec = fromJson(parseJson(specJson))
  result = newNetworkWeights(spec)

  # テンソル数
  var numTensors: int32
  s.read(numTensors)

  # 各テンソル
  for _ in 0..<numTensors:
    # テンソル名
    var nameLen: int32
    s.read(nameLen)
    var name = newString(nameLen)
    discard s.readData(addr name[0], nameLen)

    # 形状
    var numDims: int32
    s.read(numDims)
    var shape = newSeq[int](numDims)
    for i in 0..<numDims:
      var dim: int32
      s.read(dim)
      shape[i] = dim

    # データ
    var dataLen: int32
    s.read(dataLen)

    var tensor = newWeightTensor(name, shape)
    for i in 0..<dataLen:
      s.read(tensor.data[i])

    result.tensors[name] = tensor

# ========== JSON形式 ==========

proc saveJSON*(weights: NetworkWeights, path: string) =
  ## JSON形式で保存 (デバッグ用)
  var json = %*{
    "spec": weights.spec.toJson(),
    "metadata": %weights.metadata,
    "tensors": %*{}
  }

  for name, tensor in weights.tensors:
    json["tensors"][name] = %*{
      "shape": tensor.shape,
      "data": tensor.data
    }

  writeFile(path, json.pretty())

proc loadJSON*(path: string): NetworkWeights =
  ## JSON形式から読み込み
  let json = parseJson(readFile(path))
  let spec = fromJson(json["spec"])
  result = newNetworkWeights(spec)

  for key, val in json["metadata"].pairs:
    result.metadata[key] = val.getStr()

  for name, tensorJson in json["tensors"].pairs:
    var shape = newSeq[int]()
    for s in tensorJson["shape"]:
      shape.add(s.getInt())

    var tensor = newWeightTensor(name, shape)
    for i, val in tensorJson["data"].pairs:
      tensor.data[i.parseInt] = val.getFloat().float32

    result.tensors[name] = tensor

# ========== フラット配列への変換 ==========

proc toFlatArray*(weights: NetworkWeights): seq[float32] =
  ## 全重みをフラットな配列に変換 (GPU転送用)
  for layer in weights.spec.layers:
    if layer.kind == lkDense:
      let weightName = layer.name & ".weight"
      result.add(weights.tensors[weightName].data)

      if layer.useBias:
        let biasName = layer.name & ".bias"
        result.add(weights.tensors[biasName].data)

proc fromFlatArray*(weights: var NetworkWeights, data: seq[float32]) =
  ## フラット配列から重みを復元
  var offset = 0

  for layer in weights.spec.layers:
    if layer.kind == lkDense:
      let weightName = layer.name & ".weight"
      let weightSize = weights.tensors[weightName].size
      for i in 0..<weightSize:
        weights.tensors[weightName].data[i] = data[offset + i]
      offset += weightSize

      if layer.useBias:
        let biasName = layer.name & ".bias"
        let biasSize = weights.tensors[biasName].size
        for i in 0..<biasSize:
          weights.tensors[biasName].data[i] = data[offset + i]
        offset += biasSize

# ========== ユーティリティ ==========

proc summary*(weights: NetworkWeights): string =
  ## 重みのサマリー表示
  result = fmt"""
Network Weights: {weights.spec.name}
{'='.repeat(60)}
"""

  var totalSize = 0
  for name, tensor in weights.tensors:
    let size = tensor.size
    totalSize += size
    result.add fmt"  {name:<30} {tensor.shape} -> {size:>8} params\n"

  result.add fmt"{'='.repeat(60)}\n"
  result.add fmt"Total: {totalSize} parameters ({totalSize * 4 / 1024 / 1024:.2f} MB)" & "\n"

# ========== テスト ==========

when isMainModule:
  import std/tempfiles

  echo "=== Network Weights Test ==="

  # ネットワーク定義
  let spec = koanClassifierSpec()
  var weights = newNetworkWeights(spec)

  # Xavier初期化
  weights.initXavier(42)
  echo weights.summary()

  # バイナリ保存/読み込みテスト
  let (tmpFile, tmpPath) = createTempFile("weights_", ".nmw")
  tmpFile.close()

  weights.saveNMW(tmpPath)
  echo fmt"Saved to: {tmpPath}"

  let loaded = loadNMW(tmpPath)
  echo "Loaded weights:"
  echo loaded.summary()

  # フラット配列変換テスト
  let flat = weights.toFlatArray()
  echo fmt"Flat array size: {flat.len}"

  # JSON保存テスト
  let jsonPath = tmpPath.replace(".nmw", ".json")
  weights.saveJSON(jsonPath)
  echo fmt"JSON saved to: {jsonPath}"

  removeFile(tmpPath)
  removeFile(jsonPath)

  echo "\n✅ Weights test passed!"
