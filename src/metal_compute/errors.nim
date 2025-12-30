## Error Handling Module for nim-metal-compute
## Result型ベースのエラーハンドリング
##
## v0.0.2: エラーハンドリング改善

import std/[strformat]

type
  # ========== エラー種別 ==========

  NMCErrorKind* = enum
    ## エラーの種類
    ekNone = "No error"
    # Network Spec errors
    ekEmptyNetwork = "Network has no layers"
    ekLayerMismatch = "Layer size mismatch"
    ekInvalidInputSize = "Invalid input size"
    ekInvalidOutputSize = "Invalid output size"
    ekInvalidActivation = "Invalid activation function"
    ekInvalidLayerName = "Invalid layer name"
    # Weight errors
    ekFileNotFound = "File not found"
    ekFileWriteError = "Failed to write file"
    ekInvalidFormat = "Invalid file format"
    ekVersionMismatch = "Version mismatch"
    ekTensorNotFound = "Tensor not found"
    ekShapeMismatch = "Shape mismatch"
    ekDataCorrupted = "Data corrupted"
    # Inference errors
    ekNotInitialized = "Engine not initialized"
    ekInvalidBatchSize = "Invalid batch size"
    ekDimensionMismatch = "Dimension mismatch"
    # Metal errors (v0.0.3+)
    ekMetalNotAvailable = "Metal not available"
    ekShaderCompileError = "Shader compilation failed"
    ekBufferAllocationError = "Buffer allocation failed"
    ekPipelineError = "Pipeline creation failed"
    ekDeviceNotFound = "Metal device not found"
    # Metal errors (v0.0.5+)
    ekDevice = "Device error"
    ekBuffer = "Buffer error"
    ekShader = "Shader error"
    ekPipeline = "Pipeline error"
    ekCommand = "Command error"
    ekEncoder = "Encoder error"
    ekPlatform = "Platform not supported"

  NMCError* = object
    ## 構造化エラー
    kind*: NMCErrorKind
    message*: string
    details*: string
    source*: string  # エラー発生元

  # ========== Result型 ==========

  NMCResult*[T] = object
    ## 成功/失敗を表すResult型
    case isOk*: bool
    of true:
      value*: T
    of false:
      error*: NMCError

# ========== エラー生成 ==========

proc newError*(kind: NMCErrorKind,
               message: string = "",
               details: string = "",
               source: string = ""): NMCError =
  ## 新しいエラーを作成
  result = NMCError(
    kind: kind,
    message: if message.len > 0: message else: $kind,
    details: details,
    source: source
  )

# ========== Result生成 ==========

proc ok*[T](value: T): NMCResult[T] =
  ## 成功Resultを作成
  result = NMCResult[T](isOk: true, value: value)

proc err*[T](error: NMCError): NMCResult[T] =
  ## 失敗Resultを作成
  result = NMCResult[T](isOk: false, error: error)

proc err*[T](kind: NMCErrorKind, message: string = ""): NMCResult[T] =
  ## エラー種別から失敗Resultを作成
  result = NMCResult[T](isOk: false, error: newError(kind, message))

# ========== Result操作 ==========

proc isOk*[T](r: NMCResult[T]): bool =
  ## 成功かどうか
  r.isOk

proc isErr*[T](r: NMCResult[T]): bool =
  ## 失敗かどうか
  not r.isOk

proc get*[T](r: NMCResult[T]): T =
  ## 値を取得（失敗時はpanic）
  if r.isOk:
    result = r.value
  else:
    raise newException(CatchableError,
      fmt"Attempted to get value from error result: {r.error.message}")

proc getOr*[T](r: NMCResult[T], default: T): T =
  ## 値を取得、失敗時はデフォルト値
  if r.isOk:
    result = r.value
  else:
    result = default

proc getError*[T](r: NMCResult[T]): NMCError =
  ## エラーを取得（成功時はpanic）
  if not r.isOk:
    result = r.error
  else:
    raise newException(CatchableError, "Attempted to get error from success result")

proc map*[T, U](r: NMCResult[T], fn: proc(v: T): U): NMCResult[U] =
  ## 値を変換
  if r.isOk:
    result = ok(fn(r.value))
  else:
    result = err[U](r.error)

proc flatMap*[T, U](r: NMCResult[T], fn: proc(v: T): NMCResult[U]): NMCResult[U] =
  ## 値を変換（Result返却）
  if r.isOk:
    result = fn(r.value)
  else:
    result = err[U](r.error)

proc mapError*[T](r: NMCResult[T], fn: proc(e: NMCError): NMCError): NMCResult[T] =
  ## エラーを変換
  if r.isOk:
    result = r
  else:
    result = err[T](fn(r.error))

# ========== チェインマクロ ==========

template `?`*[T](r: NMCResult[T]): T =
  ## 失敗時に早期リターン
  let tmp = r
  if not tmp.isOk:
    return err[typeof(result.value)](tmp.error)
  tmp.value

template tryOr*[T](r: NMCResult[T], body: untyped): T =
  ## 失敗時に代替処理
  let tmp = r
  if tmp.isOk:
    tmp.value
  else:
    body

# ========== 文字列変換 ==========

proc `$`*(e: NMCError): string =
  ## エラーを文字列に変換
  result = fmt"[{e.kind}] {e.message}"
  if e.details.len > 0:
    result.add fmt" - {e.details}"
  if e.source.len > 0:
    result.add fmt" (at {e.source})"

proc `$`*[T](r: NMCResult[T]): string =
  ## Resultを文字列に変換
  if r.isOk:
    result = fmt"Ok({r.value})"
  else:
    result = fmt"Err({r.error})"

# ========== バリデーションヘルパー ==========

proc validatePositive*(value: int, name: string): NMCResult[int] =
  ## 正の値かどうか検証
  if value > 0:
    result = ok(value)
  else:
    result = err[int](ekInvalidInputSize, fmt"{name} must be positive, got {value}")

proc validateNonEmpty*[T](items: seq[T], name: string): NMCResult[seq[T]] =
  ## 空でないか検証
  if items.len > 0:
    result = ok(items)
  else:
    result = err[seq[T]](ekEmptyNetwork, fmt"{name} cannot be empty")

proc validateRange*(value: float, min, max: float, name: string): NMCResult[float] =
  ## 範囲内かどうか検証
  if value >= min and value <= max:
    result = ok(value)
  else:
    result = err[float](ekInvalidInputSize,
      fmt"{name} must be in range [{min}, {max}], got {value}")

# ========== Void Result用 ==========

type VoidResult* = NMCResult[bool]

proc okVoid*(): VoidResult =
  ## Voidの成功Result
  result = ok(true)

proc errVoid*(kind: NMCErrorKind, message: string = ""): VoidResult =
  ## Voidの失敗Result
  result = err[bool](kind, message)

proc errVoid*(error: NMCError): VoidResult =
  ## Voidの失敗Result（エラーオブジェクト）
  result = err[bool](error)

# ========== テスト ==========

when isMainModule:
  echo "=== Error Handling Test ==="

  # 基本的なResult操作
  let success = ok(42)
  let failure = err[int](ekInvalidInputSize, "Value must be positive")

  echo "Success: ", success
  echo "Failure: ", failure

  echo "success.isOk: ", success.isOk
  echo "failure.isErr: ", failure.isErr

  echo "success.get: ", success.get
  echo "failure.getOr(0): ", failure.getOr(0)

  # map操作
  let doubled = success.map(proc(x: int): int = x * 2)
  echo "Doubled: ", doubled

  # バリデーション
  let pos = validatePositive(10, "size")
  let neg = validatePositive(-5, "size")
  echo "Positive 10: ", pos
  echo "Positive -5: ", neg

  # VoidResult
  let voidOk = okVoid()
  let voidErr = errVoid(ekEmptyNetwork, "No layers defined")
  echo "VoidOk: ", voidOk
  echo "VoidErr: ", voidErr

  echo "\n✅ Error handling test passed!"
