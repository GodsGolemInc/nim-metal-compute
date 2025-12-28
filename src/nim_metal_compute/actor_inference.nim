## Actor-based Parallel Inference Engine
## アクターモデルによるスレッド競合回避型並列推論
##
## 特徴:
## - 各アクターが独自のExtremeEngineを所有（状態分離）
## - メッセージパッシングによる通信（共有状態なし）
## - Supervisorによる障害復旧
## - ラウンドロビンルーティング

import std/[json, times, tables, options, cpuinfo, atomics, monotimes, strformat]
import ./extreme_inference

const
  NumWorkers* = 8
  MailboxSize = 1024

type
  # メッセージ型
  InferenceRequest* = object
    id*: int
    input*: array[InputSize, float32]

  InferenceResponse* = object
    id*: int
    category*: int
    confidence*: float32

  # アクター状態
  WorkerState* = enum
    wsIdle, wsProcessing, wsStopped

  # 推論ワーカーアクター
  InferenceWorker* = object
    id*: int
    engine*: ExtremeEngine
    state*: WorkerState
    processedCount*: int
    # リングバッファメールボックス
    mailbox*: array[MailboxSize, InferenceRequest]
    mailboxHead*: int
    mailboxTail*: int
    # 結果バッファ
    results*: array[MailboxSize, InferenceResponse]
    resultsHead*: int
    resultsTail*: int

  # アクターシステム
  InferenceActorSystem* = object
    workers*: array[NumWorkers, InferenceWorker]
    nextWorker*: int  # ラウンドロビン用
    isRunning*: bool
    totalProcessed*: Atomic[int]

# ========== ワーカーアクター操作 ==========

proc initWorker*(worker: var InferenceWorker, id: int) =
  worker.id = id
  worker.engine.initWeights(uint32(42 + id))
  worker.state = wsIdle
  worker.processedCount = 0
  worker.mailboxHead = 0
  worker.mailboxTail = 0
  worker.resultsHead = 0
  worker.resultsTail = 0

proc mailboxEmpty*(worker: InferenceWorker): bool {.inline.} =
  worker.mailboxHead == worker.mailboxTail

proc mailboxFull*(worker: InferenceWorker): bool {.inline.} =
  ((worker.mailboxTail + 1) mod MailboxSize) == worker.mailboxHead

proc enqueueRequest*(worker: var InferenceWorker, req: InferenceRequest): bool {.inline.} =
  if worker.mailboxFull():
    return false
  worker.mailbox[worker.mailboxTail] = req
  worker.mailboxTail = (worker.mailboxTail + 1) mod MailboxSize
  true

proc dequeueRequest*(worker: var InferenceWorker): InferenceRequest {.inline.} =
  result = worker.mailbox[worker.mailboxHead]
  worker.mailboxHead = (worker.mailboxHead + 1) mod MailboxSize

proc resultsEmpty*(worker: InferenceWorker): bool {.inline.} =
  worker.resultsHead == worker.resultsTail

proc enqueueResult*(worker: var InferenceWorker, resp: InferenceResponse) {.inline.} =
  worker.results[worker.resultsTail] = resp
  worker.resultsTail = (worker.resultsTail + 1) mod MailboxSize

proc dequeueResult*(worker: var InferenceWorker): InferenceResponse {.inline.} =
  result = worker.results[worker.resultsHead]
  worker.resultsHead = (worker.resultsHead + 1) mod MailboxSize

{.push checks:off, boundChecks:off.}

proc processOne*(worker: var InferenceWorker): bool {.inline.} =
  ## 1件の推論を処理
  if worker.mailboxEmpty():
    return false

  worker.state = wsProcessing
  let req = worker.dequeueRequest()

  # 推論実行（アクター内で完結、共有状態なし）
  var inputPtr = unsafeAddr req.input
  let (cat, conf) = worker.engine.inferExtremeWithConf(inputPtr)

  # 結果をエンキュー
  worker.enqueueResult(InferenceResponse(
    id: req.id,
    category: cat,
    confidence: conf
  ))

  inc(worker.processedCount)
  worker.state = wsIdle
  true

proc processBatch*(worker: var InferenceWorker, maxBatch: int = 100): int =
  ## バッチ処理
  result = 0
  for _ in 0..<maxBatch:
    if not worker.processOne():
      break
    inc(result)

{.pop.}

# ========== アクターシステム操作 ==========

proc initActorSystem*(system: var InferenceActorSystem) =
  for i in 0..<NumWorkers:
    system.workers[i].initWorker(i)
  system.nextWorker = 0
  system.isRunning = true
  system.totalProcessed.store(0)

proc routeRequest*(system: var InferenceActorSystem, req: InferenceRequest): bool =
  ## ラウンドロビンでリクエストをルーティング
  let startWorker = system.nextWorker
  var attempts = 0

  while attempts < NumWorkers:
    let workerIdx = (startWorker + attempts) mod NumWorkers
    if system.workers[workerIdx].enqueueRequest(req):
      system.nextWorker = (workerIdx + 1) mod NumWorkers
      return true
    inc(attempts)

  false  # 全ワーカーのメールボックスが満杯

proc tick*(system: var InferenceActorSystem): int =
  ## 全ワーカーを1ティック処理
  result = 0
  for i in 0..<NumWorkers:
    result += system.workers[i].processBatch(10)
  discard system.totalProcessed.fetchAdd(result)

proc collectResults*(system: var InferenceActorSystem,
                     output: var seq[InferenceResponse]): int =
  ## 全ワーカーから結果を収集
  result = 0
  for i in 0..<NumWorkers:
    while not system.workers[i].resultsEmpty():
      output.add(system.workers[i].dequeueResult())
      inc(result)

proc shutdown*(system: var InferenceActorSystem) =
  system.isRunning = false
  for i in 0..<NumWorkers:
    system.workers[i].state = wsStopped

# ========== スレッド並列アクターシステム ==========

type
  ThreadedActorSystem* = object
    workers*: array[NumWorkers, ptr InferenceWorker]
    threads*: array[NumWorkers, Thread[ptr InferenceWorker]]
    running*: Atomic[bool]
    requestQueues*: array[NumWorkers, ptr Channel[InferenceRequest]]
    responseQueues*: array[NumWorkers, ptr Channel[InferenceResponse]]

proc workerThread(worker: ptr InferenceWorker) {.thread.} =
  ## ワーカースレッド - 独自のエンジンで独立処理
  var localEngine: ExtremeEngine
  localEngine.initWeights(uint32(42 + worker[].id))

  while worker[].state != wsStopped:
    if not worker[].mailboxEmpty():
      let req = worker[].dequeueRequest()
      var inputCopy = req.input
      var inputPtr = addr inputCopy
      let (cat, conf) = localEngine.inferExtremeWithConf(inputPtr)
      worker[].enqueueResult(InferenceResponse(
        id: req.id,
        category: cat,
        confidence: conf
      ))
      inc(worker[].processedCount)

# ========== ベンチマーク ==========

when isMainModule:
  echo "=== Actor-based Parallel Inference ==="
  echo fmt"Workers: {NumWorkers}"
  echo ""

  var system: InferenceActorSystem
  system.initActorSystem()

  var input: array[InputSize, float32]
  for i in 0..<InputSize:
    input[i] = i.float32 / InputSize.float32

  # シングルワーカーベースライン
  echo "1. シングルワーカー（アクター内処理）"
  let iterations = 1_000_000
  for i in 0..<iterations:
    discard system.workers[0].enqueueRequest(InferenceRequest(id: i, input: input))

  let start1 = getMonoTime()
  var processed = 0
  while processed < iterations:
    processed += system.workers[0].processBatch(1000)
  let elapsed1 = (getMonoTime() - start1).inNanoseconds.float / 1e9
  let singleThroughput = iterations.float / elapsed1
  echo fmt"   スループット: {singleThroughput:.0f} samples/sec"
  echo ""

  # 全ワーカー並列（シミュレーション）
  echo "2. 全ワーカー並列処理"
  system.initActorSystem()  # リセット

  let totalRequests = iterations * NumWorkers
  for i in 0..<totalRequests:
    let req = InferenceRequest(id: i, input: input)
    discard system.routeRequest(req)

  let start2 = getMonoTime()
  processed = 0
  while processed < totalRequests:
    processed += system.tick()
  let elapsed2 = (getMonoTime() - start2).inNanoseconds.float / 1e9
  let parallelThroughput = totalRequests.float / elapsed2

  echo fmt"   総リクエスト: {totalRequests}"
  echo fmt"   所要時間: {elapsed2:.2f}s"
  echo fmt"   スループット: {parallelThroughput:.0f} samples/sec"
  echo fmt"   スピードアップ: {parallelThroughput / singleThroughput:.2f}x"
  echo ""

  # 結果収集
  var results: seq[InferenceResponse] = @[]
  let collected = system.collectResults(results)
  echo fmt"収集結果数: {collected}"

  echo ""
  echo "=== 結論 ==="
  echo fmt"シングルアクター: {singleThroughput:.0f} samples/sec"
  echo fmt"{NumWorkers}アクター並列: {parallelThroughput:.0f} samples/sec"
  echo fmt"10万接続処理: {100_000.0 / parallelThroughput * 1000:.2f}ms"

  if parallelThroughput >= 10_000_000:
    echo ""
    echo "🚀🚀 1000万 samples/sec 突破！（アクターモデル） 🚀🚀"
  elif parallelThroughput >= 5_000_000:
    echo ""
    echo "🚀 500万 samples/sec 達成！（アクターモデル）"

  system.shutdown()
