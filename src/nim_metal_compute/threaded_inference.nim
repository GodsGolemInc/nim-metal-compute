## Threaded Inference Engine v3
## 壁時計時間で正確測定

import std/[cpuinfo, times, strformat, monotimes]
import ./extreme_inference

const NumThreads = 8

type
  PaddedWorker = object
    engine: ExtremeEngine
    input: array[InputSize, float32]
    iterations: int
    pad: array[128, byte]

var
  workers: array[NumThreads, Thread[ptr PaddedWorker]]
  workerData: array[NumThreads, PaddedWorker]

proc workerProc(data: ptr PaddedWorker) {.thread.} =
  for _ in 0..<data.iterations:
    discard data.engine.inferExtreme(addr data.input)

when isMainModule:
  echo "=== Threaded Inference v3 (Wall-Clock) ==="
  echo fmt"Physical Cores: {countProcessors()}"
  echo fmt"Using Threads: {NumThreads}"
  echo ""

  let iterationsPerThread = 10_000_000
  for i in 0..<NumThreads:
    workerData[i].engine.initWeights(uint32(42 + i))
    for j in 0..<InputSize:
      workerData[i].input[j] = j.float32 / InputSize.float32
    workerData[i].iterations = iterationsPerThread

  # シングルスレッド
  echo "1. シングルスレッド"
  var singleWorker: PaddedWorker
  singleWorker.engine.initWeights(42)
  for j in 0..<InputSize:
    singleWorker.input[j] = j.float32 / InputSize.float32

  let start1 = getMonoTime()
  for _ in 0..<iterationsPerThread:
    discard singleWorker.engine.inferExtreme(addr singleWorker.input)
  let elapsed1 = (getMonoTime() - start1).inNanoseconds.float / 1e9
  let singleThroughput = iterationsPerThread.float / elapsed1
  echo fmt"   所要時間: {elapsed1:.2f}s"
  echo fmt"   スループット: {singleThroughput:.0f} samples/sec"
  echo fmt"   レイテンシ: {elapsed1 / iterationsPerThread.float * 1e9:.0f} ns"
  echo ""

  # マルチスレッド（壁時計時間）
  echo fmt"2. {NumThreads}スレッド並列"
  let start2 = getMonoTime()
  for i in 0..<NumThreads:
    createThread(workers[i], workerProc, addr workerData[i])
  for i in 0..<NumThreads:
    joinThread(workers[i])
  let elapsed2 = (getMonoTime() - start2).inNanoseconds.float / 1e9

  let totalIterations = iterationsPerThread * NumThreads
  let parallelThroughput = totalIterations.float / elapsed2
  let speedup = parallelThroughput / singleThroughput

  echo fmt"   総反復回数: {totalIterations}"
  echo fmt"   所要時間: {elapsed2:.2f}s"
  echo fmt"   スループット: {parallelThroughput:.0f} samples/sec"
  echo fmt"   スピードアップ: {speedup:.2f}x"
  echo ""

  echo "=== 最終結果 ==="
  echo fmt"シングルコア: {singleThroughput:.0f} samples/sec"
  echo fmt"{NumThreads}コア並列: {parallelThroughput:.0f} samples/sec"
  echo fmt"10万接続処理: {100_000.0 / parallelThroughput * 1000:.2f}ms"

  if parallelThroughput >= 15_000_000:
    echo ""
    echo "🚀🚀🚀 1500万 samples/sec 突破！ 🚀🚀🚀"
  elif parallelThroughput >= 10_000_000:
    echo ""
    echo "🚀🚀 1000万 samples/sec 突破！ 🚀🚀"
  elif parallelThroughput >= 5_000_000:
    echo ""
    echo "🚀 500万 samples/sec 達成！"
