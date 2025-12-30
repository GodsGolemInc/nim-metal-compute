## Metal Buffer Pool
## v0.0.6: Efficient buffer reuse for GPU operations
##
## Buffer allocation is expensive. This module provides:
## - Buffer pooling for reuse
## - Size-based bucketing
## - Automatic cleanup

import std/[tables, algorithm, strformat]
import ./errors
import ./metal_device
import ./metal_buffer
import ./metal_wrapper

type
  BufferPool* = object
    ## Pool of reusable Metal buffers
    device*: MetalDevice
    pools: Table[int, seq[MetalBuffer]]  # Size bucket -> available buffers
    inUse: seq[MetalBuffer]              # Currently allocated buffers
    maxPoolSize: int                      # Max buffers per bucket
    totalAllocated: int
    totalReused: int
    valid*: bool

  PooledBuffer* = object
    ## A buffer borrowed from the pool
    buffer*: MetalBuffer
    pool: ptr BufferPool
    returned: bool

# Size buckets for efficient pooling
const
  SizeBuckets = [
    1024,           # 1 KB
    4096,           # 4 KB
    16384,          # 16 KB
    65536,          # 64 KB
    262144,         # 256 KB
    1048576,        # 1 MB
    4194304,        # 4 MB
    16777216,       # 16 MB
    67108864,       # 64 MB
    268435456       # 256 MB
  ]

proc findBucket(size: int): int =
  ## Find the appropriate size bucket for a given size
  for bucket in SizeBuckets:
    if size <= bucket:
      return bucket
  # For sizes larger than max bucket, use exact size
  return size

# ========== Buffer Pool Management ==========

proc newBufferPool*(device: MetalDevice, maxPoolSize: int = 10): NMCResult[BufferPool] =
  ## Create a new buffer pool for a device
  if not device.valid:
    return err[BufferPool](ekDevice, "Invalid device")

  result = ok(BufferPool(
    device: device,
    pools: initTable[int, seq[MetalBuffer]](),
    inUse: @[],
    maxPoolSize: maxPoolSize,
    totalAllocated: 0,
    totalReused: 0,
    valid: true
  ))

proc acquire*(pool: var BufferPool, size: int,
              mode: MTLStorageMode = smShared): NMCResult[PooledBuffer] =
  ## Acquire a buffer from the pool (or allocate new one)
  if not pool.valid:
    return err[PooledBuffer](ekBuffer, "Buffer pool is not valid")

  let bucket = findBucket(size)

  # Check if we have a buffer available in this bucket
  if bucket in pool.pools and pool.pools[bucket].len > 0:
    var buffer = pool.pools[bucket].pop()
    pool.inUse.add(buffer)
    pool.totalReused.inc
    return ok(PooledBuffer(
      buffer: buffer,
      pool: addr pool,
      returned: false
    ))

  # Allocate a new buffer
  let bufferResult = pool.device.newBuffer(bucket, mode)
  if not bufferResult.isOk:
    return err[PooledBuffer](bufferResult.error)

  var buffer = bufferResult.get
  pool.inUse.add(buffer)
  pool.totalAllocated.inc

  result = ok(PooledBuffer(
    buffer: buffer,
    pool: addr pool,
    returned: false
  ))

proc release*(pooled: var PooledBuffer) =
  ## Return a buffer to the pool for reuse
  if pooled.returned:
    return

  pooled.returned = true

  if pooled.pool == nil or not pooled.pool[].valid:
    # Pool is gone, just release the buffer
    pooled.buffer.release()
    return

  let bucket = findBucket(pooled.buffer.length)

  # Remove from in-use list
  let idx = pooled.pool[].inUse.find(pooled.buffer)
  if idx >= 0:
    pooled.pool[].inUse.delete(idx)

  # Add to pool if not full
  if bucket notin pooled.pool[].pools:
    pooled.pool[].pools[bucket] = @[]

  if pooled.pool[].pools[bucket].len < pooled.pool[].maxPoolSize:
    pooled.pool[].pools[bucket].add(pooled.buffer)
  else:
    # Pool is full, release the buffer
    pooled.buffer.release()

proc clear*(pool: var BufferPool) =
  ## Release all buffers in the pool
  # Release pooled buffers
  for bucket, buffers in pool.pools.mpairs:
    for buffer in buffers.mitems:
      buffer.release()
  pool.pools.clear()

  # Release in-use buffers (shouldn't happen normally)
  for buffer in pool.inUse.mitems:
    buffer.release()
  pool.inUse.setLen(0)

proc destroy*(pool: var BufferPool) =
  ## Destroy the buffer pool
  pool.clear()
  pool.valid = false

proc stats*(pool: BufferPool): string =
  ## Get pool statistics
  var pooledCount = 0
  for bucket, buffers in pool.pools:
    pooledCount += buffers.len

  let total = pool.totalAllocated + pool.totalReused
  let reuseRate = if total > 0: pool.totalReused * 100 div total else: 0

  result = fmt"""BufferPool Stats:
  Total Allocated: {pool.totalAllocated}
  Total Reused:    {pool.totalReused}
  Currently In Use: {pool.inUse.len}
  Currently Pooled: {pooledCount}
  Reuse Rate:      {reuseRate}%"""

# ========== Convenience Functions ==========

proc `$`*(pool: BufferPool): string =
  if not pool.valid:
    return "BufferPool(invalid)"
  result = fmt"BufferPool(allocated: {pool.totalAllocated}, reused: {pool.totalReused})"

proc `$`*(pooled: PooledBuffer): string =
  if pooled.returned:
    return "PooledBuffer(returned)"
  result = fmt"PooledBuffer(size: {pooled.buffer.length})"

# ========== Test ==========

when isMainModule:
  echo "=== Metal Buffer Pool Test ==="
  echo ""

  let deviceResult = getDefaultDevice()
  if not deviceResult.isOk:
    echo "Error: ", deviceResult.error
    quit(1)

  let device = deviceResult.get
  echo "Device: ", device.info.name
  echo ""

  # Create buffer pool
  let poolResult = newBufferPool(device)
  if not poolResult.isOk:
    echo "Pool error: ", poolResult.error
    quit(1)

  var pool = poolResult.get
  echo "Pool created: ", pool
  echo ""

  # Test buffer acquisition and release
  echo "--- Acquiring buffers ---"
  var buffers: seq[PooledBuffer] = @[]

  for i in 1..5:
    let size = i * 1024
    let bufResult = pool.acquire(size)
    if bufResult.isOk:
      buffers.add(bufResult.get)
      echo fmt"  Acquired buffer {i}: {size} bytes -> bucket {findBucket(size)}"

  echo ""
  echo pool.stats()
  echo ""

  # Release buffers
  echo "--- Releasing buffers ---"
  for buffer in buffers.mitems:
    buffer.release()
  buffers.setLen(0)

  echo pool.stats()
  echo ""

  # Acquire again (should reuse)
  echo "--- Acquiring again (should reuse) ---"
  for i in 1..5:
    let size = i * 1024
    let bufResult = pool.acquire(size)
    if bufResult.isOk:
      buffers.add(bufResult.get)
      echo fmt"  Acquired buffer {i}: {size} bytes"

  echo ""
  echo pool.stats()
  echo ""

  # Cleanup
  for buffer in buffers.mitems:
    buffer.release()
  pool.destroy()

  echo "✅ Metal buffer pool test complete"
