/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef MSSCLKERNELIMPL_H
#define MSSCLKERNELIMPL_H

#include "device.h"
#include "primitives.h"
#include "collectives.h"

#include "msccl/msccl_struct.h"
#include "msccl/msccl_kernel.h"

extern __shared__ struct mscclShmemData mscclShmem;

#define MSCCL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
  MSCCL_MAX_ITER*MSCCL_MAX_NUM_STEPS*(uint64_t)__WORKINDEX__ + ((uint64_t)__GRIDOFFSET_ITER__ * MSCCL_MAX_NUM_STEPS + (uint64_t)__STEP__)

#define GET_WORKINDEX_FROM_FLAG(__FLAG__) \
  (__FLAG__) / (MSCCL_MAX_ITER*MSCCL_MAX_NUM_STEPS)

#ifdef ENABLE_COLLTRACE
  #define INC_COLL_TRACE \
    uint32_t pos = atomicAdd(&ncclShmem.collTraceTail->tail, 1)%COLLTRACE_NUM_ITEMS; \
    struct ncclCollTrace* collTrace = ncclShmem.collTrace+pos; \
    collTrace->timeStamp = wall_clock64(); \
    collTrace->bid = blockIdx.x;
    // TODO: switch to atomicInc after llvm crash is fixed
    // uint32_t pos = atomicInc(&ncclShmem.collTraceTail->tail, COLLTRACE_NUM_ITEMS)

  #define traceData(data2, data4, data8_0, data8_1) { \
    INC_COLL_TRACE \
    collTrace->funcIndex = data2; \
    collTrace->data_0 = data4; \
    collTrace->opCount = data8_0; \
    collTrace->data_1 = data8_1; \
    collTrace->type = ncclCollTraceDataType; \
  }
#else
#define traceData(data2, data4, data8_0, data8_1)
#endif

inline __device__ static void barrier(int nthreads) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HCC__) || defined(__HIPCC__)
  assert(nthreads == NCCL_MAX_NTHREADS);
  __asm__ __volatile__("s_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier");
#else
  asm volatile ("bar.sync %1, %0;" :: "r"(nthreads), "r"(15));
#endif
}

// Copy 8-byte aligned data. You must call with at least `(bytes+7)/8` threads.
inline __device__ static void copyToShmem8(int tid, void* dst, void const* src, int bytes) {
  int offset = sizeof(uint32_t) * tid;
  if (offset < bytes) {
    uint32_t *src2 = (uint32_t*)((char const*)src + offset);
    uint32_t *dst2 = (uint32_t*)((char*)dst + offset);
    *dst2 = *src2;
    offset += WARP_SIZE*sizeof(uint32_t);
  }
}

__device__ __forceinline__ static void threadBlockCopy(
  uint32_t *dst, uint32_t const *src, uint64_t size, int tid, int nthreads) {
  for (int i = tid; i < size; i += nthreads) {
    dst[i] = src[i];
  }
}

#define MSCCL_REDUCE_UNROLL_LOOP_A(numloops, BytePerPack) \
  for (int r = 0; r < numloops; r++) { \
    srcOffset = srcBaseOffset + (ssize_t)mscclShmem.mscclTB.reductionSrcOffsets[t->reductionPointer+r] * sizePerMscclChunk; \
    reduceInput = ld_volatile_global<BytePerPack>((uintptr_t)(srcPointer + srcOffset)); \
    o = applyReduce(redFn, reduceInput, o); \
  }

template<typename T, typename RedOp, int BytePerPack>
__device__ __forceinline__ static void mscclReduce(int c, int numReductions, int currIdx, ssize_t sizePerMscclChunk, RedOp redFn,
  struct mscclTransmission* t, ssize_t gridOffset, ssize_t &srcOffset, ssize_t dstOffset, T *srcPointer, T *dstPointer) {
  const int elemsPerPack = BytePerPack/sizeof(T);
  T* dstIndex = dstPointer + dstOffset + currIdx*elemsPerPack;
  BytePack<BytePerPack> reduceInput;
  BytePack<BytePerPack> o = ld_volatile_global<BytePerPack>((uintptr_t)dstIndex);
  ssize_t srcBaseOffset = gridOffset + (ssize_t)c * sizePerMscclChunk + currIdx*elemsPerPack;
  switch (numReductions) {
    case 7:
      #pragma unroll
      MSCCL_REDUCE_UNROLL_LOOP_A(7, BytePerPack);
      break;
#if defined(__gfx90a__)
    case 15:
      #pragma unroll
      MSCCL_REDUCE_UNROLL_LOOP_A(15, BytePerPack);
      break;
#endif
    default:
      MSCCL_REDUCE_UNROLL_LOOP_A(numReductions, BytePerPack);
      break;
  }
  st_global<BytePerPack>((uintptr_t)dstIndex, o);
}


template<typename T, typename RedOp, typename Proto, bool fullOps>
__device__ __forceinline__ void mscclRunInterpreter(
  struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork* work) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = NCCL_MAX_NTHREADS;

#if defined(ENABLE_NPKIT)
  uint64_t timestamp_entry = 0;
  if (tid == 0) {
     timestamp_entry = NPKIT_GET_GPU_TIMESTAMP();
  }
#endif
  // initialize mscclShmem.mscclTB
  threadBlockCopy(
    (uint32_t *)&mscclShmem.mscclTB, (uint32_t *)(algo->mscclTBs + bid),
    sizeof(struct mscclThreadBlock) / sizeof(uint32_t), tid, nthreads);
  __synclds(); // publish mscclShmem.mscclTB.channelId

  // initialize ncclShmem and mscclShmem.work
  int channelId = mscclShmem.mscclTB.channelId;
  {
    void *dst, *src;
    int bytes = 0;
    // Use first 3 warps to load comm, channel, and work into shmem
    switch (tid/WARP_SIZE) {
    case 0:
      dst = &ncclShmem.comm;
      src = comm;
      bytes = sizeof(ncclDevComm);
      break;
    case 1:
      // Get address of channel without incurring indirect load from ncclDevComm::channels
      dst = &ncclShmem.channel;
      src = &((ncclDevCommAndChannels*)comm)->channels[channelId];
      bytes = sizeof(ncclDevChannel);
      break;
    case 2:
      dst = &mscclShmem.work;
      src = work + blockIdx.x;
      bytes = sizeof(mscclWork);
      break;
    case 3:
      /* set abort flag to 0 */
      if (tid%WARP_SIZE == 0) ncclShmem.aborted = 0;
#ifdef ENABLE_COLLTRACE
      else if (tid%WARP_SIZE == 1) ncclShmem.collTrace = comm->collTrace + COLLTRACE_NUM_ITEMS*channelId;
      else if (tid%WARP_SIZE == 2) ncclShmem.collTraceTail = comm->collTraceTail + channelId;
#endif
      break;
    default:
      break;
    }
    if (bytes) copyToShmem8(tid%WARP_SIZE, dst, src, bytes);
  }

#if defined(ENABLE_NPKIT)
  int npKitCtxIdx = bid;
  int xcc_id = 0;
  if (tid == 0) {
    ncclShmem.event_buffer_head = 0;
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s" (xcc_id));
#endif
  }
#endif
  __synclds(); // publish shmem

  if (tid == 0)
    *mscclShmem.work.workFifoDone = mscclShmem.work.workFifoDoneAck;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
  if (tid == 0) {
    uint64_t* cpuTimestamp = ncclShmem.comm.cpuTimestamp;
    NpKit::CollectGpuEventLDS(NPKIT_EVENT_TIME_SYNC_CPU, 0, xcc_id, *cpuTimestamp);
  }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
  if (tid == 0) {
    NpKit::CollectGpuEventLDS(NPKIT_EVENT_TIME_SYNC_GPU, 0, xcc_id, NPKIT_GET_GPU_TIMESTAMP());
  }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_RUN_ENTRY)
  if (tid == 0) {
    NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_RUN_ENTRY, mscclShmem.work.sizePerMscclChunk*mscclShmem.work.nChunksPerLoop, xcc_id, timestamp_entry);
  }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_MSCCL_RUN_EXIT)
  if (tid == 0) {
    NpKit::CollectGpuEventLDS(NPKIT_EVENT_MSCCL_RUN_EXIT, mscclShmem.work.sizePerMscclChunk*mscclShmem.work.nChunksPerLoop, xcc_id, NPKIT_GET_GPU_TIMESTAMP());
  }
#endif

#if defined(ENABLE_NPKIT)
  __synclds();
  NpKitEventCollectContext* ctx = ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx;
  copyToShmem16(tid, ctx->event_buffer+ctx->event_buffer_head, ncclShmem.event_buffer, sizeof(NpKitEvent)*ncclShmem.event_buffer_head);
  if (tid == 0) ctx->event_buffer_head += ncclShmem.event_buffer_head;
#endif
}

#define MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, type, fullOps) \
__global__ void MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL, fullOps)(struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork* work) { \
  mscclRunInterpreter<type, Func##devredop<type>, ProtoLL, fullOps>(comm, algo, work); \
} \
__global__ void MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL128, fullOps)(struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork* work) { \
  mscclRunInterpreter<type, Func##devredop<type>, ProtoLL128, fullOps>(comm, algo, work); \
} \
__global__ void MSCCL_KERNEL_ENTRY_NAME(devredop, type, Simple, fullOps)(struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork* work) { \
  mscclRunInterpreter<type, Func##devredop<type>, ProtoSimple<MSCCL_CHUNKSTEPS/MSCCL_SLICESTEPS, MSCCL_SLICESTEPS>, fullOps>(comm, algo, work); \
}

#define MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(devredop, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int8_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint8_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int32_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint32_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int64_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint64_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, half, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, float, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, double, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, hip_bfloat16, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, rccl_float8, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, rccl_bfloat8, fullOps)

#define MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_NOFLOAT(devredop, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int8_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint8_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int32_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint32_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int64_t, fullOps) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint64_t, fullOps)

#define MSCCL_IMPL_KERNEL_ENTRY_FUNC() \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(Sum, false) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(Prod, false) \
  MSCCL_IMPL_KERNEL_ENTRY_FUNC_DEVREDOP(MinMax, false)

#endif
