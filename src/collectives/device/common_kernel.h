/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "devcomm.h"
#include <cstdio>
#include <cstdint>

#include <hip/hip_runtime.h>

// Define min for ssize_t
static __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

typedef uint64_t PackType;

#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)

template<class FUNC, typename T>
struct MULTI {
  __device__ PackType operator()(const PackType x, const PackType y) const
  {
    return FUNC()(x, y);
  }
};

#else

// unpack x and y to elements of type T and apply FUNC to each element
template<class FUNC, typename T>
struct MULTI {
  __device__ PackType operator()(const PackType x, const PackType y) const;
};

template<class FUNC>
struct MULTI<FUNC, int8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, uint8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, int32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(int32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      int32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, uint32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, half> {
  static_assert(sizeof(PackType) == 4 * sizeof(half),
      "PackType must be four times the size of half.");

  struct PackHalf2 {
    half2 a, b;
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    struct PackHalf2 cx, cy, cr;
    cx = *(reinterpret_cast<const struct PackHalf2*>(&x));
    cy = *(reinterpret_cast<const struct PackHalf2*>(&y));

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return *(reinterpret_cast<PackType*>(&cr));
  }
};

template<class FUNC>
struct MULTI<FUNC, float> {
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType storage;
    struct {
      float a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, double> {
  static_assert(sizeof(PackType) == sizeof(double),
      "PackType must be the same size as double.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    double rv = FUNC()(__longlong_as_double(x), __longlong_as_double(y));
    return __double_as_longlong(rv);
  }
};

template<class FUNC>
struct MULTI<FUNC, uint64_t> {
  static_assert(sizeof(PackType) == sizeof(uint64_t),
      "PackType must be the same size as uint64_t.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    uint64_t rv = FUNC()(x, y);
    return rv;
  }
};

template<class FUNC>
struct MULTI<FUNC, int64_t> {
  static_assert(sizeof(PackType) == sizeof(int64_t),
      "PackType must be the same size as int64_t.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    int64_t rv = FUNC()((int64_t)x, (int64_t)y);
    return rv;
  }
};

#endif //defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)

template<typename T> inline __device__
T vFetch(const volatile T* ptr) {
  return *ptr;
}

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) {
  *ptr = val;
}

#if CUDART_VERSION < 9000 && !(defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__))
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r.x = ptr->x;
  return r;
}

template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ptr->x = val.x;
}
#else
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r = ((half*)ptr)[0];
  return r;
}

template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ((half*)ptr)[0] = val;
}
#endif

typedef ulong2 Pack128;

template<class FUNC, typename T>
struct MULTI128 {
  __device__ void operator()(Pack128& x, Pack128& y) {
    x.x = MULTI<FUNC, T>()(x.x, y.x);
    x.y = MULTI<FUNC, T>()(x.y, y.y);
  }
};

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  v.x = p->x;
  v.y = p->y;
#else
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
#endif
}
inline __device__ void Store128(Pack128* p, Pack128& v) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  p->x = v.x;
  p->y = v.y;
#else
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
#endif
}

template<class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ void ReduceCopyMulti(const int tid, const int nthreads,
    int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
    const int offset, const int N) {
  for (int idx = offset+tid; idx < offset+N; idx += nthreads) {
    T val = vFetch(srcs[0]+idx);
    #pragma unroll
    for (int i=1; i<MINSRCS; i++) val = FUNC()(val, vFetch(srcs[i]+idx));
    #pragma unroll 1
    for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) val = FUNC()(val, vFetch(srcs[i]+idx));

    #pragma unroll
    for (int i=0; i<MINDSTS; i++) vStore(dsts[i]+idx, val);
    #pragma unroll 1
    for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) vStore(dsts[i]+idx, val);
  }
}

#define WARP_SIZE 64

template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ void ReduceCopy128bMulti( const int w, const int nw, const int t,
    int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
    const int elemOffset, const int Npack) {
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  const Pack128* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
  Pack128* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;

  while (offset < Npack) {
    Pack128 vals[UNROLL];
    // Load and reduce
    for (int u = 0; u < UNROLL; ++u) Fetch128(vals[u], srcs[0]+u*WARP_SIZE);

    for (int i=1; i<MINSRCS; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }
    #pragma unroll 1
    for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }

    // Store
    for (int i = 0; i < MINDSTS; i++) {
      for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    #pragma unroll 1
    for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
      for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
    offset += inc;
  }
}

template <typename T>
__device__ int ptrAlign128(T* ptr) { return (uint64_t)ptr % alignof(Pack128); }

#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
// Use UNROLL 4 for 2 SRCs, 2 for the rest
#define AUTOUNROLL (UNROLL*(2/MINSRCS))
#else
// Try to limit consecutive load/stores to 8.
// Use UNROLL 8 when we have a single source and a single destination, 4 otherwise
#define AUTOUNROLL (UNROLL*(4/(MINDSTS+MINSRCS)))
#endif

template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ void ReduceOrCopyMulti(const int tid, const int nthreads,
    int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
    int N) {
  int Nrem = N;
  if (Nrem <= 0) return;

  int alignDiff = 0;
  int align = ptrAlign128(srcs[0]);
  #pragma unroll
  for (int i=1; i<MINSRCS; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  #pragma unroll
  for (int i=0; i<MINDSTS; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));
  for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));

  int Npreamble = alignDiff ? Nrem :
    N < alignof(Pack128) ? N :
    (alignof(Pack128) - align) % alignof(Pack128);

  // stage 1: preamble: handle any elements up to the point of everything coming
  // into alignment
  if (Npreamble) {
    ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble);
    Nrem -= Npreamble;
    if (Nrem == 0) return;
  }
  int offset = Npreamble;

  // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
  // assuming the pointers we have are all 128-bit alignable.
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  const int packFactor = sizeof(Pack128) / sizeof(T);

  // stage 2a: main loop
  int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
      * (AUTOUNROLL * WARP_SIZE); // round down
  int Nelem2a = Npack2a * packFactor;

  ReduceCopy128bMulti<FUNC, T, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2a);

  Nrem -= Nelem2a;
  if (Nrem == 0) return;
  offset += Nelem2a;

  // stage 2b: slightly less optimized for section when we don't have full
  // unrolling

  int Npack2b = Nrem / packFactor;
  int Nelem2b = Npack2b * packFactor;

  ReduceCopy128bMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2b);

  Nrem -= Nelem2b;
  if (Nrem == 0) return;
  offset += Nelem2b;

  // stage 2c: tail
  ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem);
}

#ifdef ENABLE_CHECKSUM

template<typename T>
using checksum_type = typename std::conditional<sizeof(T) >= sizeof(uint32_t), uint32_t,
  typename std::conditional<sizeof(T) == sizeof(uint16_t), uint16_t, uint8_t>::type>::type;

template<typename T>
__device__ uint64_t Checksum(const int tid, const int nthreads, const T* src, size_t N,
  uint64_t* tempA, uint64_t* tempB, uint64_t* tempC) {
  uint64_t A = 0, B = 0;
  using DataType = checksum_type<T>;
  size_t data_size = N * sizeof(T) / sizeof(DataType);
  size_t block_size = DIVUP(data_size, nthreads);
  DataType* input_data = (DataType*) src;
  const uint64_t MAX_CHECKSUM = sizeof(DataType) == 4 ? 92681 : 23726746;
  const uint64_t MOD_CHECKSUM = sizeof(DataType) == 4 ? 0xffffffff : (sizeof(DataType) == 2 ? 0xffff : 0xff);
  const uint8_t SHIFT_CHECKSUM = sizeof(DataType) == 4 ? 32 : (sizeof(DataType) == 2 ? 16 : 8);

  for (ssize_t j = 0; j < block_size; j += MAX_CHECKSUM) {
    for (int i = 0; i < MAX_CHECKSUM && j+i < block_size; i ++) {
      ssize_t data_offset = (i + j) * nthreads + tid;
      if (data_offset < data_size)
        A += input_data[data_offset];
      B += A;
    }
    A %= MOD_CHECKSUM;
    B %= MOD_CHECKSUM;
  }
  tempA[tid] = A;
  tempB[tid] = B;
  tempC[tid] = A * tid;

  __syncthreads();

  int i = nthreads / 2;
  while (i != 0) {
      if (tid < i) {
          tempA[tid] += tempA[tid + i];
          tempB[tid] += tempB[tid + i];
          tempC[tid] += tempC[tid + i];
      }
      __syncthreads();
      i /= 2;
  }

  if (tid == 0) {
    uint64_t Al = tempA[0], Bl = tempB[0];
    Bl = Bl * nthreads - tempC[0];
    Bl %= MOD_CHECKSUM;
    Al %= MOD_CHECKSUM;
    return (Bl << SHIFT_CHECKSUM) | Al;
  }
  return 0;
}

#endif

#endif // COMMON_KERNEL_H_
