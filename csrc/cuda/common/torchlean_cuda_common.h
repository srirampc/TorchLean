#pragma once

#include <lean/lean.h>

#include "torchlean_size_common.h"
#include "torchlean_cuda_thread.h"

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Shared CUDA runtime bits.
// Wrappers validate shapes before launch; `checkCuda` is only for CUDA driver/runtime failures.

static inline void checkCuda(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    char detail[512];
    snprintf(detail, sizeof(detail), "%s: %s (%d)", msg, cudaGetErrorString(e), (int)e);
    lean_internal_panic(detail);
  }
}

// CUDA records the last asynchronous launch error per host thread. Clear it immediately before a
// launch when the next check is meant to diagnose that launch, not an older unchecked kernel.
static inline void torchlean_cuda_clear_pending_error() {
  (void)cudaGetLastError();
}

// Check the launch error recorded by the kernel that was just submitted.
static inline void torchlean_cuda_check_launch(const char* msg) {
  checkCuda(cudaGetLastError(), msg);
}

static inline void torchlean_cuda_free_checked(void** ptr, const char* msg) {
  if (ptr && *ptr) {
    checkCuda(cudaFree(*ptr), msg);
    *ptr = nullptr;
  }
}

static inline void torchlean_cuda_lock(torchlean_cuda_mutex_t* mutex, const char* msg) {
  if (torchlean_cuda_mutex_lock(mutex) != 0) {
    lean_internal_panic(msg);
  }
}

static inline void torchlean_cuda_unlock(torchlean_cuda_mutex_t* mutex, const char* msg) {
  if (torchlean_cuda_mutex_unlock(mutex) != 0) {
    lean_internal_panic(msg);
  }
}

struct torchlean_cuda_scratch_block {
  size_t bytes;
  void* ptr;
  cudaEvent_t ready;
};

static torchlean_cuda_mutex_t g_torchlean_cuda_scratch_mutex = TORCHLEAN_CUDA_MUTEX_INITIALIZER;
static torchlean_cuda_scratch_block* g_torchlean_cuda_scratch_cache = nullptr;
static size_t g_torchlean_cuda_scratch_count = 0;
static size_t g_torchlean_cuda_scratch_cap = 0;

static inline void torchlean_cuda_scratch_push(torchlean_cuda_scratch_block block) {
  if (g_torchlean_cuda_scratch_count == g_torchlean_cuda_scratch_cap) {
    size_t new_cap = g_torchlean_cuda_scratch_cap == 0 ? 16 : g_torchlean_cuda_scratch_cap * 2;
    if (new_cap < g_torchlean_cuda_scratch_cap) {
      lean_internal_panic("torchlean_cuda_scratch_push: capacity overflow");
    }
    const size_t bytes = checked_bytes_size(
        new_cap, sizeof(torchlean_cuda_scratch_block),
        "torchlean_cuda_scratch_push: cache byte size overflow");
    void* next = realloc(g_torchlean_cuda_scratch_cache,
                         bytes);
    if (!next) {
      lean_internal_panic_out_of_memory();
    }
    g_torchlean_cuda_scratch_cache = (torchlean_cuda_scratch_block*)next;
    g_torchlean_cuda_scratch_cap = new_cap;
  }
  g_torchlean_cuda_scratch_cache[g_torchlean_cuda_scratch_count++] = block;
}

static inline void torchlean_cuda_scratch_flush(void) {
  torchlean_cuda_lock(&g_torchlean_cuda_scratch_mutex, "mutex lock scratch flush failed");
  torchlean_cuda_scratch_block* blocks = g_torchlean_cuda_scratch_cache;
  size_t count = g_torchlean_cuda_scratch_count;
  g_torchlean_cuda_scratch_cache = nullptr;
  g_torchlean_cuda_scratch_count = 0;
  g_torchlean_cuda_scratch_cap = 0;
  torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex, "mutex unlock scratch flush failed");

  for (size_t i = 0; i < count; ++i) {
    torchlean_cuda_scratch_block block = blocks[i];
    if (block.ready) {
      checkCuda(cudaEventSynchronize(block.ready), "cudaEventSynchronize cached scratch block failed");
      checkCuda(cudaEventDestroy(block.ready), "cudaEventDestroy cached scratch block failed");
    }
    torchlean_cuda_free_checked(&block.ptr, "cudaFree cached scratch block failed");
  }
  free(blocks);
}

static inline void* torchlean_cuda_scratch_alloc_bytes(size_t bytes, const char* msg) {
  if (bytes == 0) {
    return nullptr;
  }
  torchlean_cuda_lock(&g_torchlean_cuda_scratch_mutex, "mutex lock scratch alloc failed");
  for (size_t i = 0; i < g_torchlean_cuda_scratch_count; ++i) {
    if (g_torchlean_cuda_scratch_cache[i].bytes != bytes) {
      continue;
    }
    cudaError_t ready = cudaEventQuery(g_torchlean_cuda_scratch_cache[i].ready);
    if (ready == cudaSuccess) {
      void* ptr = g_torchlean_cuda_scratch_cache[i].ptr;
      checkCuda(cudaEventDestroy(g_torchlean_cuda_scratch_cache[i].ready),
                "cudaEventDestroy scratch reuse event failed");
      g_torchlean_cuda_scratch_cache[i] =
        g_torchlean_cuda_scratch_cache[g_torchlean_cuda_scratch_count - 1];
      g_torchlean_cuda_scratch_count--;
      torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex,
                            "mutex unlock scratch alloc failed");
      return ptr;
    }
    if (ready != cudaErrorNotReady) {
      torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex,
                            "mutex unlock scratch alloc error failed");
      checkCuda(ready, "cudaEventQuery scratch reuse event failed");
    }
  }
  torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex, "mutex unlock scratch alloc failed");
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, bytes);
  if (err != cudaSuccess) {
    torchlean_cuda_scratch_flush();
    err = cudaMalloc(&ptr, bytes);
  }
  checkCuda(err, msg);
  return ptr;
}

static inline void torchlean_cuda_scratch_free_bytes(void** ptr, size_t bytes, const char* msg) {
  (void)msg;
  if (!ptr || !*ptr) {
    return;
  }
  if (bytes == 0) {
    torchlean_cuda_free_checked(ptr, msg);
    return;
  }
  cudaEvent_t ready = nullptr;
  checkCuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
            "cudaEventCreate cached scratch block failed");
  checkCuda(cudaEventRecord(ready, 0), "cudaEventRecord cached scratch block failed");
  torchlean_cuda_lock(&g_torchlean_cuda_scratch_mutex, "mutex lock scratch free failed");
  torchlean_cuda_scratch_push({bytes, *ptr, ready});
  torchlean_cuda_unlock(&g_torchlean_cuda_scratch_mutex, "mutex unlock scratch free failed");
  *ptr = nullptr;
}

template <typename T>
static inline T* torchlean_cuda_scratch_alloc(size_t count, const char* msg) {
  return (T*)torchlean_cuda_scratch_alloc_bytes(checked_bytes_size(count, sizeof(T), msg), msg);
}

template <typename T>
static inline void torchlean_cuda_scratch_free(T** ptr, size_t count, const char* msg) {
  torchlean_cuda_scratch_free_bytes((void**)ptr, checked_bytes_size(count, sizeof(T), msg), msg);
}
