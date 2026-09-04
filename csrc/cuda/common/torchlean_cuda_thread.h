#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef SRWLOCK torchlean_cuda_mutex_t;
#define TORCHLEAN_CUDA_MUTEX_INITIALIZER SRWLOCK_INIT
static inline int torchlean_cuda_mutex_lock(torchlean_cuda_mutex_t* m) {
  AcquireSRWLockExclusive(m); return 0;
}
static inline int torchlean_cuda_mutex_unlock(torchlean_cuda_mutex_t* m) {
  ReleaseSRWLockExclusive(m); return 0;
}

typedef INIT_ONCE torchlean_cuda_once_t;
#define TORCHLEAN_CUDA_ONCE_INIT INIT_ONCE_STATIC_INIT
static BOOL CALLBACK torchlean_cuda_once_trampoline(
    PINIT_ONCE once, PVOID param, PVOID* ctx) {
  (void)once; (void)ctx;
  ((void (*)(void))param)();
  return TRUE;
}
static inline void torchlean_cuda_once(
    torchlean_cuda_once_t* once, void (*fn)(void)) {
  InitOnceExecuteOnce(once, torchlean_cuda_once_trampoline, (PVOID)fn, NULL);
}
#else
#include <pthread.h>

typedef pthread_mutex_t torchlean_cuda_mutex_t;
#define TORCHLEAN_CUDA_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#define torchlean_cuda_mutex_lock pthread_mutex_lock
#define torchlean_cuda_mutex_unlock pthread_mutex_unlock

typedef pthread_once_t torchlean_cuda_once_t;
#define TORCHLEAN_CUDA_ONCE_INIT PTHREAD_ONCE_INIT
#define torchlean_cuda_once pthread_once
#endif
