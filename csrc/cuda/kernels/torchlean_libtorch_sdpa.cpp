// LibTorch SDPA forward/backward bridge (g++; nvcc cannot parse torch headers).
#include <lean/lean.h>
#include <torch/torch.h>
#include "torchlean_cuda_buffer.h"
#include <cuda_runtime.h>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <ATen/cuda/CUDAContext.h>

// Initialize Torch's CUDA interface up front through device-query APIs.
//
// Why eager initialization? In Windows, Torch's lazy CUDA init fails at a
// TORCH_CHECK (due to mixed-ABI), when reached via at::from_blob -> TensorMaker.
static void torchlean_ensure_torch_cuda() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    (void)at::cuda::is_available();
    (void)at::cuda::getCurrentDeviceProperties();
  });
}

// ----- Error Forwarding to Lean -----

// C-only error builder, safe to call from Windows `__except` handler (no C++
// destructor, no unwinding requirements in the mixed-ABI image).
static lean_obj_res c_ioError(const char* message) {
  return lean_io_result_mk_error(
      lean_mk_io_error_other_error(1, lean_mk_string(message)));
}

static lean_obj_res ioError(const std::string& message) {
  return c_ioError(message.c_str());
}

// ----- Validation -----
// std::optional<std::string> semantics: nullopt = success; an engaged value is
// the failure message. Never throws: in the mixed-ABI Windows exe a C++ exception
// thrown at the FFI boundary crashes silently, so validation failure MUST surface
// as a Lean error.

static bool checkedMul(size_t a, size_t b, size_t* out) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
  *out = a * b;
  return true;
}

static std::optional<std::string> checkedElements(uint32_t batch, uint32_t n, uint32_t d,
                                                  const char* what, size_t* out) {
  size_t bn = 0;
  size_t total = 0;
  if (!checkedMul((size_t)batch, (size_t)n, &bn) || !checkedMul(bn, (size_t)d, &total)) {
    return std::string("LibTorch SDPA: ") + what + " size overflow";
  }
  *out = total;
  return std::nullopt;
}

static std::optional<std::string> requireSize(b_lean_obj_arg object, size_t expected,
                                              const char* name) {
  const size_t actual = torchlean_cuda_buffer_unbox(object)->size;
  if (actual != expected) {
    return std::string("LibTorch SDPA: ") + name + " buffer size mismatch (expected " +
           std::to_string(expected) + ", got " + std::to_string(actual) + ")";
  }
  return std::nullopt;
}

static std::optional<std::string> validateInputs(b_lean_obj_arg Q, b_lean_obj_arg K,
                                                 b_lean_obj_arg V, b_lean_obj_arg M,
                                                 uint32_t hasMask, uint32_t batch,
                                                 uint32_t n, uint32_t d,
                                                 b_lean_obj_arg DOut = nullptr) {
  if (hasMask > 1) return std::string("LibTorch SDPA: hasMask must be 0 or 1");
  size_t qkvElements = 0;
  if (auto e = checkedElements(batch, n, d, "Q/K/V", &qkvElements)) return e;
  if (auto e = requireSize(Q, qkvElements, "Q")) return e;
  if (auto e = requireSize(K, qkvElements, "K")) return e;
  if (auto e = requireSize(V, qkvElements, "V")) return e;
  if (DOut != nullptr) {
    if (auto e = requireSize(DOut, qkvElements, "dOut")) return e;
  }
  if (hasMask != 0) {
    size_t maskElements = 0;
    if (auto e = checkedElements(batch, n, n, "mask", &maskElements)) return e;
    if (auto e = requireSize(M, maskElements, "mask")) return e;
  }
  return std::nullopt;
}
// ----- Validation -----

static at::Tensor view3(b_lean_obj_arg o, uint32_t batch, int64_t n, int64_t d) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  return at::from_blob(torchlean_cuda_buffer_unbox(o)->data,
                       {static_cast<int64_t>(batch), n, d}, options);
}

static c10::optional<at::Tensor> attnMask(b_lean_obj_arg M, uint32_t hasMask, uint32_t batch,
                                        uint32_t n) {
  if (!hasMask) return c10::nullopt;
  // PyTorch SDPA boolean masks use true for entries that participate in attention.
  return c10::optional<at::Tensor>(view3(M, batch, n, n).to(at::kBool));
}

static lean_obj_res torchlean_libtorch_sdpa_fwd_impl(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  try {
    torchlean_ensure_torch_cuda();
    if (auto err = validateInputs(Q, K, V, M, hasMask, batch, n, d)) {
      return ioError(std::string("LibTorch SDPA forward failed: ") + *err);
    }
    auto mask = attnMask(M, hasMask, batch, n);
    auto y = at::scaled_dot_product_attention(view3(Q, batch, n, d), view3(K, batch, n, d),
                                              view3(V, batch, n, d), mask, 0., false, scale)
                 .contiguous();
    auto* out = torchlean_cuda_buffer_alloc((size_t)y.numel());
    const cudaError_t copy = cudaMemcpy(out->data, y.data_ptr<float>(),
                                       (size_t)y.numel() * sizeof(float),
                                       cudaMemcpyDeviceToDevice);
    if (copy != cudaSuccess) {
      torchlean_cuda_buffer_drop_unboxed(out);
      return ioError(std::string("LibTorch SDPA forward copy failed: ") + cudaGetErrorString(copy));
    }
    return lean_io_result_mk_ok(torchlean_cuda_buffer_box(out));
  } catch (const c10::Error& e) {
    return ioError(std::string("LibTorch SDPA forward failed: ") + e.what_without_backtrace());
  } catch (const std::exception& e) {
    return ioError(std::string("LibTorch SDPA forward failed: ") + e.what());
  } catch (...) {
    return ioError("LibTorch SDPA forward failed with an unknown native exception");
  }
}

static lean_obj_res torchlean_libtorch_sdpa_bwd_impl(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M, b_lean_obj_arg DOut,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  try {
    torchlean_ensure_torch_cuda();
    if (auto err = validateInputs(Q, K, V, M, hasMask, batch, n, d, DOut)) {
      return ioError(std::string("LibTorch SDPA backward failed: ") + *err);
    }
    at::AutoGradMode enable_grad(true);
    auto mask = attnMask(M, hasMask, batch, n);
    auto q = view3(Q, batch, n, d).detach().requires_grad_(true);
    auto k = view3(K, batch, n, d).detach().requires_grad_(true);
    auto v = view3(V, batch, n, d).detach().requires_grad_(true);
    auto grad_out = view3(DOut, batch, n, d);
    auto out = at::scaled_dot_product_attention(q, k, v, mask, 0., false, scale);
    out.backward(grad_out);
    size_t numel = 0;
    if (auto err = checkedElements(batch, n, d, "gradient", &numel)) {
      return ioError(std::string("LibTorch SDPA backward failed: ") + *err);
    }
    at::Tensor grads[3] = {q.grad(), k.grad(), v.grad()};
    torchlean_cuda_buffer* bufs[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; ++i) bufs[i] = torchlean_cuda_buffer_alloc(numel);
    for (int i = 0; i < 3; ++i) {
      auto t = grads[i].contiguous();
      const cudaError_t copy = cudaMemcpy(bufs[i]->data, t.data_ptr<float>(),
                                         numel * sizeof(float), cudaMemcpyDeviceToDevice);
      if (copy != cudaSuccess) {
        for (auto* buffer : bufs) torchlean_cuda_buffer_drop_unboxed(buffer);
        return ioError(std::string("LibTorch SDPA backward copy failed: ") +
                       cudaGetErrorString(copy));
      }
    }
    return lean_io_result_mk_ok(torchlean_cuda_box_three_buffers(bufs[0], bufs[1], bufs[2]));
  } catch (const c10::Error& e) {
    return ioError(std::string("LibTorch SDPA backward failed: ") + e.what_without_backtrace());
  } catch (const std::exception& e) {
    return ioError(std::string("LibTorch SDPA backward failed: ") + e.what());
  } catch (...) {
    return ioError("LibTorch SDPA backward failed with an unknown native exception");
  }
}

// Guard used by the exported libtorch-FFI functions.
// On Linux/macOS : Pass through function that calls `f` with `args`.
// On Windows:
//   The mixed-ABI exe (MinGW/Itanium Lean runtime + clang-cl/MSVC bridge + MSVC torch DLLs)
//   cannot unwind C++ exceptions across the FFI boundary: the MSVC Exception
//   Handler (EH)  personality aborts (__CxxFrameHandler3 -> __vcrt_getptd) on
//   Lean's thread.
//   Hence, call `f` with `args` inside an SEH frame so any Windows exception
//   becomes a  normal Lean error.
//   SEH Ref:
//    https://en.wikipedia.org/wiki/Microsoft-specific_exception_handling_mechanisms#Structured_Exception_Handling
template <typename F, typename... Args>
static lean_obj_res torchlean_seh_guard(const char* what, F&& f, Args&&... args) {
#if defined(_WIN32)
  lean_obj_res r;
  __try {
    r = std::forward<F>(f)(std::forward<Args>(args)...);
  } __except (1) {
    r = c_ioError(what);
  }
  return r;
#else
  (void)what;
  return std::forward<F>(f)(std::forward<Args>(args)...);
#endif
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_sdpa_fwd(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  return torchlean_seh_guard(
      "LibTorch SDPA forward failed with a Windows exception inside torch",
      torchlean_libtorch_sdpa_fwd_impl, Q, K, V, M, hasMask, batch, n, d, scale);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_libtorch_sdpa_bwd(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M, b_lean_obj_arg DOut,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  return torchlean_seh_guard(
      "LibTorch SDPA backward failed with a Windows exception inside torch",
      torchlean_libtorch_sdpa_bwd_impl, Q, K, V, M, DOut, hasMask, batch, n, d, scale);
}
