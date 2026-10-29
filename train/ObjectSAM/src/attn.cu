#include "attn.cuh"

#include <vector>

namespace {

#if defined(FSO_FP32)
constexpr cudaDataType_t kActBlas = CUDA_R_32F;
#else
constexpr cudaDataType_t kActBlas = CUDA_R_16F;
#endif

constexpr int kWarps = 8;  // 블록당 워프(열 하나 = 워프 하나)

// 열마다(길이 N, 연속) softmax: pt = softmax(s)
__global__ void k_softmax_cols(const float* __restrict__ s, act_t* __restrict__ pt, int N, int64_t cols) {
  int64_t col = int64_t(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (col >= cols) return;
  const int lane = threadIdx.x & 31;
  const float* x = s + col * N;
  act_t* y = pt + col * N;
  float mx = -INFINITY;
  for (int i = lane; i < N; i += 32) mx = fmaxf(mx, x[i]);
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sum = 0.f;
  for (int i = lane; i < N; i += 32) sum += __expf(x[i] - mx);
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  const float inv = 1.f / sum;
  for (int i = lane; i < N; i += 32) y[i] = from_f<act_t>(__expf(x[i] - mx) * inv);
}

// 열마다 softmax 역: ds = p ⊙ (dp − Σ p·dp)
__global__ void k_softmax_bwd_cols(const act_t* __restrict__ pt, const float* __restrict__ dpt, act_t* __restrict__ ds,
                                   int N, int64_t cols) {
  int64_t col = int64_t(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (col >= cols) return;
  const int lane = threadIdx.x & 31;
  const act_t* p = pt + col * N;
  const float* g = dpt + col * N;
  act_t* o = ds + col * N;
  float dot = 0.f;
  for (int i = lane; i < N; i += 32) dot += to_f(p[i]) * g[i];
  for (int k = 16; k; k >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, k);
  for (int i = lane; i < N; i += 32) { float pv = to_f(p[i]); o[i] = from_f<act_t>(pv * (g[i] - dot)); }
}

template <class T>
T** to_dev(const std::vector<T*>& v) {
  T** d = static_cast<T**>(DevMem::alloc(v.size() * sizeof(T*)));
  CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T*), cudaMemcpyHostToDevice));
  return d;
}

void gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, float alpha,
          const void* const* A, cudaDataType_t at, int lda, const void* const* B, cudaDataType_t bt, int ldb,
          float beta, void* const* C, cudaDataType_t ct, int ldc, int batch) {
  CUBLAS_CHECK(cublasGemmBatchedEx(h, ta, tb, m, n, k, &alpha, A, at, lda, B, bt, ldb, &beta, C, ct, ldc, batch,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

}  // namespace

void AttnOp::init(cublasHandle_t h, int B_, int N_, int heads_, int kd_, int hd_, const act_t* qkv, act_t* out,
                  act_t* dqkv, const act_t* dout) {
  h_ = h; B = B_; N = N_; heads = heads_; kd = kd_; hd = hd_;
  const int E = 2 * kd + hd;
  Cq = heads * E;
  Co = heads * hd;
  scale = 1.f / sqrtf(float(kd));
  const int64_t NN = int64_t(N) * N, nb = int64_t(B) * heads;
  pt_ = dalloc<act_t>(nb * NN);
  f32_ = dalloc<float>(nb * NN);
  ds_ = dalloc<act_t>(nb * NN);
  std::vector<const void*> q, k, v, dO;
  std::vector<void*> dq, dk, dv, o, ptp, f32p, dsp;
  for (int b = 0; b < B; b++)
    for (int hh = 0; hh < heads; hh++) {
      const int64_t base = int64_t(b) * N * Cq + int64_t(hh) * E;
      q.push_back(qkv + base);
      k.push_back(qkv + base + kd);
      v.push_back(qkv + base + 2 * kd);
      dq.push_back(dqkv ? dqkv + base : nullptr);
      dk.push_back(dqkv ? dqkv + base + kd : nullptr);
      dv.push_back(dqkv ? dqkv + base + 2 * kd : nullptr);
      const int64_t ob = int64_t(b) * N * Co + int64_t(hh) * hd;
      o.push_back(out + ob);
      dO.push_back(dout ? dout + ob : nullptr);
      const int64_t pb = (int64_t(b) * heads + hh) * NN;
      ptp.push_back(pt_ + pb);
      f32p.push_back(f32_ + pb);
      dsp.push_back(ds_ + pb);
    }
  q_ = to_dev(q); k_ = to_dev(k); v_ = to_dev(v); do_ = to_dev(dO);
  dq_ = to_dev(dq); dk_ = to_dev(dk); dv_ = to_dev(dv); o_ = to_dev(o);
  ptp_ = to_dev(ptp); f32p_ = to_dev(f32p); dsp_ = to_dev(dsp);
}

void AttnOp::fwd(cudaStream_t st) {
  CUBLAS_CHECK(cublasSetStream(h_, st));
  const int nb = B * heads;
  const int64_t cols = int64_t(nb) * N;
  // Sᵀ = scale · Kᵀ Q   (N×N)
  gemm(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, N, kd, scale, k_, kActBlas, Cq, q_, kActBlas, Cq, 0.f, f32p_, CUDA_R_32F, N, nb);
  k_softmax_cols<<<cdiv(cols, kWarps), 32 * kWarps, 0, st>>>(f32_, pt_, N, cols);
  // O = V · Pᵀ   (hd×N) → out 의 머리 h 채널
  gemm(h_, CUBLAS_OP_N, CUBLAS_OP_N, hd, N, N, 1.f, v_, kActBlas, Cq, (const void* const*)ptp_, kActBlas, N, 0.f, o_,
       kActBlas, Co, nb);
}

void AttnOp::bwd(cudaStream_t st) {
  CUBLAS_CHECK(cublasSetStream(h_, st));
  const int nb = B * heads;
  const int64_t cols = int64_t(nb) * N;
  // dV = dO · P      (P = Ptᵀ)
  gemm(h_, CUBLAS_OP_N, CUBLAS_OP_T, hd, N, N, 1.f, do_, kActBlas, Co, (const void* const*)ptp_, kActBlas, N, 0.f, dv_,
       kActBlas, Cq, nb);
  // dPᵀ = Vᵀ · dO
  gemm(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, N, hd, 1.f, v_, kActBlas, Cq, do_, kActBlas, Co, 0.f, f32p_, CUDA_R_32F, N, nb);
  k_softmax_bwd_cols<<<cdiv(cols, kWarps), 32 * kWarps, 0, st>>>(pt_, f32_, ds_, N, cols);
  // dQ = scale · K · dSᵀ,  dK = scale · Q · dS
  gemm(h_, CUBLAS_OP_N, CUBLAS_OP_N, kd, N, N, scale, k_, kActBlas, Cq, (const void* const*)dsp_, kActBlas, N, 0.f, dq_,
       kActBlas, Cq, nb);
  gemm(h_, CUBLAS_OP_N, CUBLAS_OP_T, kd, N, N, scale, q_, kActBlas, Cq, (const void* const*)dsp_, kActBlas, N, 0.f, dk_,
       kActBlas, Cq, nb);
}
