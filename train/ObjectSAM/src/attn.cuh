// YOLO26 PSABlock 의 Attention 핵심(Ultralytics ultralytics/nn/modules/block.py Attention.forward 의 행렬 부분).
//
//   qkv: [B, N=H*W, Cq] NHWC, 머리 h 의 채널 [h*E, h*E+kd) = q, [+kd, +2kd) = k, [+2kd, +2kd+hd) = v  (E = 2kd+hd)
//   S[n,m] = scale · Σ_i q[i,n] k[i,m],  P = softmax_m(S),  out[n, h*hd + d] = Σ_m v[d,m] P[n,m]
// (pe(v)·proj·잔차는 바깥의 Conv+BN 노드가 한다.)
//
// 열 우선 cuBLAS 로 본다: 영상 b 의 qkv 는 (Cq × N) 행렬(ld Cq). 저장은 Pt = Pᵀ (N×N, 열 n 이 화소 n 의 확률이라
// softmax 가 연속 메모리에서 돈다). 영상·머리마다 포인터 배열로 묶음 GEMM(fp16 입력, fp32 누적).
#pragma once
#include <cublas_v2.h>

#include "core.h"

#define CUBLAS_CHECK(x)                                                                                  \
  do {                                                                                                   \
    cublasStatus_t e_ = (x);                                                                             \
    if (e_ != CUBLAS_STATUS_SUCCESS) {                                                                   \
      fprintf(stderr, "cuBLAS %s:%d %s: %d\n", __FILE__, __LINE__, #x, int(e_));                         \
      std::abort();                                                                                      \
    }                                                                                                    \
  } while (0)

class AttnOp {
 public:
  // heads·kd(key)·hd(head) 는 Ultralytics: heads = max(c/64, 1), hd = c/heads, kd = hd·attn_ratio(0.5)
  void init(cublasHandle_t h, int B, int N, int heads, int kd, int hd, const act_t* qkv, act_t* out, act_t* dqkv,
            const act_t* dout);
  void fwd(cudaStream_t st);
  // dqkv 를 (q·k·v 채널 모두) 덮어쓴다. dout 은 출력 기울기.
  void bwd(cudaStream_t st);
  int B = 0, N = 0, heads = 0, kd = 0, hd = 0, Cq = 0, Co = 0;
  float scale = 1.f;

 private:
  cublasHandle_t h_ = nullptr;
  act_t* pt_ = nullptr;     // [B*heads][N*N] softmax 확률 Pᵀ (역방향에 다시 쓴다)
  float* f32_ = nullptr;    // [B*heads][N*N] 정방향 점수 Sᵀ / 역방향 dPᵀ (fp32)
  act_t* ds_ = nullptr;     // [B*heads][N*N] dSᵀ
  // 묶음 GEMM 포인터 배열(장치): 영상·머리 순서
  const void** q_ = nullptr; const void** k_ = nullptr; const void** v_ = nullptr;
  void** dq_ = nullptr; void** dk_ = nullptr; void** dv_ = nullptr;
  void** o_ = nullptr; const void** do_ = nullptr;
  void** ptp_ = nullptr; void** f32p_ = nullptr; void** dsp_ = nullptr;
};
