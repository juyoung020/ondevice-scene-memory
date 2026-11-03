// YOLO26 Depth 머리·손실(Ultralytics 8.4 nn/modules/head.py Depth, utils/loss.py DepthLoss26)의 GPU 부분.
//
// 머리: proj(1×1) → 굵은 단계부터 bilinear 2배(align_corners=True) + 다음 단계 → refine(3×3 둘) → head(3×3, ConvT 2×2,
//       3×3, 1×1) → z(로그 깊이, 입력/4). 학습 출력 = exp(clamp(z, −4, 5)), 추론 = exp(cal_a·clamp(z) + cal_b) [m].
// 손실(정답이 모두 유효할 때 — 선생 깊이): log 공간에서
//   SILog = sqrt(mean((d − mean d)²) + 1e-6)·dlog   (dlam = 1: 크기 무관)
//   grad  = Σ_{4 단계, 2×2 평균풀링} [mean|Δx p − Δx g| + mean|Δy p − Δy g|]·dgrad
// 합은 Ultralytics 처럼 × 배치(기울기 쪽 gain 에 넣는다).
#pragma once
#include "core.h"

// out = skip + up2_bilinear_ac(in)  (in: [N, H, W, C], skip·out: [N, 2H, 2W, C])
void bilin_up_add_fwd(const act_t* in, const act_t* skip, act_t* out, int N, int H, int W, int C, cudaStream_t st);
// din (accum ? += : =) up2ᵀ(dout), dskip (skip_accum ? += : =) dout. tmp: fp32 [N·H·W·C]
void bilin_up_add_bwd(const act_t* dout, act_t* din, bool accum, act_t* dskip, bool skip_accum, float* tmp, int N,
                      int H, int W, int C, cudaStream_t st);

// 선생 로그 깊이: t = a·clamp(z, −4, 5) + b
void depth_teacher_log(const act_t* z, float a, float b, float* tlog, int64_t n, cudaStream_t st);

class DepthLoss {
 public:
  void init(int B, int H, int W, float dlog = 1.f, float dgrad = 0.5f, int scales = 4);
  // z: 학생 머리 출력(로그 깊이, clamp 전), tlog: 선생 로그 깊이. dz 를 쓴다(clamp 밖은 0).
  // gscale: 장치 float(loss scale × 배치 × 이 손실의 가중) — 기울기에만.
  void run(const act_t* z, const float* tlog, act_t* dz, const float* gscale, cudaStream_t st);
  const float* items() const { return items_; }  // [0] SILog·dlog, [1] grad·dgrad (배치 곱 전)
  int B = 0, H = 0, W = 0, scales = 4;
  float dlog = 1.f, dgrad = 0.5f;

 private:
  int nl_ = 0;
  float* p_[8] = {};   // 단계별 예측 로그 깊이(평균풀링 피라미드)
  float* g_[8] = {};
  float* dp_[8] = {};  // 단계별 기울기
  double* acc_ = nullptr;  // [0] Σd [1] Σd² [2..] 단계별 L1 합
  float* items_ = nullptr; // [0] silog·dlog [1] grad·dgrad [2] silog [3] mean d
};
