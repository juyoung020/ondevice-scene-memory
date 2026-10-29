#include "loss26.cuh"

#include <algorithm>
#include <climits>

namespace {

constexpr int kT = 256;
constexpr float kCiouEps = 1e-7f;
constexpr float kTalEps = 1e-9f;
constexpr int kKC = 32;  // 후보(3·topk) 상한

struct Lv {
  int off[4];
  int h[3], w[3], s[3];
};
__device__ __forceinline__ void anchor_of(const Lv& L, int a, int& l, int& h, int& w) {
  l = a < L.off[1] ? 0 : (a < L.off[2] ? 1 : 2);
  int r = a - L.off[l];
  h = r / L.w[l];
  w = r - h * L.w[l];
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float bce_logits(float x, float y) { return fmaxf(x, 0.f) - x * y + log1pf(expf(-fabsf(x))); }
__device__ __forceinline__ void ld8(const float* p, float o[8]) {
  float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
__device__ __forceinline__ void st8(float* p, const float o[8]) {
  *reinterpret_cast<float4*>(p) = make_float4(o[0], o[1], o[2], o[3]);
  *reinterpret_cast<float4*>(p + 4) = make_float4(o[4], o[5], o[6], o[7]);
}
__device__ __forceinline__ void ld8(const __half* p, float o[8]) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const __half2* h = reinterpret_cast<const __half2*>(&u);
  for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}
__device__ __forceinline__ void st8(__half* p, const float o[8]) {
  uint4 u;
  __half2* h = reinterpret_cast<__half2*>(&u);
  for (int i = 0; i < 4; i++) h[i] = __floats2half2_rn(o[2 * i], o[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}
__device__ __forceinline__ void ld32(const act_t* p, float o[32]) {
  for (int k = 0; k < 32; k += 8) ld8(p + k, o + k);
}
__device__ __forceinline__ void st32(act_t* p, const float o[32]) {
  for (int k = 0; k < 32; k += 8) st8(p + k, o + k);
}
__device__ __forceinline__ void zero_n(act_t* p, int n) {
  const float z[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < n; k += 8) st8(p + k, z);
}

// Ultralytics bbox_iou(xywh=False, CIoU=True), alpha 상수 취급. loss.cu 와 같은 식.
__device__ float ciou(float4 b1, float4 b2) {
  float w1 = b1.z - b1.x, h1 = b1.w - b1.y + kCiouEps, w2 = b2.z - b2.x, h2 = b2.w - b2.y + kCiouEps;
  float iw = fmaxf(fminf(b1.z, b2.z) - fmaxf(b1.x, b2.x), 0.f), ih = fmaxf(fminf(b1.w, b2.w) - fmaxf(b1.y, b2.y), 0.f);
  float inter = iw * ih, uni = w1 * h1 + w2 * h2 - inter + kCiouEps, iou = inter / uni;
  float cw = fmaxf(b1.z, b2.z) - fminf(b1.x, b2.x), ch = fmaxf(b1.w, b2.w) - fminf(b1.y, b2.y);
  float c2 = cw * cw + ch * ch + kCiouEps;
  float sx = b2.x + b2.z - b1.x - b1.z, sy = b2.y + b2.w - b1.y - b1.w;
  float rho2 = (sx * sx + sy * sy) * 0.25f;
  float dv = atanf(w2 / h2) - atanf(w1 / h1);
  float v = 0.40528473456935108f * dv * dv;
  float alpha = v / (1.f - iou + v + kCiouEps);
  return iou - (rho2 / c2 + v * alpha);
}
__device__ float ciou_grad(float4 b1, float4 b2, float4& g) {
  float w1 = b1.z - b1.x, h1 = b1.w - b1.y + kCiouEps, w2 = b2.z - b2.x, h2 = b2.w - b2.y + kCiouEps;
  float ix2 = fminf(b1.z, b2.z), ix1 = fmaxf(b1.x, b2.x), iy2 = fminf(b1.w, b2.w), iy1 = fmaxf(b1.y, b2.y);
  float iw_ = ix2 - ix1, ih_ = iy2 - iy1;
  float iw = fmaxf(iw_, 0.f), ih = fmaxf(ih_, 0.f);
  float inter = iw * ih, uni = w1 * h1 + w2 * h2 - inter + kCiouEps, iou = inter / uni;
  float cx2 = fmaxf(b1.z, b2.z), cx1 = fminf(b1.x, b2.x), cy2 = fmaxf(b1.w, b2.w), cy1 = fminf(b1.y, b2.y);
  float cw = cx2 - cx1, ch = cy2 - cy1;
  float c2 = cw * cw + ch * ch + kCiouEps;
  float sx = b2.x + b2.z - b1.x - b1.z, sy = b2.y + b2.w - b1.y - b1.w;
  float rho2 = (sx * sx + sy * sy) * 0.25f;
  float r1 = w1 / h1;
  float dv = atanf(w2 / h2) - atanf(r1);
  const float k4 = 0.40528473456935108f;
  float v = k4 * dv * dv;
  float alpha = v / (1.f - iou + v + kCiouEps);
  float out = iou - (rho2 / c2 + v * alpha);
  float dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0, dw1 = 0, dh1 = 0;
  float dinter = (uni + inter) / (uni * uni);
  float dA1 = -inter / (uni * uni);
  dw1 += dA1 * h1;
  dh1 += dA1 * w1;
  float diw = iw_ > 0.f ? dinter * ih : 0.f, dih = ih_ > 0.f ? dinter * iw : 0.f;
  if (b1.z < b2.z) dx2 += diw;
  if (b1.x > b2.x) dx1 -= diw;
  if (b1.w < b2.w) dy2 += dih;
  if (b1.y > b2.y) dy1 -= dih;
  float drho2 = -1.f / c2, dc2 = rho2 / (c2 * c2);
  dx1 += drho2 * (-sx * 0.5f);
  dx2 += drho2 * (-sx * 0.5f);
  dy1 += drho2 * (-sy * 0.5f);
  dy2 += drho2 * (-sy * 0.5f);
  float dcw = dc2 * 2.f * cw, dch = dc2 * 2.f * ch;
  if (b1.z > b2.z) dx2 += dcw;
  if (b1.x < b2.x) dx1 -= dcw;
  if (b1.w > b2.w) dy2 += dch;
  if (b1.y < b2.y) dy1 -= dch;
  float dr1 = -alpha * k4 * 2.f * dv * (-1.f / (1.f + r1 * r1));
  dw1 += dr1 / h1;
  dh1 += dr1 * (-w1 / (h1 * h1));
  dx2 += dw1;
  dx1 -= dw1;
  dy2 += dh1;
  dy1 -= dh1;
  g = make_float4(dx1, dy1, dx2, dy2);
  return out;
}
__device__ __forceinline__ float4 enlarge(float4 g) {  // select_candidates_in_gts: 변이 16(stride[1]) 미만이면 키움
  float cx = (g.x + g.z) * 0.5f, cy = (g.y + g.w) * 0.5f, w = g.z - g.x, h = g.w - g.y;
  if (w < 16.f) w = 16.f;
  if (h < 16.f) h = 16.f;
  return make_float4(cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f);
}
__device__ __forceinline__ bool inside(float4 e, float px, float py) {
  return px - e.x > kTalEps && py - e.y > kTalEps && e.z - px > kTalEps && e.w - py > kTalEps;
}
__device__ __forceinline__ float metric_of(float score, float ov) {  // score^0.5 · ov^6
  float o2 = ov * ov;
  return sqrtf(score) * o2 * o2 * o2;
}
__device__ __forceinline__ void block_add(float v, float* dst) {
  __shared__ float red[kT / 32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0 && v != 0.f) atomicAdd(dst, v);
  }
  __syncthreads();
}
__device__ __forceinline__ int block_sum_int(int v, int* sh /*[kT/32]*/) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
  __syncthreads();
  int t = 0;
  for (int i = 0; i < int(blockDim.x / 32); i++) t += sh[i];
  __syncthreads();
  return t;
}
__device__ __forceinline__ bool gt_bit(const uint32_t* m, int pi) { return (m[pi >> 5] >> (pi & 31)) & 1u; }

// ---------------------------------------------------------------- 할당

// 1) 해독: 상자(격자 단위 xyxy, reg_max 1: 거리 그대로)와 점수
struct Dec1 { const act_t* box[3]; const act_t* cls[3]; };
__global__ void k_decode1(Lv L, int A, int B, Dec1 io, float4* pbox, float* score) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= B * A) return;
  int b = q / A, a = q - b * A, l, h, w;
  anchor_of(L, a, l, h, w);
  int64_t pix = (int64_t(b) * L.h[l] + h) * L.w[l] + w;
  const act_t* p = io.box[l] + pix * 4;
  float ax = w + 0.5f, ay = h + 0.5f;
  pbox[q] = make_float4(ax - to_f(p[0]), ay - to_f(p[1]), ax + to_f(p[2]), ay + to_f(p[3]));
  score[q] = sigm(to_f(io.cls[l][pix]));
}

// 정답 마스크 화소 수(proto 해상도)
__global__ void k_garea(int G, int words, const int* n_gt, const uint32_t* mask, int* area) {
  int b = blockIdx.x / G, j = blockIdx.x - b * G;
  __shared__ int sh[kT / 32];
  int c = 0;
  if (j < n_gt[b])
    for (int w = threadIdx.x; w < words; w += blockDim.x) c += __popc(mask[(int64_t(b) * G + j) * words + w]);
  c = block_sum_int(c, sh);
  if (threadIdx.x == 0) area[blockIdx.x] = c;
}

// 2) 정답마다(블록 하나): 상자 안 앵커의 상자 지표 상위 ncand → 후보마다 예측 마스크 IoU → 합친 겹침으로 지표 →
//    상위 topk 를 양성 후보로(cnt, who). 후보 목록(앵커·합친 겹침)은 충돌 해소용으로 남긴다.
struct CandArgs {
  Lv L;
  int A, G, ncand, topk, Hp, Wp, imgsz;
  const int* n_gt;
  const float4* gtb;
  const float4* pbox;
  const float* score;
  const act_t* coef[3];
  const act_t* proto;
  const uint32_t* gmask;
  const int* garea;
  int* cnt;
  int* who;
  int* cand_a;
  float* cand_ov;
};
__global__ void __launch_bounds__(kT) k_cand(CandArgs P) {
  const int b = blockIdx.x / P.G, j = blockIdx.x - b * P.G;
  int* ca = P.cand_a + int64_t(blockIdx.x) * P.ncand;
  if (threadIdx.x < P.ncand) ca[threadIdx.x] = -1;
  if (j >= P.n_gt[b]) return;
  const float4 g = P.gtb[blockIdx.x];
  const float4 e = enlarge(g);
  float tv[kKC], tio[kKC];
  int ti[kKC];
  for (int k = 0; k < kKC; k++) { tv[k] = -1.f; ti[k] = -1; tio[k] = 0.f; }
  const int K = P.ncand;
  for (int l = 0; l < 3; l++) {
    const float s = P.L.s[l];
    int w0 = max(0, int(floorf(e.x / s - 0.5f))), w1 = min(P.L.w[l] - 1, int(ceilf(e.z / s - 0.5f)));
    int h0 = max(0, int(floorf(e.y / s - 0.5f))), h1 = min(P.L.h[l] - 1, int(ceilf(e.w / s - 0.5f)));
    int nw = w1 - w0 + 1, nh = h1 - h0 + 1;
    if (nw <= 0 || nh <= 0) continue;
    for (int t = threadIdx.x; t < nw * nh; t += blockDim.x) {
      int hh = h0 + t / nw, ww = w0 + t % nw;
      if (!inside(e, (ww + 0.5f) * s, (hh + 0.5f) * s)) continue;
      int a = P.L.off[l] + hh * P.L.w[l] + ww;
      int q = b * P.A + a;
      float4 pb = P.pbox[q];
      float ov = fmaxf(ciou(g, make_float4(pb.x * s, pb.y * s, pb.z * s, pb.w * s)), 0.f);
      float m = metric_of(P.score[q], ov) + kTalEps;  // SegTaskAlignedAssigner: eps 로 지표 0 쌍도 고를 수 있게
      if (m > tv[K - 1] || (m == tv[K - 1] && (ti[K - 1] < 0 || a < ti[K - 1]))) {
        int k = K - 1;
        while (k > 0 && (m > tv[k - 1] || (m == tv[k - 1] && a < ti[k - 1]))) {
          tv[k] = tv[k - 1]; ti[k] = ti[k - 1]; tio[k] = tio[k - 1]; k--;
        }
        tv[k] = m; ti[k] = a; tio[k] = ov;
      }
    }
  }
  // 블록 병합: K 번 최댓값
  __shared__ float sv[kT], siou_[kT];
  __shared__ int si[kT], st_[kT];
  __shared__ int ca_s[kKC];
  __shared__ float cio_s[kKC], cov_s[kKC], cfm_s[kKC];
  __shared__ int nfound;
  int head = 0;
  if (threadIdx.x == 0) nfound = 0;
  for (int r = 0; r < K; r++) {
    sv[threadIdx.x] = head < K && ti[head] >= 0 ? tv[head] : -1.f;
    si[threadIdx.x] = head < K ? ti[head] : -1;
    siou_[threadIdx.x] = head < K ? tio[head] : 0.f;
    st_[threadIdx.x] = threadIdx.x;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
      if (threadIdx.x < o) {
        float v2 = sv[threadIdx.x + o];
        int i2 = si[threadIdx.x + o];
        bool take = i2 >= 0 && (si[threadIdx.x] < 0 || v2 > sv[threadIdx.x] || (v2 == sv[threadIdx.x] && i2 < si[threadIdx.x]));
        if (take) { sv[threadIdx.x] = v2; si[threadIdx.x] = i2; siou_[threadIdx.x] = siou_[threadIdx.x + o]; st_[threadIdx.x] = st_[threadIdx.x + o]; }
      }
      __syncthreads();
    }
    int win = si[0], wt = st_[0];
    float wio = siou_[0];
    __syncthreads();
    if (win < 0) break;
    if (threadIdx.x == wt) head++;
    if (threadIdx.x == 0) { ca_s[r] = win; cio_s[r] = wio; nfound = r + 1; }
  }
  __syncthreads();
  const int nf = nfound;
  // 후보마다 예측 마스크 IoU: (coef·proto > 0) 를 예측 상자(proto 좌표)로 자른 것 vs 정답 마스크
  __shared__ float cf[32];
  __shared__ int red[kT / 32];
  const int words = P.Hp * P.Wp / 32;
  const uint32_t* gm = P.gmask + int64_t(blockIdx.x) * words;
  const float msc = float(P.Hp) / float(P.imgsz);
  for (int i = 0; i < nf; i++) {
    const int a = ca_s[i];
    int l, h, w;
    anchor_of(P.L, a, l, h, w);
    const int64_t apix = (int64_t(b) * P.L.h[l] + h) * P.L.w[l] + w;
    if (threadIdx.x < 32) cf[threadIdx.x] = to_f(P.coef[l][apix * 32 + threadIdx.x]);
    __syncthreads();
    const float s = P.L.s[l];
    const float4 pb = P.pbox[b * P.A + a];
    const float x1 = pb.x * s * msc, y1 = pb.y * s * msc, x2 = pb.z * s * msc, y2 = pb.w * s * msc;
    const int c0 = max(0, int(ceilf(x1))), c1 = min(P.Wp, int(ceilf(x2)));
    const int r0 = max(0, int(ceilf(y1))), r1 = min(P.Hp, int(ceilf(y2)));
    const int nw = c1 - c0, n = nw > 0 && r1 > r0 ? nw * (r1 - r0) : 0;
    int inter = 0, pmc = 0;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
      const int py = r0 + t / nw, px = c0 + t % nw;
      float pr[32], z = 0.f;
      ld32(P.proto + ((int64_t(b) * P.Hp + py) * P.Wp + px) * 32, pr);
      for (int k = 0; k < 32; k++) z += cf[k] * pr[k];
      if (z > 0.f) {
        pmc++;
        inter += gt_bit(gm, py * P.Wp + px);
      }
    }
    inter = block_sum_int(inter, red);
    pmc = block_sum_int(pmc, red);
    if (threadIdx.x == 0) {
      const int uni = max(P.garea[blockIdx.x] + pmc - inter, 1);
      const float miou = float(inter) / float(uni);
      const float ov = (cio_s[i] + miou) * 0.5f;
      cov_s[i] = ov;
      cfm_s[i] = metric_of(P.score[b * P.A + a], ov);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    // 합친 지표 상위 topk (같으면 앵커 번호 작은 쪽)
    bool used[kKC];
    for (int i = 0; i < nf; i++) used[i] = false;
    for (int r = 0; r < P.topk && r < nf; r++) {
      int bi = -1;
      for (int i = 0; i < nf; i++) {
        if (used[i]) continue;
        if (bi < 0 || cfm_s[i] > cfm_s[bi] || (cfm_s[i] == cfm_s[bi] && ca_s[i] < ca_s[bi])) bi = i;
      }
      used[bi] = true;
      atomicAdd(&P.cnt[b * P.A + ca_s[bi]], 1);
      P.who[b * P.A + ca_s[bi]] = j;
    }
  }
  if (threadIdx.x < nf) {
    ca[threadIdx.x] = ca_s[threadIdx.x];
    P.cand_ov[int64_t(blockIdx.x) * P.ncand + threadIdx.x] = cov_s[threadIdx.x];
  }
}

// 앵커 a 와 정답 k 의 합친 겹침: 후보면 (CIoU + 마스크 IoU)/2, 상자 안 다른 앵커는 CIoU/2, 밖은 0
__device__ float pair_ov(const int* cand_a, const float* cand_ov, int ncand, int bg, int a, float4 g, float px, float py,
                         float4 pp) {
  const int* ca = cand_a + int64_t(bg) * ncand;
  for (int i = 0; i < ncand; i++) {
    int c = ca[i];
    if (c < 0) break;
    if (c == a) return cand_ov[int64_t(bg) * ncand + i];
  }
  return inside(enlarge(g), px, py) ? fmaxf(ciou(g, pp), 0.f) * 0.5f : 0.f;
}

// 3) 앵커별 충돌 해소(여러 정답이 뽑으면 합친 겹침이 가장 큰 정답) → 정답·지표·겹침
__global__ void k_resolve26(Lv L, int A, int B, int G, int ncand, const int* n_gt, const float4* gtb, const float4* pbox,
                            const float* score, int* cnt, const int* who, const int* cand_a, const float* cand_ov,
                            int* agt, float* amet, float* aovl) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= B * A) return;
  int c = cnt[q];
  if (c == 0) { agt[q] = -1; return; }
  cnt[q] = 0;
  int b = q / A, a = q - b * A, l, h, w;
  anchor_of(L, a, l, h, w);
  float s = L.s[l], px = (w + 0.5f) * s, py = (h + 0.5f) * s;
  float4 pb = pbox[q];
  float4 pp = make_float4(pb.x * s, pb.y * s, pb.z * s, pb.w * s);
  int j = who[q];
  float ov;
  if (c > 1) {
    float best = -1.f;
    for (int k = 0; k < n_gt[b]; k++) {
      float o = pair_ov(cand_a, cand_ov, ncand, b * G + k, a, gtb[b * G + k], px, py, pp);
      if (o > best) { best = o; j = k; }
    }
    ov = best;
  } else {
    ov = pair_ov(cand_a, cand_ov, ncand, b * G + j, a, gtb[b * G + j], px, py, pp);
  }
  agt[q] = j;
  amet[q] = metric_of(score[q], ov);
  aovl[q] = ov;
}

// 3b) topk2 = 1: 정답마다 지표가 가장 큰 양성 하나만(같으면 앵커 번호 작은 쪽)
__global__ void k_best(int A, int B, int G, const int* agt, const float* amet, int* gbest) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= B * A || agt[q] < 0) return;
  atomicMax(&gbest[(q / A) * G + agt[q]], __float_as_int(amet[q]));
}
__global__ void k_win(int A, int B, int G, const int* agt, const float* amet, const int* gbest, int* gwin) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= B * A || agt[q] < 0) return;
  int bg = (q / A) * G + agt[q];
  if (__float_as_int(amet[q]) == gbest[bg]) atomicMin(&gwin[bg], q % A);
}
__global__ void k_keep(int A, int B, int G, int* agt, const int* gwin) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= B * A || agt[q] < 0) return;
  if (gwin[(q / A) * G + agt[q]] != q % A) agt[q] = -1;
}

// 4) 양성 목록·정답별 최대(정규화용)
__global__ void k_pos26(int A, int B, int G, const int* agt, const float* amet, const float* aovl, float* galign,
                        float* govl, int* pos, int* npos, float* items) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= B * A || agt[q] < 0) return;
  int b = q / A, j = agt[q];
  atomicMax(reinterpret_cast<int*>(&galign[b * G + j]), __float_as_int(amet[q]));
  atomicMax(reinterpret_cast<int*>(&govl[b * G + j]), __float_as_int(aovl[q]));
  int k = atomicAdd(&npos[b], 1);
  pos[b * A + k] = q - b * A;
  atomicAdd(&items[4], 1.f);
}
__global__ void k_tsum(int A, int B, int G, const int* npos, const int* pos, const int* agt, const float* amet,
                       const float* galign, const float* govl, float* tsc, float* items) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  float t = 0.f;
  if (q < B * A) {
    int b = q / A, i = q - b * A;
    if (i < npos[b]) {
      int a = pos[b * A + i], qa = b * A + a, j = agt[qa];
      t = amet[qa] * govl[b * G + j] / (galign[b * G + j] + kTalEps);
      tsc[qa] = t;
    }
  }
  for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
  if ((threadIdx.x & 31) == 0 && t != 0.f) atomicAdd(&items[5], t);
}

// ---------------------------------------------------------------- 손실

// 5) 점수 BCE(전 앵커), 상자 CIoU + 정규화 거리 L1(양성): 손실과 머리 기울기
struct Det1 {
  Lv L;
  int A, B, G, imgsz, Hp, Wp;
  const act_t* box[3];
  const act_t* cls[3];
  act_t* dbox[3];
  act_t* dcls[3];
  act_t* dcoef[3];
  const int* agt;
  const float* tsc;
  const float4* gtb;
  const uint32_t* ignore;
  float* items;
  const float* scale;
  float box_gain, cls_gain, l1_gain;
};
__global__ void k_det1(Det1 P) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  float lcls = 0, liou = 0, ll1 = 0;
  if (q < P.B * P.A) {
    int b = q / P.A, a = q - b * P.A, l, h, w;
    anchor_of(P.L, a, l, h, w);
    int64_t pix = (int64_t(b) * P.L.h[l] + h) * P.L.w[l] + w;
    float tss = fmaxf(P.items[5], 1.f);
    float sc = float(P.B) / tss * *P.scale;
    int j = P.agt[q];
    float t = j >= 0 ? P.tsc[q] : 0.f;
    float x = to_f(P.cls[l][pix]);
    bool ign = false;
    if (j < 0 && P.ignore) {
      int cs = P.imgsz / P.Wp;
      int cx = min(P.Wp - 1, int((w + 0.5f) * P.L.s[l]) / cs), cy = min(P.Hp - 1, int((h + 0.5f) * P.L.s[l]) / cs);
      int64_t bit = int64_t(cy) * P.Wp + cx;
      ign = P.ignore[int64_t(b) * (P.Hp * P.Wp / 32) + (bit >> 5)] >> (bit & 31) & 1u;
    }
    lcls = ign ? 0.f : bce_logits(x, t);
    P.dcls[l][pix] = from_f<act_t>(ign ? 0.f : (sigm(x) - t) * P.cls_gain * sc);
    act_t* db = P.dbox[l] + pix * 4;
    if (j < 0) {
      for (int s = 0; s < 4; s++) db[s] = from_f<act_t>(0.f);
      zero_n(P.dcoef[l] + pix * 32, 32);
    } else {
      const act_t* bl = P.box[l] + pix * 4;
      float d[4] = {to_f(bl[0]), to_f(bl[1]), to_f(bl[2]), to_f(bl[3])};
      float s_ = float(P.L.s[l]);
      float ax = w + 0.5f, ay = h + 0.5f;
      float4 pb = make_float4(ax - d[0], ay - d[1], ax + d[2], ay + d[3]);
      float4 tg = P.gtb[b * P.G + j];
      float4 tb = make_float4(tg.x / s_, tg.y / s_, tg.z / s_, tg.w / s_);
      float4 gb;
      float ci = ciou_grad(pb, tb, gb);
      liou = (1.f - ci) * t;
      float gi = -t * P.box_gain * sc;
      float dd[4] = {-gb.x * gi, -gb.y * gi, gb.z * gi, gb.w * gi};
      // L1: 거리(ltrb)·stride/imgsz 끼리, 4 개 평균 × 목표 점수
      const float tl[4] = {ax - tb.x, ay - tb.y, tb.z - ax, tb.w - ay};
      const float k = s_ / float(P.imgsz);
      float l1 = 0.f;
      for (int s = 0; s < 4; s++) {
        float r = (d[s] - tl[s]) * k;
        l1 += fabsf(r);
        float sg = r > 0.f ? 1.f : (r < 0.f ? -1.f : 0.f);
        dd[s] += t * P.l1_gain * sc * sg * k * 0.25f;
      }
      ll1 = l1 * 0.25f * t;
      for (int s = 0; s < 4; s++) db[s] = from_f<act_t>(dd[s]);
    }
  }
  block_add(lcls, &P.items[10]);
  block_add(liou, &P.items[8]);
  block_add(ll1, &P.items[11]);
}

// 6) 마스크 손실(loss.cu 와 같은 계산). dproto == nullptr 이면 proto 기울기를 쓰지 않는다(one2one, detach).
struct MaskArgs {
  Lv L;
  int A, B, G, imgsz, Hp, Wp;
  const act_t* coef[3];
  const act_t* proto;
  act_t* dproto;
  act_t* dcoef[3];
  const int* npos;
  const int* pos;
  const int* agt;
  const float4* gtb;
  const uint32_t* gmask;
  float* items;
  const float* scale;
  float box_gain;
};
__device__ __forceinline__ void crop_of(const MaskArgs& P, float4 tg, float& x1, float& y1, float& x2, float& y2,
                                        float& area) {
  float s = float(P.imgsz);
  float nx1 = tg.x / s, ny1 = tg.y / s, nx2 = tg.z / s, ny2 = tg.w / s;
  area = (nx2 - nx1) * (ny2 - ny1);
  x1 = nx1 * P.Wp; x2 = nx2 * P.Wp; y1 = ny1 * P.Hp; y2 = ny2 * P.Hp;
}
constexpr int kMC = 64;
__global__ void __launch_bounds__(kT) k_mask_tile(MaskArgs P) {
  const int b = blockIdx.y;
  const int tx = blockIdx.x % (P.Wp / 16), ty = blockIdx.x / (P.Wp / 16);
  const int px = tx * 16 + (threadIdx.x & 15), py = ty * 16 + (threadIdx.x >> 4);
  const int64_t pix = (int64_t(b) * P.Hp + py) * P.Wp + px;
  float pr[32], acc[32];
  ld32(P.proto + pix * 32, pr);
  for (int k = 0; k < 32; k++) acc[k] = 0.f;
  const float fg = fmaxf(P.items[4], 1.f);
  const float hw = float(P.Hp) * P.Wp;
  const float gk = P.box_gain * float(P.B) / fg * *P.scale;
  float lsum = 0.f;
  const int np = P.npos[b];
  const int words = P.Hp * P.Wp / 32;
  const int pi = py * P.Wp + px;
  __shared__ __align__(16) float cf[kMC][32];
  __shared__ float4 cbox[kMC];
  __shared__ float cinv[kMC];
  __shared__ int cj[kMC], canc[kMC], wcnt[kMC / 32];
  for (int base = 0; base < np; base += kMC) {
    bool hit = false;
    int a = 0, j = 0;
    float4 bx;
    float inv = 0.f;
    if (threadIdx.x < kMC && base + threadIdx.x < np) {
      a = P.pos[b * P.A + base + threadIdx.x];
      j = P.agt[b * P.A + a];
      float x1, y1, x2, y2, area;
      crop_of(P, P.gtb[b * P.G + j], x1, y1, x2, y2, area);
      bx = make_float4(x1, y1, x2, y2);
      inv = 1.f / (hw * area);
      hit = tx * 16 + 15 >= x1 && tx * 16 < x2 && ty * 16 + 15 >= y1 && ty * 16 < y2;
    }
    unsigned m = __ballot_sync(~0u, hit);
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    if (threadIdx.x < kMC && lane == 0) wcnt[wp] = __popc(m);
    __syncthreads();
    int n = 0;
    for (int w = 0; w < kMC / 32; w++) n += wcnt[w];
    if (hit) {
      int slot = __popc(m & ((1u << lane) - 1u));
      for (int w = 0; w < wp; w++) slot += wcnt[w];
      cbox[slot] = bx; cinv[slot] = inv; cj[slot] = j; canc[slot] = a;
    }
    __syncthreads();
    for (int t = threadIdx.x; t < n * 32; t += blockDim.x) {
      int i = t >> 5, k = t & 31, l, h, w;
      anchor_of(P.L, canc[i], l, h, w);
      cf[i][k] = to_f(P.coef[l][((int64_t(b) * P.L.h[l] + h) * P.L.w[l] + w) * 32 + k]);
    }
    __syncthreads();
    for (int i = 0; i < n; i++) {
      float4 c = cbox[i];
      if (!(px >= c.x && px < c.z && py >= c.y && py < c.w)) continue;
      const float4* cv = reinterpret_cast<const float4*>(cf[i]);
      float z = 0.f;
      for (int k = 0; k < 8; k++) {
        float4 v = cv[k];
        z += v.x * pr[4 * k] + v.y * pr[4 * k + 1] + v.z * pr[4 * k + 2] + v.w * pr[4 * k + 3];
      }
      float y = (P.gmask[(int64_t(b) * P.G + cj[i]) * words + (pi >> 5)] >> (pi & 31)) & 1u ? 1.f : 0.f;
      lsum += bce_logits(z, y) * cinv[i];
      if (P.dproto) {
        float g = (sigm(z) - y) * cinv[i] * gk;
        for (int k = 0; k < 8; k++) {
          float4 v = cv[k];
          acc[4 * k] += g * v.x; acc[4 * k + 1] += g * v.y; acc[4 * k + 2] += g * v.z; acc[4 * k + 3] += g * v.w;
        }
      }
    }
    __syncthreads();
  }
  if (P.dproto) st32(P.dproto + pix * 32, acc);
  block_add(lsum, &P.items[9]);
}
__global__ void k_mask_coef(MaskArgs P) {
  __shared__ float red[kT / 32][32];
  __shared__ float cf[32];
  float fg = fmaxf(P.items[4], 1.f);
  float hw = float(P.Hp) * P.Wp;
  int words = P.Hp * P.Wp / 32;
  for (int b = 0; b < P.B; b++) {
    int np = P.npos[b];
    for (int i = blockIdx.x; i < np; i += gridDim.x) {
      int a = P.pos[b * P.A + i];
      int j = P.agt[b * P.A + a];
      int l, h, w;
      anchor_of(P.L, a, l, h, w);
      int64_t apix = (int64_t(b) * P.L.h[l] + h) * P.L.w[l] + w;
      __syncthreads();
      if (threadIdx.x < 32) cf[threadIdx.x] = to_f(P.coef[l][apix * 32 + threadIdx.x]);
      __syncthreads();
      float4 tg = P.gtb[b * P.G + j];
      float x1, y1, x2, y2, area;
      crop_of(P, tg, x1, y1, x2, y2, area);
      int c0 = max(0, int(ceilf(x1))), c1 = min(P.Wp, int(ceilf(x2)));
      int r0 = max(0, int(ceilf(y1))), r1 = min(P.Hp, int(ceilf(y2)));
      float acc[32];
      for (int k = 0; k < 32; k++) acc[k] = 0.f;
      int nw = c1 - c0, n = nw > 0 && r1 > r0 ? nw * (r1 - r0) : 0;
      float sc = P.box_gain * float(P.B) / (fg * hw * area) * *P.scale;
      for (int t = threadIdx.x; t < n; t += blockDim.x) {
        int py = r0 + t / nw, px = c0 + t % nw;
        float pr[32], z = 0.f;
        ld32(P.proto + ((int64_t(b) * P.Hp + py) * P.Wp + px) * 32, pr);
        for (int k = 0; k < 32; k++) z += cf[k] * pr[k];
        int pi = py * P.Wp + px;
        float y = (P.gmask[(int64_t(b) * P.G + j) * words + (pi >> 5)] >> (pi & 31)) & 1u ? 1.f : 0.f;
        float g = (sigm(z) - y) * sc;
        for (int k = 0; k < 32; k++) acc[k] += g * pr[k];
      }
      for (int k = 0; k < 32; k++) {
        float v = acc[k];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5][k] = v;
      }
      __syncthreads();
      if (threadIdx.x < 32) {
        float v = 0.f;
        for (int wp = 0; wp < int(blockDim.x / 32); wp++) v += red[wp][threadIdx.x];
        P.dcoef[l][apix * 32 + threadIdx.x] = from_f<act_t>(v);
      }
    }
  }
}
__global__ void k_final26(float* items, float box_gain, float cls_gain, float l1_gain) {
  float tss = fmaxf(items[5], 1.f), fg = items[4];
  items[0] = items[8] / tss * box_gain;
  items[1] = fg > 0.f ? items[9] / fg * box_gain : 0.f;
  items[2] = items[10] / tss * cls_gain;
  items[3] = items[11] / tss * l1_gain;
}

// ---------------------------------------------------------------- semseg (nc = 1)
// 목표 = 정답 마스크 합집합(proto 해상도)을 최근접으로 semseg 해상도에 (F.interpolate nearest: 원본 (y·Hp/Hs, x·Wp/Ws)).
// 무시 영역 화소는 BCE·Dice 모두에서 뺀다(ObjectSAM 라벨 규칙: 물체라고도 배경이라고도 가르치지 않음).
// acc: [b*4+0] Σp, [+1] Σp·t, [+2] Σt, [+3] 유효 화소 수, [B*4] BCE 합
__global__ void k_sem_red(int B, int Hs, int Ws, int Hp, int Wp, int G, const int* n_gt, const uint32_t* gmask,
                          const uint32_t* ignore, const act_t* sem, float* acc) {
  const int64_t n = int64_t(B) * Hs * Ws;
  const int words = Hp * Wp / 32;
  float sp = 0, spt = 0, stt = 0, cnt = 0, bce = 0;
  int bcur = -1;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int b = int(i / (int64_t(Hs) * Ws)), r = int(i - int64_t(b) * Hs * Ws), y = r / Ws, x = r - y * Ws;
    if (b != bcur) {
      if (bcur >= 0) {
        atomicAdd(&acc[bcur * 4 + 0], sp); atomicAdd(&acc[bcur * 4 + 1], spt);
        atomicAdd(&acc[bcur * 4 + 2], stt); atomicAdd(&acc[bcur * 4 + 3], cnt);
      }
      sp = spt = stt = cnt = 0; bcur = b;
    }
    int pi = (y * Hp / Hs) * Wp + (x * Wp / Ws);
    if (ignore && gt_bit(ignore + int64_t(b) * words, pi)) continue;
    float t = 0.f;
    for (int j = 0; j < n_gt[b]; j++)
      if (gt_bit(gmask + (int64_t(b) * G + j) * words, pi)) { t = 1.f; break; }
    float z = to_f(sem[i]), p = sigm(z);
    sp += p; spt += p * t; stt += t; cnt += 1.f; bce += bce_logits(z, t);
  }
  if (bcur >= 0) {
    atomicAdd(&acc[bcur * 4 + 0], sp); atomicAdd(&acc[bcur * 4 + 1], spt);
    atomicAdd(&acc[bcur * 4 + 2], stt); atomicAdd(&acc[bcur * 4 + 3], cnt);
  }
  block_add(bce, &acc[B * 4]);
}
// 기울기: d/dz [ ½·mean BCE + ½·mean_b(1 − (2I+1)/(U+1)) ] × box_gain × B × scale·w
__global__ void k_sem_grad(int B, int Hs, int Ws, int Hp, int Wp, int G, const int* n_gt, const uint32_t* gmask,
                           const uint32_t* ignore, const act_t* sem, act_t* dsem, const float* acc, const float* scale,
                           float gain) {
  const int64_t n = int64_t(B) * Hs * Ws;
  const int words = Hp * Wp / 32;
  float ntot = 0.f;
  for (int b = 0; b < B; b++) ntot += acc[b * 4 + 3];
  ntot = fmaxf(ntot, 1.f);
  const float k = gain * float(B) * *scale;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int b = int(i / (int64_t(Hs) * Ws)), r = int(i - int64_t(b) * Hs * Ws), y = r / Ws, x = r - y * Ws;
    int pi = (y * Hp / Hs) * Wp + (x * Wp / Ws);
    if (ignore && gt_bit(ignore + int64_t(b) * words, pi)) { dsem[i] = from_f<act_t>(0.f); continue; }
    float t = 0.f;
    for (int j = 0; j < n_gt[b]; j++)
      if (gt_bit(gmask + (int64_t(b) * G + j) * words, pi)) { t = 1.f; break; }
    float p = sigm(to_f(sem[i]));
    float I = acc[b * 4 + 1], U = acc[b * 4 + 0] + acc[b * 4 + 2];
    float ddice = -(2.f * t * (U + 1.f) - (2.f * I + 1.f)) / ((U + 1.f) * (U + 1.f)) * p * (1.f - p);
    float g = 0.5f * (p - t) / ntot + 0.5f * ddice / float(B);
    dsem[i] = from_f<act_t>(g * k);
  }
}

__global__ void k_scales(const float* scale, const float* w, float* out) {
  out[0] = *scale * *w;
  out[1] = *scale * (1.f - *w);
}
__global__ void k_combine(int B, const float* a, const float* o, const float* sacc, const float* w, float gain,
                          float* items) {
  float ntot = 0.f, dice = 0.f;
  for (int b = 0; b < B; b++) {
    ntot += sacc[b * 4 + 3];
    float I = sacc[b * 4 + 1], U = sacc[b * 4 + 0] + sacc[b * 4 + 2];
    dice += 1.f - (2.f * I + 1.f) / (U + 1.f);
  }
  float sem = (0.5f * sacc[B * 4] / fmaxf(ntot, 1.f) + 0.5f * dice / float(B)) * gain;
  float W = *w, so = 0.f, s2 = 0.f;
  for (int k = 0; k < 4; k++) { items[k] = a[k]; items[4 + k] = o[k]; so += a[k]; s2 += o[k]; }
  items[8] = sem;
  items[9] = W * so + (1.f - W) * s2 + sem;
  items[10] = a[4];
  items[11] = o[4];
  items[12] = W;
}

Lv make_lv(const Loss26Cfg& c) {
  Lv L;
  L.off[0] = 0;
  for (int l = 0; l < 3; l++) {
    L.h[l] = c.lvl_h[l]; L.w[l] = c.lvl_w[l]; L.s[l] = c.stride[l];
    L.off[l + 1] = L.off[l] + L.h[l] * L.w[l];
  }
  return L;
}
}  // namespace

void Loss26Branch::init(const Loss26Cfg& c, int topk_, int topk2_) {
  cfg = c;
  topk = topk_;
  topk2 = topk2_;
  ncand_ = 3 * topk;
  A_ = c.A();
  FSO_ASSERT(ncand_ <= kKC, "3·topk ≤ %d", kKC);
  FSO_ASSERT(topk2 == topk || topk2 == 1, "topk2: topk or 1");
  FSO_ASSERT(c.Hp % 16 == 0 && c.Wp % 16 == 0 && (c.Hp * c.Wp) % 32 == 0, "proto size");
  const int64_t BA = int64_t(c.B) * A_, BG = int64_t(c.B) * c.G;
  pbox_ = dalloc<float4>(BA);
  score_ = dalloc<float>(BA);
  cnt_ = dalloc<int>(BA);
  CUDA_CHECK(cudaMemset(cnt_, 0, BA * sizeof(int)));
  who_ = dalloc<int>(BA);
  agt_ = dalloc<int>(BA);
  amet_ = dalloc<float>(BA);
  aovl_ = dalloc<float>(BA);
  tsc_ = dalloc<float>(BA);
  galign_ = dalloc<float>(BG);
  govl_ = dalloc<float>(BG);
  cand_a_ = dalloc<int>(BG * ncand_);
  cand_ov_ = dalloc<float>(BG * ncand_);
  gbest_ = dalloc<int>(BG);
  gwin_ = dalloc<int>(BG);
  pos_ = dalloc<int>(BA);
  npos_ = dalloc<int>(c.B);
  items_ = dalloc<float>(16);
}

void Loss26Branch::run(const Head26IO& io, const act_t* proto, act_t* dproto, const GtBatch& gt, const int* garea,
                       const float* scale, cudaStream_t st) {
  const Lv L = make_lv(cfg);
  const int B = cfg.B, A = A_, G = cfg.G;
  const int nq = cdiv(int64_t(B) * A, kT);
  const int64_t BG = int64_t(B) * G;
  CUDA_CHECK(cudaMemsetAsync(items_, 0, 16 * sizeof(float), st));
  CUDA_CHECK(cudaMemsetAsync(galign_, 0, BG * sizeof(float), st));
  CUDA_CHECK(cudaMemsetAsync(govl_, 0, BG * sizeof(float), st));
  CUDA_CHECK(cudaMemsetAsync(npos_, 0, B * sizeof(int), st));
  Dec1 dc;
  for (int l = 0; l < 3; l++) { dc.box[l] = io.box[l]; dc.cls[l] = io.cls[l]; }
  k_decode1<<<nq, kT, 0, st>>>(L, A, B, dc, pbox_, score_);
  CandArgs C;
  C.L = L; C.A = A; C.G = G; C.ncand = ncand_; C.topk = topk; C.Hp = cfg.Hp; C.Wp = cfg.Wp; C.imgsz = cfg.imgsz;
  C.n_gt = gt.n_gt; C.gtb = gt.box; C.pbox = pbox_; C.score = score_;
  for (int l = 0; l < 3; l++) C.coef[l] = io.coef[l];
  C.proto = proto; C.gmask = gt.mask; C.garea = garea; C.cnt = cnt_; C.who = who_; C.cand_a = cand_a_; C.cand_ov = cand_ov_;
  k_cand<<<int(BG), kT, 0, st>>>(C);
  k_resolve26<<<nq, kT, 0, st>>>(L, A, B, G, ncand_, gt.n_gt, gt.box, pbox_, score_, cnt_, who_, cand_a_, cand_ov_, agt_,
                                 amet_, aovl_);
  if (topk2 < topk) {
    CUDA_CHECK(cudaMemsetAsync(gbest_, 0, BG * sizeof(int), st));
    CUDA_CHECK(cudaMemsetAsync(gwin_, 0x7f, BG * sizeof(int), st));
    k_best<<<nq, kT, 0, st>>>(A, B, G, agt_, amet_, gbest_);
    k_win<<<nq, kT, 0, st>>>(A, B, G, agt_, amet_, gbest_, gwin_);
    k_keep<<<nq, kT, 0, st>>>(A, B, G, agt_, gwin_);
  }
  k_pos26<<<nq, kT, 0, st>>>(A, B, G, agt_, amet_, aovl_, galign_, govl_, pos_, npos_, items_);
  k_tsum<<<nq, kT, 0, st>>>(A, B, G, npos_, pos_, agt_, amet_, galign_, govl_, tsc_, items_);
  Det1 D;
  D.L = L; D.A = A; D.B = B; D.G = G; D.imgsz = cfg.imgsz; D.Hp = cfg.Hp; D.Wp = cfg.Wp;
  for (int l = 0; l < 3; l++) {
    D.box[l] = io.box[l]; D.cls[l] = io.cls[l]; D.dbox[l] = io.dbox[l]; D.dcls[l] = io.dcls[l]; D.dcoef[l] = io.dcoef[l];
  }
  D.agt = agt_; D.tsc = tsc_; D.gtb = gt.box; D.ignore = gt.ignore; D.items = items_; D.scale = scale;
  D.box_gain = cfg.box_gain; D.cls_gain = cfg.cls_gain; D.l1_gain = cfg.l1_gain;
  k_det1<<<nq, kT, 0, st>>>(D);
  MaskArgs M;
  M.L = L; M.A = A; M.B = B; M.G = G; M.imgsz = cfg.imgsz; M.Hp = cfg.Hp; M.Wp = cfg.Wp;
  for (int l = 0; l < 3; l++) { M.coef[l] = io.coef[l]; M.dcoef[l] = io.dcoef[l]; }
  M.proto = proto; M.dproto = dproto; M.npos = npos_; M.pos = pos_; M.agt = agt_; M.gtb = gt.box; M.gmask = gt.mask;
  M.items = items_; M.scale = scale; M.box_gain = cfg.box_gain;
  k_mask_tile<<<dim3((cfg.Hp / 16) * (cfg.Wp / 16), B), kT, 0, st>>>(M);
  k_mask_coef<<<280, kT, 0, st>>>(M);
  k_final26<<<1, 1, 0, st>>>(items_, cfg.box_gain, cfg.cls_gain, cfg.l1_gain);
}

void Loss26::init(const Loss26Cfg& c) {
  cfg = c;
  o2m.init(c, 10, 10);
  o2o.init(c, 7, 1);
  garea_ = dalloc<int>(int64_t(c.B) * c.G);
  scales_ = dalloc<float>(2);
  sem_acc_ = dalloc<float>(c.B * 4 + 4);
  items_ = dalloc<float>(16);
  wdev_ = dalloc<float>(1);
  set_epoch(0, 50);
}

void Loss26::set_epoch(int epoch, int epochs) {
  // E2ELoss: updates = epoch, o2m = max(1 − updates/max(epochs−1, 1), 0)·(0.8 − 0.1) + 0.1
  w = std::max(1.f - float(epoch) / float(std::max(epochs - 1, 1)), 0.f) * (0.8f - 0.1f) + 0.1f;
  CUDA_CHECK(cudaMemcpy(wdev_, &w, sizeof(float), cudaMemcpyHostToDevice));
}

void Loss26::run(const Head26IO& a, const Head26IO& o, const act_t* proto, act_t* dproto, const act_t* sem, act_t* dsem,
                 const GtBatch& gt, const float* scale, cudaStream_t st) {
  const int B = cfg.B, G = cfg.G, words = cfg.Hp * cfg.Wp / 32;
  k_scales<<<1, 1, 0, st>>>(scale, wdev_, scales_);
  k_garea<<<B * G, kT, 0, st>>>(G, words, gt.n_gt, gt.mask, garea_);
  o2m.run(a, proto, dproto, gt, garea_, scales_, st);
  o2o.run(o, proto, nullptr, gt, garea_, scales_ + 1, st);
  CUDA_CHECK(cudaMemsetAsync(sem_acc_, 0, (B * 4 + 4) * sizeof(float), st));
  const int64_t ns = int64_t(B) * cfg.Hs * cfg.Ws;
  const int nb = int(std::min<int64_t>(cdiv(ns, kT), 70 * 8));
  k_sem_red<<<nb, kT, 0, st>>>(B, cfg.Hs, cfg.Ws, cfg.Hp, cfg.Wp, G, gt.n_gt, gt.mask, gt.ignore, sem, sem_acc_);
  k_sem_grad<<<nb, kT, 0, st>>>(B, cfg.Hs, cfg.Ws, cfg.Hp, cfg.Wp, G, gt.n_gt, gt.mask, gt.ignore, sem, dsem, sem_acc_,
                                scales_, cfg.box_gain);
  k_combine<<<1, 1, 0, st>>>(B, o2m.items(), o2o.items(), sem_acc_, wdev_, cfg.box_gain, items_);
}
