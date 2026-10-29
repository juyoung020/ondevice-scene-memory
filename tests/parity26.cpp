// YOLO26s-seg 정합성: pytools/ref26.py 가 남긴 Ultralytics fp32 값과 우리 학습기의 정방향(층·머리·proto·semseg),
// 역방향(모든 가중치 기울기), BN running 통계를 비교한다.
//   build/parity26_fp32 yolo26s-seg-nc1.fsow ref26.fsow [tol]
#include <algorithm>
#include <cmath>
#include <cstring>

#include "../src/model.h"

namespace {
struct Cmp {
  double max_abs = 0, ref_max = 0, l2d = 0, l2r = 0;
  void add(double r, double v) {
    double d = std::fabs(r - v);
    max_abs = std::max(max_abs, d);
    ref_max = std::max(ref_max, std::fabs(r));
    l2d += d * d;
    l2r += r * r;
  }
  double rel() const { return std::sqrt(l2d / std::max(l2r, 1e-30)); }
};
int g_fail = 0, g_n = 0;
double g_tol = 1e-3;
bool g_quiet = false;
double g_abs0 = 1e-3;  // 참조가 이 아래로 작으면(수학적으로 0 인 기울기: BN 뒤 BN 의 β 등) 절대오차로 본다
void report(const std::string& what, const Cmp& c) {
  const bool zero = c.ref_max < g_abs0;
  bool bad = zero ? !(c.max_abs < g_abs0) : !(c.rel() <= g_tol);
  g_fail += bad;
  g_n++;
  if (bad || !g_quiet)
    printf("%-52s relL2 %.2e  maxabs %.2e  (ref max %.2e)%s\n", what.c_str(), c.rel(), c.max_abs, c.ref_max,
           bad ? "  <-- FAIL" : zero ? "  (≈0, abs)" : "");
}
std::vector<float> d2h(const act_t* p, int64_t n) {
  std::vector<act_t> t(n);
  CUDA_CHECK(cudaMemcpy(t.data(), p, n * sizeof(act_t), cudaMemcpyDeviceToHost));
  std::vector<float> o(n);
  for (int64_t i = 0; i < n; i++) o[i] = to_f(t[i]);
  return o;
}
void h2d(act_t* p, const std::vector<float>& v) {
  std::vector<act_t> t(v.size());
  for (size_t i = 0; i < v.size(); i++) t[i] = from_f<act_t>(v[i]);
  CUDA_CHECK(cudaMemcpy(p, t.data(), t.size() * sizeof(act_t), cudaMemcpyHostToDevice));
}
// 참조 NCHW (B,C,H,W) 와 우리 NHWC 버퍼의 채널 [choff, choff+C)
Cmp cmp_nchw(const HostTensor& ref, const Act* a, int choff) {
  int64_t B = ref.shape[0], C = ref.shape[1], H = ref.shape[2], W = ref.shape[3];
  FSO_ASSERT(H == a->H && W == a->W, "%s: %lldx%lld vs %dx%d", a->name.c_str(), (long long)H, (long long)W, a->H, a->W);
  auto v = d2h(a->p, B * a->H * a->W * a->C);
  Cmp c;
  const float* r = ref.f32();
  for (int64_t b = 0; b < B; b++)
    for (int64_t ch = 0; ch < C; ch++)
      for (int64_t y = 0; y < H; y++)
        for (int64_t x = 0; x < W; x++)
          c.add(r[((b * C + ch) * H + y) * W + x], v[((b * a->H + y) * a->W + x) * a->C + choff + ch]);
  return c;
}
// 참조 (B, C, A=Σ H_l·W_l) ↔ 단계별 NHWC 버퍼
Cmp cmp_flat(const HostTensor& ref, Act* const lv[3]) {
  int64_t B = ref.shape[0], C = ref.shape[1], A = ref.shape[2];
  const float* r = ref.f32();
  Cmp c;
  int64_t off = 0;
  for (int l = 0; l < 3; l++) {
    const Act* a = lv[l];
    FSO_ASSERT(a->C == C, "%s channels", a->name.c_str());
    auto v = d2h(a->p, B * a->H * a->W * C);
    for (int64_t b = 0; b < B; b++)
      for (int64_t ch = 0; ch < C; ch++)
        for (int64_t i = 0; i < int64_t(a->H) * a->W; i++) c.add(r[(b * C + ch) * A + off + i], v[(b * a->H * a->W + i) * C + ch]);
    off += int64_t(a->H) * a->W;
  }
  FSO_ASSERT(off == A, "anchors %lld vs %lld", (long long)off, (long long)A);
  return c;
}
void set_grad_flat(const HostTensor& G, Act* const lv[3]) {
  int64_t B = G.shape[0], C = G.shape[1], A = G.shape[2];
  const float* r = G.f32();
  int64_t off = 0;
  for (int l = 0; l < 3; l++) {
    Act* a = lv[l];
    std::vector<float> v(B * a->H * a->W * C);
    for (int64_t b = 0; b < B; b++)
      for (int64_t ch = 0; ch < C; ch++)
        for (int64_t i = 0; i < int64_t(a->H) * a->W; i++) v[(b * a->H * a->W + i) * C + ch] = r[(b * C + ch) * A + off + i];
    h2d(a->g, v);
    off += int64_t(a->H) * a->W;
  }
}
void set_grad_nchw(const HostTensor& G, Act* a) {
  int64_t B = G.shape[0], C = G.shape[1], H = G.shape[2], W = G.shape[3];
  const float* r = G.f32();
  std::vector<float> v(B * H * W * C);
  for (int64_t b = 0; b < B; b++)
    for (int64_t ch = 0; ch < C; ch++)
      for (int64_t y = 0; y < H; y++)
        for (int64_t x = 0; x < W; x++) v[((b * H + y) * W + x) * C + ch] = r[((b * C + ch) * H + y) * W + x];
  h2d(a->g, v);
}
}  // namespace

int main(int argc, char** argv) {
  FSO_ASSERT(argc >= 3, "usage: parity26 <weights.fsow> <ref26.fsow> [tol]");
  if (argc > 3) g_tol = atof(argv[3]);
  g_quiet = getenv("FSO_QUIET") != nullptr;
  auto W = fsow_load(argv[1]);
  auto R = fsow_load(argv[2]);
  const HostTensor& in = R.at("in");
  const int B = int(in.shape[0]), S = int(in.shape[2]);
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudnnHandle_t h;
  CUDNN_CHECK(cudnnCreate(&h));
  CUDNN_CHECK(cudnnSetStream(h, st));
  Model m;
  ModelCfg cfg;
  cfg.arch = 26; cfg.nc = 1; cfg.batch = B; cfg.imgsz = S; cfg.reg_max = 1;
  m.build(cfg, h, st);
  m.print_plan();
  printf("learnable params: %ld (Ultralytics yolo26s-seg nc=1: 11,434,463)\n",
         long(m.params.size[P_W] + m.params.size[P_GAMMA] + m.params.size[P_BIAS]));
  m.params.load(W);
  m.sync_weights();
  {  // 입력 NCHW → NHWC
    std::vector<float> v(int64_t(B) * S * S * kInC, 0.f);
    const float* r = in.f32();
    for (int b = 0; b < B; b++)
      for (int c = 0; c < 3; c++)
        for (int y = 0; y < S; y++)
          for (int x = 0; x < S; x++) v[((int64_t(b) * S + y) * S + x) * kInC + c] = r[((int64_t(b) * 3 + c) * S + y) * S + x];
    h2d(m.input()->p, v);
  }
  m.forward(true);
  CUDA_CHECK(cudaStreamSynchronize(st));

  // 층 출력: (참조 층 번호, 우리 버퍼, 채널 시작)
  struct L { int i; const char* act; int choff; };
  const L layers[] = {{0, "a0", 0},     {1, "a1", 0},     {2, "a2", 0},     {3, "a3", 0},     {4, "a4", 0},
                      {5, "a5", 0},     {6, "a6", 0},     {7, "a7", 0},     {8, "a8", 0},     {9, "a9", 0},
                      {10, "cat21", 256}, {11, "cat12", 0}, {12, "cat12", 0}, {13, "cat18", 128}, {14, "cat15", 0},
                      {15, "cat15", 0}, {16, "p3", 0},    {17, "cat18", 0}, {18, "cat18", 0}, {19, "p4", 0},
                      {20, "cat21", 0}, {21, "cat21", 0}, {22, "p5", 0}};
  for (auto& l : layers) report("fwd layer " + std::to_string(l.i) + " → " + l.act, cmp_nchw(R.at("act." + std::to_string(l.i)), m.act(l.act), l.choff));
  Act* o2m[3][3], *o2o[3][3];
  for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++) { o2m[k][l] = m.head(k, l); o2o[k][l] = m.head2(k, l); }
  const char* kn[3] = {"boxes", "scores", "coef"};
  for (int k = 0; k < 3; k++) {
    report(std::string("fwd o2m ") + kn[k], cmp_flat(R.at(std::string("out.o2m.") + kn[k]), o2m[k]));
    report(std::string("fwd o2o ") + kn[k], cmp_flat(R.at(std::string("out.o2o.") + kn[k]), o2o[k]));
  }
  report("fwd proto", cmp_nchw(R.at("out.proto"), m.proto(), 0));
  report("fwd semseg", cmp_nchw(R.at("out.sem"), m.sem(), 0));

  // 역방향: 출력 기울기 = G
  for (int k = 0; k < 3; k++) {
    set_grad_flat(R.at(std::string("G.o2m.") + kn[k]), o2m[k]);
    set_grad_flat(R.at(std::string("G.o2o.") + kn[k]), o2o[k]);
  }
  set_grad_nchw(R.at("G.proto"), m.proto());
  set_grad_nchw(R.at("G.sem"), m.sem());
  m.backward(false);
  CUDA_CHECK(cudaStreamSynchronize(st));
  auto grads = m.params.save("", true);
  int ng = 0;
  Cmp all;
  for (auto& [name, t] : grads) {
    auto it = R.find("grad." + name);
    FSO_ASSERT(it != R.end(), "no ref grad %s", name.c_str());
    Cmp c;
    const float* r = it->second.f32();
    const float* v = t.f32();
    for (int64_t i = 0; i < t.numel(); i++) { c.add(r[i], v[i]); all.add(r[i], v[i]); }
    report("grad " + name, c);
    ng++;
  }
  report("grad ALL (" + std::to_string(ng) + " tensors)", all);
  auto run = m.params.save("", false);
  Cmp rs;
  for (auto& [name, t] : run) {
    if (name.find("running") == std::string::npos) continue;
    const float* r = R.at("run." + name).f32();
    const float* v = t.f32();
    for (int64_t i = 0; i < t.numel(); i++) rs.add(r[i], v[i]);
  }
  report("BN running stats (all)", rs);
  printf("%s: %d / %d comparisons over tolerance %.1e\n", g_fail ? "FAIL" : "PASS", g_fail, g_n, g_tol);
  return g_fail ? 1 : 0;
}
