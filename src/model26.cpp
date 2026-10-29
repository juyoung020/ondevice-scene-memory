// YOLO26s-seg (Ultralytics 8.4 ultralytics/cfg/models/26/yolo26-seg.yaml, scale s, end2end, reg_max 1).
// 이름은 PyTorch state_dict 그대로(model.<층>.…), 채널은 scale s(폭 0.5, 깊이 0.5 → 반복 1).
#include <algorithm>

#include "model.h"

// Bottleneck(c1, c2, shortcut, k=(3,3), e): cv1 Conv(c1, hidden, 3) → cv2 Conv(hidden, c2, 3) (+x)
void Model::bottleneck(const std::string& name, Act* in, int c2, int hidden, bool shortcut, std::vector<DestSpec> out) {
  Act* t = new_act(name + ".t", hidden, in->H, in->W);
  conv_bn(name + ".cv1", in, hidden, 3, 1, {{t, 0, 0, hidden, 1}});
  conv_bn(name + ".cv2", t, c2, 3, 1, out, shortcut && in->C == c2 ? in : nullptr);
}

// C3k(c, c, n=2, shortcut, k=3): cv3(cat(m(cv1(x)), cv2(x))), c_ = c2/2, m = Bottleneck(c_, c_, e=1) × 2
void Model::c3k(const std::string& name, Act* in, int c2, bool shortcut, std::vector<DestSpec> out) {
  const int c_ = c2 / 2, H = in->H, W = in->W;
  Act* cat = new_act(name + ".cat", 2 * c_, H, W);
  Act* m0 = new_act(name + ".m0.in", c_, H, W);
  conv_bn(name + ".cv1", in, c_, 1, 1, {{m0, 0, 0, c_, 1}});
  conv_bn(name + ".cv2", in, c_, 1, 1, {{cat, c_, 0, c_, 1}});
  Act* m1 = new_act(name + ".m1.in", c_, H, W);
  bottleneck(name + ".m.0", m0, c_, c_, shortcut, {{m1, 0, 0, c_, 1}});
  bottleneck(name + ".m.1", m1, c_, c_, shortcut, {{cat, 0, 0, c_, 1}});
  conv_bn(name + ".cv3", cat, c2, 1, 1, out);
}

// PSABlock(c, attn_ratio 0.5, heads = max(c/64, 1)): x = x + attn(x); x = x + ffn(x)
//   attn: qkv = Conv(c, c + 2·kd·heads, 1, act=False) → 머리별 softmax(qᵀk·scale) → v·Pᵀ + pe(v) → proj(1×1, act=False)
//   ffn : Conv(c, 2c, 1) → Conv(2c, c, 1, act=False)
void Model::psa(const std::string& name, Act* in, std::vector<DestSpec> out) {
  const int c = in->C, H = in->H, W = in->W;
  const int heads = std::max(c / 64, 1), hd = c / heads, kd = hd / 2, E = 2 * kd + hd, Cq = heads * E;
  const std::string a = name + ".attn";
  Act* qkv = new_act(a + ".qkv.out", Cq, H, W);
  Act* v = new_act(a + ".v", c, H, W);  // pe 입력: 머리별 v 를 이어 붙인 (B, C, H, W)
  std::vector<DestSpec> qd = {{qkv, 0, 0, Cq, 1}};
  for (int h = 0; h < heads; h++) qd.push_back({v, h * hd, h * E + 2 * kd, h * E + E, 1});
  conv_bn(a + ".qkv", in, Cq, 1, 1, qd, nullptr, {}, {}, false);
  Act* ao = new_act(a + ".core", c, H, W);
  {
    auto nd = std::make_unique<Node>();
    nd->kind = Node::ATTN;
    nd->name = a + ".core";
    nd->in = qkv;
    nd->out = ao;
    nd->attn.heads = heads; nd->attn.kd = kd; nd->attn.hd = hd;  // 버퍼는 finalize 에서
    nodes_.push_back(std::move(nd));
  }
  Act* xs = new_act(a + ".sum", c, H, W);
  conv_bn(a + ".pe", v, c, 3, 1, {{xs, 0, 0, c, 1}}, ao, {}, {}, false, c);  // 깊이별 3×3, + 어텐션 출력
  Act* x1 = new_act(name + ".x1", c, H, W);
  conv_bn(a + ".proj", xs, c, 1, 1, {{x1, 0, 0, c, 1}}, in, {}, {}, false);  // + x
  Act* f = new_act(name + ".ffn.h", 2 * c, H, W);
  conv_bn(name + ".ffn.0", x1, 2 * c, 1, 1, {{f, 0, 0, 2 * c, 1}});
  conv_bn(name + ".ffn.1", f, c, 1, 1, out, x1, {}, {}, false);  // + x
}

// C3k2(c1, c2, n, c3k, e, attn, shortcut) = C2f 꼴: y = cv1(x).chunk(2); y += m_j(y[-1]); cv2(cat(y))
//   m_j: attn → Sequential(Bottleneck(c, c, e=0.5), PSABlock(c)), c3k → C3k(c, c, 2), 아니면 Bottleneck(c, c, e=0.5)
void Model::c3k2(const std::string& name, Act* in, int c2, int n, bool c3k_, float e, bool attn, bool shortcut,
                 std::vector<DestSpec> out) {
  const int c = int(c2 * e), H = in->H, W = in->W;
  Act* cat = new_act(name + ".cat", (2 + n) * c, H, W);
  Act* bin = new_act(name + ".m0.in", c, H, W);
  conv_bn(name + ".cv1", in, 2 * c, 1, 1, {{cat, 0, 0, 2 * c, 1}, {bin, 0, c, 2 * c, 1}});
  for (int j = 0; j < n; j++) {
    const std::string mj = name + ".m." + std::to_string(j);
    std::vector<DestSpec> d = {{cat, (2 + j) * c, 0, c, 1}};
    Act* next = nullptr;
    if (j + 1 < n) { next = new_act(name + ".m" + std::to_string(j + 1) + ".in", c, H, W); d.push_back({next, 0, 0, c, 1}); }
    if (attn) {
      Act* t = new_act(mj + ".0.out", c, H, W);
      bottleneck(mj + ".0", bin, c, c / 2, shortcut, {{t, 0, 0, c, 1}});
      psa(mj + ".1", t, d);
    } else if (c3k_) {
      c3k(mj, bin, c, shortcut, d);
    } else {
      bottleneck(mj, bin, c, c / 2, shortcut, d);
    }
    bin = next;
  }
  conv_bn(name + ".cv2", cat, c2, 1, 1, out);
}

// Segment26 머리 한 단계(Detect 비legacy + 마스크 계수). o2o 면 one2one_ 사본이고 입력 기울기를 막는다.
//   cv2: Conv(x,32,3)·Conv(32,32,3)·1×1(4)   cv3: [DW(x)+Conv(x,128,1)]·[DW(128)+Conv(128,128,1)]·1×1(nc)
//   cv4: Conv(x,32,3)·Conv(32,32,3)·1×1(32)  (cv2·cv4 첫 conv 는 같은 입력이라 하나로 합친다)
void Model::head26(const std::string& pre, Act* x, int l, bool o2o) {
  const std::string L = std::to_string(l), p = pre + (o2o ? "one2one_" : "");
  const std::string tag = std::string(o2o ? "o2o" : "o2m") + ".h" + L;
  const int H = x->H, W = x->W, c2 = 32, c3 = 128, c4 = 32, nm = 32;
  Act* b1 = new_act(tag + ".b1", c2, H, W);
  Act* m1 = new_act(tag + ".m1", c4, H, W);
  conv_bn(tag + ".bm0", x, c2 + c4, 3, 1, {{b1, 0, 0, c2, 1}, {m1, 0, c2, c2 + c4, 1}}, nullptr,
          {p + "cv2." + L + ".0", p + "cv4." + L + ".0"}, {c2, c4});
  nodes_.back()->detach = o2o;
  Act* b2 = new_act(tag + ".b2", c2, H, W);
  conv_bn(p + "cv2." + L + ".1", b1, c2, 3, 1, {{b2, 0, 0, c2, 1}});
  Act* m2 = new_act(tag + ".m2", c4, H, W);
  conv_bn(p + "cv4." + L + ".1", m1, c4, 3, 1, {{m2, 0, 0, c4, 1}});
  // 점수 가지
  Act* d0 = new_act(tag + ".c0dw", x->C, H, W);
  conv_bn(p + "cv3." + L + ".0.0", x, x->C, 3, 1, {{d0, 0, 0, x->C, 1}}, nullptr, {}, {}, true, x->C);
  nodes_.back()->detach = o2o;
  Act* c0 = new_act(tag + ".c0", c3, H, W);
  conv_bn(p + "cv3." + L + ".0.1", d0, c3, 1, 1, {{c0, 0, 0, c3, 1}});
  Act* d1 = new_act(tag + ".c1dw", c3, H, W);
  conv_bn(p + "cv3." + L + ".1.0", c0, c3, 3, 1, {{d1, 0, 0, c3, 1}}, nullptr, {}, {}, true, c3);
  Act* c1 = new_act(tag + ".c1", c3, H, W);
  conv_bn(p + "cv3." + L + ".1.1", d1, c3, 1, 1, {{c1, 0, 0, c3, 1}});
  Act* (*hd)[3] = o2o ? head2_ : head_;
  hd[0][l] = new_act("out." + tag + ".box", 4, H, W);
  hd[1][l] = new_act("out." + tag + ".cls", cfg.nc, H, W);
  hd[2][l] = new_act("out." + tag + ".coef", nm, H, W);
  head_conv(p + "cv2." + L + ".2", b2, 4, hd[0][l]);
  head_conv(p + "cv3." + L + ".2", c1, cfg.nc, hd[1][l]);
  head_conv(p + "cv4." + L + ".2", m2, nm, hd[2][l]);
}

void Model::build26() {
  const int S = cfg.imgsz;
  auto M = [](int i) { return "model." + std::to_string(i); };
  Act* in = new_act("in", kInC, S, S);
  in->need_grad = false;
  // ---- 백본
  Act* a0 = new_act("a0", 32, S / 2, S / 2);
  conv_bn(M(0), in, 32, 3, 2, {{a0, 0, 0, 0, 1}});
  Act* a1 = new_act("a1", 64, S / 4, S / 4);
  conv_bn(M(1), a0, 64, 3, 2, {{a1, 0, 0, 0, 1}});
  Act* a2 = new_act("a2", 128, S / 4, S / 4);
  c3k2(M(2), a1, 128, 1, false, 0.25f, false, true, {{a2, 0, 0, 0, 1}});
  Act* a3 = new_act("a3", 128, S / 8, S / 8);
  conv_bn(M(3), a2, 128, 3, 2, {{a3, 0, 0, 0, 1}});
  // 목 concat 버퍼: cat12 = [up(10) 512, 6 256] @S/16, cat15 = [up(13) 256, 4 256] @S/8,
  //                cat18 = [17 128, 13 256] @S/16, cat21 = [20 256, 10 512] @S/32
  Act* cat12 = new_act("cat12", 768, S / 16, S / 16);
  Act* cat15 = new_act("cat15", 512, S / 8, S / 8);
  Act* cat18 = new_act("cat18", 384, S / 16, S / 16);
  Act* cat21 = new_act("cat21", 768, S / 32, S / 32);
  Act* a4 = new_act("a4", 256, S / 8, S / 8);
  c3k2(M(4), a3, 256, 1, false, 0.25f, false, true, {{a4, 0, 0, 0, 1}, {cat15, 256, 0, 0, 1}});
  Act* a5 = new_act("a5", 256, S / 16, S / 16);
  conv_bn(M(5), a4, 256, 3, 2, {{a5, 0, 0, 0, 1}});
  Act* a6 = new_act("a6", 256, S / 16, S / 16);
  c3k2(M(6), a5, 256, 1, true, 0.5f, false, true, {{a6, 0, 0, 0, 1}, {cat12, 512, 0, 0, 1}});
  Act* a7 = new_act("a7", 512, S / 32, S / 32);
  conv_bn(M(7), a6, 512, 3, 2, {{a7, 0, 0, 0, 1}});
  Act* a8 = new_act("a8", 512, S / 32, S / 32);
  c3k2(M(8), a7, 512, 1, true, 0.5f, false, true, {{a8, 0, 0, 0, 1}});
  // SPPF(512, 512, k5, n3, shortcut): cv1 act 없음, 풀링 3번, cv2 + x
  Act* sp = new_act(M(9) + ".cat", 1024, S / 32, S / 32);
  conv_bn(M(9) + ".cv1", a8, 256, 1, 1, {{sp, 0, 0, 0, 1}}, nullptr, {}, {}, false);
  {
    auto nd = std::make_unique<Node>();
    nd->kind = Node::POOL;
    nd->name = M(9) + ".m";
    nd->cat = sp;
    nd->pc = 256;
    nodes_.push_back(std::move(nd));
  }
  Act* a9 = new_act("a9", 512, S / 32, S / 32);
  conv_bn(M(9) + ".cv2", sp, 512, 1, 1, {{a9, 0, 0, 0, 1}}, a8);
  // C2PSA(512): a, b = cv1(x).split(256); cv2(cat(a, PSABlock(b)))
  Act* c10 = new_act(M(10) + ".cat", 512, S / 32, S / 32);
  Act* b10 = new_act(M(10) + ".b", 256, S / 32, S / 32);
  conv_bn(M(10) + ".cv1", a9, 512, 1, 1, {{c10, 0, 0, 256, 1}, {b10, 0, 256, 512, 1}});
  psa(M(10) + ".m.0", b10, {{c10, 256, 0, 256, 1}});
  conv_bn(M(10) + ".cv2", c10, 512, 1, 1, {{cat12, 0, 0, 0, 2}, {cat21, 256, 0, 0, 1}});
  // ---- 목
  c3k2(M(13), cat12, 256, 1, true, 0.5f, false, true, {{cat15, 0, 0, 0, 2}, {cat18, 128, 0, 0, 1}});
  Act* p3 = new_act("p3", 128, S / 8, S / 8);
  Act* feat = new_act("proto.feat", 128, S / 8, S / 8);  // Proto26: P3 + up2(refine(P4)) + up4(refine(P5))
  c3k2(M(16), cat15, 128, 1, true, 0.5f, false, true, {{p3, 0, 0, 0, 1}, {feat, 0, 0, 0, 1}});
  conv_bn(M(17), p3, 128, 3, 2, {{cat18, 0, 0, 0, 1}});
  Act* p4 = new_act("p4", 256, S / 16, S / 16);
  c3k2(M(19), cat18, 256, 1, true, 0.5f, false, true, {{p4, 0, 0, 0, 1}});
  conv_bn(M(20), p4, 256, 3, 2, {{cat21, 0, 0, 0, 1}});
  Act* p5 = new_act("p5", 512, S / 32, S / 32);
  c3k2(M(22), cat21, 512, 1, true, 0.5f, true, true, {{p5, 0, 0, 0, 1}});
  // ---- 머리(Segment26)
  const std::string hd = M(23) + ".";
  Act* P[3] = {p3, p4, p5};
  for (int l = 0; l < 3; l++) head26(hd, P[l], l, false);
  for (int l = 0; l < 3; l++) head26(hd, P[l], l, true);
  // Proto26(ch, c_=128, nm=32, nc)
  const std::string pr = hd + "proto.";
  DestSpec f4{feat, 0, 0, 0, 2}; f4.acc = true;   // DestSpec 에 acc 를 넘긴다(아래 mk_dl)
  conv_bn(pr + "feat_refine.0", p4, 128, 1, 1, {f4});
  DestSpec f5{feat, 0, 0, 0, 4}; f5.acc = true;
  conv_bn(pr + "feat_refine.1", p5, 128, 1, 1, {f5});
  Act* ff = new_act("proto.ff", 128, S / 8, S / 8);
  conv_bn(pr + "feat_fuse", feat, 128, 3, 1, {{ff, 0, 0, 0, 1}});
  Act* pr1 = new_act("proto.1", 128, S / 8, S / 8);
  conv_bn(pr + "cv1", ff, 128, 3, 1, {{pr1, 0, 0, 0, 1}});
  Act* pru = new_act("proto.up", 128, S / 4, S / 4);
  {
    auto nd = std::make_unique<Node>();
    nd->kind = Node::CONVT;
    nd->name = pr + "upsample";
    nd->in = pr1;
    nd->out = pru;
    nd->K = 128;
    params.add(nd->name + ".weight", P_W, {128, 128, 2, 2}, 2);
    params.add(nd->name + ".bias", P_BIAS, {128});
    nd->wnames = {nd->name + ".weight"};
    nd->bias = nd->name + ".bias";
    nd->conv.init(h_, ConvGeom{cfg.batch, pr1->H, pr1->W, 128, 128, 2, 2, 2, 0}, true);
    nodes_.push_back(std::move(nd));
  }
  Act* pr2 = new_act("proto.2", 128, S / 4, S / 4);
  conv_bn(pr + "cv2", pru, 128, 3, 1, {{pr2, 0, 0, 0, 1}});
  proto_ = new_act("out.proto", 32, S / 4, S / 4);
  conv_bn(pr + "cv3", pr2, 32, 1, 1, {{proto_, 0, 0, 0, 1}});
  // semseg(학습 때만): Conv(128,128,3)·Conv(128,128,3)·1×1(nc) on feat
  Act* s0 = new_act("sem.0", 128, S / 8, S / 8);
  conv_bn(pr + "semseg.0", feat, 128, 3, 1, {{s0, 0, 0, 0, 1}});
  Act* s1 = new_act("sem.1", 128, S / 8, S / 8);
  conv_bn(pr + "semseg.1", s0, 128, 3, 1, {{s1, 0, 0, 0, 1}});
  sem_ = new_act("out.sem", cfg.nc, S / 8, S / 8);
  head_conv(pr + "semseg.2", s1, cfg.nc, sem_);
}
