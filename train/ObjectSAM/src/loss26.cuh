// YOLO26-seg 손실(Ultralytics 8.4 E2ELoss(SegmentLoss26), overlap_mask=False, 클래스 1)의 GPU 구현.
//
//   전체 = w·L(one2many) + (1−w)·L(one2one),  w: 0.8 → 0.1 (epoch 따라 선형, E2ELoss.decay)
//   L = box(CIoU)·7.5 + seg(마스크 BCE)·7.5 + cls(BCE)·0.5 + l1(정규화 거리 L1)·6.0 + sem(BCE·½ + Dice·½)·7.5
//   할당: SegTaskAlignedAssigner — 겹침 = (상자 CIoU + 예측 마스크 IoU)/2, 마스크 IoU 는 정답마다 상자 지표 상위
//         3·topk 앵커만. one2many topk 10, one2one topk 7 → 충돌 해소 뒤 정답마다 1 개(topk2 = 1).
//   one2one 은 proto 를 detach 해서 받는다(마스크 손실이 proto 로 기울기를 보내지 않음). semseg 도 one2one 쪽은 detach
//   → sem 기울기는 w 배, 값은 1 배.
// 버퍼는 고정 크기, 개수는 장치 메모리에서 읽는다 → 호스트 동기 없음, CUDA graph 가능.
#pragma once
#include "loss.cuh"

struct Loss26Cfg {
  int B = 2, imgsz = 256;
  int lvl_h[3], lvl_w[3];
  int stride[3] = {8, 16, 32};
  int Hp, Wp;             // proto = 입력/4, 정답 마스크도 이 해상도
  int Hs, Ws;             // semseg = 입력/8
  int G = 256;
  float box_gain = 7.5f, cls_gain = 0.5f, l1_gain = 6.0f;
  int A() const { return lvl_h[0] * lvl_w[0] + lvl_h[1] * lvl_w[1] + lvl_h[2] * lvl_w[2]; }
};

// 머리 출력·기울기(NHWC act_t): box [B,H,W,4] (ltrb 거리, 격자 단위), cls [B,H,W,1], coef [B,H,W,32]
struct Head26IO {
  const act_t* box[3];
  const act_t* cls[3];
  const act_t* coef[3];
  act_t* dbox[3];
  act_t* dcls[3];
  act_t* dcoef[3];
};

class Loss26Branch {
 public:
  void init(const Loss26Cfg& c, int topk, int topk2);
  // dproto: nullptr 이면 proto 로 기울기를 보내지 않는다(one2one). 계수·머리 기울기는 늘 쓴다.
  void run(const Head26IO& io, const act_t* proto, act_t* dproto, const GtBatch& gt, const int* garea,
           const float* scale, cudaStream_t st);
  const float* items() const { return items_; }  // [box, seg, cls, l1, fg, tss] (이득 곱, 배치 곱 전)
  Loss26Cfg cfg;
  int topk = 10, topk2 = 10;

 private:
  int A_ = 0, ncand_ = 30;
  float4* pbox_ = nullptr;
  float* score_ = nullptr;
  int *cnt_ = nullptr, *who_ = nullptr, *agt_ = nullptr;
  float *amet_ = nullptr, *aovl_ = nullptr, *tsc_ = nullptr;
  float *galign_ = nullptr, *govl_ = nullptr;
  int *cand_a_ = nullptr;      // [B*G][ncand] 정답별 후보 앵커(−1 끝)
  float* cand_ov_ = nullptr;   // [B*G][ncand] 합친 겹침
  int *gbest_ = nullptr, *gwin_ = nullptr;  // topk2 = 1: 정답별 최대 지표(float 비트)·이긴 앵커
  int *pos_ = nullptr, *npos_ = nullptr;
  float* items_ = nullptr;
};

class Loss26 {
 public:
  void init(const Loss26Cfg& c);
  // epoch 시작마다(호스트): one2many 가중 w = E2ELoss.decay(epoch) (Ultralytics: update() 가 epoch 끝마다)
  void set_epoch(int epoch, int epochs);
  void run(const Head26IO& o2m, const Head26IO& o2o, const act_t* proto, act_t* dproto, const act_t* sem,
           act_t* dsem, const GtBatch& gt, const float* scale, cudaStream_t st);
  // [0..3] o2m box seg cls l1, [4..7] o2o box seg cls l1, [8] sem, [9] 합계(= Ultralytics loss.sum()/B 와 같은 꼴),
  // [10] o2m 양성 수, [11] o2o 양성 수, [12] w
  const float* items() const { return items_; }
  float w = 0.8f;
  Loss26Cfg cfg;
  Loss26Branch o2m, o2o;

 private:
  int* garea_ = nullptr;     // [B*G] 정답 마스크 화소 수
  float* scales_ = nullptr;  // [2] 장치: scale·w, scale·(1−w)
  float* sem_acc_ = nullptr; // [B*4 + 4] semseg 영상별 Σp, Σpt, Σt, 유효 화소 수 + BCE 합
  float* items_ = nullptr;
  float* wdev_ = nullptr;
};
