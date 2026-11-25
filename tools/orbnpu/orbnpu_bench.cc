// ORB-SLAM3 ORB 추출을 단계별로 재고(time), 원본 결과를 쏟고(dump), NPU 입력 모자이크를 만들고(mosaic),
// NPU(또는 ONNX)가 낸 지도(FAST 점수 S · 블러)로 나머지 CPU 단계를 돌려 같은 형식으로 쏟는다(maps).
// 원본 ORBextractor.cc 를 그대로 끌어와(정적 함수 · 보호 멤버 사용) 같은 코드로 비교한다. 시험 도구.
//
// 빌드: tools/orbnpu/build.sh
// 쓰기: orbnpu_bench time   <seq> <assoc> <n>
//       orbnpu_bench dump   <seq> <assoc> <n> <out.bin>         원본 operator() 결과
//       orbnpu_bench mosaic <seq> <assoc> <n> <outdir>          모자이크 f%04d.raw(u8) + layout.txt
//       orbnpu_bench maps   <seq> <assoc> <n> <mapdir> <out.bin> [cpuS]  지도 → 점 · 기술자
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <cstring>
#define protected public
#define private public
#include "ORBextractor.cc"
#undef protected
#undef private

using namespace ORB_SLAM3;
using namespace cv;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

struct Acc { double pyr = 0, fast = 0, dist = 0, orient = 0, blur = 0, desc = 0, rest = 0, total = 0; long kps = 0; };

static std::vector<std::string> read_assoc(const std::string& seq, const std::string& assoc, int n) {
    std::ifstream f(assoc);
    std::vector<std::string> v;
    std::string t, rgb, td, dep;
    while (f >> t >> rgb >> td >> dep) { v.push_back(seq + "/" + rgb); if ((int)v.size() >= n) break; }
    return v;
}

static Mat load_gray(const std::string& p) {
    Mat im = imread(p, IMREAD_UNCHANGED), g;
    cvtColor(im, g, COLOR_RGB2GRAY);  // TUM1.yaml RGB: 1, Tracking::GrabImageRGBD 와 같음
    return g;
}

// ComputeKeyPointsOctTree 를 단계 시간만 재도록 그대로 옮김
static void kp_timed(ORBextractor& e, std::vector<std::vector<KeyPoint>>& allKeypoints, Acc& a) {
    allKeypoints.resize(e.nlevels);
    const float W = 35;
    for (int level = 0; level < e.nlevels; ++level) {
        auto t0 = Clock::now();
        const int minBorderX = EDGE_THRESHOLD - 3, minBorderY = minBorderX;
        const int maxBorderX = e.mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
        const int maxBorderY = e.mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;
        std::vector<KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(e.nfeatures * 10);
        const float width = (maxBorderX - minBorderX), height = (maxBorderY - minBorderY);
        const int nCols = width / W, nRows = height / W;
        const int wCell = ceil(width / nCols), hCell = ceil(height / nRows);
        for (int i = 0; i < nRows; i++) {
            const float iniY = minBorderY + i * hCell;
            float maxY = iniY + hCell + 6;
            if (iniY >= maxBorderY - 3) continue;
            if (maxY > maxBorderY) maxY = maxBorderY;
            for (int j = 0; j < nCols; j++) {
                const float iniX = minBorderX + j * wCell;
                float maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 6) continue;
                if (maxX > maxBorderX) maxX = maxBorderX;
                std::vector<KeyPoint> vKeysCell;
                FAST(e.mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, e.iniThFAST, true);
                if (vKeysCell.empty())
                    FAST(e.mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, e.minThFAST, true);
                for (auto& k : vKeysCell) { k.pt.x += j * wCell; k.pt.y += i * hCell; vToDistributeKeys.push_back(k); }
            }
        }
        auto t1 = Clock::now();
        std::vector<KeyPoint>& keypoints = allKeypoints[level];
        keypoints = e.DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX, minBorderY, maxBorderY, e.mnFeaturesPerLevel[level], level);
        const int scaledPatchSize = PATCH_SIZE * e.mvScaleFactor[level];
        for (auto& k : keypoints) { k.pt.x += minBorderX; k.pt.y += minBorderY; k.octave = level; k.size = scaledPatchSize; }
        auto t2 = Clock::now();
        a.fast += ms(t0, t1); a.dist += ms(t1, t2);
    }
    auto t3 = Clock::now();
    for (int level = 0; level < e.nlevels; ++level) computeOrientation(e.mvImagePyramid[level], allKeypoints[level], e.umax);
    a.orient += ms(t3, Clock::now());
}

static void time_one(ORBextractor& e, const Mat& im, Acc& a) {
    auto t0 = Clock::now();
    e.ComputePyramid(im);
    auto t1 = Clock::now();
    a.pyr += ms(t0, t1);
    std::vector<std::vector<KeyPoint>> all;
    kp_timed(e, all, a);
    for (int level = 0; level < e.nlevels; ++level) {
        auto& kps = all[level];
        if (kps.empty()) continue;
        auto b0 = Clock::now();
        Mat w = e.mvImagePyramid[level].clone();
        GaussianBlur(w, w, Size(7, 7), 2, 2, BORDER_REFLECT_101);
        auto b1 = Clock::now();
        Mat desc(kps.size(), 32, CV_8U);
        computeDescriptors(w, kps, desc, e.pattern);
        auto b2 = Clock::now();
        a.blur += ms(b0, b1); a.desc += ms(b1, b2);
        a.kps += kps.size();
    }
    a.total += ms(t0, Clock::now());
}

static void write_frame(std::ofstream& o, const std::vector<KeyPoint>& k, const Mat& d) {
    int n = k.size();
    o.write((char*)&n, 4);
    for (auto& p : k) {
        float r[5] = {p.pt.x, p.pt.y, p.angle, p.response, (float)p.octave};
        o.write((char*)r, sizeof r);
    }
    if (n) o.write((char*)d.data, n * 32);
}

// ---------- 모자이크 배치(모든 층을 테두리 붙은 채로 한 장에) ----------
struct Slot { int x, y, w, h; };  // 테두리 포함 사각형
static const int MOS_W = 1280;
static std::vector<Slot> layout(const ORBextractor& e, int cols, int rows, int& H) {
    std::vector<Slot> s;
    int x = 0, y = 0, rowh = 0;
    for (int l = 0; l < e.nlevels; ++l) {
        int w = cvRound(cols * e.mvInvScaleFactor[l]) + 2 * EDGE_THRESHOLD, h = cvRound(rows * e.mvInvScaleFactor[l]) + 2 * EDGE_THRESHOLD;
        if (x + w > MOS_W) { x = 0; y += rowh; rowh = 0; }
        s.push_back({x, y, w, h});
        x += w; rowh = std::max(rowh, h);
    }
    H = (y + rowh + 31) / 32 * 32;
    return s;
}

static Mat make_mosaic(const ORBextractor& e, const std::vector<Slot>& s, int H) {
    Mat m = Mat::zeros(H, MOS_W, CV_8U);
    for (int l = 0; l < e.nlevels; ++l) {
        const Mat& lv = e.mvImagePyramid[l];  // 테두리 안쪽 ROI, 부모가 테두리 붙은 사각형
        Size whole; Point ofs;
        lv.locateROI(whole, ofs);
        Mat full = lv;
        full.adjustROI(EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD);
        full.copyTo(m(Rect(s[l].x, s[l].y, s[l].w, s[l].h)));
    }
    return m;
}

// FAST 점수 S: 16 개 이웃 중 이어진 9 개 호의 (가운데 − 이웃) 최소 또는 (이웃 − 가운데) 최소의 최댓값, [0,255].
// 구석 ⇔ S > 문턱, OpenCV cornerScore = max(문턱, S) − 1
static Mat fast_score_cpu(const Mat& img) {
    static const int off[16][2] = {{0,3},{1,3},{2,2},{3,1},{3,0},{3,-1},{2,-2},{1,-3},{0,-3},{-1,-3},{-2,-2},{-3,-1},{-3,0},{-3,1},{-2,2},{-1,3}};
    Mat S = Mat::zeros(img.size(), CV_8U);
    for (int y = 3; y < img.rows - 3; ++y)
        for (int x = 3; x < img.cols - 3; ++x) {
            int c = img.at<uchar>(y, x), d[16];
            for (int k = 0; k < 16; ++k) d[k] = c - img.at<uchar>(y + off[k][1], x + off[k][0]);
            int best = 0;
            for (int k = 0; k < 16; ++k) {
                int mn = 1000, mx = -1000;
                for (int j = 0; j < 9; ++j) { int v = d[(k + j) & 15]; mn = std::min(mn, v); mx = std::max(mx, v); }
                best = std::max(best, std::max(mn, -mx));
            }
            S.at<uchar>(y, x) = (uchar)std::min(best, 255);
        }
    return S;
}

// 칸 하나 안에서 OpenCV FAST(문턱 th, 비최대 억제)를 S 지도로 그대로 흉내: 칸 가장자리 3 픽셀 제외, 이웃 중 구석 아닌 것은 0
static void fast_from_S(const Mat& S, int x0, int y0, int x1, int y1, int th, std::vector<KeyPoint>& out) {
    int w = x1 - x0, h = y1 - y0;
    auto sc = [&](int yy, int xx) -> int {
        if (yy < 3 || yy >= h - 3 || xx < 3 || xx >= w - 3) return 0;
        int v = S.at<uchar>(y0 + yy, x0 + xx);
        return v > th ? v : 0;
    };
    for (int y = 3; y < h - 3; ++y)
        for (int x = 3; x < w - 3; ++x) {
            int v = sc(y, x);
            if (!v) continue;
            bool keep = true;
            for (int dy = -1; dy <= 1 && keep; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    if ((dy || dx) && sc(y + dy, x + dx) >= v) { keep = false; break; }
            if (keep) out.emplace_back(Point2f((float)x, (float)y), 7.f, -1, (float)(v - 1));
        }
}

// 지도 → 원본 operator() 와 같은 점 · 기술자(가장자리 겹침 구역 없음 = RGB-D)
static void extract_from_maps(ORBextractor& e, const std::vector<Mat>& S, const std::vector<Mat>& B,
                              const std::vector<Mat>& M10, const std::vector<Mat>& M01,
                              std::vector<KeyPoint>& outk, Mat& outd) {
    std::vector<std::vector<KeyPoint>> all(e.nlevels);
    const float W = 35;
    for (int level = 0; level < e.nlevels; ++level) {
        const int minBorderX = EDGE_THRESHOLD - 3, minBorderY = minBorderX;
        const int maxBorderX = S[level].cols - EDGE_THRESHOLD + 3, maxBorderY = S[level].rows - EDGE_THRESHOLD + 3;
        std::vector<KeyPoint> dk;
        const float width = (maxBorderX - minBorderX), height = (maxBorderY - minBorderY);
        const int nCols = width / W, nRows = height / W;
        const int wCell = ceil(width / nCols), hCell = ceil(height / nRows);
        for (int i = 0; i < nRows; i++) {
            const float iniY = minBorderY + i * hCell;
            float maxY = iniY + hCell + 6;
            if (iniY >= maxBorderY - 3) continue;
            if (maxY > maxBorderY) maxY = maxBorderY;
            for (int j = 0; j < nCols; j++) {
                const float iniX = minBorderX + j * wCell;
                float maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 6) continue;
                if (maxX > maxBorderX) maxX = maxBorderX;
                std::vector<KeyPoint> cell;
                fast_from_S(S[level], (int)iniX, (int)iniY, (int)maxX, (int)maxY, e.iniThFAST, cell);
                if (cell.empty()) fast_from_S(S[level], (int)iniX, (int)iniY, (int)maxX, (int)maxY, e.minThFAST, cell);
                for (auto& k : cell) { k.pt.x += j * wCell; k.pt.y += i * hCell; dk.push_back(k); }
            }
        }
        all[level] = e.DistributeOctTree(dk, minBorderX, maxBorderX, minBorderY, maxBorderY, e.mnFeaturesPerLevel[level], level);
        const int sps = PATCH_SIZE * e.mvScaleFactor[level];
        for (auto& k : all[level]) {
            k.pt.x += minBorderX; k.pt.y += minBorderY; k.octave = level; k.size = sps;
            int x = cvRound(k.pt.x), y = cvRound(k.pt.y);
            k.angle = M10.empty() ? IC_Angle(e.mvImagePyramid[level], k.pt, e.umax)  // 방향은 CPU(원본 그대로)
                                  : fastAtan2(M01[level].at<float>(y, x), M10[level].at<float>(y, x));
        }
    }
    int n = 0;
    for (auto& v : all) n += v.size();
    outk.clear(); outd.create(n, 32, CV_8U);
    int off = 0;
    for (int level = 0; level < e.nlevels; ++level) {
        auto& kps = all[level];
        if (kps.empty()) continue;
        Mat desc(kps.size(), 32, CV_8U);
        computeDescriptors(B[level], kps, desc, e.pattern);
        for (size_t i = 0; i < kps.size(); ++i) {
            KeyPoint k = kps[i];
            if (level) k.pt *= e.mvScaleFactor[level];
            outk.push_back(k);
            desc.row(i).copyTo(outd.row(off++));
        }
    }
}

// 같은 원본 계산으로 만든 지도(검산용: 이 경로가 원본과 비트 같아야 함)
static void cpu_maps(ORBextractor& e, std::vector<Mat>& S, std::vector<Mat>& B, std::vector<Mat>& M10, std::vector<Mat>& M01) {
    S.clear(); B.clear(); M10.clear(); M01.clear();
    for (int l = 0; l < e.nlevels; ++l) {
        const Mat& lv = e.mvImagePyramid[l];
        S.push_back(fast_score_cpu(lv));
        Mat w = lv.clone();
        GaussianBlur(w, w, Size(7, 7), 2, 2, BORDER_REFLECT_101);
        B.push_back(w);
        Mat m10 = Mat::zeros(lv.size(), CV_32F), m01 = Mat::zeros(lv.size(), CV_32F);
        for (int y = HALF_PATCH_SIZE; y < lv.rows - HALF_PATCH_SIZE; ++y)
            for (int x = HALF_PATCH_SIZE; x < lv.cols - HALF_PATCH_SIZE; ++x) {
                int a = 0, b = 0;
                for (int v = -HALF_PATCH_SIZE; v <= HALF_PATCH_SIZE; ++v) {
                    int d = e.umax[std::abs(v)];
                    for (int u = -d; u <= d; ++u) { int p = lv.at<uchar>(y + v, x + u); a += u * p; b += v * p; }
                }
                m10.at<float>(y, x) = a; m01.at<float>(y, x) = b;
            }
        M10.push_back(m10); M01.push_back(m01);
    }
}

int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "쓰기: time|dump|mosaic|maps <seq> <assoc> <n> ...\n"; return 1; }
    std::string mode = argv[1];
    auto files = read_assoc(argv[2], argv[3], atoi(argv[4]));
    ORBextractor e(1000, 1.2f, 8, 20, 7);  // TUM1.yaml
    if (mode == "time") {
        Acc a;
        for (int rep = 0; rep < 2; ++rep) {
            if (rep == 1) a = Acc();  // 첫 바퀴는 데우기
            for (auto& f : files) { Mat g = load_gray(f); time_one(e, g, a); }
        }
        double n = files.size();
        printf("장면 %d, 점 평균 %.0f\n", (int)n, a.kps / n);
        printf("피라미드 %.3f  FAST(칸) %.3f  나무분배 %.3f  방향 %.3f  블러 %.3f  기술자 %.3f  합 %.3f ms/장\n",
               a.pyr / n, a.fast / n, a.dist / n, a.orient / n, a.blur / n, a.desc / n, a.total / n);
        return 0;
    }
    if (mode == "dump") {
        std::ofstream o(argv[5], std::ios::binary);
        std::vector<int> lap = {0, 0};
        for (auto& f : files) {
            Mat g = load_gray(f), d;
            std::vector<KeyPoint> k;
            e(g, Mat(), k, d, lap);
            write_frame(o, k, d);
        }
        return 0;
    }
    if (mode == "mosaic") {
        std::string out = argv[5];
        Mat g0 = load_gray(files[0]);
        int H;
        auto s = layout(e, g0.cols, g0.rows, H);
        std::ofstream lo(out + "/layout.txt");
        lo << MOS_W << " " << H << " " << e.nlevels << "\n";
        for (auto& q : s) lo << q.x << " " << q.y << " " << q.w << " " << q.h << "\n";
        for (size_t i = 0; i < files.size(); ++i) {
            Mat g = load_gray(files[i]);
            e.ComputePyramid(g);
            Mat m = make_mosaic(e, s, H);
            char p[512]; snprintf(p, sizeof p, "%s/f%04zu.raw", out.c_str(), i);
            std::ofstream r(p, std::ios::binary);
            r.write((char*)m.data, m.total());
        }
        return 0;
    }
    if (mode == "maps") {
        // START 환경 변수: 그 장면부터(지도 파일 번호도 그만큼), 결과는 이어 쓰기 — 큰 지도를 조금씩 만들고 지우려고
        std::string md = argv[5];
        size_t start = getenv("START") ? atoi(getenv("START")) : 0;
        std::ofstream o(argv[6], std::ios::binary | (start ? std::ios::app : std::ios::trunc));
        bool cpuS = argc > 7 && std::string(argv[7]) == "cpuS";
        Acc tm;
        for (size_t i = start; i < files.size(); ++i) {
            Mat g = load_gray(files[i]);
            e.ComputePyramid(g);
            std::vector<Mat> S, B, M10, M01;
            if (cpuS) cpu_maps(e, S, B, M10, M01);
            else {
                // ONNX/NPU 출력: m%04d.bin = [2][H][W] f32 (S, 블러) 모자이크 좌표. 방향은 CPU
                int H;
                auto s = layout(e, g.cols, g.rows, H);
                char p[512]; snprintf(p, sizeof p, "%s/m%04zu.bin", md.c_str(), i);
                std::ifstream r(p, std::ios::binary);
                if (!r) { std::cerr << p << " 없음\n"; return 1; }
                std::vector<float> buf((size_t)2 * H * MOS_W);
                r.read((char*)buf.data(), buf.size() * 4);
                for (int l = 0; l < e.nlevels; ++l) {
                    Rect roi(s[l].x + EDGE_THRESHOLD, s[l].y + EDGE_THRESHOLD, s[l].w - 2 * EDGE_THRESHOLD, s[l].h - 2 * EDGE_THRESHOLD);
                    Mat ch[2];
                    for (int c = 0; c < 2; ++c) ch[c] = Mat(H, MOS_W, CV_32F, buf.data() + (size_t)c * H * MOS_W)(roi).clone();
                    Mat s8, b8;
                    ch[0].convertTo(s8, CV_8U);  // 반올림 · 0–255 포화
                    ch[1].convertTo(b8, CV_8U);
                    S.push_back(s8); B.push_back(b8);
                }
            }
            std::vector<KeyPoint> k; Mat d;
            auto t0 = Clock::now();
            extract_from_maps(e, S, B, M10, M01, k, d);
            tm.total += ms(t0, Clock::now());
            write_frame(o, k, d);
        }
        fprintf(stderr, "지도 → 점 · 기술자(CPU 나머지) %.3f ms/장\n", tm.total / std::max<size_t>(1, files.size() - start));
        return 0;
    }
    std::cerr << "모르는 방식 " << mode << "\n";
    return 1;
}
