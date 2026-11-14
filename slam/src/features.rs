//! 한 장면 → 특징(특징점 · 몸체 축 3D 점 · BRIEF 32 바이트). F2F 위치 추적이 쓰는 RegistrationVis 경로 그대로.
//! 원본(RTAB-Map 0.24.2): RegistrationVis.cpp computeTransformationImpl(깊이를 가리개로 generateKeypoints → generateDescriptors → generateKeypoints3D),
//!   Features2d.cpp Feature2D::generateKeypoints · GFTT · GFTT_BRIEF, util3d_features.cpp generateKeypoints3DDepth, util2d.cpp getDepth, util3d.cpp projectDepthTo3D.
//! 원본(OpenCV 4.10): featureselect.cpp goodFeaturesToTrack, corner.cpp cornerMinEigenVal(Sobel · boxFilter · calcMinEigenVal),
//!   filter.simd.hpp RowVec_8u32f · SymmColumnSmallVec_32f, box_filter.simd.hpp RowSum · ColumnSum, contrib xfeatures2d brief.cpp · generated_32.i.
//!
//! 결과는 원본과 비트까지 같게 한다(점 순서·좌표·response·3D 점·설명자). 그래서 실수 계산 순서·자료형을 원본 그대로 따른다:
//! - Sobel: OpenCV 가 쓰는 AVX2(x86)·NEON(aarch64) 경로는 둘 다 곱셈·덧셈을 한 번에 반올림(fma)한다. 여기서도 mul_add.
//! - 3×3 상자 합: 가로 합·세로 누적 합 모두 f64, 세로는 맨 위 줄부터 "더하고 빼는" 누적이라 줄마다 나눠 돌리면 비트가 달라진다
//!   → 여러 코어는 세로 띠(열 묶음)로 나눈다.
//! - 최소 고유값: x86 은 AVX(곱·덧셈 따로), aarch64 는 NEON v_muladd(fma). 대상 구조에 맞춰 고른다.
use crate::params::Params;
use crate::{Features, Frame, KeyPoint};
use nalgebra::Point3;
use rayon::prelude::*;

/// BRIEF 32 바이트 비교 쌍(y1, x1, y2, x2). OpenCV contrib xfeatures2d/src/generated_32.i 를 그대로 옮김.
/// 바이트 i 의 비트 7..0 = 쌍 8i..8i+7 이고, 비트는 "합(y1,x1) < 합(y2,x2)" 일 때 1
#[rustfmt::skip]
const BRIEF32: [[i8; 4]; 256] = [
    [-2, -1, 7, -1], [-14, -1, -3, 3], [1, -2, 11, 2], [1, 6, -10, -7],
    [13, 2, -1, 0], [-14, 5, 5, -3], [-2, 8, 2, 4], [-11, 8, -15, 5],
    [-6, -23, 8, -9], [-12, 6, -10, 8], [-3, -1, 8, 1], [3, 6, 5, 6],
    [-7, -6, 5, -5], [22, -2, -11, -8], [14, 7, 8, 5], [-1, 14, -5, -14],
    [-14, 9, 2, 0], [7, -3, 22, 6], [-6, 6, -8, -5], [-5, 9, 7, -1],
    [-3, -7, -10, -18], [4, -5, 0, 11], [2, 3, 9, 10], [-10, 3, 4, 9],
    [0, 12, -3, 19], [1, 15, -11, -5], [14, -1, 7, 8], [7, -23, -5, 5],
    [0, -6, -10, 17], [13, -4, -3, -4], [-12, 1, -12, 2], [0, 8, 3, 22],
    [-13, 13, 3, -1], [-16, 17, 6, 10], [7, 15, -5, 0], [2, -12, 19, -2],
    [3, -6, -4, -15], [8, 3, 0, 14], [4, -11, 5, 5], [11, -7, 7, 1],
    [6, 12, 21, 3], [-3, 2, 14, 1], [5, 1, -5, 11], [3, -17, -6, 2],
    [6, 8, 5, -10], [-14, -2, 0, 4], [5, -7, -6, 5], [10, 4, 4, -7],
    [22, 0, 7, -18], [-1, -3, 0, 18], [-4, 22, -5, 3], [1, -7, 2, -3],
    [19, -20, 17, -2], [3, -10, -8, 24], [-5, -14, 7, 5], [-2, 12, -4, -15],
    [4, 12, 0, -19], [20, 13, 3, 5], [-8, -12, 5, 0], [-5, 6, -7, -11],
    [6, -11, -3, -22], [15, 4, 10, 1], [-7, -4, 15, -6], [5, 10, 0, 24],
    [3, 6, 22, -2], [-13, 14, 4, -4], [-13, 8, -18, -22], [-1, -1, -7, 3],
    [-19, -12, 4, 3], [8, 10, 13, -2], [-6, -1, -6, -5], [2, -21, -3, 2],
    [4, -7, 0, 16], [-6, -5, -12, -1], [1, -1, 9, 18], [-7, 10, -11, 6],
    [4, 3, 19, -7], [-18, 5, -4, 5], [4, 0, -20, 4], [7, -11, 18, 12],
    [-20, 17, -18, 7], [2, 15, 19, -11], [-18, 6, -7, 3], [-4, 1, -14, 13],
    [17, 3, 2, -8], [-7, 2, 1, 6], [17, -9, -2, 8], [-8, -6, -1, 12],
    [-2, 4, -1, 6], [-2, 7, 6, 8], [-8, -1, -7, -9], [8, -9, 15, 0],
    [0, 22, -4, -15], [-14, -1, 3, -2], [-7, -4, 17, -7], [-8, -2, 9, -4],
    [5, -7, 7, 7], [-5, 13, -8, 11], [11, -4, 0, 8], [5, -11, -9, -6],
    [2, -6, 3, -20], [-6, 2, 6, 10], [-6, -6, -15, 7], [-6, -3, 2, 1],
    [11, 0, -3, 2], [7, -12, 14, 5], [0, -7, -1, -1], [-16, 0, 6, 8],
    [22, 11, 0, -3], [19, 0, 5, -17], [-23, -14, -13, -19], [-8, 10, -11, -2],
    [-11, 6, -10, 13], [1, -7, 14, 0], [-12, 1, -5, -5], [4, 7, 8, -1],
    [-1, -5, 15, 2], [-3, -1, 7, -10], [3, -6, 10, -18], [-7, -13, -13, 10],
    [1, -1, 13, -10], [-19, 14, 8, -14], [-4, -13, 7, 1], [1, -2, 12, -7],
    [3, -5, 1, -5], [-2, -2, 8, -10], [2, 14, 8, 7], [3, 9, 8, 2],
    [-9, 1, -18, 0], [4, 0, 1, 12], [0, 9, -14, -10], [-13, -9, -2, 6],
    [1, 5, 10, 10], [-3, -6, -16, -5], [11, 6, -5, 0], [-23, 10, 1, 2],
    [13, -5, -3, 9], [-4, -1, -13, -5], [10, 13, -11, 8], [19, 20, -9, 2],
    [4, -8, 0, -9], [-14, 10, 15, 19], [-14, -12, -10, -3], [-23, -3, 17, -2],
    [-3, -11, 6, -14], [19, -2, -4, 2], [-5, 5, 3, -13], [2, -2, -5, 4],
    [17, 4, 17, -11], [-7, -2, 1, 23], [8, 13, 1, -16], [-13, -5, 1, -17],
    [4, 6, -8, -3], [-5, -9, -2, -10], [-9, 0, -7, -2], [5, 0, 5, 2],
    [-4, -16, 6, 3], [2, -15, -2, 12], [4, -1, 6, 2], [1, 1, -2, -8],
    [-2, 12, -5, -2], [-8, 8, -9, 9], [2, -10, 3, 1], [-4, 10, -9, 4],
    [6, 12, 2, 5], [-3, -8, 0, 5], [-13, 1, -7, 2], [-1, -10, 7, -18],
    [-1, 8, -9, -10], [-23, -1, 6, 2], [-5, -3, 3, 2], [0, 11, -4, -7],
    [15, 2, -10, -3], [-20, -8, -13, 3], [-19, -12, 5, -11], [-17, -13, -3, 2],
    [7, 4, -12, 0], [5, -1, -14, -6], [-4, 11, 0, -4], [3, 10, 7, -3],
    [13, 21, -11, 6], [-12, 24, -7, -4], [4, 16, 3, -14], [-3, 5, -7, -12],
    [0, -4, 7, -5], [-17, -9, 13, -7], [22, -6, -11, 5], [2, -8, 23, -11],
    [7, -10, -1, 14], [-3, -10, 8, 3], [-13, 1, -6, 0], [-7, -21, 6, -14],
    [18, 19, -4, -6], [10, 7, -1, -4], [-1, 21, 1, -5], [-10, 6, -11, -2],
    [18, -3, -1, 7], [-3, -9, -5, 10], [-13, 14, 17, -3], [11, -19, -1, -18],
    [8, -2, -18, -23], [0, -5, -2, -9], [-4, -11, 2, -8], [14, 6, -3, -6],
    [-3, 0, -15, 0], [-9, 4, -15, -9], [-1, 11, 3, 11], [-10, -16, -7, 7],
    [-2, -10, -10, -2], [-5, -3, 5, -23], [13, -8, -15, -11], [-15, 11, 6, -6],
    [-16, -3, -2, 2], [6, 12, -16, 24], [-10, 0, 8, 11], [-7, 7, -19, -7],
    [5, 16, 9, -3], [9, 7, -7, -16], [3, 2, -10, 9], [21, 1, 8, 7],
    [7, 0, 1, 17], [-8, 12, 9, 6], [11, -7, -8, -6], [19, 0, 9, 3],
    [1, -7, -5, -11], [0, 8, -2, 14], [12, -2, -15, -6], [4, 12, 0, -21],
    [17, -4, -6, -7], [-10, -9, -14, -7], [-15, -10, -15, -14], [-7, -5, 5, -12],
    [-4, 0, 15, -4], [5, 2, -6, -23], [-4, -21, -6, 4], [-10, 5, -15, 6],
    [4, -3, -1, 5], [-4, 19, -23, -4], [-4, 17, 13, -11], [1, 12, 4, -14],
    [-11, -6, -20, 10], [4, 5, 3, 20], [-8, -20, 3, 1], [-19, 9, 9, -3],
    [18, 15, 11, -4], [12, 16, 8, 7], [-14, -8, -3, 9], [-6, 0, 2, -4],
    [1, -10, -1, 2], [8, -7, -6, 18], [9, 12, -7, -23], [8, -6, 5, 2],
    [-9, 6, -12, -7], [-1, -2, -7, 2], [9, 9, 7, 15], [6, 2, -6, 6],
];

/// 특징 추출기: 위치 추적(RegistrationVis)이 Vis/* 로 만든 GFTT/BRIEF
#[derive(Clone, Debug)]
pub struct Extractor {
    /// Vis/MaxFeatures(→ Kp/MaxFeatures, GFTT maxCorners). 0 = 제한 없음
    pub max_features: i32,
    pub quality_level: f64,
    pub min_distance: f64,
    /// Vis/MinDepth · Vis/MaxDepth(미터, 0 = 제한 없음)
    pub min_depth: f32,
    pub max_depth: f32,
    /// Vis/DepthAsMask: 깊이가 있는 화소에서만 특징점을 찾음
    pub depth_as_mask: bool,
    /// 상자 합을 f32 로(원본은 f64). 약 25 % 빠르지만 response 아래 자리가 달라져 점 순서가 바뀜. 기본 끔, 설정 이름 Slam/FeatureBoxF32
    pub box_f32: bool,
}

impl Extractor {
    /// 기본값은 RTAB-Map 0.24.2 Parameters.h. RegistrationVis 가 Vis/* 를 Kp/* 로 옮겨 Feature2D 를 만든다(RegistrationVis.cpp 157-171, 306-349)
    pub fn new(p: &Params) -> Self {
        let need = |ok: bool, what: &str| {
            assert!(ok, "features: 옮기지 않은 설정 {what} (GFTT/BRIEF·격자 1×1·ROI 없음·부화소 끔만 지원)");
        };
        need(p.get("Vis/FeatureType", 6) == 6, "Vis/FeatureType");
        need(p.get("BRIEF/Bytes", 32) == 32, "BRIEF/Bytes");
        need(p.get("GFTT/BlockSize", 3) == 3, "GFTT/BlockSize");
        need(!p.get_bool("GFTT/UseHarrisDetector", false), "GFTT/UseHarrisDetector");
        need(p.get("Vis/GridRows", 1) == 1 && p.get("Vis/GridCols", 1) == 1, "Vis/GridRows·GridCols");
        need(p.get("Vis/SubPixIterations", 0) == 0, "Vis/SubPixIterations");
        need(!p.get_bool("Vis/SSC", false), "Vis/SSC");
        need(p.get("Vis/DepthMaskFloorThr", 0.0f32) == 0.0, "Vis/DepthMaskFloorThr");
        let roi = p.0.get("Vis/RoiRatios").map(|s| s.as_str()).unwrap_or("0.0 0.0 0.0 0.0");
        need(roi.split_whitespace().all(|v| v.parse::<f32>().map(|x| x == 0.0).unwrap_or(false)), "Vis/RoiRatios");
        Extractor {
            max_features: p.get("Vis/MaxFeatures", 1000),
            quality_level: p.get("GFTT/QualityLevel", 0.001),
            min_distance: p.get("GFTT/MinDistance", 7.0),
            min_depth: p.get("Vis/MinDepth", 0.0),
            max_depth: p.get("Vis/MaxDepth", 0.0),
            depth_as_mask: p.get_bool("Vis/DepthAsMask", true),
            box_f32: p.get_bool("Slam/FeatureBoxF32", false),
        }
    }

    /// 특징점 → 설명자(가장자리 점은 빠짐) → 3D 점. RegistrationVis 와 같은 순서
    pub fn extract(&self, f: &Frame) -> Features {
        let mut kps = self.detect(f);
        let desc = describe(&f.gray, f.cam.width as usize, f.cam.height as usize, &mut kps);
        let pts3d = self.keypoints_3d(f, &kps);
        Features { kps, pts3d, desc }
    }

    /// Feature2D::generateKeypoints(image, depthMask): 깊이 → 가리개, GFTT(goodFeaturesToTrack)
    pub fn detect(&self, f: &Frame) -> Vec<KeyPoint> {
        let (w, h) = (f.cam.width as usize, f.cam.height as usize);
        assert_eq!(f.gray.len(), w * h);
        if self.depth_as_mask && !f.depth_mm.is_empty() {
            assert_eq!(f.depth_mm.len(), w * h);
            with_buf(&MASK_BUF, |m| {
                depth_mask(&f.depth_mm, self.min_depth, self.max_depth, m);
                gftt_opt(&f.gray, w, h, Some(m), self.max_features, self.quality_level, self.min_distance, self.box_f32)
            })
        } else {
            gftt_opt(&f.gray, w, h, None, self.max_features, self.quality_level, self.min_distance, self.box_f32)
        }
    }

    /// Feature2D::generateKeypoints3D → util3d::generateKeypoints3DDepth. 깊이 없으면 NaN, 있으면 몸체 축
    pub fn keypoints_3d(&self, f: &Frame, kps: &[KeyPoint]) -> Vec<Point3<f32>> {
        keypoints_3d(&f.depth_mm, &f.cam, kps, self.min_depth, self.max_depth)
    }
}

/// 깊이(mm) → 8 비트 가리개(Feature2D::generateKeypoints 앞부분): 0 < 값 < 65535, 미터로 min < d ≤ max
fn depth_mask(depth: &[u16], min_depth: f32, max_depth: f32, out: &mut Vec<u8>) {
    out.clear();
    out.extend(depth.iter().map(|&d| {
        let v = if d > 0 && d < u16::MAX { d as f32 * 0.001 } else { 0.0 };
        if v > min_depth && (max_depth == 0.0 || v <= max_depth) && v.is_finite() { 255 } else { 0 }
    }));
}

thread_local! {
    /// 깊이 가리개 버퍼(스레드마다 하나, 장면마다 다시 씀)
    static MASK_BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// 스레드 버퍼를 빌려 씀. rayon 이 기다리는 동안 같은 스레드에서 다른 장면을 처리해 이미 빌려 갔으면 새로 만듦
fn with_buf<T, R>(key: &'static std::thread::LocalKey<std::cell::RefCell<Vec<T>>>, f: impl FnOnce(&mut Vec<T>) -> R) -> R {
    let mut f = Some(f);
    let r = key.with(|c| c.try_borrow_mut().ok().map(|mut b| (f.take().unwrap())(&mut b)));
    match r {
        Some(r) => r,
        None => (f.take().unwrap())(&mut Vec::new()),
    }
}

/// OpenCV BORDER_REFLECT_101: -1 → 1, n → n-2
#[inline]
fn refl(p: isize, n: usize) -> usize {
    let n = n as isize;
    (if p < 0 { -p } else if p >= n { 2 * n - 2 - p } else { p }) as usize
}

/// 최소 고유값 그림(cornerMinEigenVal, blockSize 3, Sobel 3)과 가리개 안 최댓값(minMaxLoc, 화소가 없으면 0).
/// 세로 띠로 나눠 여러 코어에서(띠마다 결과 비트는 같음: 상자 세로 누적이 열마다 따로라서)
pub fn min_eigen(gray: &[u8], w: usize, h: usize, mask: Option<&[u8]>) -> (Vec<f32>, f32) {
    let mut eig = Vec::new();
    let mx = min_eigen_into(gray, w, h, mask, &mut eig);
    (eig, mx)
}

/// min_eigen 을 이미 있는 버퍼에(장면마다 1.2 MB 를 새로 받으면 쪽 나눔 비용이 계산만큼 큼)
pub fn min_eigen_into(gray: &[u8], w: usize, h: usize, mask: Option<&[u8]>, eig: &mut Vec<f32>) -> f32 {
    eig_pass(gray, w, h, mask, eig, false, false, 0.0).0
}

/// 띠마다 고유값 계산 + (원하면) 띠 안쪽 열의 3×3 극대점(0 보다 크고 가리개 안) 화소 번호.
/// 반환: (가리개 안 최댓값, 띠 안쪽 극대점, 띠 경계 열 목록)
fn eig_pass(gray: &[u8], w: usize, h: usize, mask: Option<&[u8]>, eig: &mut Vec<f32>, peaks: bool, box_f32: bool, quality: f64) -> (f32, Vec<u32>, Vec<usize>) {
    assert!(w >= 3 && h >= 3 && gray.len() == w * h);
    // 모든 칸을 띠들이 덮어쓰므로 0 으로 채울 필요 없음(크기만 맞춤)
    eig.resize(w * h, 0.0);
    // 띠 하나 최소 64 열(띠마다 좌우 2 열을 더 계산)
    let strips = rayon::current_num_threads().min(w / 64).max(1);
    let out = SendPtr(eig.as_mut_ptr());
    let res: Vec<(Option<f32>, Vec<u32>)> = (0..strips)
        .into_par_iter()
        .map(|s| {
            let (xa, xb) = (w * s / strips, w * (s + 1) / strips);
            // SAFETY: 띠마다 열 [xa, xb) 만 쓰고, 읽는 것도 자기 열뿐이라 서로 겹치지 않음
            let out = unsafe { std::slice::from_raw_parts_mut(out.get(), w * h) };
            eig_strip(gray, w, h, xa, xb, out, mask, peaks, box_f32, quality)
        })
        .collect();
    // 띠 경계 열(왼쪽 띠의 마지막 열, 오른쪽 띠의 첫 열)은 이웃 띠 값이 필요해 나중에 따로 봄
    let mut edges = Vec::new();
    for s in 1..strips {
        let x = w * s / strips;
        edges.extend([x - 1, x]);
    }
    let mx = res.iter().filter_map(|r| r.0).reduce(f32::max).unwrap_or(0.0);
    let pk = res.into_iter().flat_map(|r| r.1).collect();
    (mx, pk, edges)
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[allow(clippy::too_many_arguments)]
fn eig_strip(gray: &[u8], w: usize, h: usize, xa: usize, xb: usize, out: &mut [f32], mask: Option<&[u8]>, peaks: bool, box_f32: bool, quality: f64) -> (Option<f32>, Vec<u32>) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx2") {
            // SAFETY: 위에서 CPU 가 fma·avx2 를 지원함을 확인
            return unsafe {
                if box_f32 { eig_strip_fma::<f32>(gray, w, h, xa, xb, out, mask, peaks, quality) } else { eig_strip_fma::<f64>(gray, w, h, xa, xb, out, mask, peaks, quality) }
            };
        }
    }
    if box_f32 { eig_strip_impl::<f32>(gray, w, h, xa, xb, out, mask, peaks, quality) } else { eig_strip_impl::<f64>(gray, w, h, xa, xb, out, mask, peaks, quality) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma,avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn eig_strip_fma<A: BoxAcc>(gray: &[u8], w: usize, h: usize, xa: usize, xb: usize, out: &mut [f32], mask: Option<&[u8]>, peaks: bool, quality: f64) -> (Option<f32>, Vec<u32>) {
    eig_strip_impl::<A>(gray, w, h, xa, xb, out, mask, peaks, quality)
}

/// 상자 합 누적 자료형. 원본(boxFilter 32F)은 f64. f32 는 비트가 달라지는 빠른 선택(Extractor::box_f32)
trait BoxAcc: Copy + Default + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> {
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
}
impl BoxAcc for f64 {
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v as f64
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl BoxAcc for f32 {
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

/// Sobel 배율 s = 1/(4·3·255) 를 float 로(corner.cpp cornerEigenValsVecs, Mat *= scale)
const SOBEL_S: f32 = (1.0f64 / 3060.0) as f32;

/// 회색조 한 줄 → 가로 거른 두 줄(RowVec_8u32f: s = fma(x_k, k_k, s), k = 0,1,2)
/// d = [-1 0 1] (정수라 r - l 과 비트가 같음), m = [s 2s s]
#[inline(always)]
fn row_filter(row: &[u8], idx: &ColIdx, d: &mut [f32], m: &mut [f32]) {
    let (s, s2) = (SOBEL_S, SOBEL_S * 2.0);
    let n = d.len();
    let (i0, i1) = idx.run;
    // 그림 가장자리(반사) 칸만 하나씩, 가운데는 이어진 구간으로
    for i in (0..i0).chain(i1..n) {
        let (l, c, r) = (row[idx.l[i]] as f32, row[idx.c[i]] as f32, row[idx.r[i]] as f32);
        d[i] = r - l;
        m[i] = r.mul_add(s, c.mul_add(s2, l * s));
    }
    if i0 < i1 {
        let x0 = idx.l[i0];
        let k = i1 - i0;
        let (l, c, r) = (&row[x0..x0 + k], &row[x0 + 1..x0 + k + 1], &row[x0 + 2..x0 + k + 2]);
        for ((((d, m), &l), &c), &r) in d[i0..i1].iter_mut().zip(m[i0..i1].iter_mut()).zip(l).zip(c).zip(r) {
            let (l, c, r) = (l as f32, c as f32, r as f32);
            *d = r - l;
            *m = r.mul_add(s, c.mul_add(s2, l * s));
        }
    }
}

/// cov 열마다 쓰는 회색조 열. run = 반사 없이 이어진 칸 구간 [i0, i1)
struct ColIdx {
    l: Vec<usize>,
    c: Vec<usize>,
    r: Vec<usize>,
    run: (usize, usize),
}

/// 세 줄의 가로 거른 값 → Dx·Dy(세로 거르기) → cov → 상자 가로 합(세 채널 한 번에, ((a + b) + c))
/// Dx 세로 = fma(위+아래, s, fma(가운데, 2s, 0)) (SymmColumnSmallVec_32f 대칭·일반 계수), Dy 세로 = (아래 - 위) + 0 (반대칭 ±1)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cov_row_sum<A: BoxAcc>(du: &[f32], dm: &[f32], dd: &[f32], su: &[f32], sd: &[f32], cov: &mut [[f32; 3]], ra: &mut [A], rb: &mut [A], rc: &mut [A]) {
    let (s, s2) = (SOBEL_S, SOBEL_S * 2.0);
    for (((((c, &du), &dm), &dd), &su), &sd) in cov.iter_mut().zip(du).zip(dm).zip(dd).zip(su).zip(sd) {
        let dx = (du + dd).mul_add(s, dm.mul_add(s2, 0.0));
        let dy = (sd - su) + 0.0;
        *c = [dx * dx, dx * dy, dy * dy];
    }
    // 세 채널을 한 번에 돌되 결과는 채널마다 따로 둠(세로 누적이 채널별 연속 배열이라 SIMD 로 잘 바뀜)
    let cw = ra.len();
    let (c0, c1, c2) = (&cov[..cw], &cov[1..cw + 1], &cov[2..cw + 2]);
    for (((((a, b), c), p), q), r) in ra.iter_mut().zip(rb.iter_mut()).zip(rc.iter_mut()).zip(c0).zip(c1).zip(c2) {
        *a = A::from_f32(p[0]) + A::from_f32(q[0]) + A::from_f32(r[0]);
        *b = A::from_f32(p[1]) + A::from_f32(q[1]) + A::from_f32(r[1]);
        *c = A::from_f32(p[2]) + A::from_f32(q[2]) + A::from_f32(r[2]);
    }
}

/// 띠 [xa, xb) 의 최소 고유값과 가리개 안 최댓값(+ peaks 면 띠 안쪽 극대점). 원본 순서:
/// 1) Sobel(8U→32F): 가로 거르기 → 세로 거르기. Dx = 가로 [-1 0 1] · 세로 [s 2s s],  Dy = 가로 [s 2s s] · 세로 [-1 0 1]
///    (deriv.cpp Sobel: 미분 아닌 쪽 커널에 배율). 그림 밖은 BORDER_REFLECT_101(회색조 단계)
/// 2) cov = (dx², dx·dy, dy²)  3) 3×3 상자 합(boxFilter 정규화 없음): 가로 RowSum<float,double>, 세로 ColumnSum<double,float>
///    (cov 단계에서 다시 반사, 세로는 SUM 에 더해 내보내고 맨 윗줄을 빼는 누적)
/// 4) calcMinEigenVal: λmin = (a+c) - √((a-c)² + b²), a = cov0·0.5, c = cov2·0.5
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn eig_strip_impl<A: BoxAcc>(gray: &[u8], w: usize, h: usize, xa: usize, xb: usize, out: &mut [f32], mask: Option<&[u8]>, peaks: bool, quality: f64) -> (Option<f32>, Vec<u32>) {
    let cw = xb - xa;
    let n = cw + 2; // cov 열: xa-1 ..= xb
    let c: Vec<usize> = (0..n).map(|i| refl(xa as isize + i as isize - 1, w)).collect();
    let l: Vec<usize> = c.iter().map(|&c| refl(c as isize - 1, w)).collect();
    let r: Vec<usize> = c.iter().map(|&c| refl(c as isize + 1, w)).collect();
    // 반사 없는 칸: c = xa-1+i 이고 양옆이 그 ±1. 가장자리 띠에서도 처음·끝 1~2 칸만 빠짐
    let plain = |i: usize| xa + i >= 2 && xa + i < w && c[i] + 1 == xa + i && l[i] + 1 == c[i] && c[i] + 1 == r[i];
    let i0 = (0..n).find(|&i| plain(i)).unwrap_or(n);
    let i1 = (i0..n).find(|&i| !plain(i)).unwrap_or(n);
    let idx = ColIdx { l, c, r, run: (i0, i1) };

    // 가로 거른 줄(줄 번호 % 3 칸), 상자 가로 합(줄 번호 % 3 칸, 채널 셋을 한 칸에)
    let mut dxr = vec![0f32; 3 * n];
    let mut smr = vec![0f32; 3 * n];
    let mut ra = vec![A::default(); 3 * cw];
    let mut rb = vec![A::default(); 3 * cw];
    let mut rc = vec![A::default(); 3 * cw];
    let mut cov = vec![[0f32; 3]; n];
    // 클로저 대신 매크로: 클로저는 target_feature(fma) 를 물려받지 않아 mul_add 가 느린 함수 호출이 됨
    macro_rules! filt {
        ($g:expr) => {{
            let g: usize = $g;
            let k = g % 3 * n;
            row_filter(&gray[g * w..(g + 1) * w], &idx, &mut dxr[k..k + n], &mut smr[k..k + n]);
        }};
    }
    // R(r): 회색조 줄 r-1, r, r+1(반사) 의 가로 거른 값이 이미 칸에 있어야 함
    macro_rules! rsum {
        ($r:expr) => {{
            let r: usize = $r;
            let (up, mid, dn) = (refl(r as isize - 1, h) % 3 * n, r % 3 * n, refl(r as isize + 1, h) % 3 * n);
            let k = r % 3 * cw;
            cov_row_sum::<A>(&dxr[up..up + n], &dxr[mid..mid + n], &dxr[dn..dn + n], &smr[up..up + n], &smr[dn..dn + n], &mut cov, &mut ra[k..k + cw], &mut rb[k..k + cw], &mut rc[k..k + cw]);
        }};
    }

    filt!(0);
    filt!(1);
    filt!(2);
    rsum!(0);
    rsum!(1);
    // ColumnSum 시작: SUM = 0 + R(-1) + R(0), R(-1) = R(1)
    let mut sa: Vec<A> = (0..cw).map(|j| (A::default() + ra[cw + j]) + ra[j]).collect();
    let mut sb: Vec<A> = (0..cw).map(|j| (A::default() + rb[cw + j]) + rb[j]).collect();
    let mut sc: Vec<A> = (0..cw).map(|j| (A::default() + rc[cw + j]) + rc[j]).collect();
    // 극대점 검사 열: 띠 안에서 양옆이 있는 열, 그림 안쪽 1..w-1
    let (pa, pb) = ((xa + 1).max(1), (xb - 1).min(w - 1));
    let mut pk = Vec::new();
    let pw = pb.saturating_sub(pa);
    let mut hm = vec![0f32; 3 * pw];
    let mut flag = vec![0u8; pw];
    let mut lb = 0f32; // 극대점 아래 경계(0 보다 커야 함)
    let mut cmax = vec![f32::NEG_INFINITY; cw];
    for y in 0..h {
        if y >= 1 && y + 1 < h {
            if y + 2 < h {
                filt!(y + 2);
            }
            rsum!(y + 1);
        }
        // 내보냄 = SUM + R(y+1), 그다음 SUM = 그 값 - R(y-1)
        let p = refl(y as isize + 1, h) % 3 * cw;
        let m = refl(y as isize - 1, h) % 3 * cw;
        let o = &mut out[y * w + xa..y * w + xb];
        let (pa_, pb_, pc_) = (&ra[p..p + cw], &rb[p..p + cw], &rc[p..p + cw]);
        let (ma, mb, mc) = (&ra[m..m + cw], &rb[m..m + cw], &rc[m..m + cw]);
        for (((((((((o, sa), sb), sc), &pa_), &pb_), &pc_), &ma), &mb), &mc) in
            o.iter_mut().zip(sa.iter_mut()).zip(sb.iter_mut()).zip(sc.iter_mut()).zip(pa_).zip(pb_).zip(pc_).zip(ma).zip(mb).zip(mc)
        {
            let (va, vb, vc) = (*sa + pa_, *sb + pb_, *sc + pc_);
            *sa = va - ma;
            *sb = vb - mb;
            *sc = vc - mc;
            let a = va.to_f32() * 0.5;
            let b = vb.to_f32();
            let c = vc.to_f32() * 0.5;
            let t = a - c;
            *o = (a + c) - eig_q(t, b).sqrt();
        }
        // minMaxLoc(가리개): 열마다 누적 최댓값(가리개 밖은 -∞), 띠 끝에서 한 번 줄임. 최댓값은 순서와 무관해 비트가 같음
        match mask {
            Some(mk) => {
                for ((c, &v), &m) in cmax.iter_mut().zip(o.iter()).zip(&mk[y * w + xa..y * w + xb]) {
                    *c = fmax(*c, if m != 0 { v } else { f32::NEG_INFINITY });
                }
            }
            None => {
                for (c, &v) in cmax.iter_mut().zip(o.iter()) {
                    *c = fmax(*c, v);
                }
            }
        }
        // 줄 y 가 나왔으니 줄 y-1 의 극대점(위·아래 줄이 다 있음)
        if peaks && pa < pb {
            // 줄 y 의 가로 3 칸 최댓값(열 pa..pb), 줄 번호 % 3 칸
            let k = y % 3 * pw;
            let row = &out[y * w..(y + 1) * w];
            for (((o, &a), &b), &c) in hm[k..k + pw].iter_mut().zip(&row[pa - 1..pb - 1]).zip(&row[pa..pb]).zip(&row[pa + 1..pb + 1]) {
                *o = fmax(fmax(a, b), c);
            }
            if y >= 2 {
                if y % 16 == 0 {
                    // 지금까지 본 최댓값으로 문턱 아래 경계: 전체 최댓값 이상은 안 되므로 (그 값 × q) 이하는 어차피 문턱 이하
                    lb = lb.max((cmax.iter().fold(f32::NEG_INFINITY, |a, &b| fmax(a, b)) as f64 * quality) as f32);
                }
                find_peaks(out, w, y - 1, pa, pb, mask, &hm, pw, lb, &mut flag, &mut pk);
            }
        }
    }
    let rmax = cmax.iter().fold(f32::NEG_INFINITY, |a, &b| fmax(a, b));
    ((rmax > f32::NEG_INFINITY).then_some(rmax), pk)
}

/// 줄 yy, 열 [xa, xb) 에서 0 보다 크고 3×3 이웃(원래 값) 모두 이상이며 가리개 안인 화소 번호.
/// hm = 줄 번호 % 3 칸의 가로 3 칸 최댓값(열 xa.. 기준, 폭 pw)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn find_peaks(eig: &[f32], w: usize, yy: usize, xa: usize, xb: usize, mask: Option<&[u8]>, hm: &[f32], pw: usize, lb: f32, flag: &mut [u8], pk: &mut Vec<u32>) {
    let r1 = &eig[yy * w + xa..yy * w + xb];
    let (h0, h1, h2) = (&hm[(yy - 1) % 3 * pw..][..pw], &hm[yy % 3 * pw..][..pw], &hm[(yy + 1) % 3 * pw..][..pw]);
    for ((((f, &v), &a), &b), &c) in flag.iter_mut().zip(r1).zip(h0).zip(h1).zip(h2) {
        *f = ((v > lb) & (v >= fmax(fmax(a, b), c))) as u8;
    }
    if let Some(m) = mask {
        for (f, &m) in flag.iter_mut().zip(&m[yy * w + xa..yy * w + xb]) {
            *f &= (m != 0) as u8;
        }
    }
    // 대부분 0 이라 8 칸씩 건너뜀
    for (ci, ch) in flag.chunks(8).enumerate() {
        if u64::from_ne_bytes(std::array::from_fn(|k| ch.get(k).copied().unwrap_or(0))) == 0 {
            continue;
        }
        for (k, &f) in ch.iter().enumerate() {
            if f != 0 {
                pk.push((yy * w + xa + ci * 8 + k) as u32);
            }
        }
    }
}

/// (a-c)² + b²: x86 은 calcMinEigenValLine_AVX(곱·덧셈 따로), aarch64 는 NEON v_muladd(b, b, t·t) = fma
#[inline(always)]
fn eig_q(t: f32, b: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        b.mul_add(b, t * t)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        b * b + t * t
    }
}

/// f32 → 크기 순서가 같은 u32(부호 비트 뒤집기)
#[inline]
fn ord_f32(v: f32) -> u32 {
    let b = v.to_bits();
    if b >> 31 == 1 { !b } else { b | 0x8000_0000 }
}

/// cv::goodFeaturesToTrack(Harris 아님) + GFTTDetector::detect(KeyPoint 크기 = blockSize 3, 각도 -1)
pub fn gftt(gray: &[u8], w: usize, h: usize, mask: Option<&[u8]>, max_corners: i32, quality: f64, min_distance: f64) -> Vec<KeyPoint> {
    gftt_opt(gray, w, h, mask, max_corners, quality, min_distance, false)
}

/// gftt + box_f32(상자 합을 f32 로: 빠르지만 원본과 비트가 다름, 기본 끔)
#[allow(clippy::too_many_arguments)]
pub fn gftt_opt(gray: &[u8], w: usize, h: usize, mask: Option<&[u8]>, max_corners: i32, quality: f64, min_distance: f64, box_f32: bool) -> Vec<KeyPoint> {
    with_buf(&EIG_BUF, |eig| gftt_with(gray, w, h, mask, max_corners, quality, min_distance, eig, box_f32))
}

thread_local! {
    /// 최소 고유값 그림 버퍼(스레드마다 하나, 장면마다 다시 씀)
    static EIG_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[allow(clippy::too_many_arguments)]
fn gftt_with(gray: &[u8], w: usize, h: usize, mask: Option<&[u8]>, max_corners: i32, quality: f64, min_distance: f64, eig: &mut Vec<f32>, box_f32: bool) -> Vec<KeyPoint> {
    // 1 단계: 고유값 + 띠 안쪽 극대점(문턱은 전체 최댓값을 알아야 정해지므로 아직 안 거름)
    let (max_val, peaks, edges) = eig_pass(gray, w, h, mask, eig, true, box_f32, quality);
    let eig: &[f32] = eig;
    // threshold(TOZERO): 문턱 = (float)(최댓값 × qualityLevel), src > 문턱 이면 남김
    let th = (max_val as f64 * quality) as f32;
    // 2 단계 후보: 원본은 "문턱 뒤 값 ≠ 0 이고 문턱 뒤 3×3 dilate 값과 같음 + 가리개".
    // 문턱이 0 이상이면 이는 "원래 값 > 문턱 이고 3×3 이웃 원래 값 모두 이상"과 같다
    // (문턱 이하 이웃은 0 이 되어 그 값보다 작고, 문턱 위 이웃은 값 그대로라서). 그래서 1 단계 극대점을 문턱으로만 거름.
    // 문턱이 음수(최댓값 < 0)면 이 같음이 깨지므로 원본 그대로 다시 계산
    let mut cand: Vec<u64> = if th >= 0.0 {
        let key = |pix: usize| ((ord_f32(eig[pix]) as u64) << 32) | pix as u64;
        let mut c: Vec<u64> = peaks.into_iter().map(|p| p as usize).filter(|&p| eig[p] > th).map(key).collect();
        // 띠 경계 열은 이웃 띠 값으로 여기서 봄
        for &x in &edges {
            if x == 0 || x + 1 >= w {
                continue;
            }
            for y in 1..h - 1 {
                let p = y * w + x;
                let v = eig[p];
                if v > th && mask.is_none_or(|m| m[p] != 0) {
                    let nb = (y - 1..=y + 1).flat_map(|yy| (x - 1..=x + 1).map(move |xx| yy * w + xx));
                    if nb.into_iter().all(|q| eig[q] <= v) {
                        c.push(key(p));
                    }
                }
            }
        }
        c
    } else {
        let bands = rayon::current_num_threads().min((h - 2) / 32).max(1);
        (0..bands)
            .into_par_iter()
            .flat_map_iter(|b| {
                let (ya, yb) = (1 + (h - 2) * b / bands, 1 + (h - 2) * (b + 1) / bands);
                candidates(eig, w, mask, th, ya, yb)
            })
            .collect()
    };
    // greaterThanPtr: 값 내림차순, 같으면 주소(화소 번호) 큰 쪽 먼저 → 열쇠 (값 순서 << 32 | 번호) 의 내림차순.
    // 위쪽 일부만 먼저 정렬하고, 다 쓰고도 모자라면 나머지를 정렬(결과 순서는 전체 정렬과 같음)
    let want = if max_corners > 0 { max_corners as usize } else { usize::MAX };
    let head = cand.len().min(want.saturating_mul(8).max(4096));
    if head < cand.len() {
        cand.select_nth_unstable_by(head, |a, b| b.cmp(a));
    }
    cand[..head].sort_unstable_by(|a, b| b.cmp(a));
    let mut sorted_tail = false;

    let mut kps = Vec::new();
    let cell = min_distance.round_ties_even() as usize; // cvRound
    let use_grid = min_distance >= 1.0;
    let (gw, gh) = if use_grid { (w.div_ceil(cell), h.div_ceil(cell)) } else { (1, 1) };
    let mut grid: Vec<Vec<(f32, f32)>> = vec![Vec::new(); gw * gh];
    let md2 = min_distance * min_distance;
    let mut i = 0;
    while i < cand.len() && kps.len() < want {
        if i == head && !sorted_tail {
            cand[head..].sort_unstable_by(|a, b| b.cmp(a));
            sorted_tail = true;
        }
        let key = cand[i];
        i += 1;
        let pix = (key & 0xFFFF_FFFF) as usize;
        let v = eig[pix];
        let (y, x) = (pix / w, pix % w);
        if use_grid {
            let (xc, yc) = (x / cell, y / cell);
            let (x1, y1) = (xc.saturating_sub(1), yc.saturating_sub(1));
            let (x2, y2) = ((xc + 1).min(gw - 1), (yc + 1).min(gh - 1));
            let good = (y1..=y2).all(|yy| {
                (x1..=x2).all(|xx| {
                    grid[yy * gw + xx].iter().all(|m| {
                        let (dx, dy) = (x as f32 - m.0, y as f32 - m.1);
                        ((dx * dx + dy * dy) as f64) >= md2
                    })
                })
            });
            if !good {
                continue;
            }
            grid[yc * gw + xc].push((x as f32, y as f32));
        }
        kps.push(KeyPoint { x: x as f32, y: y as f32, size: 3.0, angle: -1.0, response: v, octave: 0 });
    }
    kps
}

/// 비교로 고르는 최댓값(NaN 없음). f32::max 보다 SIMD 로 잘 바뀜
#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// 줄 [ya, yb) 의 후보 열쇠. threshold → dilate(3×3, 그림 밖 무시) 를 문턱 줄 → 가로 최댓값 → 세로 최댓값으로 나눠 계산
fn candidates(eig: &[f32], w: usize, mask: Option<&[u8]>, th: f32, ya: usize, yb: usize) -> Vec<u64> {
    // 줄 번호 % 3 칸: t = 문턱 뒤 값, hm = 가로 3 칸 최댓값(x 는 1..w-1 만 씀)
    let mut t = vec![0f32; 3 * w];
    let mut hm = vec![0f32; 3 * w];
    let prep = |y: usize, t: &mut [f32], hm: &mut [f32]| {
        let (t, hm) = (&mut t[y % 3 * w..][..w], &mut hm[y % 3 * w..][..w]);
        for (o, &v) in t.iter_mut().zip(&eig[y * w..(y + 1) * w]) {
            *o = if v > th { v } else { 0.0 };
        }
        for (((o, &a), &b), &c) in hm[1..w - 1].iter_mut().zip(&t[..w - 2]).zip(&t[1..w - 1]).zip(&t[2..]) {
            *o = fmax(fmax(a, b), c);
        }
    };
    prep(ya - 1, &mut t, &mut hm);
    prep(ya, &mut t, &mut hm);
    let mut flag = vec![0u8; w];
    let mut out = Vec::new();
    for y in ya..yb {
        prep(y + 1, &mut t, &mut hm);
        let (h0, h1, h2) = (&hm[(y - 1) % 3 * w..][..w], &hm[y % 3 * w..][..w], &hm[(y + 1) % 3 * w..][..w]);
        let tv = &t[y % 3 * w..][..w];
        for ((((f, &v), &a), &b), &c) in flag[1..w - 1].iter_mut().zip(&tv[1..w - 1]).zip(&h0[1..w - 1]).zip(&h1[1..w - 1]).zip(&h2[1..w - 1]) {
            *f = ((v != 0.0) & (v == fmax(fmax(a, b), c))) as u8;
        }
        if let Some(mk) = mask {
            for (f, &m) in flag.iter_mut().zip(&mk[y * w..(y + 1) * w]) {
                *f &= (m != 0) as u8;
            }
        }
        // 대부분 0 이라 8 칸씩 건너뜀
        for (ci, ch) in flag.chunks(8).enumerate() {
            if ch.iter().all(|&f| f == 0) {
                continue;
            }
            for (k, &f) in ch.iter().enumerate() {
                if f != 0 {
                    let x = ci * 8 + k;
                    out.push(((ord_f32(tv[x]) as u64) << 32) | (y * w + x) as u64);
                }
            }
        }
    }
    out
}

thread_local! {
    /// 9×9 상자 합 그림 버퍼(스레드마다 하나)
    static BOX_BUF: std::cell::RefCell<Vec<i32>> = const { std::cell::RefCell::new(Vec::new()) };
    static COL_BUF: std::cell::RefCell<Vec<i32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// BriefDescriptorExtractor::compute(32 바이트, 방향 없음): 가장자리 28 화소(PATCH 48/2 + KERNEL 9/2) 안의 점은 지우고
/// 9×9 합끼리 비교. kps 에서 지운 점은 빠진다(원래 순서 유지)
/// 원본은 적분 그림(i32)에서 네 귀퉁이로 9×9 합을 구한다. 정수라 값이 같으므로 여기서는 9×9 합 그림을 한 번 만들어 한 번씩만 읽는다
pub fn describe(gray: &[u8], w: usize, h: usize, kps: &mut Vec<KeyPoint>) -> Vec<[u8; 32]> {
    const B: i32 = 28;
    let (wi, hi) = (w as i32, h as i32);
    if hi <= B * 2 || wi <= B * 2 {
        kps.clear();
        return Vec::new();
    }
    // KeyPointsFilter::runByImageBorder: Rect(28, 28, w-56, h-56).contains(Point(cvRound(x), cvRound(y)))
    kps.retain(|k| {
        let (x, y) = (k.x.round_ties_even() as i32, k.y.round_ties_even() as i32);
        x >= B && x < wi - B && y >= B && y < hi - B
    });
    if kps.is_empty() {
        return Vec::new();
    }
    with_buf(&BOX_BUF, |bx| {
        box9(gray, w, h, bx);
        // 쌍마다 중심에서의 칸 거리(줄 · w + 열)
        let off: Vec<[isize; 2]> = BRIEF32.iter().map(|p| [p[0] as isize * w as isize + p[1] as isize, p[2] as isize * w as isize + p[3] as isize]).collect();
        let bx = &bx[..];
        kps.par_iter()
            .with_min_len(128)
            .map(|k| {
                // smoothedSum: 중심 = (int)(pt + 0.5)(double 로 더하고 0 쪽으로 버림). 상자 중심은 그림 [4, n-4) 안
                let cy = (k.y as f64 + 0.5) as isize;
                let cx = (k.x as f64 + 0.5) as isize;
                let base = cy * w as isize + cx;
                let mut d = [0u8; 32];
                for (byte, o8) in d.iter_mut().zip(off.chunks_exact(8)) {
                    let mut v = 0u8;
                    for (b, o) in o8.iter().enumerate() {
                        v |= ((bx[(base + o[0]) as usize] < bx[(base + o[1]) as usize]) as u8) << (7 - b);
                    }
                    *byte = v;
                }
                d
            })
            .collect()
    })
}

/// 9×9 상자 합 그림: out[y·w + x] = Σ gray[y-4..=y+4][x-4..=x+4] (4 ≤ y < h-4, 4 ≤ x < w-4 만, 나머지는 쓰지 않음)
fn box9(gray: &[u8], w: usize, h: usize, out: &mut Vec<i32>) {
    out.resize(w * h, 0);
    // 세로 9 줄 합(열마다), 줄을 내려가며 한 줄 더하고 한 줄 뺌
    with_buf(&COL_BUF, |col| box9_with(gray, w, h, out, col));
}

fn box9_with(gray: &[u8], w: usize, h: usize, out: &mut [i32], col: &mut Vec<i32>) {
    col.clear();
    col.resize(w, 0);
    for y in 0..9 {
        for (c, &g) in col.iter_mut().zip(&gray[y * w..(y + 1) * w]) {
            *c += g as i32;
        }
    }
    for y in 4..h - 4 {
        if y > 4 {
            let (add, sub) = (&gray[(y + 4) * w..(y + 5) * w], &gray[(y - 5) * w..(y - 4) * w]);
            for ((c, &a), &s) in col.iter_mut().zip(add).zip(sub) {
                *c += a as i32 - s as i32;
            }
        }
        let o = &mut out[y * w + 4..y * w + w - 4];
        let n = w - 8;
        // 가로 9 칸 합: 앞 9 칸을 한 번 더하고 한 칸씩 밀며 더하고 뺌(정수라 값이 같음)
        let mut s: i32 = col[..9].iter().sum();
        o[0] = s;
        for k in 1..n {
            s += col[k + 8] - col[k - 1];
            o[k] = s;
        }
    }
}

/// util2d::getDepth(smoothing = true, depthErrorRatio = 0.02, 이웃 추정 없음). mm → m, 3×3 가중(1 2 1 / 2 4 2 / 1 2 1) 평균
fn get_depth(depth: &[u16], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let (wi, hi) = (w as i32, h as i32);
    let mut u = (x + 0.5) as i32;
    let mut v = (y + 0.5) as i32;
    if u == wi && x < wi as f32 {
        u = wi - 1;
    }
    if v == hi && y < hi as f32 {
        v = hi - 1;
    }
    if !(u >= 0 && u < wi && v >= 0 && v < hi) {
        return 0.0;
    }
    let mm = |uu: i32, vv: i32| {
        let d = depth[vv as usize * w + uu as usize];
        if d > 0 && d < u16::MAX { d as f32 * 0.001 } else { 0.0 }
    };
    let mut d0 = mm(u, v);
    if d0 == 0.0 || !d0.is_finite() {
        return 0.0;
    }
    let (us, vs, ue, ve) = ((u - 1).max(0), (v - 1).max(0), (u + 1).min(wi - 1), (v + 1).min(hi - 1));
    let mut sw = 0f32;
    let mut sd = 0f32;
    // 원본 순서: 바깥 uu(열), 안쪽 vv(줄)
    for uu in us..=ue {
        for vv in vs..=ve {
            if !(uu == u && vv == v) {
                let mut d = mm(uu, vv);
                let err = 0.02f32 * d0;
                if d != 0.0 && d.is_finite() && (d - d0).abs() < err {
                    if uu == u || vv == v {
                        sw += 2.0;
                        d *= 2.0;
                    } else {
                        sw += 1.0;
                    }
                    sd += d;
                }
            }
        }
    }
    d0 *= 4.0;
    sw += 4.0;
    (d0 + sd) / sw
}

/// util3d::generateKeypoints3DDepth(카메라 1 대): 깊이 → 광학 축 점 → localTransform(float 3×4) 곱 → 몸체 축
pub fn keypoints_3d(depth: &[u16], cam: &crate::Camera, kps: &[KeyPoint], min_depth: f32, max_depth: f32) -> Vec<Point3<f32>> {
    let (w, h) = (cam.width as usize, cam.height as usize);
    let nan = Point3::new(f32::NAN, f32::NAN, f32::NAN);
    if depth.is_empty() {
        return Vec::new();
    }
    assert_eq!(depth.len(), w * h);
    // 깊이 크기 = 영상 크기라 배율 1(원본 식 그대로 계산)
    let sub_w = w as f32;
    let fxr = 1.0f32 / if cam.width > 0 { cam.width as f32 / sub_w } else { 1.0 };
    let fyr = 1.0f32 / if cam.height > 0 { cam.height as f32 / h as f32 } else { 1.0 };
    // CameraModel 의 cx() 등은 double, × float 배율 → double → float 인자
    let cx0 = (cam.cx * fxr as f64) as f32;
    let cy0 = (cam.cy * fyr as f64) as f32;
    let fx = (cam.fx * fxr as f64) as f32;
    let fy = (cam.fy * fyr as f64) as f32;
    let cx = if cx0 > 0.0 { cx0 } else { (w / 2) as f32 - 0.5 };
    let cy = if cy0 > 0.0 { cy0 } else { (h / 2) as f32 - 0.5 };
    // Transform 은 float 3×4
    let m = cam.local.to_homogeneous();
    let t: [[f32; 4]; 3] = std::array::from_fn(|r| std::array::from_fn(|c| m[(r, c)] as f32));
    let identity = (0..3).all(|r| (0..4).all(|c| t[r][c] == if r == c { 1.0 } else { 0.0 }));
    kps.iter()
        .map(|k| {
            let x = k.x * fxr;
            let y = k.y * fyr;
            let d = get_depth(depth, w, h, x, y);
            if !(d > 0.0) {
                return nan;
            }
            let p = [(x - cx) * d / fx, (y - cy) * d / fy, d];
            if !(p.iter().all(|v| v.is_finite()) && (min_depth < 0.0 || p[2] > min_depth) && (max_depth <= 0.0 || p[2] <= max_depth)) {
                return nan;
            }
            if identity {
                return Point3::new(p[0], p[1], p[2]);
            }
            // util3d::transformPoint: 왼쪽부터 float 로 더함
            let tr = |r: usize| t[r][0] * p[0] + t[r][1] * p[1] + t[r][2] * p[2] + t[r][3];
            Point3::new(tr(0), tr(1), tr(2))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Camera, Pose};

    fn frame(gray: Vec<u8>, depth: Vec<u16>, w: u32, h: u32) -> Frame {
        Frame {
            id: 0,
            t: 0.0,
            gray,
            color: vec![],
            color_is_rgb: false,
            depth_mm: depth.into(),
            depth_raw: Default::default(),
            depth_per_m: 1000.0,
            cam: Camera { fx: 500.0, fy: 500.0, cx: 159.5, cy: 119.5, width: w, height: h, local: crate::io::optical_rotation() },
        }
    }

    /// 밝은 사각형 하나: 네 모서리 근처가 특징점, 깊이 1 m 면 몸체 축 x = 1
    #[test]
    fn square_corners() {
        let (w, h) = (320usize, 240usize);
        let mut g = vec![20u8; w * h];
        for y in 80..160 {
            for x in 100..220 {
                g[y * w + x] = 220;
            }
        }
        let f = frame(g, vec![1000; w * h], w as u32, h as u32);
        let ex = Extractor::new(&Params::default());
        let out = ex.extract(&f);
        assert!(out.kps.len() >= 4, "{}", out.kps.len());
        for (cx, cy) in [(100.0, 80.0), (219.0, 80.0), (100.0, 159.0), (219.0, 159.0)] {
            assert!(out.kps.iter().any(|k| (k.x - cx).abs() <= 1.0 && (k.y - cy).abs() <= 1.0), "모서리 ({cx},{cy}) 없음");
        }
        // response 내림차순
        assert!(out.kps.windows(2).all(|p| p[0].response >= p[1].response));
        assert_eq!(out.kps.len(), out.desc.len());
        assert_eq!(out.kps.len(), out.pts3d.len());
        for (k, p) in out.kps.iter().zip(&out.pts3d) {
            assert_eq!(p.x, 1.0); // 몸체 x = 광학 z
            assert!((p.y + (k.x - 159.5) / 500.0).abs() < 1e-5);
            assert!((p.z + (k.y - 119.5) / 500.0).abs() < 1e-5);
        }
        // 깊이가 없으면 가리개 때문에 점이 없음
        let f2 = frame(f.gray.clone(), vec![0; w * h], w as u32, h as u32);
        assert!(ex.extract(&f2).kps.is_empty());
    }

    /// 가장자리 28 화소 안의 점은 설명자 단계에서 빠짐(원래 순서 유지, 좌표는 반올림해서 봄)
    #[test]
    fn border_removed() {
        let (w, h) = (100usize, 80usize);
        let g = vec![0u8; w * h];
        let mk = |x: f32, y: f32| KeyPoint { x, y, size: 3.0, angle: -1.0, response: 1.0, octave: 0 };
        let mut kps = vec![mk(27.0, 40.0), mk(28.0, 40.0), mk(71.0, 40.0), mk(72.0, 40.0), mk(50.0, 27.6), mk(50.0, 51.0), mk(50.0, 52.0)];
        let d = describe(&g, w, h, &mut kps);
        let xy: Vec<(f32, f32)> = kps.iter().map(|k| (k.x, k.y)).collect();
        assert_eq!(xy, vec![(28.0, 40.0), (71.0, 40.0), (50.0, 27.6), (50.0, 51.0)]);
        assert_eq!(d.len(), 4);
        // 평평한 그림: 모든 비교가 같음 → 비트 0
        assert!(d.iter().all(|x| x.iter().all(|&b| b == 0)));
    }

    /// 비트 배치: 바이트 i 의 비트 7-b = 쌍 8i+b. 9×9 합을 직접 세어 모든 비트를 맞춰 봄
    #[test]
    fn descriptor_bit_layout() {
        let (w, h) = (100usize, 100usize);
        let g: Vec<u8> = (0..w * h).map(|i| ((i * 2654435761usize) >> 7) as u8).collect();
        let mut kps = vec![KeyPoint { x: 50.0, y: 49.6, size: 3.0, angle: -1.0, response: 1.0, octave: 0 }];
        let d = describe(&g, w, h, &mut kps)[0];
        // 중심 (int)(49.6 + 0.5) = 50
        let s = |y: i32, x: i32| -> i64 {
            let mut t = 0i64;
            for yy in (50 + y - 4)..=(50 + y + 4) {
                for xx in (50 + x - 4)..=(50 + x + 4) {
                    t += g[yy as usize * w + xx as usize] as i64;
                }
            }
            t
        };
        for (i, p) in BRIEF32.iter().enumerate() {
            let bit = (s(p[0] as i32, p[1] as i32) < s(p[2] as i32, p[3] as i32)) as u8;
            assert_eq!((d[i / 8] >> (7 - i % 8)) & 1, bit, "쌍 {i}");
        }
        // 첫 쌍은 바이트 0 의 맨 위 비트
        assert_eq!(BRIEF32[0], [-2, -1, 7, -1]);
    }

    /// 세로 띠 나누기와 상관없이 비트가 같음(띠 경계의 반사·이웃)
    #[test]
    fn strips_match_single() {
        let (w, h) = (300usize, 50usize);
        let g: Vec<u8> = (0..w * h).map(|i| ((i * 7919) % 251) as u8).collect();
        let mut one = vec![0f32; w * h];
        eig_strip(&g, w, h, 0, w, &mut one, None, false, false, 0.0);
        let mut many = vec![0f32; w * h];
        for (a, b) in [(0, 64), (64, 130), (130, 299), (299, 300)] {
            eig_strip(&g, w, h, a, b, &mut many, None, false, false, 0.0);
        }
        assert!(one.iter().zip(&many).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// 1 단계 극대점 + 문턱 거르기가 원본식 후보(문턱 → dilate 비교)와 같은 집합(여러 스레드 = 여러 띠 포함)
    #[test]
    fn peaks_match_candidates() {
        let (w, h) = (640usize, 120usize);
        let g: Vec<u8> = (0..w * h).map(|i| (((i * 2654435761usize) >> 9) % 256) as u8 / 4 + ((i % w) / 20) as u8).collect();
        let mask: Vec<u8> = (0..w * h).map(|i| if (i / 7) % 5 == 0 { 0 } else { 255 }).collect();
        for m in [None, Some(&mask[..])] {
            for threads in [1, 3, 8] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                let got = pool.install(|| gftt(&g, w, h, m, 0, 0.001, 0.0));
                let mut eig = Vec::new();
                let mx = min_eigen_into(&g, w, h, m, &mut eig);
                let th = (mx as f64 * 0.001) as f32;
                let mut want = candidates(&eig, w, m, th, 1, h - 1);
                want.sort_unstable_by(|a, b| b.cmp(a));
                let gp: Vec<u64> = got.iter().map(|k| k.y as u64 * w as u64 + k.x as u64).collect();
                let wp: Vec<u64> = want.iter().map(|k| k & 0xFFFF_FFFF).collect();
                assert_eq!(gp, wp, "threads {threads}");
            }
        }
    }

    #[test]
    fn identity_local_keeps_optical() {
        let mut cam = frame(vec![0; 16], vec![2000; 16], 4, 4).cam.clone();
        cam.local = Pose::identity();
        let k = KeyPoint { x: 2.0, y: 1.0, ..Default::default() };
        let p = keypoints_3d(&[2000; 16], &cam, &[k], 0.0, 0.0)[0];
        assert_eq!(p.z, 2.0);
        // 깊이 범위 밖은 NaN
        assert!(keypoints_3d(&[2000; 16], &cam, &[k], 0.0, 1.5)[0].x.is_nan());
    }
}
