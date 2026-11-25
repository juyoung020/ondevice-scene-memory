//! ORB-SLAM3(RGB-D 경로)가 부르는 OpenCV 4.6 영상 함수 중 ORB 추출기 밖의 것을 같은 정수 · 실수 연산 순서로 옮긴 것.
//! (resize · GaussianBlur · FAST 는 ORB 추출기와 함께 `opt/orb3-rust` 의 `orb3rs::extract` 에 있다.)
//! 원본: imgproc/src/color_rgb.simd.hpp(RGB2Gray<uchar>), core/src/convert_scale.simd.hpp(cvtScale 16U→32F),
//! calib3d/src/undistort.dispatch.cpp(cvUndistortPointsInternal), core/src/copy.cpp(copyMakeBorder, BORDER_REFLECT_101).
//! 이 파일은 crate 의 다른 모듈을 쓰지 않는다(저장소 밖 하네스가 `#[path]` 로 그대로 끌어와 비교).

/// cvtColor 의 채널 배치
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorOrder {
    Rgb,
    Bgr,
    Rgba,
    Bgra,
}

impl ColorOrder {
    fn cn(self) -> usize {
        match self {
            ColorOrder::Rgb | ColorOrder::Bgr => 3,
            _ => 4,
        }
    }
    fn blue_first(self) -> bool {
        matches!(self, ColorOrder::Bgr | ColorOrder::Bgra)
    }
}

// RGB2Gray<uchar>: 15 비트 고정소수 계수(합 32768), (Σ + 2^14) >> 15
const RY15: u32 = 9798;
const GY15: u32 = 19235;
const BY15: u32 = 3735;

/// cvtColor(COLOR_{RGB,BGR,RGBA,BGRA}2GRAY), 8 비트. `src` 는 w×h 연속(줄 간격 = w·채널)
pub fn cvt_gray(src: &[u8], w: usize, h: usize, order: ColorOrder, dst: &mut Vec<u8>) {
    let cn = order.cn();
    assert!(src.len() >= w * h * cn);
    // 첫 칸이 파랑이면 첫 칸 계수 BY15
    let (c0, c2) = if order.blue_first() { (BY15, RY15) } else { (RY15, BY15) };
    dst.clear();
    dst.extend(src[..w * h * cn].chunks_exact(cn).map(|p| ((p[0] as u32 * c0 + p[1] as u32 * GY15 + p[2] as u32 * c2 + (1 << 14)) >> 15) as u8));
}

/// imDepth.convertTo(CV_32F, scale): 16 비트 → f32, 값 = (f32)v · (f32)scale(원본 SIMD v_fma(v, a, 0) 과 같은 한 번 반올림)
pub fn depth_to_f32(src: &[u16], scale: f32, dst: &mut Vec<f32>) {
    dst.clear();
    dst.extend(src.iter().map(|&v| v as f32 * scale));
}

/// copyMakeBorder(BORDER_REFLECT_101): 위 · 아래 · 왼 · 오른 칸 수만큼 반사(가장자리 칸은 되풀이 안 함). 1 채널 8 비트
pub fn copy_make_border_reflect101(src: &[u8], w: usize, h: usize, top: usize, bottom: usize, left: usize, right: usize) -> Vec<u8> {
    assert!(w > 1 && h > 1 && src.len() >= w * h);
    let ow = w + left + right;
    let oh = h + top + bottom;
    let r101 = |p: isize, n: usize| -> usize {
        // borderInterpolate(BORDER_REFLECT_101): 크기보다 테두리가 넓으면 되풀이 반사
        let n = n as isize;
        let mut p = p;
        while p < 0 || p >= n {
            if p < 0 {
                p = -p;
            } else {
                p = 2 * n - 2 - p;
            }
        }
        p as usize
    };
    let xs: Vec<usize> = (0..ow).map(|x| r101(x as isize - left as isize, w)).collect();
    let mut out = vec![0u8; ow * oh];
    for y in 0..oh {
        let sy = r101(y as isize - top as isize, h);
        let s = &src[sy * w..sy * w + w];
        let o = &mut out[y * ow..y * ow + ow];
        for (d, &sx) in o.iter_mut().zip(&xs) {
            *d = s[sx];
        }
    }
    out
}

/// 핀홀 카메라 행렬(f32, ORB-SLAM3 Pinhole::toK 와 같은 값)
#[derive(Clone, Copy, Debug)]
pub struct PinholeK {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

/// cv::undistortPoints(점, 점, K, 왜곡, Mat(), K) — ORB-SLAM3 Frame::UndistortKeyPoints · ComputeImageBounds.
/// 왜곡 계수는 f32(k1 k2 p1 p2 [k3 [k4 k5 k6]]), 반복은 기본 기준(COUNT 5 번). 출력은 다시 화소 좌표(P = K).
pub fn undistort_points(pts: &[(f32, f32)], k: &PinholeK, dist: &[f32], out: &mut Vec<(f32, f32)>) {
    assert!(matches!(dist.len(), 4 | 5 | 8 | 12 | 14));
    let mut kk = [0f64; 14];
    for (d, &s) in kk.iter_mut().zip(dist) {
        *d = s as f64;
    }
    let (fx, fy, cx, cy) = (k.fx as f64, k.fy as f64, k.cx as f64, k.cy as f64);
    let ifx = 1.0 / fx;
    let ify = 1.0 / fy;
    // RR = P(3×3) · I (cvMatMul: 행 · 열 곱 합, 0 부터 더함)
    let p = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]];
    let mut rr = [[0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for t in 0..3 {
                s += p[i][t] * if t == j { 1.0 } else { 0.0 };
            }
            rr[i][j] = s;
        }
    }
    assert!(kk[12] == 0.0 && kk[13] == 0.0, "기울임 왜곡(τx, τy)은 안 옮김");
    out.clear();
    for &(u, v) in pts {
        let (u, v) = (u as f64, v as f64);
        let mut x = (u - cx) * ifx;
        let mut y = (v - cy) * ify;
        // 기울임 보정: invMatTilt = 단위, Matx33d · Vec3d(x, y, 1)
        let ux = 0.0 + 1.0 * x + 0.0 * y + 0.0 * 1.0;
        let uy = 0.0 + 0.0 * x + 1.0 * y + 0.0 * 1.0;
        let uz = 0.0 + 0.0 * x + 0.0 * y + 1.0 * 1.0;
        let inv_proj = if uz != 0.0 { 1.0 / uz } else { 1.0 };
        x = inv_proj * ux;
        y = inv_proj * uy;
        let (x0, y0) = (x, y);
        for _ in 0..5 {
            let r2 = x * x + y * y;
            let icdist = (1.0 + ((kk[7] * r2 + kk[6]) * r2 + kk[5]) * r2) / (1.0 + ((kk[4] * r2 + kk[1]) * r2 + kk[0]) * r2);
            if icdist < 0.0 {
                x = (u - cx) * ifx;
                y = (v - cy) * ify;
                break;
            }
            let dx = 2.0 * kk[2] * x * y + kk[3] * (r2 + 2.0 * x * x) + kk[8] * r2 + kk[9] * r2 * r2;
            let dy = kk[2] * (r2 + 2.0 * y * y) + 2.0 * kk[3] * x * y + kk[10] * r2 + kk[11] * r2 * r2;
            x = (x0 - dx) * icdist;
            y = (y0 - dy) * icdist;
        }
        let xx = rr[0][0] * x + rr[0][1] * y + rr[0][2];
        let yy = rr[1][0] * x + rr[1][1] * y + rr[1][2];
        let ww = 1.0 / (rr[2][0] * x + rr[2][1] * y + rr[2][2]);
        out.push(((xx * ww) as f32, (yy * ww) as f32));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn border_reflect101() {
        // 1 2 3 → 왼 2 칸: 3 2 | 1 2 3 | 2 1
        let s = [1u8, 2, 3, 4, 5, 6];
        let o = copy_make_border_reflect101(&s, 3, 2, 0, 0, 2, 2);
        assert_eq!(&o[..7], &[3, 2, 1, 2, 3, 2, 1]);
    }

    #[test]
    fn gray_weights_sum() {
        let mut d = Vec::new();
        cvt_gray(&[255, 255, 255, 0, 0, 0], 2, 1, ColorOrder::Rgb, &mut d);
        assert_eq!(d, vec![255, 0]);
    }

    #[test]
    fn undistort_zero_is_identity_ish() {
        let k = PinholeK { fx: 600.0, fy: 600.0, cx: 320.0, cy: 240.0 };
        let mut o = Vec::new();
        undistort_points(&[(100.0, 50.0)], &k, &[0.0; 5], &mut o);
        assert!((o[0].0 - 100.0).abs() < 1e-3 && (o[0].1 - 50.0).abs() < 1e-3);
    }
}
