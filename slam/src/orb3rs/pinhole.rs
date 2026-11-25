//! ORB-SLAM3 `GeometricCamera` / `Pinhole`(src/CameraModels/Pinhole.cpp) 의 투영 부분.
//!
//! 매개변수는 원본처럼 f32 `[fx, fy, cx, cy]`, f64 판 식은 f32 를 f64 로 올려 같은 순서로 계산한다
//! (`fx * x / z + cx` = `((fx·x)/z) + cx`, 나눗셈이 끼어 FMA 합치기 없음).

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pinhole {
    /// mvParameters: fx, fy, cx, cy
    pub p: [f32; 4],
}

impl Pinhole {
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self { p: [fx, fy, cx, cy] }
    }

    /// `Pinhole::project(const Eigen::Vector3d&)`
    #[inline]
    pub fn project(&self, v: &[f64; 3]) -> [f64; 2] {
        let [fx, fy, cx, cy] = self.p.map(f64::from);
        [fx * v[0] / v[2] + cx, fy * v[1] / v[2] + cy]
    }

    /// `Pinhole::project(const Eigen::Vector3f&)`
    #[inline]
    pub fn project_f(&self, v: &[f32; 3]) -> [f32; 2] {
        let [fx, fy, cx, cy] = self.p;
        [fx * v[0] / v[2] + cx, fy * v[1] / v[2] + cy]
    }

    /// `Pinhole::unprojectEig` — 정규 평면 (x, y, 1)
    #[inline]
    pub fn unproject(&self, u: f32, v: f32) -> [f32; 3] {
        let [fx, fy, cx, cy] = self.p;
        [(u - cx) / fx, (v - cy) / fy, 1.0]
    }

    /// `Pinhole::projectJac(const Eigen::Vector3d&)` — 2x3, 열 우선 `[j00, j10, j01, j11, j02, j12]`
    #[inline]
    pub fn project_jac(&self, v: &[f64; 3]) -> [f64; 6] {
        let fx = f64::from(self.p[0]);
        let fy = f64::from(self.p[1]);
        let z2 = v[2] * v[2];
        [fx / v[2], 0.0, 0.0, fy / v[2], -fx * v[0] / z2, -fy * v[1] / z2]
    }

    /// `Pinhole::uncertainty2` — 핀홀은 1
    #[inline]
    pub fn uncertainty2(&self) -> f32 {
        1.0
    }
}
