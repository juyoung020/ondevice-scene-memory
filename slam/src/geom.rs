//! 기하 도구: 강체 변환(float 3×4 `Tf`), PnP RANSAC(EPnP 6 점), 3×3 · SVD 수학.
//! 위치 추적(klt) · 되찾기 · 루프 검증 · IMU 가 같이 쓴다.
pub mod cvmath;
mod epnp;
pub(crate) mod pnp;
pub mod rtf;

pub use rtf::Tf;

use crate::Camera;
use cvmath::K;

/// PnP RANSAC(EPnP 6 점): 세계 점 op → 화소 ip. (세계 → 광학 축 변환, 안쪽 번호)
pub fn pnp_ransac_k(op: &[[f32; 3]], ip: &[[f32; 2]], k4: &[f64; 4], iterations: i32, reproj: f32) -> Option<(Tf, Vec<usize>)> {
    let k = K { fx: k4[0], fy: k4[1], cx: k4[2], cy: k4[3] };
    let (mut m1, mut m2) = (Vec::new(), Vec::new());
    let r = pnp::solve_pnp_ransac(op, ip, &k, iterations, reproj, &mut m1, &mut m2)?;
    let rm = cvmath::rodrigues_v2m(r.rvec);
    let t = Tf([rm[0] as f32, rm[1] as f32, rm[2] as f32, r.tvec[0] as f32, rm[3] as f32, rm[4] as f32, rm[5] as f32, r.tvec[1] as f32, rm[6] as f32, rm[7] as f32, rm[8] as f32, r.tvec[2] as f32]);
    Some((t, r.inliers))
}

/// Camera.local(몸체 → 광학)을 float 변환으로. 광학 회전(0, ±1 만)은 정확한 값으로 맞춤
pub fn cam_local(cam: &Camera) -> Tf {
    let mut t = Tf::from_pose(&cam.local);
    for v in t.0.iter_mut() {
        for c in [-1.0f32, 0.0, 1.0] {
            if (*v - c).abs() < 1e-6 {
                *v = c;
            }
        }
    }
    t
}
