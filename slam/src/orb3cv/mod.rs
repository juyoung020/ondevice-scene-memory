//! 포터 CV: ORB-SLAM3 가 부르는 OpenCV · Eigen 계산 중 ORB 추출기 밖의 것을 원본과 비트 같게 옮긴 것.
//! img = cvtColor · convertTo · copyMakeBorder · undistortPoints(OpenCV 4.6), svd = Eigen 3.4 JacobiSVD · GeometricTools::Triangulate.
pub mod img;
pub mod svd;
pub mod settings;

impl settings::TrackingParams {
    /// 원본 Frame 이 쓰는 카메라 값(`orb3rs::types::Camera`). w · h = 들어오는 영상 크기
    pub fn camera(&self, w: i32, h: i32) -> crate::orb3rs::types::Camera {
        crate::orb3rs::types::Camera {
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            invfx: self.invfx,
            invfy: self.invfy,
            bf: self.bf,
            b: self.b,
            th_depth: self.th_depth,
            width: w,
            height: h,
        }
    }
}
pub mod atlas_io;
pub mod system;
pub mod g2osim3;
pub mod sim3opt;
