//! RTAB-Map Transform(3x4 float) 를 비트 단위로 같게 옮긴 것.
//! 원본: corelib/src/Transform.cpp. 곱·역행렬은 Eigen 3.4(SSE) 의 Matrix4f 연산, 회전 정규화는 Quaternionf,
//! 오일러 각은 pcl::getTransformation / getTranslationAndEulerAngles(float) 를 같은 연산 순서로 따라 한다.
//! 위치 추적의 모든 위치 계산(추측 움직임, 키프레임 기준 움직임, 누적 위치)이 이 float 연산을 거친다.
use crate::Pose;
use nalgebra::{Matrix3, Rotation3, Translation3, UnitQuaternion};

/// 3x4 float 행렬(행 우선). RTAB-Map Transform::data() 와 같은 배치
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Tf(pub [f32; 12]);

impl Default for Tf {
    fn default() -> Self {
        Tf::NULL
    }
}

impl Tf {
    pub const NULL: Tf = Tf([0.0; 12]);
    pub const IDENTITY: Tf = Tf([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> f32 {
        self.0[r * 4 + c]
    }
    pub fn x(&self) -> f32 {
        self.0[3]
    }
    pub fn y(&self) -> f32 {
        self.0[7]
    }
    pub fn z(&self) -> f32 {
        self.0[11]
    }

    /// Transform::isNull: 모두 0 이거나 NaN 이 하나라도 있으면 빈 값
    pub fn is_null(&self) -> bool {
        self.0.iter().all(|&v| v == 0.0) || self.0.iter().any(|v| v.is_nan())
    }

    /// Transform::isIdentity
    pub fn is_identity(&self) -> bool {
        self.0 == Tf::IDENTITY.0
    }

    /// Transform::operator*: Matrix4f 곱 뒤 회전을 Quaternionf 로 정규화
    pub fn mul(&self, o: &Tf) -> Tf {
        let a = &self.0;
        let b = &o.0;
        // Eigen 4x4 float 곱(열 단위 패킷): res(i,j) = ((a(i,0)b(0,j) + a(i,1)b(1,j)) + a(i,2)b(2,j)) + a(i,3)b(3,j)
        let mut m = [0f32; 12];
        for i in 0..3 {
            for j in 0..4 {
                let b3 = if j == 3 { 1.0f32 } else { 0.0f32 };
                m[i * 4 + j] = ((a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j]) + a[i * 4 + 2] * b[8 + j]) + a[i * 4 + 3] * b3;
            }
        }
        let r = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
        let q = quat_from_mat3f(&r);
        let q = quat_normalized_f(q);
        let rr = quat_to_mat3f(q);
        Tf([rr[0], rr[1], rr[2], m[3], rr[3], rr[4], rr[5], m[7], rr[6], rr[7], rr[8], m[11]])
    }

    /// Transform::inverse: Matrix4f::computeInverseAndDetWithCheck(Eigen SSE 4x4 역행렬)
    pub fn inverse(&self) -> Tf {
        let a = &self.0;
        // 열 우선 4x4
        let col = |c: usize| -> [f32; 4] { [a[c], a[4 + c], a[8 + c], if c == 3 { 1.0 } else { 0.0 }] };
        let inv = inverse4_sse(col(0), col(1), col(2), col(3));
        // inv[c] = 결과의 c 번째 열
        Tf([
            inv[0][0], inv[1][0], inv[2][0], inv[3][0], inv[0][1], inv[1][1], inv[2][1], inv[3][1], inv[0][2], inv[1][2],
            inv[2][2], inv[3][2],
        ])
    }

    /// Transform(x,y,z,roll,pitch,yaw): pcl::getTransformation(float).
    /// 원본은 std::cos(float)=cosf 와 이름만 쓴 sin(=double sin) 을 섞어 쓴다(시험으로 확인)
    pub fn from_xyz_rpy(x: f32, y: f32, z: f32, roll: f32, pitch: f32, yaw: f32) -> Tf {
        let a = yaw.cos();
        let b = (yaw as f64).sin() as f32;
        let c = pitch.cos();
        let d = (pitch as f64).sin() as f32;
        let e = roll.cos();
        let f = (roll as f64).sin() as f32;
        let de = d * e;
        let df = d * f;
        Tf([
            a * c,
            a * df - b * e,
            b * f + a * de,
            x,
            b * c,
            a * e + b * df,
            b * de - a * f,
            y,
            -d,
            c * f,
            c * e,
            z,
        ])
    }

    /// getTranslationAndEulerAngles: pcl(float). atan2 는 float, asin 은 double
    pub fn xyz_rpy(&self) -> [f32; 6] {
        let t = &self.0;
        let roll = t[9].atan2(t[10]);
        let pitch = ((-t[8]) as f64).asin() as f32;
        let yaw = t[4].atan2(t[0]);
        [t[3], t[7], t[11], roll, pitch, yaw]
    }

    /// util3d::transformPoint(cv::Point3f, Transform)
    #[inline]
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let t = &self.0;
        [
            t[0] * p[0] + t[1] * p[1] + t[2] * p[2] + t[3],
            t[4] * p[0] + t[5] * p[1] + t[6] * p[2] + t[7],
            t[8] * p[0] + t[9] * p[1] + t[10] * p[2] + t[11],
        ]
    }

    /// f64 위치(nalgebra) 로. 회전은 가장 가까운 회전 행렬로 맞춤
    pub fn to_pose(&self) -> Pose {
        let t = &self.0;
        let m = Matrix3::new(t[0] as f64, t[1] as f64, t[2] as f64, t[4] as f64, t[5] as f64, t[6] as f64, t[8] as f64, t[9] as f64, t[10] as f64);
        let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(&m));
        Pose::from_parts(Translation3::new(t[3] as f64, t[7] as f64, t[11] as f64), q)
    }

    pub fn from_pose(p: &Pose) -> Tf {
        let m = p.to_homogeneous();
        let mut o = [0f32; 12];
        for r in 0..3 {
            for c in 0..4 {
                o[r * 4 + c] = m[(r, c)] as f32;
            }
        }
        Tf(o)
    }

    /// Transform::toEigen3d 의 3x4 (double)
    pub fn to_f64(&self) -> [f64; 12] {
        let mut o = [0f64; 12];
        for i in 0..12 {
            o[i] = self.0[i] as f64;
        }
        o
    }
}

/// Eigen quaternionbase_assign_impl<3,3>(float). 반환 [x,y,z,w]
pub fn quat_from_mat3f(m: &[f32; 9]) -> [f32; 4] {
    // trace = d0 + (d1 + d2) (Eigen redux 펼침)
    let t = m[0] + (m[4] + m[8]);
    let mut q = [0f32; 4];
    if t > 0.0 {
        let t = (t + 1.0).sqrt();
        q[3] = 0.5 * t;
        let t = 0.5 / t;
        q[0] = (m[7] - m[5]) * t;
        q[1] = (m[2] - m[6]) * t;
        q[2] = (m[3] - m[1]) * t;
    } else {
        let mut i = 0;
        if m[4] > m[0] {
            i = 1;
        }
        if m[8] > m[i * 4] {
            i = 2;
        }
        let j = (i + 1) % 3;
        let k = (j + 1) % 3;
        let t = (m[i * 4] - m[j * 4] - m[k * 4] + 1.0).sqrt();
        q[i] = 0.5 * t;
        let t = 0.5 / t;
        q[3] = (m[k * 3 + j] - m[j * 3 + k]) * t;
        q[j] = (m[j * 3 + i] + m[i * 3 + j]) * t;
        q[k] = (m[k * 3 + i] + m[i * 3 + k]) * t;
    }
    q
}

/// Quaternionf::normalized: 제곱합은 SSE 축약 (x²+z²)+(y²+w²)
pub fn quat_normalized_f(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3]);
    if n > 0.0 {
        let s = n.sqrt();
        [q[0] / s, q[1] / s, q[2] / s, q[3] / s]
    } else {
        q
    }
}

/// QuaternionBase::toRotationMatrix(float), 행 우선 3x3
pub fn quat_to_mat3f(q: [f32; 4]) -> [f32; 9] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let tx = 2.0 * x;
    let ty = 2.0 * y;
    let tz = 2.0 * z;
    let twx = tx * w;
    let twy = ty * w;
    let twz = tz * w;
    let txx = tx * x;
    let txy = ty * x;
    let txz = tz * x;
    let tyy = ty * y;
    let tyz = tz * y;
    let tzz = tz * z;
    [1.0 - (tyy + tzz), txy - twz, txz + twy, txy + twz, 1.0 - (txx + tzz), tyz - twx, txz - twy, tyz + twx, 1.0 - (txx + tyy)]
}

// ---- Eigen InverseSize4.h(float, SSE) 를 4 칸 배열로 흉내 ----
type P4 = [f32; 4];
#[inline]
fn mul4(a: P4, b: P4) -> P4 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}
#[inline]
fn add4(a: P4, b: P4) -> P4 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}
#[inline]
fn sub4(a: P4, b: P4) -> P4 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}
/// vec4f_swizzle2(a,b,p,q,r,s) = [a[p], a[q], b[r], b[s]]
#[inline]
fn sw2(a: P4, b: P4, p: usize, q: usize, r: usize, s: usize) -> P4 {
    [a[p], a[q], b[r], b[s]]
}
/// _mm_movelh_ps(a,b) = [a0,a1,b0,b1]
#[inline]
fn movelh(a: P4, b: P4) -> P4 {
    [a[0], a[1], b[0], b[1]]
}
/// _mm_movehl_ps(a,b) = [b2,b3,a2,a3]
#[inline]
fn movehl(a: P4, b: P4) -> P4 {
    [b[2], b[3], a[2], a[3]]
}
#[inline]
fn dup(a: P4, p: usize) -> P4 {
    [a[p]; 4]
}

/// 열 우선 4x4 의 역행렬(열 4 개 반환)
fn inverse4_sse(l1: P4, l2: P4, l3: P4, l4: P4) -> [P4; 4] {
    let a = movelh(l1, l2);
    let b = movehl(l2, l1);
    let c = movelh(l3, l4);
    let d = movehl(l4, l3);

    let ab = sub4(mul4(sw2(a, a, 3, 3, 0, 0), b), mul4(sw2(a, a, 1, 1, 2, 2), sw2(b, b, 2, 3, 0, 1)));
    let dc = sub4(mul4(sw2(d, d, 3, 3, 0, 0), c), mul4(sw2(d, d, 1, 1, 2, 2), sw2(c, c, 2, 3, 0, 1)));

    let mut da = mul4(sw2(a, a, 3, 3, 1, 1), a);
    da = sub4(da, movehl(da, da));
    let mut db = mul4(sw2(b, b, 3, 3, 1, 1), b);
    db = sub4(db, movehl(db, db));
    let mut dcc = mul4(sw2(c, c, 3, 3, 1, 1), c);
    dcc = sub4(dcc, movehl(dcc, dcc));
    let mut dd = mul4(sw2(d, d, 3, 3, 1, 1), d);
    dd = sub4(dd, movehl(dd, dd));

    let mut dt = mul4(sw2(dc, dc, 0, 2, 1, 3), ab);
    dt = add4(dt, movehl(dt, dt));
    dt = add4(dt, sw2(dt, dt, 1, 0, 0, 0));
    let d1 = mul4(da, dd);
    let d2 = mul4(db, dcc);
    let det = dup(sub4(add4(d1, d2), dt), 0);
    let rd0 = [1.0f32 / det[0], 1.0 / det[1], 1.0 / det[2], 1.0 / det[3]];

    let mut id = mul4(sw2(c, c, 0, 0, 2, 2), movelh(ab, ab));
    id = add4(id, mul4(sw2(c, c, 1, 1, 3, 3), movehl(ab, ab)));
    id = sub4(mul4(d, dup(da, 0)), id);

    let mut ia = mul4(sw2(b, b, 0, 0, 2, 2), movelh(dc, dc));
    ia = add4(ia, mul4(sw2(b, b, 1, 1, 3, 3), movehl(dc, dc)));
    ia = sub4(mul4(a, dup(dd, 0)), ia);

    let mut ib = mul4(d, sw2(ab, ab, 3, 0, 3, 0));
    ib = sub4(ib, mul4(sw2(d, d, 1, 0, 3, 2), sw2(ab, ab, 2, 1, 2, 1)));
    ib = sub4(mul4(c, dup(db, 0)), ib);

    let mut ic = mul4(a, sw2(dc, dc, 3, 0, 3, 0));
    ic = sub4(ic, mul4(sw2(a, a, 1, 0, 3, 2), sw2(dc, dc, 2, 1, 2, 1)));
    ic = sub4(mul4(b, dup(dcc, 0)), ic);

    let rd = [rd0[0], -rd0[1], -rd0[2], rd0[3]];
    let ia = mul4(ia, rd);
    let ib = mul4(ib, rd);
    let ic = mul4(ic, rd);
    let id = mul4(id, rd);
    [sw2(ia, ib, 3, 1, 3, 1), sw2(ia, ib, 2, 0, 2, 0), sw2(ic, id, 3, 1, 3, 1), sw2(ic, id, 2, 0, 2, 0)]
}

/// CameraModel::opticalRotation(): 광학 축 → 몸체 축
pub const OPTICAL: Tf = Tf([0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_and_mul_roundtrip() {
        let a = Tf::from_xyz_rpy(0.3, -0.2, 1.1, 0.1, -0.2, 0.7);
        let i = a.inverse();
        let e = a.mul(&i);
        for k in 0..12 {
            assert!((e.0[k] - Tf::IDENTITY.0[k]).abs() < 1e-5, "{:?}", e);
        }
        let r = a.xyz_rpy();
        assert!((r[3] - 0.1).abs() < 1e-5 && (r[4] + 0.2).abs() < 1e-5 && (r[5] - 0.7).abs() < 1e-5);
    }
}
