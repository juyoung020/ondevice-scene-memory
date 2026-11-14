//! 시험·벤치용 합성 장면과 D455 비슷한 보정값(가정 값: 실제 값은 rs_info 로 받는다).
//! 장면은 깊이 카메라 좌표(OpenCV: x 오른쪽, y 아래, z 앞, m)의 평면·상자로 만들고 광선을 쏴서 깊이를 그린다.
use crate::fast_align::{dist, Extrinsics, Intrinsics};
use crate::rsref;

pub fn make_intrin(w: i32, h: i32, fx: f32, fy: f32, ppx: f32, ppy: f32, model: i32, k: &[f32]) -> Intrinsics {
    let mut coeffs = [0.0f32; 5];
    coeffs[..k.len()].copy_from_slice(k);
    Intrinsics { width: w, height: h, ppx, ppy, fx, fy, model, coeffs }
}

/// D455 비슷한 값(공개 덤프 수준의 가정): 깊이 = 왼쪽 IR, Brown-Conrady 계수 0. 컬러 = Inverse Brown-Conrady 계수 있음.
pub fn d455_depth_640() -> Intrinsics {
    make_intrin(640, 480, 390.6, 390.6, 319.4, 238.7, dist::BROWN_CONRADY, &[])
}
pub fn d455_depth_848() -> Intrinsics {
    make_intrin(848, 480, 425.9, 425.9, 423.6, 238.9, dist::BROWN_CONRADY, &[])
}
pub fn d455_color_640() -> Intrinsics {
    make_intrin(640, 480, 383.1, 382.4, 321.6, 244.9, dist::INVERSE_BROWN_CONRADY, &[-0.0566, 0.0679, 0.00015, 0.00043, -0.0218])
}
pub fn d455_color_1280x720() -> Intrinsics {
    make_intrin(1280, 720, 642.0, 641.6, 652.6, 360.3, dist::INVERSE_BROWN_CONRADY, &[-0.0566, 0.0679, 0.00015, 0.00043, -0.0218])
}

/// 깊이→컬러: 약 59 mm 옆, 아주 작은 회전(열 우선)
pub fn d455_extrin() -> Extrinsics {
    let (ax, ay, az) = (0.0011f32, -0.0024f32, 0.0007f32); // 라디안(작은 각 근사로 회전 행렬)
    // R = I + [w]x, 열 우선
    let r = [[1.0, -az, ay], [az, 1.0, -ax], [-ay, ax, 1.0f32]];
    // 정규 직교로 다듬기(그람-슈미트)
    let norm = |v: &mut [f32; 3]| {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        for x in v.iter_mut() {
            *x /= n;
        }
    };
    let mut c0 = [r[0][0], r[1][0], r[2][0]];
    let mut c1 = [r[0][1], r[1][1], r[2][1]];
    norm(&mut c0);
    let d = c0[0] * c1[0] + c0[1] * c1[1] + c0[2] * c1[2];
    for i in 0..3 {
        c1[i] -= d * c0[i];
    }
    norm(&mut c1);
    let c2 = [c0[1] * c1[2] - c0[2] * c1[1], c0[2] * c1[0] - c0[0] * c1[2], c0[0] * c1[1] - c0[1] * c1[0]];
    let mut e = Extrinsics::default();
    for i in 0..3 {
        e.rotation[i] = c0[i];
        e.rotation[3 + i] = c1[i];
        e.rotation[6 + i] = c2[i];
    }
    e.translation = [-0.0591, 0.00012, 0.00041];
    e
}

pub fn shift_x(tx: f32) -> Extrinsics {
    let mut e = Extrinsics::default();
    e.translation[0] = tx;
    e
}

// ---- 기하 ----
/// n·p = d, 범위 제한 없음
#[derive(Clone, Copy)]
pub struct Plane {
    pub n: [f64; 3],
    pub d: f64,
}
/// 축 정렬 상자(깊이 카메라 좌표)
#[derive(Clone, Copy)]
pub struct Box3 {
    pub lo: [f64; 3],
    pub hi: [f64; 3],
}
#[derive(Clone, Default)]
pub struct Scene {
    pub planes: Vec<Plane>,
    pub boxes: Vec<Box3>,
}

impl Scene {
    /// 원점 o, 방향 v 의 광선이 처음 닿는 거리 t(없으면 -1)
    pub fn hit(&self, o: &[f64; 3], v: &[f64; 3]) -> f64 {
        let mut best = 1e30;
        for p in &self.planes {
            let nv = p.n[0] * v[0] + p.n[1] * v[1] + p.n[2] * v[2];
            if nv.abs() < 1e-12 {
                continue;
            }
            let t = (p.d - (p.n[0] * o[0] + p.n[1] * o[1] + p.n[2] * o[2])) / nv;
            if t > 1e-6 && t < best {
                best = t;
            }
        }
        for b in &self.boxes {
            let (mut t0, mut t1) = (-1e30f64, 1e30f64);
            let mut ok = true;
            for i in 0..3 {
                if !ok {
                    break;
                }
                if v[i].abs() < 1e-15 {
                    if o[i] < b.lo[i] || o[i] > b.hi[i] {
                        ok = false;
                    }
                    continue;
                }
                let (mut a, mut c) = ((b.lo[i] - o[i]) / v[i], (b.hi[i] - o[i]) / v[i]);
                if a > c {
                    std::mem::swap(&mut a, &mut c);
                }
                t0 = if t0 < a { a } else { t0 };
                t1 = if c < t1 { c } else { t1 };
            }
            if ok && t1 >= t0 && t0 > 1e-6 && t0 < best {
                best = t0;
            }
        }
        if best < 1e29 { best } else { -1.0 }
    }
}

/// 방: 뒷벽(살짝 기울어짐) + 바닥 + 상자 두 개(가까운 것은 0.6 m)
pub fn room() -> Scene {
    Scene {
        planes: vec![
            Plane { n: [0.05, 0.0, 0.9987], d: 3.2 },  // 뒷벽
            Plane { n: [0.0, -1.0, 0.0], d: -0.75 },   // 바닥(y = 0.75 m 아래)
        ],
        boxes: vec![
            Box3 { lo: [-0.45, -0.25, 1.4], hi: [0.05, 0.75, 1.9] },  // 책상 크기 상자
            Box3 { lo: [0.15, -0.05, 0.62], hi: [0.32, 0.20, 0.75] }, // 가까운 물건
        ],
    }
}
/// 정면 평면 z = d
pub fn front_plane(d: f64) -> Scene {
    Scene { planes: vec![Plane { n: [0.0, 0.0, 1.0], d }], boxes: vec![] }
}
/// 계단: 왼쪽 절반 z=1.0, 오른쪽 z=2.0(앞 판이 화면 왼쪽 절반을 가림) + 뒷벽 2.0
pub fn step() -> Scene {
    Scene { planes: vec![Plane { n: [0.0, 0.0, 1.0], d: 2.0 }], boxes: vec![Box3 { lo: [-10.0, -10.0, 1.0], hi: [0.0, 10.0, 1.01] }] }
}

/// 깊이 카메라 화소 중심 광선으로 깊이(z, m) 그리기. 왜곡 없는 깊이 카메라로 봄
pub fn render_depth(s: &Scene, di: &Intrinsics, unit: f32, noise_rel: f64, hole_frac: f64, seed: u32) -> Vec<u16> {
    let mut z = vec![0u16; (di.width * di.height) as usize];
    let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
    let mut rnd = || {
        // [0,1)
        st ^= st << 13;
        st ^= st >> 17;
        st ^= st << 5;
        (st & 0xFFFFFF) as f64 / 16777216.0
    };
    let o = [0.0f64; 3];
    for y in 0..di.height {
        for x in 0..di.width {
            let v = [((x as f32 - di.ppx) / di.fx) as f64, ((y as f32 - di.ppy) / di.fy) as f64, 1.0];
            let t = s.hit(&o, &v);
            let (r1, r2, r3) = (rnd(), rnd(), rnd());
            if t <= 0.0 || t > 10.0 {
                continue;
            }
            if r1 < hole_frac {
                continue;
            }
            let mut zz = t; // v[2]=1 이라 t = z
            if noise_rel > 0.0 {
                let g = (-2.0 * (if r2 < 1e-12 { 1e-12 } else { r2 }).ln()).sqrt() * (6.283185307 * r3).cos();
                zz *= 1.0 + noise_rel * g;
            }
            let q = (zz / unit as f64).round() as i64;
            z[(y * di.width + x) as usize] = q.clamp(0, 65535) as u16;
        }
    }
    z
}

/// 정답: 컬러 화소 중심이 보는 면의 점을 깊이 카메라 좌표 z 로(정렬 결과가 담는 값과 같은 뜻). 0 = 깊이 카메라에서 안 보임
/// (컬러에서 보이는 점이 깊이 카메라에서 가려졌는지는 깊이 카메라 쪽 광선으로 다시 확인).
pub fn truth_in_color(s: &Scene, di: &Intrinsics, ci: &Intrinsics, d2c: &Extrinsics) -> Vec<f32> {
    let mut out = vec![0.0f32; (ci.width * ci.height) as usize];
    // 컬러 → 깊이: p_d = R^T (p_c - t)
    let r = &d2c.rotation; // 열 우선: p_c = R p_d + t, R(i,j) = rotation[j*3+i]
    let t = &d2c.translation;
    let mut oc = [0.0f64; 3];
    for i in 0..3 {
        oc[i] = -(r[i * 3] * t[0] + r[i * 3 + 1] * t[1] + r[i * 3 + 2] * t[2]) as f64;
    }
    for y in 0..ci.height {
        for x in 0..ci.width {
            let ray = rsref::rs2_deproject_pixel_to_point(ci, &[x as f32, y as f32], 1.0); // 컬러 왜곡 풀기(rsutil 규칙)
            let mut vd = [0.0f64; 3];
            for i in 0..3 {
                vd[i] = (r[i * 3] * ray[0] + r[i * 3 + 1] * ray[1] + r[i * 3 + 2] * ray[2]) as f64; // R^T v
            }
            let tt = s.hit(&oc, &vd);
            if tt <= 0.0 {
                continue;
            }
            let p = [oc[0] + tt * vd[0], oc[1] + tt * vd[1], oc[2] + tt * vd[2]];
            if p[2] <= 0.0 {
                continue;
            }
            // 깊이 카메라에서 보이는지(같은 점까지 막힘 없음)
            let v2 = [p[0] / p[2], p[1] / p[2], 1.0];
            let t2 = s.hit(&[0.0; 3], &v2);
            if t2 <= 0.0 || t2 < p[2] - 1e-4 {
                continue;
            }
            // 깊이 영상 안에 들어오는지(화소 중심 기준 반 화소까지)
            let ud = di.fx as f64 * v2[0] + di.ppx as f64;
            let vdd = di.fy as f64 * v2[1] + di.ppy as f64;
            if ud < -0.5 || vdd < -0.5 || ud >= di.width as f64 - 0.5 || vdd >= di.height as f64 - 0.5 {
                continue;
            }
            out[(y * ci.width + x) as usize] = p[2] as f32;
        }
    }
    out
}

/// 무작위 깊이(겹침·경계 최악 조건): 덩어리 + 잡음 + 구멍
pub fn random_depth(di: &Intrinsics, seed: u32) -> Vec<u16> {
    let mut st = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    (0..(di.width * di.height) as usize)
        .map(|_| {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            let r = st;
            if r % 17 == 0 { 0 } else { (250 + (r >> 8) % 12000) as u16 }
        })
        .collect()
}
