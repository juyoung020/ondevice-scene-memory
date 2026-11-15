//! 빠른 위치 추적(기본 경로, Odom/Strategy=9): 장면 사이 광류(피라미드 Lucas-Kanade)로 점을 따라가고 PnP 로 위치를 구한다.
//!   1) 등속 예측: 지난 속도로 이번 위치를 미리 짐작하고, 지도 점을 그 위치로 투영해 광류 시작점으로 씀(빠르게 움직여도 찾는 범위가 작음)
//!   2) 광류: 앞 장면의 점 위치 → 이번 장면(4 층 피라미드, 11×11 창). 기술자 계산·짝짓기 없음
//!   3) PnP RANSAC(EPnP 6 점) + 안쪽 점 가우스-뉴턴 다듬기(Huber). 장면마다 묶음 조정(BA) 없음
//!   4) 따라가는 점이 줄면(키프레임) 그 장면에서만 GFTT 로 빈 자리를 채우고 깊이로 3D 점을 만든다
//!   5) 예측으로 실패하면 '움직임 없음' 시작점으로 한 번 더. 그래도 실패면 놓침(마지막 성공 장면·점을 그대로 둬 다음 장면이 거기서 다시 시도)
//! 지도 점은 한 번 만든 뒤 바꾸지 않는다(세계 좌표, 몸체 축). 루프 클로저·지도용 특징(BRIEF)은 지도 노드 장면에서만 따로 뽑는다(app::Engine).
use crate::features::{keypoints_3d, Extractor};
use crate::register::RegistrationVis;
use crate::odom::OdomResult;
use crate::params::Params;
use crate::register::rtf::Tf;
use crate::register::pnp_ransac_k;
use crate::{Features, Frame, KeyPoint};
use nalgebra::{Matrix6, Rotation3, Vector3, Vector6};
use rayon::prelude::*;

const LEVELS: usize = 4;
const HW: i32 = 5; // 창 반쪽(11×11)
const ITERS: usize = 10;

/// 회색조 피라미드(층마다 가로·세로 절반, 2×2 평균)
pub struct Pyr {
    pub lv: Vec<(Vec<u8>, usize, usize)>,
}

impl Pyr {
    pub fn new(gray: &[u8], w: usize, h: usize) -> Pyr {
        let mut p = Pyr { lv: Vec::new() };
        p.fill(gray, w, h);
        p
    }

    /// 버퍼를 다시 써서 채움(장면마다 새로 잡지 않음). 0 층은 회색조를 그대로 복사(앞 장면으로 남겨야 해서)
    pub fn fill(&mut self, gray: &[u8], w: usize, h: usize) {
        self.lv.resize_with(LEVELS, || (Vec::new(), 0, 0));
        self.lv[0].0.clear();
        self.lv[0].0.extend_from_slice(gray);
        self.lv[0].1 = w;
        self.lv[0].2 = h;
        for l in 1..LEVELS {
            let (a, b) = self.lv.split_at_mut(l);
            let (src, sw, _) = (&a[l - 1].0, a[l - 1].1, a[l - 1].2);
            let (nw, nh) = (sw / 2, a[l - 1].2 / 2);
            let d = &mut b[0];
            d.0.resize(nw * nh, 0);
            d.1 = nw;
            d.2 = nh;
            for (y, row) in d.0.chunks_exact_mut(nw).enumerate() {
                let r0 = &src[(2 * y) * sw..(2 * y) * sw + 2 * nw];
                let r1 = &src[(2 * y + 1) * sw..(2 * y + 1) * sw + 2 * nw];
                for x in 0..nw {
                    let s = r0[2 * x] as u32 + r0[2 * x + 1] as u32 + r1[2 * x] as u32 + r1[2 * x + 1] as u32;
                    row[x] = ((s + 2) >> 2) as u8;
                }
            }
        }
    }
}

/// 한 점 광류: prev 의 p0 → cur, 시작점 init(0 층 화소). 실패면 None
fn track_one(prev: &Pyr, cur: &Pyr, p0: [f32; 2], init: [f32; 2]) -> Option<[f32; 2]> {
    const N: usize = (2 * HW + 3) as usize; // 기울기용 1 칸 테두리
    const M: usize = (2 * HW + 1) as usize;
    let mut tpl = [0f32; N * N];
    let mut gx = [0f32; M * M];
    let mut gy = [0f32; M * M];
    let top = (LEVELS - 1) as i32;
    let sc = 1.0 / (1 << top) as f32;
    let mut g = [(init[0] - p0[0]) * sc, (init[1] - p0[1]) * sc];
    for l in (0..LEVELS).rev() {
        let s = 1.0 / (1 << l) as f32;
        let (pi, w, h) = (&prev.lv[l].0, prev.lv[l].1, prev.lv[l].2);
        let ci = &cur.lv[l].0;
        let (px, py) = (p0[0] * s, p0[1] * s);
        let lim = |x: f32, y: f32| x >= (HW + 1) as f32 && y >= (HW + 1) as f32 && x < (w as i32 - HW - 2) as f32 && y < (h as i32 - HW - 2) as f32;
        if !lim(px, py) {
            if l == 0 {
                return None;
            }
            g = [2.0 * g[0], 2.0 * g[1]];
            continue;
        }
        // 앞 장면 창(테두리 포함)을 같은 이중 선형 가중치로
        let (ix, iy) = (px.floor(), py.floor());
        let (ax, ay) = (px - ix, py - iy);
        let (w00, w01, w10, w11) = ((1.0 - ax) * (1.0 - ay), ax * (1.0 - ay), (1.0 - ax) * ay, ax * ay);
        let (bx, by) = (ix as i32 - HW - 1, iy as i32 - HW - 1);
        for j in 0..N {
            let r = ((by + j as i32) as usize) * w + bx as usize;
            for i in 0..N {
                let o = r + i;
                tpl[j * N + i] = w00 * pi[o] as f32 + w01 * pi[o + 1] as f32 + w10 * pi[o + w] as f32 + w11 * pi[o + w + 1] as f32;
            }
        }
        let (mut a11, mut a12, mut a22) = (0f32, 0f32, 0f32);
        for j in 0..M {
            for i in 0..M {
                let c = (j + 1) * N + i + 1;
                let dx = 0.5 * (tpl[c + 1] - tpl[c - 1]);
                let dy = 0.5 * (tpl[c + N] - tpl[c - N]);
                gx[j * M + i] = dx;
                gy[j * M + i] = dy;
                a11 += dx * dx;
                a12 += dx * dy;
                a22 += dy * dy;
            }
        }
        let det = a11 * a22 - a12 * a12;
        let mn = (M * M) as f32;
        // 최소 고윳값 / 화소 수 가 너무 작으면(밋밋한 곳) 못 따라감
        let min_eig = (a11 + a22 - ((a11 - a22) * (a11 - a22) + 4.0 * a12 * a12).sqrt()) * 0.5 / mn;
        if det.abs() < 1e-6 || min_eig < 1e-2 {
            return None;
        }
        let inv = 1.0 / det;
        let mut d = [0f32; 2];
        for _ in 0..ITERS {
            let (qx, qy) = (px + g[0] + d[0], py + g[1] + d[1]);
            if !(qx >= HW as f32 && qy >= HW as f32 && qx < (w as i32 - HW - 1) as f32 && qy < (h as i32 - HW - 1) as f32) {
                return None;
            }
            let (jx, jy) = (qx.floor(), qy.floor());
            let (cx, cy) = (qx - jx, qy - jy);
            let (v00, v01, v10, v11) = ((1.0 - cx) * (1.0 - cy), cx * (1.0 - cy), (1.0 - cx) * cy, cx * cy);
            let (ox, oy) = (jx as i32 - HW, jy as i32 - HW);
            let (mut b1, mut b2) = (0f32, 0f32);
            for j in 0..M {
                let r = ((oy + j as i32) as usize) * w + ox as usize;
                for i in 0..M {
                    let o = r + i;
                    let v = v00 * ci[o] as f32 + v01 * ci[o + 1] as f32 + v10 * ci[o + w] as f32 + v11 * ci[o + w + 1] as f32;
                    let e = tpl[(j + 1) * N + i + 1] - v;
                    b1 += e * gx[j * M + i];
                    b2 += e * gy[j * M + i];
                }
            }
            let dx = inv * (a22 * b1 - a12 * b2);
            let dy = inv * (a11 * b2 - a12 * b1);
            d[0] += dx;
            d[1] += dy;
            if dx * dx + dy * dy < 1e-4 {
                break;
            }
        }
        if l > 0 {
            g = [2.0 * (g[0] + d[0]), 2.0 * (g[1] + d[1])];
        } else {
            let r = [p0[0] + g[0] + d[0], p0[1] + g[1] + d[1]];
            return Some(r);
        }
    }
    None
}

#[derive(Clone)]
struct Track {
    w: [f32; 3],
    uv: [f32; 2],
}

/// 세계(몸체 축) 점 → 화소. tcw = 세계 → 광학 축
fn project(tcw: &Tf, k: &[f64; 4], p: [f32; 3]) -> Option<[f32; 2]> {
    let c = tcw.transform_point(p);
    if c[2] <= 0.05 {
        return None;
    }
    Some([(k[0] * c[0] as f64 / c[2] as f64 + k[2]) as f32, (k[1] * c[1] as f64 / c[2] as f64 + k[3]) as f32])
}

fn tf_from_rt(r: &Rotation3<f64>, t: &Vector3<f64>) -> Tf {
    let m = r.matrix();
    Tf([
        m[(0, 0)] as f32, m[(0, 1)] as f32, m[(0, 2)] as f32, t[0] as f32,
        m[(1, 0)] as f32, m[(1, 1)] as f32, m[(1, 2)] as f32, t[1] as f32,
        m[(2, 0)] as f32, m[(2, 1)] as f32, m[(2, 2)] as f32, t[2] as f32,
    ])
}

fn rt_from_tf(t: &Tf) -> (Rotation3<f64>, Vector3<f64>) {
    let m = nalgebra::Matrix3::from_fn(|r, c| t.at(r, c) as f64);
    (Rotation3::from_matrix(&m), Vector3::new(t.x() as f64, t.y() as f64, t.z() as f64))
}

/// 재투영 오차 가우스-뉴턴(Huber), tcw(세계 → 광학) 다듬기
fn refine(tcw: &Tf, k: &[f64; 4], w: &[[f32; 3]], uv: &[[f32; 2]], idx: &[usize], huber: f64) -> Tf {
    let (mut r, mut t) = rt_from_tf(tcw);
    for _ in 0..6 {
        let mut h = Matrix6::<f64>::zeros();
        let mut b = Vector6::<f64>::zeros();
        for &i in idx {
            let p = Vector3::new(w[i][0] as f64, w[i][1] as f64, w[i][2] as f64);
            let c = r * p + t;
            if c[2] <= 0.05 {
                continue;
            }
            let iz = 1.0 / c[2];
            let u = k[0] * c[0] * iz + k[2];
            let v = k[1] * c[1] * iz + k[3];
            let e = [u - uv[i][0] as f64, v - uv[i][1] as f64];
            let n = (e[0] * e[0] + e[1] * e[1]).sqrt();
            let wt = if n > huber { huber / n } else { 1.0 };
            // d(u,v)/d c
            let du = [k[0] * iz, 0.0, -k[0] * c[0] * iz * iz];
            let dv = [0.0, k[1] * iz, -k[1] * c[1] * iz * iz];
            // d c / d[ω, v] = [-[c]x, I]
            let jc = |d: [f64; 3]| -> Vector6<f64> {
                // d · (-[c]x) = (c × d)ᵀ ... 행 벡터 d 곱하기 -[c]x = (d × c)?: -dᵀ[c]x = (c × d)ᵀ
                let cr = Vector3::new(d[0], d[1], d[2]);
                let a = c.cross(&cr);
                Vector6::new(a[0], a[1], a[2], d[0], d[1], d[2])
            };
            let ju = jc(du);
            let jv = jc(dv);
            h += wt * (ju * ju.transpose() + jv * jv.transpose());
            b -= wt * (ju * e[0] + jv * e[1]);
        }
        let Some(dx) = h.cholesky().map(|c| c.solve(&b)) else { break };
        let dr = Rotation3::new(Vector3::new(dx[0], dx[1], dx[2]));
        r = dr * r;
        t = dr * t + Vector3::new(dx[3], dx[4], dx[5]);
        if dx.norm() < 1e-7 {
            break;
        }
    }
    tf_from_rt(&r, &t)
}

/// 깊이 성긴 격자(SUB 화소마다 한 점, mm). 깊이 ICP 의 앞 장면 정보: 영상 전체를 들고 있지 않고 이것만 다시 씀
const SUB: usize = 4;

fn sub_depth(depth: &[u16], w: usize, h: usize, out: &mut Vec<u16>) -> usize {
    let (ws, hs) = (w / SUB, h / SUB);
    out.clear();
    out.reserve(ws * hs);
    let mut n = 0;
    for j in 0..hs {
        let row = &depth[(j * SUB + SUB / 2) * w..];
        for i in 0..ws {
            let d = row[i * SUB + SUB / 2];
            let d = if d == u16::MAX { 0 } else { d };
            n += (d > 0) as usize;
            out.push(d);
        }
    }
    n
}

/// 격자 칸 (i, j) 의 광학 축 3D 점(m)
#[inline]
fn vtx(d: &[u16], ws: usize, i: usize, j: usize, k: &[f64; 4]) -> Option<Vector3<f64>> {
    let z = d[j * ws + i];
    if z == 0 || z > 6000 {
        return None;
    }
    let z = z as f64 * 1e-3;
    let (u, v) = ((i * SUB + SUB / 2) as f64, (j * SUB + SUB / 2) as f64);
    Some(Vector3::new((u - k[2]) * z / k[0], (v - k[3]) * z / k[1], z))
}

/// 점-평면 ICP(투영 짝짓기): 이번 깊이 격자 cur 를 앞 깊이 격자 prev 에 맞춤. t_init·결과 = 이번 광학 → 앞 광학
fn icp(prev: &[u16], cur: &[u16], ws: usize, hs: usize, k: &[f64; 4], t_init: &Tf) -> Option<Tf> {
    icp_q(prev, cur, ws, hs, k, t_init, 0.25, 1e-3)
}

/// min_ratio: 짝이 된 점 비율 하한, min_eig: 6 방향 구속(정규화 최소 고윳값) 하한
fn icp_q(prev: &[u16], cur: &[u16], ws: usize, hs: usize, k: &[f64; 4], t_init: &Tf, min_ratio: f64, min_eig: f64) -> Option<Tf> {
    let (mut r, mut t) = rt_from_tf(t_init);
    // 앞 장면 법선(이웃 격자 차의 외적)
    let normal = |i: usize, j: usize| -> Option<(Vector3<f64>, Vector3<f64>)> {
        if i == 0 || j == 0 || i + 1 >= ws || j + 1 >= hs {
            return None;
        }
        let c = vtx(prev, ws, i, j, k)?;
        let a = vtx(prev, ws, i + 1, j, k)? - vtx(prev, ws, i - 1, j, k)?;
        let b = vtx(prev, ws, i, j + 1, k)? - vtx(prev, ws, i, j - 1, k)?;
        if a.norm() > 0.1 * c[2] || b.norm() > 0.1 * c[2] {
            return None; // 깊이 끊김
        }
        let n = a.cross(&b);
        let l = n.norm();
        (l > 1e-12).then(|| (c, n / l))
    };
    let mut pts = Vec::with_capacity(ws * hs / 4);
    for j in (1..hs - 1).step_by(2) {
        for i in (1..ws - 1).step_by(2) {
            if let Some(p) = vtx(cur, ws, i, j, k) {
                pts.push(p);
            }
        }
    }
    if pts.len() < 200 {
        return None;
    }
    // 잔차 상한: 스테레오 깊이 잡음 ≈ 0.0042·z²(D435, 기준선 5 cm·초점 383 px·부화소 0.08)의 2.5 배, 1–4 cm
    let mut zs: Vec<f64> = pts.iter().map(|p| p[2]).collect();
    let zm = *zs.select_nth_unstable_by(pts.len() / 2, |a, b| a.total_cmp(b)).1;
    let rms_max = (2.5 * 0.0042 * zm * zm).clamp(0.01, 0.04);
    let mut good = 0usize;
    let mut rms = 0.0;
    let mut h = Matrix6::<f64>::zeros();
    for it in 0..15 {
        let max_d = if it < 5 { 0.15 } else { 0.05 };
        h = Matrix6::<f64>::zeros();
        let mut b = Vector6::<f64>::zeros();
        good = 0;
        let mut se = 0.0;
        for p in &pts {
            let q = r * p + t;
            if q[2] <= 0.1 {
                continue;
            }
            let u = k[0] * q[0] / q[2] + k[2];
            let v = k[1] * q[1] / q[2] + k[3];
            if u < 0.0 || v < 0.0 {
                continue;
            }
            let (i, j) = ((u / SUB as f64) as usize, (v / SUB as f64) as usize);
            if i >= ws || j >= hs {
                continue;
            }
            let Some((vp, n)) = normal(i, j) else { continue };
            let d = q - vp;
            if d.norm() > max_d {
                continue;
            }
            let res = n.dot(&d);
            let wt = if res.abs() > 0.02 { 0.02 / res.abs() } else { 1.0 };
            let a = q.cross(&n);
            let jr = Vector6::new(a[0], a[1], a[2], n[0], n[1], n[2]);
            h += wt * jr * jr.transpose();
            b -= wt * jr * res;
            good += 1;
            se += res * res;
        }
        if good < 100 {
            return None;
        }
        rms = (se / good as f64).sqrt();
        let dx = h.cholesky().map(|c| c.solve(&b))?;
        let dr = Rotation3::new(Vector3::new(dx[0], dx[1], dx[2]));
        r = dr * r;
        t = dr * t + Vector3::new(dx[3], dx[4], dx[5]);
        if dx.norm() < 1e-6 {
            break;
        }
    }
    // 받아들임: 짝 25 % 이상, 잔차 3 cm 아래(D435 깊이 잡음), 6 방향 모두 잡힘(평평한 벽 하나면 미끄러짐)
    let eig = h.symmetric_eigen().eigenvalues;
    let min_e = eig.iter().cloned().fold(f64::MAX, f64::min) / good.max(1) as f64;
    if std::env::var("KLT_PROF").is_ok() {
        eprintln!("icp good {good}/{} rms {rms:.4} min_e {min_e:.2e}", pts.len());
    }
    if (good as f64) < min_ratio * pts.len() as f64 || rms > rms_max || min_e < min_eig {
        return None;
    }
    Some(tf_from_rt(&r, &t))
}

/// 일꾼 스레드에 넘기는 키프레임 기술자 일
struct DescJob {
    gray: Vec<u8>,
    depth: crate::DepthBuf,
    cam: crate::Camera,
    kps: Vec<KeyPoint>,
    pose: Tf,
    min_depth: f32,
    max_depth: f32,
}

/// 광류 한 장의 질(위치 질 표시·통계용)
#[derive(Clone, Copy, Debug, Default)]
pub struct Quality {
    pub inliers: usize,
    pub ratio: f32,
    pub rms: f32,
    pub spread: f32,
}

/// 이번 장면 위치를 어떻게 냈나: 0 광류, 1 깊이 ICP, 2 기술자 되찾기, 3 첫 장면, 4 놓침
#[derive(Clone, Copy, Debug, Default)]
pub struct Diag {
    pub method: u8,
    pub q: Quality,
    /// 등속 예측과 낸 위치 사이 거리(m), 예측 없으면 -1
    pub jump: f32,
}

/// 영상이 낸 장면 사이 움직임(transform, 몸체)의 회전이 IMU 자이로 회전과 5° + 회전량의 10 % 넘게 다른지. IMU 회전이 없으면 false
fn imu_disagree(imu_rot: &Option<crate::simd::M34>, transform: &Tf) -> bool {
    let Some(r) = imu_rot else { return false };
    let e = crate::imu::so3_log(&crate::simd::m34_mul(&crate::simd::m34_t(r), &crate::imu::m34_of_tf(transform)));
    let w = crate::imu::so3_log(r);
    let n = |v: [f32; 4]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().to_degrees();
    n(e) > 5.0 + 0.1 * n(w)
}

fn jump_of(pred: &Option<Tf>, pose: &Tf) -> f32 {
    match pred {
        Some(p) => {
            let (a, b) = (p, pose);
            ((a.x() - b.x()).powi(2) + (a.y() - b.y()).powi(2) + (a.z() - b.z()).powi(2)).sqrt()
        }
        None => -1.0,
    }
}

pub struct OdometryKlt {
    pose: Tf,
    /// 마지막 성공 장면의 회색조 피라미드·시각
    prev: Option<Pyr>,
    /// 마지막 키프레임 특징과 그때 위치(광류가 끊겼을 때 되찾기)
    kf: Option<(Features, Tf)>,
    kf_t: f64,
    desc_tx: Option<std::sync::mpsc::Sender<DescJob>>,
    desc_rx: Option<std::sync::mpsc::Receiver<(Features, Tf)>>,
    /// 되찾기용 기술자를 새로 뽑는 최소 간격(초)
    desc_every: f64,
    desc_every_kf: u32,
    kf_count: u32,
    ext: Extractor,
    reg: RegistrationVis,
    /// 다음 장면 피라미드에 다시 쓸 버퍼
    spare: Option<Pyr>,
    prev_t: f64,
    tracks: Vec<Track>,
    /// 몸체 축 속도(xyz rpy / 초). NULL = 모름
    velocity: Tf,
    max_features: i32,
    min_distance: f64,
    min_depth: f32,
    max_depth: f32,
    min_inliers: usize,
    reproj: f32,
    iterations: i32,
    redetect: f32,
    /// 다듬은 위치로 재투영 오차가 이 화소 안이면 점을 계속 따라감
    keep_err: f32,
    /// 광류를 여러 스레드로(rayon)
    par: bool,
    /// 기술자 맞추기로 되찾은 장면 수
    pub recovered: usize,
    /// 마지막 장면 진단
    pub diag: Diag,
    /// 약한 광류 결과는 받지 않음(OdomKlt/Strict, 기본 켬)
    strict: bool,
    min_good: usize,
    /// 깊이 ICP 로 이어 간 장면 수
    pub by_icp: usize,
    /// 다시 시작한 첫 장면이 마지막 성공 장면 깊이와 ICP 로 맞았는지(이어 붙인 자리가 확인됨)
    pub bridge_verified: bool,
    pub bridges_verified: usize,
    /// 시험용: 광류 없이 깊이 ICP 만(OdomKlt/DepthOnly)
    depth_only: bool,
    /// 마지막 성공 장면의 깊이 격자(깊이 ICP)
    prev_sub: Vec<u16>,
    cur_sub: Vec<u16>,
    /// IMU 융합(None = 카메라 전용, 결과가 IMU 없는 판과 비트까지 같음)
    pub imu: Option<crate::imu::Fusion>,
    /// 다음 첫 장면을 중력으로 수평 맞춤(처음 · 원점 리셋 뒤)
    level_pending: bool,
    /// 지금 지도가 중력으로 수평 맞춰져 있음(지도 z = 위)
    pub gravity_aligned: bool,
    /// IMU 회전으로 이어 붙인 장면 수
    pub by_imu: usize,
    /// 이번 장면에서 영상 결과를 IMU 회전과 어긋나 버렸는지 · 그런 장면 수
    pub imu_rejected: bool,
}

impl OdometryKlt {
    pub fn new(p: &Params) -> OdometryKlt {
        OdometryKlt {
            pose: Tf::IDENTITY,
            prev: None,
            kf: None,
            kf_t: 0.0,
            desc_tx: None,
            desc_rx: None,
            desc_every: p.get("OdomKlt/DescEvery", 0.0f64),
            desc_every_kf: p.get("OdomKlt/DescEveryKf", 1u32),
            kf_count: 0,
            ext: Extractor::new(p),
            reg: RegistrationVis::new(p),
            spare: None,
            prev_t: 0.0,
            tracks: Vec::new(),
            velocity: Tf::NULL,
            max_features: p.get("Vis/MaxFeatures", 400i32),
            min_distance: p.get("GFTT/MinDistance", 7.0),
            min_depth: p.get("Vis/MinDepth", 0.0f32),
            max_depth: p.get("Vis/MaxDepth", 0.0f32),
            min_inliers: p.get("Vis/MinInliers", 20i32).max(6) as usize,
            reproj: p.get("Vis/PnPReprojError", 2.0f32),
            iterations: p.get("OdomKlt/Iterations", 100i32),
            redetect: p.get("OdomKlt/RedetectRatio", 0.6f32),
            keep_err: p.get("OdomKlt/KeepError", 3.0f32),
            recovered: 0,
            diag: Diag::default(),
            strict: p.get_bool("OdomKlt/Strict", true),
            min_good: p.get("OdomKlt/MinGood", 30usize),
            by_icp: 0,
            bridge_verified: false,
            bridges_verified: 0,
            depth_only: p.get_bool("OdomKlt/DepthOnly", false),
            prev_sub: Vec::new(),
            cur_sub: Vec::new(),
            imu: None,
            level_pending: true,
            gravity_aligned: false,
            by_imu: 0,
            imu_rejected: false,
            par: p.get_bool("OdomKlt/Parallel", false), // 기본 한 스레드: 전체 프로그램에선 rayon 풀을 memmap 이 같이 써서 기다림이 길어짐(보드 walk1 밀려 버린 장면 284 → 150)
        }
    }

    pub fn pose_tf(&self) -> Tf {
        self.pose
    }

    /// 이 위치에서 처음부터(다음 장면이 첫 장면)
    pub fn reset_tf(&mut self, pose: Tf) {
        // 같은 자리에서 다시 시작(이어 붙이기)이면 마지막 성공 장면 깊이를 남겨, 첫 장면에서 그 깊이와 ICP 로 실제 움직임을 잡는다
        let same_place = pose == self.pose;
        // 원점에서 처음부터(리셋)면 다음 첫 장면을 다시 중력으로 수평 맞춤
        if pose == Tf::IDENTITY {
            self.level_pending = true;
            self.gravity_aligned = false;
        }
        self.pose = pose;
        self.prev = None;
        self.tracks.clear();
        self.kf = None;
        if !same_place {
            self.prev_sub.clear();
        }
        self.velocity = Tf::NULL;
        self.prev_t = 0.0;
    }

    pub fn tracks(&self) -> usize {
        self.tracks.len()
    }

    /// 키프레임: 이 장면 특징(GFTT·BRIEF·3D)을 뽑아 되찾기용으로 두고, 따라가는 점 둘레(min_distance 칸)가 빈 곳의 점을 더함
    fn top_up(&mut self, f: &Frame) {
        // 되찾기용 특징(BRIEF 기술자)은 0.5 초에 한 번만 새로 뽑음: 그 사이 키프레임은 GFTT·3D 점만(장면당 시간 줄임)
        // OdomKlt/DescEvery 초(기본 0 = 키프레임마다, 결과가 시간과 무관하게 같음)
        // 또는 키프레임 N 번에 한 번(OdomKlt/DescEveryKf, 장면 수로 정해 결과가 시간과 무관)
        self.kf_count += 1;
        let full = self.kf.as_ref().is_none_or(|_| f.t - self.kf_t >= self.desc_every && self.kf_count >= self.desc_every_kf);
        if full {
            self.kf_count = 0;
        }
        let feats = if full && self.desc_tx.is_some() {
            // 기술자는 일꾼 스레드에서(되찾기에만 쓰여 한두 장 늦어도 됨): 여기선 GFTT·3D 점만, 회색조 사본과 깊이(나눠 듦)를 넘김
            let kps = self.ext.detect(f);
            let pts3d = self.ext.keypoints_3d(f, &kps);
            let _ = self.desc_tx.as_ref().unwrap().send(DescJob {
                gray: f.gray.clone(),
                depth: f.depth_mm.clone(),
                cam: f.cam.clone(),
                kps: kps.clone(),
                pose: self.pose,
                min_depth: self.min_depth,
                max_depth: self.max_depth,
            });
            self.kf_t = f.t;
            Features { kps, pts3d, desc: Vec::new() }
        } else if full {
            self.ext.extract(f)
        } else {
            let kps = self.ext.detect(f);
            let pts3d = self.ext.keypoints_3d(f, &kps);
            Features { kps, pts3d, desc: Vec::new() }
        };
        let (w, h) = (f.cam.width as usize, f.cam.height as usize);
        let cs = self.min_distance.max(1.0) as f32;
        let (gw, gh) = ((w as f32 / cs) as usize + 1, (h as f32 / cs) as usize + 1);
        let mut occ = vec![false; gw * gh];
        for t in &self.tracks {
            let (gx, gy) = ((t.uv[0] / cs) as usize, (t.uv[1] / cs) as usize);
            if gx < gw && gy < gh {
                occ[gy * gw + gx] = true;
            }
        }
        for (k, p) in feats.kps.iter().zip(feats.pts3d.iter()) {
            if self.tracks.len() >= self.max_features as usize {
                break;
            }
            let (gx, gy) = ((k.x / cs) as usize, (k.y / cs) as usize);
            if !p.x.is_finite() || gx >= gw || gy >= gh || occ[gy * gw + gx] {
                continue;
            }
            occ[gy * gw + gx] = true;
            let wp = self.pose.transform_point([p.x, p.y, p.z]);
            self.tracks.push(Track { w: wp, uv: [k.x, k.y] });
        }
        if full && self.desc_tx.is_none() {
            self.kf = Some((feats, self.pose));
            self.kf_t = f.t;
        }
    }

    /// 실시간 입력: 키프레임 기술자를 일꾼 스레드에서 뽑음(재생은 결정적이게 그 자리에서)
    pub fn set_async_desc(&mut self) {
        if self.desc_tx.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel::<DescJob>();
        let (rtx, rrx) = std::sync::mpsc::channel::<(Features, Tf)>();
        std::thread::spawn(move || {
            common::threads::apply(5, 0); // 위치 추적보다 뒤
            while let Ok(mut j) = rx.recv() {
                // 쌓였으면 가장 최근 것만
                while let Ok(n) = rx.try_recv() {
                    j = n;
                }
                let (w, h) = (j.cam.width as usize, j.cam.height as usize);
                let mut kps = j.kps;
                let desc = crate::features::describe(&j.gray, w, h, &mut kps);
                let pts3d = keypoints_3d(&j.depth, &j.cam, &kps, j.min_depth, j.max_depth);
                if rtx.send((Features { kps, pts3d, desc }, j.pose)).is_err() {
                    break;
                }
            }
        });
        self.desc_tx = Some(tx);
        self.desc_rx = Some(rrx);
    }

    /// 광류가 모두 실패했을 때: 마지막 키프레임 특징과 이번 장면 특징을 기술자로 맞춤(RTAB-Map 맞추기 그대로). 성공하면 점을 이 장면에서 새로
    fn recover(&mut self, f: &Frame, pred: Option<Tf>) -> Option<(Tf, [[f64; 6]; 6], usize)> {
        let (kff, kfp) = self.kf.as_ref()?;
        let kfp = *kfp;
        let cf = self.ext.extract(f);
        let g = pred.map(|p| kfp.inverse().mul(&p)).unwrap_or(Tf::NULL);
        let mut info = self.reg.compute_tf(kff, &cf, &f.cam, g);
        if info.tf.is_null() && !g.is_null() {
            info = self.reg.compute_tf(kff, &cf, &f.cam, Tf::NULL);
        }
        if info.tf.is_null() {
            return None;
        }
        let pose = kfp.mul(&info.tf);
        self.tracks.clear();
        for (k, p) in cf.kps.iter().zip(cf.pts3d.iter()) {
            if p.x.is_finite() {
                self.tracks.push(Track { w: pose.transform_point([p.x, p.y, p.z]), uv: [k.x, k.y] });
            }
        }
        self.kf = Some((cf, pose));
        Some((pose, info.covariance, info.inliers))
    }

    fn eye(v: f64) -> [[f64; 6]; 6] {
        let mut m = [[0.0; 6]; 6];
        for (i, r) in m.iter_mut().enumerate() {
            r[i] = v;
        }
        m
    }

    fn result(&self, transform: Tf, cov: [[f64; 6]; 6], inliers: usize, matches: usize, kf: bool) -> OdomResult {
        let lost = transform.is_null();
        OdomResult {
            pose: if lost { None } else { Some(self.pose.to_pose()) },
            tf: if lost { Tf::NULL } else { self.pose },
            lost,
            covariance: cov,
            inliers,
            matches,
            keyframe_updated: kf,
            transform,
            features: Features::default(),
            reg: None,
        }
    }

    /// 한 번 시도: 시작점들 → 광류 → PnP → 다듬기. (새 위치, 안쪽 번호, 이번 화소)
    /// need: 받아들일 최소 안쪽 점 수(빈틈 뒤엔 더 엄격하게)
    fn attempt(&self, prev: &Pyr, cur: &Pyr, inits: &[[f32; 2]], k4: &[f64; 4], local: &Tf, need: usize) -> Option<(Tf, Vec<usize>, Vec<[f32; 2]>, Quality)> {
        let ta = std::time::Instant::now();
        let res: Vec<Option<[f32; 2]>> = if self.par {
            self.tracks.par_iter().zip(inits.par_iter()).map(|(t, &i)| track_one(prev, cur, t.uv, i)).collect()
        } else {
            self.tracks.iter().zip(inits.iter()).map(|(t, &i)| track_one(prev, cur, t.uv, i)).collect()
        };
        if std::env::var("KLT_PROF").is_ok() {
            eprintln!("klt flow {:.2} ms", ta.elapsed().as_secs_f64() * 1e3);
        }
        let mut ids = Vec::with_capacity(res.len());
        let mut op = Vec::with_capacity(res.len());
        let mut ip = Vec::with_capacity(res.len());
        for (i, r) in res.iter().enumerate() {
            if let Some(q) = r {
                ids.push(i);
                op.push(self.tracks[i].w);
                ip.push(*q);
            }
        }
        if op.len() < self.min_inliers {
            return None;
        }
        let r = pnp_ransac_k(&op, &ip, k4, self.iterations, self.reproj)?;
        if r.1.len() < self.min_inliers {
            return None;
        }
        // 다듬은 뒤 안쪽 점을 다시 고름
        let tcw = refine(&r.0, k4, &op, &ip, &r.1, self.reproj as f64);
        let thr = self.keep_err * self.keep_err;
        let inl: Vec<usize> = (0..op.len())
            .filter(|&i| project(&tcw, k4, op[i]).map(|p| (p[0] - ip[i][0]).powi(2) + (p[1] - ip[i][1]).powi(2) <= thr).unwrap_or(false))
            .collect();
        if inl.len() < need {
            return None;
        }
        let tcw = refine(&tcw, k4, &op, &ip, &inl, self.reproj as f64);
        // 질: 재투영 RMS, 안쪽 비율, 안쪽 점이 퍼진 넓이(화면 대비). 약하면 받지 않고 되찾기·놓침으로(어긋난 자리에 붙어 이어 가지 않게)
        let mut se = 0.0f64;
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &i in &inl {
            if let Some(p) = project(&tcw, k4, op[i]) {
                se += ((p[0] - ip[i][0]).powi(2) + (p[1] - ip[i][1]).powi(2)) as f64;
            }
            x0 = x0.min(ip[i][0]);
            y0 = y0.min(ip[i][1]);
            x1 = x1.max(ip[i][0]);
            y1 = y1.max(ip[i][1]);
        }
        let rms = (se / inl.len().max(1) as f64).sqrt() as f32;
        let ratio = inl.len() as f32 / op.len().max(1) as f32;
        let spread = ((x1 - x0).max(0.0) * (y1 - y0).max(0.0)) / ((2.0 * k4[2]) * (2.0 * k4[3])) as f32;
        let q = Quality { inliers: inl.len(), ratio, rms, spread };
        if self.strict && (inl.len() < self.min_good || ratio < 0.3 || spread < 0.08 || rms > 2.0) {
            return None;
        }
        let pose = local.mul(&tcw).inverse();
        let uvs: Vec<[f32; 2]> = ip.clone();
        let ids_in: Vec<usize> = inl.iter().map(|&i| ids[i]).collect();
        let uv_in: Vec<[f32; 2]> = inl.iter().map(|&i| uvs[i]).collect();
        Some((pose, ids_in, uv_in, q))
    }

    pub fn process(&mut self, f: &Frame) -> OdomResult {
        self.imu_rejected = false;
        let (w, h) = (f.cam.width as usize, f.cam.height as usize);
        let t0 = std::time::Instant::now();
        if let Some(rx) = &self.desc_rx {
            while let Ok(kf) = rx.try_recv() {
                self.kf = Some(kf);
            }
        }
        let mut cur = self.spare.take().unwrap_or(Pyr { lv: Vec::new() });
        cur.fill(&f.gray, w, h);
        let t_pyr = t0.elapsed().as_secs_f64();
        let local = Tf::from_pose(&f.cam.local);
        let k4 = [f.cam.fx, f.cam.fy, f.cam.cx, f.cam.cy];
        let Some(prev) = self.prev.take() else {
            // 첫 장면: 점만 만들고 '새로 시작'(공분산 9999)
            self.tracks.clear();
            // 이어 붙이기: 마지막 성공 장면 깊이가 남아 있으면 움직임 없음에서 시작하는 ICP 로 그사이 움직임을 바로잡음
            self.bridge_verified = false;
            // IMU: 원점에서 시작하는 첫 장면이면 지도 z 를 중력 위로(가장 작은 회전, 방향은 그대로)
            if let Some(u) = self.imu.as_mut() {
                u.poll(&local, f.t, f.t);
                if std::mem::take(&mut self.level_pending) && self.pose == Tf::IDENTITY {
                    if let Some(up) = u.up_body() {
                        self.pose = crate::imu::tf_of(&crate::imu::rot_between(&up, &[0.0, 0.0, 1.0, 0.0]), [0.0; 3]);
                        self.gravity_aligned = true;
                        common::info!("IMU: 첫 장면을 중력으로 수평 맞춤({:.1}° 기울어 있었음)", up[2].clamp(-1.0, 1.0).acos().to_degrees());
                    }
                }
            }
            if !self.prev_sub.is_empty() && !f.depth_mm.is_empty() && sub_depth(&f.depth_mm, w, h, &mut self.cur_sub) > 1000 {
                // 확인용이라 엄격하게: 점 절반 이상이 짝, 바닥·벽 하나처럼 미끄러지는 장면은 확인 안 된 것으로
                if let Some(tpo) = icp_q(&self.prev_sub, &self.cur_sub, w / SUB, h / SUB, &k4, &Tf::IDENTITY, 0.5, 1e-2) {
                    self.pose = self.pose.mul(&local.mul(&tpo).mul(&local.inverse()));
                    self.bridge_verified = true;
                    self.bridges_verified += 1;
                }
            }
            self.top_up(f);
            let nd = if f.depth_mm.is_empty() { 0 } else { sub_depth(&f.depth_mm, w, h, &mut self.prev_sub) };
            // 어두워 점이 모자라도 깊이가 있으면 시작(깊이 ICP 로 이어 감)
            if self.tracks.len() < self.min_inliers && nd < 2000 {
                self.tracks.clear();
                self.prev_sub.clear();
                self.spare = Some(cur);
                return self.result(Tf::NULL, Self::eye(9999.0), 0, 0, false);
            }
            self.prev = Some(cur);
            self.prev_t = f.t;
            self.diag = Diag { method: 3, q: Quality::default(), jump: -1.0 };
            return self.result(Tf::IDENTITY, Self::eye(9999.0), 0, 0, true);
        };
        let dt = f.t - self.prev_t;
        // 1) 등속 예측 시작점
        // 0.2 초 넘게 비면(녹화 빈틈·놓친 뒤) 등속 예측은 틀리기 쉬워 쓰지 않음(움직임 없음에서 시작)
        // IMU: 마지막 성공 장면 → 이번 장면 몸체 회전(이어 붙이기 한도 안)
        let imu_rot = match self.imu.as_mut() {
            Some(u) => {
                u.poll(&local, f.t, self.prev_t);
                if dt > 0.0 && dt <= u.bridge_max { u.rotation(self.prev_t, f.t) } else { None }
            }
            None => None,
        };
        let pred = if let Some(r) = &imu_rot {
            // 회전은 IMU, 이동은 등속(0.2 초 안일 때만)
            let tr = if !self.velocity.is_null() && dt < 0.2 {
                let v = self.velocity.xyz_rpy();
                let m = |x: f32| (x as f64 * dt) as f32;
                [m(v[0]), m(v[1]), m(v[2])]
            } else {
                [0.0; 3]
            };
            Some(self.pose.mul(&crate::imu::tf_of(r, tr)))
        } else if !self.velocity.is_null() && dt > 0.0 && dt < 0.2 {
            let v = self.velocity.xyz_rpy();
            let m = |x: f32| (x as f64 * dt) as f32;
            Some(self.pose.mul(&Tf::from_xyz_rpy(m(v[0]), m(v[1]), m(v[2]), m(v[3]), m(v[4]), m(v[5]))))
        } else {
            None
        };
        // 0.5 초 넘는 빈틈 뒤: 비슷해 보이는 다른 자리에 잘못 붙지 않게 따라가던 점의 절반 이상이 맞아야 받아들임
        let need = if dt > 0.5 { self.min_inliers.max(self.tracks.len() / 2) } else { self.min_inliers };
        let still: Vec<[f32; 2]> = self.tracks.iter().map(|t| t.uv).collect();
        let mut out = None;
        if let Some(pp) = pred.filter(|_| !self.depth_only) {
            let tcw = pp.mul(&local).inverse();
            let inits: Vec<[f32; 2]> = self.tracks.iter().map(|t| project(&tcw, &k4, t.w).unwrap_or(t.uv)).collect();
            out = self.attempt(&prev, &cur, &inits, &k4, &local, need);
        }
        // IMU: 영상 회전이 자이로 회전과 크게 어긋나면 그 영상 결과는 받지 않음(엉뚱한 자리에 붙음). 카메라 전용 모드는 늘 false
        let base = self.pose;
        let rejected = std::cell::Cell::new(false);
        let disagree = |p: &Tf| {
            let d = imu_disagree(&imu_rot, &base.inverse().mul(p));
            rejected.set(rejected.get() || d);
            d
        };
        if out.as_ref().is_some_and(|o| disagree(&o.0)) {
            out = None;
        }
        if out.is_none() && !self.depth_only {
            out = self.attempt(&prev, &cur, &still, &k4, &local, need).filter(|o| !disagree(&o.0));
        }
        self.imu_rejected = rejected.get();
        let t_trk = t0.elapsed().as_secs_f64();
        let n_tracks = self.tracks.len();
        let Some((pose, ids, uvs, q)) = out else {
            // 먼저 마지막 키프레임과 기술자 맞추기(밝기만 있으면 더 정확), 안 되면 깊이 ICP
            self.imu_rejected = rejected.get();
            if let Some((pose, cov, inl)) = self.recover(f, pred).filter(|r| !disagree(&r.0)) {
                if !f.depth_mm.is_empty() {
                    sub_depth(&f.depth_mm, w, h, &mut self.prev_sub);
                }
                let transform = self.pose.inverse().mul(&pose);
                if dt > 0.0 {
                    let v = transform.xyz_rpy();
                    let d = |x: f32| (x as f64 / dt) as f32;
                    self.velocity = Tf::from_xyz_rpy(d(v[0]), d(v[1]), d(v[2]), d(v[3]), d(v[4]), d(v[5]));
                }
                self.pose = pose;
                self.prev = Some(cur);
                self.spare = Some(prev);
                self.prev_t = f.t;
                self.recovered += 1;
                self.diag = Diag { method: 2, q: Quality { inliers: inl, ..Default::default() }, jump: jump_of(&pred, &pose) };
                if std::env::var("KLT_PROF").is_ok() {
                    eprintln!("klt recover t {:.3} inliers {inl}", f.t);
                }
                return self.result(transform, cov, inl, n_tracks, true);
            }
            // 광류가 끊김(어둡거나 번짐): 깊이는 적외선이라 보통 멀쩡하다 -> 앞 장면 깊이와 점-평면 ICP
            let nd = if f.depth_mm.is_empty() || self.prev_sub.is_empty() { 0 } else { sub_depth(&f.depth_mm, w, h, &mut self.cur_sub) };
            // 빈틈(0.5 초 넘게) 뒤의 ICP 는 바닥·벽처럼 평평한 곳에서 엉뚱한 자리로 미끄러져도 잔차가 작아(fr1_floor 1.3 초 빈틈 0.4 m 어긋남) 쓰지 않음
            if nd > 1000 && dt <= 0.5 {
                let rel = pred.map(|p| self.pose.inverse().mul(&p)).unwrap_or(Tf::IDENTITY);
                let linv = local.inverse();
                let init = linv.mul(&rel).mul(&local);
                if let Some(tpo) = icp(&self.prev_sub, &self.cur_sub, w / SUB, h / SUB, &k4, &init).filter(|tpo| !disagree(&self.pose.mul(&local.mul(tpo).mul(&linv)))) {
                    let transform = local.mul(&tpo).mul(&linv);
                    let pose = self.pose.mul(&transform);
                    if dt > 0.0 {
                        let v = transform.xyz_rpy();
                        let d = |x: f32| (x as f64 / dt) as f32;
                        self.velocity = Tf::from_xyz_rpy(d(v[0]), d(v[1]), d(v[2]), d(v[3]), d(v[4]), d(v[5]));
                    }
                    self.pose = pose;
                    // 점은 이 장면에서 다시(보이는 만큼)
                    self.tracks.clear();
                    self.top_up(f);
                    std::mem::swap(&mut self.prev_sub, &mut self.cur_sub);
                    self.prev = Some(cur);
                    self.spare = Some(prev);
                    self.prev_t = f.t;
                    self.by_icp += 1;
                    self.diag = Diag { method: 1, q: Quality::default(), jump: jump_of(&pred, &pose) };
                    if std::env::var("KLT_PROF").is_ok() {
                        eprintln!("klt icp t {:.3}", f.t);
                    }
                    return self.result(transform, Self::eye(1e-3), 0, n_tracks, true);
                }
            }
            self.imu_rejected = rejected.get();
            // IMU: 영상이 다 실패한 짧은 구간은 마지막 성공 위치 × IMU 회전으로 이어 붙임(이동 0, 불확실).
            // 마지막 성공 장면 · 점은 그대로 둬 다음 장면이 IMU 예측으로 거기서 다시 따라감
            if let Some(r) = &imu_rot {
                let pose_b = self.pose.mul(&crate::imu::tf_of(r, [0.0; 3]));
                self.prev = Some(prev);
                self.spare = Some(cur);
                self.by_imu += 1;
                self.diag = Diag { method: 5, q: Quality::default(), jump: -1.0 };
                let mut cov = Self::eye(0.1);
                for (i, row) in cov.iter_mut().enumerate().skip(3) {
                    row[i] = 1e-3;
                }
                return OdomResult {
                    pose: Some(pose_b.to_pose()),
                    tf: pose_b,
                    lost: false,
                    covariance: cov,
                    inliers: 0,
                    matches: n_tracks,
                    keyframe_updated: false,
                    transform: self.pose.inverse().mul(&pose_b),
                    features: Features::default(),
                    reg: None,
                };
            }
            // 놓침: 마지막 성공 장면 그대로 둠(다음 장면이 거기서 다시)
            self.prev = Some(prev);
            self.spare = Some(cur);
            self.diag = Diag { method: 4, q: Quality::default(), jump: -1.0 };
            return self.result(Tf::NULL, Self::eye(9999.0), 0, n_tracks, false);
        };
        let transform = self.pose.inverse().mul(&pose);
        if dt > 0.0 {
            let v = transform.xyz_rpy();
            let d = |x: f32| (x as f64 / dt) as f32;
            self.velocity = Tf::from_xyz_rpy(d(v[0]), d(v[1]), d(v[2]), d(v[3]), d(v[4]), d(v[5]));
        }
        // IMU: 광류가 잘 된 장면의 영상 회전으로 자이로 치우침을 고침
        if let (Some(u), Some(r)) = (self.imu.as_mut(), &imu_rot) {
            if std::env::var("IMU_DBG").is_ok() {
                let (a, b) = (crate::imu::so3_log(r), crate::imu::so3_log(&crate::imu::m34_of_tf(&transform)));
                let d = |v: [f32; 4]| v.map(|x| x.to_degrees() / dt as f32);
                eprintln!("imu {:?} vis {:?} °/s bias {:?}", d(a), d(b), u.bias);
            }
            u.correct_bias(r, &crate::imu::m34_of_tf(&transform), dt);
        }
        self.pose = pose;
        let inliers = ids.len();
        // 공분산(RTAB-Map estimateMotion3DTo2D 와 같은 꼴): 이번 깊이로 본 점과 지도 점의 거리² 1/4 지점 × 2.1981
        let kps: Vec<KeyPoint> = uvs.iter().map(|u| KeyPoint { x: u[0], y: u[1], ..Default::default() }).collect();
        let p3 = keypoints_3d(&f.depth_mm, &f.cam, &kps, self.min_depth, self.max_depth);
        let mut ed: Vec<f32> = Vec::new();
        let mut ea: Vec<f32> = Vec::new();
        for (j, &i) in ids.iter().enumerate() {
            if let Some(p) = p3.get(j).filter(|p| p.x.is_finite()) {
                let q = pose.transform_point([p.x, p.y, p.z]);
                let o = self.tracks[i].w;
                let (dx, dy, dz) = (o[0] - q[0], o[1] - q[1], o[2] - q[2]);
                ed.push(dx * dx + dy * dy + dz * dz);
                let (a, b) = (Vector3::new(o[0], o[1], o[2]), Vector3::new(q[0], q[1], q[2]));
                ea.push(a.angle(&b));
            }
        }
        let mut cov = Self::eye(1.0);
        if ed.len() >= 4 {
            ed.sort_by(|a, b| a.total_cmp(b));
            ea.sort_by(|a, b| a.total_cmp(b));
            let (lin, ang) = (2.1981 * ed[ed.len() / 4] as f64, 2.1981 * ea[ea.len() / 4] as f64);
            for i in 0..3 {
                cov[i][i] = lin + 1e-6;
                cov[i + 3][i + 3] = ang + 1e-6;
            }
        } else {
            cov = Self::eye(1e-4);
        }
        // 안쪽 점만 남기고 이번 화소로
        let mut nt = Vec::with_capacity(ids.len());
        for (j, &i) in ids.iter().enumerate() {
            nt.push(Track { w: self.tracks[i].w, uv: uvs[j] });
        }
        self.tracks = nt;
        let mut kf = false;
        if (self.tracks.len() as f32) < self.redetect * self.max_features as f32 {
            self.top_up(f);
            kf = true;
        }
        if std::env::var("KLT_PROF").is_ok() {
            eprintln!("klt pyr {:.2} track+pnp {:.2} total {:.2} ms tracks {} kf {}", t_pyr * 1e3, (t_trk - t_pyr) * 1e3, t0.elapsed().as_secs_f64() * 1e3, n_tracks, kf);
        }
        if !f.depth_mm.is_empty() {
            sub_depth(&f.depth_mm, w, h, &mut self.prev_sub);
        }
        self.spare = Some(prev);
        self.prev = Some(cur);
        self.prev_t = f.t;
        self.diag = Diag { method: 0, q, jump: jump_of(&pred, &pose) };
        self.result(transform, cov, inliers, n_tracks, kf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 무늬를 옮긴 두 영상에서 광류가 이동량을 찾는다(큰 이동은 시작점으로)
    #[test]
    fn klt_shift() {
        let (w, h) = (320usize, 240usize);
        let img = |dx: f32, dy: f32| -> Vec<u8> {
            (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f32 - dx, (i / w) as f32 - dy);
                    (128.0 + 60.0 * (x * 0.11).sin() * (y * 0.07).cos() + 40.0 * ((x + y) * 0.05).sin()) as u8
                })
                .collect()
        };
        let a = Pyr::new(&img(0.0, 0.0), w, h);
        let b = Pyr::new(&img(6.0, -3.0), w, h);
        let r = track_one(&a, &b, [160.0, 120.0], [160.0, 120.0]).unwrap();
        assert!((r[0] - 166.0).abs() < 0.3 && (r[1] - 117.0).abs() < 0.3, "{r:?}");
        let c = Pyr::new(&img(45.0, 20.0), w, h);
        let r = track_one(&a, &c, [120.0, 100.0], [163.0, 121.0]).unwrap();
        assert!((r[0] - 165.0).abs() < 0.3 && (r[1] - 120.0).abs() < 0.3, "{r:?}");
    }
}
