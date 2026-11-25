//! 포터 TRACK — ORB-SLAM3 LocalMapping.cc 의 BA 호출을 뺀 판단들(RGB-D, 관성 없음).
//! MapPointCulling · KeyFrameCulling(KeyFrame/MapPoint::SetBadFlag 가 관측 수에 미치는 효과 포함).
//! 지역 BA 는 opt/orb3-rs-ba 몫이라 부르는 자리만: [`LocalMapperOps::local_ba`].
use std::collections::BTreeMap;
use super::mapdata::{kf_set_bad_flag, mp_set_bad_flag};
use super::track::{obs_weight, TrackExt};
use super::types::*;

/// LocalMapping::MapPointCulling. `recent` = mlpRecentAddedMapPoints(순서 유지), 돌려줌: SetBadFlag 부른 점들(순서대로).
/// 점 상태(found · visible · first_kf · bad · nObs)는 atlas.
pub fn map_point_culling(recent: &mut Vec<MpId>, cur_kf: KfId, monocular: bool, atlas: &mut Atlas, ext: &mut TrackExt) -> Vec<MpId> {
    let th: i32 = if monocular { 2 } else { 3 };
    let mut set_bad = Vec::new();
    let mut keep = Vec::with_capacity(recent.len());
    for &mp in recent.iter() {
        let m = &atlas.mps[&mp];
        let age = (cur_kf as i32).wrapping_sub(m.first_kf as i32);
        if m.bad {
            continue;
        }
        let ratio = m.found as f32 / m.visible as f32;
        if ratio < 0.25 {
            set_bad.push(mp);
            mp_set_bad_flag(atlas, mp);
        } else if age >= 2 && atlas.mps[&mp].n_obs <= th {
            set_bad.push(mp);
            mp_set_bad_flag(atlas, mp);
        } else if age >= 3 {
        } else {
            keep.push(mp);
        }
    }
    *recent = keep;
    set_bad
}

/// LocalMapping::KeyFrameCulling(관성 없음). `local` = 지금 키프레임의 공시성 순서 목록(UpdateBestCovisibles 뒤).
/// `abort_ba()` = mbAbortBA(반복 중 바뀔 수 있음). 돌려줌: SetBadFlag 를 부른 키프레임들.
pub fn keyframe_culling(
    local: &[KfId],
    monocular: bool,
    th_depth: &dyn Fn(KfId) -> f32,
    abort_ba: &dyn Fn() -> bool,
    atlas: &mut Atlas,
    ext: &mut TrackExt,
) -> Vec<KfId> {
    let redundant_th: f32 = 0.9;
    let mut count = 0;
    let mut called = Vec::new();
    for &kf in local {
        count += 1;
        let k = &atlas.kfs[&kf];
        let is_init = atlas.map_ref(k.map).is_some_and(|m| m.init_kf_id == kf);
        if !(is_init || k.bad) {
            let thd = th_depth(kf);
            let th_obs = 3;
            let mut n_red = 0;
            let mut n_mps = 0;
            for (i, slot) in k.map_points.iter().enumerate() {
                let Some(mp) = *slot else { continue };
                let m = &atlas.mps[&mp];
                if m.bad {
                    continue;
                }
                if !monocular && (k.depth[i] > thd || k.depth[i] < 0.0) {
                    continue;
                }
                n_mps += 1;
                if m.n_obs > th_obs {
                    let lvl = k.keys[i].octave;
                    let mut n = 0;
                    for (&okf, &(l, _r)) in m.obs.iter() {
                        if okf == kf {
                            continue;
                        }
                        let lvli = atlas.kfs[&okf].keys[l as usize].octave;
                        if lvli <= lvl + 1 {
                            n += 1;
                            if n > th_obs {
                                break;
                            }
                        }
                    }
                    if n > th_obs {
                        n_red += 1;
                    }
                }
            }
            if (n_red as f32) > redundant_th * n_mps as f32 {
                called.push(kf);
                kf_set_bad_flag(atlas, kf);
            }
        }
        if (count > 20 && abort_ba()) || count > 100 {
            break;
        }
    }
    called
}

/// LocalMapping::Run 한 바퀴의 차례(RGB-D, 관성 없음). 하위 단계는 다른 몫 · 원본.
pub trait LocalMapperOps {
    fn process_new_keyframe(&mut self);
    fn map_point_culling(&mut self);
    fn create_new_map_points(&mut self);
    fn check_new_keyframes(&mut self) -> bool;
    fn search_in_neighbors(&mut self);
    fn stop_requested(&mut self) -> bool;
    fn kfs_in_map(&mut self) -> i32;
    /// Optimizer::LocalBundleAdjustment(opt/orb3-rs-ba)
    fn local_ba(&mut self);
    fn keyframe_culling(&mut self);
    fn insert_in_loop_closer(&mut self);
}

/// LocalMapping::Run 의 키프레임 하나 처리(관성 · IMU 초기화 갈래 뺌)
pub fn run_once(ops: &mut dyn LocalMapperOps, abort_ba: &dyn Fn() -> bool) {
    ops.process_new_keyframe();
    ops.map_point_culling();
    ops.create_new_map_points();
    if !ops.check_new_keyframes() {
        ops.search_in_neighbors();
    }
    if !ops.check_new_keyframes() && !ops.stop_requested() {
        if ops.kfs_in_map() > 2 {
            ops.local_ba();
        }
        let _ = abort_ba;
        ops.keyframe_culling();
    }
    ops.insert_in_loop_closer();
}

// ---------------- 삼각측량(GeometricTools::Triangulate = Eigen JacobiSVD<Matrix4f>, 원본 빌드의 곱합 합치기 그대로) ----------------

const FLT_MIN: f32 = f32::MIN_POSITIVE;
/// Eigen JacobiSVD 의 precision = 2·ε
const SVD_PREC: f32 = 2.0 * f32::EPSILON;

/// JacobiRotation(c, s)를 행 p · q 에(applyOnTheLeft): x' = fma(c, x, s·y), y' = fma(−s, x, c·y)
#[inline]
fn rot_left(m: &mut [[f32; 4]; 4], p: usize, q: usize, c: f32, s: f32) {
    if c == 1.0 && s == 0.0 {
        return;
    }
    for k in 0..4 {
        let (x, y) = (m[p][k], m[q][k]);
        m[p][k] = c.mul_add(x, s * y);
        m[q][k] = (-s).mul_add(x, c * y);
    }
}

/// 열 p · q 에(applyOnTheRight, 회전의 전치): x' = fma(c, x, (−s)·y), y' = fma(s, x, c·y)
#[inline]
fn rot_right(m: &mut [[f32; 4]; 4], p: usize, q: usize, c: f32, s: f32) {
    if c == 1.0 && s == 0.0 {
        return;
    }
    for k in 0..4 {
        let (x, y) = (m[k][p], m[k][q]);
        m[k][p] = c.mul_add(x, (-s) * y);
        m[k][q] = s.mul_add(x, c * y);
    }
}

/// Eigen internal::real_2x2_jacobi_svd → (j_left, j_right) 각 (c, s)
fn real_2x2_jacobi_svd(w: &[[f32; 4]; 4], p: usize, q: usize) -> ((f32, f32), (f32, f32)) {
    let (mut m00, mut m01, mut m10, mut m11) = (w[p][p], w[p][q], w[q][p], w[q][q]);
    let d = m10 - m01;
    let (c1, s1);
    if d.abs() < FLT_MIN {
        c1 = 1.0;
        s1 = 0.0;
    } else {
        let t = m11 + m00;
        let u = t / d;
        let tmp = u.mul_add(u, 1.0).sqrt();
        s1 = 1.0 / tmp;
        c1 = u / tmp;
        if !(c1 == 1.0 && s1 == 0.0) {
            let (a, b) = (m00, m10);
            m00 = c1.mul_add(a, s1 * b);
            m10 = (-s1).mul_add(a, c1 * b);
            let (a, b) = (m01, m11);
            m01 = c1.mul_add(a, s1 * b);
            m11 = (-s1).mul_add(a, c1 * b);
        }
    }
    let _ = m10;
    // makeJacobi(x = m00, y = m01, z = m11)
    let (cr, sr);
    let deno = m01.abs() + m01.abs();
    if deno < FLT_MIN {
        cr = 1.0;
        sr = 0.0;
    } else {
        let tau = (m00 - m11) / deno;
        let w = tau.mul_add(tau, 1.0).sqrt();
        let t = if tau > 0.0 { 1.0 / (w + tau) } else { 1.0 / (tau - w) };
        let nsign: f32 = if t > 0.0 { -1.0 } else { 1.0 };
        let n = 1.0 / t.mul_add(t, 1.0).sqrt();
        sr = ((m01 / m01.abs()) * nsign) * t.abs() * n;
        cr = n;
    }
    // j_left = rot1 · j_right^T, j_right^T = (cr, −sr)
    let ms = -sr;
    let sl = c1.mul_add(ms, s1 * cr);
    let cl = (-s1).mul_add(ms, c1 * cr);
    ((cl, sl), (cr, sr))
}

/// GeometricTools::Triangulate. t1 · t2 = Tc1w · Tc2w 의 3×4([R|t], 행 우선). 실패(동차 좌표 w = 0)면 None.
pub fn triangulate(xn1: &[f32; 3], xn2: &[f32; 3], t1: &[[f32; 4]; 3], t2: &[[f32; 4]; 3]) -> Option<[f32; 3]> {
    let mut a = [[0f32; 4]; 4];
    for c in 0..4 {
        a[0][c] = xn1[0].mul_add(t1[2][c], -t1[0][c]);
        a[1][c] = xn1[1].mul_add(t1[2][c], -t1[1][c]);
        a[2][c] = xn2[0].mul_add(t2[2][c], -t2[0][c]);
        a[3][c] = xn2[1].mul_add(t2[2][c], -t2[1][c]);
    }
    let mut scale = 0f32;
    for r in &a {
        for v in r {
            scale = scale.max(v.abs());
        }
    }
    if !scale.is_finite() {
        return None;
    }
    if scale == 0.0 {
        scale = 1.0;
    }
    let mut w = a;
    for r in w.iter_mut() {
        for v in r.iter_mut() {
            *v /= scale;
        }
    }
    let mut v = [[0f32; 4]; 4];
    for i in 0..4 {
        v[i][i] = 1.0;
    }
    let mut max_diag = 0f32;
    for i in 0..4 {
        max_diag = max_diag.max(w[i][i].abs());
    }
    loop {
        let mut finished = true;
        for p in 1..4 {
            for q in 0..p {
                let m = SVD_PREC * max_diag;
                let thr = if !(FLT_MIN < m) { FLT_MIN } else { m };
                if w[p][q].abs() > thr || w[q][p].abs() > thr {
                    finished = false;
                    let ((cl, sl), (cr, sr)) = real_2x2_jacobi_svd(&w, p, q);
                    rot_left(&mut w, p, q, cl, sl);
                    rot_right(&mut w, p, q, cr, sr);
                    rot_right(&mut v, p, q, cr, sr);
                    max_diag = max_diag.max(w[p][p].abs().max(w[q][q].abs()));
                }
            }
        }
        if finished {
            break;
        }
    }
    let mut sv = [0f32; 4];
    for i in 0..4 {
        sv[i] = w[i][i].abs() * scale;
    }
    for i in 0..4 {
        let mut pos = 0;
        let mut mx = sv[i];
        for j in 1..4 - i {
            if sv[i + j] > mx {
                pos = j;
            }
            mx = sv[i + j].max(mx);
        }
        if mx == 0.0 {
            break;
        }
        if pos != 0 {
            sv.swap(i, i + pos);
            for r in v.iter_mut() {
                r.swap(i, i + pos);
            }
        }
    }
    let h = [v[0][3], v[1][3], v[2][3], v[3][3]];
    if h[3] == 0.0 {
        return None;
    }
    Some([h[0] / h[3], h[1] / h[3], h[2] / h[3]])
}

// ---------------- LocalMapping::CreateNewMapPoints ----------------

use super::track::{pose_inverse_matrices, quat_to_rot, unproject};

/// 3 원소 내적(Eigen redux a0b0 + (a1b1 + a2b2), GCC 합치기: fma(a0, b0, fma(a2, b2, a1·b1)))
#[inline]
pub fn dot3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0].mul_add(b[0], a[2].mul_add(b[2], a[1] * b[1]))
}
#[inline]
fn mat_vec(m: &[[f32; 3]; 3], x: &[f32; 3]) -> [f32; 3] {
    [dot3(&m[0], x), dot3(&m[1], x), dot3(&m[2], x)]
}
#[inline]
fn norm3(a: &[f32; 3]) -> f32 {
    dot3(a, a).sqrt()
}

/// 키프레임 자세에서 나오는 행렬들(Sophus matrix3x4 · 카메라 중심)
pub struct KfPose {
    pub rcw: [[f32; 3]; 3],
    /// Rcw 전치(CreateNewMapPoints 의 광선용 Rwc1 · Rwc2)
    pub rwc: [[f32; 3]; 3],
    /// KeyFrame::mRwc = mTwc.rotationMatrix() — 켤레 사원수를 다시 정규화해 만들어 전치와 끝자리가 다를 수 있음(UnprojectStereo 용)
    pub rwc_stored: [[f32; 3]; 3],
    pub tcw: [f32; 3],
    pub ow: [f32; 3],
    pub t34: [[f32; 4]; 3],
}
pub fn kf_pose(t: &Se3) -> KfPose {
    let rcw = quat_to_rot(t.q);
    let mut rwc = [[0f32; 3]; 3];
    let mut t34 = [[0f32; 4]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rwc[i][j] = rcw[j][i];
            t34[i][j] = rcw[i][j];
        }
        t34[i][3] = t.t[i];
    }
    let (rwc_stored, ow) = pose_inverse_matrices(t);
    KfPose { rcw, rwc, rwc_stored, tcw: t.t, ow, t34 }
}

/// 삼각측량 설정(LocalMapping · 키프레임 상수)
pub struct TriParams {
    pub cam: Camera,
    /// mvScaleFactors · mvLevelSigma2
    pub scale: Vec<f32>,
    pub sigma2: Vec<f32>,
    pub scale_factor: f32,
    pub monocular: bool,
    pub inertial: bool,
    pub far_points: bool,
    pub th_far_points: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewMp {
    pub kf2: KfId,
    pub idx1: usize,
    pub idx2: usize,
    pub pos: [f32; 3],
}

/// KeyFrame::UnprojectStereo: mRwc·(x,y,z) + Ow (mvKeys 좌표)
fn kf_unproject(k: &KeyFrame, p: &KfPose, cam: &Camera, i: usize) -> Option<[f32; 3]> {
    let z = k.depth[i];
    if z > 0.0 { Some(unproject(&p.rwc_stored, &p.ow, cam, k.keys_raw[i][0], k.keys_raw[i][1], z)) } else { None }
}

/// LocalMapping::CreateNewMapPoints(RGB-D · 관성 없음 · 오른쪽 카메라 없음).
/// `neighbors` = GetBestCovisibilityKeyFrames(10), `search(kf2)` = ORBmatcher::SearchForTriangulation 의 짝(지금 키프레임 번호, kf2 번호),
/// `check_new()` = CheckNewKeyFrames(새 키프레임이 오면 그만). 새 점마다 `on_new` 를 부름(부르는 쪽이 지도에 넣음 — 다음 이웃 짝 찾기가 그걸 봄).
pub fn create_new_map_points(
    cur: &KeyFrame,
    neighbors: &[&KeyFrame],
    prm: &TriParams,
    search: &mut dyn FnMut(KfId) -> Vec<(usize, usize)>,
    check_new: &mut dyn FnMut() -> bool,
    on_new: &mut dyn FnMut(NewMp),
) {
    assert!(!prm.monocular && !prm.inertial);
    let cam = &prm.cam;
    let p1 = kf_pose(&cur.tcw);
    let ratio_factor = 1.5f32 * prm.scale_factor;
    for (i, k2) in neighbors.iter().enumerate() {
        if i > 0 && check_new() {
            return;
        }
        let p2 = kf_pose(&k2.tcw);
        let vb = [p2.ow[0] - p1.ow[0], p2.ow[1] - p1.ow[1], p2.ow[2] - p1.ow[2]];
        let baseline = norm3(&vb);
        if baseline < cam.b {
            continue;
        }
        let matches = search(k2.id);
        for &(idx1, idx2) in &matches {
            let kp1 = &cur.keys[idx1];
            let kp1_ur = cur.u_right[idx1];
            let st1 = kp1_ur >= 0.0;
            let kp2 = &k2.keys[idx2];
            let kp2_ur = k2.u_right[idx2];
            let st2 = kp2_ur >= 0.0;
            let xn1 = [(kp1.x - cam.cx) / cam.fx, (kp1.y - cam.cy) / cam.fy, 1.0];
            let xn2 = [(kp2.x - cam.cx) / cam.fx, (kp2.y - cam.cy) / cam.fy, 1.0];
            let ray1 = mat_vec(&p1.rwc, &xn1);
            let ray2 = mat_vec(&p2.rwc, &xn2);
            let cpr = dot3(&ray1, &ray2) / (norm3(&ray1) * norm3(&ray2));
            let cps0 = cpr + 1.0;
            let mut cps1 = cps0;
            let mut cps2 = cps0;
            if st1 {
                cps1 = (2.0 * (cam.b / 2.0).atan2(cur.depth[idx1])).cos();
            } else if st2 {
                cps2 = (2.0 * (cam.b / 2.0).atan2(k2.depth[idx2])).cos();
            }
            let cps = cps1.min(cps2);
            let x3d = if cpr < cps && cpr > 0.0 && (st1 || st2 || (cpr as f64) < 0.9998) {
                match triangulate(&xn1, &xn2, &p1.t34, &p2.t34) {
                    Some(x) => x,
                    None => continue,
                }
            } else if st1 && cps1 < cps2 {
                match kf_unproject(cur, &p1, cam, idx1) {
                    Some(x) => x,
                    None => continue,
                }
            } else if st2 && cps2 < cps1 {
                match kf_unproject(k2, &p2, cam, idx2) {
                    Some(x) => x,
                    None => continue,
                }
            } else {
                continue;
            };
            let z1 = dot3(&p1.rcw[2], &x3d) + p1.tcw[2];
            if z1 <= 0.0 {
                continue;
            }
            let z2 = dot3(&p2.rcw[2], &x3d) + p2.tcw[2];
            if z2 <= 0.0 {
                continue;
            }
            if !reproj_ok(cam, &p1, &x3d, z1, kp1, kp1_ur, st1, prm.sigma2[kp1.octave as usize]) {
                continue;
            }
            if !reproj_ok(cam, &p2, &x3d, z2, kp2, kp2_ur, st2, prm.sigma2[kp2.octave as usize]) {
                continue;
            }
            let n1 = [x3d[0] - p1.ow[0], x3d[1] - p1.ow[1], x3d[2] - p1.ow[2]];
            let d1 = norm3(&n1);
            let n2 = [x3d[0] - p2.ow[0], x3d[1] - p2.ow[1], x3d[2] - p2.ow[2]];
            let d2 = norm3(&n2);
            if d1 == 0.0 || d2 == 0.0 {
                continue;
            }
            if prm.far_points && (d1 >= prm.th_far_points || d2 >= prm.th_far_points) {
                continue;
            }
            let ratio_dist = d2 / d1;
            let ratio_oct = prm.scale[kp1.octave as usize] / prm.scale[kp2.octave as usize];
            if ratio_dist * ratio_factor < ratio_oct || ratio_dist > ratio_oct * ratio_factor {
                continue;
            }
            on_new(NewMp { kf2: k2.id, idx1, idx2, pos: x3d });
        }
    }
}

/// 다시 투영 오차 검사(단안 5.991 σ², 깊이 있으면 7.8 σ²). 원본은 두 번째 키프레임에도 지금 키프레임의 mbf 를 씀(같은 카메라라 같은 값).
#[allow(clippy::too_many_arguments)]
fn reproj_ok(cam: &Camera, p: &KfPose, x3d: &[f32; 3], z: f32, kp: &KeyPoint, ur: f32, stereo: bool, s2: f32) -> bool {
    let x = dot3(&p.rcw[0], x3d) + p.tcw[0];
    let y = dot3(&p.rcw[1], x3d) + p.tcw[1];
    let invz = (1.0f64 / z as f64) as f32;
    if !stereo {
        let u = cam.fx * x / z + cam.cx;
        let v = cam.fy * y / z + cam.cy;
        let (ex, ey) = (u - kp.x, v - kp.y);
        !((ex.mul_add(ex, ey * ey) as f64) > 5.991 * s2 as f64)
    } else {
        let u = (cam.fx * x).mul_add(invz, cam.cx);
        let ur2 = (-cam.bf).mul_add(invz, u);
        let v = (cam.fy * y).mul_add(invz, cam.cy);
        let (ex, ey, er) = (u - kp.x, v - kp.y, ur2 - ur);
        !((er.mul_add(er, ex.mul_add(ex, ey * ey)) as f64) > 7.8 * s2 as f64)
    }
}




