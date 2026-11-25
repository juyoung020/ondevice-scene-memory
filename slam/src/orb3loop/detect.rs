//! LoopClosing 의 공통 영역 찾기(src/LoopClosing.cc): `DetectCommonRegionsFromBoW` · `DetectCommonRegionsFromLastKF` ·
//! `DetectAndReffineSim3FromLastKF` · `FindMatchesByProjection`. 원본 순서 · 버릇 그대로(아래 "원본 버릇").
//! `Optimizer::OptimizeSim3` 는 트레잇([`OptimizeSim3`]): 대조 시험은 원본 FFI 를 끼운다(BA 몫 Rust 판이 오면 바꿔 낌).
//!
//! 원본 버릇(결과가 같아야 해서 그대로 둠):
//! - BoW 단계의 `pMostBoWMatchesKF` 는 늘 후보 자신(가장 많이 맞은 공시 키프레임 번호만 세고 포인터는 안 바꿈).
//! - Sim3Solver 에 키프레임별 짝 목록을 주지만 원본 생성자는 비어 있지 않으면 무시하고 늘 pKF2 로 색인을 찾음(sim3.rs PairIn).
//! - `DetectCommonRegionsFromLastKF` 가 BoW 단계의 `vpMapPoints` 를 덮어씀 → 최고 후보의 점 목록은 마지막 확인 호출의 것.
//! - `FindMatchesByProjection` 은 공시 키프레임이 10 개 미만이면 2 단계 이웃을 거르지 않고 통째로 붙임(중복 포함).
//! - `DetectAndReffineSim3FromLastKF` 는 OptimizeSim3 결과(gScm)를 버리고 들어온 gScw 의 회전 · 이동(크기 1)으로 다시 찾음.
use std::collections::{BTreeMap, BTreeSet};

use super::sim3::{PairIn, Pinhole as SPinhole, Sim3Solver};
use crate::orb3rs::matcher::Matcher;
use crate::orb3rs::{KeyFrame, KfId, MapPoint, MpId, Se3, Sim3};

// ---------------- g2o::Sim3 (double) ----------------

/// g2o::Sim3: 사원수 r(x, y, z, w) · t · s
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G2oSim3 {
    pub r: [f64; 4],
    pub t: [f64; 3],
    pub s: f64,
}

impl Default for G2oSim3 {
    fn default() -> Self {
        G2oSim3 { r: [0.0, 0.0, 0.0, 1.0], t: [0.0; 3], s: 1.0 }
    }
}

/// Eigen Quaterniond 곱(SSE2 Packet2d 판: t1 = ww·b_xy + yy·b_zw, t2 = zz·b_xy − xx·b_zw, xy = t1 ∓ swap(t2) …)
pub fn quatd_mul(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    let (xx, yy, zz, ww) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    let t1 = [ww * bx + yy * bz, ww * by + yy * bw];
    let t2 = [zz * bx - xx * bz, zz * by - xx * bw];
    let x = t1[0] - t2[1];
    let y = t1[1] + t2[0];
    let u1 = [ww * bz - yy * bx, ww * bw - yy * by];
    let u2 = [zz * bz + xx * bx, zz * bw + xx * by];
    let z = u1[0] + u2[1];
    let w = u1[1] - u2[0];
    [x, y, z, w]
}

fn crossd(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Quaterniond * Vector3d(_transformVector): uv = v×p, uv += uv, (p + w·uv) + v×uv
pub fn quatd_rotate(q: &[f64; 4], p: &[f64; 3]) -> [f64; 3] {
    let v = [q[0], q[1], q[2]];
    let mut uv = crossd(&v, p);
    uv = [uv[0] + uv[0], uv[1] + uv[1], uv[2] + uv[2]];
    let c = crossd(&v, &uv);
    [p[0] + q[3] * uv[0] + c[0], p[1] + q[3] * uv[1] + c[1], p[2] + q[3] * uv[2] + c[2]]
}

/// Eigen Quaterniond(Matrix3d)(Shoemake, 행 우선 m)
pub fn quatd_from_mat(m: &[[f64; 3]; 3]) -> [f64; 4] {
    let mut q = [0.0f64; 4];
    let t = (m[0][0] + m[1][1]) + m[2][2];
    if t > 0.0 {
        let t = (t + 1.0).sqrt();
        q[3] = 0.5 * t;
        let t = 0.5 / t;
        q[0] = (m[2][1] - m[1][2]) * t;
        q[1] = (m[0][2] - m[2][0]) * t;
        q[2] = (m[1][0] - m[0][1]) * t;
    } else {
        let mut i = 0;
        if m[1][1] > m[0][0] {
            i = 1;
        }
        if m[2][2] > m[i][i] {
            i = 2;
        }
        let j = (i + 1) % 3;
        let k = (j + 1) % 3;
        let t = ((m[i][i] - m[j][j]) - m[k][k] + 1.0).sqrt();
        q[i] = 0.5 * t;
        let t = 0.5 / t;
        q[3] = (m[k][j] - m[j][k]) * t;
        q[j] = (m[j][i] + m[i][j]) * t;
        q[k] = (m[k][i] + m[i][k]) * t;
    }
    q
}

/// Quaterniond::toRotationMatrix(행 우선)
pub fn quatd_to_mat(q: &[f64; 4]) -> [[f64; 3]; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let (tx, ty, tz) = (2.0 * x, 2.0 * y, 2.0 * z);
    let (twx, twy, twz) = (tx * w, ty * w, tz * w);
    let (txx, txy, txz) = (tx * x, ty * x, tz * x);
    let (tyy, tyz, tzz) = (ty * y, tz * y, tz * z);
    [[1.0 - (tyy + tzz), txy - twz, txz + twy], [txy + twz, 1.0 - (txx + tzz), tyz - twx], [txz - twy, tyz + twx, 1.0 - (txx + tyy)]]
}

impl G2oSim3 {
    /// Sim3(Quaterniond, t, s)
    pub fn new(r: [f64; 4], t: [f64; 3], s: f64) -> G2oSim3 {
        G2oSim3 { r, t, s }
    }
    /// Sim3(Matrix3d, t, s)
    pub fn from_mat(m: &[[f64; 3]; 3], t: [f64; 3], s: f64) -> G2oSim3 {
        G2oSim3 { r: quatd_from_mat(m), t, s }
    }
    /// Sim3(SE3f.cast<double>().unit_quaternion(), translation, 1): Sophus 의 cast 는 SO3d 생성자를 거쳐 double 로 다시 정규화
    pub fn from_se3(p: &Se3) -> G2oSim3 {
        let q = p.q.map(f64::from);
        let n = ((q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3])).sqrt();
        G2oSim3 { r: [q[0] / n, q[1] / n, q[2] / n, q[3] / n], t: p.t.map(f64::from), s: 1.0 }
    }
    /// Sim3(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>(), 1): 사원수만 내려받음(정규화 없음)
    pub fn from_se3_raw(p: &Se3) -> G2oSim3 {
        G2oSim3 { r: p.q.map(f64::from), t: p.t.map(f64::from), s: 1.0 }
    }
    /// Sim3(R(float 행 우선).cast<double>(), t.cast<double>(), s)
    pub fn from_mat_f(m: &[[f32; 3]; 3], t: &[f32; 3], s: f64) -> G2oSim3 {
        G2oSim3::from_mat(&m.map(|r| r.map(f64::from)), t.map(f64::from), s)
    }
    /// operator*: r = r·o.r, t = s·(r·o.t) + t, s = s·o.s
    pub fn mul(&self, o: &G2oSim3) -> G2oSim3 {
        let rt = quatd_rotate(&self.r, &o.t);
        G2oSim3 { r: quatd_mul(&self.r, &o.r), t: [self.s * rt[0] + self.t[0], self.s * rt[1] + self.t[1], self.s * rt[2] + self.t[2]], s: self.s * o.s }
    }
    /// Converter::toSophus: Sim3f(RxSO3d((float)s, r.matrix()).cast<float>(), t.cast<float>())
    pub fn to_sophus(&self) -> Sim3 {
        let q = quatd_from_mat(&quatd_to_mat(&self.r));
        let sq = ((self.s as f32) as f64).sqrt();
        Sim3 { q: q.map(|v| (v * sq) as f32), t: self.t.map(|v| v as f32) }
    }
}

// ---------------- 키프레임 그래프 ----------------

/// GetBestCovisibilityKeyFrames(n)
pub fn best_covisibles(kf: &KeyFrame, n: usize) -> Vec<KfId> {
    kf.ordered_conn.iter().take(n).map(|c| c.0).collect()
}

/// GetConnectedKeyFrames(집합, 찾기만 씀)
pub fn connected(kf: &KeyFrame) -> BTreeSet<KfId> {
    kf.conn_weights.keys().copied().collect()
}

fn mp_ok(m: &Option<MpId>, mps: &BTreeMap<MpId, MapPoint>) -> Option<MpId> {
    m.filter(|id| !mps[id].bad)
}

fn index_in(mp: &MapPoint, kf: KfId) -> i32 {
    mp.obs.get(&kf).map_or(-1, |o| o.0)
}

/// `Optimizer::OptimizeSim3(pKF1, pKF2, vpMatches1, g2oS12, th2, bFixScale, H, bAllPoints)`: 지운 짝은 None 으로, g2oS12 갱신, 내점 수
pub trait OptimizeSim3 {
    #[allow(clippy::too_many_arguments)]
    fn optimize_sim3(
        &mut self,
        kf1: &KeyFrame,
        kf2: &KeyFrame,
        matches: &mut [Option<MpId>],
        s12: &mut G2oSim3,
        th2: f32,
        fix_scale: bool,
        all_points: bool,
        mps: &BTreeMap<MpId, MapPoint>,
    ) -> i32;
}

pub struct World<'a> {
    pub kfs: &'a BTreeMap<KfId, KeyFrame>,
    pub mps: &'a BTreeMap<MpId, MapPoint>,
}

/// FindMatchesByProjection: (짝 수, vpMapPoints, vpMatchedMapPoints)
pub fn find_matches_by_projection(w: &World, cur: KfId, matched: KfId, scw: &G2oSim3) -> (i32, Vec<MpId>, Vec<Option<MpId>>) {
    let n_cov = 10;
    let mkf = &w.kfs[&matched];
    let mut cov = best_covisibles(mkf, n_cov);
    let n_init = cov.len();
    cov.push(matched);
    let mut check: BTreeSet<KfId> = cov.iter().copied().collect();
    let cur_cov = connected(&w.kfs[&cur]);
    if n_init < n_cov {
        for i in 0..n_init {
            let v = best_covisibles(&w.kfs[&cov[i]], n_cov);
            let mut ins = 0;
            let mut j = 0;
            while j < v.len() && ins < n_cov {
                if !check.contains(&v[j]) && !cur_cov.contains(&v[j]) {
                    check.insert(v[j]);
                    ins += 1;
                }
                j += 1;
            }
            cov.extend(v);
        }
    }
    let mut seen = BTreeSet::new();
    let mut pts = Vec::new();
    for k in &cov {
        for m in &w.kfs[k].map_points {
            if let Some(id) = mp_ok(m, w.mps) {
                if seen.insert(id) {
                    pts.push(id);
                }
            }
        }
    }
    let ckf = &w.kfs[&cur];
    let mut out = vec![None; ckf.map_points.len()];
    let n = Matcher::new(0.9, true).search_by_projection_sim3(ckf, &scw.to_sophus(), &pts, &mut out, w.mps, 3, 1.5);
    (n, pts, out)
}

/// DetectCommonRegionsFromLastKF: (참이면 ≥ 30, 짝 수, 점들, 짝)
pub fn detect_common_regions_from_last_kf(w: &World, cur: KfId, matched: KfId, scw: &G2oSim3) -> (bool, i32, Vec<MpId>, Vec<Option<MpId>>) {
    let (n, p, m) = find_matches_by_projection(w, cur, matched, scw);
    (n >= 30, n, p, m)
}

/// DetectAndReffineSim3FromLastKF 결과(gScw 는 참일 때만 바뀜)
pub struct LastOut {
    pub ok: bool,
    pub scw: G2oSim3,
    pub n_proj: i32,
    pub mps: Vec<MpId>,
    pub matched: Vec<Option<MpId>>,
}

/// DetectAndReffineSim3FromLastKF(pCurrentKF, pMatchedKF, gScw, …). `member_cur` = LoopClosing::mpCurrentKF(OptimizeSim3 의 KF1)
pub fn detect_and_refine_sim3_from_last_kf<O: OptimizeSim3>(w: &World, cur: KfId, member_cur: KfId, matched: KfId, scw: &G2oSim3, fix_scale: bool, opt: &mut O) -> LastOut {
    let (n, p, mut m) = find_matches_by_projection(w, cur, matched, scw);
    let mut o = LastOut { ok: false, scw: *scw, n_proj: n, mps: p, matched: m.clone() };
    if n >= 30 {
        let mkf = &w.kfs[&matched];
        let swm = G2oSim3::from_se3(&mkf.twc);
        let mut scm = scw.mul(&swm);
        let nopt = opt.optimize_sim3(&w.kfs[&member_cur], mkf, &mut m, &mut scm, 10.0, fix_scale, true, w.mps);
        o.matched = m;
        if nopt > 50 {
            let est = G2oSim3::new(scw.r, scw.t, 1.0);
            let (n2, p2, m2) = find_matches_by_projection(w, cur, matched, &est);
            o.n_proj = n2;
            o.mps = p2;
            o.matched = m2;
            if n2 >= 100 {
                o.scw = est;
                o.ok = true;
            }
        }
    }
    o
}

/// DetectCommonRegionsFromBoW 의 출력 칸(원본은 참조 인자 · LoopClosing 멤버. 최고 후보가 없으면 그대로 둠)
#[derive(Clone, Debug, Default)]
pub struct BowOut {
    pub matched_kf2: Option<KfId>,
    pub last_current_kf: Option<KfId>,
    pub scw: G2oSim3,
    pub n_coincidences: i32,
    pub mps: Vec<Option<MpId>>,
    pub matched_mps: Vec<Option<MpId>>,
    /// 원본은 여기서 pMatchedKF2->SetNotErase() — 부르는 쪽이 처리
    pub set_not_erase: Option<KfId>,
}

fn rot_cm(m: &[[f32; 3]; 3]) -> [f32; 9] {
    [m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]]
}

/// DetectCommonRegionsFromBoW(vpBowCand, …). 난수는 원본 DUtils::Random::RandomInt 자리(시스템 glibc rand 하나)
#[allow(clippy::too_many_arguments)]
pub fn detect_common_regions_from_bow<O: OptimizeSim3>(
    w: &World,
    cur: KfId,
    fix_scale: bool,
    cands: &[Option<KfId>],
    opt: &mut O,
    rand: &mut dyn FnMut(i32, i32) -> i32,
    out: &mut BowOut,
) -> bool {
    let (n_bow_matches, n_bow_inliers, n_sim3_inliers, n_proj_matches, n_proj_opt_matches) = (20, 15, 20, 50, 80);
    let ckf = &w.kfs[&cur];
    let conn = connected(ckf);
    let n_cov = 10;
    let matcher_bow = Matcher::new(0.9, true);
    let matcher = Matcher::new(0.75, true);
    let mut best_reproj = 0;
    let mut best_coinc = 0;
    let mut best_kf = None;
    let mut best_scw = G2oSim3::default();
    let mut best_mps: Vec<MpId> = Vec::new();
    let mut best_matched: Vec<Option<MpId>> = Vec::new();
    let ncur = ckf.map_points.len();
    for c in cands {
        let Some(ki) = *c else { continue };
        let kfi = &w.kfs[&ki];
        if kfi.bad {
            continue;
        }
        let mut cov = best_covisibles(kfi, n_cov);
        if cov.is_empty() {
            cov.push(ki);
        } else {
            cov.push(cov[0]);
            cov[0] = ki;
        }
        if cov.iter().any(|k| conn.contains(k)) {
            continue;
        }
        let mut vv: Vec<Vec<Option<MpId>>> = vec![Vec::new(); cov.len()];
        for (j, k) in cov.iter().enumerate() {
            let kj = &w.kfs[k];
            if kj.bad {
                continue;
            }
            vv[j] = matcher_bow.search_by_bow_kf_kf(ckf, kj, w.mps).1;
        }
        let most = ki;
        let mut seen = BTreeSet::new();
        let mut nbow = 0;
        let mut matched_pts: Vec<Option<MpId>> = vec![None; ncur];
        for v in &vv {
            for (k, m) in v.iter().enumerate() {
                let Some(id) = mp_ok(m, w.mps) else { continue };
                if seen.insert(id) {
                    nbow += 1;
                    matched_pts[k] = Some(id);
                }
            }
        }
        if nbow < n_bow_matches {
            continue;
        }
        let mkf = &w.kfs[&most];
        // Sim3Solver(mpCurrentKF, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP)
        let pairs: Vec<PairIn> = (0..ncur)
            .map(|i1| {
                let Some(m2) = matched_pts[i1] else { return PairIn::Unmatched };
                let Some(m1) = ckf.map_points[i1] else { return PairIn::NoMp1 };
                let (p1, p2) = (&w.mps[&m1], &w.mps[&m2]);
                if p1.bad || p2.bad {
                    return PairIn::Bad;
                }
                let (i1k, i2k) = (index_in(p1, cur), index_in(p2, most));
                if i1k < 0 || i2k < 0 {
                    return PairIn::NoIndex;
                }
                PairIn::Use { pos1: p1.pos, pos2: p2.pos, oct1: ckf.keys[i1k as usize].octave, oct2: mkf.keys[i2k as usize].octave }
            })
            .collect();
        let cam = |k: &KeyFrame| SPinhole { fx: k.ctx.cam.fx, fy: k.ctx.cam.fy, cx: k.ctx.cam.cx, cy: k.ctx.cam.cy };
        let mut solver = Sim3Solver::new(
            &rot_cm(&ckf.tcw.rot()),
            &ckf.tcw.t,
            &rot_cm(&mkf.tcw.rot()),
            &mkf.tcw.t,
            &ckf.ctx.pyr.sigma2,
            &mkf.ctx.pyr.sigma2,
            &pairs,
            cam(ckf),
            cam(mkf),
            fix_scale,
        );
        solver.set_ransac_parameters(0.99, n_bow_inliers, 300);
        let (mut converge, mut no_more) = (false, false);
        while !converge && !no_more {
            let o = solver.iterate_with(20, rand);
            converge = o.converge;
            no_more = o.no_more;
        }
        if !converge {
            continue;
        }
        let mut cov2 = best_covisibles(mkf, n_cov);
        cov2.push(most);
        let mut seen = BTreeSet::new();
        let mut pts: Vec<MpId> = Vec::new();
        let mut pts_kf: Vec<KfId> = Vec::new();
        for k in &cov2 {
            for m in &w.kfs[k].map_points {
                if let Some(id) = mp_ok(m, w.mps) {
                    if seen.insert(id) {
                        pts.push(id);
                        pts_kf.push(*k);
                    }
                }
            }
        }
        let br = &solver.best_r;
        let rm = [[br[0], br[3], br[6]], [br[1], br[4], br[7]], [br[2], br[5], br[8]]];
        let mut scm = G2oSim3::from_mat_f(&rm, &solver.best_t, solver.best_s as f64);
        let smw = G2oSim3::from_mat_f(&mkf.tcw.rot(), &mkf.tcw.t, 1.0);
        let scw = scm.mul(&smw);
        let mut mmp = vec![None; ncur];
        let mut mkfv = vec![None; ncur];
        let nproj = matcher.search_by_projection_sim3_kfs(ckf, &scw.to_sophus(), &pts, &pts_kf, &mut mmp, &mut mkfv, w.mps, 8, 1.5);
        if nproj < n_proj_matches {
            continue;
        }
        let nopt = opt.optimize_sim3(ckf, kfi, &mut mmp, &mut scm, 10.0, fix_scale, true, w.mps);
        if nopt < n_sim3_inliers {
            continue;
        }
        let scw = scm.mul(&smw);
        let mut mmp = vec![None; ncur];
        let nprojopt = matcher.search_by_projection_sim3(ckf, &scw.to_sophus(), &pts, &mut mmp, w.mps, 5, 1.0);
        if nprojopt < n_proj_opt_matches {
            continue;
        }
        let ccov = best_covisibles(ckf, n_cov);
        let mut nkfs = 0;
        let mut j = 0;
        while nkfs < 3 && j < ccov.len() {
            let kj = &w.kfs[&ccov[j]];
            let tjc = kj.tcw.mul(&ckf.twc);
            let sjw = G2oSim3::from_se3(&tjc).mul(&scw);
            let (valid, _, p, _) = detect_common_regions_from_last_kf(w, ccov[j], most, &sjw);
            pts = p;
            if valid {
                nkfs += 1;
            }
            j += 1;
        }
        if best_reproj < nprojopt {
            best_reproj = nprojopt;
            best_coinc = nkfs;
            best_kf = Some(most);
            best_scw = scw;
            best_mps = pts.clone();
            best_matched = mmp;
        }
    }
    if best_reproj > 0 {
        out.last_current_kf = Some(cur);
        out.n_coincidences = best_coinc;
        out.matched_kf2 = best_kf;
        out.set_not_erase = best_kf;
        out.scw = best_scw;
        out.mps = best_mps.into_iter().map(Some).collect();
        out.matched_mps = best_matched;
        return best_coinc >= 3;
    }
    false
}
