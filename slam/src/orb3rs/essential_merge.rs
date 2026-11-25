//! 포터 OPT — `Optimizer::OptimizeEssentialGraph(pCurKF, vpFixedKFs, vpFixedCorrectedKFs, vpNonFixedKFs, vpNonCorrectedMPs)`
//! (LoopClosing::MergeLocal 이 지도 합치기 뒤 부르는 판)을 Rust 로. g2o 경로는 [`super::sim3pg`].
//! 원본과 비트 동일(덤프 대조: 아래 `bit`, docs/orb3_port.md "합치기 본질 그래프").
//!
//! 원본 그대로 옮긴 동작(겉보기에 이상해도):
//! - 정점: 고정(척도 고정) → 고정 보정(합치기 전 위치를 vScw 로) → 비고정 순. 같은 id 정점을 두 번 넣으면 g2o 가 뒤엣것을 버리지만
//!   좋음 · 나쁨 표시와 vScw · vCorrectedSwc 는 덮어씀.
//! - 간선 만들기는 세 목록을 이어 붙인 것(중복 · 나쁜 kf 포함)을 돈다. 정점이 없는 kf 와의 간선은 g2o 가 받지 않음.
//! - 루프 간선은 집합 순서(원본 주소 순, 여기 id 순)로 넣고, 간선 순서가 헤세 합 순서라 결과 비트에 닿는다.
//! - 결과 반영은 비고정 목록 전부(고정 보정에 이미 있어 건너뛴 kf 도 다시 SetPose · 합치기 전 위치 덮어씀).
use super::mapdata::{mp_erase_observation, mp_update_normal_and_depth};
use super::sim3pg::{optimize, PgEdge, PgVertex, PoseGraph};
use super::types::*;
use crate::orb3ba::sim3::Sim3;
use std::collections::{BTreeMap, BTreeSet};

/// Sophus SO3d(Quaterniond) 정규화(묶음 제곱합 (x²+z²)+(y²+w²), 칸마다 나눔)
fn q64_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = ((q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3])).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}
/// g2o::Sim3(Tcw.cast<double>().unit_quaternion(), translation, 1)
fn sim_of(t: &Se3) -> Sim3 {
    let q = q64_normalize(t.q.map(|x| x as f64));
    Sim3 { q, t: t.t.map(|x| x as f64), s: 1.0 }
}
/// Sophus::SE3d(S.rotation(), S.translation() / s).cast<float>()
fn se3_of(s: &Sim3) -> Se3 {
    let q = q64_normalize(s.q);
    let t = [s.t[0] / s.s, s.t[1] / s.s, s.t[2] / s.s];
    Se3::new(q.map(|x| x as f32), t.map(|x| x as f32))
}

pub fn optimize_essential_graph_merge(a: &mut Atlas, _cur_kf: KfId, fixed: &[KfId], fixed_corrected: &[KfId], non_fixed: &[KfId], non_corrected_mps: &[MpId]) {
    let bad = |a: &Atlas, k: KfId| a.kfs[&k].bad;
    let mut g = PoseGraph { lambda_init: 1e-16, ..Default::default() };
    let mut vidx: BTreeMap<KfId, usize> = BTreeMap::new();
    let mut scw: BTreeMap<KfId, Sim3> = BTreeMap::new();
    let mut corrected_swc: BTreeMap<KfId, Sim3> = BTreeMap::new();
    let mut good: BTreeMap<KfId, bool> = BTreeMap::new();
    let mut badp: BTreeMap<KfId, bool> = BTreeMap::new();
    let mut add_vertex = |g: &mut PoseGraph, vidx: &mut BTreeMap<KfId, usize>, id: KfId, fixed: bool, fs: bool, est: Sim3| {
        if !vidx.contains_key(&id) {
            vidx.insert(id, g.vertices.len());
            g.vertices.push(PgVertex { id: id as i32, fixed, fix_scale: fs, est });
        }
    };
    for &k in fixed {
        if bad(a, k) {
            continue;
        }
        let siw = sim_of(&a.kfs[&k].tcw);
        corrected_swc.insert(k, siw.inverse());
        add_vertex(&mut g, &mut vidx, k, true, true, siw);
        good.insert(k, true);
        badp.insert(k, false);
    }
    let mut sid: BTreeSet<KfId> = BTreeSet::new();
    for &k in fixed_corrected {
        if bad(a, k) {
            continue;
        }
        let siw = sim_of(&a.kfs[&k].tcw);
        corrected_swc.insert(k, siw.inverse());
        scw.insert(k, sim_of(&a.kfs[&k].tcw_bef_merge));
        add_vertex(&mut g, &mut vidx, k, true, false, siw);
        sid.insert(k);
        good.insert(k, true);
        badp.insert(k, true);
    }
    for &k in non_fixed {
        if bad(a, k) || sid.contains(&k) {
            continue;
        }
        let siw = sim_of(&a.kfs[&k].tcw);
        scw.insert(k, siw);
        add_vertex(&mut g, &mut vidx, k, false, false, siw);
        sid.insert(k);
        good.insert(k, false);
        badp.insert(k, true);
    }
    let all: Vec<KfId> = fixed.iter().chain(fixed_corrected).chain(non_fixed).copied().collect();
    let sp: BTreeSet<KfId> = all.iter().copied().collect();
    let gd = |k: KfId| good.get(&k).copied().unwrap_or(false);
    let bd = |k: KfId| badp.get(&k).copied().unwrap_or(false);
    let scw_of = |k: KfId| scw.get(&k).copied().unwrap_or_else(Sim3::identity);
    let cswc_of = |k: KfId| corrected_swc.get(&k).copied().unwrap_or_else(Sim3::identity);
    let mut add_edge = |g: &mut PoseGraph, i: KfId, j: KfId, m: Sim3| {
        if let (Some(&v0), Some(&v1)) = (vidx.get(&i), vidx.get(&j)) {
            g.edges.push(PgEdge { v0, v1, meas: m });
        }
    };
    for &ki in &all {
        let swi = if bd(ki) { scw_of(ki).inverse() } else { Sim3::identity() };
        let kf = &a.kfs[&ki];
        let parent = kf.parent;
        let relation = |kj: KfId| -> Option<Sim3> {
            if gd(ki) && gd(kj) {
                Some(cswc_of(kj).inverse())
            } else if bd(ki) && bd(kj) {
                Some(scw_of(kj))
            } else {
                None
            }
        };
        if let Some(p) = parent.filter(|p| sp.contains(p)) {
            if let Some(sjw) = relation(p) {
                add_edge(&mut g, ki, p, sjw.mul(&swi));
            }
        }
        let loops = kf.loop_edges.clone();
        for &l in &loops {
            if sp.contains(&l) && l < ki {
                if let Some(slw) = relation(l) {
                    add_edge(&mut g, ki, l, slw.mul(&swi));
                }
            }
        }
        for n in kf.covisibles_by_weight(100) {
            if Some(n) != parent && !kf.has_child(n) && !loops.contains(&n) && sp.contains(&n) && !bad(a, n) && n < ki {
                if let Some(snw) = relation(n) {
                    add_edge(&mut g, ki, n, snw.mul(&swi));
                }
            }
        }
    }
    let out = optimize(&g, 20);
    for &k in non_fixed {
        if bad(a, k) {
            continue;
        }
        let s = out.est[vidx[&k]];
        let kf = a.kfs.get_mut(&k).unwrap();
        kf.tcw_bef_merge = kf.tcw;
        kf.twc_bef_merge = kf.twc;
        kf.set_pose(se3_of(&s));
    }
    for &mp in non_corrected_mps {
        if a.mps[&mp].bad {
            continue;
        }
        let mut r = a.mps[&mp].ref_kf;
        while a.kfs[&r].bad {
            mp_erase_observation(a, mp, r);
            let r2 = a.mps[&mp].ref_kf;
            if r2 == r {
                // 관측이 다 없어져 기준이 안 바뀜(원본은 빈 map 의 begin 을 읽는 미정의 동작으로 무한 반복) — 그만둠
                break;
            }
            r = r2;
        }
        if bd(r) {
            let rk = &a.kfs[&r];
            let t = rk.twc.mul(&rk.twc_bef_merge.inverse());
            let p = t.mul_point(&a.mps[&mp].pos);
            a.mps.get_mut(&mp).unwrap().pos = p;
            mp_update_normal_and_depth(a, mp);
        }
    }
}
