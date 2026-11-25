//! 패치 옮겨 붙이기(`slam/cpp/ORB_SLAM3.patch` 의 Tracking::TryGraft · DoPendingGraft · GraftMap · RequestGraft) Rust 판.
//! 지금 지도가 다른 지도에 같은 변환으로 두 번 재위치되면 지금 지도를 그 지도 좌표로 옮겨 하나로 합친다.
//! 계산 순서는 원본 Sophus · Eigen float 식 그대로(`orb3rs::types::Se3`).
use crate::orb3rs::types::{quat_from_mat, Atlas, KfId, MapId, Se3};

/// Tracking 의 옮겨 붙이기 칸(mpGraftMap · mGraftS · mnGraftFrame · mnGraftTry · mpPendFrom/Into · mPendS)
#[derive(Clone, Debug, Default)]
pub struct GraftState {
    pub graft_map: Option<MapId>,
    pub graft_s: Option<Se3>,
    pub graft_frame: u64,
    pub graft_try: u64,
    /// (from, into, S): 다음 장면 시작에 합침
    pub pending: Option<(MapId, MapId, Se3)>,
}

/// TryGraft 의 판단: 지도 pM 에 재위치된 자세로 S(pM ← 지금 지도)를 받아 다음에 할 일을 정함.
/// 돌려줌: true 면 합치기 예약(pending 설정), false 면 이 장면은 여기서 끝(첫 관측 기록 또는 어긋남)
pub fn try_graft_step(g: &mut GraftState, frame_id: u64, cur: MapId, m: MapId, s: Se3) -> bool {
    if g.graft_map != Some(m) || frame_id > g.graft_frame + 30 {
        g.graft_map = Some(m);
        g.graft_s = Some(s);
        g.graft_frame = frame_id;
        return false;
    }
    let (dt, da) = graft_delta(&g.graft_s.expect("앞 변환"), &s);
    g.graft_s = Some(s);
    g.graft_frame = frame_id;
    if dt > 0.10f32 || da > 2.0f32 {
        return false;
    }
    g.pending = Some((cur, m, s));
    g.graft_map = None;
    true
}

/// D = A⁻¹·B 의 (이동 크기, 회전 각 도)
pub fn graft_delta(a: &Se3, b: &Se3) -> (f32, f32) {
    let d = a.inverse().mul(b);
    // float × 180.f / M_PI(double) → double 나눗셈 뒤 float
    (norm3(&d.t), ((so3_log_norm(&d.q) * 180.0f32) as f64 / std::f64::consts::PI) as f32)
}

/// RequestGraft(바깥 요청): 이미 예약이 있거나 지도가 없으면 거절
pub fn request_graft(g: &mut GraftState, a: &Atlas, from: MapId, into: MapId, s: Se3) -> bool {
    let has = |id: MapId| a.maps.iter().any(|m| m.id == id);
    if !has(from) || !has(into) || from == into || g.pending.is_some() {
        return false;
    }
    g.pending = Some((from, into, s));
    true
}

/// Eigen 크기 3 squaredNorm(반 나눔 순서 a0 + (a1 + a2))
fn sq3(v: &[f32; 3]) -> f32 {
    v[0] * v[0] + (v[1] * v[1] + v[2] * v[2])
}
fn norm3(v: &[f32; 3]) -> f32 {
    sq3(v).sqrt()
}

/// Sophus SO3f::log().norm()(logAndTheta 의 ω 크기)
pub fn so3_log_norm(q: &[f32; 4]) -> f32 {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let sn = sq3(&[x, y, z]);
    let n = sn.sqrt();
    // Sophus Constants<float>::epsilon() = 1e-5
    let eps = 1e-5f32;
    let two_atan_nbyw_by_n = if sn < eps * eps {
        let sw = w * w;
        2.0f32 / w - 2.0f32 / 3.0f32 * sn / (w * sw)
    } else if w.abs() < eps {
        if w > 0.0 {
            std::f32::consts::PI / n
        } else {
            -std::f32::consts::PI / n
        }
    } else {
        2.0f32 * (n / w).atan() / n
    };
    let o = [two_atan_nbyw_by_n * x, two_atan_nbyw_by_n * y, two_atan_nbyw_by_n * z];
    norm3(&o)
}

/// YawOnly: z 축 방위 + 이동만(Eigen AngleAxisf(yaw, UnitZ).toRotationMatrix() → SE3f(R, t))
pub fn yaw_only(s: &Se3) -> Se3 {
    let r = s.rot();
    // 원본 atan2(float, float) 는 이름공간 없는 C atan2(double) 로 풀림
    let yaw = (r[1][0] as f64).atan2(r[0][0] as f64) as f32;
    let (sn, c) = (yaw.sin(), yaw.cos());
    // AngleAxis::toRotationMatrix, 축 (0, 0, 1): 대각 tmp·axis + c, (2, 2) = (1 − c) + c
    let one_c = 1.0f32 - c;
    let m = [[0.0f32 * 0.0 + c, 0.0f32 * 0.0 - sn, 0.0f32 * 1.0 + 0.0], [0.0f32 * 0.0 + sn, 0.0f32 * 0.0 + c, 0.0f32 * 1.0 - 0.0], [one_c * 0.0 - 0.0, one_c * 0.0 + 0.0, one_c * 1.0 + c]];
    Se3 { q: quat_from_rows_rev(&m), t: s.t }
}

/// GraftMap: from 지도의 키프레임 · 점을 S(into ← from, 방위만)로 옮겨 into 에 넣고 from 을 지움.
/// 돌려줌: 옮긴 키프레임 수
pub fn graft_map(a: &mut Atlas, from: MapId, into: MapId, s0: &Se3) -> usize {
    let s = yaw_only(s0);
    let sinv = s.inverse();
    let fi = a.maps.iter().position(|m| m.id == from).expect("from 지도");
    let ii = a.maps.iter().position(|m| m.id == into).expect("into 지도");
    let kfs = std::mem::take(&mut a.maps[fi].kfs);
    let mps = std::mem::take(&mut a.maps[fi].mps);
    let mut origin: Option<KfId> = None;
    for &k in &kfs {
        let kf = a.kfs.get_mut(&k).expect("키프레임");
        if !kf.bad && origin.is_none_or(|o| k < o) {
            origin = Some(k);
        }
        let p = kf.tcw.mul(&sinv);
        kf.set_pose(p);
        kf.map = into;
        a.maps[ii].kfs.push(k);
    }
    for &m in &mps {
        let mp = a.mps.get_mut(&m).expect("지도 점");
        mp.pos = s.mul_point(&mp.pos);
        mp.normal = s.rotate(&mp.normal);
        mp.map = into;
        a.maps[ii].mps.push(m);
    }
    // 옮긴 지도의 뿌리를 합친 지도의 가장 가까운 키프레임에 매달기(원본은 포인터 순 순회 — 같은 거리면 앞, Rust 는 번호 순)
    if let Some(o) = origin {
        if a.kfs[&o].parent.is_none() {
            let c = a.kfs[&o].camera_center();
            let mut best: Option<KfId> = None;
            let mut bd = 1e9f32;
            let moved: std::collections::BTreeSet<KfId> = kfs.iter().copied().collect();
            let mut ids: Vec<KfId> = a.maps[ii].kfs.clone();
            ids.sort_unstable();
            for k in ids {
                let kf = &a.kfs[&k];
                if kf.bad || moved.contains(&k) {
                    continue;
                }
                let cc = kf.camera_center();
                let d = norm3(&[cc[0] - c[0], cc[1] - c[1], cc[2] - c[2]]);
                if d < bd {
                    bd = d;
                    best = Some(k);
                }
            }
            if let Some(b) = best {
                // ChangeParent: 부모 칸 + 새 부모의 자식 목록
                a.kfs.get_mut(&o).unwrap().parent = Some(b);
                let ch = &mut a.kfs.get_mut(&b).unwrap().children;
                if !ch.contains(&o) {
                    ch.push(o);
                }
            }
        }
    }
    // SetMapBad · IncreaseChangeIndex · RemoveBadMaps
    a.maps[ii].change_id += 1;
    a.maps[fi].bad = true;
    let cur_id = a.maps[a.current].id;
    a.maps.retain(|m| !m.bad);
    a.current = a.maps.iter().position(|m| m.id == cur_id).unwrap_or(0);
    kfs.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 원본 계산(기준 빌드 플래그 probe, WSL ~/opt/orb3_sys/graft/probe) 5 만 건과 비트 비교:
    /// YawOnly · 키프레임 자세 × S⁻¹ · S × 점 · S.so3() × 법선 · TryGraft 의 dt · da
    /// ORB3GRAFT=/tmp/graft.bin cargo test --release -p slam --lib orb3sys::graft
    #[test]
    fn graft_math_matches_original() {
        let p = std::env::var("ORB3GRAFT").expect("입력 없음: ORB3GRAFT(probe 출력)");
        let b = std::fs::read(p).unwrap();
        let f: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        assert!(!f.is_empty());
        let se = |v: &[f32]| Se3 { q: [v[0], v[1], v[2], v[3]], t: [v[4], v[5], v[6]] };
        let bits = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let flat = |s: &Se3| [s.q[0], s.q[1], s.q[2], s.q[3], s.t[0], s.t[1], s.t[2]];
        let mut bad = [0usize; 6];
        let n = f.len() / 56;
        for r in f.chunks_exact(56) {
            let (s0, p, a, bb) = (se(&r[0..7]), se(&r[7..14]), se(&r[14..21]), se(&r[21..28]));
            let x = [r[28], r[29], r[30]];
            let nv = [r[31], r[32], r[33]];
            let s = yaw_only(&s0);
            let (dt, da) = graft_delta(&a, &bb);
            let got: [Vec<f32>; 6] = [flat(&s).to_vec(), flat(&p.mul(&s.inverse())).to_vec(), s.mul_point(&x).to_vec(), s.rotate(&nv).to_vec(), vec![dt], vec![da]];
            let want: [&[f32]; 6] = [&r[34..41], &r[41..48], &r[48..51], &r[51..54], &r[54..55], &r[55..56]];
            for k in 0..6 {
                if !bits(&got[k], want[k]) {
                    if bad[k] < 3 {
                        eprintln!("{k}: 우리 {:?} 원본 {:?}", got[k], want[k]);
                    }
                    bad[k] += 1;
                }
            }
        }
        eprintln!("{n} 건, 다름 {:?}", bad);
        assert_eq!(bad, [0; 6]);
    }
}

/// Eigen Quaternionf(Matrix3f) — 이 자리(GraftMap 의 YawOnly) 대각 합 순서 (m22 + m11) + m00
pub fn quat_from_rows_rev(m: &[[f32; 3]; 3]) -> [f32; 4] {
    let t = (m[2][2] + m[1][1]) + m[0][0];
    if t > 0.0 {
        let t = (t + 1.0).sqrt();
        let h = 0.5 / t;
        [(m[2][1] - m[1][2]) * h, (m[0][2] - m[2][0]) * h, (m[1][0] - m[0][1]) * h, 0.5 * t]
    } else {
        quat_from_mat(m)
    }
}
