//! 키프레임 공시성 그래프 · 지도 점 갱신(KeyFrame::UpdateConnections · AddConnection · UpdateBestCovisibles · AddLoopEdge,
//! MapPoint::UpdateNormalAndDepth). 원본 std::map<KeyFrame*> 순회 · 정렬 동률은 번호(mnId) 순으로 둔다
//! (통합 담당의 번호 순 비교자 패치(OrderById)와 같은 뜻 — 대조 시험은 그 패치를 넣은 기준 빌드 `~/opt/orb3_refid` 와 비교).
use std::collections::{BTreeMap, BTreeSet};

use crate::orb3rs::{add, norm, sub, KeyFrame, KfId, MapPoint, MpId, V3};

/// UpdateBestCovisibles: (무게, 번호) 오름차순 정렬 뒤 뒤집고 나쁜 키프레임은 뺌
pub fn update_best_covisibles(kf: KfId, kfs: &mut BTreeMap<KfId, KeyFrame>) {
    let mut v: Vec<(i32, KfId)> = kfs[&kf].conn_weights.iter().map(|(&k, &w)| (w, k)).collect();
    v.sort();
    let oc: Vec<(KfId, i32)> = v.into_iter().rev().filter(|(_, k)| !kfs.get(k).is_some_and(|x| x.bad)).map(|(w, k)| (k, w)).collect();
    kfs.get_mut(&kf).unwrap().ordered_conn = oc;
}

/// AddConnection(pKF, weight) — `on` 에 `other` 와의 무게를 두고 바뀌었을 때만 다시 정렬
pub fn add_connection(on: KfId, other: KfId, w: i32, kfs: &mut BTreeMap<KfId, KeyFrame>) {
    let k = kfs.get_mut(&on).unwrap();
    if k.conn_weights.get(&other) == Some(&w) {
        return;
    }
    k.conn_weights.insert(other, w);
    update_best_covisibles(on, kfs);
}

/// UpdateConnections(upParent). 첫 연결(mbFirstConnection)일 때 부모를 정하는 부분은 `first_connection` 이 참이고
/// 지도 첫 키프레임이 아닐 때만(루프 · 합치기 단계의 키프레임은 이미 연결돼 있어 거짓).
pub fn update_connections(kf: KfId, kfs: &mut BTreeMap<KfId, KeyFrame>, mps: &BTreeMap<MpId, MapPoint>, first_connection: bool) {
    let me = &kfs[&kf];
    let mut counter: BTreeMap<KfId, i32> = BTreeMap::new();
    for m in me.map_points.iter().flatten() {
        let p = &mps[m];
        if p.bad {
            continue;
        }
        for (&o, _) in &p.obs {
            let ok = &kfs[&o];
            if o == kf || ok.bad || ok.map != me.map {
                continue;
            }
            *counter.entry(o).or_insert(0) += 1;
        }
    }
    if counter.is_empty() {
        return;
    }
    let (mut nmax, mut kmax) = (0, None);
    let mut pairs: Vec<(i32, KfId)> = Vec::new();
    for (&k, &w) in &counter {
        if w > nmax {
            nmax = w;
            kmax = Some(k);
        }
        if w >= 15 {
            pairs.push((w, k));
            add_connection(k, kf, w, kfs);
        }
    }
    if pairs.is_empty() {
        let k = kmax.unwrap();
        pairs.push((nmax, k));
        add_connection(k, kf, nmax, kfs);
    }
    pairs.sort();
    let me = kfs.get_mut(&kf).unwrap();
    me.conn_weights = counter;
    me.ordered_conn = pairs.into_iter().rev().map(|(w, k)| (k, w)).collect();
    if first_connection {
        let p = me.ordered_conn[0].0;
        me.parent = Some(p);
        let c = &mut kfs.get_mut(&p).unwrap().children;
        if !c.contains(&kf) {
            c.push(kf);
            c.sort();
        }
    }
}

/// AddLoopEdge(pKF): mbNotErase = true(부르는 쪽), mspLoopEdges.insert
pub fn add_loop_edge(kf: KfId, other: KfId, kfs: &mut BTreeMap<KfId, KeyFrame>) {
    let e = &mut kfs.get_mut(&kf).unwrap().loop_edges;
    if !e.contains(&other) {
        e.push(other);
        e.sort();
    }
}

/// 원본 GetConnectedKeyFrames(집합)
pub fn connected_set(kf: &KeyFrame) -> BTreeSet<KfId> {
    kf.conn_weights.keys().copied().collect()
}

/// MapPoint::UpdateNormalAndDepth. 원본은 기준 키프레임이 관측에 없으면 observations[pRefKF] 가 (0, 0) 을 넣어 0 번 점의 층을 씀
pub fn update_normal_and_depth(mp: &mut MapPoint, kfs: &BTreeMap<KfId, KeyFrame>) {
    if mp.bad || mp.obs.is_empty() {
        return;
    }
    let pos = mp.pos;
    let mut normal: V3 = [0.0; 3];
    let mut n = 0i32;
    for (k, &(l, r)) in &mp.obs {
        let c = kfs[k].camera_center();
        for idx in [l, r] {
            if idx != -1 {
                let ni = sub(&pos, &c);
                let d = norm(&ni);
                normal = add(&normal, &[ni[0] / d, ni[1] / d, ni[2] / d]);
                n += 1;
            }
        }
    }
    let rk = &kfs[&mp.ref_kf];
    let pc = sub(&pos, &rk.camera_center());
    let dist = norm(&pc);
    let left = mp.obs.get(&mp.ref_kf).map_or(0, |o| o.0);
    let level = rk.keys[left as usize].octave as usize;
    let sc = &rk.ctx.pyr.scale;
    let lsf = sc[level];
    mp.max_dist = dist * lsf;
    mp.min_dist = mp.max_dist / sc[sc.len() - 1];
    let nf = n as f32;
    mp.normal = [normal[0] / nf, normal[1] / nf, normal[2] / nf];
}
