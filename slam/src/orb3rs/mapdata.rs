//! 포터 OPT — 지도 자료 클래스의 메서드: KeyFrame.cc(공시성 그래프 · 신장 트리 · 지도 점 칸 · SetBadFlag),
//! MapPoint.cc(관측 · 나쁨 · 바꾸기 · 법선 · 깊이 범위), Map.cc(키프레임 · 점 넣고 빼기, 가장 낮은 id, 큰 변화 번호).
//! 원본과 비트 동일(덤프 대조: 아래 `bit` 시험, docs/orb3_port.md "지도 자료 메서드").
//!
//! 순서: 원본 std::map/set<KeyFrame*> 는 포인터 주소 순, 여기는 id 순(types.rs 약속). 결과가 순서에 닿는 자리
//! (UpdateBestCovisibles 같은 무게 동률 · UpdateConnections 최대 개수 동률 · SetBadFlag 자식 동률 · EraseObservation 새 기준)는
//! 주소 순 = id 순일 때 원본과 같다.
//! 이미 다른 포터가 옮긴 것은 그대로 씀: MapPoint::add_observation · compute_distinctive_descriptors · predict_scale(frame.rs).
//! KeyFrameDatabase 에서 빼기(SetBadFlag 끝)는 LOOP 몫이라 [`kf_set_bad_flag`] 가 true 를 돌려주면 부른 쪽이 한다.
use super::types::*;
use std::collections::BTreeMap;

/// 정렬된 id 목록에 넣기(std::set::insert)
pub fn set_insert(v: &mut Vec<u64>, x: u64) -> bool {
    match v.binary_search(&x) {
        Ok(_) => false,
        Err(i) => {
            v.insert(i, x);
            true
        }
    }
}
/// 정렬된 id 목록에서 빼기(std::set::erase)
pub fn set_remove(v: &mut Vec<u64>, x: u64) -> bool {
    match v.binary_search(&x) {
        Ok(i) => {
            v.remove(i);
            true
        }
        Err(_) => false,
    }
}

// ---------------- Map ----------------
impl Map {
    /// Map::AddKeyFrame
    pub fn add_keyframe(&mut self, kf: KfId) {
        if self.kfs.is_empty() {
            self.init_kf_id = kf;
            self.origin_kf = Some(kf);
            self.lower_kf = Some(kf);
        }
        set_insert(&mut self.kfs, kf);
        if kf > self.max_kf_id {
            self.max_kf_id = kf;
        }
        if self.lower_kf.is_some_and(|l| kf < l) {
            self.lower_kf = Some(kf);
        }
    }
    /// Map::AddMapPoint
    pub fn add_map_point(&mut self, mp: MpId) {
        set_insert(&mut self.mps, mp);
    }
    /// Map::EraseMapPoint
    pub fn erase_map_point(&mut self, mp: MpId) {
        set_remove(&mut self.mps, mp);
    }
    /// Map::EraseKeyFrame: 가장 낮은 id 를 지웠으면 남은 것 중 가장 낮은 id(lId 정렬 첫째)
    pub fn erase_keyframe(&mut self, kf: KfId) {
        set_remove(&mut self.kfs, kf);
        if !self.kfs.is_empty() {
            if self.lower_kf == Some(kf) {
                self.lower_kf = Some(self.kfs[0]);
            }
        } else {
            self.lower_kf = None;
        }
    }
    pub fn set_reference_map_points(&mut self, mps: Vec<MpId>) {
        self.ref_mps = mps;
    }
    pub fn inform_new_big_change(&mut self) {
        self.big_change_idx += 1;
    }
    pub fn last_big_change_idx(&self) -> i32 {
        self.big_change_idx
    }
    pub fn increase_change_index(&mut self) {
        self.change_id += 1;
    }
    pub fn set_last_map_change(&mut self, c: i32) {
        self.change_notified = c;
    }
    /// GetLowerKFID(없으면 0)
    pub fn lower_kf_id(&self) -> KfId {
        self.lower_kf.unwrap_or(0)
    }
    pub fn keyframes_in_map(&self) -> usize {
        self.kfs.len()
    }
    pub fn map_points_in_map(&self) -> usize {
        self.mps.len()
    }
}

impl Atlas {
    pub fn map_mut(&mut self, id: MapId) -> Option<&mut Map> {
        self.maps.iter_mut().find(|m| m.id == id)
    }
    pub fn map_ref(&self, id: MapId) -> Option<&Map> {
        self.maps.iter().find(|m| m.id == id)
    }
    fn kf_bad(&self, k: KfId) -> bool {
        self.kfs.get(&k).is_some_and(|x| x.bad)
    }
}

// ---------------- KeyFrame: 자기 칸만 보는 것 ----------------
impl KeyFrame {
    /// GetWeight
    pub fn weight(&self, o: KfId) -> i32 {
        self.conn_weights.get(&o).copied().unwrap_or(0)
    }
    /// GetConnectedKeyFrames(집합 순)
    pub fn connected_keyframes(&self) -> Vec<KfId> {
        self.conn_weights.keys().copied().collect()
    }
    /// GetVectorCovisibleKeyFrames
    pub fn vector_covisible(&self) -> Vec<KfId> {
        self.ordered_conn.iter().map(|x| x.0).collect()
    }
    /// GetBestCovisibilityKeyFrames(N)
    pub fn best_covisibility(&self, n: i32) -> Vec<KfId> {
        let v = self.vector_covisible();
        if (v.len() as i32) < n { v } else { v[..n.max(0) as usize].to_vec() }
    }
    /// GetCovisiblesByWeight(w): 무게 내림차순 목록에서 upper_bound(w, a>b) 앞까지
    pub fn covisibles_by_weight(&self, w: i32) -> Vec<KfId> {
        if self.ordered_conn.is_empty() {
            return Vec::new();
        }
        let n = self.ordered_conn.partition_point(|&(_, e)| !(w > e));
        if n == self.ordered_conn.len() && self.ordered_conn.last().unwrap().1 < w {
            return Vec::new();
        }
        self.ordered_conn[..n].iter().map(|x| x.0).collect()
    }
    /// GetNumberMPs
    pub fn number_mps(&self) -> i32 {
        self.map_points.iter().filter(|m| m.is_some()).count() as i32
    }
    /// AddMapPoint · ReplaceMapPointMatch
    pub fn add_map_point(&mut self, mp: MpId, idx: usize) {
        self.map_points[idx] = Some(mp);
    }
    /// EraseMapPointMatch(idx)
    pub fn erase_map_point_match_idx(&mut self, idx: usize) {
        self.map_points[idx] = None;
    }
    pub fn add_child(&mut self, c: KfId) {
        set_insert(&mut self.children, c);
    }
    pub fn erase_child(&mut self, c: KfId) {
        set_remove(&mut self.children, c);
    }
    pub fn has_child(&self, c: KfId) -> bool {
        self.children.binary_search(&c).is_ok()
    }
    /// AddLoopEdge: 지우지 말기 표시 + 집합에 넣기
    pub fn add_loop_edge(&mut self, k: KfId) {
        self.not_erase = true;
        set_insert(&mut self.loop_edges, k);
    }
    pub fn add_merge_edge(&mut self, k: KfId) {
        self.not_erase = true;
        set_insert(&mut self.merge_edges, k);
    }
    pub fn set_not_erase(&mut self) {
        self.not_erase = true;
    }
}

// ---------------- KeyFrame: 다른 키프레임 · 점을 건드리는 것 ----------------

/// UpdateBestCovisibles: (무게, kf) 오름차순 정렬 뒤 앞에 끼워 넣기 = 내림차순, 나쁜 kf 는 뺌
pub fn kf_update_best_covisibles(a: &mut Atlas, kf: KfId) {
    let mut v: Vec<(i32, KfId)> = a.kfs[&kf].conn_weights.iter().map(|(&k, &w)| (w, k)).collect();
    v.sort_unstable();
    let ordered: Vec<(KfId, i32)> = v.iter().rev().filter(|(_, k)| !a.kf_bad(*k)).map(|&(w, k)| (k, w)).collect();
    a.kfs.get_mut(&kf).unwrap().ordered_conn = ordered;
}

/// AddConnection: 무게가 같으면 아무것도 안 함
pub fn kf_add_connection(a: &mut Atlas, kf: KfId, o: KfId, w: i32) {
    let k = a.kfs.get_mut(&kf).unwrap();
    if k.conn_weights.get(&o) == Some(&w) {
        return;
    }
    k.conn_weights.insert(o, w);
    kf_update_best_covisibles(a, kf);
}

/// EraseConnection
pub fn kf_erase_connection(a: &mut Atlas, kf: KfId, o: KfId) {
    if a.kfs.get_mut(&kf).unwrap().conn_weights.remove(&o).is_some() {
        kf_update_best_covisibles(a, kf);
    }
}

/// UpdateConnections(upParent 는 원본에서 출력만 바꿈): 지도 점을 같이 보는 같은 지도의 키프레임 수를 세어
/// 15 이상(없으면 최대 하나)과 양쪽 연결, 공시성 표는 센 것 전부, 처음이면 가장 무거운 것을 부모로.
pub fn kf_update_connections(a: &mut Atlas, kf: KfId, _up_parent: bool) {
    let mut counter: BTreeMap<KfId, i32> = BTreeMap::new();
    {
        let k = &a.kfs[&kf];
        for mp in k.map_points.iter().flatten() {
            let Some(p) = a.mps.get(mp) else { continue };
            if p.bad {
                continue;
            }
            for &o in p.obs.keys() {
                let Some(ok) = a.kfs.get(&o) else { continue };
                if o == kf || ok.bad || ok.map != k.map {
                    continue;
                }
                *counter.entry(o).or_insert(0) += 1;
            }
        }
    }
    if counter.is_empty() {
        return;
    }
    const TH: i32 = 15;
    let (mut nmax, mut kmax) = (0, 0);
    let mut pairs: Vec<(i32, KfId)> = Vec::with_capacity(counter.len());
    for (&o, &c) in &counter {
        if c > nmax {
            nmax = c;
            kmax = o;
        }
        if c >= TH {
            pairs.push((c, o));
            kf_add_connection(a, o, kf, c);
        }
    }
    if pairs.is_empty() {
        pairs.push((nmax, kmax));
        kf_add_connection(a, kmax, kf, nmax);
    }
    pairs.sort_unstable();
    let ordered: Vec<(KfId, i32)> = pairs.iter().rev().map(|&(w, k)| (k, w)).collect();
    let map = a.kfs[&kf].map;
    let init = a.map_ref(map).map(|m| m.init_kf_id);
    let k = a.kfs.get_mut(&kf).unwrap();
    k.conn_weights = counter;
    k.ordered_conn = ordered;
    if !k.first_conn_done && Some(kf) != init {
        let p = k.ordered_conn[0].0;
        k.parent = Some(p);
        k.first_conn_done = true;
        a.kfs.get_mut(&p).unwrap().add_child(kf);
    }
}

/// ChangeParent(원본은 자기 자신이면 예외)
pub fn kf_change_parent(a: &mut Atlas, kf: KfId, p: KfId) {
    assert_ne!(kf, p, "The parent and child can not be the same");
    a.kfs.get_mut(&kf).unwrap().parent = Some(p);
    a.kfs.get_mut(&p).unwrap().add_child(kf);
}

/// EraseMapPointMatch(MapPoint*): 점이 기억하는 이 kf 번호 칸을 비움
pub fn kf_erase_map_point_match(a: &mut Atlas, kf: KfId, mp: MpId) {
    let (l, r) = a.mps.get(&mp).and_then(|p| p.obs.get(&kf).copied()).unwrap_or((-1, -1));
    let k = a.kfs.get_mut(&kf).unwrap();
    for i in [l, r] {
        if i != -1 {
            k.map_points[i as usize] = None;
        }
    }
}

/// TrackedMapPoints(minObs)
pub fn kf_tracked_map_points(a: &Atlas, kf: KfId, min_obs: i32) -> i32 {
    let k = &a.kfs[&kf];
    let mut n = 0;
    for i in 0..k.keys.len() {
        if let Some(mp) = k.map_points[i] {
            let p = &a.mps[&mp];
            if !p.bad && (min_obs <= 0 || p.n_obs >= min_obs) {
                n += 1;
            }
        }
    }
    n
}

/// SetErase: 루프 간선이 없으면 지우기 허용, 지우기 미뤄 둔 것이면 지금 SetBadFlag
pub fn kf_set_erase(a: &mut Atlas, kf: KfId) -> bool {
    let k = a.kfs.get_mut(&kf).unwrap();
    if k.loop_edges.is_empty() {
        k.not_erase = false;
    }
    if k.to_be_erased { kf_set_bad_flag(a, kf) } else { false }
}

/// SetBadFlag. 지도 첫 키프레임이면 그대로, 지우지 말기면 미룸(false). 아니면 연결 끊기 · 점 관측 지우기 ·
/// 신장 트리 다시 잇기(자식마다 부모 후보와의 공시성 무게가 가장 큰 짝부터) · 부모 기준 위치 mTcp · 지도에서 빼기 → true
/// (KeyFrameDatabase 에서 빼기는 부른 쪽).
pub fn kf_set_bad_flag(a: &mut Atlas, kf: KfId) -> bool {
    {
        let map = a.kfs[&kf].map;
        let init = a.map_ref(map).map(|m| m.init_kf_id);
        let k = a.kfs.get_mut(&kf).unwrap();
        if Some(kf) == init {
            return false;
        } else if k.not_erase {
            k.to_be_erased = true;
            return false;
        }
    }
    let conns = a.kfs[&kf].connected_keyframes();
    for o in conns {
        if a.kfs.contains_key(&o) {
            kf_erase_connection(a, o, kf);
        }
    }
    let n = a.kfs[&kf].map_points.len();
    for i in 0..n {
        if let Some(mp) = a.kfs[&kf].map_points[i] {
            mp_erase_observation(a, mp, kf);
        }
    }
    let parent = {
        let k = a.kfs.get_mut(&kf).unwrap();
        k.conn_weights.clear();
        k.ordered_conn.clear();
        k.parent
    };
    let mut cands: Vec<KfId> = parent.into_iter().collect();
    let mut children = a.kfs[&kf].children.clone();
    while !children.is_empty() {
        let mut best: Option<(KfId, KfId)> = None;
        let mut max = -1;
        for &c in &children {
            let ck = &a.kfs[&c];
            if ck.bad {
                continue;
            }
            for &(v, _) in &ck.ordered_conn {
                for &cand in &cands {
                    if v == cand {
                        let w = ck.weight(v);
                        if w > max {
                            best = Some((c, v));
                            max = w;
                        }
                    }
                }
            }
        }
        match best {
            Some((c, p)) => {
                kf_change_parent(a, c, p);
                set_insert(&mut cands, c);
                set_remove(&mut children, c);
            }
            None => break,
        }
    }
    if let Some(p) = parent {
        for &c in &children {
            kf_change_parent(a, c, p);
        }
    }
    a.kfs.get_mut(&kf).unwrap().children = children;
    if let Some(p) = parent {
        let ptwc = {
            let pk = a.kfs.get_mut(&p).unwrap();
            pk.erase_child(kf);
            pk.twc
        };
        let k = a.kfs.get_mut(&kf).unwrap();
        k.tcp = k.tcw.mul(&ptwc);
    }
    let map = {
        let k = a.kfs.get_mut(&kf).unwrap();
        k.bad = true;
        k.map
    };
    if let Some(m) = a.map_mut(map) {
        m.erase_keyframe(kf);
    }
    true
}

// ---------------- MapPoint ----------------
impl MapPoint {
    pub fn increase_visible(&mut self, n: i32) {
        self.visible += n;
    }
    pub fn increase_found(&mut self, n: i32) {
        self.found += n;
    }
    /// GetFoundRatio
    pub fn found_ratio(&self) -> f32 {
        self.found as f32 / self.visible as f32
    }
    pub fn is_in_keyframe(&self, kf: KfId) -> bool {
        self.obs.contains_key(&kf)
    }
    /// GetIndexInKeyFrame
    pub fn index_in_keyframe(&self, kf: KfId) -> (i32, i32) {
        self.obs.get(&kf).copied().unwrap_or((-1, -1))
    }
}

/// MapPoint::EraseObservation: 관측 수 빼기(오른쪽 좌표 있으면 2), 기준 kf 를 지웠으면 남은 첫째로, 2 이하면 SetBadFlag.
/// (관측이 다 없어지면 원본은 빈 map 의 begin 을 읽는 미정의 동작 — 기준은 그대로 둔다. 그때 점은 어차피 나쁨이 됨.)
pub fn mp_erase_observation(a: &mut Atlas, mp: MpId, kf: KfId) {
    let Some(p) = a.mps.get_mut(&mp) else { return };
    let Some((l, r)) = p.obs.remove(&kf) else { return };
    if l != -1 {
        let stereo = a.kfs.get(&kf).is_some_and(|k| k.u_right[l as usize] >= 0.0);
        p.n_obs -= if stereo { 2 } else { 1 };
    }
    if r != -1 {
        p.n_obs -= 1;
    }
    if p.ref_kf == kf {
        if let Some(&first) = p.obs.keys().next() {
            p.ref_kf = first;
        }
    }
    if p.n_obs <= 2 {
        mp_set_bad_flag(a, mp);
    }
}

/// MapPoint::SetBadFlag: 나쁨, 관측 비우고 그 kf 칸들 비우기, 지도에서 빼기
pub fn mp_set_bad_flag(a: &mut Atlas, mp: MpId) {
    let Some(p) = a.mps.get_mut(&mp) else { return };
    p.bad = true;
    let obs = std::mem::take(&mut p.obs);
    let map = p.map;
    for (kf, (l, r)) in obs {
        if let Some(k) = a.kfs.get_mut(&kf) {
            for i in [l, r] {
                if i != -1 {
                    k.map_points[i as usize] = None;
                }
            }
        }
    }
    if let Some(m) = a.map_mut(map) {
        m.erase_map_point(mp);
    }
}

/// MapPoint::Replace(old → new): 관측을 new 로 옮기고(이미 보던 kf 는 칸 비움) 찾은 · 보인 수 더하고 기술자 다시, old 는 지도에서 빼기
pub fn mp_replace(a: &mut Atlas, old: MpId, new: MpId) {
    if old == new {
        return;
    }
    let (obs, nvisible, nfound, map) = {
        let o = a.mps.get_mut(&old).unwrap();
        o.bad = true;
        o.replaced = Some(new);
        (std::mem::take(&mut o.obs), o.visible, o.found, o.map)
    };
    for (k, (l, r)) in obs {
        let in_kf = a.mps[&new].obs.contains_key(&k);
        for idx in [l, r] {
            if idx == -1 {
                continue;
            }
            if !in_kf {
                a.kfs.get_mut(&k).unwrap().map_points[idx as usize] = Some(new);
                let kk = &a.kfs[&k];
                a.mps.get_mut(&new).unwrap().add_observation(kk, idx as usize);
            } else {
                a.kfs.get_mut(&k).unwrap().map_points[idx as usize] = None;
            }
        }
    }
    let n = a.mps.get_mut(&new).unwrap();
    n.found += nfound;
    n.visible += nvisible;
    n.compute_distinctive_descriptors(&a.kfs);
    if let Some(m) = a.map_mut(map) {
        m.erase_map_point(old);
    }
}

/// MapPoint::UpdateNormalAndDepth: 관측 카메라 중심 → 점 단위 벡터의 평균, 기준 kf 거리 × 그 층 배율 = 최대 거리
pub fn mp_update_normal_and_depth(a: &mut Atlas, mp: MpId) {
    let p = &a.mps[&mp];
    if p.bad || p.obs.is_empty() {
        return;
    }
    let pos = p.pos;
    let mut normal = [0.0f32; 3];
    let mut n = 0i32;
    for (k, &(l, r)) in &p.obs {
        if l != -1 {
            let ni = sub(&pos, &a.kfs[k].camera_center());
            let nn = norm(&ni);
            normal = add(&normal, &[ni[0] / nn, ni[1] / nn, ni[2] / nn]);
            n += 1;
        }
        assert!(r == -1, "오른쪽 카메라 관측(RGB-D 에는 없음)");
    }
    let rk = &a.kfs[&p.ref_kf];
    let dist = norm(&sub(&pos, &rk.camera_center()));
    let (l, _) = p.index_in_keyframe(p.ref_kf);
    let level = rk.keys[l as usize].octave as usize;
    let pyr = &rk.ctx.pyr;
    let max = dist * pyr.scale[level];
    let min = max / pyr.scale[pyr.nlevels as usize - 1];
    let nf = n as f32;
    let p = a.mps.get_mut(&mp).unwrap();
    p.max_dist = max;
    p.min_dist = min;
    p.normal = [normal[0] / nf, normal[1] / nf, normal[2] / nf];
}

#[cfg(test)]
#[path = "mapdata_bit.rs"]
mod bit;
