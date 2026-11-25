//! LoopClosing::NewDetectCommonRegions(RGB-D · 관성 없음)를 원본 순서대로: 루프 · 합치기 후보 각각 앞 키프레임에서 이어 찾기
//! (DetectAndReffineSim3FromLastKF) → 못 이었으면 KeyFrameDatabase 후보(DetectNBestCandidates 3) → DetectCommonRegionsFromBoW
//! → KeyFrameDatabase 에 지금 키프레임 넣기 → 아무것도 없으면 SetErase. 계산은 포터 LOOP(`orb3loop::detect`) · CV(OptimizeSim3).
//! 원본 std::set<KeyFrame*> 주소 순 자리는 id 순(`AtlasGraph`).
use super::kfgraph::AtlasGraph;
use super::loopflow::Found;
use super::mapping::MapState;
use crate::orb3loop::crand::GlibcRand;
use crate::orb3loop::detect::{detect_and_refine_sim3_from_last_kf, detect_common_regions_from_bow, BowOut, G2oSim3, OptimizeSim3, World};
use crate::orb3loop::kfdb::{KeyFrameDatabase, KfId as DbKf};
use crate::orb3rs::mapdata::kf_set_erase;
use crate::orb3rs::types::{KfId, MpId};

/// 한 갈래(루프 또는 합치기)의 LoopClosing 칸: mpXMatchedKF · mpXLastCurrentKF · mg2oXSlw · mnXNumCoincidences · mnXNumNotFound ·
/// mvpXMPs · mvpXMatchedMPs · mbXDetected
#[derive(Clone, Debug, Default)]
pub struct Branch {
    pub matched_kf: Option<KfId>,
    pub last_current_kf: Option<KfId>,
    pub slw: G2oSim3,
    pub n_coincidences: i32,
    pub n_not_found: i32,
    pub mps: Vec<Option<MpId>>,
    pub matched_mps: Vec<Option<MpId>>,
    pub detected: bool,
}

impl Branch {
    fn bow_out(&self) -> BowOut {
        BowOut {
            matched_kf2: self.matched_kf,
            last_current_kf: self.last_current_kf,
            scw: self.slw,
            n_coincidences: self.n_coincidences,
            mps: self.mps.clone(),
            matched_mps: self.matched_mps.clone(),
            set_not_erase: None,
        }
    }
    fn take_bow(&mut self, o: BowOut) {
        self.matched_kf = o.matched_kf2;
        self.last_current_kf = o.last_current_kf;
        self.slw = o.scw;
        self.n_coincidences = o.n_coincidences;
        self.mps = o.mps;
        self.matched_mps = o.matched_mps;
    }
}

/// 앞 키프레임에서 이어 찾기(루프 · 합치기 공통 몸통). 돌려줌: 이 키프레임에서 찾음(bXDetectedInKF)
fn from_last<O: OptimizeSim3>(st: &mut MapState, b: &mut Branch, cur: KfId, is_loop: bool, opt: &mut O) -> bool {
    if b.n_coincidences <= 0 {
        return false;
    }
    let (last, matched) = (b.last_current_kf.expect("mpXLastCurrentKF"), b.matched_kf.expect("mpXMatchedKF"));
    let a = &st.atlas;
    // Sophus::SE3d mTcl = (pCur->GetPose() * pLast->GetPoseInverse()).cast<double>(); g2o::Sim3(q, t, 1)
    let tcl = a.kfs[&cur].tcw.mul(&a.kfs[&last].twc);
    let scl = G2oSim3::from_se3(&tcl);
    let scw = scl.mul(&b.slw);
    let out = {
        let w = World { kfs: &a.kfs, mps: &a.mps };
        detect_and_refine_sim3_from_last_kf(&w, cur, cur, matched, &scw, false, opt)
    };
    if out.ok {
        b.n_coincidences += 1;
        kf_set_erase(&mut st.atlas, last);
        b.last_current_kf = Some(cur);
        b.slw = scw;
        b.matched_mps = out.matched;
        b.mps = out.mps.into_iter().map(Some).collect();
        b.detected = b.n_coincidences >= 3;
        if is_loop {
            b.n_not_found = 0;
        }
        true
    } else {
        if !is_loop {
            b.detected = false;
        }
        b.n_not_found += 1;
        if b.n_not_found >= 2 {
            kf_set_erase(&mut st.atlas, last);
            kf_set_erase(&mut st.atlas, matched);
            b.n_coincidences = 0;
            b.matched_mps.clear();
            b.mps.clear();
            b.n_not_found = 0;
        }
        false
    }
}

/// NewDetectCommonRegions 상태
pub struct Detector {
    pub lp: Branch,
    pub mg: Branch,
}

impl Detector {
    pub fn new() -> Self {
        Detector { lp: Branch::default(), mg: Branch::default() }
    }

    /// 키프레임 하나(대기열에서 꺼낸 mpCurrentKF). 지도 갱신 잠금 안에서 부른다
    pub fn detect<O: OptimizeSim3>(&mut self, st: &mut MapState, db: &mut KeyFrameDatabase, rand: &mut GlibcRand, cur: KfId, opt: &mut O) -> Found {
        // SetNotErase · mpLastMap
        st.atlas.kfs.get_mut(&cur).unwrap().not_erase = true;
        let map = st.atlas.kfs[&cur].map;
        let add = |st: &MapState, db: &mut KeyFrameDatabase| {
            let k = &st.atlas.kfs[&cur];
            db.add(cur as DbKf, &k.bow.iter().map(|(w, v)| (*w, *v)).collect());
        };
        let nkfs = st.atlas.maps.iter().find(|m| m.id == map).map_or(0, |m| m.keyframes_in_map());
        if nkfs < 12 {
            add(st, db);
            kf_set_erase(&mut st.atlas, cur);
            return Found::default();
        }
        let loop_in_kf = from_last(st, &mut self.lp, cur, true, opt);
        let merge_in_kf = from_last(st, &mut self.mg, cur, false, opt);
        if self.mg.detected || self.lp.detected {
            add(st, db);
            return Found { merge: self.mg.detected, looped: self.lp.detected };
        }
        let (mut lcand, mut mcand) = (Vec::new(), Vec::new());
        if !merge_in_kf || !loop_in_kf {
            let g = AtlasGraph::new(&st.atlas);
            let (l, m) = db.detect_n_best(&g, cur as DbKf, 3);
            lcand = l.into_iter().map(|k| Some(k as KfId)).collect();
            mcand = m.into_iter().map(|k| Some(k as KfId)).collect();
        }
        let mut rnd = |a: i32, b: i32| rand.random_int(a, b);
        for (inkf, cands, b) in [(loop_in_kf, &lcand, &mut self.lp), (merge_in_kf, &mcand, &mut self.mg)] {
            if inkf || cands.is_empty() {
                continue;
            }
            let mut o = b.bow_out();
            let found = {
                let w = World { kfs: &st.atlas.kfs, mps: &st.atlas.mps };
                detect_common_regions_from_bow(&w, cur, false, cands, opt, &mut rnd, &mut o)
            };
            if let Some(k) = o.set_not_erase {
                st.atlas.kfs.get_mut(&k).unwrap().not_erase = true;
            }
            b.take_bow(o);
            b.detected = found;
        }
        add(st, db);
        if self.mg.detected || self.lp.detected {
            return Found { merge: self.mg.detected, looped: self.lp.detected };
        }
        kf_set_erase(&mut st.atlas, cur);
        Found::default()
    }
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

/// 루프 스레드 계산(`loopflow::LoopOps`) Rust 판. 공통 영역 찾기 · 받아들임 · 표지 지우기는 Rust, CorrectLoop · MergeLocal 은 빈자리.
pub struct RsLoop<O: OptimizeSim3 + Send> {
    pub det: Detector,
    pub db: std::sync::Arc<std::sync::Mutex<Option<KeyFrameDatabase>>>,
    pub rand: std::sync::Arc<std::sync::Mutex<GlibcRand>>,
    pub opt: O,
    /// 점 mnCorrectedByKF · mnCorrectedReference
    pub corrected_by: std::collections::BTreeMap<MpId, (KfId, KfId)>,
}

impl<O: OptimizeSim3 + Send> super::loopflow::LoopOps<MapState> for RsLoop<O> {
    fn detect(&mut self, map: &super::threads::MapUpdate<MapState>, kf: u64) -> Found {
        let mut g = map.lock();
        let mut db = self.db.lock().unwrap();
        let db = db.as_mut().expect("KeyFrameDatabase(set_vocabulary)");
        let mut r = self.rand.lock().unwrap();
        self.det.detect(&mut g, db, &mut r, kf as KfId, &mut self.opt)
    }
    /// 관성 없음: 늘 합침
    fn merge_accept(&mut self, _: &MapState) -> super::loopflow::MergeVerdict {
        super::loopflow::MergeVerdict::Merge
    }
    fn merge_local(&mut self, _: &mut MapState) -> Option<u64> {
        super::rsys::hole("LoopClosing::MergeLocal")
    }
    /// 관성 없음: 기울기 검사 없이 받아들임
    fn loop_accept(&mut self, _: &MapState) -> bool {
        true
    }
    /// CorrectLoop 계산(포터 LOOP `correct_loop`, 원본 2/2 비트 동일) + 본질 그래프(`essops::RsEss`, CV 판 · 원본 2/2 비트 동일).
    /// 입력 mg2oLoopScw = mg2oLoopSlw · mvpLoopMapPoints = mvpLoopMPs · mvpLoopMatchedMPs(Run 이 넘기는 그대로)
    fn correct_loop(&mut self, st: &mut MapState) -> Option<u64> {
        let b = &self.det.lp;
        let cur = b.last_current_kf.expect("mpCurrentKF");
        let matched = b.matched_kf.expect("mpLoopMatchedKF");
        let map = st.atlas.kfs[&cur].map;
        let init = st.atlas.maps.iter().find(|m| m.id == map).map(|m| m.init_kf_id);
        let inp = crate::orb3loop::correct::LoopIn { cur, matched, scw: b.slw, loop_mps: &b.mps, loop_matched: &b.matched_mps, fix_scale: true, imu_init: false };
        let mut ess = super::essops::RsEss { init_kf: init };
        let out = crate::orb3loop::correct::correct_loop(&inp, &mut st.atlas, &mut self.corrected_by, &mut ess);
        for k in out.not_erase {
            if let Some(kf) = st.atlas.kfs.get_mut(&k) {
                kf.not_erase = true;
            }
        }
        if let Some(m) = st.atlas.maps.iter_mut().find(|m| m.id == map) {
            m.increase_change_index();
        }
        out.run_gba.then_some(map as u64)
    }
    /// mpMergeLastCurrentKF->SetErase() · mpMergeMatchedKF->SetErase() · 일치 수 0 · 짝 목록 비움
    fn clear_merge(&mut self, st: &mut MapState) {
        clear(&mut self.det.mg, st);
    }
    fn clear_loop(&mut self, st: &mut MapState) {
        clear(&mut self.det.lp, st);
    }
    fn done(&mut self, _kf: u64) {}
}

fn clear(b: &mut Branch, st: &mut MapState) {
    for k in [b.last_current_kf, b.matched_kf].into_iter().flatten() {
        if st.atlas.kfs.contains_key(&k) {
            kf_set_erase(&mut st.atlas, k);
        }
    }
    b.n_coincidences = 0;
    b.matched_mps.clear();
    b.mps.clear();
    b.n_not_found = 0;
    b.detected = false;
}
