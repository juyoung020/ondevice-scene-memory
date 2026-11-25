//! LocalMapping 키프레임 단계의 Rust 판(포터 TRACK). 원본 덤프 재생으로 비트 동일 확인된 단계만 채우고
//! 나머지는 [`super::rsys::hole`] 로 멈춘다(`slam/examples/orb3rs_track_replay.rs`, docs/orb3_port_track.md).
use super::mapping::{KfSteps, MapState};
use super::rsys::hole;
use crate::orb3rs::localmap;
use crate::orb3rs::types::{KfId, MpId};

/// mlpRecentAddedMapPoints 와 설정(RGB-D: 단안 아님)
pub struct TrackSteps {
    /// LocalMapping::mlpRecentAddedMapPoints
    pub recent: Vec<MpId>,
    /// KeyFrame::mThDepth(한 카메라라 모든 키프레임 같음)
    pub th_depth: f32,
    /// KeyFrameCulling 의 mbAbortBA 읽기
    pub abort_ba: Box<dyn Fn() -> bool + Send>,
}

impl KfSteps for TrackSteps {
    fn process_new_keyframe(&mut self, _: &mut MapState, _: KfId) {
        hole("LocalMapping::ProcessNewKeyFrame")
    }
    fn map_point_culling(&mut self, st: &mut MapState, kf: KfId) {
        localmap::map_point_culling(&mut self.recent, kf, false, &mut st.atlas, &mut st.ext);
    }
    fn create_new_map_points(&mut self, _: &mut MapState, _: KfId) {
        // 삼각측량 계산(localmap::create_new_map_points)은 비트 동일 — 새 MapPoint 만들기(UpdateNormalAndDepth 등)를 아직 잇지 않음
        hole("LocalMapping::CreateNewMapPoints(새 점 지도 넣기)")
    }
    fn search_in_neighbors(&mut self, _: &mut MapState, _: KfId) {
        hole("LocalMapping::SearchInNeighbors")
    }
    fn keyframe_culling(&mut self, st: &mut MapState, kf: KfId) {
        // GetVectorCovisibleKeyFrames: 무게 순 이웃 전체
        let local: Vec<KfId> = st.atlas.kfs[&kf].ordered_conn.iter().map(|&(k, _)| k).collect();
        let th = self.th_depth;
        let abort = &self.abort_ba;
        localmap::keyframe_culling(&local, false, &|_| th, &|| abort(), &mut st.atlas, &mut st.ext);
    }
}
