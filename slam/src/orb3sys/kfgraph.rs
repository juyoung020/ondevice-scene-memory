//! KeyFrameDatabase 가 보는 키프레임 그래프(`orb3loop::kfdb::KfGraph`)를 Atlas 위에: DB 번호는 u32, 원본 주소 순 자리는 id 순.
use crate::orb3loop::dbow2::BowVector;
use crate::orb3loop::kfdb::{KfGraph, KfId as DbKf, MapId as DbMap};
use crate::orb3rs::types::{Atlas, KfId};
use std::collections::HashMap;

/// KeyFrameDatabase 가 보는 키프레임 그래프(Atlas 위, DB 번호는 u32)
pub struct AtlasGraph<'a> {
    a: &'a Atlas,
    /// 질의가 닿은 키프레임만 그때 바꿈(전부 미리 바꾸면 키프레임 900 개에서 재위치 한 번에 약 100 ms)
    bow: HashMap<DbKf, std::cell::OnceCell<BowVector>>,
    covis: HashMap<DbKf, std::cell::OnceCell<Vec<DbKf>>>,
}

impl<'a> AtlasGraph<'a> {
    pub fn new(a: &'a Atlas) -> Self {
        let bow = a.kfs.keys().map(|k| (*k as DbKf, std::cell::OnceCell::new())).collect();
        let covis = a.kfs.keys().map(|k| (*k as DbKf, std::cell::OnceCell::new())).collect();
        AtlasGraph { a, bow, covis }
    }
}

impl KfGraph for AtlasGraph<'_> {
    fn bow(&self, kf: DbKf) -> &BowVector {
        self.bow[&kf].get_or_init(|| self.a.kfs[&(kf as KfId)].bow.iter().map(|(w, v)| (*w, *v)).collect())
    }
    fn map_of(&self, kf: DbKf) -> DbMap {
        self.a.kfs[&(kf as KfId)].map as DbMap
    }
    fn is_bad(&self, kf: DbKf) -> bool {
        self.a.kfs[&(kf as KfId)].bad
    }
    fn map_is_bad(&self, m: DbMap) -> bool {
        self.a.maps.iter().find(|x| x.id as DbMap == m).is_none_or(|x| x.bad)
    }
    fn is_connected(&self, kf: DbKf, other: DbKf) -> bool {
        self.a.kfs[&(kf as KfId)].ordered_conn.iter().any(|c| c.0 as DbKf == other)
    }
    fn best_covis(&self, kf: DbKf, n: usize) -> &[DbKf] {
        let v = self.covis[&kf].get_or_init(|| self.a.kfs[&(kf as KfId)].ordered_conn.iter().map(|c| c.0 as DbKf).collect());
        &v[..n.min(v.len())]
    }
}

