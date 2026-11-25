//! KeyFrameDatabase 가 보는 키프레임 그래프(`orb3loop::kfdb::KfGraph`)를 Atlas 위에: DB 번호는 u32, 원본 주소 순 자리는 id 순.
use crate::orb3loop::dbow2::BowVector;
use crate::orb3loop::kfdb::{KfGraph, KfId as DbKf, MapId as DbMap};
use crate::orb3rs::types::{Atlas, KfId};
use std::collections::HashMap;

/// KeyFrameDatabase 가 보는 키프레임 그래프(Atlas 위, DB 번호는 u32)
pub struct AtlasGraph<'a> {
    a: &'a Atlas,
    bow: HashMap<DbKf, BowVector>,
    covis: HashMap<DbKf, Vec<DbKf>>,
}

impl<'a> AtlasGraph<'a> {
    pub fn new(a: &'a Atlas) -> Self {
        let bow = a.kfs.iter().map(|(k, f)| (*k as DbKf, f.bow.iter().map(|(w, v)| (*w, *v)).collect())).collect();
        let covis = a.kfs.iter().map(|(k, f)| (*k as DbKf, f.ordered_conn.iter().map(|c| c.0 as DbKf).collect())).collect();
        AtlasGraph { a, bow, covis }
    }
}

impl KfGraph for AtlasGraph<'_> {
    fn bow(&self, kf: DbKf) -> &BowVector {
        &self.bow[&kf]
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
        let v = &self.covis[&kf];
        &v[..n.min(v.len())]
    }
}

