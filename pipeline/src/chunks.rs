//! 점 지도(청크): 가까운 키프레임 몇 장의 점을 한 청크에 그 청크 기준 자세에서 본 좌표로 담는다.
//! 점 = i16 × 3(기준 자세 좌표, mm) + 색 u8 × 3, 청크 안에서만 칸(--scene-vox)으로 겹침을 거른다(본 키프레임 수 ≥ --pts-obs 인 칸만 점).
//! 루프 클로저 · 수평 맞춤은 점을 다시 쌓지 않고 청크 기준 자세만 고친다(기준 키프레임의 고친 위치) → 뷰어는 바뀐 자세만 받는다.
//! 파일: chunks/<판>/c<id>.bin = u32 n + n × i16 xyz(mm, little-endian) + n × u8 rgb. 목록은 view.json scene_points.submap
use crate::geom::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// 청크를 닫는 거리(m) · 각도(°) · 키프레임 수: 기준 자세에서 이만큼 벗어나면 새 청크
pub const CHUNK_M: f64 = 0.6;
pub const CHUNK_DEG: f64 = 25.0;
pub const CHUNK_KF: u32 = 30;
/// i16 mm 로 담을 수 있는 거리(기준 자세에서)
const LIM_MM: f64 = 32000.0;

/// 청크 안 칸 하나(쌓는 동안만): 위치 합, 가장 가까이서 본 색, 본 키프레임 수
#[derive(Clone, Copy)]
struct Acc {
    s: [f32; 3],
    n: u16,
    col: [u8; 3],
    best: u16,
    nf: u8,
    last: u32,
}

struct Chunk {
    id: u32,
    /// 기준 키프레임(보관 장면 순번): 고친 위치가 오면 이 청크 자세가 됨. None = 보관 안 된 장면(자세 고정)
    seq: Option<u64>,
    r: M3,
    p: V3,
    nkf: u32,
    /// 쌓는 중인 칸(닫으면 비움)
    acc: Option<HashMap<u64, Acc>>,
    pts: Vec<[i16; 3]>,
    col: Vec<[u8; 3]>,
    ver: u32,
    written: u32,
    pose_ver: u32,
}

impl Chunk {
    /// 쌓는 중인 칸 → 점(본 키프레임 수 ≥ min_obs)
    fn confirmed(acc: &HashMap<u64, Acc>, min_obs: u8) -> (Vec<[i16; 3]>, Vec<[u8; 3]>) {
        let mut ks: Vec<(&u64, &Acc)> = acc.iter().filter(|(_, a)| a.nf >= min_obs).collect();
        ks.sort_unstable_by_key(|(k, _)| **k); // 해시 순서와 상관없이 같은 파일
        let pts = ks.iter().map(|(_, a)| {
            let n = a.n.max(1) as f32;
            [(a.s[0] / n).round() as i16, (a.s[1] / n).round() as i16, (a.s[2] / n).round() as i16]
        }).collect();
        (pts, ks.iter().map(|(_, a)| a.col).collect())
    }
    fn close(&mut self, min_obs: u8) {
        if let Some(acc) = self.acc.take() {
            (self.pts, self.col) = Self::confirmed(&acc, min_obs);
        }
    }
    fn n_points(&self, min_obs: u8) -> usize {
        match &self.acc {
            Some(a) => a.values().filter(|x| x.nf >= min_obs).count(),
            None => self.pts.len(),
        }
    }
    fn bytes(&self) -> usize {
        self.pts.capacity() * 6 + self.col.capacity() * 3 + self.acc.as_ref().map_or(0, |a| a.capacity() * (8 + std::mem::size_of::<Acc>() + 1))
    }
}

pub struct ChunkMap {
    pub vox: f64,
    pub min_obs: u8,
    pub run: u64,
    chunks: Vec<Chunk>,
    next_id: u32,
    /// 넣은 보관 장면(다시 넣지 않게)
    added: HashSet<u64>,
    /// 지금까지 자세를 고친 청크 수 · 걸린 시간(초)
    pub n_repose: usize,
    pub t_repose: f64,
}

impl ChunkMap {
    pub fn new(vox: f64, min_obs: u8) -> ChunkMap {
        ChunkMap { vox, min_obs, run: new_run(), chunks: vec![], next_id: 0, added: HashSet::new(), n_repose: 0, t_repose: 0.0 }
    }

    pub fn has(&self, seq: u64) -> bool {
        self.added.contains(&seq)
    }

    /// 한 장면의 점(카메라 좌표 m, 색, 깊이 m)을 넣는다. r, pwc: 카메라 → 지도, kf: 장면 번호(칸마다 한 번 셈), seq: 보관 장면 순번
    pub fn add(&mut self, kf: u32, seq: Option<u64>, r: &M3, pwc: &V3, pts: impl Iterator<Item = (V3, [u8; 3], f32)>) {
        if let Some(s) = seq {
            if !self.added.insert(s) {
                return;
            }
        }
        let open = self.chunks.last().is_some_and(|c| {
            c.acc.is_some() && c.nkf < CHUNK_KF && norm(&sub(pwc, &c.p)) < CHUNK_M && rot_angle(r, &c.r) < CHUNK_DEG.to_radians()
        });
        if !open {
            let min_obs = self.min_obs;
            if let Some(c) = self.chunks.last_mut() {
                c.close(min_obs);
            }
            self.chunks.push(Chunk { id: self.next_id, seq, r: *r, p: *pwc, nkf: 0, acc: Some(HashMap::new()), pts: vec![], col: vec![], ver: 0, written: 0, pose_ver: 0 });
            self.next_id += 1;
        }
        let vox_mm = self.vox * 1000.0;
        let c = self.chunks.last_mut().unwrap();
        // 카메라 → 청크 기준: q = Rcᵀ (R pc + p − pc0) = (Rcᵀ R) pc + Rcᵀ (p − pc0), mm 로
        let rt = transpose(&c.r);
        let rr = mm(&rt, r);
        let tt = mv(&rt, &sub(pwc, &c.p));
        let acc = c.acc.as_mut().unwrap();
        for (pc, col, d) in pts {
            let q = add(&mv(&rr, &pc), &tt).map(|x| x * 1000.0);
            if q.iter().any(|x| x.abs() > LIM_MM) {
                continue;
            }
            let k = crate::maps::key(&q, vox_mm);
            let dm = (d * 1000.0).clamp(0.0, 65535.0) as u16;
            let e = acc.entry(k).or_insert(Acc { s: [0.0; 3], n: 0, col, best: u16::MAX, nf: 0, last: u32::MAX });
            if e.n < u16::MAX {
                e.s = [e.s[0] + q[0] as f32, e.s[1] + q[1] as f32, e.s[2] + q[2] as f32];
                e.n += 1;
            }
            if dm < e.best {
                (e.best, e.col) = (dm, col);
            }
            if e.last != kf {
                e.last = kf;
                e.nf = e.nf.saturating_add(1);
            }
        }
        c.nkf += 1;
        c.ver += 1;
    }

    /// 고친 키프레임 위치(순번 → 자세)로 청크 기준 자세를 고침: 바뀐 청크 수. 쌓던 청크는 닫는다(그 안 장면끼리 위치가 바뀌었을 수 있음)
    pub fn repose(&mut self, poses: &HashMap<u64, (M3, V3)>) -> usize {
        let c0 = std::time::Instant::now();
        let min_obs = self.min_obs;
        if let Some(c) = self.chunks.last_mut() {
            c.close(min_obs);
        }
        let mut n = 0;
        for c in &mut self.chunks {
            if let Some((r, p)) = c.seq.and_then(|s| poses.get(&s)) {
                if *r != c.r || *p != c.p {
                    (c.r, c.p) = (*r, *p);
                    c.pose_ver += 1;
                    n += 1;
                }
            }
        }
        self.n_repose += n;
        self.t_repose += c0.elapsed().as_secs_f64();
        n
    }

    pub fn write_count(&self) -> usize {
        self.chunks.len()
    }
    pub fn n_points(&self) -> usize {
        self.chunks.iter().map(|c| c.n_points(self.min_obs)).sum()
    }
    pub fn bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.bytes()).sum::<usize>() + self.added.capacity() * 9
    }

    /// 바뀐 청크 파일을 쓰고 목록 JSON 을 돌려줌(list: [id, 판, 점 수, r00..r22, px, py, pz])
    pub fn write(&mut self, dir: &Path) -> serde_json::Value {
        let sub_dir = format!("chunks/{}", self.run);
        std::fs::create_dir_all(dir.join(&sub_dir)).unwrap();
        let min_obs = self.min_obs;
        let mut list = Vec::with_capacity(self.chunks.len());
        let mut total = 0usize;
        for c in &mut self.chunks {
            if c.ver != c.written {
                let tmp;
                let (pts, col) = match &c.acc {
                    Some(a) => {
                        tmp = Chunk::confirmed(a, min_obs);
                        (&tmp.0, &tmp.1)
                    }
                    None => (&c.pts, &c.col),
                };
                let mut b = Vec::with_capacity(4 + pts.len() * 9);
                b.extend_from_slice(&(pts.len() as u32).to_le_bytes());
                for q in pts {
                    for x in q {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                for cc in col {
                    b.extend_from_slice(cc);
                }
                c.written = c.ver;
                crate::out::write_owned(&dir.join(format!("{sub_dir}/c{}.bin", c.id)), b);
            }
            let n = c.n_points(min_obs);
            total += n;
            let r4 = |x: f64| (x * 1e6).round() / 1e6;
            let mut e = vec![serde_json::json!(c.id), serde_json::json!(c.written), serde_json::json!(n)];
            for row in &c.r {
                e.extend(row.iter().map(|x| serde_json::json!(r4(*x))));
            }
            e.extend(c.p.iter().map(|x| serde_json::json!(r4(*x))));
            list.push(serde_json::Value::Array(e));
        }
        serde_json::json!({"run": self.run, "dir": sub_dir, "voxel": self.vox, "n": total, "chunks": self.chunks.len(), "list": list})
    }
}

static RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 새 지도 번호(뷰어가 받은 청크를 비우게): 시각(ms) × 16 + 프로세스 안 순번
pub fn new_run() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    t * 16 + RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 16
}

fn transpose(m: &M3) -> M3 {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}

fn rot_angle(a: &M3, b: &M3) -> f64 {
    let t: f64 = (0..3).map(|i| (0..3).map(|j| a[i][j] * b[i][j]).sum::<f64>()).sum();
    ((t - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    const I: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    #[test]
    fn chunk_points_follow_pose() {
        let mut m = ChunkMap::new(0.01, 1);
        let pts = || vec![([0.1, 0.0, 1.0], [10, 20, 30], 1.0f32), ([0.0, 0.2, 2.0], [1, 2, 3], 2.0)].into_iter();
        m.add(0, Some(7), &I, &[1.0, 0.0, 0.0], pts());
        let c = &m.chunks[0];
        let (p, col) = Chunk::confirmed(c.acc.as_ref().unwrap(), 1);
        assert_eq!(p.len(), 2);
        assert!(p.contains(&[100, 0, 1000]) && col.contains(&[10, 20, 30]));
        // 루프 클로저: 기준 키프레임이 2 m 옮겨짐 → 점은 그대로, 자세만
        let mut poses = HashMap::new();
        poses.insert(7u64, (I, [3.0, 0.0, 0.0]));
        assert_eq!(m.repose(&poses), 1);
        assert_eq!(m.chunks[0].p, [3.0, 0.0, 0.0]);
        assert_eq!(m.chunks[0].pts.len(), 2);
        // 같은 보관 장면은 다시 넣지 않음
        m.add(1, Some(7), &I, &[3.0, 0.0, 0.0], pts());
        assert_eq!(m.chunks.len(), 1);
    }
}
