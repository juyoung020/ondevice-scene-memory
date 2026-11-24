//! 점 지도(청크): 가까운 키프레임 몇 장의 점을 한 청크에 그 청크 기준 자세에서 본 좌표로 담는다.
//! 점 = i16 × 3(기준 자세 좌표, mm) + 색 u8 × 3, 청크 안에서만 칸(--scene-vox)으로 겹침을 거른다(본 키프레임 수 ≥ --pts-obs 인 칸만 점).
//! 루프 클로저 · 수평 맞춤은 점을 다시 쌓지 않고 청크 기준 자세만 고친다(청크에 든 첫 보관 장면의 고친 위치 × 그 장면에서 본 청크 자세) → 뷰어는 바뀐 자세만 받는다.
//! 누적: 점은 버리지 않고(새 지도 · 다시 이어짐 · 다시 쌓기 모두), 새 장면 점은 이미 점이 있는 칸(세계 좌표 --scene-vox 칸, 다른 청크 포함, 고친 자세 기준)이면 건너뛰어
//! 빈 곳만 채운다(같은 곳을 다시 돌아도 점이 늘지 않음). 이미 있는지는 DUP_CELL 칸에 점이 DUP_MIN 개 넘게 있는지로 본다.
//! 파일: chunks/<판>/c<id>.bin = u32 n + n × i16 xyz(mm, little-endian) + n × u8 rgb. 목록은 view.json scene_points.submap
use crate::geom::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// 청크를 닫는 거리(m) · 각도(°) · 키프레임 수: 기준 자세에서 이만큼 벗어나면 새 청크
pub const CHUNK_M: f64 = 0.6;
pub const CHUNK_DEG: f64 = 25.0;
pub const CHUNK_KF: u32 = 30;
/// 이미 점이 있는지 보는 세계 칸(m)과 그 칸이 찼다고 볼 점 수: 같은 곳을 다시 볼 때 위치가 1 – 2 cm 어긋나도 새 점으로 세지 않게
/// 점 칸(--scene-vox, 1 cm)보다 굵게. 3 cm 칸에 표면 점은 보통 9 개 안팎이라 3 개부터 참
pub const DUP_CELL: f64 = 0.03;
pub const DUP_MIN: u16 = 3;
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
    /// 닻 키프레임(청크에 처음 든 보관 장면 순번)과 그 장면 자세에서 본 청크 자세: 고친 위치가 오면 청크 자세 = 고친 위치 × rel. None = 아직 보관 장면 없음(자세 고정)
    seq: Option<u64>,
    rel: (M3, V3),
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
    /// 닫은 청크 점이 든 세계 DUP_CELL 칸(키 → 점 수): 찬 칸의 새 점은 건너뜀. 청크 자세가 바뀌면 그 청크 몫을 옮김
    occ: HashMap<u64, u16>,
    /// 이미 점이 있어 건너뛴 새 점 수 · 자세를 고친 뒤 겹쳐 뺀 점 수
    pub n_dup: u64,
    pub n_dedup: u64,
    /// 지금까지 자세를 고친 청크 수 · 걸린 시간(초)
    pub n_repose: usize,
    pub t_repose: f64,
}

impl ChunkMap {
    pub fn new(vox: f64, min_obs: u8) -> ChunkMap {
        ChunkMap { vox, min_obs, run: new_run(), chunks: vec![], next_id: 0, added: HashSet::new(), occ: HashMap::new(), n_dup: 0, n_dedup: 0, n_repose: 0, t_repose: 0.0 }
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
            self.close_last();
            self.chunks.push(Chunk { id: self.next_id, seq, rel: (I3, [0.0; 3]), r: *r, p: *pwc, nkf: 0, acc: Some(HashMap::new()), pts: vec![], col: vec![], ver: 0, written: 0, pose_ver: 0 });
            self.next_id += 1;
        }
        let vox_mm = self.vox * 1000.0;
        let c = self.chunks.last_mut().unwrap();
        // 닻이 없던 청크에 처음 보관 장면이 들면 그 장면에 묶음(그 장면 자세에서 본 청크 자세)
        if let (None, Some(s)) = (c.seq, seq) {
            c.seq = Some(s);
            c.rel = inv_mul(&(*r, *pwc), &(c.r, c.p));
        }
        // 카메라 → 청크 기준: q = Rcᵀ (R pc + p − pc0) = (Rcᵀ R) pc + Rcᵀ (p − pc0), mm 로
        let rt = transpose(&c.r);
        let rr = mm(&rt, r);
        let tt = mv(&rt, &sub(pwc, &c.p));
        let acc = c.acc.as_mut().unwrap();
        let occ = &self.occ;
        let mut dup = 0u64;
        for (pc, col, d) in pts {
            let q = add(&mv(&rr, &pc), &tt).map(|x| x * 1000.0);
            if q.iter().any(|x| x.abs() > LIM_MM) {
                continue;
            }
            // 이미 다른 청크 점이 있는 세계 칸은 건너뜀(빈 곳만 채움)
            if !occ.is_empty() && occ.get(&crate::maps::key(&add(&mv(r, &pc), pwc), DUP_CELL)).is_some_and(|&n| n >= DUP_MIN) {
                dup += 1;
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
        self.n_dup += dup;
    }

    /// 세계 높이 z 가 top 보다 위인 점을 모든 청크에서 지움(천장을 정한 순간 그 전에 쌓인 천장 점): 지운 점 수
    pub fn cut_above(&mut self, top: f64) -> usize {
        let mut n = 0;
        for c in &mut self.chunks {
            let wz = |q: [f64; 3]| (c.r[2][0] * q[0] + c.r[2][1] * q[1] + c.r[2][2] * q[2]) / 1000.0 + c.p[2];
            let before = n;
            if let Some(acc) = c.acc.as_mut() {
                let l = acc.len();
                acc.retain(|_, a| {
                    let k = a.n.max(1) as f64;
                    wz([a.s[0] as f64 / k, a.s[1] as f64 / k, a.s[2] as f64 / k]) <= top
                });
                n += l - acc.len();
            } else {
                let keep: Vec<bool> = c.pts.iter().map(|q| wz([q[0] as f64, q[1] as f64, q[2] as f64]) <= top).collect();
                let l = c.pts.len();
                let mut it = keep.iter();
                c.pts.retain(|_| *it.next().unwrap());
                let mut it = keep.iter();
                c.col.retain(|_| *it.next().unwrap());
                n += l - c.pts.len();
            }
            if n > before {
                c.ver += 1;
            }
        }
        n
    }

    /// 쌓던 마지막 청크를 닫고 그 점을 세계 칸에 올림
    fn close_last(&mut self) {
        let min_obs = self.min_obs;
        if let Some(c) = self.chunks.last_mut()
            && c.acc.is_some()
        {
            c.close(min_obs);
            occ_add(&mut self.occ, c, 1);
        }
    }

    /// 고친 키프레임 위치(순번 → 자세)로 청크 기준 자세를 고침: 바뀐 청크 수. 쌓던 청크는 닫는다(그 안 장면끼리 위치가 바뀌었을 수 있음)
    pub fn repose(&mut self, poses: &HashMap<u64, (M3, V3)>) -> usize {
        let c0 = std::time::Instant::now();
        self.close_last();
        let mut moved = HashSet::new();
        for c in &mut self.chunks {
            if let Some(kp) = c.seq.and_then(|s| poses.get(&s)) {
                let (r, p) = mul(kp, &c.rel);
                // 닻 자세 × rel 의 반올림 차는 움직임으로 보지 않음(0.1 mm · 약 0.0006°)
                if norm(&sub(&p, &c.p)) > 1e-4 || rot_angle(&r, &c.r) > 1e-5 {
                    (c.r, c.p) = (r, p);
                    c.pose_ver += 1;
                    moved.insert(c.id);
                }
            }
        }
        let n = moved.len();
        if n > 0 {
            self.dedupe(&moved);
        }
        self.n_repose += n;
        self.t_repose += c0.elapsed().as_secs_f64();
        n
    }

    /// 자세를 고친 뒤 세계 칸을 처음부터 다시 셈(청크 id 차례): 옮겨진 청크의 점 가운데 앞 청크들로 이미 찬 칸에 든 점은 뺀다.
    /// 이어지기 전 새 지도(임시 위치)로 넣은 점이 이어진 뒤 옛 점과 겹치면 여기서 한 벌만 남음(덮인 곳은 그대로, 겹친 점만 빠짐)
    fn dedupe(&mut self, moved: &HashSet<u32>) {
        let mut occ: HashMap<u64, u16> = HashMap::with_capacity(self.occ.len());
        let mut dropped = 0u64;
        for c in &mut self.chunks {
            if c.acc.is_some() {
                continue;
            }
            if moved.contains(&c.id) && !occ.is_empty() {
                let keep: Vec<bool> = c.pts.iter().map(|q| occ.get(&world_key(c, q)).is_none_or(|&k| k < DUP_MIN)).collect();
                let before = c.pts.len();
                let mut it = keep.iter();
                c.pts.retain(|_| *it.next().unwrap());
                let mut it = keep.iter();
                c.col.retain(|_| *it.next().unwrap());
                if c.pts.len() != before {
                    dropped += (before - c.pts.len()) as u64;
                    c.ver += 1;
                }
            }
            occ_add(&mut occ, c, 1);
        }
        self.occ = occ;
        self.n_dedup += dropped;
    }

    pub fn write_count(&self) -> usize {
        self.chunks.len()
    }
    pub fn n_points(&self) -> usize {
        self.chunks.iter().map(|c| c.n_points(self.min_obs)).sum()
    }
    pub fn bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.bytes()).sum::<usize>() + self.added.capacity() * 9 + self.occ.capacity() * 11
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

const I3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// 닫은 청크 점을 지금 자세로 세계 DUP_CELL 칸에 올림(d = 1) · 내림(d = −1)
fn occ_add(occ: &mut HashMap<u64, u16>, c: &Chunk, d: i32) {
    for q in &c.pts {
        let k = world_key(c, q);
        if d > 0 {
            let e = occ.entry(k).or_insert(0);
            *e = e.saturating_add(1);
        } else if let Some(e) = occ.get_mut(&k) {
            *e -= 1;
            if *e == 0 {
                occ.remove(&k);
            }
        }
    }
}

/// 청크 점의 세계 DUP_CELL 칸
fn world_key(c: &Chunk, q: &[i16; 3]) -> u64 {
    let w = add(&mv(&c.r, &[q[0] as f64 * 1e-3, q[1] as f64 * 1e-3, q[2] as f64 * 1e-3]), &c.p);
    crate::maps::key(&w, DUP_CELL)
}

/// a⁻¹ · b
fn inv_mul(a: &(M3, V3), b: &(M3, V3)) -> (M3, V3) {
    let rt = transpose(&a.0);
    (mm(&rt, &b.0), mv(&rt, &sub(&b.1, &a.1)))
}

/// a · b
fn mul(a: &(M3, V3), b: &(M3, V3)) -> (M3, V3) {
    (mm(&a.0, &b.0), add(&mv(&a.0, &b.1), &a.1))
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

    #[test]
    fn fills_only_empty_cells_and_anchor_follows_later_kf() {
        let mut m = ChunkMap::new(0.01, 1);
        // 3 cm 칸 하나(x 0.09 – 0.12 · y 0 – 0.03 · z 0.99 – 1.02)에 서로 다른 1 cm 점 3 개 = 찬 칸
        let pts = || vec![([0.1, 0.0, 1.0], [10, 20, 30], 1.0f32), ([0.1, 0.012, 1.0], [1, 2, 3], 1.0), ([0.1, 0.0, 1.012], [1, 2, 3], 1.0)].into_iter();
        // 보관 안 된 장면으로 청크 시작, 같은 청크의 다음 장면이 보관 장면(순번 5) → 닻
        m.add(0, None, &I, &[0.0, 0.0, 0.0], pts());
        m.add(1, Some(5), &I, &[0.1, 0.0, 0.0], std::iter::empty());
        assert_eq!(m.chunks[0].seq, Some(5));
        // 멀리 가서 새 청크(앞 청크 닫힘) → 같은 세계 점을 다시 보면 건너뜀
        m.add(2, None, &I, &[5.0, 0.0, 0.0], std::iter::empty());
        assert_eq!(m.chunks.len(), 2);
        m.add(3, None, &I, &[5.0, 0.0, 0.0], vec![([-4.895, 0.0, 1.0], [0, 0, 0], 1.0f32), ([-4.0, 0.0, 1.0], [0, 0, 0], 1.0)].into_iter());
        assert_eq!(m.n_dup, 1);
        // 닻 장면이 1 m 옮겨지면 청크도 같은 몫(rel)만큼: 기준 (0,0,0) 이 닻 (0.1,0,0) 에서 −0.1 → 새 닻 (1.1,0,0) 이면 (1.0,0,0)
        let mut poses = HashMap::new();
        poses.insert(5u64, (I, [1.1, 0.0, 0.0]));
        assert_eq!(m.repose(&poses), 1);
        assert!((m.chunks[0].p[0] - 1.0).abs() < 1e-9);
        // 옮긴 뒤 세계 칸도 옮겨짐: 옛 자리(0.1, 0, 1)는 비고 새 자리(1.1, 0, 1)가 참
        let vk = |x: f64, z: f64| crate::maps::key(&[x, 0.0, z], DUP_CELL);
        assert!(!m.occ.contains_key(&vk(0.1, 1.0)) && m.occ.contains_key(&vk(1.1, 1.0)));
    }
}
