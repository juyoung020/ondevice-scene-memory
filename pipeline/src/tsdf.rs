//! 표면 지도(3D 스캔처럼): TSDF 융합 + Surface Nets 로 면 뽑기.
//! 위치가 확인된 장면마다 깊이를 칸 격자에 '표면까지의 거리(잘린 부호 거리)'로 누적 평균하고(여러 장면이 겹칠수록 잡음이 줄어듦),
//! 거리가 0 을 지나는 곳에서 매끄러운 삼각형 면을 만든다. 칸은 표면 근처 블록(8×8×8)만 해시로 잡는다.
//! 면 파일은 구역(블록 4×4×4)마다 따로 쓰고, 바뀐 구역만 다시 쓴다.
use crate::geom::*;
use crate::geom::{FastMap, FastSet};

const B: i32 = 8; // 블록 한 변의 칸 수
const R: i32 = 4; // 구역 한 변의 블록 수

struct Block {
    d: Vec<f32>,        // 잘린 거리(−1…1)
    w: Vec<f32>,        // 무게(관측 수, 상한 있음)
    c: Vec<[f32; 3]>,   // 색(가중 평균)
    cw: Vec<f32>,       // 색 무게(가깝고 정면으로 본 장면일수록 큼)
}

pub struct Tsdf {
    pub vox: f64,
    trunc: f64,
    blocks: FastMap<(i32, i32, i32), Block>,
    dirty: FastSet<(i32, i32, i32)>, // 바뀐 구역
    pub ver: FastMap<(i32, i32, i32), u32>, // 구역 → 판 번호(뷰어가 바뀐 것만 받음)
}

#[inline]
fn div_floor(a: i32, b: i32) -> i32 {
    a.div_euclid(b)
}

impl Tsdf {
    pub fn new(vox: f64) -> Tsdf {
        Tsdf { vox, trunc: 3.0 * vox, blocks: FastMap::default(), dirty: FastSet::default(), ver: FastMap::default() }
    }

    /// 한 장면 더하기: depth(미터, w×h), rgb, 카메라→세계 (r, p), 보정값 k.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate(&mut self, depth: &[f32], rgb: &[u8], w: usize, h: usize, k: [f64; 4], r: &M3, p: &V3, max_depth: f64) {
        use rayon::prelude::*;
        let [fx, fy, cx, cy] = k;
        let vs = self.vox;
        let bs = vs * B as f64;
        let trunc = self.trunc;
        // 1) 표면 근처 블록 잡기(화소 4 간격, 표면 앞뒤 trunc). 줄 안에서 바로 앞 화소와 같은 블록이면 건너뛰어
        //    해시에는 화소당 약 0.4 번만 넣고, 블록은 좌표 순으로 정렬
        let mut set: FastSet<(i32, i32, i32)> = FastSet::default();
        for v in (0..h).step_by(4) {
            let mut last = [(i32::MIN, 0, 0); 3];
            for u in (0..w).step_by(4) {
                let z = depth[v * w + u] as f64;
                if !(0.2..max_depth).contains(&z) {
                    continue;
                }
                let ray = [(u as f64 - cx) / fx, (v as f64 - cy) / fy, 1.0];
                for (j, s) in [-trunc, 0.0, trunc].into_iter().enumerate() {
                    let zz = z + s;
                    let x = add(&mv(r, &ray.map(|a| a * zz)), p);
                    let key = ((x[0] / bs).floor() as i32, (x[1] / bs).floor() as i32, (x[2] / bs).floor() as i32);
                    if key != last[j] {
                        last[j] = key;
                        set.insert(key);
                    }
                }
            }
        }
        let mut touch: Vec<(i32, i32, i32)> = set.into_iter().collect();
        touch.sort_unstable();
        // 2) 잡은 블록의 칸마다: 칸 중심을 이번 장면에 투영해 깊이와 비교. 블록끼리 독립이라 코어마다 나눠 돌린다.
        let rt = [0, 1, 2].map(|i| [0, 1, 2].map(|j| r[j][i]));
        let n = (B * B * B) as usize;
        let mut work: Vec<((i32, i32, i32), Block, bool)> = touch
            .into_iter()
            .map(|key| {
                let blk = self.blocks.remove(&key).unwrap_or_else(|| Block { d: vec![1.0; n], w: vec![0.0; n], c: vec![[0.0; 3]; n], cw: vec![0.0; n] });
                (key, blk, false)
            })
            .collect();
        let one = |key: (i32, i32, i32), blk: &mut Block| -> bool {
            // 칸 중심의 카메라 좌표 Rᵀ·(x − p) 를 축마다 미리 곱해 둔다: 성분마다 (x 항 + y 항) + z 항 — 칸마다 행렬을 곱할 때와 같은 순서·같은 값
            let axis = |a: usize, k0: i32| -> [[f64; 3]; B as usize] {
                std::array::from_fn(|i| {
                    let d = (k0 * B + i as i32) as f64 * vs + vs * 0.5 - p[a];
                    [rt[0][a] * d, rt[1][a] * d, rt[2][a] * d]
                })
            };
            let (ax, ay, az) = (axis(0, key.0), axis(1, key.1), axis(2, key.2));
            let mut changed = false;
            let mut i = 0usize;
            for cz in &az {
                for cy_ in &ay {
                    for cx_ in &ax {
                        let ii = i;
                        i += 1;
                        let pc = [cx_[0] + cy_[0] + cz[0], cx_[1] + cy_[1] + cz[1], cx_[2] + cy_[2] + cz[2]];
                        if pc[2] < 0.15 {
                            continue;
                        }
                        let u = (fx * pc[0] / pc[2] + cx).round();
                        let v = (fy * pc[1] / pc[2] + cy).round();
                        if u < 0.0 || v < 0.0 || u >= w as f64 || v >= h as f64 {
                            continue;
                        }
                        let (u, v) = (u as usize, v as usize);
                        let z = depth[v * w + u] as f64;
                        if !(0.2..max_depth).contains(&z) {
                            continue;
                        }
                        let sdf = z - pc[2];
                        if sdf < -trunc {
                            continue; // 표면 뒤 멀리: 모름
                        }
                        let t = (sdf / trunc).min(1.0) as f32;
                        // 형상 무게 = 비스듬함 코사인(이웃 깊이로 구한 면 법선 · 시선). 거리와는 무관하게 둔다 —
                        // 1/거리² 를 곱하면 F 가 낮아짐(책상 0.40 → 0.30, surfcmp). 스치듯 본 먼 바닥(겹친 판·날개)은 무게가 작고, 약 80° 넘게 스치면 버림
                        let cos = incidence(depth, w, h, u, v, fx, fy);
                        if cos < GRAZE_MIN {
                            continue;
                        }
                        let wt = cos as f32;
                        let w0 = blk.w[ii];
                        blk.d[ii] = (blk.d[ii] * w0 + t * wt) / (w0 + wt);
                        if t.abs() < 1.0 {
                            // 색 무게 = 코사인 × (1 m / 거리)²(형상 무게와 별개) — 가깝고 정면인 장면의 색이 앞섬
                            let wc = (cos / (z * z).max(0.25)) as f32;
                            let c0 = blk.cw[ii];
                            let o = (v * w + u) * 3;
                            for q in 0..3 {
                                blk.c[ii][q] = (blk.c[ii][q] * c0 + rgb[o + q] as f32 * wc) / (c0 + wc);
                            }
                            blk.cw[ii] = (c0 + wc).min(64.0);
                        }
                        blk.w[ii] = (w0 + wt).min(64.0);
                        changed = true;
                    }
                }
            }
            changed
        };
        work.par_iter_mut().for_each(|(key, blk, ch)| *ch = one(*key, blk));
        for (key, blk, ch) in work {
            if ch {
                self.dirty.insert((div_floor(key.0, R), div_floor(key.1, R), div_floor(key.2, R)));
            }
            self.blocks.insert(key, blk);
        }
    }
    /// 바뀐 구역의 면(꼭짓점: 위치·색, 삼각형)을 뽑아 돌려준다(구역 → 면). 판 번호도 올린다.
    #[allow(clippy::type_complexity)]
    #[allow(dead_code)]
    pub fn take_dirty_meshes(&mut self) -> Vec<((i32, i32, i32), Vec<([f32; 3], [u8; 3])>, Vec<[u32; 3]>)> {
        self.take_dirty_meshes_max(usize::MAX)
    }

    /// 바뀐 구역 중 많아야 max 개만(구역 순서 앞쪽부터) 면을 뽑고 나머지는 바뀐 채로 둔다
    #[allow(clippy::type_complexity)]
    pub fn take_dirty_meshes_max(&mut self, max: usize) -> Vec<((i32, i32, i32), Vec<([f32; 3], [u8; 3])>, Vec<[u32; 3]>)> {
        let regions: Vec<(i32, i32, i32)> = if self.dirty.len() <= max {
            self.dirty.drain().collect()
        } else {
            let mut all: Vec<(i32, i32, i32)> = self.dirty.iter().copied().collect();
            all.sort();
            all.truncate(max);
            for r in &all {
                self.dirty.remove(r);
            }
            all
        };
        // 구역끼리 독립: 코어마다 나눠 뽑는다(읽기만)
        let this = &*self;
        let mut out: Vec<_> = {
            use rayon::prelude::*;
            regions.par_iter().map(|&reg| { let (v, t) = this.mesh_region(reg); (reg, v, t) }).collect()
        };
        for (reg, _, _) in &out {
            *self.ver.entry(*reg).or_insert(0) += 1;
        }
        out.sort_by_key(|x| x.0);
        out
    }

    /// Surface Nets: 칸 8 개로 된 상자마다 부호가 바뀌면 그 안에 꼭짓점 하나(변의 0 지점 평균),
    /// 부호가 바뀌는 변마다 그 변을 둘러싼 상자 4 개의 꼭짓점으로 사각형 하나.
    fn mesh_region(&self, reg: (i32, i32, i32)) -> (Vec<([f32; 3], [u8; 3])>, Vec<[u32; 3]>) {
        let n = B * R; // 구역 한 변의 칸 수
        let (x0, y0, z0) = (reg.0 * n, reg.1 * n, reg.2 * n);
        let vs = self.vox as f32;
        // 구역 주변(−1 … n+1) 칸을 작은 배열로 한 번에 복사(해시 조회를 칸마다 하지 않게)
        let m = n + 3;
        let li = |x: i32, y: i32, z: i32| (((z + 1) * m + (y + 1)) * m + (x + 1)) as usize;
        let mut loc: Vec<Option<(f32, [f32; 3])>> = vec![None; (m * m * m) as usize];
        let (bx0, bx1) = (div_floor(x0 - 1, B), div_floor(x0 + n + 1, B));
        let (by0, by1) = (div_floor(y0 - 1, B), div_floor(y0 + n + 1, B));
        let (bz0, bz1) = (div_floor(z0 - 1, B), div_floor(z0 + n + 1, B));
        for bz in bz0..=bz1 {
            for by in by0..=by1 {
                for bx in bx0..=bx1 {
                    let Some(blk) = self.blocks.get(&(bx, by, bz)) else { continue };
                    for i in 0..(B * B * B) as usize {
                        if blk.w[i] < 2.0 {
                            continue; // 한 번만 본 칸은 잡음이 커서 면에 쓰지 않음
                        }
                        let (ix, iy, iz) = ((i as i32) % B, (i as i32 / B) % B, i as i32 / (B * B));
                        let (x, y, z) = (bx * B + ix - x0, by * B + iy - y0, bz * B + iz - z0);
                        if x < -1 || y < -1 || z < -1 || x > n + 1 || y > n + 1 || z > n + 1 {
                            continue;
                        }
                        loc[li(x, y, z)] = Some((blk.d[i], blk.c[i]));
                    }
                }
            }
        }
        let at = |x: i32, y: i32, z: i32| loc[li(x, y, z)];
        // 상자(셀) 꼭짓점: 구역 −1 … n−1 (경계 사각형을 위해 한 칸 바깥까지)
        let idx = |x: i32, y: i32, z: i32| (((z + 1) * (n + 1) + (y + 1)) * (n + 1) + (x + 1)) as usize;
        let mut cell_v = vec![u32::MAX; ((n + 1) * (n + 1) * (n + 1)) as usize];
        let mut verts: Vec<([f32; 3], [u8; 3])> = vec![];
        const CORN: [(i32, i32, i32); 8] = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)];
        const EDGES: [(usize, usize); 12] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];
        for z in -1..n {
            for y in -1..n {
                for x in -1..n {
                    let mut c = [(0f32, [0f32; 3]); 8];
                    let mut ok = true;
                    for (k, &(dx, dy, dz)) in CORN.iter().enumerate() {
                        match at(x + dx, y + dy, z + dz) {
                            Some(v) => c[k] = v,
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let neg = c.iter().filter(|v| v.0 < 0.0).count();
                    if neg == 0 || neg == 8 {
                        continue;
                    }
                    // 진짜 0 지점이 하나도 없으면(가짜 0 지점만) 꼭짓점을 만들지 않음
                    if !EDGES.iter().any(|&(a, b)| (c[a].0 < 0.0) != (c[b].0 < 0.0) && c[a].0.abs() <= 0.95 && c[b].0.abs() <= 0.95) {
                        continue;
                    }
                    // 변의 0 지점 평균(상자 안 좌표 0…1)
                    let (mut s, mut m) = ([0f32; 3], 0f32);
                    let mut col = [0f32; 3];
                    for &(a, b) in &EDGES {
                        let (da, db) = (c[a].0, c[b].0);
                        if (da < 0.0) == (db < 0.0) || da.abs() > 0.95 || db.abs() > 0.95 {
                            continue; // 잘린 값(±1) 과 맞닿은 0 지점은 가짜 면
                        }
                        let t = da / (da - db);
                        let (pa, pb) = (CORN[a], CORN[b]);
                        s[0] += pa.0 as f32 + t * (pb.0 - pa.0) as f32;
                        s[1] += pa.1 as f32 + t * (pb.1 - pa.1) as f32;
                        s[2] += pa.2 as f32 + t * (pb.2 - pa.2) as f32;
                        for q in 0..3 {
                            col[q] += c[a].1[q] * (1.0 - t) + c[b].1[q] * t;
                        }
                        m += 1.0;
                    }
                    let pos = [0, 1, 2].map(|q| ([x0 + x, y0 + y, z0 + z][q] as f32 + 0.5 + s[q] / m) * vs);
                    cell_v[idx(x, y, z)] = verts.len() as u32;
                    verts.push((pos, col.map(|q| (q / m).clamp(0.0, 255.0) as u8)));
                }
            }
        }
        // 사각형: 구역 안(0…n−1) 칸에서 +축 방향 변마다
        let mut tris = vec![];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let Some((d0, _)) = at(x, y, z) else { continue };
                    for ax in 0..3 {
                        let (dx, dy, dz) = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][ax];
                        let Some((d1, _)) = at(x + dx, y + dy, z + dz) else { continue };
                        if (d0 < 0.0) == (d1 < 0.0) || d0.abs() > 0.95 || d1.abs() > 0.95 {
                            continue;
                        }
                        // 변을 둘러싼 상자 4 개(다른 두 축으로 0/−1)
                        let (b, c2) = [(1, 2), (2, 0), (0, 1)][ax]; // 순환 순서(x→y,z / y→z,x / z→x,y)여야 세 축의 면이 같은 쪽을 향함
                        let cell = |ob: i32, oc: i32| {
                            let mut q = [x, y, z];
                            q[b] -= ob;
                            q[c2] -= oc;
                            cell_v[idx(q[0], q[1], q[2])]
                        };
                        let (v00, v10, v01, v11) = (cell(1, 1), cell(0, 1), cell(1, 0), cell(0, 0));
                        if [v00, v10, v01, v11].contains(&u32::MAX) {
                            continue;
                        }
                        // 안(음수)→밖 방향이 바깥 면이 되도록 감는 방향을 정함
                        if d0 < 0.0 {
                            tris.push([v00, v10, v11]);
                            tris.push([v00, v11, v01]);
                        } else {
                            tris.push([v00, v11, v10]);
                            tris.push([v00, v01, v11]);
                        }
                    }
                }
            }
        }
        (verts, tris)
    }
}

/// 비스듬함 문턱(면 법선과 시선의 코사인): 이보다 작으면(약 80° 넘게 스침) 측정을 버림
const GRAZE_MIN: f64 = 0.17;

/// 화소 (u, v) 의 면 법선(옆·아래 화소 깊이 차이)과 시선 사이 코사인. 이웃 깊이가 없으면 1(모름 = 정면 취급)
#[inline]
fn incidence(depth: &[f32], w: usize, h: usize, u: usize, v: usize, fx: f64, fy: f64) -> f64 {
    if u + 1 >= w || v + 1 >= h {
        return 1.0;
    }
    let z = depth[v * w + u] as f64;
    let (zr, zd) = (depth[v * w + u + 1] as f64, depth[(v + 1) * w + u] as f64);
    if zr <= 0.0 || zd <= 0.0 || (zr - z).abs() > 0.1 * z || (zd - z).abs() > 0.1 * z {
        return 1.0;
    }
    // 화면 기울기 → 카메라 좌표 법선: 깊이 변화 / 화소 크기(z/f)
    let gx = (zr - z) * fx / z;
    let gy = (zd - z) * fy / z;
    1.0 / (1.0 + gx * gx + gy * gy).sqrt()
}

// ------------------------------------------------------------------ 서브맵 합치기(cblox getProjectedMap 방식)
// 서브맵 = 기준 노드에 붙은 자기 좌표의 TSDF. 전체 표면 = 서브맵마다 지금 자세로 옮겨(삼선형 보간) 무게 평균한 것.
// 칸 값은 (거리, 무게, 색) 평균이고, 더하기·빼기는 무게 합 산술이라 같은 자세로 더한 것을 그대로 뺄 수 있다.

/// 블록 키(블록 단위 정수 좌표)
pub type BKey = (i32, i32, i32);
/// 자세: 회전·이동(원본 좌표 → 이 지도 좌표)
pub type Pose = (M3, V3);

impl Clone for Block {
    fn clone(&self) -> Block {
        Block { d: self.d.clone(), w: self.w.clone(), c: self.c.clone(), cw: self.cw.clone() }
    }
}

/// 칸 하나에 (거리 d, 무게 w, 색 c) 를 부호 s(+1 더하기, −1 빼기)로 섞는다
#[inline]
fn mix(blk: &mut Block, i: usize, d: f32, w: f32, c: [f32; 3], s: f32) {
    let w0 = blk.w[i];
    let w1 = w0 + s * w;
    if w1 < 1e-3 {
        (blk.d[i], blk.w[i], blk.c[i]) = (1.0, 0.0, [0.0; 3]);
        return;
    }
    blk.d[i] = ((blk.d[i] * w0 + s * d * w) / w1).clamp(-1.0, 1.0);
    for q in 0..3 {
        blk.c[i][q] = ((blk.c[i][q] * w0 + s * c[q] * w) / w1).clamp(0.0, 255.0);
    }
    blk.w[i] = w1;
}

fn empty_block() -> Block {
    let n = (B * B * B) as usize;
    Block { d: vec![1.0; n], w: vec![0.0; n], c: vec![[0.0; 3]; n], cw: vec![0.0; n] }
}

impl Tsdf {
    /// 블록 한 변(m)
    pub fn block_size(&self) -> f64 {
        self.vox * B as f64
    }

    /// 칸 배열이 차지하는 바이트(대략)
    pub fn bytes(&self) -> usize {
        self.blocks.len() * (B * B * B) as usize * 20
    }

    /// 면을 다시 뽑을 구역 수
    pub fn n_dirty(&self) -> usize {
        self.dirty.len()
    }

    /// 바뀐 구역을 꺼낸다(면은 뽑지 않음): 서브맵에서 전체 표면으로 옮길 곳
    pub fn drain_dirty_regions(&mut self) -> Vec<BKey> {
        self.dirty.drain().collect()
    }

    /// 구역들 안에 있는 블록 키
    pub fn blocks_in_regions(&self, regs: &[BKey]) -> Vec<BKey> {
        let mut out = vec![];
        for r in regs {
            for z in 0..R {
                for y in 0..R {
                    for x in 0..R {
                        let k = (r.0 * R + x, r.1 * R + y, r.2 * R + z);
                        if self.blocks.contains_key(&k) {
                            out.push(k);
                        }
                    }
                }
            }
        }
        out
    }

    /// 다른 지도의 블록을 그대로 복사(같은 키)
    pub fn copy_blocks_from(&mut self, src: &Tsdf, keys: &[BKey]) {
        for k in keys {
            if let Some(b) = src.blocks.get(k) {
                self.blocks.insert(*k, b.clone());
            }
        }
    }

    /// 판 번호를 넘겨받고 다시 쓸 구역을 정한다(전체 표면을 새로 합친 뒤 바뀐 곳만 뷰어가 받게)
    pub fn adopt_versions(&mut self, old: &mut Tsdf, dirty: impl IntoIterator<Item = BKey>) {
        self.ver = std::mem::take(&mut old.ver);
        self.dirty.clear();
        self.dirty.extend(dirty);
        self.dirty.extend(old.dirty.drain());
    }

    /// 원본 지도의 블록들이 자세 pose 에서 덮는 구역
    pub fn footprint(src: &Tsdf, pose: &Pose) -> FastSet<BKey> {
        let bs = src.block_size();
        let rs = bs * R as f64;
        let mut out = FastSet::default();
        for k in src.blocks.keys() {
            for c in 0..8 {
                let q = [(k.0 + (c & 1)) as f64 * bs, (k.1 + ((c >> 1) & 1)) as f64 * bs, (k.2 + (c >> 2)) as f64 * bs];
                let x = add(&mv(&pose.0, &q), &pose.1);
                out.insert(((x[0] / rs).floor() as i32, (x[1] / rs).floor() as i32, (x[2] / rs).floor() as i32));
            }
        }
        out
    }

    /// 자세가 회전 없이 블록 단위로만 옮기면 그 블록 수(칸을 보간 없이 그대로 옮길 수 있음)
    pub fn aligned(&self, pose: &Pose) -> Option<BKey> {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        if (0..3).any(|i| (0..3).any(|j| (pose.0[i][j] - id[i][j]).abs() > 1e-9)) {
            return None;
        }
        let bs = self.block_size();
        let o = pose.1.map(|v| v / bs);
        if o.iter().any(|v| (v - v.round()).abs() > 1e-6) {
            return None;
        }
        Some((o[0].round() as i32, o[1].round() as i32, o[2].round() as i32))
    }

    /// 원본 지도 src 를 자세 pose(src → 이 지도)로 옮겨 더하거나(s = 1) 빼기(s = −1). 바뀐 구역은 면을 다시 뽑게 표시.
    /// 블록 단위 이동이면 칸을 그대로, 아니면 이 지도 칸 중심을 원본 좌표로 되돌려 이웃 8 칸을 삼선형 보간(무게를 곱해 섞으니 빈 칸은 무게 0).
    /// keys: 블록 단위 이동일 때 이 원본 블록만(None = 전부). 칸 크기는 같아야 한다.
    pub fn merge(&mut self, src: &Tsdf, pose: &Pose, s: f32, keys: Option<&[BKey]>) {
        use rayon::prelude::*;
        let n = (B * B * B) as usize;
        if let Some(o) = self.aligned(pose) {
            let all: Vec<BKey>;
            let ks = match keys {
                Some(k) => k,
                None => {
                    all = src.blocks.keys().copied().collect();
                    &all
                }
            };
            for k in ks {
                let Some(sb) = src.blocks.get(k) else { continue };
                let tk = (k.0 + o.0, k.1 + o.1, k.2 + o.2);
                let tb = self.blocks.entry(tk).or_insert_with(empty_block);
                let mut ch = false;
                for i in 0..n {
                    if sb.w[i] > 0.0 {
                        mix(tb, i, sb.d[i], sb.w[i], sb.c[i], s);
                        ch = true;
                    }
                }
                if ch {
                    self.dirty.insert((div_floor(tk.0, R), div_floor(tk.1, R), div_floor(tk.2, R)));
                }
            }
            return;
        }
        assert!(keys.is_none(), "블록 단위 이동이 아닐 때는 지도 전체만 옮김");
        let vs = self.vox;
        let bs = self.block_size();
        // 대상 블록: 원본 블록 상자(칸 하나 넓혀)의 꼭짓점 8 개를 옮긴 범위
        let mut tset: FastSet<BKey> = FastSet::default();
        for k in src.blocks.keys() {
            let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
            for c in 0..8 {
                let e = [c & 1, (c >> 1) & 1, c >> 2];
                let q = [
                    (k.0 + e[0]) as f64 * bs + if e[0] == 0 { -vs } else { vs },
                    (k.1 + e[1]) as f64 * bs + if e[1] == 0 { -vs } else { vs },
                    (k.2 + e[2]) as f64 * bs + if e[2] == 0 { -vs } else { vs },
                ];
                let x = add(&mv(&pose.0, &q), &pose.1);
                for a in 0..3 {
                    lo[a] = lo[a].min(x[a]);
                    hi[a] = hi[a].max(x[a]);
                }
            }
            let (a0, a1) = (lo.map(|v| (v / bs).floor() as i32), hi.map(|v| (v / bs).floor() as i32));
            for z in a0[2]..=a1[2] {
                for y in a0[1]..=a1[1] {
                    for x in a0[0]..=a1[0] {
                        tset.insert((x, y, z));
                    }
                }
            }
        }
        let mut tv: Vec<BKey> = tset.into_iter().collect();
        tv.sort_unstable();
        let r = &pose.0;
        let rt = [0, 1, 2].map(|i| [0, 1, 2].map(|j| r[j][i]));
        let p = pose.1;
        // 대상 블록끼리 독립(원본은 읽기만): 코어마다 나눠 보간
        type Samp = Vec<(u16, f32, f32, [f32; 3])>;
        let res: Vec<(BKey, Samp)> = tv
            .par_iter()
            .filter_map(|&tk| {
                let mut out: Samp = vec![];
                let mut cache: (BKey, Option<&Block>) = ((i32::MIN, 0, 0), None);
                for i in 0..n {
                    let (ix, iy, iz) = ((i as i32) % B, (i as i32 / B) % B, i as i32 / (B * B));
                    let xg = [((tk.0 * B + ix) as f64 + 0.5) * vs, ((tk.1 * B + iy) as f64 + 0.5) * vs, ((tk.2 * B + iz) as f64 + 0.5) * vs];
                    let xs = mv(&rt, &sub(&xg, &p));
                    let q = xs.map(|v| v / vs - 0.5);
                    let fl = q.map(|v| v.floor());
                    let f = [q[0] - fl[0], q[1] - fl[1], q[2] - fl[2]];
                    let i0 = fl.map(|v| v as i32);
                    let (mut sw, mut sd, mut sc) = (0f64, 0f64, [0f64; 3]);
                    for c in 0..8 {
                        let e = [c & 1, (c >> 1) & 1, c >> 2];
                        let tw = (0..3).map(|a| if e[a] == 1 { f[a] } else { 1.0 - f[a] }).product::<f64>();
                        if tw <= 0.0 {
                            continue;
                        }
                        let v = [i0[0] + e[0], i0[1] + e[1], i0[2] + e[2]];
                        let bk = (div_floor(v[0], B), div_floor(v[1], B), div_floor(v[2], B));
                        if cache.0 != bk {
                            cache = (bk, src.blocks.get(&bk));
                        }
                        let Some(b) = cache.1 else { continue };
                        let li = ((v[2].rem_euclid(B) * B + v[1].rem_euclid(B)) * B + v[0].rem_euclid(B)) as usize;
                        let ww = tw * b.w[li] as f64;
                        if ww <= 0.0 {
                            continue;
                        }
                        sw += ww;
                        sd += ww * b.d[li] as f64;
                        for q in 0..3 {
                            sc[q] += ww * b.c[li][q] as f64;
                        }
                    }
                    if sw > 1e-6 {
                        out.push((i as u16, (sd / sw) as f32, sw as f32, sc.map(|v| (v / sw) as f32)));
                    }
                }
                (!out.is_empty()).then_some((tk, out))
            })
            .collect();
        for (tk, out) in res {
            let tb = self.blocks.entry(tk).or_insert_with(empty_block);
            for (i, d, w, c) in out {
                mix(tb, i as usize, d, w, c, s);
            }
            self.dirty.insert((div_floor(tk.0, R), div_floor(tk.1, R), div_floor(tk.2, R)));
        }
    }

    /// 칸 하나 값(시험용): (거리, 무게)
    #[cfg(test)]
    pub fn at(&self, x: &V3) -> Option<(f32, f32)> {
        let v = x.map(|a| (a / self.vox).floor() as i32);
        let b = self.blocks.get(&(div_floor(v[0], B), div_floor(v[1], B), div_floor(v[2], B)))?;
        let li = ((v[2].rem_euclid(B) * B + v[1].rem_euclid(B)) * B + v[0].rem_euclid(B)) as usize;
        Some((b.d[li], b.w[li]))
    }

    /// 무게 있는 칸 전부(시험용): (칸 중심, 거리, 무게)
    #[cfg(test)]
    pub fn voxels(&self) -> Vec<(V3, f32, f32)> {
        let mut out = vec![];
        for (k, b) in &self.blocks {
            for i in 0..(B * B * B) as usize {
                if b.w[i] <= 0.0 {
                    continue;
                }
                let (ix, iy, iz) = ((i as i32) % B, (i as i32 / B) % B, i as i32 / (B * B));
                let c = [((k.0 * B + ix) as f64 + 0.5) * self.vox, ((k.1 * B + iy) as f64 + 0.5) * self.vox, ((k.2 * B + iz) as f64 + 0.5) * self.vox];
                out.push((c, b.d[i], b.w[i]));
            }
        }
        out
    }

    /// 표면 근처 거리 값 위치 오차 확인용 잘림 거리
    #[cfg(test)]
    pub fn trunc(&self) -> f64 {
        self.trunc
    }
}
