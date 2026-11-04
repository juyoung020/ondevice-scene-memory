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

    /// 한 장면 더하기: depth(미터, w×h), rgb, 카메라→세계 (r, p), 보정값 k. step: 블록 잡을 때 화소 간격.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate(&mut self, depth: &[f32], rgb: &[u8], w: usize, h: usize, k: [f64; 4], r: &M3, p: &V3, max_depth: f64) {
        let [fx, fy, cx, cy] = k;
        let vs = self.vox;
        let bs = vs * B as f64;
        // 1) 표면 근처 블록 잡기(화소 4 간격, 표면 앞뒤 trunc)
        let mut touch: FastSet<(i32, i32, i32)> = FastSet::default();
        for v in (0..h).step_by(4) {
            for u in (0..w).step_by(4) {
                let z = depth[v * w + u] as f64;
                if !(0.2..max_depth).contains(&z) {
                    continue;
                }
                let ray = [(u as f64 - cx) / fx, (v as f64 - cy) / fy, 1.0];
                for s in [-self.trunc, 0.0, self.trunc] {
                    let zz = z + s;
                    let x = add(&mv(r, &ray.map(|a| a * zz)), p);
                    touch.insert(((x[0] / bs).floor() as i32, (x[1] / bs).floor() as i32, (x[2] / bs).floor() as i32));
                }
            }
        }
        // 2) 잡은 블록의 칸마다: 칸 중심을 이번 장면에 투영해 깊이와 비교. 블록끼리 독립이라 코어마다 나눠 돌린다.
        let rt = [0, 1, 2].map(|i| [0, 1, 2].map(|j| r[j][i]));
        let n = (B * B * B) as usize;
        let trunc = self.trunc;
        let mut work: Vec<((i32, i32, i32), Block, bool)> = touch
            .into_iter()
            .map(|key| {
                let blk = self.blocks.remove(&key).unwrap_or_else(|| Block { d: vec![1.0; n], w: vec![0.0; n], c: vec![[0.0; 3]; n] });
                (key, blk, false)
            })
            .collect();
        let one = |key: (i32, i32, i32), blk: &mut Block| -> bool {
            let mut changed = false;
            for i in 0..n {
                let (ix, iy, iz) = ((i as i32) % B, (i as i32 / B) % B, i as i32 / (B * B));
                let xw = [
                    (key.0 * B + ix) as f64 * vs + vs * 0.5,
                    (key.1 * B + iy) as f64 * vs + vs * 0.5,
                    (key.2 * B + iz) as f64 * vs + vs * 0.5,
                ];
                let pc = mv(&rt, &sub(&xw, p));
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
                let tr = trunc;
                if sdf < -tr {
                    continue; // 표면 뒤 멀리: 모름
                }
                let t = (sdf / tr).min(1.0) as f32;
                // 균일 가중: 1/거리² 가중은 먼 면을 덜 만들어 걸러낼 뿐, 정확도·완전도를 합친 F 는 오히려 낮았다(책상 0.40 → 0.30, surfcmp)
                let wt = 1.0f32;
                let w0 = blk.w[i];
                blk.d[i] = (blk.d[i] * w0 + t * wt) / (w0 + wt);
                if t.abs() < 1.0 {
                    let o = (v * w + u) * 3;
                    for q in 0..3 {
                        blk.c[i][q] = (blk.c[i][q] * w0 + rgb[o + q] as f32 * wt) / (w0 + wt);
                    }
                }
                blk.w[i] = (w0 + wt).min(64.0);
                changed = true;
            }
            changed
        };
        let nth = std::thread::available_parallelism().map_or(4, |x| x.get()).min(8);
        let per = work.len().div_ceil(nth).max(1);
        std::thread::scope(|sc| {
            for part in work.chunks_mut(per) {
                sc.spawn(|| {
                    for (key, blk, ch) in part.iter_mut() {
                        *ch = one(*key, blk);
                    }
                });
            }
        });
        for (key, blk, ch) in work {
            if ch {
                self.dirty.insert((div_floor(key.0, R), div_floor(key.1, R), div_floor(key.2, R)));
            }
            self.blocks.insert(key, blk);
        }
    }
    /// 바뀐 구역의 면(꼭짓점: 위치·색, 삼각형)을 뽑아 돌려준다(구역 → 면). 판 번호도 올린다.
    #[allow(clippy::type_complexity)]
    pub fn take_dirty_meshes(&mut self) -> Vec<((i32, i32, i32), Vec<([f32; 3], [u8; 3])>, Vec<[u32; 3]>)> {
        let regions: Vec<(i32, i32, i32)> = self.dirty.drain().collect();
        // 구역끼리 독립: 코어마다 나눠 뽑는다(읽기만)
        let nth = std::thread::available_parallelism().map_or(4, |x| x.get()).min(8);
        let per = regions.len().div_ceil(nth).max(1);
        let this = &*self;
        let mut out: Vec<_> = std::thread::scope(|sc| {
            let hs: Vec<_> = regions.chunks(per).map(|part| sc.spawn(move || part.iter().map(|&reg| { let (v, t) = this.mesh_region(reg); (reg, v, t) }).collect::<Vec<_>>())).collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
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
