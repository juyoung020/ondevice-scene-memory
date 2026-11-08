//! 표면 텍스처(Open3D color map·mvs-texturing 을 줄인 방식): 표면 구역의 삼각형마다 가장 잘 본 장면 사진을 골라 붙인다.
//! - 장면 사진 보관: 지도에 넣은 장면 중 선명하고(라플라시안 분산) 서로 다른 자리에서 본 것만 JPEG 로 보관한다.
//!   비슷한 자리(12 cm·12° 안)에 이미 있으면 10 % 넘게 선명할 때만 바꿔 넣는다. 위치는 장면 보관(`kfstore`)과 같이 기준 노드 × 상대 위치라 루프 클로저를 따른다.
//! - 고르기: 삼각형마다 정면일수록·가까울수록·선명할수록 높은 점수, 보이는지는 보관한 깊이(4 화소 간격)와 맞대 본다(가려짐·깊이 경계 제외).
//!   이웃 삼각형과 같은 장면을 쓰도록 몇 번 다듬고(ICM), 같은 장면 삼각형 덩어리(차트)마다 사진 조각을 잘라 구역 아틀라스 한 장에 모은다.
//! - 장면마다 노출이 달라 생기는 이음매는 꼭짓점마다 표면 색(모든 장면 평균)에 맞춘 색 배율(꼭짓점 색)로 줄이고,
//!   차트 경계에 닿은 삼각형은 이웃 사진과 섞어(꼭짓점 사진 무게를 보간) 경계 양쪽 색을 잇는다.
//! - 작업 스레드 하나가 맡는다: 실시간 장면 처리는 사진을 넘기기만 하고, 2 초 넘게 안 바뀐 구역부터 하나씩(코어 하나의 절반만 씀) 붙인다.
use crate::geom::*;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

pub type Reg = (i32, i32, i32);
pub type Verts = Vec<([f32; 3], [u8; 3])>;
pub type Tris = Vec<[u32; 3]>;

const Q: usize = 4; // 보관 깊이 화소 간격
const SIM_T: f64 = 0.12; // 비슷한 자리: 거리(m)
const SIM_R: f64 = 12.0; // 비슷한 자리: 보는 방향(°)
const MOVE_T: f64 = 0.01; // 장면이 이만큼 움직이면 그 장면을 쓴 구역을 다시 붙임
const MOVE_R: f64 = 0.5;
const STABLE: f64 = 2.0; // 구역 면이 이 초 동안 안 바뀌면 붙임
const MAX_CAND: usize = 24; // 구역마다 따져 볼 장면 수
const ATLAS: usize = 1024; // 구역 아틀라스 최대 한 변
const PAD: usize = 2; // 차트 둘레 여백(화소)
const MARGIN: f64 = 6.0; // 사진 가장자리 제외(화소)
const KB: usize = 6; // 삼각형마다 남기는 후보 장면 수

/// 실시간 장면 하나(작업 스레드로 넘김). rel: 기준 노드에서 본 상대 위치(기준 노드 0 = 받은 위치 그대로)
pub struct FrameIn {
    pub anchor: u32,
    pub rel: (M3, V3),
    pub w: usize,
    pub h: usize,
    pub k: [f64; 4],
    pub rgb: Vec<u8>,
    pub dq: Vec<u16>, // 깊이 mm, 4 화소 간격
    pub fix: M3,      // 수평 맞춤 회전
    pub z0: f64,      // 바닥 높이 원점
}

enum Ctl {
    Graph(Vec<(u32, M3, V3)>),
    Mesh(Reg, u32, Arc<(Verts, Tris)>),
    Finish,
}

enum Out {
    Done(Reg, u32, u32), // 구역, 면 판 번호, 텍스처 번호(0 = 없음)
    Finished(String),
}

/// 깊이(m) → 4 화소 간격 mm
pub fn depth_grid(depth: &[f32], w: usize, h: usize) -> Vec<u16> {
    let (qw, qh) = (w.div_ceil(Q), h.div_ceil(Q));
    (0..qw * qh).map(|i| (depth[(i / qw) * Q * w + (i % qw) * Q] * 1000.0).round().clamp(0.0, 65535.0) as u16).collect()
}

/// 선명도: 절반 크기 회색의 라플라시안 분산
pub fn sharpness(rgb: &[u8], w: usize, h: usize) -> f32 {
    let (hw, hh) = (w / 2, h / 2);
    let g: Vec<f32> = (0..hw * hh)
        .map(|i| {
            let (x, y) = (i % hw * 2, i / hw * 2);
            let mut s = 0.0;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let o = ((y + dy) * w + x + dx) * 3;
                s += 0.299 * rgb[o] as f32 + 0.587 * rgb[o + 1] as f32 + 0.114 * rgb[o + 2] as f32;
            }
            s * 0.25
        })
        .collect();
    let (mut s, mut s2, mut n) = (0f64, 0f64, 0f64);
    for y in 1..hh.saturating_sub(1) {
        for x in 1..hw - 1 {
            let i = y * hw + x;
            let l = (g[i - 1] + g[i + 1] + g[i - hw] + g[i + hw] - 4.0 * g[i]) as f64;
            s += l;
            s2 += l * l;
            n += 1.0;
        }
    }
    if n < 1.0 { 0.0 } else { (s2 / n - (s / n).powi(2)) as f32 }
}

/// 텍스처를 고를 때 쓰는 장면(위치는 지도 좌표, 카메라 → 지도)
#[derive(Clone)]
pub struct Cam {
    pub id: u32,
    pub r: M3,
    pub p: V3,
    pub k: [f64; 4],
    pub w: usize,
    pub h: usize,
    pub dq: Arc<Vec<u16>>,
    pub sharp: f32,
}

impl Cam {
    fn proj(&self, x: &V3) -> Option<(f64, f64, f64)> {
        let d = sub(x, &self.p);
        let r = &self.r;
        let c = [r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2], r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2], r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2]];
        (c[2] > 0.1).then(|| (self.k[0] * c[0] / c[2] + self.k[2], self.k[1] * c[1] / c[2] + self.k[3], c[2]))
    }
    fn grid(&self, u: f64, v: f64) -> (usize, usize, usize, usize) {
        let (qw, qh) = (self.w.div_ceil(Q), self.h.div_ceil(Q));
        (((u / Q as f64).round().max(0.0) as usize).min(qw - 1), ((v / Q as f64).round().max(0.0) as usize).min(qh - 1), qw, qh)
    }
    /// 보관 깊이와 맞으면 보이는 점
    fn visible(&self, u: f64, v: f64, z: f64) -> bool {
        let (i, j, qw, _) = self.grid(u, v);
        let d = self.dq[j * qw + i] as f64 * 1e-3;
        d > 0.0 && (d - z).abs() < 0.03 + 0.015 * z * z
    }
    /// 깊이 경계 근처(이웃 3×3 칸에 빈 값이 있거나 깊이 차가 큼): 앞 물체 색이 뒤 면에 번지는 자리
    fn edge(&self, u: f64, v: f64) -> bool {
        let (i, j, qw, qh) = self.grid(u, v);
        let (mut lo, mut hi) = (f64::MAX, 0f64);
        for y in j.saturating_sub(1)..=(j + 1).min(qh - 1) {
            for x in i.saturating_sub(1)..=(i + 1).min(qw - 1) {
                let d = self.dq[y * qw + x] as f64 * 1e-3;
                if d <= 0.0 {
                    return true;
                }
                (lo, hi) = (lo.min(d), hi.max(d));
            }
        }
        hi - lo > 0.15 * lo
    }
}

/// 텍스처 붙인 구역 면: 꼭짓점(위치, 법선, UV, 색). 그릴 색 = 아틀라스 × 꼭짓점 색 × 2: 사진을 붙인 꼭짓점은 노출 맞춤 배율 ÷ 2, 사진이 없는 삼각형은 흰 칸 + 표면 색 ÷ 2
pub struct TexMesh {
    pub verts: Vec<([f32; 3], [f32; 3], [f32; 2], [u8; 3])>,
    pub tris: Vec<[u32; 3]>,
    pub atlas: (usize, usize, Vec<u8>),
    pub used: Vec<u32>,
    #[allow(dead_code)]
    pub n_tex: usize, // 사진을 붙인 삼각형 수
}

fn p64(v: &[f32; 3]) -> V3 {
    [v[0] as f64, v[1] as f64, v[2] as f64]
}

/// 삼각형마다 점수 높은 장면 최대 KB 개(점수, 장면 번호). 법선 방향(바깥 = 빈 공간 쪽)에서 본 장면만.
fn score_views(verts: &Verts, tris: &Tris, cams: &[Cam]) -> (Vec<[(f32, u16); KB]>, Vec<usize>) {
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for v in verts {
        for q in 0..3 {
            lo[q] = lo[q].min(v.0[q] as f64);
            hi[q] = hi[q].max(v.0[q] as f64);
        }
    }
    let c = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    let rad = norm(&sub(&hi, &c));
    let mut cand: Vec<(f64, usize)> = cams
        .iter()
        .enumerate()
        .filter_map(|(i, cam)| {
            let d = norm(&sub(&c, &cam.p));
            let (u, v, z) = if d < rad { (cam.k[2], cam.k[3], rad) } else { cam.proj(&c)? };
            let m = rad * cam.k[0] / z;
            (z < 6.0 && u > -m && v > -m && u < cam.w as f64 + m && v < cam.h as f64 + m).then(|| (cam.sharp as f64 / (z * z).max(0.25), i))
        })
        .collect();
    cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    cand.truncate(MAX_CAND);
    let cand: Vec<usize> = cand.into_iter().map(|x| x.1).collect();
    let mut sh: Vec<f32> = cand.iter().map(|&i| cams[i].sharp).collect();
    sh.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sh.get(sh.len() / 2).copied().unwrap_or(1.0).max(1e-3) as f64;
    let best = tris
        .iter()
        .map(|t| {
            let (a, b, cc) = (p64(&verts[t[0] as usize].0), p64(&verts[t[1] as usize].0), p64(&verts[t[2] as usize].0));
            let n = cross(&sub(&b, &a), &sub(&cc, &a));
            let ln = norm(&n);
            let mut top = [(0f32, 0u16); KB];
            if ln < 1e-12 {
                return top;
            }
            let ctr = [(a[0] + b[0] + cc[0]) / 3.0, (a[1] + b[1] + cc[1]) / 3.0, (a[2] + b[2] + cc[2]) / 3.0];
            for (ci, &i) in cand.iter().enumerate() {
                let cam = &cams[i];
                let d = sub(&cam.p, &ctr);
                let cos = dot(&n, &d) / (ln * norm(&d));
                if cos < 0.2 {
                    continue;
                }
                let ok = [a, b, cc].iter().all(|x| match cam.proj(x) {
                    Some((u, v, z)) => u >= MARGIN && v >= MARGIN && u < cam.w as f64 - MARGIN && v < cam.h as f64 - MARGIN && cam.visible(u, v, z),
                    None => false,
                });
                let Some((u, v, z)) = cam.proj(&ctr) else { continue };
                if !ok || cam.edge(u, v) {
                    continue;
                }
                let rr = ((u - cam.k[2]) / cam.k[2]).powi(2) + ((v - cam.k[3]) / cam.k[3]).powi(2);
                let s = (cos / (z * z) * (cam.sharp as f64 / med).sqrt().min(2.0) * (1.0 - 0.15 * rr)) as f32;
                if s > top[KB - 1].0 {
                    top[KB - 1] = (s, ci as u16);
                    top.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
                }
            }
            top
        })
        .collect();
    (best, cand)
}

/// 삼각형 이웃(변을 같이 쓰는 삼각형)
fn neighbours(tris: &Tris) -> Vec<[u32; 3]> {
    let mut e: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (i, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            e.entry((a.min(b), a.max(b))).or_default().push(i as u32);
        }
    }
    let mut nb = vec![[u32::MAX; 3]; tris.len()];
    for (i, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if let Some(o) = e[&(a.min(b), a.max(b))].iter().find(|&&j| j != i as u32) {
                nb[i][k] = *o;
            }
        }
    }
    nb
}

/// 장면 고르기 + 이웃끼리 같은 장면 쓰게 다듬기(ICM: 점수 손해 + 이웃과 다른 수 × λ 를 줄임). u16::MAX = 사진 없음
pub fn label(best: &[[(f32, u16); KB]], nb: &[[u32; 3]]) -> Vec<u16> {
    const LAMBDA: f32 = 1.0;
    let mut lab: Vec<u16> = best.iter().map(|b| if b[0].0 > 0.0 { b[0].1 } else { u16::MAX }).collect();
    for _ in 0..6 {
        let mut changed = false;
        for i in 0..best.len() {
            let b = &best[i];
            if b[0].0 <= 0.0 {
                continue;
            }
            let (mut bc, mut bl) = (f32::MAX, lab[i]);
            for &(s, l) in b.iter().filter(|x| x.0 > 0.0) {
                let diff = nb[i].iter().filter(|&&j| j != u32::MAX && lab[j as usize] != u16::MAX && lab[j as usize] != l).count();
                let cost = (1.0 - s / b[0].0) + LAMBDA * diff as f32;
                if cost < bc - 1e-6 {
                    (bc, bl) = (cost, l);
                }
            }
            if bl != lab[i] {
                lab[i] = bl;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    lab
}

/// 선반 쌓기: 높이 큰 것부터 줄마다 왼쪽에서 오른쪽. 돌려줌: (높이, 자리)
pub fn pack(sizes: &[(usize, usize)], width: usize) -> (usize, Vec<(usize, usize)>) {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&a, &b| sizes[b].1.cmp(&sizes[a].1).then(sizes[b].0.cmp(&sizes[a].0)));
    let (mut x, mut y, mut sh) = (0, 0, 0);
    let mut pos = vec![(0, 0); sizes.len()];
    for i in order {
        let (w, h) = sizes[i];
        if x + w > width && x > 0 {
            (x, y, sh) = (0, y + sh, 0);
        }
        pos[i] = (x, y);
        x += w;
        sh = sh.max(h);
    }
    (y + sh, pos)
}

fn bilinear(img: &[u8], w: usize, h: usize, u: f64, v: f64) -> [f32; 3] {
    let (u, v) = (u.clamp(0.0, w as f64 - 1.001), v.clamp(0.0, h as f64 - 1.001));
    let (x, y) = (u as usize, v as usize);
    let (fx, fy) = ((u - x as f64) as f32, (v - y as f64) as f32);
    let mut o = [0f32; 3];
    for (dx, dy, wt) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
        let i = ((y + dy) * w + x + dx) * 3;
        for q in 0..3 {
            o[q] += wt * img[i + q] as f32;
        }
    }
    o
}

/// 노출 맞춤 배율: 표면 색(모든 장면 평균) ÷ 사진 색(사진 좌표 u, v 둘레 5×5 흐린 값)
fn raw_gain(im: &[u8], cam: &Cam, u: f64, v: f64, surf: &[u8; 3]) -> [f32; 3] {
    let mut c = [0f32; 3];
    for dy in -2..=2 {
        for dx in -2..=2 {
            let s = bilinear(im, cam.w, cam.h, u + 2.0 * dx as f64, v + 2.0 * dy as f64);
            for q in 0..3 {
                c[q] += s[q] / 25.0;
            }
        }
    }
    [0, 1, 2].map(|q| (surf[q] as f32 + 10.0) / (c[q] + 10.0))
}

/// 구역 면 하나에 텍스처: 사진을 붙일 삼각형이 하나도 없으면 None. img: 장면 id → 컬러(w×h×3)
pub fn texture_region(verts: &Verts, tris: &Tris, cams: &[Cam], img: &dyn Fn(u32) -> Option<Arc<Vec<u8>>>) -> Option<TexMesh> {
    if tris.is_empty() || cams.is_empty() {
        return None;
    }
    let (best, cand) = score_views(verts, tris, cams);
    let nb = neighbours(tris);
    let lab = label(&best, &nb);
    let n_tex = lab.iter().filter(|&&l| l != u16::MAX).count();
    if n_tex == 0 {
        return None;
    }
    // 차트: 같은 장면 삼각형끼리 이어진 덩어리(합집합 찾기)
    let mut par: Vec<u32> = (0..tris.len() as u32).collect();
    fn find(p: &mut [u32], mut i: u32) -> u32 {
        while p[i as usize] != i {
            p[i as usize] = p[p[i as usize] as usize];
            i = p[i as usize];
        }
        i
    }
    for i in 0..tris.len() {
        if lab[i] == u16::MAX {
            continue;
        }
        for &j in &nb[i] {
            if j != u32::MAX && lab[j as usize] == lab[i] {
                let (a, b) = (find(&mut par, i as u32), find(&mut par, j));
                if a != b {
                    par[a.max(b) as usize] = a.min(b);
                }
            }
        }
    }
    let mut chart_of = vec![u32::MAX; tris.len()];
    let mut charts: Vec<(u16, Vec<u32>)> = vec![]; // (장면, 삼각형)
    let mut root_id: HashMap<u32, u32> = HashMap::new();
    for i in 0..tris.len() {
        if lab[i] == u16::MAX {
            continue;
        }
        let r = find(&mut par, i as u32);
        let id = *root_id.entry(r).or_insert_with(|| {
            charts.push((lab[i], vec![]));
            charts.len() as u32 - 1
        });
        charts[id as usize].1.push(i as u32);
        chart_of[i] = id;
    }
    // 사진 풀기(이긴 장면만)
    let mut used: Vec<u16> = charts.iter().map(|c| c.0).collect();
    used.sort();
    used.dedup();
    let mut imgs: HashMap<u16, Arc<Vec<u8>>> = HashMap::new();
    for &l in &used {
        let im = img(cams[cand[l as usize]].id)?;
        imgs.insert(l, im);
    }
    // 차트마다: 꼭짓점의 사진 좌표, 둘레 상자, 밝기 배율
    struct Ch {
        cam: usize,
        uv: HashMap<u32, (f64, f64)>,
        lo: (f64, f64),
        hi: (f64, f64),
        gain: HashMap<u32, [f32; 3]>, // 꼭짓점 색 배율
    }
    let mut chs: Vec<Ch> = charts
        .iter()
        .map(|(l, ts)| {
            let cam = &cams[cand[*l as usize]];
            let mut uv = HashMap::new();
            let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
            for &t in ts {
                for &vi in &tris[t as usize] {
                    let e = uv.entry(vi).or_insert_with(|| {
                        let (u, v, _) = cam.proj(&p64(&verts[vi as usize].0)).unwrap_or((cam.k[2], cam.k[3], 1.0));
                        (u.clamp(0.0, cam.w as f64 - 1.0), v.clamp(0.0, cam.h as f64 - 1.0))
                    });
                    lo = (lo.0.min(e.0), lo.1.min(e.1));
                    hi = (hi.0.max(e.0), hi.1.max(e.1));
                }
            }
            // 노출 맞춤(mvs-texturing 의 이음매 고르기를 줄인 것): 꼭짓점마다 표면 색(모든 장면 평균) ÷ 사진 색(5×5 흐린 값),
            // 차트 안 이웃끼리 세 번 평균해 꼭짓점 색(배율)으로 둔다 → 차트 경계 양쪽이 같은 표면 색에 맞아 이음매가 줄고, 안쪽 무늬는 사진 그대로
            let im = &imgs[l];
            let mut gain: HashMap<u32, [f32; 3]> = uv.iter().map(|(&vi, &(u, v))| (vi, raw_gain(im, cam, u, v, &verts[vi as usize].1))).collect();
            let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
            for &t in ts {
                let tv = tris[t as usize];
                for k in 0..3 {
                    adj.entry(tv[k]).or_default().push(tv[(k + 1) % 3]);
                    adj.entry(tv[(k + 1) % 3]).or_default().push(tv[k]);
                }
            }
            for _ in 0..3 {
                gain = gain
                    .iter()
                    .map(|(&vi, g)| {
                        let mut s = *g;
                        let nb = &adj[&vi];
                        for n in nb {
                            let gn = gain[n];
                            for q in 0..3 {
                                s[q] += gn[q];
                            }
                        }
                        (vi, s.map(|x| x / (1 + nb.len()) as f32))
                    })
                    .collect();
            }
            for g in gain.values_mut() {
                *g = g.map(|x| x.clamp(0.7, 1.4));
            }
            Ch { cam: cand[*l as usize], uv, lo, hi, gain }
        })
        .collect();
    let untex = n_tex < tris.len();
    // 아틀라스 크기: 사진 화소 그대로(배율 1), 넘치면 줄임
    let area: f64 = chs.iter().map(|c| (c.hi.0 - c.lo.0 + 1.0 + 2.0 * PAD as f64) * (c.hi.1 - c.lo.1 + 1.0 + 2.0 * PAD as f64)).sum();
    let mut s = (0.8 * (ATLAS * ATLAS) as f64 / area).sqrt().min(1.0);
    let (aw, ah, pos, sizes) = loop {
        let mut sizes: Vec<(usize, usize)> = chs.iter().map(|c| (((c.hi.0 - c.lo.0) * s).ceil() as usize + 1 + 2 * PAD, ((c.hi.1 - c.lo.1) * s).ceil() as usize + 1 + 2 * PAD)).collect();
        if untex {
            sizes.push((4, 4));
        }
        let a: usize = sizes.iter().map(|x| x.0 * x.1).sum();
        let maxw = sizes.iter().map(|x| x.0).max().unwrap_or(4);
        let width = (((a as f64 * 1.15).sqrt().ceil() as usize).next_power_of_two()).max(maxw.next_power_of_two()).clamp(16, ATLAS);
        let (hgt, pos) = pack(&sizes, width);
        if (hgt <= ATLAS && maxw <= width) || s < 0.05 {
            break (width, hgt.next_power_of_two().max(16).min(ATLAS.max(hgt.next_power_of_two())), pos, sizes);
        }
        s *= 0.9;
    };
    let mut atlas = vec![128u8; aw * ah * 3];
    for (ci, c) in chs.iter_mut().enumerate() {
        let cam = &cams[c.cam];
        let im = &imgs[&charts[ci].0];
        let (x0, y0) = pos[ci];
        let (rw, rh) = sizes[ci];
        // 아틀라스 화소 (x0 + PAD + i) 중심 ↔ 사진 좌표 lo + i / s
        for j in 0..rh {
            for i in 0..rw {
                let (u, v) = (c.lo.0 + (i as f64 - PAD as f64) / s, c.lo.1 + (j as f64 - PAD as f64) / s);
                let col = bilinear(im, cam.w, cam.h, u, v);
                let o = ((y0 + j) * aw + x0 + i) * 3;
                if o + 2 < atlas.len() {
                    for q in 0..3 {
                        atlas[o + q] = col[q].round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        // 꼭짓점 사진 좌표 → 아틀라스 화소 좌표(화소 i 의 가운데 = i + 0.5)
        for e in c.uv.values_mut() {
            *e = (x0 as f64 + PAD as f64 + (e.0 - c.lo.0) * s + 0.5, y0 as f64 + PAD as f64 + (e.1 - c.lo.1) * s + 0.5);
        }
    }
    blend_seams(verts, tris, &lab, &chart_of, &chs.iter().map(|c| (c.cam, &c.uv, &c.gain)).collect::<Vec<_>>(), cams, &cand, &imgs, (aw, ah, &mut atlas));
    // 아틀라스 화소 좌표 → UV(텍스처 v 는 아래가 0)
    for c in chs.iter_mut() {
        for e in c.uv.values_mut() {
            *e = (e.0 / aw as f64, 1.0 - e.1 / ah as f64);
        }
    }
    let white = untex.then(|| {
        let (x0, y0) = pos[chs.len()];
        for j in 0..4 {
            for i in 0..4 {
                let o = ((y0 + j) * aw + x0 + i) * 3;
                atlas[o..o + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
        ((x0 as f64 + 2.0) / aw as f64, 1.0 - (y0 as f64 + 2.0) / ah as f64)
    });
    // 꼭짓점 법선(넓이 무게, 원래 면에서: 차트 경계에서도 매끈)
    let mut vn = vec![[0f64; 3]; verts.len()];
    for t in tris {
        let (a, b, c) = (p64(&verts[t[0] as usize].0), p64(&verts[t[1] as usize].0), p64(&verts[t[2] as usize].0));
        let n = cross(&sub(&b, &a), &sub(&c, &a));
        for &i in t {
            vn[i as usize] = add(&vn[i as usize], &n);
        }
    }
    let mut out_v = vec![];
    let mut map: HashMap<(u32, u32), u32> = HashMap::new();
    let mut out_t = Vec::with_capacity(tris.len());
    for (ti, t) in tris.iter().enumerate() {
        let ch = chart_of[ti];
        out_t.push(t.map(|vi| {
            *map.entry((vi, ch)).or_insert_with(|| {
                let v = &verts[vi as usize];
                let n = vn[vi as usize];
                let l = norm(&n).max(1e-12);
                let (uv, col) = if ch == u32::MAX { (white.unwrap(), v.1.map(|x| x / 2)) } else { (chs[ch as usize].uv[&vi], chs[ch as usize].gain[&vi].map(|g| (g * 127.5).round() as u8)) };
                out_v.push((v.0, [(n[0] / l) as f32, (n[1] / l) as f32, (n[2] / l) as f32], [uv.0 as f32, uv.1 as f32], col));
                out_v.len() as u32 - 1
            })
        }));
    }
    Some(TexMesh { verts: out_v, tris: out_t, atlas: (aw, ah, atlas), used: used.iter().map(|&l| cams[cand[l as usize]].id).collect(), n_tex })
}

/// 이음매 섞기(경계 띠만 여러 장면 가중 평균): 꼭짓점마다 장면 무게 = 둘레 삼각형 중 그 장면 삼각형의 비율.
/// 차트 경계 꼭짓점은 양쪽 차트에서 같은 무게라 경계 양쪽 색이 이어지고, 무게는 삼각형 안에서 보간돼 삼각형 한 개 폭 안에서 이웃 장면 사진으로 넘어간다.
/// 이웃 장면 색은 (그 장면 노출 배율 ÷ 이 차트 배율)로 맞추고, 그 장면에서 가려진 점은 섞지 않는다. 경계에 닿은 삼각형만 칠한다
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn blend_seams(
    verts: &Verts, tris: &Tris, lab: &[u16], chart_of: &[u32], chs: &[(usize, &HashMap<u32, (f64, f64)>, &HashMap<u32, [f32; 3]>)], cams: &[Cam], cand: &[usize],
    imgs: &HashMap<u16, Arc<Vec<u8>>>, atlas: (usize, usize, &mut [u8]),
) {
    let (aw, ah, atlas) = atlas;
    let mut vw: HashMap<u32, Vec<(u16, f32)>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        if lab[ti] == u16::MAX {
            continue;
        }
        for &vi in t {
            let e = vw.entry(vi).or_default();
            match e.iter_mut().find(|x| x.0 == lab[ti]) {
                Some(x) => x.1 += 1.0,
                None => e.push((lab[ti], 1.0)),
            }
        }
    }
    for e in vw.values_mut() {
        let s: f32 = e.iter().map(|x| x.1).sum();
        e.iter_mut().for_each(|x| x.1 /= s);
    }
    let inside = |cam: &Cam, u: f64, v: f64| u >= MARGIN && v >= MARGIN && u < cam.w as f64 - MARGIN && v < cam.h as f64 - MARGIN;
    for (ti, t) in tris.iter().enumerate() {
        let ch = chart_of[ti];
        if ch == u32::MAX || t.iter().all(|vi| vw[vi].len() < 2) {
            continue;
        }
        let own = lab[ti];
        let (_, uv, gain) = chs[ch as usize];
        // 이웃 장면: (장면, 꼭짓점별 무게, 노출 배율 비)
        let mut others: Vec<(u16, [f32; 3], [f32; 3])> = vec![];
        for (k, vi) in t.iter().enumerate() {
            for &(l, w) in &vw[vi] {
                if l == own {
                    continue;
                }
                let i = others.iter().position(|o| o.0 == l).unwrap_or_else(|| {
                    others.push((l, [0.0; 3], [0.0; 3]));
                    others.len() - 1
                });
                others[i].1[k] = w;
            }
        }
        others.retain_mut(|o| {
            let (cam, im) = (&cams[cand[o.0 as usize]], &imgs[&o.0]);
            let (mut r, mut n) = ([0f32; 3], 0f32);
            for &vi in t {
                let v = &verts[vi as usize];
                if let Some((u, vv, _)) = cam.proj(&p64(&v.0)).filter(|&(u, vv, _)| inside(cam, u, vv)) {
                    let (g, ga) = (raw_gain(im, cam, u, vv, &v.1), gain[&vi]);
                    for q in 0..3 {
                        r[q] += g[q].clamp(0.7, 1.4) / ga[q];
                    }
                    n += 1.0;
                }
            }
            o.2 = r.map(|x| x / n.max(1.0));
            n > 0.0
        });
        if others.is_empty() {
            continue;
        }
        let p = t.map(|vi| uv[&vi]);
        let x = t.map(|vi| p64(&verts[vi as usize].0));
        let wa = t.map(|vi| vw[&vi].iter().find(|x| x.0 == own).map_or(0.0, |x| x.1));
        let det = (p[1].0 - p[0].0) * (p[2].1 - p[0].1) - (p[2].0 - p[0].0) * (p[1].1 - p[0].1);
        if det.abs() < 1e-9 {
            continue;
        }
        let lo = |a: f64, b: f64, c: f64| (a.min(b).min(c) - 0.5).floor().max(0.0) as usize;
        let (x0, y0) = (lo(p[0].0, p[1].0, p[2].0), lo(p[0].1, p[1].1, p[2].1));
        let (x1, y1) = ((p[0].0.max(p[1].0).max(p[2].0) as usize).min(aw - 1), (p[0].1.max(p[1].1).max(p[2].1) as usize).min(ah - 1));
        for py in y0..=y1 {
            for px in x0..=x1 {
                let (cx, cy) = (px as f64 + 0.5, py as f64 + 0.5);
                let b1 = ((cx - p[0].0) * (p[2].1 - p[0].1) - (p[2].0 - p[0].0) * (cy - p[0].1)) / det;
                let b2 = ((p[1].0 - p[0].0) * (cy - p[0].1) - (cx - p[0].0) * (p[1].1 - p[0].1)) / det;
                let b = [1.0 - b1 - b2, b1, b2];
                if b.iter().any(|&v| v < -0.1) {
                    continue;
                }
                let b = b.map(|v| v.max(0.0));
                let bs = b[0] + b[1] + b[2];
                let b = b.map(|v| v / bs);
                let o = (py * aw + px) * 3;
                let w_own = (0..3).map(|k| b[k] as f32 * wa[k]).sum::<f32>();
                let mut acc = [0, 1, 2].map(|q| atlas[o + q] as f32 * w_own);
                let mut ws = w_own;
                let xp = [0, 1, 2].map(|q| b[0] * x[0][q] + b[1] * x[1][q] + b[2] * x[2][q]);
                for (l, w3, ratio) in &others {
                    let w = (0..3).map(|k| b[k] as f32 * w3[k]).sum::<f32>();
                    if w < 0.01 {
                        continue;
                    }
                    let cam = &cams[cand[*l as usize]];
                    let Some((u, v, z)) = cam.proj(&xp) else { continue };
                    if !inside(cam, u, v) || !cam.visible(u, v, z) {
                        continue;
                    }
                    let c = bilinear(&imgs[l], cam.w, cam.h, u, v);
                    for q in 0..3 {
                        acc[q] += w * c[q] * ratio[q];
                    }
                    ws += w;
                }
                if ws > w_own {
                    for q in 0..3 {
                        atlas[o + q] = (acc[q] / ws).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
}

/// 텍스처 면 PLY: 위치·법선·UV(s, t)·색 + 삼각형
pub fn ply_tex_bytes(m: &TexMesh) -> Vec<u8> {
    use std::io::Write;
    let mut b = Vec::with_capacity(m.verts.len() * 35 + m.tris.len() * 13 + 320);
    write!(b, "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float s\nproperty float t\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n", m.verts.len(), m.tris.len()).unwrap();
    for (p, n, uv, c) in &m.verts {
        for v in p.iter().chain(n).chain(uv) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(c);
    }
    for t in &m.tris {
        b.push(3);
        for v in t {
            b.extend_from_slice(&(*v as i32).to_le_bytes());
        }
    }
    b
}

fn jpeg(rgb: &[u8], w: usize, h: usize, q: u8) -> Vec<u8> {
    let mut v = vec![];
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut v, q).encode(rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8).unwrap();
    v
}

fn mul(a: &(M3, V3), b: &(M3, V3)) -> (M3, V3) {
    (mm(&a.0, &b.0), add(&mv(&a.0, &b.1), &a.1))
}

fn angle(a: &M3, b: &M3) -> f64 {
    let mut tr = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            tr += a[j][i] * b[j][i];
        }
    }
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}

struct View {
    anchor: u32,
    rel: (M3, V3),
    w: usize,
    h: usize,
    k: [f64; 4],
    jpg: Arc<Vec<u8>>,
    dq: Arc<Vec<u16>>,
    sharp: f32,
    pose: (M3, V3), // 지도 좌표(카메라 → 지도)
    epoch: u32,     // 위치가 바뀔 때마다 늘어남
}

impl View {
    fn bytes(&self) -> usize {
        self.jpg.len() + self.dq.len() * 2
    }
}

struct RegState {
    ver: u32,
    mesh: Arc<(Verts, Tris)>,
    t_mesh: Instant,
    done: Option<(u32, Vec<(u32, u32)>)>, // 붙인 면 판 번호, 쓴 장면(id, epoch)
    tv: u32,                              // 텍스처 번호(파일 이름)
}

struct Worker {
    dir: PathBuf,
    cap: usize,
    nodes: HashMap<u32, (M3, V3)>,
    fix: M3,
    z0: f64,
    views: BTreeMap<u32, View>,
    next_id: u32,
    bytes: usize,
    regs: BTreeMap<Reg, RegState>,
    cache: Mutex<(HashMap<u32, Arc<Vec<u8>>>, VecDeque<u32>)>,
    n_frames: usize,
    n_regions: usize,
    secs: f64,
    secs_max: f64,
    secs_frames: f64,
    n_reused: usize, // 면이 그대로라 텍스처를 다시 붙이지 않은 횟수
}

/// 두 면이 같은 모양인가: 삼각형이 같고 꼭짓점이 1 mm 안(색은 보지 않음 — 텍스처가 덮음)
fn same_mesh(a: &(Verts, Tris), b: &(Verts, Tris)) -> bool {
    a.1 == b.1 && a.0.len() == b.0.len() && a.0.iter().zip(&b.0).all(|(p, q)| (0..3).all(|i| (p.0[i] - q.0[i]).abs() < 1e-3))
}

impl Worker {
    fn world(&self, anchor: u32, rel: &(M3, V3)) -> (M3, V3) {
        let (r, p) = match self.nodes.get(&anchor) {
            Some(n) if anchor != 0 => mul(n, rel),
            _ => *rel,
        };
        let q = mv(&self.fix, &p);
        (mm(&self.fix, &r), [q[0], q[1], q[2] - self.z0])
    }

    fn add_frame(&mut self, f: FrameIn) {
        let c0 = Instant::now();
        (self.fix, self.z0) = (f.fix, f.z0);
        self.n_frames += 1;
        let pose = self.world(f.anchor, &f.rel);
        let sharp = sharpness(&f.rgb, f.w, f.h);
        let axis = |r: &M3| [r[0][2], r[1][2], r[2][2]];
        let a0 = axis(&pose.0);
        let sim: Vec<u32> = self
            .views
            .iter()
            .filter(|(_, v)| norm(&sub(&v.pose.1, &pose.1)) < SIM_T && dot(&axis(&v.pose.0), &a0).clamp(-1.0, 1.0).acos().to_degrees() < SIM_R)
            .map(|(id, _)| *id)
            .collect();
        let best = sim.iter().map(|id| self.views[id].sharp).fold(0f32, f32::max);
        if !sim.is_empty() && sharp <= 1.1 * best {
            self.secs_frames += c0.elapsed().as_secs_f64();
            return;
        }
        for id in sim {
            if let Some(v) = self.views.remove(&id) {
                self.bytes -= v.bytes();
            }
        }
        let v = View { anchor: f.anchor, rel: f.rel, w: f.w, h: f.h, k: f.k, jpg: Arc::new(jpeg(&f.rgb, f.w, f.h, 90)), dq: Arc::new(f.dq), sharp, pose, epoch: 0 };
        self.bytes += v.bytes();
        self.views.insert(self.next_id, v);
        self.next_id += 1;
        // 예산을 넘으면: 가까운(비슷한 자리 두 배 안) 이웃이 있는 장면 중 가장 흐린 것부터 버림
        while self.bytes > self.cap && self.views.len() > 2 {
            let ids: Vec<u32> = self.views.keys().copied().collect();
            let victim = ids
                .iter()
                .filter(|&&i| {
                    let v = &self.views[&i];
                    ids.iter().any(|&j| j != i && norm(&sub(&self.views[&j].pose.1, &v.pose.1)) < 2.0 * SIM_T && angle(&self.views[&j].pose.0, &v.pose.0) < 2.0 * SIM_R)
                })
                .min_by(|&&a, &&b| self.views[&a].sharp.partial_cmp(&self.views[&b].sharp).unwrap())
                .or(ids.first())
                .copied()
                .unwrap();
            let v = self.views.remove(&victim).unwrap();
            self.bytes -= v.bytes();
        }
        crate::out::slow(c0);
        self.secs_frames += c0.elapsed().as_secs_f64();
    }

    fn graph(&mut self, g: Vec<(u32, M3, V3)>) {
        for (id, r, p) in g {
            self.nodes.insert(id, (r, p));
        }
        let ids: Vec<u32> = self.views.keys().copied().collect();
        for id in ids {
            let (anchor, rel) = (self.views[&id].anchor, self.views[&id].rel);
            let np = self.world(anchor, &rel);
            let v = self.views.get_mut(&id).unwrap();
            if norm(&sub(&np.1, &v.pose.1)) > MOVE_T || angle(&np.0, &v.pose.0) > MOVE_R {
                v.pose = np;
                v.epoch += 1;
            }
        }
    }

    fn stale(&self, st: &RegState) -> bool {
        match &st.done {
            None => true,
            Some((ver, used)) => *ver != st.ver || used.iter().any(|(id, ep)| self.views.get(id).is_none_or(|v| v.epoch != *ep)),
        }
    }

    fn cams(&self) -> Vec<Cam> {
        self.views.iter().map(|(&id, v)| Cam { id, r: v.pose.0, p: v.pose.1, k: v.k, w: v.w, h: v.h, dq: v.dq.clone(), sharp: v.sharp }).collect()
    }

    fn image(&self, id: u32) -> Option<Arc<Vec<u8>>> {
        if let Some(im) = self.cache.lock().unwrap().0.get(&id) {
            return Some(im.clone());
        }
        let v = self.views.get(&id)?;
        let im = Arc::new(image::load_from_memory_with_format(&v.jpg, image::ImageFormat::Jpeg).ok()?.into_rgb8().into_raw());
        let mut c = self.cache.lock().unwrap();
        c.0.insert(id, im.clone());
        c.1.push_back(id);
        while c.1.len() > 16 {
            let old = c.1.pop_front().unwrap();
            c.0.remove(&old);
        }
        Some(im)
    }

    fn texture(&self, reg: Reg, cams: &[Cam]) -> (Option<TexMesh>, f64) {
        let c0 = Instant::now();
        let st = &self.regs[&reg];
        let m = texture_region(&st.mesh.0, &st.mesh.1, cams, &|id| self.image(id));
        crate::out::slow(c0);
        (m, c0.elapsed().as_secs_f64())
    }

    /// 결과 쓰기: mesh/t_x_y_z_n.ply + .jpg(두 번 전 텍스처 파일은 지움)
    fn store(&mut self, reg: Reg, m: Option<TexMesh>, secs: f64, out: &mpsc::Sender<Out>) {
        self.n_regions += 1;
        self.secs += secs;
        self.secs_max = self.secs_max.max(secs);
        let views = &self.views;
        let st = self.regs.get_mut(&reg).unwrap();
        let used = m.as_ref().map_or(vec![], |m| m.used.iter().map(|id| (*id, views[id].epoch)).collect());
        st.done = Some((st.ver, used));
        let Some(m) = m else {
            if st.tv > 0 {
                let _ = out.send(Out::Done(reg, st.ver, 0));
            }
            return;
        };
        st.tv += 1;
        let base = format!("mesh/t_{}_{}_{}_", reg.0, reg.1, reg.2);
        if st.tv > 2 {
            for e in ["ply", "jpg"] {
                let _ = std::fs::remove_file(self.dir.join(format!("{base}{}.{e}", st.tv - 2)));
            }
        }
        let (aw, ah, ref px) = m.atlas;
        crate::out::write_owned(&self.dir.join(format!("{base}{}.jpg", st.tv)), jpeg(px, aw, ah, 85));
        crate::out::write_owned(&self.dir.join(format!("{base}{}.ply", st.tv)), ply_tex_bytes(&m));
        let _ = out.send(Out::Done(reg, st.ver, st.tv));
    }

    fn pick(&self) -> Option<Reg> {
        // 붙인 적 없는 구역부터, 그다음 오래전에 바뀐 구역
        self.regs
            .iter()
            .filter(|(_, st)| st.t_mesh.elapsed().as_secs_f64() >= STABLE && self.stale(st))
            .min_by_key(|(_, st)| (st.done.is_some(), st.t_mesh))
            .map(|(r, _)| *r)
    }

    fn run(mut self, ctl: mpsc::Receiver<Ctl>, frames: mpsc::Receiver<FrameIn>, out: mpsc::Sender<Out>) {
        loop {
            let mut finish = false;
            while let Ok(c) = ctl.try_recv() {
                match c {
                    Ctl::Graph(g) => self.graph(g),
                    Ctl::Mesh(reg, ver, m) => {
                        // 다시 뽑았어도 면이 그대로(꼭짓점 1 mm 안, 같은 삼각형)이고 붙일 때 쓴 사진도 움직이지 않았으면 붙인 텍스처를 그대로 씀
                        let reuse = self.regs.get(&reg).is_some_and(|st| st.tv > 0 && !self.stale(st) && same_mesh(&st.mesh, &m));
                        let st = self.regs.entry(reg).or_insert(RegState { ver, mesh: m.clone(), t_mesh: Instant::now(), done: None, tv: 0 });
                        if reuse {
                            let used = st.done.take().map(|d| d.1).unwrap_or_default();
                            (st.ver, st.done) = (ver, Some((ver, used)));
                            self.n_reused += 1;
                            let _ = out.send(Out::Done(reg, ver, st.tv));
                        } else {
                            (st.ver, st.mesh, st.t_mesh) = (ver, m, Instant::now());
                        }
                    }
                    Ctl::Finish => finish = true,
                }
            }
            if finish {
                while let Ok(f) = frames.try_recv() {
                    self.add_frame(f);
                }
                // 끝: 남은 구역 모두(코어 여럿으로)
                use rayon::prelude::*;
                let cams = self.cams();
                let todo: Vec<Reg> = self.regs.iter().filter(|(_, st)| self.stale(st)).map(|(r, _)| *r).collect();
                let c0 = Instant::now();
                let res: Vec<(Reg, Option<TexMesh>, f64)> = todo.par_iter().map(|&r| { let (m, s) = self.texture(r, &cams); (r, m, s) }).collect();
                let wall = c0.elapsed().as_secs_f64();
                let n_final = res.len();
                for (r, m, s) in res {
                    self.store(r, m, s, &out);
                }
                let ntex = self.regs.values().filter(|s| s.tv > 0 && s.done.as_ref().is_some_and(|d| d.0 == s.ver)).count();
                let msg = format!(
                    "텍스처: 보관 사진 {} 장({:.0} MB, 받은 장면 {} 장, 고르기·JPEG {:.1} ms/장), 구역 {} 개 중 {} 개 붙임, 붙이기 {} 번(면이 그대로라 다시 씀 {} 번) 평균 {:.0} ms·최대 {:.0} ms, 끝 마무리 {} 구역 {:.2} 초",
                    self.views.len(), self.bytes as f64 / 1048576.0, self.n_frames, self.secs_frames / self.n_frames.max(1) as f64 * 1e3,
                    self.regs.len(), ntex, self.n_regions, self.n_reused, self.secs / self.n_regions.max(1) as f64 * 1e3, self.secs_max * 1e3, n_final, wall
                );
                let _ = out.send(Out::Finished(msg));
                return;
            }
            let mut got = false;
            while let Ok(f) = frames.try_recv() {
                self.add_frame(f);
                got = true;
            }
            match self.pick() {
                Some(reg) => {
                    let cams = self.cams();
                    let (m, s) = self.texture(reg, &cams);
                    self.store(reg, m, s, &out);
                    std::thread::sleep(Duration::from_secs_f64(s.min(0.5))); // 코어 하나의 절반만
                }
                None if !got => std::thread::sleep(Duration::from_millis(30)),
                None => {}
            }
        }
    }
}

/// 본 스레드 쪽 손잡이: 작업 스레드로 장면·그래프·구역 면을 넘기고, 붙인 결과(구역 → (면 판 번호, 텍스처 번호))를 받는다
pub struct Texturer {
    ctl: mpsc::Sender<Ctl>,
    frames: mpsc::SyncSender<FrameIn>,
    out: mpsc::Receiver<Out>,
    th: Option<std::thread::JoinHandle<()>>,
    pub tex: HashMap<Reg, (u32, u32)>,
}

impl Texturer {
    pub fn start(dir: &Path, cap_mb: usize) -> Texturer {
        let (ctl, crx) = mpsc::channel();
        let (frames, frx) = mpsc::sync_channel(8);
        let (otx, out) = mpsc::channel();
        let w = Worker {
            dir: dir.to_path_buf(), cap: cap_mb << 20, nodes: HashMap::new(), fix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], z0: 0.0,
            views: BTreeMap::new(), next_id: 1, bytes: 0, regs: BTreeMap::new(), cache: Mutex::new((HashMap::new(), VecDeque::new())),
            n_frames: 0, n_regions: 0, secs: 0.0, secs_max: 0.0, secs_frames: 0.0, n_reused: 0,
        };
        let th = std::thread::Builder::new().name("texture".into()).spawn(move || { crate::out::thread_role(crate::out::Role::Texture); w.run(crx, frx, otx) }).unwrap();
        Texturer { ctl, frames, out, th: Some(th), tex: HashMap::new() }
    }
    /// 장면 넘기기(작업 스레드가 밀려 있으면 버림)
    pub fn frame(&self, f: FrameIn) {
        let _ = self.frames.try_send(f);
    }
    pub fn graph(&self, g: &[(u32, M3, V3)]) {
        let _ = self.ctl.send(Ctl::Graph(g.to_vec()));
    }
    pub fn mesh(&self, reg: Reg, ver: u32, verts: Verts, tris: Tris) {
        let _ = self.ctl.send(Ctl::Mesh(reg, ver, Arc::new((verts, tris))));
    }
    fn apply(&mut self, o: Out) -> Option<String> {
        match o {
            Out::Done(reg, ver, 0) => {
                let _ = ver;
                self.tex.remove(&reg);
                None
            }
            Out::Done(reg, ver, tv) => {
                self.tex.insert(reg, (ver, tv));
                None
            }
            Out::Finished(s) => Some(s),
        }
    }
    /// 끝난 결과 받기
    pub fn poll(&mut self) {
        while let Ok(o) = self.out.try_recv() {
            self.apply(o);
        }
    }
    /// 남은 구역을 모두 붙이고 작업 스레드를 닫는다
    pub fn finish(&mut self) {
        let _ = self.ctl.send(Ctl::Finish);
        while let Ok(o) = self.out.recv() {
            if let Some(s) = self.apply(o) {
                eprintln!("{s}");
                break;
            }
        }
        if let Some(th) = self.th.take() {
            let _ = th.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_no_overlap() {
        let sizes: Vec<(usize, usize)> = (0..40).map(|i| (5 + (i * 37) % 60, 4 + (i * 53) % 45)).collect();
        let (h, pos) = pack(&sizes, 128);
        for i in 0..sizes.len() {
            assert!(pos[i].0 + sizes[i].0 <= 128 && pos[i].1 + sizes[i].1 <= h);
            for j in 0..i {
                let sep = pos[i].0 + sizes[i].0 <= pos[j].0 || pos[j].0 + sizes[j].0 <= pos[i].0 || pos[i].1 + sizes[i].1 <= pos[j].1 || pos[j].1 + sizes[j].1 <= pos[i].1;
                assert!(sep, "{i} {j} 겹침");
            }
        }
        let area: usize = sizes.iter().map(|s| s.0 * s.1).sum();
        assert!(h * 128 < area * 2, "빈 자리가 너무 많음: {h}");
    }

    #[test]
    fn icm_smooths_isolated_label() {
        // 띠 모양 삼각형 5 개, 가운데만 장면 1 이 조금 높음 → 이웃과 같은 장면 0 으로
        let tris: Tris = (0..5u32).map(|i| if i % 2 == 0 { [i, i + 1, i + 2] } else { [i + 1, i, i + 2] }).collect();
        let nb = neighbours(&tris);
        let mut best = vec![[(1.0f32, 0u16), (0.9, 1), (0.0, 0), (0.0, 0), (0.0, 0), (0.0, 0)]; 5];
        best[2] = [(1.0, 1), (0.9, 0), (0.0, 0), (0.0, 0), (0.0, 0), (0.0, 0)];
        let lab = label(&best, &nb);
        assert!(lab.iter().all(|&l| l == 0), "{lab:?}");
        // 이웃 장면이 이 삼각형을 못 보면 그대로
        best[2] = [(1.0, 1), (0.0, 0), (0.0, 0), (0.0, 0), (0.0, 0), (0.0, 0)];
        assert_eq!(label(&best, &nb)[2], 1);
    }

    /// 바닥(z = 0) 0.5 m × 0.5 m 격자, 위에서 보는 카메라
    fn floor() -> (Verts, Tris) {
        let n = 21;
        let verts: Verts = (0..n * n).map(|i| ([(i % n) as f32 * 0.025, (i / n) as f32 * 0.025, 0.0], [100, 100, 100])).collect();
        let mut tris = vec![];
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let i = (y * n + x) as u32;
                tris.push([i, i + 1, i + n as u32 + 1]);
                tris.push([i, i + n as u32 + 1, i + n as u32]);
            }
        }
        (verts, tris)
    }

    fn down_cam(id: u32, p: V3, zfloor: f64, sharp: f32) -> Cam {
        // 아래(−z)를 보는 카메라: 카메라 z → 세계 −z, 카메라 x → 세계 x, 카메라 y → 세계 −y
        let (w, h) = (320usize, 240usize);
        let k = [300.0, 300.0, 160.0, 120.0];
        let dq = vec![((p[2] - zfloor) * 1000.0) as u16; w.div_ceil(Q) * h.div_ceil(Q)];
        Cam { id, r: [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], p, k, w, h, dq: Arc::new(dq), sharp }
    }

    #[test]
    fn view_selection_prefers_near_sharp_visible() {
        let (verts, tris) = floor();
        // 장면 1: 가까움, 장면 2: 멂, 장면 3: 가깝지만 흐림, 장면 4: 가깝지만 깊이가 안 맞음(가려짐)
        let mut occl = down_cam(4, [0.25, 0.25, 0.9], 0.0, 100.0);
        occl.dq = Arc::new(vec![300; occl.dq.len()]);
        let cams = vec![down_cam(1, [0.25, 0.25, 1.0], 0.0, 100.0), down_cam(2, [0.25, 0.25, 2.5], 0.0, 100.0), down_cam(3, [0.25, 0.25, 1.0], 0.0, 5.0), occl];
        let (best, cand) = score_views(&verts, &tris, &cams);
        let lab = label(&best, &neighbours(&tris));
        assert!(lab.iter().all(|&l| l != u16::MAX && cams[cand[l as usize]].id == 1), "{:?}", lab.iter().map(|&l| cand.get(l as usize).map(|&i| cams[i].id)).collect::<Vec<_>>());
        // 바닥 아래에서 보는 카메라는 고르지 않음(면 뒤)
        let mut under = down_cam(5, [0.25, 0.25, -1.0], 0.0, 100.0);
        under.r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        under.dq = Arc::new(vec![1000; under.dq.len()]);
        let (best, _) = score_views(&verts, &tris, &[under]);
        assert!(best.iter().all(|b| b[0].0 == 0.0));
    }

    #[test]
    fn uv_maps_back_to_image() {
        // 텍스처 붙인 꼭짓점의 UV 로 아틀라스를 읽으면 그 점을 사진에서 본 색과 같아야 함
        let (verts, tris) = floor();
        let cam = down_cam(1, [0.25, 0.25, 1.0], 0.0, 100.0);
        let (w, h) = (cam.w, cam.h);
        let img: Vec<u8> = (0..w * h).flat_map(|i| { let (x, y) = (i % w, i / w); [(x * 255 / w) as u8, (y * 255 / h) as u8, 100] }).collect();
        let img = Arc::new(img);
        // 표면 색 = 사진 색(밝기 배율 1 이 되게)
        let mut verts = verts;
        for v in verts.iter_mut() {
            let (u, vv, _) = cam.proj(&p64(&v.0)).unwrap();
            let c = bilinear(&img, w, h, u, vv);
            v.1 = c.map(|x| x.round() as u8);
        }
        let m = texture_region(&verts, &tris, std::slice::from_ref(&cam), &|_| Some(img.clone())).unwrap();
        assert_eq!(m.n_tex, tris.len());
        let (aw, ah, ref at) = m.atlas;
        let mut worst = 0f32;
        for (p, _, uv, col) in &m.verts {
            assert!(col.iter().all(|&c| (120..=136).contains(&c)), "노출 배율 {col:?}");
            let (u, v, _) = cam.proj(&p64(p)).unwrap();
            let want = bilinear(&img, w, h, u, v);
            let got = bilinear(at, aw, ah, uv[0] as f64 * aw as f64 - 0.5, (1.0 - uv[1] as f64) * ah as f64 - 0.5);
            for q in 0..2 {
                worst = worst.max((want[q] - got[q]).abs());
            }
        }
        assert!(worst < 3.0, "UV 어긋남: 색 차 {worst}");
    }

    #[test]
    fn sharpness_orders_blur() {
        let (w, h) = (64, 48);
        let sharp: Vec<u8> = (0..w * h).flat_map(|i| { let v = if (i % w / 4 + i / w / 4) % 2 == 0 { 230 } else { 20 }; [v, v, v] }).collect();
        let mut blur = sharp.clone();
        for _ in 0..3 {
            let src = blur.clone();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    for q in 0..3 {
                        let s: u32 = [(0i32, 0i32), (1, 0), (-1, 0), (0, 1), (0, -1)].iter().map(|(dx, dy)| src[(((y as i32 + dy) as usize) * w + (x as i32 + dx) as usize) * 3 + q] as u32).sum();
                        blur[(y * w + x) * 3 + q] = (s / 5) as u8;
                    }
                }
            }
        }
        assert!(sharpness(&sharp, w, h) > 2.0 * sharpness(&blur, w, h));
    }
}
