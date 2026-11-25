//! 하네스 덤프(저장소 밖 ~/opt/orb3_ba_bg)와 비트 비교.
//! ORB3BA_DUMP = 현장 LBA 덤프(ds_run), ORB3BA_STAGE = replay 가 한 호출의 첫 반복 중간값을 남긴 파일, ORB3BA_CALL = 그 호출 번호.
use super::edges::{Edge, Kind};
use super::solver::{Sys, NONE};
use super::{Est, Problem, Vertex};
use std::collections::HashMap;

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        s
    }
    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn done(&self) -> bool {
        self.i >= self.b.len()
    }
}

/// 덤프의 정점 목록: (id, fixed, 값)
fn read_vertices(r: &mut Rd) -> Vec<Vertex> {
    let n = r.u32();
    (0..n)
        .map(|_| {
            let k = r.u8();
            let id = r.i32();
            let fixed = r.u8() != 0;
            let _marg = r.u8();
            let est = if k == 0 {
                let q = [r.f64(), r.f64(), r.f64(), r.f64()];
                let t = [r.f64(), r.f64(), r.f64()];
                Est::Pose { q, t }
            } else {
                Est::Point([r.f64(), r.f64(), r.f64()])
            };
            Vertex { id, fixed, est }
        })
        .collect()
}

pub struct Call {
    pub idx: i32,
    pub problem: Problem,
    pub iterations: i32,
    #[allow(dead_code)]
    pub stopped: bool,
    pub out: Vec<Vertex>,
    pub chi2: Vec<(f64, u8)>,
}

pub fn read_calls(path: &str) -> Vec<Call> {
    let b = std::fs::read(path).expect("덤프 읽기");
    let mut r = Rd { b: &b, i: 0 };
    let mut calls = Vec::new();
    while !r.done() {
        let _magic = r.u32();
        let idx = r.i32();
        let lambda_init = r.f64();
        let vertices = read_vertices(&mut r);
        let pos: HashMap<i32, usize> = vertices.iter().enumerate().map(|(i, v)| (v.id, i)).collect();
        let ne = r.u32();
        let mut edges = Vec::new();
        for _ in 0..ne {
            let k = r.u8();
            let a = r.i32();
            let b2 = r.i32();
            let mut meas = [0.0; 3];
            let mut info = [0.0; 9];
            let (kind, d) = match k {
                0 => {
                    meas[0] = r.f64();
                    meas[1] = r.f64();
                    for v in info.iter_mut().take(4) {
                        *v = r.f64();
                    }
                    let d = r.f64();
                    let cam = [r.f32(), r.f32(), r.f32(), r.f32()];
                    (Kind::Mono { cam }, d)
                }
                1 => {
                    for v in meas.iter_mut() {
                        *v = r.f64();
                    }
                    for v in info.iter_mut() {
                        *v = r.f64();
                    }
                    let d = r.f64();
                    let (fx, fy, cx, cy, bf) = (r.f64(), r.f64(), r.f64(), r.f64(), r.f64());
                    (Kind::Stereo { fx, fy, cx, cy, bf }, d)
                }
                3 => {
                    for v in meas.iter_mut() {
                        *v = r.f64();
                    }
                    for v in info.iter_mut() {
                        *v = r.f64();
                    }
                    (Kind::Gravity, r.f64())
                }
                _ => panic!("모르는 간선 {k}"),
            };
            let v1 = if b2 >= 0 { pos[&b2] } else { 0 };
            edges.push(Edge { kind, v0: pos[&a], v1, meas, info, delta: if d >= 0.0 { Some(d) } else { None } });
        }
        let iterations = r.i32();
        let stopped = r.u8() != 0;
        let out = read_vertices(&mut r);
        let n2 = r.u32();
        let chi2 = (0..n2).map(|_| (r.f64(), r.u8())).collect();
        calls.push(Call { idx, problem: Problem { vertices, edges, lambda_init }, iterations, stopped, out, chi2 });
    }
    calls
}

pub fn read_stage(path: &str) -> HashMap<String, Vec<f64>> {
    let b = std::fs::read(path).expect("단계 파일 읽기");
    let mut r = Rd { b: &b, i: 0 };
    let mut m = HashMap::new();
    while !r.done() {
        let l = r.u32() as usize;
        let name = String::from_utf8(r.take(l).to_vec()).unwrap();
        let c = u64::from_le_bytes(r.take(8).try_into().unwrap()) as usize;
        let v = (0..c).map(|_| r.f64()).collect();
        m.insert(name, v);
    }
    m
}

/// 비트 비교: (다른 개수, 처음 다른 자리)
fn cmp(name: &str, a: &[f64], b: &[f64]) -> usize {
    assert_eq!(a.len(), b.len(), "{name} 길이");
    let mut bad = 0;
    for i in 0..a.len() {
        if a[i].to_bits() != b[i].to_bits() && !(a[i] == 0.0 && b[i] == 0.0) {
            if bad < 5 {
                eprintln!("  {name}[{i}] 원본 {:e} 우리 {:e}", a[i], b[i]);
            }
            bad += 1;
        }
    }
    eprintln!("{name}: {} 중 {} 다름", a.len(), bad);
    bad
}


#[test]
fn stage_first_iteration() {
    let (dump, stage, call) = (need("ORB3BA_DUMP"), need("ORB3BA_STAGE"), need("ORB3BA_CALL"));
    let call: i32 = call.parse().unwrap();
    let calls = read_calls(&dump);
    let c = calls.iter().find(|c| c.idx == call).expect("호출 번호");
    let st = read_stage(&stage);
    let p = &c.problem;
    let mut s = Sys::new(p);
    s.compute_errors();
    let mut err = Vec::new();
    let mut chi = Vec::new();
    for k in 0..p.edges.len() {
        if s.active[k] {
            err.extend_from_slice(&s.err[k][..p.edges[k].dim()]);
            chi.push(s.chi2(k));
        }
    }
    let mut bad = cmp("err", &st["err"], &err);
    bad += cmp("chi2", &st["chi2"], &chi);
    bad += cmp("rchi", &st["rchi"], &[s.robust_chi2()]);
    // 간선마다: 야코비안 · 누적 블록
    {
        let mut snap = Vec::new();
        let mut segs: Vec<(usize, usize, &str)> = Vec::new();
        let seg = |snap: &Vec<f64>, k: usize, name: &'static str, segs: &mut Vec<(usize, usize, &str)>| segs.push((snap.len(), k, name));
        for h in s.hpp.iter_mut() {
            *h = [0.0; 36];
        }
        for k in 0..p.edges.len() {
            if !s.active[k] {
                continue;
            }
            let (xi, xj) = s.add_edge(k);
            let e = &p.edges[k];
            let d = e.dim();
            match e.kind {
                Kind::Gravity => {
                    seg(&snap, k, "gJ", &mut segs);
                    snap.extend_from_slice(&xj[..18]);
                    seg(&snap, k, "gHpp", &mut segs);
                    let pi = s.hidx[e.v0];
                    snap.extend_from_slice(&s.hpp[pi]);
                    snap.extend_from_slice(&s.bp[pi]);
                }
                _ => {
                    seg(&snap, k, "xi", &mut segs);
                    snap.extend_from_slice(&xi[..d * 3]);
                    seg(&snap, k, "xj", &mut segs);
                    let pi = s.hidx[e.v1];
                    if pi == NONE {
                        snap.extend(std::iter::repeat_n(0.0, d * 6));
                    } else {
                        snap.extend_from_slice(&xj[..d * 6]);
                    }
                    let li = s.hidx[e.v0];
                    seg(&snap, k, "Hll", &mut segs);
                    snap.extend_from_slice(&s.hll[li]);
                    seg(&snap, k, "bl", &mut segs);
                    snap.extend_from_slice(&s.bl[li]);
                    seg(&snap, k, "Hpp", &mut segs);
                    if pi == NONE {
                        snap.extend(std::iter::repeat_n(0.0, 42));
                        snap.extend(std::iter::repeat_n(0.0, 18));
                    } else {
                        snap.extend_from_slice(&s.hpp[pi]);
                        snap.extend_from_slice(&s.bp[pi]);
                        snap.extend_from_slice(&s.hpl[&(pi, li)]);
                    }
                }
            }
        }
        let o = &st["edgesnap"];
        let mut shown = 0;
        for i in 0..o.len().min(snap.len()) {
            if o[i].to_bits() != snap[i].to_bits() && shown < 12 {
                let g = segs.iter().rev().find(|g| g.0 <= i).unwrap();
                eprintln!("  간선 {} {:?} {}+{}: 원본 {:e} 우리 {:e}", g.1, p.edges[g.1].kind, g.2, i - g.0, o[i], snap[i]);
                shown += 1;
            }
        }
        bad += cmp("edgesnap", &st["edgesnap"], &snap);
    }
    eprintln!("다른 값 합 {bad}");
    assert_eq!(bad, 0);
}

fn need(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("{k} 없음: 비교 입력이 있어야 한다"))
}

/// 첫 반복의 슈어 보수 · 풀이 · 갱신(replay 단계 덤프)
#[test]
fn stage_solve() {
    let (dump, stage, call) = (need("ORB3BA_DUMP"), need("ORB3BA_STAGE"), need("ORB3BA_CALL"));
    let call: i32 = call.parse().unwrap();
    let calls = read_calls(&dump);
    let c = calls.iter().find(|c| c.idx == call).expect("호출 번호");
    let st = read_stage(&stage);
    let p = &c.problem;
    let mut s = Sys::new(p);
    let mut sc = s.schur_pattern();
    s.compute_errors();
    s.build_system();
    let lam = if p.lambda_init > 0.0 { p.lambda_init } else { 1e-5 * s.max_diag() };
    let mut bad = cmp("lambda", &st["lambda"], &[lam]);
    bad += cmp("b", &st["b"], &s.b_vec());
    let x = s.solve(&mut sc, lam).expect("분해 실패");
    // Hschur: 블록 열 순서로 (행, 열, 값 36)
    let mut hs = Vec::new();
    for (&(c, r), v) in &sc.h {
        hs.push(r as f64);
        hs.push(c as f64);
        hs.extend_from_slice(v);
    }
    bad += cmp("Hschur", &st["Hschur"], &hs);
    bad += cmp("bschur", &st["bschur"], &sc.bschur);
    bad += cmp("x", &st["x"], &x);
    s.update(&x);
    let mut est = Vec::new();
    for i in s.act_v.clone() {
        match &s.est[i] {
            Est::Pose { q, t } => {
                est.extend_from_slice(q);
                est.extend_from_slice(t);
            }
            Est::Point(v) => est.extend_from_slice(v),
        }
    }
    bad += cmp("est1", &st["est1"], &est);
    eprintln!("다른 값 합 {bad}");
    assert_eq!(bad, 0);
}

/// 현장 덤프의 모든 호출을 끝까지 풀어 결과 정점 · χ² · 반복 수 비교
#[test]
fn full_calls() {
    let dump = need("ORB3BA_DUMP");
    let calls = read_calls(&dump);
    assert!(!calls.is_empty(), "덤프에 호출 없음");
    let mut nbad = 0;
    for c in &calls {
        let o = super::optimize(&c.problem, 10);
        let pos: HashMap<i32, usize> = c.problem.vertices.iter().enumerate().map(|(i, v)| (v.id, i)).collect();
        let mut vb = 0;
        for v in &c.out {
            let a = &o.est[pos[&v.id]];
            let same = match (a, &v.est) {
                (Est::Pose { q, t }, Est::Pose { q: q2, t: t2 }) => {
                    q.iter().chain(t).zip(q2.iter().chain(t2)).all(|(x, y)| x.to_bits() == y.to_bits())
                }
                (Est::Point(x), Est::Point(y)) => x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits()),
                _ => false,
            };
            if !same {
                vb += 1;
            }
        }
        // 비활성 간선(정점이 모두 고정)의 원본 χ² 는 초기화 안 된 오차라 비교하지 않음
        let act = Sys::new(&c.problem).active;
        let differs = |k: usize, a: &f64, b: &(f64, u8)| act[k] && a.to_bits() != b.0.to_bits();
        let cb = o.chi2.iter().zip(&c.chi2).enumerate().filter(|(k, (a, b))| differs(*k, a, b)).count();
        for (k, (a, b)) in o.chi2.iter().zip(&c.chi2).enumerate() {
            if differs(k, a, b) {
                let e = &c.problem.edges[k];
                eprintln!("  χ² 간선 {k} {:?} 우리 {a:e} 원본 {:e} 고정 {} {}", e.kind, b.0, c.problem.vertices[e.v0].fixed, c.problem.vertices[e.v1].fixed);
            }
        }
        eprintln!(
            "호출 {}: 정점 {} 간선 {} 반복 {}/{} 다른 정점 {} 다른 χ² {}",
            c.idx,
            c.out.len(),
            c.chi2.len(),
            c.iterations,
            o.iterations,
            vb,
            cb
        );
        if vb > 0 || cb > 0 || c.iterations as usize != o.iterations {
            nbad += 1;
        }
    }
    eprintln!("호출 {} 중 다름 {}", calls.len(), nbad);
    assert_eq!(nbad, 0);
}

/// SE3Quat::exp · VertexSE3Expmap::oplus 무작위 3 만 건(하네스 se3ref 덤프)
#[test]
fn se3_exp_oplus() {
    let b = std::fs::read(need("ORB3BA_SE3")).expect("se3 덤프");
    let f: Vec<f64> = b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    assert!(!f.is_empty());
    let (mut be, mut bo) = (0, 0);
    for r in f.chunks_exact(27) {
        let d: [f64; 6] = r[0..6].try_into().unwrap();
        let mut q: [f64; 4] = r[6..10].try_into().unwrap();
        let mut t: [f64; 3] = r[10..13].try_into().unwrap();
        let (eq, et) = super::se3::exp(&d);
        if eq.iter().chain(&et).zip(&r[13..20]).any(|(a, b)| a.to_bits() != b.to_bits()) {
            if be < 3 {
                eprintln!("exp {:?}: {:?} {:?} / {:?}", d, eq, et, &r[13..20]);
            }
            be += 1;
        }
        super::se3::normalize_rot(&mut q); // SE3Quat(q, t) 생성자
        super::se3::oplus_se3(&mut q, &mut t, &d);
        if q.iter().chain(&t).zip(&r[20..27]).any(|(a, b)| a.to_bits() != b.to_bits()) {
            bo += 1;
        }
    }
    eprintln!("{} 건: exp 다름 {be}, oplus 다름 {bo}", f.len() / 27);
    assert_eq!(be + bo, 0);
}

/// g2o::Sim3 exp · log · 곱 · 역 · map 무작위 4 만 건(하네스 sim3ref 덤프)
#[test]
fn sim3_ops() {
    use super::sim3::Sim3;
    let b = std::fs::read(need("ORB3BA_SIM3")).expect("sim3 덤프");
    let f: Vec<f64> = b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    assert!(!f.is_empty());
    let sim = |r: &[f64]| Sim3 { q: r[0..4].try_into().unwrap(), t: r[4..7].try_into().unwrap(), s: r[7] };
    let flat = |s: &Sim3| -> Vec<f64> { s.q.iter().chain(&s.t).copied().chain([s.s]).collect() };
    let mut bad = [0usize; 7];
    let names = ["exp", "log", "log(exp)", "곱", "역", "map", "C·a·b⁻¹ log"];
    let mut shown = 0;
    for r in f.chunks_exact(74) {
        let d: [f64; 7] = r[0..7].try_into().unwrap();
        let a = sim(&r[7..15]);
        let bb = sim(&r[15..23]);
        let p: [f64; 3] = r[23..26].try_into().unwrap();
        let e = Sim3::exp(&d);
        let got: [Vec<f64>; 7] = [
            flat(&e),
            a.log().to_vec(),
            e.log().to_vec(),
            flat(&a.mul(&bb)),
            flat(&a.inverse()),
            a.map(&p).to_vec(),
            a.mul(&e).mul(&bb.inverse()).log().to_vec(),
        ];
        let want: [&[f64]; 7] = [&r[26..34], &r[34..41], &r[41..48], &r[48..56], &r[56..64], &r[64..67], &r[67..74]];
        for k in 0..7 {
            if got[k].iter().zip(want[k]).any(|(x, y)| x.to_bits() != y.to_bits()) {
                if shown < 6 {
                    eprintln!("{} d={:?}\n  우리 {:?}\n  원본 {:?}", names[k], d, got[k], want[k]);
                    shown += 1;
                }
                bad[k] += 1;
            }
        }
    }
    for k in 0..7 {
        eprintln!("{}: 다름 {}", names[k], bad[k]);
    }
    assert_eq!(bad.iter().sum::<usize>(), 0);
}
