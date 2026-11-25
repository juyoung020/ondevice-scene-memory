//! 원본 대조: `slam/cpp/orb3_sim3pg_probe.cc` 덤프(무작위 Sim3 자세 그래프를 원본 g2o 로 푼 것).
//! `ORBPG_DUMP=pg.bin cargo test --release -p slam --lib orb3rs::sim3pg` — 입력이 없으면 실패.
use super::*;

struct R<'a> {
    b: &'a [u8],
    p: usize,
}
impl R<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let v: [u8; N] = self.b[self.p..self.p + N].try_into().unwrap();
        self.p += N;
        v
    }
    fn u8(&mut self) -> u8 {
        self.take::<1>()[0]
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take())
    }
    fn f(&mut self) -> f64 {
        f64::from_le_bytes(self.take())
    }
    fn sim(&mut self) -> Sim3 {
        Sim3 { r: [self.f(), self.f(), self.f(), self.f()], t: [self.f(), self.f(), self.f()], s: self.f() }
    }
}
fn bits(s: &Sim3) -> [u64; 8] {
    [s.r[0], s.r[1], s.r[2], s.r[3], s.t[0], s.t[1], s.t[2], s.s].map(f64::to_bits)
}
fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (*x == 0.0 && *y == 0.0))
}

#[test]
fn original_probe() {
    let path = std::env::var("ORBPG_DUMP").expect("ORBPG_DUMP(원본 g2o 덤프) 가 필요함");
    let b = std::fs::read(&path).unwrap();
    assert!(!b.is_empty());
    let mut r = R { b: &b, p: 0 };
    let names = ["err", "chi2", "jac", "H", "b", "lambda", "x", "trace", "iters", "final"];
    let mut bad = [0usize; 10];
    let mut ng = 0;
    let mut shown = 0;
    let mut hshown = 0;
    while r.p < b.len() {
        assert_eq!(r.u32(), 0x31524750);
        ng += 1;
        let n = r.i32() as usize;
        let mut g = PoseGraph { lambda_init: 1e-16, ..Default::default() };
        let mut idx = std::collections::BTreeMap::new();
        for i in 0..n {
            let (id, fixed, fs) = (r.i32(), r.u8() != 0, r.u8() != 0);
            idx.insert(id, i);
            g.vertices.push(PgVertex { id, fixed, fix_scale: fs, est: r.sim() });
        }
        let ne = r.i32() as usize;
        for _ in 0..ne {
            let (a, c) = (r.i32() as usize, r.i32() as usize);
            g.edges.push(PgEdge { v0: a, v1: c, meas: r.sim() });
        }
        let fl = first_linearization(&g);
        let mut diff = vec![false; 10];
        let chi = r.f();
        diff[1] |= chi.to_bits() != fl.chi2.to_bits();
        let nae = r.i32() as usize;
        let act: Vec<usize> = (0..ne).filter(|&k| !(g.vertices[g.edges[k].v0].fixed && g.vertices[g.edges[k].v1].fixed)).collect();
        assert_eq!(nae, act.len());
        for &k in &act {
            let e: Vec<f64> = (0..7).map(|_| r.f()).collect();
            let _c = r.f();
            diff[0] |= !same(&e, &fl.err[k]);
            let (_fi, _fj) = (r.u8(), r.u8());
            let ji: Vec<f64> = (0..49).map(|_| r.f()).collect();
            let jj: Vec<f64> = (0..49).map(|_| r.f()).collect();
            // 원본 야코비안은 공유 작업 공간이라 덤프 때는 마지막 간선 값 — 비교 안 함
            let _ = (ji, jj);
        }
        let np = r.i32() as usize;
        for c in 0..np {
            let nb = r.i32();
            for _ in 0..nb {
                let row = r.i32() as usize;
                let v: Vec<f64> = (0..49).map(|_| r.f()).collect();
                let ok = fl.h.get(&(row, c)).is_some_and(|h| same(&v, h));
                if !ok && hshown < 3 {
                    hshown += 1;
                    let h = &fl.h[&(row, c)];
                    let pos: Vec<String> = (0..49).filter(|&k| v[k].to_bits() != h[k].to_bits()).map(|k| format!("({},{}) {:e}/{:e}", k % 7, k / 7, v[k], h[k])).collect();
                    eprintln!("    H({row},{c}) {}", pos[..pos.len().min(8)].join(" "));
                }
                diff[3] |= !ok;
            }
        }
        let bb: Vec<f64> = (0..np * 7).map(|_| r.f()).collect();
        diff[4] |= !same(&bb, &fl.b);
        diff[5] |= r.f().to_bits() != fl.lambda.to_bits();
        let ok = r.u8() != 0;
        let x: Vec<f64> = (0..np * 7).map(|_| r.f()).collect();
        diff[6] |= ok != fl.x.is_some() || (ok && !same(&x, fl.x.as_ref().unwrap()));
        let it = r.i32() as usize;
        let nt = r.i32() as usize;
        let out = optimize(&g, 20);
        diff[8] |= it != out.iterations || nt != out.trace.len();
        for t in 0..nt {
            for v in 0..n {
                let s = r.sim();
                if t < out.trace.len() && bits(&s) != bits(&out.trace[t][v]) {
                    diff[7] = true;
                }
            }
        }
        if diff.iter().any(|&d| d) {
            for k in 0..10 {
                bad[k] += diff[k] as usize;
            }
            if shown < 5 {
                shown += 1;
                let which: Vec<&str> = (0..10).filter(|&k| diff[k]).map(|k| names[k]).collect();
                eprintln!("  그래프 {ng}(정점 {n}, 간선 {ne}): {}", which.join(" "));
            }
        }
    }
    eprintln!("== 그래프 {ng} 개, 항목별 다름: {:?}", names.iter().zip(&bad).collect::<Vec<_>>());
    assert!(bad.iter().all(|&x| x == 0), "원본과 다름");
}
