//! 포터 LOOP 비트 대조: 저장소 밖 하네스(~/opt/orb3_loop_bg)가 원본 C++ 로 덤프한 입력 · 결과를 읽어 같은지 센다.
//! ORBLOOP_DUMPS="dir1:dir2" (bow.bin 등), ORBLOOP_VOC=ORBvoc.txt. 없으면 건너뜀.
use slam::orb3loop::dbow2::*;
use std::sync::OnceLock;

struct R<'a>(&'a [u8], usize);
impl R<'_> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.0[self.1..self.1 + 4].try_into().unwrap());
        self.1 += 4;
        v
    }
    fn i32(&mut self) -> i32 {
        self.u32() as i32
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.0[self.1..self.1 + 8].try_into().unwrap());
        self.1 += 8;
        v
    }
    fn f64(&mut self) -> f64 {
        let v = f64::from_le_bytes(self.0[self.1..self.1 + 8].try_into().unwrap());
        self.1 += 8;
        v
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn desc(&mut self) -> Desc {
        let d = self.0[self.1..self.1 + 32].try_into().unwrap();
        self.1 += 32;
        d
    }
    fn done(&self) -> bool {
        self.1 >= self.0.len()
    }
}

fn dumps() -> Vec<String> {
    std::env::var("ORBLOOP_DUMPS").map(|s| s.split(':').map(String::from).collect()).unwrap_or_default()
}

fn voc() -> Option<&'static Vocabulary> {
    static V: OnceLock<Option<Vocabulary>> = OnceLock::new();
    V.get_or_init(|| {
        let p = std::env::var("ORBLOOP_VOC").ok()?;
        let t = std::time::Instant::now();
        let v = Vocabulary::load(&p).unwrap();
        eprintln!("어휘 {} 노드 {} 단어, {:.2} s", v.parent.len(), v.size(), t.elapsed().as_secs_f64());
        Some(v)
    })
    .as_ref()
}

#[test]
fn vocab_text_eq_original_and_bin() {
    let Some(p) = std::env::var("ORBLOOP_VOC").ok() else { return };
    let t = std::time::Instant::now();
    let v = Vocabulary::load_text(&p).unwrap();
    let tt = t.elapsed().as_secs_f64();
    let bin = std::env::temp_dir().join("orbloop_voc_test.rsbin");
    v.save_bin(bin.to_str().unwrap()).unwrap();
    let t = std::time::Instant::now();
    let vb = Vocabulary::load_bin(bin.to_str().unwrap()).unwrap();
    let tb = t.elapsed().as_secs_f64();
    let _ = std::fs::remove_file(&bin);
    eprintln!("텍스트 로딩 {tt:.2} s, 이진 {tb:.3} s");
    assert_eq!(v.parent, vb.parent);
    assert_eq!(v.desc, vb.desc);
    assert!(v.weight.iter().zip(&vb.weight).all(|(a, b)| a.to_bits() == b.to_bits()));
    // 원본이 읽어 들인 노드 표(voc.bin) 와 비교
    for d in dumps() {
        let Ok(b) = std::fs::read(format!("{d}/voc.bin")) else { continue };
        let mut r = R(&b, 0);
        let (k, l, sc, wt, n) = (r.u32(), r.u32(), r.u32(), r.u32(), r.u32() as usize);
        assert_eq!((k, l, sc, wt, n), (v.k, v.l, v.scoring, v.weighting, v.parent.len()));
        let mut bad = 0usize;
        for i in 0..n {
            let (p, leaf, w, wt) = (r.u32(), r.0[r.1] != 0, { r.1 += 1; r.u32() }, r.f64());
            let d = r.desc();
            let nc = r.u32() as usize;
            let ch: Vec<u32> = (0..nc).map(|_| r.u32()).collect();
            let a = v.child_off[i] as usize;
            let mine = &v.child[a..v.child_off[i + 1] as usize];
            let ok = (i == 0 || p == v.parent[i])
                && leaf == v.leaf[i]
                && (!leaf || w == v.word_id[i])
                && wt.to_bits() == v.weight[i].to_bits()
                && d == v.desc[i]
                && ch == mine;
            if !ok {
                bad += 1;
            }
        }
        let nw = r.u32() as usize;
        let words: Vec<u32> = (0..nw).map(|_| r.u32()).collect();
        assert_eq!(words, v.words);
        eprintln!("어휘 노드 {n} 개 중 다른 것 {bad}");
        assert_eq!(bad, 0);
    }
}

#[test]
fn bow_transform_score_eq_original() {
    let Some(v) = voc() else { return };
    for d in dumps() {
        let Ok(b) = std::fs::read(format!("{d}/bow.bin")) else { continue };
        let mut r = R(&b, 0);
        let n = r.u32() as usize;
        let (mut bow_ok, mut fv_ok, mut feats) = (0, 0, 0usize);
        let mut bows = vec![];
        for _ in 0..n {
            let nf = r.u32() as usize;
            let ds: Vec<Desc> = (0..nf).map(|_| r.desc()).collect();
            feats += nf;
            let nb = r.u32() as usize;
            let bv: BowVector = (0..nb).map(|_| (r.u32(), r.f64())).collect();
            let nfv = r.u32() as usize;
            let fv: FeatureVector = (0..nfv)
                .map(|_| {
                    let id = r.u32();
                    let c = r.u32() as usize;
                    (id, (0..c).map(|_| r.u32()).collect())
                })
                .collect();
            let (mb, mf) = v.transform(&ds, 4);
            if mb.len() == bv.len() && mb.iter().zip(&bv).all(|(a, b)| a.0 == b.0 && a.1.to_bits() == b.1.to_bits()) {
                bow_ok += 1;
            }
            if mf == fv {
                fv_ok += 1;
            }
            bows.push(mb);
        }
        let mut sc_ok = 0usize;
        for i in 0..n {
            for j in 0..n {
                if v.score(&bows[i], &bows[j]).to_bits() == r.f64().to_bits() {
                    sc_ok += 1;
                }
            }
        }
        eprintln!("{d}: 장면 {n} (특징 {feats}) BowVector 같음 {bow_ok}/{n}, FeatureVector {fv_ok}/{n}, 점수 {sc_ok}/{}", n * n);
        assert_eq!((bow_ok, fv_ok, sc_ok), (n, n, n * n));
        assert!(r.done());
    }
}

#[allow(dead_code)]
fn _unused(r: &mut R) -> (i32, f32) {
    (r.i32(), r.f32())
}

mod kfdb_replay {
    use super::*;
    use slam::orb3loop::kfdb::*;

    #[derive(Default)]
    struct G {
        bows: Vec<BowVector>,
        kf_bow: Vec<usize>,
        map: Vec<u32>,
        bad: Vec<bool>,
        map_bad: [bool; 3],
        ord: Vec<Vec<u32>>,
    }
    impl KfGraph for G {
        fn bow(&self, k: KfId) -> &BowVector {
            &self.bows[self.kf_bow[k as usize]]
        }
        fn map_of(&self, k: KfId) -> MapId {
            self.map[k as usize]
        }
        fn is_bad(&self, k: KfId) -> bool {
            self.bad[k as usize]
        }
        fn map_is_bad(&self, m: MapId) -> bool {
            self.map_bad[m as usize]
        }
        fn is_connected(&self, k: KfId, o: KfId) -> bool {
            self.ord[k as usize].contains(&o)
        }
        fn best_covis(&self, k: KfId, n: usize) -> &[KfId] {
            let o = &self.ord[k as usize];
            &o[..o.len().min(n)]
        }
    }

    fn load_bows(b: &[u8]) -> Vec<BowVector> {
        let mut r = R(b, 0);
        let n = r.u32() as usize;
        (0..n)
            .map(|_| {
                let nf = r.u32() as usize;
                r.1 += 32 * nf;
                let nb = r.u32() as usize;
                let bv: BowVector = (0..nb).map(|_| (r.u32(), r.f64())).collect();
                let nfv = r.u32() as usize;
                for _ in 0..nfv {
                    r.u32();
                    let c = r.u32() as usize;
                    r.1 += 4 * c;
                }
                bv
            })
            .collect()
    }

    #[test]
    fn kfdb_eq_original() {
        if voc().is_none() {
            return;
        }
        let nwords = voc().unwrap().size();
        for d in dumps() {
            let (Ok(bb), Ok(sb)) = (std::fs::read(format!("{d}/bow.bin")), std::fs::read(format!("{d}/kfdb.bin"))) else { continue };
            let mut g = G { bows: load_bows(&bb), ..Default::default() };
            let mut db = KeyFrameDatabase::with_words(nwords);
            let mut r = R(&sb, 0);
            let (mut nq, mut nok, mut rq, mut rok, mut ncand) = (0, 0, 0, 0, 0usize);
            let (mut st_ok, mut st_n) = (0, 0);
            while !r.done() {
                match r.u32() {
                    0 => {
                        let (k, m, bi) = (r.u32() as usize, r.u32(), r.u32() as usize);
                        g.kf_bow.resize(k + 1, 0);
                        g.map.resize(k + 1, 0);
                        g.bad.resize(k + 1, false);
                        g.ord.resize(k + 1, vec![]);
                        g.kf_bow[k] = bi;
                        g.map[k] = m;
                    }
                    1 => {
                        let k = r.u32();
                        let b = g.bow(k).clone();
                        db.add(k, &b);
                    }
                    2 => {
                        let k = r.u32();
                        let b = g.bow(k).clone();
                        db.erase(k, &b);
                    }
                    3 => {
                        let k = r.u32() as usize;
                        g.bad[k] = true;
                    }
                    4 => {
                        let m = r.u32() as usize;
                        g.map_bad[m] = true;
                    }
                    5 => {
                        let m = r.u32();
                        db.clear_map(&g, m);
                    }
                    6 => {
                        let k = r.u32() as usize;
                        let n = r.u32() as usize;
                        g.ord[k] = (0..n).map(|_| r.u32()).collect();
                    }
                    7 => {
                        let (k, nc) = (r.u32(), r.u32() as usize);
                        let l: Vec<u32> = { let n = r.u32() as usize; (0..n).map(|_| r.u32()).collect() };
                        let m: Vec<u32> = { let n = r.u32() as usize; (0..n).map(|_| r.u32()).collect() };
                        let (ml, mm) = db.detect_n_best(&g, k, nc);
                        nq += 1;
                        ncand += l.len() + m.len();
                        if ml == l && mm == m {
                            nok += 1;
                        }
                    }
                    8 => {
                        let (bi, m) = (r.u32() as usize, r.u32());
                        let v: Vec<u32> = { let n = r.u32() as usize; (0..n).map(|_| r.u32()).collect() };
                        let b = g.bows[bi].clone();
                        let mine = db.detect_reloc(&g, &b, m);
                        rq += 1;
                        ncand += v.len();
                        if mine == v {
                            rok += 1;
                        }
                    }
                    9 => {
                        let n = r.u32();
                        for k in 0..n {
                            let (pq, pw, ps, rw, rs) = (r.u32() as u64, r.i32(), r.f32(), r.i32(), r.f32());
                            let p = db.place.get(k as usize).copied().unwrap_or_default();
                            let rr = db.reloc.get(k as usize).copied().unwrap_or_default();
                            st_n += 1;
                            if p.query == pq && p.words == pw && p.score.to_bits() == ps.to_bits() && rr.words == rw && rr.score.to_bits() == rs.to_bits() {
                                st_ok += 1;
                            }
                        }
                    }
                    x => panic!("op {x}"),
                }
            }
            eprintln!("{d}: 루프/합치기 질의 {nok}/{nq}, 재위치 질의 {rok}/{rq} (후보 {ncand} 개), 키프레임 남은 상태 {st_ok}/{st_n}");
            assert_eq!((nok, rok, st_ok), (nq, rq, st_n));
        }
    }
}

mod sim3_replay {
    use super::*;
    use slam::orb3loop::crand::*;
    use slam::orb3loop::sim3::*;

    fn bits_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn glibc_rand_matches() {
        // glibc 2.39 rand() 씨앗 1 첫 값들(원본 수열)
        let mut g = GlibcRand::new(1);
        let v: Vec<i32> = (0..5).map(|_| g.rand()).collect();
        assert_eq!(v, vec![1804289383, 846930886, 1681692777, 1714636915, 1957747793]);
    }

    #[test]
    fn compute_sim3_eq_original() {
        for d in dumps() {
            let Ok(b) = std::fs::read(format!("{d}/sim3c.bin")) else { continue };
            let mut r = R(&b, 0);
            let (mut n, mut ok, mut ok_r, mut ok_s) = (0, 0, 0, 0);
            let mut first_bad = None;
            while !r.done() {
                let p1: [f32; 9] = std::array::from_fn(|_| r.f32());
                let p2: [f32; 9] = std::array::from_fn(|_| r.f32());
                let fs = r.i32() != 0;
                let rr: [f32; 9] = std::array::from_fn(|_| r.f32());
                let t: [f32; 3] = std::array::from_fn(|_| r.f32());
                let s = r.f32();
                let t12: [f32; 16] = std::array::from_fn(|_| r.f32());
                let t21: [f32; 16] = std::array::from_fn(|_| r.f32());
                let m = compute_sim3(&p1, &p2, fs);
                n += 1;
                if bits_eq(&m.r, &rr) {
                    ok_r += 1;
                }
                if m.s.to_bits() == s.to_bits() {
                    ok_s += 1;
                }
                if bits_eq(&m.r, &rr) && bits_eq(&m.t, &t) && m.s.to_bits() == s.to_bits() && bits_eq(&m.t12, &t12) && bits_eq(&m.t21, &t21) {
                    ok += 1;
                } else if first_bad.is_none() {
                    first_bad = Some((n, rr, m.r, t, m.t, t21, m.t21));
                }
            }
            eprintln!("{d}: ComputeSim3 전부 같음 {ok}/{n} (R {ok_r}, s {ok_s})");
            if let Some(fb) = first_bad {
                eprintln!("첫 차이 #{}: R 원본 {:?}\n  R 우리 {:?}\n  t {:?} / {:?}\n  T21 {:?}\n  T21 {:?}", fb.0, fb.1, fb.2, fb.3, fb.4, fb.5, fb.6);
            }
            assert_eq!(ok, n);
        }
    }

    #[test]
    fn sim3_iterate_eq_original() {
        for d in dumps() {
            let Ok(b) = std::fs::read(format!("{d}/sim3.bin")) else { continue };
            let mut r = R(&b, 0);
            let (mut n, mut ok, mut ok_dec, mut rng_ok, mut rng_n) = (0, 0, 0, 0, 0);
            while !r.done() {
                assert_eq!(r.u32(), 0x53494D33);
                let rc = { let lo = r.u32() as u64; let hi = r.u32() as u64; lo | hi << 32 };
                let nit = r.i32();
                let nn = r.i32() as usize;
                let n1 = r.i32() as usize;
                let (max_its, min_inl, n_iter0, best0, fs) = (r.i32(), r.i32(), r.i32(), r.i32(), r.i32() != 0);
                let c1 = Pinhole { fx: r.f32(), fy: r.f32(), cx: r.f32(), cy: r.f32() };
                let c2 = Pinhole { fx: r.f32(), fy: r.f32(), cx: r.f32(), cy: r.f32() };
                let (mut x1, mut x2, mut p1, mut p2, mut e1, mut e2, mut ix1, mut all) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
                let u64r = |r: &mut R| { let lo = r.u32() as u64; let hi = r.u32() as u64; lo | hi << 32 };
                for _ in 0..nn {
                    x1.push([r.f32(), r.f32(), r.f32()]);
                    x2.push([r.f32(), r.f32(), r.f32()]);
                    p1.push([r.f32(), r.f32()]);
                    p2.push([r.f32(), r.f32()]);
                    e1.push(u64r(&mut r));
                    e2.push(u64r(&mut r));
                    ix1.push(u64r(&mut r) as usize);
                    all.push(u64r(&mut r) as usize);
                }
                let nr = r.i32() as usize;
                let rv: Vec<i32> = (0..nr).map(|_| r.i32()).collect();
                let (no_more, conv, ninl, nit_after, best_after) = (r.i32() != 0, r.i32() != 0, r.i32(), r.i32(), r.i32());
                let tt: [f32; 16] = std::array::from_fn(|_| r.f32());
                let _br: [f32; 9] = std::array::from_fn(|_| r.f32());
                let _bt: [f32; 3] = std::array::from_fn(|_| r.f32());
                let _bs = r.f32();
                let inl: Vec<bool> = (0..n1).map(|_| { let v = r.0[r.1] != 0; r.1 += 1; v }).collect();
                // 난수 생성기: 원본이 앞서 부른 rand 횟수만큼 건너뛰면 같은 값인지(다른 스레드가 사이에 부르면 어긋날 수 있음)
                let mut g = GlibcRand::new(1);
                for _ in 0..rc {
                    g.rand();
                }
                rng_n += 1;
                if (0..nr).all(|i| g.rand() == rv[i]) {
                    rng_ok += 1;
                }
                let mut s = Sim3Solver::from_matches(x1, x2, e1, e2, ix1, n1, c1, c2, fs);
                // 덤프의 투영 좌표가 우리 FromCameraToImage 와 같은지도 같이
                let proj_ok = s.p1im1 == p1 && s.p2im2 == p2;
                s.all_indices = all;
                s.ransac_max_its = max_its;
                s.ransac_min_inliers = min_inl;
                s.n_iterations = n_iter0;
                s.best_inliers_n = best0;
                let mut k = 0;
                let o = s.iterate_with(nit, &mut |lo, hi| {
                    let v = random_int_from(rv[k], lo, hi - lo + 1);
                    k += 1;
                    v
                });
                n += 1;
                let dec = o.no_more == no_more && o.converge == conv && o.n_inliers == ninl && s.n_iterations == nit_after && s.best_inliers_n == best_after && o.inliers == inl && k == nr;
                if dec {
                    ok_dec += 1;
                }
                if dec && proj_ok && (!conv || bits_eq(&o.t, &tt)) {
                    ok += 1;
                }
            }
            eprintln!("{d}: Sim3 iterate 호출 {n}: 결정(인라이어 · 수렴 · 반복 수) 같음 {ok_dec}, 행렬 · 투영까지 비트 같음 {ok}; rand 수열 {rng_ok}/{rng_n}");
            assert_eq!(ok, n);
        }
    }
}

mod eig_stages {
    use super::*;
    use slam::orb3loop::eig4::*;

    fn eq(a: &[f32], b: &[f32]) -> bool {
        a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// 원본 Eigen 내부 단계(Hessenberg · Francis QR · 두 행 떼기 · 고유벡터 역대입)를 같은 입력으로 비교
    #[test]
    fn eigen_stages_eq_original() {
        for d in dumps() {
            let Ok(b) = std::fs::read(format!("{d}/eig.bin")) else { continue };
            let mut r = R(&b, 0);
            let mut cnt = [[0usize; 2]; 5];
            let mut shown = [false; 5];
            while !r.done() {
                let tag = r.u32() as usize;
                let n = r.u32() as usize / 4;
                let v: Vec<f32> = (0..n).map(|_| r.f32()).collect();
                let (ok, dbg) = match tag {
                    1 => {
                        let mut a: [f32; 16] = v[0..16].try_into().unwrap();
                        let h = hessenberg(&mut a);
                        (eq(&a, &v[16..32]) && eq(&h, &v[32..35]), format!("{:?}\n{:?}", &a, &v[16..32]))
                    }
                    2 => {
                        let mut s = Schur { t: v[0..16].try_into().unwrap(), u: v[16..32].try_into().unwrap(), exshift: 0.0 };
                        let (il, im, iu) = (v[67] as usize, v[68] as usize, v[69] as usize);
                        s.francis_step(il, im, iu, [v[64], v[65], v[66]]);
                        (eq(&s.t, &v[32..48]) && eq(&s.u, &v[48..64]), format!("il{il} im{im} iu{iu}\nT {:?}\nT {:?}\nU {:?}\nU {:?}", s.t, &v[32..48], s.u, &v[48..64]))
                    }
                    3 => {
                        let mut s = Schur { t: v[0..16].try_into().unwrap(), u: v[16..32].try_into().unwrap(), exshift: v[65] };
                        s.split_off_two_rows(v[64] as usize);
                        (eq(&s.t, &v[32..48]) && eq(&s.u, &v[48..64]), format!("T {:?}\nT {:?}", s.t, &v[32..48]))
                    }
                    4 => {
                        let mut ev: [f32; 16] = v[0..16].try_into().unwrap();
                        let ere: [f32; 4] = [v[16], v[18], v[20], v[22]];
                        let mut t: [f32; 16] = v[24..40].try_into().unwrap();
                        let norm = eig_norm(&t);
                        do_compute_eigenvectors(&mut ev, &mut t, &ere, norm);
                        (eq(&ev, &v[40..56]), format!("{:?}\n{:?}", ev, &v[40..56]))
                    }
                    _ => panic!(),
                };
                cnt[tag][ok as usize] += 1;
                if !ok && !shown[tag] {
                    shown[tag] = true;
                    eprintln!("단계 {tag} 첫 차이:\n{dbg}");
                }
            }
            eprintln!("{d}: Hessenberg {}/{} · Francis {}/{} · 두 행 {}/{} · 고유벡터 {}/{}",
                cnt[1][1], cnt[1][0] + cnt[1][1], cnt[2][1], cnt[2][0] + cnt[2][1], cnt[3][1], cnt[3][0] + cnt[3][1], cnt[4][1], cnt[4][0] + cnt[4][1]);
        }
    }
}

#[test]
fn eigen_whole_vs_stage4() {
    use slam::orb3loop::eig4::*;
    for d in dumps() {
        let (Ok(b), Ok(c)) = (std::fs::read(format!("{d}/eig.bin")), std::fs::read(format!("{d}/sim3c.bin"))) else { continue };
        let mut r = R(&b, 0);
        let mut t4 = vec![];
        while !r.done() {
            let tag = r.u32();
            let n = r.u32() as usize / 4;
            let v: Vec<f32> = (0..n).map(|_| r.f32()).collect();
            if tag == 4 { t4.push(v); }
        }
        let mut rc = R(&c, 0);
        let (mut k, mut ok_u, mut ok_e, mut ok_v) = (0, 0, 0, 0);
        while !rc.done() {
            let p1: [f32; 9] = std::array::from_fn(|_| rc.f32());
            let p2: [f32; 9] = std::array::from_fn(|_| rc.f32());
            rc.1 += 4 + 36 + 12 + 4 + 128;
            let n = slam::orb3loop::sim3::n_matrix(&p1, &p2);
            let e = eigen_solver4(&n);
            let v = &t4[k];
            let ere = [v[16], v[18], v[20], v[22]];
            if e.u.iter().zip(&v[0..16]).all(|(a, b)| a.to_bits() == b.to_bits()) { ok_u += 1; }
            if e.eval_re.iter().zip(&ere).all(|(a, b)| a.to_bits() == b.to_bits()) { ok_e += 1; }
            if e.evec_raw.iter().zip(&v[40..56]).all(|(a, b)| a.to_bits() == b.to_bits()) { ok_v += 1; }
            k += 1;
        }
        eprintln!("전체 고유 풀이 {k}: U {ok_u}, 고유값 {ok_e}, 정규화 전 벡터 {ok_v}");
    }
}

/// Sim3Solver 생성자: ~/opt/orb3_loop_bg/ctor_hook.so 가 원본 생성자 입출력을 덤프(ctor.bin)
mod sim3_ctor {
    use super::*;
    use slam::orb3loop::sim3::*;

    fn bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn sim3_ctor_eq_original() {
        let mut total = 0;
        for d in dumps() {
            let Ok(b) = std::fs::read(format!("{d}/ctor.bin")) else { continue };
            let mut r = R(&b, 0);
            let (mut n, mut ok, mut skipped) = (0, 0, 0);
            while !r.done() {
                assert_eq!(r.u32(), 0x43544F52);
                let stable = r.i32() != 0;
                let fs = r.i32() != 0;
                let _km = r.i32();
                let mut pose = || -> ([f32; 4], [f32; 3]) { (std::array::from_fn(|_| r.f32()), std::array::from_fn(|_| r.f32())) };
                let (_q1, t1) = pose();
                let (_q2, t2) = pose();
                let r1: M3 = std::array::from_fn(|_| r.f32());
                let r2: M3 = std::array::from_fn(|_| r.f32());
                let nl = r.i32() as usize;
                let s1: Vec<f32> = (0..nl).map(|_| r.f32()).collect();
                let s2: Vec<f32> = (0..nl).map(|_| r.f32()).collect();
                let mut cam = || Pinhole { fx: r.f32(), fy: r.f32(), cx: r.f32(), cy: r.f32() };
                let (c1, c2) = (cam(), cam());
                let m = r.i32() as usize;
                let mut pairs = Vec::with_capacity(m);
                for _ in 0..m {
                    let code = r.i32();
                    let (o1, o2) = (r.i32(), r.i32());
                    let p1: [f32; 3] = std::array::from_fn(|_| r.f32());
                    let p2: [f32; 3] = std::array::from_fn(|_| r.f32());
                    pairs.push(match code {
                        0 => PairIn::Unmatched,
                        1 => PairIn::NoMp1,
                        2 => PairIn::Bad,
                        3 => PairIn::NoIndex,
                        _ => PairIn::Use { pos1: p1, pos2: p2, oct1: o1, oct2: o2 },
                    });
                }
                let nn = r.i32() as usize;
                let mut x1 = vec![];
                let mut x2 = vec![];
                let mut i1 = vec![];
                let mut i2 = vec![];
                let mut e = vec![];
                for _ in 0..nn {
                    x1.extend((0..3).map(|_| r.f32()));
                    x2.extend((0..3).map(|_| r.f32()));
                    i1.extend((0..2).map(|_| r.f32()));
                    i2.extend((0..2).map(|_| r.f32()));
                    e.extend((0..4).map(|_| r.u64()));
                }
                let prob = r.f64();
                let (mi, mx, nit, nb) = (r.i32(), r.i32(), r.i32(), r.i32());
                n += 1;
                if !stable {
                    skipped += 1;
                    continue;
                }
                let s = Sim3Solver::new(&r1, &t1, &r2, &t2, &s1, &s2, &pairs, c1, c2, fs);
                let fl = |v: &[[f32; 3]]| v.iter().flatten().copied().collect::<Vec<f32>>();
                let fl2 = |v: &[[f32; 2]]| v.iter().flatten().copied().collect::<Vec<f32>>();
                let ee: Vec<u64> = (0..s.n()).flat_map(|i| [s.max_err1[i], s.max_err2[i], s.indices1[i] as u64, s.all_indices[i] as u64]).collect();
                let good = s.n() == nn
                    && bits(&fl(&s.x3dc1), &x1)
                    && bits(&fl(&s.x3dc2), &x2)
                    && bits(&fl2(&s.p1im1), &i1)
                    && bits(&fl2(&s.p2im2), &i2)
                    && ee == e
                    && s.ransac_prob == prob
                    && (s.ransac_min_inliers, s.ransac_max_its, s.n_iterations, s.best_inliers_n) == (mi, mx, nit, nb);
                if good {
                    ok += 1;
                } else if n - skipped - ok < 3 {
                    eprintln!("#{n} 다름: n {}/{nn} x1 {} x2 {} im1 {} im2 {} e {} ransac {:?}/{:?}", s.n(),
                        bits(&fl(&s.x3dc1), &x1), bits(&fl(&s.x3dc2), &x2), bits(&fl2(&s.p1im1), &i1), bits(&fl2(&s.p2im2), &i2), ee == e,
                        (s.ransac_min_inliers, s.ransac_max_its, s.n_iterations, s.best_inliers_n), (mi, mx, nit, nb));
                }
            }
            eprintln!("{d}: Sim3Solver 생성자 {ok}/{} (포즈 바뀜 제외 {skipped})", n - skipped);
            assert_eq!(ok, n - skipped);
            total += n - skipped;
        }
        assert!(total > 0, "ctor.bin 덤프가 ORBLOOP_DUMPS 어디에도 없음");
    }
}
