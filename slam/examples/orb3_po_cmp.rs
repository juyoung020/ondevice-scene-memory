//! 원본(기준 빌드) PoseOptimization 덤프(steps_dump 의 ORBSYS_PO)를 Rust `pose_opt` 에 같은 입력으로 넣어 비트 비교.
//! 쓰임: cargo run --release -p slam --example orb3_po_cmp -- 덤프.txt
use slam::orb3rs::pinhole::Pinhole;
use slam::orb3rs::pose_opt::{pose_optimization, PoseCam, PoseObs, Se3f};

fn f(s: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(s, 16).unwrap())
}

fn main() {
    let path = std::env::args().nth(1).expect("덤프 경로");
    let text = std::fs::read_to_string(path).unwrap();
    run(&text, true);
}

fn run(text: &str, verbose: bool) -> usize {
    let mut lines = text.lines();
    let (mut n_call, mut n_bad) = (0usize, 0usize);
    while let Some(l) = lines.next() {
        let w: Vec<&str> = l.split_whitespace().collect();
        if w.first() != Some(&"PO") {
            continue;
        }
        let fid: u64 = w[1].parse().unwrap();
        let n: usize = w[2].parse().unwrap();
        let v: Vec<f32> = w[3..].iter().map(|s| f(s)).collect();
        let pc = PoseCam { cam: Pinhole::new(v[0], v[1], v[2], v[3]), fx: v[4], fy: v[5], cx: v[6], cy: v[7], bf: v[8] };
        let tcw = Se3f { q: [v[9], v[10], v[11], v[12]], t: [v[13], v[14], v[15]] };
        let mut obs = Vec::with_capacity(n);
        for _ in 0..n {
            let o: Vec<f32> = lines.next().unwrap().split_whitespace().skip(1).map(f).collect();
            obs.push(PoseObs { u: o[0], v: o[1], ur: o[2], inv_sigma2: o[3], xw: [o[4], o[5], o[6]] });
        }
        let r: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
        let ret: i32 = r[1].parse().unwrap();
        let pr: Vec<f32> = r[2..9].iter().map(|s| f(s)).collect();
        let outs: Vec<bool> = r.get(9).map(|s| s.bytes().map(|b| b == b'1').collect()).unwrap_or_default();
        let mut out = vec![false; n];
        let (rn, p) = pose_optimization(&tcw, &pc, &obs, &mut out);
        n_call += 1;
        let same_pose = p.q.iter().chain(p.t.iter()).zip(pr.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
        let same = same_pose && (n < 3 || (rn == ret && out == outs));
        if !same {
            n_bad += 1;
            if verbose && n_bad <= 20 {
                let nd = out.iter().zip(&outs).filter(|(a, b)| a != b).count();
                println!(
                    "다름 장면 {fid} 관측 {n} 반환 {rn}/{ret} 이상치차 {nd} 자세 rs {:?} {:?} / cpp {:?} {:?}",
                    p.q, p.t, &pr[..4], &pr[4..]
                );
            }
        }
    }
    if verbose {
        println!("호출 {n_call} 다름 {n_bad}");
    }
    n_bad
}
