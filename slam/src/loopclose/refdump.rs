//! 원본 덤프(~/slamref/out/<seq>/{main,flann,opt}.txt, 형식은 ~/slamref/README.md)와 한 단계씩 비트 비교.
//! FLANN·최적화·정합은 덤프에 적힌 결과를 그대로 돌려주는 대역으로 바꿔 이 모듈의 논리만 따로 검증한다.
use super::mem::*;
use super::rtf::Tf;
use super::*;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

fn hf(s: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(s, 16).unwrap())
}
fn hd(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}
fn tf(t: &[&str]) -> (Tf, usize) {
    if t[0] == "null" {
        return (Tf::NULL, 1);
    }
    let mut a = [0f32; 12];
    for i in 0..12 {
        a[i] = hf(t[i]);
    }
    (Tf(a), 12)
}
fn mat(t: &[&str]) -> (Mat6, usize) {
    if t[0] == "empty" {
        return ([0.0; 36], 1);
    }
    let mut m = [0.0; 36];
    for i in 0..36 {
        m[i] = hd(t[1 + i]);
    }
    (m, 37)
}

#[derive(Default, Clone)]
struct Step {
    input: NodeInput,
    last: i32,
    loop_id: i32,
    prox: i32,
    mapcorr: Tf,
    post: BTreeMap<i32, u32>,
    lik: BTreeMap<i32, u32>,
    rawlik: BTreeMap<i32, u32>,
    wm: Vec<i32>,
    stm: Vec<i32>,
    sigs: BTreeMap<i32, Vec<(i32, i32)>>,
    links_add: Vec<(i32, i32, i32, Tf, Mat6)>,
    links_del: Vec<(i32, i32, i32)>,
    local: BTreeMap<i32, Tf>,
    regs: Vec<(i32, i32, RegOut)>,
    trigger_before: bool,
}

fn parse_main(path: &str) -> Vec<Step> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut steps: Vec<Step> = Vec::new();
    let mut cur = Step::default();
    let mut pending_regs = Vec::new();
    let mut trig = false;
    let mut last_sig = 0;
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        match t[0] {
            "TRIGGER" => trig = true,
            "STEP" => {
                cur = Step::default();
                cur.input.stamp = hd(t[7]);
                cur.input.image_size = (640, 480);
                cur.trigger_before = trig;
                trig = false;
            }
            "POSE" => cur.input.pose = tf(&t[1..]).0,
            "COV" => cur.input.cov = mat(&t[1..]).0,
            "VEL" => cur.input.velocity = t[2..].iter().map(|s| hf(s)).collect(),
            "k" => {
                let k = Kp { x: hf(t[1]), y: hf(t[2]), size: hf(t[3]), angle: hf(t[4]), response: hf(t[5]), octave: t[6].parse().unwrap(), class_id: t[7].parse().unwrap() };
                cur.input.features.kps.push(k);
                cur.input.features.pts3d.push([hf(t[8]), hf(t[9]), hf(t[10])]);
                let h = t[11];
                let mut d = [0u8; 32];
                for i in 0..32 {
                    d[i] = u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).unwrap();
                }
                cur.input.features.desc.push(d);
            }
            "REG" => {
                let from: i32 = t[1].parse().unwrap();
                let to: i32 = t[2].parse().unwrap();
                let mut i = 6;
                let (_g, n) = tf(&t[i..]);
                i += n + 1; // "out"
                let (o, n) = tf(&t[i..]);
                i += n;
                let inliers = t[i + 1].parse().unwrap();
                let matches = t[i + 3].parse().unwrap();
                i += 8 + 1; // 4 실수 + "cov"
                let (c, _) = mat(&t[i..]);
                pending_regs.push((from, to, RegOut { t: o, cov: c, inliers, matches }));
            }
            "OUT" => {
                cur.last = t[2].parse().unwrap();
                cur.loop_id = t[4].parse().unwrap();
                cur.prox = t[6].parse().unwrap();
                cur.regs = std::mem::take(&mut pending_regs);
            }
            "MAPCORR" => cur.mapcorr = tf(&t[1..]).0,
            "POST" | "LIK" | "RAWLIK" => {
                let mut m = BTreeMap::new();
                let mut i = 2;
                while i + 1 < t.len() {
                    m.insert(t[i].parse().unwrap(), u32::from_str_radix(t[i + 1], 16).unwrap());
                    i += 2;
                }
                match t[0] {
                    "POST" => cur.post = m,
                    "LIK" => cur.lik = m,
                    _ => cur.rawlik = m,
                }
            }
            "WM" => cur.wm = t[2..].iter().map(|s| s.parse().unwrap()).collect(),
            "STM" => cur.stm = t[2..].iter().map(|s| s.parse().unwrap()).collect(),
            "SIG" => {
                last_sig = t[1].parse().unwrap();
                cur.sigs.insert(last_sig, Vec::new());
            }
            "w" => {
                let v = t[1..].iter().map(|p| {
                    let (a, b) = p.split_once(':').unwrap();
                    (a.parse().unwrap(), b.parse().unwrap())
                });
                cur.sigs.get_mut(&last_sig).unwrap().extend(v);
            }
            "LINK+" => {
                let (tt, n) = tf(&t[4..]);
                let (m, _) = mat(&t[4 + n..]);
                cur.links_add.push((t[1].parse().unwrap(), t[2].parse().unwrap(), t[3].parse().unwrap(), tt, m));
            }
            "LINK-" => cur.links_del.push((t[1].parse().unwrap(), t[2].parse().unwrap(), t[3].parse().unwrap())),
            "lp" => {
                cur.local.insert(t[1].parse().unwrap(), tf(&t[2..]).0);
            }
            "END" => steps.push(cur.clone()),
            _ => {}
        }
    }
    steps
}

/// flann.txt 에서 사전 색인(단계 1 의 첫 BUILD 주소)의 호출 결과만 차례로
#[derive(Default)]
struct FlannLog {
    adds: VecDeque<Vec<u32>>,
    knns: VecDeque<(Vec<i32>, Vec<f32>)>,
}

fn parse_flann(path: &str) -> FlannLog {
    use std::io::BufRead;
    let f = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut log = FlannLog::default();
    let mut dict: Option<String> = None;
    let mut mode = 0; // 1 = ADD(사전), 2 = KNN(사전)
    let mut knn = 0usize;
    let mut idx: Vec<i32> = Vec::new();
    let mut dist: Vec<f32> = Vec::new();
    let mut rows_left = 0usize;
    for line in f.lines() {
        let line = line.unwrap();
        let mut it = line.split(' ');
        let tag = it.next().unwrap_or("");
        match tag {
            "BUILD" | "ADD" | "KNN" | "REM" => {
                let t: Vec<&str> = line.split_whitespace().collect();
                if tag == "BUILD" && dict.is_none() {
                    dict = Some(t[2].to_string());
                }
                let mine = dict.as_deref() == Some(t[2]);
                mode = 0;
                if mine && tag == "ADD" {
                    mode = 1;
                }
                if mine && tag == "KNN" {
                    mode = 2;
                    knn = t[10].parse().unwrap();
                    rows_left = t[4].parse().unwrap();
                    idx.clear();
                    dist.clear();
                }
            }
            "idx" if mode == 1 => {
                log.adds.push_back(line.split_whitespace().skip(2).map(|s| s.parse().unwrap()).collect());
                mode = 0;
            }
            "q" if mode == 2 => {
                let r = line.split('|').nth(1).unwrap();
                let mut n = 0;
                for p in r.split_whitespace() {
                    let (a, b) = p.split_once(':').unwrap();
                    idx.push(a.parse().unwrap());
                    dist.push(hf(b));
                    n += 1;
                }
                assert_eq!(n, knn);
                rows_left -= 1;
                if rows_left == 0 {
                    log.knns.push_back((std::mem::take(&mut idx), std::mem::take(&mut dist)));
                    mode = 0;
                }
            }
            _ => {}
        }
    }
    log
}

/// 덤프 결과를 돌려주는 색인 대역
struct ReplayIndex {
    log: Arc<Mutex<FlannLog>>,
    built: bool,
}
impl WordIndex for ReplayIndex {
    fn is_built(&self) -> bool {
        self.built
    }
    fn build(&mut self, _rows: &[[u8; 32]]) {
        self.built = true;
    }
    fn add_points(&mut self, rows: &[[u8; 32]]) -> Vec<u32> {
        let v = self.log.lock().unwrap().adds.pop_front().expect("ADD 덤프 부족");
        assert_eq!(v.len(), rows.len());
        v
    }
    fn remove_point(&mut self, _idx: u32) {}
    fn knn_search(&mut self, q: &[[u8; 32]], k: usize, _checks: i32) -> (Vec<i32>, Vec<f32>) {
        let (i, d) = self.log.lock().unwrap().knns.pop_front().expect("KNN 덤프 부족");
        assert_eq!(i.len(), q.len() * k, "질의 수가 다름");
        (i, d)
    }
    fn release(&mut self) {
        self.built = false;
    }
}

struct OptCall {
    root: i32,
    poses: BTreeMap<i32, Tf>,
    links: Vec<(i32, i32, i32, Tf, Mat6)>,
    out: OptOut,
}

fn parse_opt(path: &str) -> VecDeque<OptCall> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut v = VecDeque::new();
    let mut cur: Option<OptCall> = None;
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t[0] {
            "OPT" => {
                if let Some(c) = cur.take() {
                    v.push_back(c);
                }
                cur = Some(OptCall { root: t[5].parse().unwrap(), poses: BTreeMap::new(), links: Vec::new(), out: OptOut::default() });
            }
            "p" => {
                cur.as_mut().unwrap().poses.insert(t[1].parse().unwrap(), tf(&t[2..]).0);
            }
            "l" => {
                let (tt, n) = tf(&t[4..]);
                let (m, _) = mat(&t[4 + n..]);
                cur.as_mut().unwrap().links.push((t[1].parse().unwrap(), t[2].parse().unwrap(), t[3].parse().unwrap(), tt, m));
            }
            "out" => {
                let c = cur.as_mut().unwrap();
                c.out.error = hd(t[3]);
                c.out.iterations = t[5].parse().unwrap();
            }
            "o" => {
                cur.as_mut().unwrap().out.poses.insert(t[1].parse().unwrap(), tf(&t[2..]).0);
            }
            _ => {}
        }
    }
    if let Some(c) = cur.take() {
        v.push_back(c);
    }
    v
}

/// 최적화 대역: 입력을 덤프와 비트 비교하고 덤프의 결과를 돌려줌
struct ReplayOpt {
    calls: VecDeque<OptCall>,
    mismatch: Arc<Mutex<Vec<String>>>,
    n: usize,
}
impl PoseGraphOptimizer for ReplayOpt {
    fn optimize(&mut self, root: i32, poses: &BTreeMap<i32, Tf>, links: &LinkMap) -> OptOut {
        let c = self.calls.pop_front().expect("OPT 덤프 부족");
        let n = self.n;
        self.n += 1;
        let mut bad = Vec::new();
        if c.root != root {
            bad.push(format!("root {} vs {}", root, c.root));
        }
        if c.poses.len() != poses.len() || c.poses.iter().zip(poses.iter()).any(|(a, b)| a.0 != b.0 || a.1 .0.map(f32::to_bits) != b.1 .0.map(f32::to_bits)) {
            bad.push("poses".into());
        }
        let mine: Vec<_> = links.iter().map(|(_, l)| (l.from, l.to, l.ty as i32, l.t.0.map(f32::to_bits), l.inf.map(f64::to_bits))).collect();
        let theirs: Vec<_> = c.links.iter().map(|l| (l.0, l.1, l.2, l.3 .0.map(f32::to_bits), l.4.map(f64::to_bits))).collect();
        if mine != theirs {
            bad.push(format!("links {} vs {}", mine.len(), theirs.len()));
        }
        if !bad.is_empty() {
            self.mismatch.lock().unwrap().push(format!("OPT {n}: {}", bad.join(", ")));
        }
        c.out
    }
}

/// 정합 대역: 차례대로 덤프의 REG 를 돌려줌(from/to 확인)
struct ReplayReg {
    bad: Arc<Mutex<Option<String>>>,
    q: Arc<Mutex<VecDeque<(i32, i32, RegOut)>>>,
}
impl LoopRegistration for ReplayReg {
    fn compute(&mut self, from: &Signature, to: &Signature, _guess: Tf) -> RegOut {
        let got = self.q.lock().unwrap().pop_front();
        match got {
            Some((f, t, r)) if (f, t) == (from.id, to.id) => r,
            other => {
                *self.bad.lock().unwrap() = Some(format!("정합 호출이 다름: {}->{} vs {:?}", from.id, to.id, other.map(|o| (o.0, o.1))));
                RegOut { t: Tf::NULL, cov: eye6(1.0), inliers: 0, matches: 0 }
            }
        }
    }
}

fn run(seq: &str, real: bool) -> Option<(usize, usize, Vec<String>)> {
    let home = std::env::var("HOME").ok()?;
    let dir = format!("{home}/slamref/out/{seq}");
    if !std::path::Path::new(&format!("{dir}/main.txt")).exists() {
        eprintln!("{dir} 없음: 건너뜀");
        return None;
    }
    let steps = parse_main(&format!("{dir}/main.txt"));
    let flog = Arc::new(Mutex::new(parse_flann(&format!("{dir}/flann.txt"))));
    let calls = parse_opt(&format!("{dir}/opt.txt"));
    let mism = Arc::new(Mutex::new(Vec::new()));
    let regq = Arc::new(Mutex::new(VecDeque::new()));
    let regbad: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let mut params = crate::params::Params::read_ini("/mnt/c/ondevice/docs/rtabmap_pi.ini").unwrap();
    if let Ok(c) = std::env::var("DICT_CHECKS") {
        params.set("Rust/DictChecks", &c);
    }
    let sync = Arc::new(Mutex::new((0u32, 0u64)));
    let index: Box<dyn WordIndex> = if real { Box::new(SyncedIndex { inner: flann::KdTreeIndex::from_params(&params), sync: sync.clone() }) } else { Box::new(ReplayIndex { log: flog.clone(), built: false }) };
    let opt: Box<dyn PoseGraphOptimizer> = if real { Box::new(TimedOpt(GtsamOptimizer(crate::graph::OptimizerParams::from_params(&params)))) } else { Box::new(ReplayOpt { calls, mismatch: mism.clone(), n: 0 }) };
    let rng = parse_rng(&format!("{dir}/flann.txt"));
    let mut slam_ms = 0.0f64;
    let (mut my_loops, mut my_prox) = (0usize, 0usize);
    let mut slam = Slam::new(
        &params,
        index,
        opt,
        Box::new(ReplayReg { bad: regbad.clone(), q: regq.clone() }),
    );
    let mut ok_steps = 0;
    let mut diffs: Vec<String> = Vec::new();
    let mut links: BTreeMap<(i32, i32, i32), (Tf, Mat6)> = BTreeMap::new();
    for (n, st) in steps.iter().enumerate() {
        if st.trigger_before {
            slam.trigger_new_map();
        }
        *regq.lock().unwrap() = st.regs.iter().cloned().collect();
        *sync.lock().unwrap() = rng.get(n).copied().unwrap_or((0, 0));
        let c0 = std::time::Instant::now();
        slam.process(&st.input);
        slam_ms += c0.elapsed().as_secs_f64() * 1e3;
        my_loops += (slam.loop_closure_id() > 0) as usize;
        my_prox += (slam.info.proximity_id > 0) as usize;
        let mut d = Vec::new();
        if let Some(b) = regbad.lock().unwrap().take() {
            d.push(b);
        }
        let bits = |m: &BTreeMap<i32, f32>| m.iter().map(|(k, v)| (*k, v.to_bits())).collect::<BTreeMap<_, _>>();
        if slam.last_location_id() != st.last {
            d.push(format!("last {} vs {}", slam.last_location_id(), st.last));
        }
        if slam.loop_closure_id() != st.loop_id {
            d.push(format!("loop {} vs {}", slam.loop_closure_id(), st.loop_id));
        }
        if slam.info.proximity_id != st.prox {
            d.push(format!("prox {} vs {}", slam.info.proximity_id, st.prox));
        }
        if slam.map_correction().0.map(f32::to_bits) != st.mapcorr.0.map(f32::to_bits) {
            d.push("mapcorr".into());
        }
        if bits(&slam.info.raw_likelihood) != st.rawlik {
            d.push("rawlik".into());
        }
        if bits(&slam.info.likelihood) != st.lik {
            d.push("lik".into());
        }
        if bits(&slam.info.posterior) != st.post {
            let mine = bits(&slam.info.posterior);
            let dif: Vec<String> = mine.iter().filter(|(k, v)| st.post.get(k) != Some(v)).take(6).map(|(k, v)| format!("{k}:{:e}/{:e}", f32::from_bits(*v), f32::from_bits(*st.post.get(k).unwrap_or(&0)))).collect();
            d.push(format!("post n{}/{} {}", mine.len(), st.post.len(), dif.join(" ")));
        }
        if slam.mem.wm.iter().copied().collect::<Vec<_>>() != st.wm {
            d.push("wm".into());
        }
        if slam.mem.stm.iter().copied().collect::<Vec<_>>() != st.stm {
            d.push("stm".into());
        }
        for (id, w) in &st.sigs {
            match slam.mem.sig(*id) {
                Some(s) if s.word_pairs().collect::<Vec<_>>() == *w => {}
                _ => d.push(format!("words {id}")),
            }
        }
        for (f, t, ty, tt, m) in &st.links_add {
            links.insert((*f, *t, *ty), (*tt, *m));
        }
        for k in &st.links_del {
            links.remove(k);
        }
        let mut mine = BTreeMap::new();
        for s in slam.mem.signatures.values() {
            for l in s.links.values() {
                mine.insert((l.from, l.to, l.ty as i32), (l.t.0.map(f32::to_bits), l.inf.map(f64::to_bits)));
            }
        }
        let theirs: BTreeMap<_, _> = links.iter().map(|(k, v)| (*k, (v.0 .0.map(f32::to_bits), v.1.map(f64::to_bits)))).collect();
        if mine != theirs {
            d.push(format!("links {} vs {}", mine.len(), theirs.len()));
        }
        let lp: BTreeMap<i32, [u32; 12]> = slam.local_optimized_tf().iter().map(|(k, v)| (*k, v.0.map(f32::to_bits))).collect();
        let lpr: BTreeMap<i32, [u32; 12]> = st.local.iter().map(|(k, v)| (*k, v.0.map(f32::to_bits))).collect();
        if lp != lpr {
            d.push("local".into());
        }
        if d.is_empty() {
            ok_steps += 1;
        } else {
            diffs.push(format!("단계 {n}: {}", d.join(", ")));
        }
    }
    let gd = check_graphs(&mut slam, &dir);
    diffs.extend(gd);
    diffs.extend(mism.lock().unwrap().iter().cloned());
    {
        let l = flog.lock().unwrap();
        if !real && (!l.knns.is_empty() || !l.adds.is_empty()) {
            diffs.push(format!("FLANN 덤프 남음: knn {} add {}", l.knns.len(), l.adds.len()));
        }
    }
    let loops = steps.iter().filter(|s| s.loop_id > 0).count();
    let proxs = steps.iter().filter(|s| s.prox > 0).count();
    let posts = steps.iter().filter(|s| !s.post.is_empty()).count();
    eprintln!("{seq}: getConnectedGraph {} 번", slam.connected_graph_calls);
    eprintln!("{seq}: Rust 루프 {my_loops} 근접 {my_prox}");
    eprintln!("{seq}: 루프 {loops} 근접 {proxs} 사후 확률 있는 단계 {posts}");
    let orig: f64 = std::fs::read_to_string(format!("{dir}/time.txt")).unwrap_or_default().lines().filter_map(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok()).sum();
    let ap = APPLY_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    let ad = ADD_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    let op = OPT_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    eprintln!("{seq}: 사전 더하기 {ad:.1} ms, 그래프 최적화 {op:.1} ms");
    let kn = KNN_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    let slam_ms = slam_ms - ap;
    eprintln!("{seq}: 그중 사전 knn {kn:.1} ms, 시험용 난수 맞추기 {ap:.1} ms(뺌)");
    eprintln!("{seq}: process 합 Rust {slam_ms:.1} ms(정합은 덤프 대역) / 원본 {orig:.1} ms(정합·특징 포함)");
    Some((ok_steps, steps.len(), diffs))
}

fn check(seq: &str, real: bool) {
    if let Some((ok, n, diffs)) = run(seq, real) {
        eprintln!("{seq}: 단계 {ok}/{n} 비트 일치");
        for d in diffs.iter().take(20) {
            eprintln!("  {d}");
        }
        assert!(diffs.is_empty(), "{seq}: 다른 곳 {}", diffs.len());
    }
}

#[test]
fn replay_desk() {
    check("desk", false);
}

#[test]
fn replay_digiteo2() {
    check("digiteo2", false);
}

/// flann.txt 의 단계 시작마다 random_device·rand() 호출 수
fn parse_rng(path: &str) -> Vec<(u32, u64)> {
    use std::io::BufRead;
    let f = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut v = Vec::new();
    for line in f.lines() {
        let line = line.unwrap();
        if line.starts_with("STEP ") {
            let t: Vec<&str> = line.split_whitespace().collect();
            let rd = t.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            let rn = t.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
            v.push((rd, rn));
        }
    }
    v
}

/// 실제 FLANN 이식판. 단계 시작의 무작위 상태(원본 프로세스 전체 공유)를 덤프 값으로 맞춘 뒤 짓기
struct SyncedIndex {
    inner: flann::KdTreeIndex,
    sync: Arc<Mutex<(u32, u64)>>,
}
impl SyncedIndex {
    fn apply(&mut self) {
        let _t = tick(&APPLY_NS);
        let (rd, rn) = *self.sync.lock().unwrap();
        let mut k = rd;
        self.inner.set_seed_source(move || {
            k = k.wrapping_add(1);
            k.wrapping_mul(2654435761)
        });
        self.inner.rand = flann::GlibcRand::new(1);
        for _ in 0..rn {
            self.inner.rand.next();
        }
    }
}
impl WordIndex for SyncedIndex {
    fn is_built(&self) -> bool {
        self.inner.is_built()
    }
    fn build(&mut self, rows: &[[u8; 32]]) {
        self.apply();
        self.inner.build(rows)
    }
    fn add_points(&mut self, rows: &[[u8; 32]]) -> Vec<u32> {
        self.apply();
        let _t = tick(&ADD_NS);
        self.inner.add_points(rows)
    }
    fn remove_point(&mut self, idx: u32) {
        self.apply();
        self.inner.remove_point(idx)
    }
    fn knn_search(&mut self, q: &[[u8; 32]], k: usize, checks: i32) -> (Vec<i32>, Vec<f32>) {
        let _t = tick(&KNN_NS);
        WordIndex::knn_search(&mut self.inner, q, k, checks)
    }
    fn release(&mut self) {
        self.inner.release()
    }
}

#[test]
fn real_desk() {
    check("desk", true);
}

#[test]
fn real_digiteo2() {
    check("digiteo2", true);
}

#[test]
fn replay_loh() {
    check("loh", false);
}

#[test]
fn real_loh() {
    check("loh", true);
}

/// graph::findNearestNodes 를 격자 위치(같은 거리 흔함)로 부른 원본 결과(~/slamref/out/nn.txt)와 비교
#[test]
fn nearest_nodes_ties() {
    let Ok(home) = std::env::var("HOME") else { return };
    let Ok(text) = std::fs::read_to_string(format!("{home}/slamref/out/nn.txt")) else { return };
    let (mut ok, mut n) = (0, 0);
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        let np: usize = t[1].parse().unwrap();
        let mut i = 2;
        let mut poses = BTreeMap::new();
        for _ in 0..np {
            let id: i32 = t[i].parse().unwrap();
            poses.insert(id, tf(&t[i + 1..]).0);
            i += 13;
        }
        let target = tf(&t[i + 1..]).0;
        i += 13;
        let r = hf(t[i + 1]);
        let a = hf(t[i + 3]);
        let k: usize = t[i + 5].parse().unwrap();
        if r > 0.0 && k > 0 {
            continue; // 반지름+k 함께는 우리 경로에 없음
        }
        let want: Vec<(i32, u32)> = t[i + 8..].iter().map(|p| {
            let (x, y) = p.split_once(':').unwrap();
            (x.parse().unwrap(), u32::from_str_radix(y, 16).unwrap())
        }).collect();
        let got: Vec<(i32, u32)> = find_nearest_nodes(&target, &poses, r, a, k).into_iter().map(|(x, y)| (x, y.to_bits())).collect();
        n += 1;
        if got == want {
            ok += 1;
        } else if n - ok <= 5 {
            eprintln!("다름 r={r} a={a} k={k}: {got:?} vs {want:?}");
        }
    }
    eprintln!("findNearestNodes {ok}/{n} 같음");
    assert_eq!(ok, n);
}

#[test]
fn replay_static() {
    check("static", false);
}

#[test]
fn replay_join() {
    check("join", false);
}

#[test]
fn real_join() {
    check("join", true);
}

#[test]
fn real_static() {
    check("static", true);
}

static APPLY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static KNN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// 구간 시간을 더하는 시계(검증용)
struct Tick<'a>(&'a std::sync::atomic::AtomicU64, std::time::Instant);

fn tick(c: &std::sync::atomic::AtomicU64) -> Tick<'_> {
    Tick(c, std::time::Instant::now())
}
impl Drop for Tick<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(self.1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}
static ADD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
struct TimedOpt(GtsamOptimizer);
impl PoseGraphOptimizer for TimedOpt {
    fn optimize(&mut self, root: i32, poses: &BTreeMap<i32, Tf>, links: &LinkMap) -> OptOut {
        let _t = tick(&OPT_NS);
        self.0.optimize(root, poses, links)
    }
    fn iterations(&self) -> i32 {
        self.0.iterations()
    }
}

/// main.txt 끝의 GRAPH 블록: (optimized, global) → (위치, 링크)
fn parse_graphs(path: &str) -> Vec<((bool, bool), BTreeMap<i32, [u32; 12]>, Vec<(i32, i32, i32, [u32; 12], [u64; 36])>)> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut v: Vec<((bool, bool), BTreeMap<i32, [u32; 12]>, Vec<(i32, i32, i32, [u32; 12], [u64; 36])>)> = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("GRAPH") => v.push(((t[1] == "1", t[2] == "1"), BTreeMap::new(), Vec::new())),
            Some("gp") => {
                v.last_mut().unwrap().1.insert(t[1].parse().unwrap(), tf(&t[2..]).0 .0.map(f32::to_bits));
            }
            Some("gl") => {
                let (tt, n) = tf(&t[4..]);
                let (m, _) = mat(&t[4 + n..]);
                v.last_mut().unwrap().2.push((t[1].parse().unwrap(), t[2].parse().unwrap(), t[3].parse().unwrap(), tt.0.map(f32::to_bits), m.map(f64::to_bits)));
            }
            _ => {}
        }
    }
    v
}

/// 끝난 뒤 get_graph 네 가지를 원본 getGraph 와 비교
fn check_graphs(slam: &mut Slam, dir: &str) -> Vec<String> {
    let mut d = Vec::new();
    for ((opt, glob), poses, links) in parse_graphs(&format!("{dir}/main.txt")) {
        let (p, l) = slam.get_graph(opt, glob);
        let p: BTreeMap<i32, [u32; 12]> = p.iter().map(|(k, v)| (*k, v.0.map(f32::to_bits))).collect();
        let l: Vec<_> = l.iter().map(|(_, x)| (x.from, x.to, x.ty as i32, x.t.0.map(f32::to_bits), x.inf.map(f64::to_bits))).collect();
        if p != poses || l != links {
            d.push(format!("getGraph({opt},{glob}): 위치 {}/{} 링크 {}/{}", p.len(), poses.len(), l.len(), links.len()));
        }
    }
    d
}
