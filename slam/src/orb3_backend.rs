//! 위치 백엔드: ORB-SLAM3 RGB-D(컬러 + 컬러에 맞춘 깊이). `backend::open` 이 이것을 연다.
//!
//! 좌표: 보내는 세계 = 기준 키프레임에 못 박은 좌표(처음 10 키프레임을 넘긴 지도의 첫 키프레임). ORB-SLAM3 가 지도를 합치거나
//!   옮겨 붙여 그 지도 좌표가 통째로 바뀌어도, 기준 키프레임의 보낸 위치가 그대로이게 `A`(보내는 세계 ← ORB 지도)를 다시 구한다.
//!   기준 키프레임이 지워지면 이미 보낸 다른 키프레임으로 다시 못 박는다.
//! 중력: 장면마다 IMU 위쪽(1 초 평균 가속도)을 ORB-SLAM3 에 넘겨, 원본 패치가 새 지도를 위쪽 = z 로 시작하고 지역 최적화의 중력 항으로
//!   지도 안 기울기 표류를 막는다. 지도 사이 옮겨 붙이기는 방위 · 이동만. 그래서 보내는 세계 z 가 곧 위(GRAVITY).
//! 노드 = ORB-SLAM3 키프레임(노드 id = 키프레임 id + 1). 장면의 기준 노드 = 그 장면의 기준 키프레임.
//!   기준 지도의 노드는 그대로, 아직 기준 지도와 합쳐지지 않은 지도(빈틈 뒤 새 지도)의 노드와 장면은 맡아 둠(HELD):
//!   받는 쪽이 보관해 두었다가, ORB-SLAM3 가 그 지도를 기준 지도에 옮겨 붙여 노드가 맡아 둠 없이 다시 오면 고친 위치로 지도에 넣는다.
use crate::backend::SlamBackend;
use crate::imu::{CsvReplay, Ring};
use crate::io::FrameSource;
use crate::orb3::{System, OK};
use crate::simd::{m34_vec, M34};
use crate::stitch::{self, Stitch};
use crate::stream::{GraphNode, GRAVITY, HELD, LOST, MAP_CORRECTED, NEW_MAP};
use crate::Frame;
use crate::stream::OutFrame;
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

/// 3×4 행 우선 강체 변환
pub type T34 = [f64; 12];

pub fn mul(a: &T34, b: &T34) -> T34 {
    let mut o = [0.0; 12];
    for i in 0..3 {
        for j in 0..4 {
            o[i * 4 + j] = a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j] + a[i * 4 + 2] * b[8 + j] + if j == 3 { a[i * 4 + 3] } else { 0.0 };
        }
    }
    o
}

pub fn inv(a: &T34) -> T34 {
    let mut o = [0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 4 + j] = a[j * 4 + i];
        }
        o[i * 4 + 3] = -(o[i * 4] * a[3] + o[i * 4 + 1] * a[7] + o[i * 4 + 2] * a[11]);
    }
    o
}

/// 두 위치의 이동(m) · 회전(rad) 차
fn moved(a: &T34, b: &T34) -> (f64, f64) {
    let d = ((a[3] - b[3]).powi(2) + (a[7] - b[7]).powi(2) + (a[11] - b[11]).powi(2)).sqrt();
    let tr: f64 = (0..3).map(|i| (0..3).map(|k| a[i * 4 + k] * b[i * 4 + k]).sum::<f64>()).sum();
    (d, ((tr - 1.0) * 0.5).clamp(-1.0, 1.0).acos())
}

/// TUM 줄: t x y z qx qy qz qw
pub fn tum_line(t: f64, p: &T34) -> String {
    let (m00, m01, m02, m10, m11, m12, m20, m21, m22) = (p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]);
    let tr = m00 + m11 + m22;
    let (x, y, z, w) = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    };
    format!("{t:.6} {:.6} {:.6} {:.6} {x:.6} {y:.6} {z:.6} {w:.6}", p[3], p[7], p[11])
}

/// 기준 지도로 칠 지도의 최소 키프레임 수
const REF_MIN_KF: usize = 10;
/// 노드 다시 보내기: 이만큼(초)마다 움직인 노드를 훑음, 움직임 문턱(m, rad)
const SWEEP_S: f64 = 1.0;
const SWEEP_M: f64 = 0.01;
const SWEEP_RAD: f64 = 0.5 * std::f64::consts::PI / 180.0;

pub struct Orb3 {
    sys: System,
    ring: Option<Arc<Ring>>,
    replay: Option<CsvReplay>,
    /// IMU → 카메라 회전, t_imu = t_cam + shift
    r_ci: M34,
    shift: f64,
    imu: Vec<[f64; 5]>,
    /// 앞 장면의 보낸 좌표 위치 · 그 뒤 자이로 회전 크기 합(rad)
    prev_out: Option<T34>,
    spike_run: u32,
    gyro_ang: f64,
    q: VecDeque<OutFrame>,
    cap: usize,
    block: bool,
    send_every: i64,
    reset_file: String,
    n: i64,
    /// 보낸 노드: 키프레임 id → (보낸 위치, 맡아 둠)
    sent: HashMap<u64, (T34, bool)>,
    /// 기준 키프레임과 그 보낸 위치(못 박음)
    ref_kf: Option<(u64, T34)>,
    /// 보내는 세계 ← ORB 지도(현재 지도가 기준 지도일 때), 맡아 둔 지도는 마지막 값을 그대로 씀
    a: T34,
    trusted: bool,
    /// 현재 지도 키프레임(ORB 좌표)과 그 지도 id, 마지막으로 읽은 때
    kfs: HashMap<u64, T34>,
    kfs_map: u64,
    kfs_t: f64,
    /// 기준 키프레임 카메라 축으로 모은 가속도 합 · 표본 수, 정한 위 방향
    g_dir: Option<[f64; 3]>,
    last_sweep: f64,
    last_map: Option<u64>,
    poses_f: Option<std::io::BufWriter<std::fs::File>>,
    final_path: String,
    /// 지도 잇기(신경망 장소 인식) 스레드, 최근 가속도(카메라 축, 0.5 초), 마지막 장면을 보낸 때
    stitch: Stitch,
    acc: VecDeque<(f64, [f64; 3])>,
    last_node_t: f64,
    /// 키프레임 → 지도 id(모든 지도), 지도별 키프레임 수, 읽은 때
    kfmap: Arc<HashMap<u64, u64>>,
    map_kfs: HashMap<u64, usize>,
    kfmap_t: f64,
    /// 확인을 기다리는 잇기: (into, from, into ← from, 시각)
    pend: Vec<(u64, u64, T34, f64)>,
    n_graft: i64,
    n_gapgraft: i64,
    /// 모든 잇기 짝(약한 짝 포함, 끝난 뒤 지도 통째 맞춤 후보), 끝난 뒤 맞춤용 작은 깊이(장면 5 장마다: 시각, 기준 키프레임, 상대 위치, 160×120 mm)
    hits_all: Vec<stitch::Hit>,
    /// 빈틈 뒤 새 지도: (새 지도, 앞 지도, 앞 지도 ← 새 지도(카메라가 빈틈 동안 거의 안 움직였다고 봄), 시도 수), 앞 장면의 지도 · 위치
    gap_hyp: Option<(u64, u64, T34, u32)>,
    last_pose: Option<(u64, T34)>,
    last_ok_t: f64,
    deps: Vec<(f64, u64, T34, Vec<u16>)>,
    /// IMU 기록(카메라 시각, 카메라 축 가속도, 자이로 크기): 끝난 지도 기울기 바로잡기
    imu_hist: Vec<(f64, [f64; 3], f64)>,
    dk: Option<crate::mapmerge::Intr>,
    /// 위치가 나온 장면: (시각, 기준 키프레임, 키프레임에서 본 장면 위치)
    tracked: Vec<(f64, u64, T34)>,
    settings_path: std::path::PathBuf,
    t_run: Instant,
    n_sent_ok: i64,
    n_held: i64,
    n_lost: i64,
    n_corr: i64,
    n_newmap: i64,
    n_graph: i64,
    n_nodes: i64,
    n_regrav: i64,
    n_switch: i64,
    /// 최근 1 초 가속도 평균 크기, 기울기 튐으로 거른 장면
    acc_g: Option<f64>,
    n_spike: i64,
    n_repin: i64,
    dropped: i64,
    s_track: Vec<f64>,
}

fn summary(v: &[f64]) -> String {
    if v.is_empty() {
        return "-".into();
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    format!("평균 {:.1} · 중앙 {:.1} · p95 {:.1} · 최대 {:.1} ({} 번)", s.iter().sum::<f64>() / s.len() as f64, q(0.5), q(0.95), q(1.0), s.len())
}

/// 단어장: 빌드 때 정한 ORB-SLAM3 폴더의 Vocabulary/ORBvoc.txt(ORB3_VOC 로 바꿈)
fn vocab() -> String {
    std::env::var("ORB3_VOC").unwrap_or_else(|_| concat!(env!("ORB_SLAM3_DIR"), "/Vocabulary/ORBvoc.txt").to_string())
}

impl Orb3 {
    /// 첫 장면(내부값 · 크기 · 깊이 단위)으로 설정을 만들고 시작한다
    fn new(a: &crate::app::Args, first: &Frame, ring: Option<Arc<Ring>>, replay: Option<CsvReplay>, calib: &crate::imu::Calib) -> Result<Orb3, String> {
        let c = &first.cam;
        let yaml = crate::orb3::settings_yaml(c.fx, c.fy, c.cx, c.cy, c.width, c.height, 30.0, 1000.0, first.color_is_rgb);
        let settings_path = std::env::temp_dir().join(format!("ondevice_orb3_{}.yaml", std::process::id()));
        std::fs::write(&settings_path, yaml).map_err(|e| format!("{}: {e}", settings_path.display()))?;
        let t0 = Instant::now();
        let voc = vocab();
        let sys = System::new(&voc, &settings_path.to_string_lossy())?;
        common::info!("ORB-SLAM3 RGB-D 준비 {:.1} 초(단어장 {voc})", t0.elapsed().as_secs_f64());
        let stitch = Stitch::start(&common::models::dir().to_string_lossy())?;
        let poses_f = if a.poses.is_empty() {
            None
        } else {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&a.poses).map_err(|e| format!("{}: {e}", a.poses))?);
            let _ = writeln!(f, "# timestamp x y z qx qy qz qw (컬러 카메라 광학 → 지도, 보낸 그때 위치)");
            Some(f)
        };
        Ok(Orb3 {
            sys,
            ring,
            replay,
            r_ci: calib.r_ci,
            shift: calib.shift,
            imu: Vec::new(),
            prev_out: None,
            spike_run: 0,
            gyro_ang: 0.0,
            q: VecDeque::new(),
            cap: a.queue.max(1),
            block: a.block_mode(),
            send_every: a.send_every.max(1),
            reset_file: a.reset_file.clone(),
            n: 0,
            sent: HashMap::new(),
            ref_kf: None,
            a: crate::stream::IDENTITY,
            trusted: false,
            kfs: HashMap::new(),
            kfs_map: u64::MAX,
            kfs_t: f64::MIN,
            g_dir: None,
            last_sweep: f64::MIN,
            last_map: None,
            poses_f,
            final_path: a.final_poses.clone(),
            stitch,
            acc: VecDeque::new(),
            last_node_t: f64::MIN,
            kfmap: Arc::new(HashMap::new()),
            map_kfs: HashMap::new(),
            kfmap_t: f64::MIN,
            pend: Vec::new(),
            n_graft: 0,
            n_gapgraft: 0,
            hits_all: Vec::new(),
            gap_hyp: None,
            last_pose: None,
            last_ok_t: f64::MIN,
            deps: Vec::new(),
            imu_hist: Vec::new(),
            dk: None,
            tracked: Vec::new(),
            settings_path,
            t_run: Instant::now(),
            n_sent_ok: 0,
            n_held: 0,
            n_lost: 0,
            n_corr: 0,
            n_newmap: 0,
            n_graph: 0,
            n_nodes: 0,
            n_regrav: 0,
            n_switch: 0,
            acc_g: None,
            n_spike: 0,
            n_repin: 0,
            dropped: 0,
            s_track: Vec::new(),
        })
    }

    /// 줄에 넣기: 실시간(기다리지 않음)이면 장면이 cap 을 넘을 때 가장 오래된 장면을 버림(그래프 블록은 남김)
    fn push(&mut self, f: OutFrame) {
        if !f.is_graph() && !f.reset && !self.block && self.q.iter().filter(|x| !x.is_graph() && !x.reset).count() >= self.cap {
            if let Some(i) = self.q.iter().position(|x| !x.is_graph() && !x.reset) {
                if let Some(mut old) = self.q.remove(i) {
                    crate::bufpool::give_rgb(std::mem::take(&mut old.color));
                    crate::bufpool::give_depth_buf(std::mem::take(&mut old.depth));
                    self.dropped += 1;
                }
            }
        }
        self.q.push_back(f);
    }

    /// 키프레임 하나의 ORB 위치(현재 지도면 읽어 둔 것, 아니면 모든 지도에서)
    fn sys_kf_pose(&mut self, k: u64) -> Option<T34> {
        if let Some(p) = self.kfs.get(&k) {
            return Some(*p);
        }
        self.sys.keyframes(true).into_iter().find(|x| x.0 == k).map(|x| x.2)
    }

    /// 현재 지도 키프레임 다시 읽기
    fn read_kfs(&mut self, t: f64) {
        let v = self.sys.keyframes(false);
        self.kfs_map = v.first().map_or(u64::MAX, |x| x.1);
        self.kfs = v.into_iter().map(|(id, _, p)| (id, p)).collect();
        self.kfs_t = t;
    }

    /// 기준 지도인지 정하고 A(보내는 세계 ← ORB 지도)를 다시 구함
    fn update_frame(&mut self) {
        // 기준 키프레임이 지금 지도에 있으면 그것으로, 없으면(지워짐) 이미 믿고 보낸 다른 키프레임으로 다시 못 박음
        if let Some((id, out)) = self.ref_kf {
            if let Some(p) = self.kfs.get(&id) {
                self.a = mul(&out, &inv(p));
                self.trusted = true;
                return;
            }
            let pin = self.kfs.keys().filter(|k| self.sent.get(k).is_some_and(|s| !s.1)).min().copied();
            if let Some(k) = pin {
                let out = self.sent[&k].0;
                common::info!("위치: 기준 키프레임 {id} 이 지도에서 빠져 키프레임 {k} 로 다시 못 박음");
                self.ref_kf = Some((k, out));
                self.n_repin += 1;
                self.a = mul(&out, &inv(&self.kfs[&k]));
                self.trusted = true;
                return;
            }
            self.trusted = false;
            // 기준 지도보다 지금 지도가 커지면(안 합쳐진 채로) 지금 지도를 기준으로: 예전 기준 지도 노드는 맡아 둠으로 돌리고 중력은 새로 모음
            let ref_map = self.kfmap.get(&id).copied().or_else(|| self.sent.iter().filter(|s| !s.1 .1).find_map(|s| self.kfmap.get(s.0).copied()));
            let n_ref = ref_map.and_then(|m| self.map_kfs.get(&m).copied()).unwrap_or(0);
            if self.kfs.len() >= REF_MIN_KF && self.kfs.len() > n_ref {
                let mut nodes = Vec::new();
                for (k, (p, h)) in self.sent.iter_mut() {
                    if !*h && !self.kfs.contains_key(k) {
                        *h = true;
                        nodes.push(GraphNode { id: (*k + 1) as u32, held: 1, t: *p });
                    }
                }
                if !nodes.is_empty() {
                    self.n_graph += 1;
                    self.n_nodes += nodes.len() as i64;
                    self.push(OutFrame { graph: nodes, ..Default::default() });
                }
                common::info!("위치: 지금 지도 {} (키프레임 {} 개)가 기준 지도 {:?} (키프레임 {n_ref} 개)보다 커져 기준을 옮김", self.kfs_map, self.kfs.len(), ref_map);
                self.ref_kf = None;
                self.g_dir = None;
                self.n_switch += 1;
            } else {
                return;
            }
        }
        // 처음: 키프레임이 REF_MIN_KF 개를 넘긴 지도의 첫 키프레임(보낸 위치 = ORB 위치, 중력은 모은 뒤 돌림)
        if self.kfs.len() >= REF_MIN_KF {
            let k = *self.kfs.keys().min().unwrap();
            let p = self.kfs[&k];
            self.ref_kf = Some((k, p));
            self.a = crate::stream::IDENTITY;
            self.trusted = true;
            common::info!("위치: 기준 지도 {} (키프레임 {} 개), 기준 키프레임 {k}", self.kfs_map, self.kfs.len());
        }
    }

    /// 중력: ORB-SLAM3 지도 좌표는 원본 패치로 z 가 IMU 위(새 지도를 IMU 위쪽으로 시작, 지역 최적화 중력 항)라 기준 지도가 정해지면 바로 GRAVITY.
    /// 처음 정할 때 true(노드를 맡아 둠 없이 모두 다시 보냄)
    fn add_gravity(&mut self) -> bool {
        if self.g_dir.is_some() || self.ref_kf.is_none() {
            return false;
        }
        self.g_dir = Some([0.0, 0.0, 1.0]);
        true
    }

    /// 노드 보내기: 현재 지도의 키프레임 중 새 것 · 움직인 것 · 맡아 둠이 바뀐 것(full 이면 모두)
    fn send_nodes(&mut self, full: bool, held: bool) {
        let mut nodes = Vec::new();
        let mut ids: Vec<u64> = self.kfs.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            let p = mul(&self.a, &self.kfs[&id]);
            let new = match self.sent.get(&id) {
                None => true,
                Some((q, h)) => {
                    let (d, a) = moved(q, &p);
                    full || *h != held || d > SWEEP_M || a > SWEEP_RAD
                }
            };
            if new {
                nodes.push(GraphNode { id: (id + 1) as u32, held: held as u32, t: p });
                self.sent.insert(id, (p, held));
            }
        }
        if !nodes.is_empty() {
            self.n_graph += 1;
            self.n_nodes += nodes.len() as i64;
            self.push(OutFrame { graph: nodes, ..Default::default() });
        }
    }

    /// 모든 지도의 키프레임 위치 · 지도
    fn all_kfs(&mut self, t: f64) -> HashMap<u64, (u64, T34)> {
        let v = self.sys.keyframes(true);
        let mut m = HashMap::new();
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for (id, map, p) in v {
            m.insert(id, (map, p));
            *counts.entry(map).or_default() += 1;
        }
        self.kfmap = Arc::new(m.iter().map(|(k, v)| (*k, v.0)).collect());
        self.map_kfs = counts;
        self.kfmap_t = t;
        m
    }

    /// 지도 잇기 결과: 두 번 같은 변환이 나오면 작은 지도를 큰 지도로 옮겨 붙이기 부탁
    fn on_hit(&mut self, h: stitch::Hit, t: f64) {
        let all = self.all_kfs(t);
        let (Some(&(ma, pa)), Some(&(mb, pb))) = (all.get(&h.q_kf), all.get(&h.c_kf)) else { return };
        if ma == mb {
            return;
        }
        // 질의 카메라의 b 지도 위치 = 후보 카메라(b) × (질의 ← 후보)⁻¹ → b ← a = 그것 × (질의 카메라의 a 위치)⁻¹
        let q_a = mul(&pa, &h.q_rel);
        let c_b = mul(&pb, &h.c_rel);
        let s_ba = mul(&mul(&c_b, &inv(&h.t_qc)), &inv(&q_a));
        common::info!("지도 잇기 후보: 지도 {ma} 장면 {:.2} ↔ 지도 {mb} 장면 {:.2} (맞은 점 {}, MegaLoc {:.2})", h.t, h.c_t, h.inliers, h.score);
        // 같은 두 지도의 최근(60 초) 후보 가운데 하나라도 같은 변환이면 받음(사이에 틀린 후보가 끼어도 됨)
        self.pend.retain(|p| t - p.3 < 60.0);
        let mut best = (f64::MAX, f64::MAX);
        for &(pi, pf, ps, _) in &self.pend {
            if (pi, pf) == (mb, ma) || (pi, pf) == (ma, mb) {
                let s = if (pi, pf) == (mb, ma) { ps } else { inv(&ps) };
                let (d, r) = moved(&s, &s_ba);
                if d + r < best.0 + best.1 {
                    best = (d, r);
                }
            }
        }
        let agree = best.0 < 0.2 && best.1 < 4f64.to_radians();
        if best.0 < f64::MAX {
            common::info!("지도 잇기: 가장 가까운 앞 후보와의 차 {:.3} m · {:.2}°", best.0, best.1.to_degrees());
        }
        // 맞은 점이 아주 많으면(60 개 이상) 한 번으로, 아니면 두 번 같은 변환이 나와야
        if !agree && h.inliers < 60 {
            self.pend.push((mb, ma, s_ba, t));
            return;
        }
        if self.try_graft(ma, mb, &s_ba, &all) {
            self.pend.retain(|p| p.0 != ma && p.1 != ma && p.0 != mb && p.1 != mb);
        }
    }

    /// 지도 a 와 b(b ← a 변환 s_ba)를 지도 통째 자유 공간 검사 뒤 옮겨 붙임(작은 쪽을 큰 쪽으로). 옮겨 붙였으면 참
    fn try_graft(&mut self, ma: u64, mb: u64, s_ba: &T34, all: &HashMap<u64, (u64, T34)>) -> bool {
        let s_ba = *s_ba;
        let (na, nb) = (self.map_kfs.get(&ma).copied().unwrap_or(0), self.map_kfs.get(&mb).copied().unwrap_or(0));
        // 지도 통째 자유 공간 검사(장면 한 장 짝은 반복되는 책상 · 모니터 줄에서 두 번 같은 틀린 변환을 낼 수 있음):
        // 작은 지도의 깊이 점을 큰 지도 장면 깊이에 투영해 맞음 비율 ≥ 0.62 · 작은 지도 깊이 6 장 이상일 때만 옮겨 붙이고, 다듬은 변환을 씀
        let (from, into, start) = if na <= nb { (ma, mb, s_ba) } else { (mb, ma, inv(&s_ba)) };
        let Some(k) = self.dk else { return false };
        let mut ff = Vec::new();
        let mut fi = Vec::new();
        for (_, kf, rel, d) in &self.deps {
            if let Some(&(m, p)) = all.get(kf) {
                let fr = crate::mapmerge::Frame { pose: mul(&p, rel), d };
                if m == from {
                    ff.push(fr);
                } else if m == into {
                    fi.push(fr);
                }
            }
        }
        let t0 = Instant::now();
        let chk = crate::mapmerge::check(&ff, &fi, &start, &k);
        let Some((s_ok, r, n_ag)) = chk else { return false };
        common::info!("지도 잇기 검사: 지도 {from}(깊이 {} 장) → 지도 {into}(깊이 {} 장) 맞음 비율 {r:.3} · 맞음 {n_ag}, {:.1} 초", ff.len(), fi.len(), t0.elapsed().as_secs_f64());
        if r < 0.62 || n_ag < 300 || ff.len() < 6 || fi.len() < 6 {
            return false;
        }
        let ok = self.sys.graft(from, into, &s_ok);
        if ok {
            self.n_graft += 1;
            common::info!("위치: 지도 {from} 를 지도 {into} 에 옮겨 붙임(신경망 장소 인식 + 지도 통째 자유 공간 검사)");
        }
        ok
    }

    fn step(&mut self, mut f: Frame, _read_ms: f64) -> bool {
        if common::control::take_reset() || (!self.reset_file.is_empty() && std::fs::remove_file(&self.reset_file).is_ok()) {
            // 뷰어 지도 초기화: 물체 기억 · 점 지도만 비우고, 노드는 다음 훑기에 모두 다시 보냄
            common::info!("위치: 뷰어 지도 초기화 → 리셋 블록, 노드 다시 보냄");
            self.push(OutFrame { reset: true, ..Default::default() });
            self.sent.clear();
        }
        // 이 장면 시각까지의 IMU 표본(가속도만 씀)
        self.imu.clear();
        if let Some(ring) = &self.ring {
            if let Some(r) = &mut self.replay {
                r.feed(ring, f.t + self.shift);
            }
            let imu = &mut self.imu;
            ring.drain(|s| imu.push([s.t, s.a[0] as f64, s.a[1] as f64, s.a[2] as f64, (s.w[0] * s.w[0] + s.w[1] * s.w[1] + s.w[2] * s.w[2]).sqrt() as f64]));
        }
        for s in &self.imu {
            let a = m34_vec(&self.r_ci, &[s[1] as f32, s[2] as f32, s[3] as f32, 0.0]);
            self.acc.push_back((s[0] - self.shift, [a[0] as f64, a[1] as f64, a[2] as f64]));
            self.imu_hist.push((s[0] - self.shift, [a[0] as f64, a[1] as f64, a[2] as f64], s[4]));
        }
        while self.acc.front().is_some_and(|x| f.t - x.0 > 1.0) {
            self.acc.pop_front();
        }
        // 최대 속도 재생(받는 쪽을 기다림)은 잇기 결과도 기다려 실시간처럼 제때 반영
        while let Some(r) = self.stitch.recv(self.block) {
            // 약한 짝(깊이 맞춤 미달)도 옮겨 붙이기 후보로: 받는 것은 지도 통째 자유 공간 검사가 정함
            for h in r {
                self.hits_all.push(h.clone());
                self.on_hit(h, f.t);
            }
        }
        let c0 = Instant::now();
        let (w, h) = (f.cam.width, f.cam.height);
        if self.block {
            self.sys.wait_idle();
        }
        // IMU 위쪽: 최근 1 초 가속도 평균(카메라 축)
        let up = (self.acc.len() >= 20).then(|| {
            let m = self.acc.iter().fold([0.0; 3], |s, x| [s[0] + x.1[0], s[1] + x.1[1], s[2] + x.1[2]]);
            let l = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            m.map(|v| v / l)
        });
        self.acc_g = (self.acc.len() >= 20).then(|| {
            let m = self.acc.iter().fold([0.0; 3], |s, x| [s[0] + x.1[0], s[1] + x.1[1], s[2] + x.1[2]]).map(|v| v / self.acc.len() as f64);
            (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt()
        });
        let tr = self.sys.track(&f.color, &f.depth_mm, w, h, f.t, up);
        self.s_track.push(c0.elapsed().as_secs_f64() * 1e3);
        let changed = self.sys.map_changed();
        let mut flags = 0u32;
        if self.last_map != Some(tr.map_id) {
            if self.last_map.is_some() {
                flags |= NEW_MAP;
                self.n_newmap += 1;
            }
            self.last_map = Some(tr.map_id);
        }
        let ok = tr.state == OK && tr.twc.iter().all(|x| x.is_finite());
        // 빈틈 뒤 새 지도는 앞 지도의 마지막 위치에서 이어진다고 보고 바로 옮겨 붙이기를 시도(새 지도 깊이 6 장부터 1 초마다, 최대 8 번)
        if ok {
            match self.last_pose {
                // 빈틈이 짧을 때(1.5 초 이하)만: 5 초 넘는 빈틈은 카메라가 다른 곳으로 가 있어 반복되는 책상 줄의 엉뚱한 자리에 붙음(37 차 950 빈틈)
                Some((m, p)) if m != tr.map_id && self.gap_hyp.is_none_or(|g| g.0 != tr.map_id) && f.t - self.last_ok_t <= 1.5 => {
                    self.gap_hyp = Some((tr.map_id, m, mul(&p, &inv(&tr.twc)), 0));
                }
                _ => {}
            }
            self.last_pose = Some((tr.map_id, tr.twc));
            self.last_ok_t = f.t;
        }
        if let Some((gn, go, gs, tries)) = self.gap_hyp
            && ok
            && gn == tr.map_id
            && self.tracked.len() % 30 == 0
        {
            let all = self.all_kfs(f.t);
            let alive = |m: u64| all.values().any(|x| x.0 == m);
            if tries >= 8 || !alive(gn) || !alive(go) {
                self.gap_hyp = None;
            } else {
                self.gap_hyp = Some((gn, go, gs, tries + 1));
                if self.try_graft(gn, go, &gs, &all) {
                    self.gap_hyp = None;
                    self.n_gapgraft += 1;
                }
            }
        }
        for s in &self.imu {
            self.gyro_ang += s[4] / 200.0;
        }
        if ok && (changed || tr.map_id != self.kfs_map || f.t - self.kfs_t >= SWEEP_S || tr.ref_kf.is_some_and(|k| !self.kfs.contains_key(&k))) {
            self.read_kfs(f.t);
        }
        if ok {
            self.update_frame();
        }
        // 튐 거르기: 보낼 좌표에서 앞 장면과의 회전이 그사이 자이로 회전 × 1.5 + 3° 보다 크거나 30 cm 넘게 뛰면 그 장면은 놓침
        // (번진 장면에서 최적화가 잘못 수렴 · 지도 좌표가 바뀐 첫 장면)
        let pose_now = mul(&self.a, &tr.twc);
        let spike = ok
            && self.prev_out.is_some_and(|p| {
                let (d, a) = moved(&p, &pose_now);
                a > self.gyro_ang * 1.5 + 3f64.to_radians() || d > 0.3
            });
        // 튀면 마지막 좋은 장면을 그대로 두고 자이로 회전을 이어 더함. 5 장 넘게 이어지면(좌표가 바뀐 것) 지금 장면부터 다시
        if spike {
            self.n_spike += 1;
            self.spike_run += 1;
            if self.spike_run > 5 {
                (self.prev_out, self.gyro_ang, self.spike_run) = (Some(pose_now), 0.0, 0);
            }
        } else {
            self.prev_out = ok.then_some(pose_now);
            (self.gyro_ang, self.spike_run) = (0.0, 0);
        }
        let ok = ok && !spike;
        let mut regrav = false;
        if ok && self.trusted {
            regrav = self.add_gravity();
        }
        let usable = ok && self.trusted && self.g_dir.is_some();
        let mut anchor = 0u32;
        if ok {
            let sweep = changed || regrav || f.t - self.last_sweep >= SWEEP_S || tr.ref_kf.is_some_and(|k| !self.sent.contains_key(&k));
            if sweep {
                self.last_sweep = f.t;
                self.send_nodes(regrav, !usable);
            }
            // 기준 노드: 장면의 기준 키프레임, 아직 지도에 안 들어간 새 키프레임이면(지도 스레드가 처리 전) 지금 지도에서 보낸 가장 새 키프레임
            let k = tr.ref_kf.filter(|k| self.sent.contains_key(k) && self.kfs.contains_key(k)).or_else(|| self.kfs.keys().filter(|k| self.sent.contains_key(k)).max().copied());
            if let Some(k) = k {
                anchor = (k + 1) as u32;
                let rel = mul(&inv(&self.kfs[&k]), &tr.twc);
                self.tracked.push((f.t, k, rel));
                // 끝난 뒤 지도 맞춤용 작은 깊이(5 장마다)
                if self.tracked.len() % 5 == 1 && f.depth_mm.len() == (w * h) as usize && w as usize == crate::mapmerge::SW * 4 && h as usize == crate::mapmerge::SH * 4 {
                    use crate::mapmerge::{SH, SW};
                    let d: Vec<u16> = (0..SH * SW).map(|i| f.depth_mm[(i / SW) * 4 * w as usize + (i % SW) * 4]).collect();
                    self.deps.push((f.t, k, rel, d));
                    self.dk.get_or_insert(crate::mapmerge::Intr { fx: f.cam.fx, fy: f.cam.fy, cx: f.cam.cx, cy: f.cam.cy });
                }
            }
        }
        // 지도 잇기에 장면 보내기(0.25 초마다, 1 초 평균 가속도로 본 IMU 위쪽이 믿을 만할 때)
        if ok && f.t - self.last_node_t >= 0.25 && self.acc.len() >= 20 {
            if f.t - self.kfmap_t >= 1.0 || changed {
                self.all_kfs(f.t);
            }
            let m = self.acc.iter().fold([0.0; 3], |s, x| [s[0] + x.1[0], s[1] + x.1[1], s[2] + x.1[2]]).map(|v| v / self.acc.len() as f64);
            let g = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            if let (Some(k), true) = ((anchor > 0).then(|| (anchor - 1) as u64), (g - 9.81).abs() < 2.0) {
                if !self.kfmap.contains_key(&k) {
                    self.all_kfs(f.t);
                }
                let p = self.kfmap.contains_key(&k).then(|| self.sys_kf_pose(k)).flatten();
                if let Some(pk) = p {
                    self.last_node_t = f.t;
                    let c = &f.cam;
                    self.stitch.send(stitch::Node {
                        seq: self.n as u64,
                        t: f.t,
                        kf: k,
                        rel: mul(&inv(&pk), &tr.twc),
                        color: f.color.clone(),
                        bgr: !f.color_is_rgb,
                        depth: f.depth_mm.clone(),
                        k: crate::lcverify::Intr { fx: c.fx, fy: c.fy, cx: c.cx, cy: c.cy },
                        w: c.width as usize,
                        h: c.height as usize,
                        up: m.map(|v| v / g),
                        maps: self.kfmap.clone(),
                    });
                }
            }
        }
        if changed && usable {
            flags |= MAP_CORRECTED;
            self.n_corr += 1;
        }
        let pose = mul(&self.a, &tr.twc);
        if usable && anchor != 0 {
            flags |= GRAVITY;
            self.n_sent_ok += 1;
            if let Some(pf) = &mut self.poses_f {
                let _ = writeln!(pf, "{}", tum_line(f.t, &pose));
            }
        } else if ok && anchor != 0 {
            // 기준 지도와 아직 안 합쳐진 지도(또는 중력 전): 맡아 둠
            flags |= HELD | LOST;
            self.n_held += 1;
        } else {
            flags |= LOST;
            anchor = 0;
            self.n_lost += 1;
        }
        let seq = self.n as u64;
        self.n += 1;
        if (seq as i64) % self.send_every != 0 {
            return true;
        }
        let depth = f.take_stream_depth();
        let o = OutFrame {
            flags,
            map_id: tr.map_id as u32,
            seq,
            t: f.t,
            pose,
            fx: f.cam.fx,
            fy: f.cam.fy,
            cx: f.cam.cx,
            cy: f.cam.cy,
            depth_per_m: f.depth_per_m,
            width: w,
            height: h,
            color: std::mem::take(&mut f.color),
            color_is_rgb: f.color_is_rgb,
            depth,
            anchor,
            ..Default::default()
        };
        self.push(o);
        true
    }
}

/// 바로잡음을 매끈하게 하는 가우스 폭(초): 장면마다 가속도로 잰 위는 손 흔들림 · 치우침으로 1–2° 씩 튀어, 그대로 쓰면 먼 점이 몇 cm 씩 흩어져 벽 · 바닥이 여러 겹(병렬 A 1 차)
const LEVEL_SIG: f64 = 3.0;
/// 측정값이 매끈한 값과 이만큼(°) 넘게 다르면 그 장면은 측정값(ORB 회전 튐)
const LEVEL_SPIKE: f64 = 4.0;

/// 끝난 지도 기울기 바로잡기: 장면마다 ±0.5 초 가속도 평균(가만히 있을 때: 크기 g ± 0.3, 자이로 평균 < 0.6 rad/s)을 지도 축으로 보낸 '위' 가
/// 지도 z 와 어긋난 만큼 장면 방향을 돌림. 바로잡음은 같은 ORB 지도에서 가만한 장면들 어긋남의 가우스(LEVEL_SIG) 평균(튐만 제 측정값), 없으면 5 초 안 가장 가까운 것.
/// ORB 지도 안 기울기 표류 · 몇 장짜리 회전 튐을 중력 측정으로 누름. fr: (시각, 장면 위치, 원래 ORB 지도). 돌려줌: (돌린 장면 수, 가장 큰 각 °)
pub(crate) fn level(fr: &mut [(f64, T34, u64)], imu: &[(f64, [f64; 3], f64)]) -> (usize, f64) {
    if imu.len() < 2 {
        return (0, 0.0);
    }
    // 누적 합으로 창 평균
    let mut cs = vec![[0.0f64; 4]; imu.len() + 1];
    for (i, x) in imu.iter().enumerate() {
        cs[i + 1] = [cs[i][0] + x.1[0], cs[i][1] + x.1[1], cs[i][2] + x.1[2], cs[i][3] + x.2];
    }
    let rate = (imu.len() - 1) as f64 / (imu[imu.len() - 1].0 - imu[0].0).max(1e-6);
    let idx = |t: f64| imu.partition_point(|x| x.0 < t);
    // 가만한 장면의 바로잡을 회전(가로축 회전 벡터)
    let mut meas: Vec<Option<[f64; 3]>> = vec![None; fr.len()];
    for (i, (t, p, _)) in fr.iter().enumerate() {
        let (a, b) = (idx(t - 0.5), idx(t + 0.5));
        if ((b - a) as f64) < 0.75 * rate {
            continue;
        }
        let n = (b - a) as f64;
        let m = [(cs[b][0] - cs[a][0]) / n, (cs[b][1] - cs[a][1]) / n, (cs[b][2] - cs[a][2]) / n];
        let g = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
        if (g - 9.81).abs() > 0.3 || (cs[b][3] - cs[a][3]) / n > 0.6 {
            continue;
        }
        let u = [(p[0] * m[0] + p[1] * m[1] + p[2] * m[2]) / g, (p[4] * m[0] + p[5] * m[1] + p[6] * m[2]) / g, (p[8] * m[0] + p[9] * m[1] + p[10] * m[2]) / g];
        // u → z 회전: 축 u × z = (u.y, −u.x, 0), 각 acos(u.z)
        let ang = u[2].clamp(-1.0, 1.0).acos();
        let l = (u[0] * u[0] + u[1] * u[1]).sqrt();
        if l > 1e-9 {
            meas[i] = Some([u[1] / l * ang, -u[0] / l * ang, 0.0]);
        }
    }
    let mut n_done = 0;
    let mut max_a: f64 = 0.0;
    let corr: Vec<Option<[f64; 3]>> = (0..fr.len())
        .map(|i| {
            let (t, m) = (fr[i].0, fr[i].2);
            let (mut w, mut r) = (0.0, [0.0; 3]);
            let mut near: Option<(f64, [f64; 3])> = None;
            for (j, x) in fr.iter().enumerate() {
                let Some(v) = meas[j] else { continue };
                if x.2 != m {
                    continue;
                }
                let dt = (x.0 - t).abs();
                if dt <= 3.0 * LEVEL_SIG {
                    let k = (-0.5 * (dt / LEVEL_SIG).powi(2)).exp();
                    w += k;
                    for c in 0..3 {
                        r[c] += k * v[c];
                    }
                }
                if dt <= 5.0 && near.is_none_or(|n| dt < n.0) {
                    near = Some((dt, v));
                }
            }
            if w > 0.0 {
                let sm = r.map(|v| v / w);
                match meas[i] {
                    // 몇 장짜리 회전 튐: 매끈한 값과 크게 다르면 그 장면 측정값
                    Some(v) if ((v[0] - sm[0]).powi(2) + (v[1] - sm[1]).powi(2)).sqrt().to_degrees() > LEVEL_SPIKE => Some(v),
                    _ => Some(sm),
                }
            } else {
                near.map(|n| n.1)
            }
        })
        .collect();
    let mut rots: Vec<[f64; 9]> = Vec::with_capacity(fr.len());
    for (f, c) in fr.iter_mut().zip(corr) {
        let r = c.unwrap_or([0.0; 3]);
        let a = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        if a < 1e-9 {
            rots.push([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
            continue;
        }
        let k = r.map(|v| v / a);
        let (s_, c_) = a.sin_cos();
        let v = 1.0 - c_;
        let rot = [
            c_ + k[0] * k[0] * v,
            k[0] * k[1] * v - k[2] * s_,
            k[0] * k[2] * v + k[1] * s_,
            k[1] * k[0] * v + k[2] * s_,
            c_ + k[1] * k[1] * v,
            k[1] * k[2] * v - k[0] * s_,
            k[2] * k[0] * v - k[1] * s_,
            k[2] * k[1] * v + k[0] * s_,
            c_ + k[2] * k[2] * v,
        ];
        let p = f.1;
        let mut q = p;
        for i in 0..3 {
            for j in 0..3 {
                q[i * 4 + j] = rot[i * 3] * p[j] + rot[i * 3 + 1] * p[4 + j] + rot[i * 3 + 2] * p[8 + j];
            }
        }
        f.1 = q;
        rots.push(rot);
        n_done += 1;
        max_a = max_a.max(a.to_degrees());
    }
    // 위치도 같은 만큼: ORB 지도마다 시간 순으로 앞 장면과의 이동을 그 장면 바로잡음 회전으로 돌려 이어 붙임(지도가 기운 채 쌓인 높이 비탈을 폄,
    // 장면 방향만 돌리면 이웃 장면끼리 깊이가 어긋남). 지도의 첫 장면은 그대로
    let mut by_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, f) in fr.iter().enumerate() {
        by_map.entry(f.2).or_default().push(i);
    }
    for v in by_map.values_mut() {
        v.sort_by(|a, b| fr[*a].0.total_cmp(&fr[*b].0));
        let mut prev_old = [fr[v[0]].1[3], fr[v[0]].1[7], fr[v[0]].1[11]];
        let mut prev_new = prev_old;
        for &i in v.iter().skip(1) {
            let old = [fr[i].1[3], fr[i].1[7], fr[i].1[11]];
            let d = [old[0] - prev_old[0], old[1] - prev_old[1], old[2] - prev_old[2]];
            let r = &rots[i];
            let nw = [
                prev_new[0] + r[0] * d[0] + r[1] * d[1] + r[2] * d[2],
                prev_new[1] + r[3] * d[0] + r[4] * d[1] + r[5] * d[2],
                prev_new[2] + r[6] * d[0] + r[7] * d[1] + r[8] * d[2],
            ];
            fr[i].1[3] = nw[0];
            fr[i].1[7] = nw[1];
            fr[i].1[11] = nw[2];
            (prev_old, prev_new) = (old, nw);
        }
    }
    (n_done, max_a)
}

/// 같은 프로세스 처리 루프용: 입력을 열고, 첫 장면을 읽어 ORB-SLAM3 를 만든다(첫 장면은 첫 step 에 넣음)
pub struct Orb3Backend {
    inner: Option<Orb3>,
    pending: Option<(Arc<Ring>, Option<CsvReplay>)>,
    args: crate::app::Args,
    calib: crate::imu::Calib,
}

impl Orb3Backend {
    fn get(&mut self, f: &Frame) -> Result<&mut Orb3, String> {
        if self.inner.is_none() {
            let (ring, replay) = match self.pending.take() {
                Some((r, p)) => (Some(r), p),
                None => (None, None),
            };
            self.inner = Some(Orb3::new(&self.args, f, ring, replay, &self.calib)?);
        }
        Ok(self.inner.as_mut().unwrap())
    }
}

impl SlamBackend for Orb3Backend {
    fn step(&mut self, f: Frame, read_ms: f64, _prepped: Option<crate::Features>) -> bool {
        match self.get(&f) {
            Ok(o) => o.step(f, read_ms),
            Err(e) => {
                common::info!("위치: {e}");
                false
            }
        }
    }
    fn pop(&mut self) -> Option<OutFrame> {
        self.inner.as_mut()?.q.pop_front()
    }
    fn frames(&self) -> i64 {
        self.inner.as_ref().map_or(0, |o| o.n)
    }
    fn finish(self: Box<Self>) -> Result<i32, String> {
        let Some(mut s) = self.inner else { return Ok(0) };
        if let Some(r) = &s.ring {
            r.closed.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let wall = s.t_run.elapsed().as_secs_f64();
        // 끝난 지도: 장면마다 기준 키프레임의 지금 위치(지워졌으면 부모를 따라감) · 지도. 가장 많은 장면이 든 지도 = 한 덩어리 지도
        let all = s.sys.keyframes(true);
        let mut per: Vec<(f64, u64, T34)> = Vec::new();
        let mut gone = 0usize;
        for (t, kf, rel) in &s.tracked {
            match s.sys.kf_now(*kf) {
                Some((m, p)) => per.push((*t, m, mul(&p, rel))),
                None => gone += 1,
            }
        }
        let mut cnt: HashMap<u64, usize> = HashMap::new();
        for x in &per {
            *cnt.entry(x.1).or_default() += 1;
        }
        let main_map = cnt.iter().max_by_key(|x| *x.1).map(|x| *x.0);
        // 끝난 뒤 지도 통째 맞춤: 못 이은 지도를 가장 큰 지도에(깊이 자유 공간 검사, mapmerge)
        let mut xf: HashMap<u64, T34> = HashMap::new();
        if let (Some(mm), Some(k)) = (main_map, s.dk) {
            let mut dfr: Vec<(u64, T34, &[u16])> = Vec::new();
            for (_, kf, rel, d) in &s.deps {
                if let Some((m, p)) = s.sys.kf_now(*kf) {
                    dfr.push((m, mul(&p, rel), d));
                }
            }
            let mut maps: HashMap<u64, Vec<crate::mapmerge::Frame>> = HashMap::new();
            for (m, p, d) in &dfr {
                maps.entry(*m).or_default().push(crate::mapmerge::Frame { pose: *p, d });
            }
            let mut hyps = Vec::new();
            for h in &s.hits_all {
                if let (Some((ma, pa)), Some((mb, pb))) = (s.sys.kf_now(h.q_kf), s.sys.kf_now(h.c_kf)) {
                    if ma != mb {
                        let q_a = mul(&pa, &h.q_rel);
                        let c_b = mul(&pb, &h.c_rel);
                        hyps.push((h.inliers, ma, mb, mul(&mul(&c_b, &inv(&h.t_qc)), &inv(&q_a))));
                    }
                }
            }
            let t0 = Instant::now();
            let mut log = Vec::new();
            // 맞은 점 많은 짝부터(지도마다 시작점 수에 한계가 있음)
            hyps.sort_by_key(|x| std::cmp::Reverse(x.0));
            let hyps: Vec<(u64, u64, T34)> = hyps.into_iter().map(|x| (x.1, x.2, x.3)).collect();
            xf = crate::mapmerge::merge(&maps, &hyps, &k, mm, &mut log);
            for l in &log {
                common::info!("끝난 뒤 지도 맞춤: {l}");
            }
            common::info!("끝난 뒤 지도 맞춤: 후보 {} 개, 붙인 지도 {} 개, {:.1} 초", hyps.len(), xf.len() - 1, t0.elapsed().as_secs_f64());
        }
        let orig: Vec<u64> = per.iter().map(|x| x.1).collect();
        let per: Vec<(f64, u64, T34)> = per.into_iter().map(|(t, m, p)| match xf.get(&m) {
            Some(x) => (t, main_map.unwrap(), mul(x, &p)),
            None => (t, m, p),
        }).collect();
        let ref_map = s.ref_kf.and_then(|(id, _)| s.sys.kf_now(id));
        // 보낸 세계 ← 끝 ORB 지도(기준 키프레임이 큰 지도에 있으면 그 못 박음, 아니면 지금 값)
        let a_end = match ref_map {
            Some((m, p)) if Some(m) == main_map => mul(&s.ref_kf.unwrap().1, &inv(&p)),
            _ => s.a,
        };
        let in_main = per.iter().filter(|x| Some(x.1) == main_map).count();
        let other = per.len() - in_main;
        let mut outp: Vec<(f64, T34, u64)> = per.iter().zip(&orig).filter(|x| Some(x.0 .1) == main_map).map(|(x, o)| (x.0, mul(&a_end, &x.2), *o)).collect();
        let (n_lv, max_lv) = level(&mut outp, &s.imu_hist);
        common::info!("끝난 지도 기울기 바로잡기: 장면 {n_lv} 개, 가장 큰 바로잡음 {max_lv:.1}°");
        // 끝난 지도 다듬기: 다른 때 같은 곳을 본 장면끼리 깊이 ICP + 방위 · 이동 위치 그래프(refine)
        if let Some(k) = s.dk {
            let at: HashMap<u64, usize> = outp.iter().enumerate().map(|(i, x)| (x.0.to_bits(), i)).collect();
            let mut nodes = Vec::new();
            let mut nt = Vec::new();
            for (j, (t, _, _, d)) in s.deps.iter().enumerate() {
                if j % 2 != 0 {
                    continue;
                }
                if let Some(&i) = at.get(&t.to_bits()) {
                    nodes.push(crate::refine::Node { t: *t, pose: outp[i].1, seg: outp[i].2, d });
                    nt.push((*t, outp[i].2));
                }
            }
            let t0 = Instant::now();
            let mut log = Vec::new();
            let dl = crate::refine::refine(&nodes, &k, &mut log);
            crate::refine::apply_deltas(&mut outp, &nt, &dl);
            for l in &log {
                common::info!("끝난 지도 다듬기: {l}, {:.1} 초", t0.elapsed().as_secs_f64());
            }
        }
        let lines: Vec<String> = outp.iter().map(|x| tum_line(x.0, &x.1)).collect();
        if !s.final_path.is_empty() {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&s.final_path).map_err(|e| format!("{}: {e}", s.final_path))?);
            let _ = writeln!(f, "# timestamp x y z qx qy qz qw (컬러 카메라 광학 → 지도, 끝난 기준 지도)");
            for l in &lines {
                let _ = writeln!(f, "{l}");
            }
        }
        // 시험용: 모든 지도의 장면 위치(STITCH_LOG 옆 .maps): t 지도 TUM
        if let Ok(p) = std::env::var("STITCH_LOG") {
            if let Ok(f) = std::fs::File::create(format!("{p}.maps")) {
                let mut f = std::io::BufWriter::new(f);
                for x in &per {
                    let _ = writeln!(f, "{} {}", x.1, tum_line(x.0, &x.2));
                }
            }
        }
        let mut maps: HashMap<u64, usize> = HashMap::new();
        for (_, m, _) in &all {
            *maps.entry(*m).or_default() += 1;
        }
        common::info!(
            "위치(ORB-SLAM3): 장면 {} 개, {wall:.1} 초({:.1} 장/초). 위치 나온 장면 {} 개 중 끝 기준 지도 {in_main} · 다른 지도 {other} · 키프레임 지워짐 {gone}. 지도 {} 개(키프레임 {:?})",
            s.n, s.n as f64 / wall.max(1e-9), s.tracked.len(), maps.len(), maps
        );
        common::info!(
            "보냄: 지도에 넣음 {} · 맡아 둠 {} · 놓침 {} · 새 지도 {} · 큰 지도 바뀜 {} · 중력 다시 맞춤 {} · 다시 못 박음 {} · 기준 옮김 {} · 기울기 튐 {} · 그래프 블록 {}(노드 {}) · 밀려 버린 장면 {}",
            s.n_sent_ok, s.n_held, s.n_lost, s.n_newmap, s.n_corr, s.n_regrav, s.n_repin, s.n_switch, s.n_spike, s.n_graph, s.n_nodes, s.dropped
        );
        common::info!("ms/장 위치추적 {}", summary(&s.s_track));
        common::info!("지도 잇기: 보낸 장면 {} · 맞은 짝 {} · 옮겨 붙임 {}(그 가운데 빈틈 이어 붙이기 {})", s.stitch.sent, s.stitch.hits, s.n_graft, s.n_gapgraft);
        drop(s.poses_f.take());
        let _ = std::fs::remove_file(&s.settings_path);
        Ok(0)
    }
}

/// 백엔드와 장면 입력을 연다(입력: --tum/--rsdrec 재생 또는 D435, IMU: --imu-csv 재생 또는 --imu 실시간)
pub fn open(a: crate::app::Args) -> Result<(Box<dyn SlamBackend>, Box<dyn FrameSource>, Option<crate::io::Prep>), String> {
    let calib = crate::imu::Calib::parse(&if a.imu_calib.is_empty() { crate::imu::DEFAULT_CALIB.to_string() } else { std::fs::read_to_string(&a.imu_calib).map_err(|e| format!("{}: {e}", a.imu_calib))? })?;
    let ring = crate::imu::Ring::new();
    let mut replay = None;
    let src: Box<dyn FrameSource> = if !a.tum.is_empty() {
        let ts = crate::io::TumSource::with_stride(&a.tum, a.rate, a.stride)?;
        if !a.imu_csv.is_empty() {
            let r = CsvReplay::load(&a.imu_csv, calib.shift)?;
            if a.rate > 0.0 {
                r.spawn_timed(ring.clone(), ts.clock(), a.rate);
            } else {
                replay = Some(r);
            }
        }
        Box::new(ts)
    } else {
        if !a.imu.is_empty() {
            crate::imu::spawn_reader(&a.imu, ring.clone())?;
        }
        crate::app::open_camera(&a)?
    };
    let has_imu = !a.imu_csv.is_empty() || !a.imu.is_empty();
    if !has_imu {
        return Err("ORB-SLAM3 위치: 지도 수평(중력)을 잡을 IMU 가 필요(--imu 또는 --imu-csv)".into());
    }
    common::info!("입력: {}", src.describe());
    let b = Orb3Backend { inner: None, pending: Some((ring, replay)), args: a, calib };
    Ok((Box::new(b), src, None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rigid_math() {
        let a: T34 = [0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0];
        let i = mul(&a, &inv(&a));
        for r in 0..3 {
            for c in 0..4 {
                assert!((i[r * 4 + c] - if c == r { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }
}
