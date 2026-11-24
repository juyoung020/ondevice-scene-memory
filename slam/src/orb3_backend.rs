//! 위치 백엔드: ORB-SLAM3 RGB-D(컬러 + 컬러에 맞춘 깊이). `backend::open` 이 이것을 연다.
//!
//! 좌표: 보내는 세계 = 기준 키프레임에 못 박은 좌표(처음 10 키프레임을 넘긴 지도의 첫 키프레임). ORB-SLAM3 가 지도를 합치거나
//!   옮겨 붙여 그 지도 좌표가 통째로 바뀌어도, 기준 키프레임의 보낸 위치가 그대로이게 `A`(보내는 세계 ← ORB 지도)를 다시 구한다.
//!   기준 키프레임이 지워지면 이미 보낸 다른 키프레임으로 다시 못 박는다.
//! 중력: 기준 지도에 있는 장면 동안의 가속도(IMU → 카메라 → 기준 키프레임 카메라 축)를 모두 더한 방향이 위. 3 초 넘게 모이면 세계 z 를
//!   그 방향으로 돌리고(GRAVITY), 그 뒤에도 0.3° 넘게 바뀌면 다시 돌려 노드를 모두 다시 보낸다. 오래 더하면 움직임 가속도는 상쇄된다.
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

fn rot_vec(a: &T34, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i * 4] * v[0] + a[i * 4 + 1] * v[1] + a[i * 4 + 2] * v[2])
}

/// 두 위치의 이동(m) · 회전(rad) 차
fn moved(a: &T34, b: &T34) -> (f64, f64) {
    let d = ((a[3] - b[3]).powi(2) + (a[7] - b[7]).powi(2) + (a[11] - b[11]).powi(2)).sqrt();
    let tr: f64 = (0..3).map(|i| (0..3).map(|k| a[i * 4 + k] * b[i * 4 + k]).sum::<f64>()).sum();
    (d, ((tr - 1.0) * 0.5).clamp(-1.0, 1.0).acos())
}

/// 단위 벡터 u 를 +z 로 돌리는 회전(가장 짧은 회전)
fn rot_to_z(u: [f64; 3]) -> T34 {
    let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let u = u.map(|x| x / n);
    // 축 = u × z, 각 = acos(u·z)
    let (ax, ay) = (u[1], -u[0]);
    let s = (ax * ax + ay * ay).sqrt();
    let c = u[2];
    if s < 1e-12 {
        return if c > 0.0 { [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0] };
    }
    let (kx, ky) = (ax / s, ay / s);
    let v = 1.0 - c;
    [kx * kx * v + c, kx * ky * v, ky * s, 0.0, kx * ky * v, ky * ky * v + c, -kx * s, 0.0, -ky * s, kx * s, c, 0.0]
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
/// 중력을 정하기까지 모을 가속도 시간(초, 200 Hz)
const GRAV_MIN_S: f64 = 3.0;
/// 중력 다시 돌리기 문턱(rad)
const GRAV_REDO: f64 = 0.3 * std::f64::consts::PI / 180.0;
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
    imu: Vec<[f64; 4]>,
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
    g_sum: [f64; 3],
    g_n: u64,
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
    pend: Option<(u64, u64, T34, f64)>,
    n_graft: i64,
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
            g_sum: [0.0; 3],
            g_n: 0,
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
            pend: None,
            n_graft: 0,
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
                (self.g_sum, self.g_n, self.g_dir) = ([0.0; 3], 0, None);
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

    /// 이 장면 동안의 가속도를 기준 키프레임 카메라 축으로 더하고, 충분하면 위 방향을 정하거나 다시 돌림. 돌렸으면 true
    fn add_gravity(&mut self, twc: &T34) -> bool {
        let Some((id, out)) = self.ref_kf else { return false };
        let Some(p_ref) = self.kfs.get(&id) else { return false };
        // 기준 키프레임 카메라 ← 지금 카메라
        let rel = mul(&inv(p_ref), twc);
        for s in &self.imu {
            let ac = m34_vec(&self.r_ci, &[s[1] as f32, s[2] as f32, s[3] as f32, 0.0]);
            let v = rot_vec(&rel, [ac[0] as f64, ac[1] as f64, ac[2] as f64]);
            for i in 0..3 {
                self.g_sum[i] += v[i];
            }
            self.g_n += 1;
        }
        if (self.g_n as f64) < GRAV_MIN_S * 200.0 {
            return false;
        }
        let n = (self.g_sum.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let u = self.g_sum.map(|x| x / n);
        if let Some(d) = self.g_dir {
            let c = (d[0] * u[0] + d[1] * u[1] + d[2] * u[2]).clamp(-1.0, 1.0);
            if c.acos() < GRAV_REDO {
                return false;
            }
        }
        // 기준 키프레임의 보낸 회전 = 위(기준 카메라 축 u)를 +z 로 보내는 회전, 위치는 그대로
        let r = rot_to_z(u);
        let mut o = r;
        (o[3], o[7], o[11]) = (out[3], out[7], out[11]);
        if self.g_dir.is_some() {
            self.n_regrav += 1;
        }
        common::info!("위치: 중력 방향 {} (가속도 {:.1} 초)", if self.g_dir.is_some() { "다시 맞춤" } else { "정함" }, self.g_n as f64 / 200.0);
        self.g_dir = Some(u);
        self.ref_kf = Some((id, o));
        self.a = mul(&o, &inv(p_ref));
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
        let agree = match self.pend {
            Some((pi, pf, ps, pt)) if t - pt < 15.0 && ((pi, pf) == (mb, ma) || (pi, pf) == (ma, mb)) => {
                let s = if (pi, pf) == (mb, ma) { ps } else { inv(&ps) };
                let (d, r) = moved(&s, &s_ba);
                common::info!("지도 잇기: 앞 후보와의 차 {d:.3} m · {:.2}°", r.to_degrees());
                d < 0.25 && r < 6f64.to_radians()
            }
            _ => false,
        };
        if !agree {
            self.pend = Some((mb, ma, s_ba, t));
            return;
        }
        self.pend = None;
        let (na, nb) = (self.map_kfs.get(&ma).copied().unwrap_or(0), self.map_kfs.get(&mb).copied().unwrap_or(0));
        let ok = if na <= nb { self.sys.graft(ma, mb, &s_ba) } else { self.sys.graft(mb, ma, &inv(&s_ba)) };
        if ok {
            self.n_graft += 1;
            common::info!("위치: 지도 {} 를 지도 {} 에 옮겨 붙임(신경망 장소 인식, 두 번 같은 변환)", if na <= nb { ma } else { mb }, if na <= nb { mb } else { ma });
        }
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
            ring.drain(|s| imu.push([s.t, s.a[0] as f64, s.a[1] as f64, s.a[2] as f64]));
        }
        for s in &self.imu {
            let a = m34_vec(&self.r_ci, &[s[1] as f32, s[2] as f32, s[3] as f32, 0.0]);
            self.acc.push_back((s[0] - self.shift, [a[0] as f64, a[1] as f64, a[2] as f64]));
        }
        while self.acc.front().is_some_and(|x| f.t - x.0 > 1.0) {
            self.acc.pop_front();
        }
        // 최대 속도 재생(받는 쪽을 기다림)은 잇기 결과도 기다려 실시간처럼 제때 반영
        while let Some(r) = self.stitch.recv(self.block) {
            if let Some(h) = r {
                self.on_hit(h, f.t);
            }
        }
        let c0 = Instant::now();
        let (w, h) = (f.cam.width, f.cam.height);
        if self.block {
            self.sys.wait_idle();
        }
        let tr = self.sys.track(&f.color, &f.depth_mm, w, h, f.t);
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
        if ok && (changed || tr.map_id != self.kfs_map || f.t - self.kfs_t >= SWEEP_S || tr.ref_kf.is_some_and(|k| !self.kfs.contains_key(&k))) {
            self.read_kfs(f.t);
        }
        if ok {
            self.update_frame();
        }
        let mut regrav = false;
        if ok && self.trusted {
            regrav = self.add_gravity(&tr.twc);
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
                self.tracked.push((f.t, k, mul(&inv(&self.kfs[&k]), &tr.twc)));
            }
        }
        // 지도 잇기에 장면 보내기(0.25 초마다, 1 초 평균 가속도로 본 IMU 위쪽이 믿을 만할 때)
        if ok && f.t - self.last_node_t >= 0.25 && self.acc.len() >= 20 {
            if f.t - self.kfmap_t >= 1.0 || changed {
                self.all_kfs(f.t);
            }
            let m = self.acc.iter().fold([0.0; 3], |s, x| [s[0] + x.1[0], s[1] + x.1[1], s[2] + x.1[2]]).map(|v| v / self.acc.len() as f64);
            let g = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            if let (Some(k), true) = (tr.ref_kf, (g - 9.81).abs() < 2.0) {
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
        let ref_map = s.ref_kf.and_then(|(id, _)| s.sys.kf_now(id));
        // 보낸 세계 ← 끝 ORB 지도(기준 키프레임이 큰 지도에 있으면 그 못 박음, 아니면 지금 값)
        let a_end = match ref_map {
            Some((m, p)) if Some(m) == main_map => mul(&s.ref_kf.unwrap().1, &inv(&p)),
            _ => s.a,
        };
        let in_main = per.iter().filter(|x| Some(x.1) == main_map).count();
        let other = per.len() - in_main;
        let lines: Vec<String> = per.iter().filter(|x| Some(x.1) == main_map).map(|x| tum_line(x.0, &mul(&a_end, &x.2))).collect();
        if !s.final_path.is_empty() {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&s.final_path).map_err(|e| format!("{}: {e}", s.final_path))?);
            let _ = writeln!(f, "# timestamp x y z qx qy qz qw (컬러 카메라 광학 → 지도, 끝난 기준 지도)");
            for l in &lines {
                let _ = writeln!(f, "{l}");
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
            "보냄: 지도에 넣음 {} · 맡아 둠 {} · 놓침 {} · 새 지도 {} · 큰 지도 바뀜 {} · 중력 다시 맞춤 {} · 다시 못 박음 {} · 기준 옮김 {} · 그래프 블록 {}(노드 {}) · 밀려 버린 장면 {}",
            s.n_sent_ok, s.n_held, s.n_lost, s.n_newmap, s.n_corr, s.n_regrav, s.n_repin, s.n_switch, s.n_graph, s.n_nodes, s.dropped
        );
        common::info!("ms/장 위치추적 {}", summary(&s.s_track));
        common::info!("지도 잇기: 보낸 장면 {} · 맞은 짝 {} · 옮겨 붙임 {}", s.stitch.sent, s.stitch.hits, s.n_graft);
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
        // 아무 방향이나 +z 로
        for u in [[0.3, -0.5, 0.8], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]] {
            let r = rot_to_z(u);
            let v = rot_vec(&r, u);
            let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            assert!((v[2] / n - 1.0).abs() < 1e-9, "{u:?} → {v:?}");
            let (_, ang) = moved(&mul(&r, &inv(&r)), &crate::stream::IDENTITY);
            assert!(ang < 1e-6);
        }
    }
}
