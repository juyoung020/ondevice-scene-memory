//! 위치 백엔드: 같은 프로세스 처리 루프(pipeline `stream.rs`)와 따로 도는 `ondevice slam` 은 이 trait 만 본다.
//! 백엔드는 장면을 받아 장면 스트림 항목(`OutFrame`: 위치 · 플래그 · 컬러 · 깊이, 그래프 블록: 루프 보정 · 지도 합치기로 움직인 노드)을
//! 넣은 차례대로 내놓는다. 형식과 좌표 약속은 docs/bridge.md "스트림 형식". 물체 기억 · 점 지도 · 뷰어는 백엔드와 상관없이 같다.
//!
//! ORB-SLAM3 스테레오(기본) · 스테레오-관성(IMU 를 줄 때, `Orb3`): 적외선 좌우(+ IMU) → 적외선 왼쪽 카메라 위치 → 컬러 카메라 위치로 바꿔 보냄.
//!   - 노드 = ORB-SLAM3 키프레임(노드 id = 키프레임 id + 1), 장면의 기준 노드 = 그 장면의 기준 키프레임. 처음 보는 노드는 장면 앞 그래프 블록으로
//!   - 큰 지도 바뀜(루프 · 합치기 · IMU 초기화 단계) 때는 현재 지도의 노드를 모두, 그 밖에는 1 초마다 1 cm · 0.5° 넘게 움직인 노드만 다시 보냄
//!   - 스테레오-관성은 중력 맞춤 전(IMU 초기화 · 첫 관성 정밀화 전: 세계가 중력과 안 맞음)을, 둘 다 놓침 · 처음 믿은 지도와 아직 안 합쳐진 새 지도를 놓침으로 보냄(지도에 안 넣음).
//!     스테레오는 중력을 모르므로 받는 쪽이 바닥으로 수평을 맞춘다. 잠깐 놓친 장면(RECENTLY_LOST)은 위치만. 새 지도가 HOLD_MAX 초 넘게 안 합쳐지면 리셋 블록을 보내고 새로 쌓기
pub use crate::app::open;
use crate::stream::OutFrame;
use crate::Frame;

pub trait SlamBackend: Send {
    /// 장면 하나를 위치 추적에 넣는다(출력은 `pop` 으로). false = 끝
    fn step(&mut self, f: Frame, read_ms: f64) -> bool;
    /// 다음 출력(장면 · 그래프 블록 · 리셋 블록), 넣은 차례대로
    fn pop(&mut self) -> Option<OutFrame>;
    /// 지금까지 넣은 장면 수
    fn frames(&self) -> i64;
    /// 끝 요약 · 최종 위치 파일. 부르는 쪽은 그 전에 `pop` 으로 남은 출력을 꺼내 둔다
    fn finish(self: Box<Self>) -> Result<i32, String>;
}

/// 3×4 행 우선 강체 변환(카메라 → 세계 등)
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

/// 시간 · 횟수 요약(ms)
#[derive(Default)]
pub struct Summary {
    v: Vec<f64>,
}

impl Summary {
    pub fn add(&mut self, x: f64) {
        self.v.push(x);
    }
    pub fn str(&self) -> String {
        if self.v.is_empty() {
            return "-".into();
        }
        let mut s = self.v.clone();
        s.sort_by(f64::total_cmp);
        let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
        format!("평균 {:.1} · 중앙 {:.1} · p95 {:.1} · 최대 {:.1} ({} 번)", s.iter().sum::<f64>() / s.len() as f64, q(0.5), q(0.95), q(1.0), s.len())
    }
}

/// 백엔드를 만든다(ORB-SLAM3 단어장 읽기에 수십 초)
#[cfg(feature = "orb3")]
pub fn make(o: crate::app::OpenParts) -> Result<Box<dyn SlamBackend>, String> {
    Ok(Box::new(orb3_backend::Orb3::new(o)?))
}

#[cfg(not(feature = "orb3"))]
pub fn make(_: crate::app::OpenParts) -> Result<Box<dyn SlamBackend>, String> {
    Err("이 빌드에는 ORB-SLAM3 가 없음(slam 의 orb3 기능 · ORB_SLAM3_DIR)".into())
}

#[cfg(feature = "orb3")]
mod orb3_backend {
    use super::*;
    use crate::imu::{CsvReplay, Ring};
    use crate::orb3::{System, OK, RECENTLY_LOST};
    use crate::stream::{GraphNode, GRAVITY, LOST, MAP_CORRECTED, NEW_MAP, NO_MAP};
    use std::collections::{HashMap, VecDeque};
    use std::io::Write;
    use std::sync::Arc;
    use std::time::Instant;

    /// 처음 믿은 지도와 합쳐지지 않은 새 지도를 기다리는 최대 시간(초): 넘으면 리셋 블록 뒤 새 지도부터
    const HOLD_MAX: f64 = 10.0;
    /// 노드 다시 보내기: 이만큼(초)마다 움직인 노드를 훑음, 움직임 문턱(m, rad)
    const SWEEP_S: f64 = 1.0;
    const SWEEP_M: f64 = 0.01;
    const SWEEP_RAD: f64 = 0.5f64 * std::f64::consts::PI / 180.0;

    pub struct Orb3 {
        sys: System,
        /// IMU 표본(스테레오-관성만): 실시간 읽기 스레드 · 재생이 넣음
        ring: Option<Arc<Ring>>,
        /// 최대 속도 재생: 장면마다 그 시각까지의 표본을 링에 넣음
        replay: Option<CsvReplay>,
        /// t_imu = t_cam + shift
        shift: f64,
        /// 컬러 카메라 → 적외선 왼쪽(장면 · 노드 위치를 컬러 카메라로)
        ir_from_color: T34,
        q: VecDeque<OutFrame>,
        cap: usize,
        block: bool,
        send_every: i64,
        reset_file: String,
        n: i64,
        sent: HashMap<u64, T34>,
        /// 믿는 지도(처음 지도에 넣은 장면의 지도 id). 다른 지도로 바뀌면 보낸 노드가 그 지도로 합쳐졌는지 확인해 옮김
        trusted_map: Option<u64>,
        last_merge_check: f64,
        untrusted_since: Option<f64>,
        last_sweep: f64,
        was_init: bool,
        last_map: Option<u64>,
        poses_f: Option<std::io::BufWriter<std::fs::File>>,
        final_path: String,
        /// 지도에 넣은 장면: (시각, 기준 키프레임, 기준 노드에서 본 장면 위치)
        mapped: Vec<(f64, u64, T34)>,
        settings_path: std::path::PathBuf,
        t_run: Instant,
        // 집계
        n_mapped: i64,
        n_pre_init: i64,
        n_lost: i64,
        n_recent: i64,
        n_untrusted: i64,
        n_resets: i64,
        n_corr: i64,
        n_newmap: i64,
        n_graph: i64,
        n_nodes: i64,
        dropped: i64,
        n_imu: u64,
        s_track: Summary,
        s_read: Summary,
    }

    impl Orb3 {
        pub fn new(o: crate::app::OpenParts) -> Result<Orb3, String> {
            std::fs::write(&o.settings_path, &o.settings).map_err(|e| format!("{}: {e}", o.settings_path.display()))?;
            let t0 = Instant::now();
            let sys = System::new(&o.voc, &o.settings_path.to_string_lossy(), o.ring.is_some())?;
            common::info!("ORB-SLAM3 {} 준비 {:.1} 초(단어장 {}, 설정 {})", if o.ring.is_some() { "스테레오-관성" } else { "스테레오" }, t0.elapsed().as_secs_f64(), o.voc, o.settings_path.display());
            let poses_f = if o.poses.is_empty() {
                None
            } else {
                let mut f = std::io::BufWriter::new(std::fs::File::create(&o.poses).map_err(|e| format!("{}: {e}", o.poses))?);
                let _ = writeln!(f, "# timestamp x y z qx qy qz qw (컬러 카메라 광학 → 지도, 보낸 그때 위치)");
                Some(f)
            };
            Ok(Orb3 {
                sys,
                ring: o.ring,
                replay: o.replay,
                shift: o.shift,
                ir_from_color: inv(&o.color_from_ir),
                q: VecDeque::new(),
                cap: o.cap.max(1),
                block: o.block,
                send_every: o.send_every.max(1),
                reset_file: o.reset_file,
                n: 0,
                sent: HashMap::new(),
                trusted_map: None,
                last_merge_check: f64::MIN,
                untrusted_since: None,
                last_sweep: f64::MIN,
                was_init: false,
                last_map: None,
                poses_f,
                final_path: o.final_poses,
                mapped: Vec::new(),
                settings_path: o.settings_path,
                t_run: Instant::now(),
                n_mapped: 0,
                n_pre_init: 0,
                n_lost: 0,
                n_recent: 0,
                n_untrusted: 0,
                n_resets: 0,
                n_corr: 0,
                n_newmap: 0,
                n_graph: 0,
                n_nodes: 0,
                dropped: 0,
                n_imu: 0,
                s_track: Summary::default(),
                s_read: Summary::default(),
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

        /// 물체 기억 · 점 지도를 처음부터(리셋 블록: 받는 쪽이 지도를 비움). 위치 추적은 지금 지도로 이어 가고, 노드는 다음 장면에서 모두 다시 보냄
        fn restart(&mut self, why: &str) {
            common::info!("위치: {why} → 리셋 블록 보내고 새로 쌓기");
            self.push(OutFrame { reset: true, ..Default::default() });
            self.sent.clear();
            self.mapped.clear();
            (self.trusted_map, self.untrusted_since, self.was_init) = (None, None, false);
            self.n_resets += 1;
        }

        /// 노드 보내기: full 이면 현재 지도의 노드를 모두, 아니면 새 노드와 움직인 노드만
        fn send_nodes(&mut self, full: bool) {
            let mut nodes = Vec::new();
            for (id, twc) in self.sys.keyframes() {
                let p = mul(&twc, &self.ir_from_color);
                let new = match self.sent.get(&id) {
                    None => true,
                    Some(q) => {
                        let (d, a) = moved(q, &p);
                        full || d > SWEEP_M || a > SWEEP_RAD
                    }
                };
                if new {
                    nodes.push(GraphNode { id: (id + 1) as u32, held: 0, t: p });
                    self.sent.insert(id, p);
                }
            }
            if !nodes.is_empty() {
                self.n_graph += 1;
                self.n_nodes += nodes.len() as i64;
                self.push(OutFrame { graph: nodes, ..Default::default() });
            }
        }
    }

    impl SlamBackend for Orb3 {
        fn step(&mut self, mut f: Frame, read_ms: f64) -> bool {
            self.s_read.add(read_ms);
            // 뷰어 지도 초기화(같은 프로세스 뷰어는 공유 신호, 따로 띄운 뷰어는 리셋 파일): 물체 기억 · 점 지도만 비우고 위치 지도는 그대로 이어 감
            // (위치 지도를 비우면 움직임이 있어야 다시 시작해, 가만히 둔 카메라로는 다시 쌓이지 않음)
            if common::control::take_reset() || (!self.reset_file.is_empty() && std::fs::remove_file(&self.reset_file).is_ok()) {
                self.restart("뷰어 지도 초기화");
            }
            let c0 = Instant::now();
            let Some(ir) = f.ir.take() else {
                common::info!("장면 {}: 적외선 좌우가 없음(놓침으로 보냄)", f.id);
                self.n_lost += 1;
                let o = OutFrame { flags: LOST, seq: self.n as u64, t: f.t, ..Default::default() };
                self.n += 1;
                self.push(o);
                return true;
            };
            // 이 장면(적외선 시각)까지의 IMU 표본(카메라 시계로, 스테레오-관성만)
            let mut imu: Vec<[f64; 7]> = Vec::with_capacity(16);
            if let Some(ring) = &self.ring {
                let t_imu = ir.t + self.shift;
                if let Some(r) = &mut self.replay {
                    r.feed(ring, t_imu);
                }
                let sh = self.shift;
                ring.take_until(t_imu, |s| imu.push([s.t - sh, s.a[0] as f64, s.a[1] as f64, s.a[2] as f64, s.w[0] as f64, s.w[1] as f64, s.w[2] as f64]));
                self.n_imu += imu.len() as u64;
            }
            let tr = self.sys.track(&ir.left, &ir.right, ir.w, ir.h, ir.t, &imu);
            self.s_track.add(c0.elapsed().as_secs_f64() * 1e3);

            let changed = self.sys.map_changed();
            let mut flags = 0u32;
            if self.last_map != Some(tr.map_id) {
                if self.last_map.is_some() {
                    flags |= NEW_MAP;
                    self.n_newmap += 1;
                }
                self.last_map = Some(tr.map_id);
            }
            let tracking = tr.state == OK || tr.state == RECENTLY_LOST;
            // 믿는 지도인가: 같은 지도 id 거나, 보낸 노드가 지금 지도로 합쳐져 있으면(1 초마다 확인) 지금 지도를 믿는 지도로
            let trusted = match self.trusted_map {
                None => true,
                Some(m) if m == tr.map_id => true,
                Some(_) if f.t - self.last_merge_check >= 1.0 => {
                    self.last_merge_check = f.t;
                    let merged = self.sys.keyframes().iter().any(|(id, _)| self.sent.contains_key(id));
                    if merged {
                        common::info!("위치: 새 지도 {} 가 믿는 지도와 합쳐짐", tr.map_id);
                        self.trusted_map = Some(tr.map_id);
                    }
                    merged
                }
                Some(_) => false,
            };
            if trusted {
                self.untrusted_since = None;
            } else if f.t - *self.untrusted_since.get_or_insert(f.t) > HOLD_MAX {
                self.restart(&format!("처음 지도와 {HOLD_MAX} 초 넘게 안 합쳐짐"));
            }
            let inertial = self.ring.is_some();
            let usable = (tr.gravity || !inertial) && trusted && tracking;
            let mut anchor = 0u32;
            if usable {
                let full = changed || !self.was_init;
                self.was_init = true;
                if full || f.t - self.last_sweep >= SWEEP_S || tr.ref_kf.is_some_and(|k| !self.sent.contains_key(&k)) {
                    self.last_sweep = f.t;
                    self.send_nodes(full);
                }
                if changed {
                    flags |= MAP_CORRECTED;
                    self.n_corr += 1;
                }
                if inertial {
                    flags |= GRAVITY;
                }
                if tr.state == RECENTLY_LOST {
                    flags |= NO_MAP;
                    self.n_recent += 1;
                }
                if let Some(k) = tr.ref_kf.filter(|k| self.sent.contains_key(k)) {
                    anchor = (k + 1) as u32;
                }
                self.trusted_map.get_or_insert(tr.map_id);
            } else {
                flags |= LOST;
                if inertial && !tr.gravity {
                    self.n_pre_init += 1;
                } else if !trusted {
                    self.n_untrusted += 1;
                } else {
                    self.n_lost += 1;
                }
            }
            let pose = mul(&tr.twc, &self.ir_from_color);
            if usable && tr.state == OK {
                self.n_mapped += 1;
                if let Some(pf) = &mut self.poses_f {
                    let _ = writeln!(pf, "{}", tum_line(f.t, &pose));
                }
                if let (Some(k), true) = (tr.ref_kf, anchor != 0) {
                    let node = self.sent[&k];
                    self.mapped.push((f.t, k, mul(&inv(&node), &pose)));
                }
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
                width: f.cam.width,
                height: f.cam.height,
                color: std::mem::take(&mut f.color),
                color_is_rgb: f.color_is_rgb,
                depth,
                anchor,
                ..Default::default()
            };
            self.push(o);
            true
        }

        fn pop(&mut self) -> Option<OutFrame> {
            self.q.pop_front()
        }

        fn frames(&self) -> i64 {
            self.n
        }

        fn finish(self: Box<Self>) -> Result<i32, String> {
            let mut s = *self;
            if let Some(r) = &s.ring {
                r.closed.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            let wall = s.t_run.elapsed().as_secs_f64();
            if !s.final_path.is_empty() {
                // 끝난 지도의 키프레임 위치 × 장면이 기준 키프레임에서 본 위치
                let kfs: HashMap<u64, T34> = s.sys.keyframes().into_iter().map(|(id, t)| (id, mul(&t, &s.ir_from_color))).collect();
                let mut f = std::io::BufWriter::new(std::fs::File::create(&s.final_path).map_err(|e| format!("{}: {e}", s.final_path))?);
                let _ = writeln!(f, "# timestamp x y z qx qy qz qw (컬러 카메라 광학 → 지도, 끝난 지도 기준)");
                let mut k = 0;
                for (t, kf, rel) in &s.mapped {
                    if let Some(node) = kfs.get(kf) {
                        let _ = writeln!(f, "{}", tum_line(*t, &mul(node, rel)));
                        k += 1;
                    }
                }
                common::info!("최종 위치 {k} 장(지도에서 빠진 키프레임 기준 {} 장 뺌) → {}", s.mapped.len() - k, s.final_path);
            }
            drop(s.poses_f.take());
            let _ = std::fs::remove_file(&s.settings_path);
            let ld = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
            common::info!(
                "장면 {} 개, {wall:.1} 초({:.1} 장/초): 지도에 넣음 {} · 중력 맞춤 전 {} · 놓침 {} · 잠깐 놓침(위치만) {} · 안 합쳐진 새 지도 {} · 리셋 {}",
                s.n, s.n as f64 / wall.max(1e-9), s.n_mapped, s.n_pre_init, s.n_lost, s.n_recent, s.n_untrusted, s.n_resets
            );
            common::info!("새 지도 {} · 큰 지도 바뀜 {} · 그래프 블록 {} 개(노드 {} 개) · 밀려 버린 장면 {}", s.n_newmap, s.n_corr, s.n_graph, s.n_nodes, s.dropped);
            if let Some(r) = &s.ring {
                common::info!("IMU 표본 {} 개(장면당 {:.1}), 링 넘침 {} · 읽기 실패 {} · 거름 {:?}", s.n_imu, s.n_imu as f64 / s.n.max(1) as f64, ld(&r.dropped), ld(&r.read_fail), r.bad.iter().map(ld).collect::<Vec<_>>());
            }
            common::info!("ms/장  읽기     {}", s.s_read.str());
            common::info!("ms/장  위치추적 {}", s.s_track.str());
            Ok(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rigid_math() {
        // z 축 90° 회전 + 이동
        let a: T34 = [0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0];
        let i = mul(&a, &inv(&a));
        for r in 0..3 {
            for c in 0..4 {
                let e = if c == r { 1.0 } else { 0.0 };
                assert!((i[r * 4 + c] - e).abs() < 1e-12);
            }
        }
        let (d, ang) = moved(&a, &mul(&a, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02]));
        assert!((d - 0.02).abs() < 1e-12 && ang.abs() < 1e-7);
        // 90° 회전의 사원수 = (0, 0, sin 45°, cos 45°)
        let l = tum_line(1.0, &a);
        let v: Vec<f64> = l.split(' ').map(|x| x.parse().unwrap()).collect();
        assert!((v[6] - 0.5f64.sqrt()).abs() < 1e-6 && (v[7] - 0.5f64.sqrt()).abs() < 1e-6);
    }
}
