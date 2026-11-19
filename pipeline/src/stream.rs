//! 장면 입력(--slam · --stream): 위치 프로그램이 보내는 장면(위치·컬러·깊이)과 그 사이에 끼어 오는 그래프 블록(새로 생기거나 움직인 지도 노드 위치).
//! 같은 프로세스의 위치 프로그램(--slam)은 메모리 큐로, 녹화·파이프(--stream)는 바이트 스트림으로 온다. 형식·읽기는 `slam::stream` 하나를 쓰고,
//! 여기서는 둘 다 같은 함수로 `StreamFrame` 을 만든다. 형식과 좌표 약속은 docs/bridge.md "스트림 형식".
use crate::geom::*;
pub use slam::stream::{GRAVITY, HELD, LOST, MAP_CORRECTED, NEW_MAP, NO_MAP};
use slam::stream::OutFrame;

/// 읽기 버퍼: 머리·그래프 블록용. 사진은 이보다 커서 버퍼를 거치지 않고 받을 자리에 바로 읽힌다
const READ_BUF: usize = 1 << 16;
/// 실시간 위치 추적이 한 번에 처리 루프를 쥐는 최대 시간(카메라 한 장 간격, 30 fps)
const SERVICE_BUDGET: std::time::Duration = std::time::Duration::from_millis(33);

/// 그래프 노드: (지도 노드 id, 자세, 위치, 맡아 둠) — OpenCV 카메라 → 세계, 장면 위치와 같은 좌표.
/// 맡아 둠 = 믿는 지도와 아직 안 이어진 새 지도의 노드(이어지면 같은 id 가 맡아 둠 없이 다시 온다)
pub type Node = (u32, M3, V3, bool);

pub struct StreamFrame {
    pub flags: u32,
    pub map_id: u32,
    pub seq: u64,
    pub t: f64,
    /// 카메라(OpenCV: x 오른쪽, y 아래, z 앞) → 세계(지도: x 앞, y 왼쪽, z 위)
    pub r: M3,
    pub p: V3,
    pub k: [f64; 4],
    pub w: usize,
    pub h: usize,
    /// 깊이 값 / 미터(RealSense 1000, TUM 5000)
    pub depth_per_m: f32,
    pub rgb: Vec<u8>,
    pub depth: slam::DepthBuf,
    /// 기준 지도 노드 id(0 = 없음): 이 장면 위치 = 그 노드 위치 × 상대 위치(루프 클로저 뒤 다시 쌓기용)
    pub anchor: u32,
    /// 이 장면 앞에 온 그래프 블록의 노드
    pub graph: Vec<Node>,
}

/// 카메라 입력이면 버려지는 장면(건너뜀·밀림 등)의 버퍼를 드라이버에 돌려줌
impl Drop for StreamFrame {
    fn drop(&mut self) {
        if slam::bufpool::enabled() {
            slam::bufpool::give_rgb(std::mem::take(&mut self.rgb));
            slam::bufpool::give_depth_buf(std::mem::take(&mut self.depth));
        }
    }
}

impl StreamFrame {
    /// 위치가 없는 장면(맡아 둘 장면은 위치가 있으니 아님)
    pub fn lost(&self) -> bool {
        self.flags & LOST != 0 && !self.held()
    }
    /// 지도 z 가 중력 위(위치 쪽 IMU 수평 맞춤)
    pub fn gravity(&self) -> bool {
        self.flags & GRAVITY != 0
    }
    pub fn held(&self) -> bool {
        self.flags & HELD != 0
    }
    /// 맡아 둘 장면이지만 위치는 바로 보이는 장면(IMU 모드 · 이어지지 않은 새 지도의 임시 위치)
    pub fn held_live(&self) -> bool {
        self.held() && self.flags & LOST == 0
    }
    /// 위치만(IMU 로 이어 붙였거나 영상이 IMU 와 어긋난 뒤): 카메라만 옮기고 지도 · 물체에 넣지 않음
    pub fn no_map(&self) -> bool {
        self.flags & NO_MAP != 0
    }
}

fn pose_of(t3x4: &[f64; 12]) -> (M3, V3) {
    ([[t3x4[0], t3x4[1], t3x4[2]], [t3x4[4], t3x4[5], t3x4[6]], [t3x4[8], t3x4[9], t3x4[10]]], [t3x4[3], t3x4[7], t3x4[11]])
}

/// 장면이 오는 곳: 같은 프로세스의 위치 프로그램(처리 루프가 직접 돌림) 또는 바이트 스트림(파이프·파일)
enum Src {
    Slam(Box<SlamSide>),
    Bytes(slam::stream::Reader<Box<dyn std::io::Read + Send>>),
}

/// 미리 푼 재생 장면: (장면, 읽기 ms, 미리 뽑은 특징)
type Ahead = Result<Option<(slam::Frame, f64, Option<slam::Features>)>, String>;

/// 같은 프로세스 위치 프로그램(`--slam`): 스레드 없이 처리 루프가 장면마다 위치 추적을 부른다(백엔드는 `slam::backend`).
/// 카메라는 장치 fd 가 루프 epoll 에 걸려 장면이 오면 루프가 깨고, 녹화 재생은 다음 장면 PNG 를 작업 풀이 한 장 앞서 푼다
pub struct SlamSide {
    r: Option<Box<dyn slam::backend::SlamBackend>>,
    camera: Option<(Box<dyn slam::io::FrameSource>, Option<slam::io::Prep>)>,
    /// 재생 풀기 줄(입력 · 미리 할 일을 쥠)과 결과 줄
    replay: Option<(common::rt::SharedLane<(Box<dyn slam::io::FrameSource>, Option<slam::io::Prep>)>, std::sync::mpsc::Receiver<Ahead>)>,
    /// 푼 장면이 줄에 있거나 푸는 중
    asked: bool,
    /// 찍은 속도 재생: (배속, 재생 시계), 아직 때가 안 된 장면
    pace: Option<(f64, std::sync::Arc<std::sync::OnceLock<(f64, std::time::Instant)>>)>,
    held: Option<(slam::Frame, f64, Option<slam::Features>)>,
    /// 결과를 늘 같게(최대 속도 재생): 받는 쪽이 줄을 비울 때만 다음 장면
    pub block: bool,
    ended: bool,
    max_frames: i64,
    waker: std::sync::Arc<common::rt::Waker>,
    /// 푼 장면을 보낼 곳(재생)
    tx: Option<std::sync::mpsc::Sender<Ahead>>,
    /// 끝난 뒤 남은 항목
    rest: std::collections::VecDeque<OutFrame>,
}

impl SlamSide {
    /// 위치 프로그램 준비(설정 · IMU · 입력 열기). waker: 재생 장면을 다 풀면 루프를 깨움
    pub fn new(a: slam::app::Args, waker: std::sync::Arc<common::rt::Waker>) -> Result<SlamSide, String> {
        let (block, max_frames) = (a.block_mode(), a.max_frames);
        let (r, mut src, prep) = slam::backend::open(a)?;
        let pace = src.caller_paces();
        let (camera, replay, tx) = if src.fds().is_empty() {
            let (tx, rx) = std::sync::mpsc::channel::<Ahead>();
            let lane = common::rt::SharedLane::new("replay", (src, prep), 0, None);
            (None, Some((lane, rx)), Some(tx))
        } else {
            (Some((src, prep)), None, None)
        };
        let mut side = SlamSide { r: Some(r), camera, replay, asked: false, pace, held: None, block, ended: false, max_frames, waker, tx, rest: Default::default() };
        side.ask();
        Ok(side)
    }

    /// 카메라 장치 fd(루프 epoll 에 건다). 재생이면 빔
    pub fn fds(&self) -> Vec<i32> {
        self.camera.as_ref().map_or(vec![], |c| c.0.fds())
    }

    /// 재생: 다음 장면 풀기를 작업 풀에 맡김(한 장 앞서)
    fn ask(&mut self) {
        let Some((lane, _)) = &self.replay else { return };
        if self.asked {
            return;
        }
        self.asked = true;
        let (tx, w) = (self.tx.clone().unwrap(), self.waker.clone());
        lane.push(move |(src, prep)| {
            let r = src.next().map(|o| {
                o.map(|f| {
                    let ms = src.last_read_ms();
                    let pf = prep.as_mut().map(|p| p(&f));
                    (f, ms, pf)
                })
            });
            let _ = tx.send(r);
            w.wake();
        });
    }

    /// 다음 장면 때까지 남은 시간(찍은 속도 재생에서 잡아 둔 장면)
    pub fn due(&self) -> Option<std::time::Duration> {
        let (f, ..) = self.held.as_ref()?;
        let (rate, clock) = self.pace.as_ref()?;
        let (t0, w0) = clock.get()?;
        Some(std::time::Duration::from_secs_f64(((f.t - t0) / rate - w0.elapsed().as_secs_f64()).max(0.0)))
    }

    /// 위치 추적에 장면 하나를 넣어 봄: true = 넣음(줄에 장면 · 그래프가 생겼을 수 있음), false = 아직 없음 또는 끝
    /// wait: 장면이 올 때까지 기다림(첫 장면 · 기다리는 읽기)
    fn advance(&mut self, wait: bool) -> bool {
        if self.ended {
            return false;
        }
        let r = self.r.as_mut().unwrap();
        if self.max_frames >= 0 && r.frames() >= self.max_frames {
            self.end();
            return false;
        }
        let item = if let Some((src, prep)) = self.camera.as_mut() {
            match if wait { src.next().map(Some) } else { src.try_next() } {
                Ok(None) => return false,
                Ok(Some(None)) => None,
                Ok(Some(Some(f))) => {
                    let ms = src.last_read_ms();
                    let pf = prep.as_mut().map(|p| p(&f));
                    Some((f, ms, pf))
                }
                Err(e) => {
                    common::info!("slam 입력 오류: {e}");
                    None
                }
            }
        } else {
            let got = match self.held.take() {
                Some(h) => Ok(Some(h)),
                None => match if wait { self.replay.as_ref().unwrap().1.recv().map_err(|_| ()) } else { self.replay.as_ref().unwrap().1.try_recv().map_err(|_| ()) } {
                    Ok(x) => {
                        self.asked = false;
                        x
                    }
                    Err(()) => return false,
                },
            };
            match got {
                Ok(Some(h)) => {
                    // 찍은 속도 재생: 때가 안 됐으면 잡아 둠(루프가 그때까지 잠듦)
                    if let Some((rate, clock)) = &self.pace {
                        if let Some((t0, w0)) = clock.get() {
                            let ahead = (h.0.t - t0) / rate - w0.elapsed().as_secs_f64();
                            if ahead > 0.0 {
                                if !wait {
                                    self.held = Some(h);
                                    return false;
                                }
                                std::thread::sleep(std::time::Duration::from_secs_f64(ahead));
                            }
                        }
                    }
                    self.ask();
                    Some(h)
                }
                Ok(None) => None,
                Err(e) => {
                    common::info!("slam 입력 오류: {e}");
                    None
                }
            }
        };
        let Some((f, ms, pf)) = item else {
            self.end();
            return false;
        };
        let r = self.r.as_mut().unwrap();
        if !r.step(f, ms, pf) {
            self.end();
            return false;
        }
        true
    }

    /// 입력 끝: 위치 프로그램 끝 요약(끝 그래프 · 통계)
    fn end(&mut self) {
        self.ended = true;
        if let Some(mut r) = self.r.take() {
            // 끝 요약 전에 줄에 남은 장면을 꺼낼 수 있게 옮겨 둠
            while let Some(f) = r.pop() {
                self.rest.push_back(f);
            }
            if let Err(e) = r.finish() {
                common::info!("slam 끝: {e}");
            }
        }
    }

    /// 다음 항목(장면 · 그래프 · 리셋 블록): 줄이 비었으면 위치 추적을 한 장 돌려 채움. Err = 아직 없음
    fn next_item(&mut self, wait: bool) -> Result<Option<OutFrame>, ()> {
        loop {
            if let Some(f) = self.rest.pop_front() {
                return Ok(Some(f));
            }
            if let Some(f) = self.r.as_mut().and_then(|r| r.pop()) {
                return Ok(Some(f));
            }
            if self.ended {
                return Ok(None);
            }
            if !self.advance(wait) && !self.ended {
                return Err(());
            }
        }
    }

    /// IMU 모드 장면 사이 위치: 표본이 오면 깨울 손잡이를 단다(뷰어가 있을 때만)
    pub fn set_imu_waker(&self, w: std::sync::Arc<common::rt::Waker>) -> bool {
        match self.r.as_ref().and_then(|r| r.imu_ring()) {
            Some(ring) => ring.waker.set(w).is_ok(),
            None => false,
        }
    }

    /// 가장 새 IMU 표본까지의 장면 사이 위치(카메라 시계 시각, IMU 시각, 카메라 광학 → 지도)
    pub fn fast_pose(&self) -> Option<(f64, f64, slam::register::Tf)> {
        self.r.as_ref()?.imu_ring()?.fast_next()
    }

    /// 실시간(카메라): 준비된 장면을 모두 위치 추적에(받는 쪽이 바빠도 위치는 장면마다, 줄이 차면 오래된 장면부터 버림).
    /// 위치 추적이 카메라보다 느려지면(부하 · 열 보호) 장면이 늘 준비돼 있으므로, SERVICE_BUDGET 이 지나면 처리 루프로 돌려준다
    /// (돌려주지 않으면 루프가 물체 기억에 장면을 넘기지도, 뷰어 요청을 받지도 못한다)
    pub fn service(&mut self) {
        if !self.block {
            let t0 = std::time::Instant::now();
            while t0.elapsed() < SERVICE_BUDGET && self.advance(false) {}
        }
    }
}

/// 리셋 블록을 만난 읽기 쪽이 여기 맡겨 두면 다음 open 이 이어서 읽는다(같은 파이프, 버퍼에 남은 바이트 포함)
static HANDOVER: std::sync::Mutex<Option<Src>> = std::sync::Mutex::new(None);

/// 리셋 블록으로 끝난 스트림이 이어 읽기를 기다리는지(물으면 그대로 둠: open 이 가져감)
pub fn reset_pending() -> bool {
    HANDOVER.lock().unwrap().is_some()
}

/// --slam: 같은 프로세스 위치 프로그램을 맡겨 둔다(다음 open 이 가져감)
pub fn hand_slam(side: SlamSide) {
    *HANDOVER.lock().unwrap() = Some(Src::Slam(Box::new(side)));
}

/// 기다리지 않는 읽기의 결과
pub enum Next {
    Frame(StreamFrame),
    /// 아직 없음(같은 프로세스 큐가 비었음: 넣을 때 루프를 깨움)
    Wait,
    /// 스트림 끝 또는 리셋 블록
    End,
}

pub struct StreamReader {
    /// None = 리셋 블록을 만나 HANDOVER 에 넘김(이 읽기는 끝)
    src: Option<Src>,
    graph: Vec<Node>,
}

impl StreamReader {
    /// "-" = 표준입력, 그 밖은 파일·이름 있는 파이프(hand_slam 으로 맡긴 것이 있으면 그것)
    pub fn open(path: &str) -> StreamReader {
        if let Some(src) = HANDOVER.lock().unwrap().take() {
            return StreamReader { src: Some(src), graph: vec![] };
        }
        let r: Box<dyn std::io::Read + Send> = if path == "-" {
            Box::new(std::io::BufReader::with_capacity(READ_BUF, std::io::stdin()))
        } else {
            Box::new(std::io::BufReader::with_capacity(READ_BUF, std::fs::File::open(path).unwrap_or_else(|e| panic!("--stream {path}: {e}"))))
        };
        StreamReader { src: Some(Src::Bytes(slam::stream::Reader::new(r))), graph: vec![] }
    }

    /// 스트림이 끝난 뒤 남은 그래프 노드(마지막 장면 뒤에 온 블록)
    pub fn take_graph(&mut self) -> Vec<Node> {
        std::mem::take(&mut self.graph)
    }

    /// 다음 장면(그 앞의 그래프 블록은 graph 에 담음). 스트림이 끝나면(보내는 쪽이 닫음) 또는 리셋 블록이면 None.
    /// 메모리 큐와 바이트 스트림이 같은 값(위치·보정값은 같은 f64, 컬러·깊이 버퍼는 복사 없이 넘겨받음)
    pub fn next(&mut self) -> Option<StreamFrame> {
        loop {
            match self.step(true) {
                Next::Frame(f) => return Some(f),
                Next::End => return None,
                Next::Wait => {}
            }
        }
    }

    /// 기다리지 않는 다음 장면(처리 루프): 같은 프로세스 위치 프로그램이 아직 장면을 못 냈으면 Wait. 바이트 스트림(파일 · 파이프)은 읽힐 때까지 읽는다
    pub fn try_next(&mut self) -> Next {
        self.step(false)
    }

    /// 실시간 위치 추적을 장면마다(처리 루프가 깰 때마다 부름)
    pub fn service(&mut self) {
        if let Some(Src::Slam(s)) = self.src.as_mut() {
            s.service();
        }
    }

    /// IMU 모드 장면 사이 위치(같은 프로세스 위치 프로그램)
    pub fn fast_pose(&self) -> Option<(f64, f64, slam::register::Tf)> {
        match self.src.as_ref() {
            Some(Src::Slam(s)) => s.fast_pose(),
            _ => None,
        }
    }

    /// 찍은 속도 재생에서 다음 장면 때까지 남은 시간
    pub fn due(&self) -> Option<std::time::Duration> {
        match self.src.as_ref() {
            Some(Src::Slam(s)) => s.due(),
            _ => None,
        }
    }

    fn step(&mut self, block: bool) -> Next {
        loop {
            let Some(src) = self.src.as_mut() else { return Next::End };
            let got = match src {
                Src::Slam(s) => match s.next_item(block) {
                    Ok(f) => f,
                    Err(()) if block => continue,
                    Err(()) => return Next::Wait,
                },
                Src::Bytes(r) => r.recv(),
            };
            let Some(mut f) = got else { return Next::End };
            if f.reset {
                // 리셋 블록: 여기서 스트림을 끝내고, 읽던 곳을 다음 실행에 넘긴다
                self.graph.clear();
                *HANDOVER.lock().unwrap() = self.src.take();
                return Next::End;
            }
            if f.is_graph() {
                for n in &f.graph {
                    let (r, p) = pose_of(&n.t);
                    self.graph.push((n.id, r, p, n.held == 1));
                }
                continue;
            }
            if !f.color_is_rgb {
                for px in f.color.chunks_exact_mut(3) {
                    px.swap(0, 2);
                }
            }
            let (r, p) = pose_of(&f.pose);
            return Next::Frame(StreamFrame {
                flags: f.flags,
                map_id: f.map_id,
                seq: f.seq,
                t: f.t,
                r,
                p,
                k: [f.fx, f.fy, f.cx, f.cy],
                w: f.width as usize,
                h: f.height as usize,
                depth_per_m: f.depth_per_m,
                rgb: f.color,
                depth: f.depth,
                anchor: f.anchor,
                graph: std::mem::take(&mut self.graph),
            });
        }
    }
}
