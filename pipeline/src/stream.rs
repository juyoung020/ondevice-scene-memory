//! 장면 입력(--slam · --stream): 위치 프로그램이 보내는 장면(위치·컬러·깊이)과 그 사이에 끼어 오는 그래프 블록(새로 생기거나 움직인 지도 노드 위치).
//! 같은 프로세스의 위치 프로그램(--slam)은 메모리 큐로, 녹화·파이프(--stream)는 바이트 스트림으로 온다. 형식·읽기는 `slam::stream` 하나를 쓰고,
//! 여기서는 둘 다 같은 함수로 `StreamFrame` 을 만든다. 형식과 좌표 약속은 docs/bridge.md "스트림 형식".
use crate::geom::*;
pub use slam::stream::{HELD, LOST, MAP_CORRECTED, NEW_MAP};
use slam::stream::OutFrame;

/// 읽기 버퍼: 머리·그래프 블록용. 사진은 이보다 커서 버퍼를 거치지 않고 받을 자리에 바로 읽힌다
const READ_BUF: usize = 1 << 16;

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
    pub depth: Vec<u16>,
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
            slam::bufpool::give_depth(std::mem::take(&mut self.depth));
        }
    }
}

impl StreamFrame {
    /// 위치가 없는 장면(맡아 둘 장면은 위치가 있으니 아님)
    pub fn lost(&self) -> bool {
        self.flags & LOST != 0 && !self.held()
    }
    pub fn held(&self) -> bool {
        self.flags & HELD != 0
    }
}

fn pose_of(t3x4: &[f64; 12]) -> (M3, V3) {
    ([[t3x4[0], t3x4[1], t3x4[2]], [t3x4[4], t3x4[5], t3x4[6]], [t3x4[8], t3x4[9], t3x4[10]]], [t3x4[3], t3x4[7], t3x4[11]])
}

/// 장면이 오는 곳: 같은 프로세스의 위치 프로그램(메모리 큐) 또는 바이트 스트림(파이프·파일)
enum Src {
    Mem(slam::stream::Receiver),
    Bytes(slam::stream::Reader<Box<dyn std::io::Read + Send>>),
}

impl Src {
    fn recv(&mut self) -> Option<OutFrame> {
        match self {
            Src::Mem(rx) => rx.recv(),
            Src::Bytes(r) => r.recv(),
        }
    }
}

/// 리셋 블록을 만난 읽기 쪽이 여기 맡겨 두면 다음 open 이 이어서 읽는다(같은 파이프, 버퍼에 남은 바이트 포함)
static HANDOVER: std::sync::Mutex<Option<Src>> = std::sync::Mutex::new(None);

/// 리셋 블록으로 끝난 스트림이 이어 읽기를 기다리는지(물으면 그대로 둠: open 이 가져감)
pub fn reset_pending() -> bool {
    HANDOVER.lock().unwrap().is_some()
}

/// --slam: 같은 프로세스에서 도는 위치 프로그램의 받는 쪽을 맡겨 둔다(다음 open 이 가져감)
pub fn hand_slam(rx: slam::stream::Receiver) {
    *HANDOVER.lock().unwrap() = Some(Src::Mem(rx));
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
            let mut f = self.src.as_mut()?.recv()?;
            if f.reset {
                // 리셋 블록: 여기서 스트림을 끝내고, 읽던 곳을 다음 실행에 넘긴다
                self.graph.clear();
                *HANDOVER.lock().unwrap() = self.src.take();
                return None;
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
            return Some(StreamFrame {
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
