//! 장면 스트림 입력(--stream): 연결 프로그램(bridge/, RTAB-Map)이 보내는 장면 = 머리 176 바이트 + RGB8 + 깊이 u16,
//! 그 사이에 그래프 블록(MMG1: 새로 생기거나 움직인 지도 노드 위치)이 끼어 온다.
//! 형식과 좌표 약속은 docs/bridge.md "스트림 형식".
use crate::geom::*;
use std::io::Read;

pub const HEADER: usize = 176;
pub const LOST: u32 = 1;
pub const NEW_MAP: u32 = 2;
pub const MAP_CORRECTED: u32 = 4;
/// 맡아 둘 장면(LOST 와 함께 옴): 믿는 지도와 아직 안 이어진 새 지도의 위치·기준 노드가 들어 있다
pub const HELD: u32 = 8;
/// 그래프 블록: 머리 16 바이트(표시 4 + 머리 크기 4 + 노드 수 4 + 리셋 표시 4), 노드마다 id u32 + 맡아 둠 u32 + 3×4 f64
const GRAPH_HEADER: usize = 16;
const GRAPH_NODE: usize = 8 + 96;
/// 읽기 버퍼: 머리·그래프 블록용. 사진은 이보다 커서 버퍼를 거치지 않고 받을 자리에 바로 읽힌다
const READ_BUF: usize = 1 << 16;

/// 그래프 노드: (RTAB-Map 노드 id, 자세, 위치, 맡아 둠) — OpenCV 카메라 → 세계, 장면 위치와 같은 좌표.
/// 맡아 둠 = 믿는 지도와 아직 안 이어진 새 지도의 노드(이어지면 같은 id 가 맡아 둠 없이 다시 온다)
pub type Node = (u32, M3, V3, bool);

pub struct StreamFrame {
    pub flags: u32,
    pub map_id: u32,
    pub seq: u64,
    pub t: f64,
    /// 카메라(OpenCV: x 오른쪽, y 아래, z 앞) → 세계(RTAB-Map 지도: x 앞, y 왼쪽, z 위)
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

impl StreamFrame {
    /// 위치가 없는 장면(맡아 둘 장면은 위치가 있으니 아님)
    pub fn lost(&self) -> bool {
        self.flags & LOST != 0 && !self.held()
    }
    pub fn held(&self) -> bool {
        self.flags & HELD != 0
    }
    /// 깊이(미터, 0 = 없음). TUM Kinect 입력(tum.rs)과 같은 계산(값 / 배율)
    pub fn depth_m(&self) -> Vec<f32> {
        self.depth.iter().map(|&v| v as f32 / self.depth_per_m).collect()
    }
}

fn pose_of(t3x4: &[f64]) -> (M3, V3) {
    ([[t3x4[0], t3x4[1], t3x4[2]], [t3x4[4], t3x4[5], t3x4[6]], [t3x4[8], t3x4[9], t3x4[10]]], [t3x4[3], t3x4[7], t3x4[11]])
}

/// 리셋 블록을 만난 읽기 쪽이 여기 맡겨 두면 다음 open 이 이어서 읽는다(같은 파이프, 버퍼에 남은 바이트 포함)
static HANDOVER: std::sync::Mutex<Option<Box<dyn Read + Send>>> = std::sync::Mutex::new(None);

/// 리셋 블록으로 끝난 스트림이 이어 읽기를 기다리는지(물으면 그대로 둠: open 이 가져감)
pub fn reset_pending() -> bool {
    HANDOVER.lock().unwrap().is_some()
}

pub struct StreamReader {
    /// None = 리셋 블록을 만나 HANDOVER 에 넘김(이 읽기는 끝)
    src: Option<Box<dyn Read + Send>>,
    graph: Vec<Node>,
}

impl StreamReader {
    /// "-" = 표준입력, 그 밖은 파일·이름 있는 파이프
    pub fn open(path: &str) -> StreamReader {
        if let Some(src) = HANDOVER.lock().unwrap().take() {
            return StreamReader { src: Some(src), graph: vec![] };
        }
        let src: Box<dyn Read + Send> = if path == "-" {
            Box::new(std::io::BufReader::with_capacity(READ_BUF, std::io::stdin()))
        } else {
            Box::new(std::io::BufReader::with_capacity(READ_BUF, std::fs::File::open(path).unwrap_or_else(|e| panic!("--stream {path}: {e}"))))
        };
        StreamReader { src: Some(src), graph: vec![] }
    }

    /// 스트림이 끝난 뒤 남은 그래프 노드(마지막 장면 뒤에 온 블록)
    pub fn take_graph(&mut self) -> Vec<Node> {
        std::mem::take(&mut self.graph)
    }

    /// 다음 장면(그 앞의 그래프 블록은 graph 에 담음). 스트림이 끝나면(보내는 쪽이 닫음) None.
    pub fn next(&mut self) -> Option<StreamFrame> {
        loop {
            let mut m = [0u8; 4];
            self.src.as_mut()?.read_exact(&mut m).ok()?;
            match &m {
                b"MMF1" => return self.frame(),
                b"MMG1" => self.graph_block()?,
                _ => panic!("스트림 형식이 아님(머리 {m:?})"),
            }
        }
    }

    fn graph_block(&mut self) -> Option<()> {
        let mut h = [0u8; GRAPH_HEADER - 4];
        self.src.as_mut()?.read_exact(&mut h).ok()?;
        let u32_at = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        assert_eq!(u32_at(0) as usize, GRAPH_HEADER, "모르는 그래프 블록 머리 크기");
        let n = u32_at(4) as usize;
        // 리셋 블록(예약 칸 1): 여기서 스트림을 끝내고, 읽던 곳을 다음 실행에 넘긴다
        if u32_at(8) == 1 {
            *HANDOVER.lock().unwrap() = self.src.take();
            self.graph.clear();
            return None;
        }
        let mut b = vec![0u8; n * GRAPH_NODE];
        self.src.as_mut()?.read_exact(&mut b).ok()?;
        for c in b.chunks_exact(GRAPH_NODE) {
            let id = u32::from_le_bytes(c[0..4].try_into().unwrap());
            let t: Vec<f64> = (0..12).map(|i| f64::from_le_bytes(c[8 + 8 * i..16 + 8 * i].try_into().unwrap())).collect();
            let (r, p) = pose_of(&t);
            let held = u32::from_le_bytes(c[4..8].try_into().unwrap()) == 1;
            self.graph.push((id, r, p, held));
        }
        Some(())
    }

    fn frame(&mut self) -> Option<StreamFrame> {
        let mut h = [0u8; HEADER];
        h[0..4].copy_from_slice(b"MMF1");
        self.src.as_mut()?.read_exact(&mut h[4..]).ok()?;
        let u32_at = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let f64_at = |o: usize| f64::from_le_bytes(h[o..o + 8].try_into().unwrap());
        assert_eq!(u32_at(4) as usize, HEADER, "모르는 머리 크기");
        let t3x4: Vec<f64> = (0..12).map(|i| f64_at(32 + 8 * i)).collect();
        let (r, p) = pose_of(&t3x4);
        let (w, hh) = (u32_at(160) as usize, u32_at(164) as usize);
        let mut rgb = vec![0u8; w * hh * 3];
        // 깊이는 리틀 엔디언 u16 이라 받을 자리에 바로 읽는다
        #[cfg(not(target_endian = "little"))]
        compile_error!("스트림 깊이를 바로 읽는 것은 리틀 엔디언 기계 기준");
        let mut depth = vec![0u16; w * hh];
        let db = unsafe { std::slice::from_raw_parts_mut(depth.as_mut_ptr() as *mut u8, w * hh * 2) };
        if self.src.as_mut()?.read_exact(&mut rgb).is_err() || self.src.as_mut()?.read_exact(db).is_err() {
            return None; // 장면 중간에 끊김
        }
        Some(StreamFrame {
            flags: u32_at(8),
            map_id: u32_at(12),
            seq: u64::from_le_bytes(h[16..24].try_into().unwrap()),
            t: f64_at(24),
            r,
            p,
            k: [f64_at(128), f64_at(136), f64_at(144), f64_at(152)],
            w,
            h: hh,
            depth_per_m: f32_at(168),
            rgb,
            depth,
            anchor: u32_at(172),
            graph: std::mem::take(&mut self.graph),
        })
    }
}
