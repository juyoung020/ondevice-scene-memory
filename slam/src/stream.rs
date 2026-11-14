//! 장면 스트림 쓰기(형식: docs/bridge.md "스트림 형식"). 쓰기는 따로 스레드가 맡아 위치 추적이 memmap 을 기다리지 않는다.
//! 원본: bridge/src/stream_writer.cpp. 바이트까지 같게 쓴다(리틀 엔디언).
use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

pub const HEADER_BYTES: usize = 176;
/// 위치 놓침: 위치 칸은 의미 없음
pub const LOST: u32 = 1;
/// 이 장면에서 새 지도를 시작함
pub const NEW_MAP: u32 = 2;
/// 이 장면에서 지도 보정(루프 클로저·최적화)이 바뀌어 위치가 뛸 수 있음
pub const MAP_CORRECTED: u32 = 4;
/// 맡아 둘 장면: 믿는 지도와 아직 안 이어진 새 지도의 위치·기준 노드가 들어 있음(LOST 와 함께 켬)
pub const HELD: u32 = 8;
/// 그래프 블록 MMG1 머리: 표시 4 + 머리 크기 4 + 노드 수 4 + 예약 4
pub const GRAPH_HEADER_BYTES: usize = 16;
pub const IDENTITY: [f64; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

/// 지도 노드 위치(OpenCV 카메라 → 세계, 장면 위치와 같은 좌표)
#[derive(Clone, Debug, PartialEq)]
pub struct GraphNode {
    pub id: u32,
    /// 1 = 믿는 지도와 아직 안 이어진 새 지도의 노드
    pub held: u32,
    pub t: [f64; 12],
}

#[derive(Clone)]
pub struct OutFrame {
    pub flags: u32,
    pub map_id: u32,
    pub seq: u64,
    pub t: f64,
    /// 3×4 행 우선: OpenCV 카메라 → 세계
    pub pose: [f64; 12],
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub depth_per_m: f32,
    pub width: u32,
    pub height: u32,
    /// 컬러 3 채널: color_is_rgb 이면 그대로, 아니면(BGR) 쓸 때 RGB 로 바꿈
    pub color: Vec<u8>,
    pub color_is_rgb: bool,
    pub depth: Vec<u16>,
    /// 기준 지도 노드 id(0 = 없음): 장면 위치 = 노드 위치 × 상대 위치
    pub anchor: u32,
    /// 비어 있지 않으면 장면이 아니라 그래프 블록(MMG1): 새로 생기거나 움직인 노드. 큐가 차도 버리지 않음
    pub graph: Vec<GraphNode>,
    /// 리셋 블록(노드 0 개 MMG1, 예약 칸 1): 받는 쪽이 지도를 비우고 처음부터 다시 쌓는다
    pub reset: bool,
}

impl Default for OutFrame {
    fn default() -> Self {
        OutFrame {
            flags: 0,
            map_id: 0,
            seq: 0,
            t: 0.0,
            pose: IDENTITY,
            fx: 0.0,
            fy: 0.0,
            cx: 0.0,
            cy: 0.0,
            depth_per_m: 1000.0,
            width: 0,
            height: 0,
            color: vec![],
            color_is_rgb: false,
            depth: vec![],
            anchor: 0,
            graph: vec![],
            reset: false,
        }
    }
}

impl OutFrame {
    pub fn is_graph(&self) -> bool {
        self.reset || !self.graph.is_empty()
    }
}

/// 그래프 블록(MMG1) 바이트: 머리 16 + 노드마다 id u32 + 맡아 둠 u32 + 3×4 f64
pub fn encode_graph(nodes: &[GraphNode], reset: bool) -> Vec<u8> {
    let mut b = Vec::with_capacity(GRAPH_HEADER_BYTES + nodes.len() * 104);
    b.extend_from_slice(b"MMG1");
    b.extend_from_slice(&(GRAPH_HEADER_BYTES as u32).to_le_bytes());
    b.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
    b.extend_from_slice(&(reset as u32).to_le_bytes());
    for n in nodes {
        b.extend_from_slice(&n.id.to_le_bytes());
        b.extend_from_slice(&n.held.to_le_bytes());
        for v in n.t {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    b
}

/// 머리 176 바이트
pub fn encode_header(f: &OutFrame) -> [u8; HEADER_BYTES] {
    let mut o = [0u8; HEADER_BYTES];
    let mut put = |off: usize, b: &[u8]| o[off..off + b.len()].copy_from_slice(b);
    put(0, b"MMF1");
    put(4, &(HEADER_BYTES as u32).to_le_bytes());
    put(8, &f.flags.to_le_bytes());
    put(12, &f.map_id.to_le_bytes());
    put(16, &f.seq.to_le_bytes());
    put(24, &f.t.to_le_bytes());
    for i in 0..12 {
        put(32 + 8 * i, &f.pose[i].to_le_bytes());
    }
    put(128, &f.fx.to_le_bytes());
    put(136, &f.fy.to_le_bytes());
    put(144, &f.cx.to_le_bytes());
    put(152, &f.cy.to_le_bytes());
    put(160, &f.width.to_le_bytes());
    put(164, &f.height.to_le_bytes());
    put(168, &f.depth_per_m.to_le_bytes());
    put(172, &f.anchor.to_le_bytes());
    o
}

struct Q {
    q: VecDeque<OutFrame>,
    done: bool,
    broken: bool,
    written: usize,
    dropped: usize,
    write_ms: Vec<f64>,
}

pub struct Writer {
    sh: Arc<(Mutex<Q>, Condvar)>,
    cap: usize,
    block: bool,
    th: Option<std::thread::JoinHandle<()>>,
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
}

impl Writer {
    /// out: 파일·이름 있는 파이프(FIFO) 경로. fd 가 있으면 경로 대신 그것(표준출력을 떼어 둔 것)에 씀
    /// block = true: 큐가 차면 기다림(재생 검증용, 장면을 빠짐없이). false: 가장 오래된 장면을 버림(카메라용)
    pub fn new(path: &str, fd: Option<std::fs::File>, queue: usize, block: bool) -> Result<Writer, String> {
        let file = match fd {
            Some(f) => f,
            None => std::fs::File::create(path).map_err(|e| format!("스트림 열기 실패: {path}: {e}"))?,
        };
        // 파이프면 버퍼를 1 MB 로(기본 64 kB): 장면 하나(약 1.5 MB)의 시스템 호출·스레드 전환이 준다. 파일이면 실패하고 넘어감
        #[cfg(target_os = "linux")]
        unsafe {
            use std::os::fd::AsRawFd;
            fcntl(file.as_raw_fd(), 1031 /* F_SETPIPE_SZ */, 1i32 << 20);
        }
        let sh = Arc::new((Mutex::new(Q { q: VecDeque::new(), done: false, broken: false, written: 0, dropped: 0, write_ms: vec![] }), Condvar::new()));
        let sh2 = sh.clone();
        let th = std::thread::spawn(move || run(sh2, file));
        Ok(Writer { sh, cap: queue.max(1), block, th: Some(th) })
    }

    /// 같은 프로세스로 넘기기: 쓰기 스레드·인코딩 없이 받는 쪽(Receiver)이 이 큐에서 장면을 그대로 꺼낸다.
    /// 큐 길이·기다림/버림·그래프 블록을 버리지 않는 규칙은 파이프와 같다
    pub fn channel(queue: usize, block: bool) -> (Writer, Receiver) {
        let sh = Arc::new((Mutex::new(Q { q: VecDeque::new(), done: false, broken: false, written: 0, dropped: 0, write_ms: vec![] }), Condvar::new()));
        (Writer { sh: sh.clone(), cap: queue.max(1), block, th: None }, Receiver { sh })
    }

    /// 받는 쪽이 닫혔으면 false
    pub fn push(&self, f: OutFrame) -> bool {
        let (m, cv) = &*self.sh;
        let mut s = m.lock().unwrap();
        if s.broken {
            return false;
        }
        // 그래프 블록은 큐 길이와 상관없이 넣는다(노드 위치를 잃으면 memmap 이 장면을 고칠 수 없음)
        if !f.is_graph() && s.q.len() >= self.cap {
            if self.block {
                s = cv.wait_while(s, |s| s.q.len() >= self.cap && !s.broken).unwrap();
                if s.broken {
                    return false;
                }
            } else if let Some(i) = s.q.iter().position(|x| !x.is_graph()) {
                // 가장 오래된 장면을 버림(그래프 블록은 남김)
                s.q.remove(i);
                s.dropped += 1;
            }
        }
        s.q.push_back(f);
        cv.notify_all();
        true
    }

    pub fn close(&mut self) {
        {
            let (m, cv) = &*self.sh;
            m.lock().unwrap().done = true;
            cv.notify_all();
        }
        if let Some(th) = self.th.take() {
            let _ = th.join();
        }
    }
    pub fn written(&self) -> usize {
        self.sh.0.lock().unwrap().written
    }
    pub fn dropped(&self) -> usize {
        self.sh.0.lock().unwrap().dropped
    }
    /// 장면마다 쓰기 시간(ms): RGB 바꾸기 + 파이프 쓰기(받는 쪽이 늦으면 기다린 시간 포함)
    pub fn write_ms(&self) -> Vec<f64> {
        self.sh.0.lock().unwrap().write_ms.clone()
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        self.close();
    }
}

/// 같은 프로세스의 받는 쪽(Writer::channel): 장면·그래프·리셋 블록을 보낸 차례 그대로 꺼낸다
pub struct Receiver {
    sh: Arc<(Mutex<Q>, Condvar)>,
}

impl Receiver {
    /// 다음 항목(장면이면 컬러는 보낸 그대로: color_is_rgb 를 보고 받는 쪽이 RGB 로 맞춘다). 보내는 쪽이 끝냈고 다 꺼냈으면 None
    pub fn recv(&self) -> Option<OutFrame> {
        let (m, cv) = &*self.sh;
        let mut s = cv.wait_while(m.lock().unwrap(), |s| s.q.is_empty() && !s.done).unwrap();
        let f = s.q.pop_front()?;
        if !f.is_graph() {
            s.written += 1;
        }
        cv.notify_all();
        Some(f)
    }
}

impl Drop for Receiver {
    fn drop(&mut self) {
        // 받는 쪽이 그만둠 = 파이프가 닫힘과 같음: 보내는 쪽 push 가 false
        let (m, cv) = &*self.sh;
        m.lock().unwrap().broken = true;
        cv.notify_all();
    }
}

fn as_bytes(d: &[u16]) -> &[u8] {
    // 리틀 엔디언 기계에서 u16 배열은 스트림 바이트 그대로
    #[cfg(not(target_endian = "little"))]
    compile_error!("스트림은 리틀 엔디언 기계 기준");
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 2) }
}

fn run(sh: Arc<(Mutex<Q>, Condvar)>, mut file: std::fs::File) {
    let (m, cv) = &*sh;
    loop {
        let mut f = {
            let s = cv.wait_while(m.lock().unwrap(), |s| s.q.is_empty() && !s.done).unwrap();
            let mut s = s;
            let Some(f) = s.q.pop_front() else { return }; // 끝났고 다 썼음
            cv.notify_all();
            f
        };
        if f.is_graph() {
            if file.write_all(&encode_graph(&f.graph, f.reset)).is_err() {
                m.lock().unwrap().broken = true;
                cv.notify_all();
                return;
            }
            continue;
        }
        let c0 = Instant::now();
        // BGR 이면 받은 버퍼 안에서 바로 RGB 로(복사 없음)
        if !f.color_is_rgb {
            for p in f.color.chunks_exact_mut(3) {
                p.swap(0, 2);
            }
        }
        let ok = file.write_all(&encode_header(&f)).is_ok() && file.write_all(&f.color).is_ok() && file.write_all(as_bytes(&f.depth)).is_ok();
        let ms = c0.elapsed().as_secs_f64() * 1e3;
        let mut s = m.lock().unwrap();
        if !ok {
            s.broken = true;
            cv.notify_all();
            return;
        }
        s.written += 1;
        s.write_ms.push(ms);
    }
}

/// 시험용 스트림 읽기(pipeline/src/stream.rs 의 읽기와 같은 해석): 장면 머리 + 그래프 블록을 차례대로
#[cfg(test)]
pub(crate) mod parse {
    pub enum Item {
        Frame { head: [u8; 176], rgb_off: usize, depth_off: usize, w: usize, h: usize },
        Graph { nodes: Vec<super::GraphNode>, reset: bool },
    }
    pub fn items(b: &[u8]) -> Vec<Item> {
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let (mut o, mut out) = (0usize, vec![]);
        while o < b.len() {
            match &b[o..o + 4] {
                b"MMF1" => {
                    assert_eq!(u32_at(o + 4), 176);
                    let (w, h) = (u32_at(o + 160) as usize, u32_at(o + 164) as usize);
                    out.push(Item::Frame { head: b[o..o + 176].try_into().unwrap(), rgb_off: o + 176, depth_off: o + 176 + w * h * 3, w, h });
                    o += 176 + w * h * 5;
                }
                b"MMG1" => {
                    assert_eq!(u32_at(o + 4), 16);
                    let n = u32_at(o + 8) as usize;
                    let reset = u32_at(o + 12) == 1;
                    let mut nodes = vec![];
                    for k in 0..n {
                        let c = o + 16 + 104 * k;
                        let t: Vec<f64> = (0..12).map(|i| f64::from_le_bytes(b[c + 8 + 8 * i..c + 16 + 8 * i].try_into().unwrap())).collect();
                        nodes.push(super::GraphNode { id: u32_at(c), held: u32_at(c + 4), t: t.try_into().unwrap() });
                    }
                    out.push(Item::Graph { nodes, reset });
                    o += 16 + 104 * n;
                }
                m => panic!("스트림 형식이 아님 {m:?}"),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::parse::*;
    use super::*;

    fn tmp(name: &str) -> String {
        std::env::temp_dir().join(format!("slam_stream_{}_{name}", std::process::id())).to_string_lossy().into()
    }

    #[test]
    fn round_trip() {
        let p = tmp("rt");
        let mut w = Writer::new(&p, None, 2, true).unwrap();
        w.push(OutFrame { reset: true, ..Default::default() });
        w.push(OutFrame { graph: vec![GraphNode { id: 7, held: 1, t: IDENTITY }], ..Default::default() });
        let f = OutFrame { flags: LOST | HELD, map_id: 3, seq: 9, t: 1.5, width: 2, height: 1, color: vec![1, 2, 3, 4, 5, 6], depth: vec![10, 20], depth_per_m: 5000.0, anchor: 7, ..Default::default() };
        w.push(f.clone());
        w.close();
        let b = std::fs::read(&p).unwrap();
        let it = items(&b);
        assert!(matches!(&it[0], Item::Graph { nodes, reset: true } if nodes.is_empty()));
        assert!(matches!(&it[1], Item::Graph { nodes, reset: false } if nodes[0].id == 7 && nodes[0].held == 1));
        let Item::Frame { head, rgb_off, depth_off, .. } = &it[2] else { panic!() };
        assert_eq!(head, &encode_header(&f));
        assert_eq!(&b[*rgb_off..*rgb_off + 6], &[3, 2, 1, 6, 5, 4]); // BGR → RGB
        assert_eq!(&b[*depth_off..], &[10, 0, 20, 0]);
        let _ = std::fs::remove_file(p);
    }

    /// C++ rtab_bridge 스트림과 바이트 비교: SLAM_CPP_STREAM=<C++ --out 파일> SLAM_IO_TUM=<같은 TUM 폴더>(--rate 0, --send-every 1).
    /// C++ 스트림의 위치·표시·그래프 블록에 TumSource 장면을 붙여 이 Writer 로 다시 쓰고 파일 전체를 비교
    #[test]
    #[ignore]
    fn match_cpp_stream() {
        use crate::io::FrameSource;
        let (Ok(cs), Ok(dir)) = (std::env::var("SLAM_CPP_STREAM"), std::env::var("SLAM_IO_TUM")) else { return };
        let cpp = std::fs::read(&cs).unwrap();
        let mut src = crate::io::TumSource::new(&dir, 0.0).unwrap();
        let p = tmp("cpp");
        let mut w = Writer::new(&p, None, 4, true).unwrap();
        let (mut nf, mut ng, mut next_seq) = (0usize, 0usize, 0u64);
        let c0 = Instant::now();
        for it in items(&cpp) {
            match it {
                Item::Graph { nodes, reset } => {
                    ng += 1;
                    w.push(OutFrame { graph: nodes, reset, ..Default::default() });
                }
                Item::Frame { head, w: fw, h: fh, .. } => {
                    let u32_at = |o: usize| u32::from_le_bytes(head[o..o + 4].try_into().unwrap());
                    let f64_at = |o: usize| f64::from_le_bytes(head[o..o + 8].try_into().unwrap());
                    let seq = u64::from_le_bytes(head[16..24].try_into().unwrap());
                    let mut fr = src.next().unwrap().unwrap();
                    // 리셋 뒤에도 seq 는 장면 순번 그대로
                    while next_seq < seq {
                        fr = src.next().unwrap().unwrap();
                        next_seq += 1;
                    }
                    next_seq += 1;
                    assert_eq!((fr.cam.width as usize, fr.cam.height as usize), (fw, fh));
                    let mut pose = [0.0; 12];
                    for i in 0..12 {
                        pose[i] = f64_at(32 + 8 * i);
                    }
                    w.push(OutFrame {
                        flags: u32_at(8),
                        map_id: u32_at(12),
                        seq,
                        t: fr.t,
                        pose,
                        fx: fr.cam.fx,
                        fy: fr.cam.fy,
                        cx: fr.cam.cx,
                        cy: fr.cam.cy,
                        depth_per_m: fr.depth_per_m,
                        width: fr.cam.width,
                        height: fr.cam.height,
                        color: fr.color,
                        color_is_rgb: fr.color_is_rgb,
                        depth: fr.depth_raw,
                        anchor: u32_at(172),
                        graph: vec![],
                        reset: false,
                    });
                    nf += 1;
                }
            }
        }
        w.close();
        let mine = std::fs::read(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        eprintln!("장면 {nf} 개, 그래프 블록 {ng} 개, {} 바이트, {:.1} 초", cpp.len(), c0.elapsed().as_secs_f64());
        assert_eq!(mine.len(), cpp.len());
        if let Some(i) = mine.iter().zip(&cpp).position(|(a, b)| a != b) {
            panic!("바이트 {i} 에서 다름");
        }
    }
}
