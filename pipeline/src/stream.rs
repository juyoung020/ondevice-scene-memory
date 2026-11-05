//! 장면 스트림 입력(--stream): 연결 프로그램(bridge/, RTAB-Map)이 보내는 장면 = 머리 176 바이트 + RGB8 + 깊이 u16.
//! 형식과 좌표 약속은 docs/bridge.md "스트림 형식".
use crate::geom::*;
use std::io::Read;

pub const HEADER: usize = 176;
pub const LOST: u32 = 1;
pub const NEW_MAP: u32 = 2;
pub const MAP_CORRECTED: u32 = 4;

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
}

impl StreamFrame {
    pub fn lost(&self) -> bool {
        self.flags & LOST != 0
    }
    /// 깊이(미터, 0 = 없음). TUM Kinect 입력(tum.rs)과 같은 계산(값 / 배율)
    pub fn depth_m(&self) -> Vec<f32> {
        self.depth.iter().map(|&v| v as f32 / self.depth_per_m).collect()
    }
}

pub struct StreamReader {
    src: Box<dyn Read + Send>,
}

impl StreamReader {
    /// "-" = 표준입력, 그 밖은 파일·이름 있는 파이프
    pub fn open(path: &str) -> StreamReader {
        let src: Box<dyn Read + Send> = if path == "-" {
            Box::new(std::io::BufReader::with_capacity(1 << 22, std::io::stdin()))
        } else {
            Box::new(std::io::BufReader::with_capacity(1 << 22, std::fs::File::open(path).unwrap_or_else(|e| panic!("--stream {path}: {e}"))))
        };
        StreamReader { src }
    }

    /// 다음 장면. 스트림이 끝나면(보내는 쪽이 닫음) None.
    pub fn next(&mut self) -> Option<StreamFrame> {
        let mut h = [0u8; HEADER];
        if self.src.read_exact(&mut h).is_err() {
            return None;
        }
        assert_eq!(&h[0..4], b"MMF1", "스트림 형식이 아님(머리 MMF1 없음)");
        let u32_at = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let f64_at = |o: usize| f64::from_le_bytes(h[o..o + 8].try_into().unwrap());
        assert_eq!(u32_at(4) as usize, HEADER, "모르는 머리 크기");
        let t3x4: Vec<f64> = (0..12).map(|i| f64_at(32 + 8 * i)).collect();
        let r = [[t3x4[0], t3x4[1], t3x4[2]], [t3x4[4], t3x4[5], t3x4[6]], [t3x4[8], t3x4[9], t3x4[10]]];
        let p = [t3x4[3], t3x4[7], t3x4[11]];
        let (w, hh) = (u32_at(160) as usize, u32_at(164) as usize);
        let mut rgb = vec![0u8; w * hh * 3];
        let mut db = vec![0u8; w * hh * 2];
        if self.src.read_exact(&mut rgb).is_err() || self.src.read_exact(&mut db).is_err() {
            return None; // 장면 중간에 끊김
        }
        let depth = db.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
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
        })
    }
}
