//! 보드 흉내용 분할 대역(`--seg-cache <폴더>`): 장면별 분할 결과를 파일로 남겨 두고, 있으면 모델 대신 그 파일을 읽어 쓴다.
//! 보드는 추론을 NPU 가 맡아 CPU 는 전·후처리만 쓴다. PC 의 CPU 분할(ORT)은 memmap 몫 코어를 추론에 쓰므로,
//! 코어 2 개 흉내에서는 추론을 읽어 오기로 바꾸고 전·후처리 몫만 `--seg-cpu-ms` 만큼 읽기 스레드가 코어를 붙잡는다.
//! 같은 장면은 늘 같은 검출이 나와 실행끼리 물체 수를 견주기도 쉽다.
use crate::seg::{Det, Segmenter};
use std::path::{Path, PathBuf};

type Make = Box<dyn FnOnce() -> Box<dyn Segmenter + Send> + Send>;

pub struct CachedSeg {
    make: Option<Make>, // 파일이 없는 장면이 처음 나올 때 진짜 모델을 만든다(모두 있으면 모델을 올리지 않음)
    inner: Option<Box<dyn Segmenter + Send>>,
    dir: PathBuf,
    cpu_ms: f64,
    hits: u32,
    misses: u32,
}

impl CachedSeg {
    pub fn new(dir: &Path, cpu_ms: f64, make: Make) -> CachedSeg {
        std::fs::create_dir_all(dir).expect("--seg-cache 폴더");
        CachedSeg { make: Some(make), inner: None, dir: dir.to_path_buf(), cpu_ms, hits: 0, misses: 0 }
    }
}

/// 장면 열쇠: 화소 5 개마다 하나 + 크기 + 확신도(FNV-1a)
fn key(rgb: &[u8], w: usize, h: usize, conf: f32) -> u64 {
    let mut x = 0xcbf29ce484222325u64;
    let mut eat = |b: u8| x = (x ^ b as u64).wrapping_mul(0x100000001b3);
    for b in (w as u32).to_le_bytes().into_iter().chain((h as u32).to_le_bytes()).chain(conf.to_le_bytes()) {
        eat(b);
    }
    for &b in rgb.iter().step_by(5) {
        eat(b);
    }
    x
}

fn pack(dets: &[Det]) -> Vec<u8> {
    let mut b = (dets.len() as u32).to_le_bytes().to_vec();
    for d in dets {
        b.extend_from_slice(&(d.cls as u32).to_le_bytes());
        b.extend_from_slice(&d.score.to_le_bytes());
        for v in d.box_ {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend(d.mask.iter().map(|&m| m as u8));
    }
    lz4_flex::compress_prepend_size(&b)
}

fn unpack(z: &[u8]) -> Option<Vec<Det>> {
    let b = lz4_flex::decompress_size_prepended(z).ok()?;
    let u32_at = |i: usize| Some(u32::from_le_bytes(b.get(i..i + 4)?.try_into().ok()?));
    let (n, mut i) = (u32_at(0)? as usize, 4);
    let mut dets = Vec::with_capacity(n);
    for _ in 0..n {
        let (cls, score) = (u32_at(i)? as usize, f32::from_bits(u32_at(i + 4)?));
        let mut box_ = [0i32; 4];
        for (j, v) in box_.iter_mut().enumerate() {
            *v = u32_at(i + 8 + j * 4)? as i32;
        }
        i += 24;
        let len = ((box_[2] - box_[0]) * (box_[3] - box_[1])) as usize;
        let mask = b.get(i..i + len)?.iter().map(|&m| m != 0).collect();
        i += len;
        dets.push(Det { cls, score, box_, mask });
    }
    Some(dets)
}

impl Segmenter for CachedSeg {
    fn run(&mut self, rgb: &[u8], w: usize, h: usize, conf: f32) -> Vec<Det> {
        let c0 = std::time::Instant::now();
        let path = self.dir.join(format!("{:016x}.seg", key(rgb, w, h, conf)));
        if let Some(dets) = std::fs::read(&path).ok().and_then(|z| unpack(&z)) {
            self.hits += 1;
            // 전·후처리 몫: 잠들지 않고 코어를 붙잡음(보드 시간 그대로, --slow 와 무관)
            while c0.elapsed().as_secs_f64() * 1e3 < self.cpu_ms {
                std::hint::spin_loop();
            }
            return dets;
        }
        if let Some(make) = self.make.take() {
            self.inner = Some(make());
        }
        let dets = self.inner.as_mut().expect("분할 모델").run(rgb, w, h, conf);
        self.misses += 1;
        // 여러 실행이 같은 폴더를 써도 반쯤 쓴 파일을 읽지 않게 임시 이름으로 쓴 뒤 바꿈
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::write(&tmp, pack(&dets)).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
        dets
    }
}

impl Drop for CachedSeg {
    fn drop(&mut self) {
        eprintln!("분할 대역(--seg-cache): 읽어 옴 {} 번, 모델로 돌려 남김 {} 번", self.hits, self.misses);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        let dets = vec![
            Det { cls: 56, score: 0.81, box_: [3, 4, 8, 6], mask: (0..10).map(|i| i % 3 == 0).collect() },
            Det { cls: 0, score: 0.4, box_: [0, 0, 1, 1], mask: vec![true] },
        ];
        let back = unpack(&pack(&dets)).unwrap();
        assert_eq!(back.len(), 2);
        for (a, b) in dets.iter().zip(&back) {
            assert_eq!((a.cls, a.score, a.box_, &a.mask), (b.cls, b.score, b.box_, &b.mask));
        }
        assert!(unpack(&[1, 2, 3]).is_none());
    }
}
