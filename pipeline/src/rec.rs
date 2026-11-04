//! 웹캠 녹화와 재생. 녹화 폴더 = `rgb.txt`(시각 경로) + `rgb/<시각>.jpg`(카메라가 준 JPEG 그대로).
//! `--cam <장치> --rec <폴더>` 로 찍고, `--cam <폴더>` 로 같은 장면을 몇 번이고 다시 넣는다(PC·보드 모두).
#![cfg_attr(not(feature = "cam"), allow(dead_code))]
use std::path::{Path, PathBuf};

pub struct Recorder {
    dir: PathBuf,
    list: String,
}

impl Recorder {
    pub fn new(dir: &Path) -> Recorder {
        std::fs::create_dir_all(dir.join("rgb")).expect("rec dir");
        Recorder { dir: dir.to_path_buf(), list: "# timestamp filename\n".into() }
    }
    pub fn add(&mut self, t: f64, jpg: &[u8]) {
        let rel = format!("rgb/{t:.6}.jpg");
        crate::out::write_atomic(&self.dir.join(&rel), jpg);
        self.list += &format!("{t:.6} {rel}\n");
        crate::out::write_atomic(&self.dir.join("rgb.txt"), self.list.as_bytes());
    }
}

pub struct Player {
    dir: PathBuf,
    frames: Vec<(f64, String)>,
    i: usize,
    pub w: usize,
    pub h: usize,
}

impl Player {
    /// 폴더에 rgb.txt 가 있으면 녹화로 본다
    pub fn open(path: &str) -> Option<Player> {
        let dir = PathBuf::from(path);
        let txt = std::fs::read_to_string(dir.join("rgb.txt")).ok()?;
        let frames: Vec<(f64, String)> = txt
            .lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| {
                let mut s = l.split_whitespace();
                Some((s.next()?.parse().ok()?, s.next()?.to_string()))
            })
            .collect();
        let first = image::open(dir.join(&frames.first()?.1)).ok()?;
        Some(Player { w: first.width() as usize, h: first.height() as usize, dir, frames, i: 0 })
    }
    /// 다음 장면(찍은 시각, RGB8)
    pub fn next(&mut self) -> Option<(f64, Vec<u8>)> {
        let (t, rel) = self.frames.get(self.i)?.clone();
        self.i += 1;
        Some((t, image::open(self.dir.join(rel)).expect("rec frame").into_rgb8().into_raw()))
    }
}
