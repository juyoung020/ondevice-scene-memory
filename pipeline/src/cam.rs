//! USB 카메라(V4L2, MJPEG) 입력. 받는 스레드가 계속 최신 JPEG 한 장만 들고 있어(오래된 프레임은 버림),
//! 처리가 느려도 항상 지금 화면을 쓴다. 해독은 가져갈 때만.
use std::sync::{Arc, Mutex};
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::FourCC;

pub struct Cam {
    latest: Arc<Mutex<(u64, Vec<u8>)>>,
    seen: u64,
    pub w: usize,
    pub h: usize,
}

impl Cam {
    pub fn open(path: &str, w: u32, h: u32) -> Cam {
        let dev = Box::leak(Box::new(Device::with_path(path).expect("camera")));
        let mut f = dev.format().expect("format");
        f.width = w;
        f.height = h;
        f.fourcc = FourCC::new(b"MJPG");
        let f = dev.set_format(&f).expect("set format");
        assert_eq!(&f.fourcc.repr, b"MJPG", "camera did not accept MJPG");
        let latest = Arc::new(Mutex::new((0u64, vec![])));
        let l2 = latest.clone();
        let mut stream = MmapStream::with_buffers(dev, Type::VideoCapture, 4).expect("stream");
        std::thread::spawn(move || {
            let mut n = 0u64;
            loop {
                let Ok((buf, meta)) = stream.next() else { continue };
                n += 1;
                let used = (meta.bytesused as usize).min(buf.len());
                *l2.lock().unwrap() = (n, buf[..used].to_vec());
            }
        });
        Cam { latest, seen: 0, w: f.width as usize, h: f.height as usize }
    }

    /// 새 프레임이 올 때까지 기다렸다 RGB8 로 해독
    pub fn grab(&mut self) -> Vec<u8> {
        loop {
            let (n, jpg) = {
                let g = self.latest.lock().unwrap();
                (g.0, if g.0 > self.seen { g.1.clone() } else { vec![] })
            };
            if n > self.seen && !jpg.is_empty() {
                self.seen = n;
                if let Ok(img) = image::load_from_memory_with_format(&jpg, image::ImageFormat::Jpeg) {
                    let img = img.into_rgb8();
                    if img.width() as usize == self.w && img.height() as usize == self.h {
                        return img.into_raw();
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
}
