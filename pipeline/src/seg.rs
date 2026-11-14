//! YOLO26-seg, end-to-end 출력: [1,300,38] = x1 y1 x2 y2 점수 클래스 + 계수 32, 원형 [1,32,160,160].
//! 실행기 두 개가 같은 전처리(레터박스 RGB, 회색 114)·후처리(상자 안 화소만 계수·원형 내적)를 쓴다.
//!   - PC: ONNX Runtime (feature "onnx", models/yolo26s-seg.onnx, 입력 f32 NCHW /255)
//!   - Pi5 + DX-M1: DXRT C API (feature "dxrt", yolo26-s-seg_640x640.dxnn, 입력 u8 NHWC)

pub const COCO: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
];

pub struct Det {
    pub cls: usize,
    pub score: f32,
    pub box_: [i32; 4], // 영상 화소 x1 y1 x2 y2 (잘라 맞춤)
    pub mask: Vec<bool>, // 상자 크기 (x2-x1)*(y2-y1), 행 우선
}

/// 분할 모델 입력 한 변
/// 분할 모델이 내는 이름(`--seg-names`). 정하지 않으면 COCO 80 개
static NAMES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

/// 클래스 이름 목록을 바꾼다: 이름 없이 물체만 자르는 1 클래스 모델(ObjectSAM)은 `object`
pub fn set_names(list: &str) {
    let _ = NAMES.set(list.split(',').map(|s| s.trim().to_string()).collect());
}

/// 분할 모델이 낼 수 있는 이름 전체
pub fn names() -> Vec<String> {
    NAMES.get().cloned().unwrap_or_else(|| COCO.iter().map(|s| s.to_string()).collect())
}

/// 클래스 번호의 이름(목록 밖이면 object)
pub fn name(cls: usize) -> &'static str {
    match NAMES.get() {
        Some(n) => n.get(cls).map_or("object", |s| s.as_str()),
        None => COCO.get(cls).copied().unwrap_or("object"),
    }
}

pub const SEG_S: usize = 640;

/// 레터박스 자리: 배율 r, 여백 (px, py)
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct LbGeom {
    pub r: f32,
    pub px: usize,
    pub py: usize,
}

/// 레터박스: RGB(w×h) → buf(u8 NHWC s×s×3, 가까운 화소, 여백 회색 114). buf 는 다시 쓰는 버퍼(크기만 맞춤).
/// 배율이 1(640×480 → 640)이면 줄 단위 복사.
#[allow(dead_code)]
pub fn letterbox(rgb: &[u8], w: usize, h: usize, s: usize, buf: &mut Vec<u8>) -> LbGeom {
    letterbox_keep(rgb, w, h, s, buf, false)
}

/// `letterbox` 와 같고, bands_ok 이면(이 버퍼에 같은 w·h·s 로 이미 레터박스한 적이 있음) 회색 띠를 다시 칠하지 않는다(띠는 늘 같은 자리·같은 값)
pub fn letterbox_keep(rgb: &[u8], w: usize, h: usize, s: usize, buf: &mut Vec<u8>, bands_ok: bool) -> LbGeom {
    let r = (s as f32 / w as f32).min(s as f32 / h as f32);
    let (nw, nh) = ((w as f32 * r).round() as usize, (h as f32 * r).round() as usize);
    let (px, py) = ((s - nw) / 2, (s - nh) / 2);
    let bands_ok = bands_ok && buf.len() == s * s * 3;
    buf.resize(s * s * 3, 114);
    // 여백: 위·아래 띠, 줄마다 왼쪽·오른쪽
    if !bands_ok {
        buf[..py * s * 3].fill(114);
        buf[(py + nh) * s * 3..].fill(114);
    }
    let same = r == 1.0;
    // 가로 원본 위치는 줄마다 같으니 한 번만 계산
    let xs: Vec<usize> = if same { vec![] } else { (0..nw).map(|x| (((x as f32 + 0.5) / r) as usize).min(w - 1) * 3).collect() };
    for y in 0..nh {
        let row = &mut buf[(py + y) * s * 3..(py + y + 1) * s * 3];
        if !bands_ok {
            row[..px * 3].fill(114);
            row[(px + nw) * 3..].fill(114);
        }
        let dst = &mut row[px * 3..(px + nw) * 3];
        if same {
            dst.copy_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
        } else {
            let sy = (((y as f32 + 0.5) / r) as usize).min(h - 1);
            let src = &rgb[sy * w * 3..(sy + 1) * w * 3];
            for (d, &o) in dst.chunks_exact_mut(3).zip(&xs) {
                d.copy_from_slice(&src[o..o + 3]);
            }
        }
    }
    LbGeom { r, px, py }
}

/// u8 NHWC s×s×3 → f32 NCHW(값 / 255)(ONNX 실행기의 입력)
#[cfg(feature = "onnx")]
pub fn nchw_f32(buf: &[u8], s: usize, out: &mut Vec<f32>) {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    let lut = LUT.get_or_init(|| std::array::from_fn(|i| i as f32 / 255.0));
    out.resize(3 * s * s, 0.0);
    let (r, gb) = out.split_at_mut(s * s);
    let (g, b) = gb.split_at_mut(s * s);
    for (((p, r), g), b) in buf.chunks_exact(3).zip(r).zip(g).zip(b) {
        *r = lut[p[0] as usize];
        *g = lut[p[1] as usize];
        *b = lut[p[2] as usize];
    }
}

/// 원형 격자 칸 하나의 네 귀퉁이가 모두 이 범위면 그 칸 화소는 모두 물체(쌍선형 = 네 값의 볼록 조합이라 양수)
const INSIDE_MIN: f32 = 1e-20;
const INSIDE_MAX: f32 = 1e30;

/// 상자·마스크 해독(입력 좌표 → 영상 좌표). 모델이 end-to-end(겹친 상자 정리가 모델 안)라 점수 문턱만 본다.
/// 마스크 = 계수·원형 내적을 원형 격자(160²)에서 한 번 하고, 화소마다 쌍선형으로 읽은 값의 부호.
/// 격자 칸의 네 귀퉁이 부호가 같으면 그 칸 화소는 계산 없이 채우고, 부호가 섞인 칸(물체 가장자리)만 화소마다 계산한다.
#[allow(clippy::too_many_arguments)]
pub fn decode(det: &[f32], proto: &[f32], pc: usize, ph: usize, pw: usize, s: usize, lb: LbGeom, w: usize, h: usize, conf: f32) -> Vec<Det> {
    let (r, px, py) = (lb.r, lb.px as f32, lb.py as f32);
    let scale = ph as f32 / s as f32;
    let mut dets = vec![];
    // 상자마다 다시 쓰는 버퍼
    let (mut g, mut ys, mut xs): (Vec<f32>, Vec<(usize, usize, f32)>, Vec<(usize, usize, f32)>) = (vec![], vec![], vec![]);
    let mut runs: Vec<(usize, usize, usize, usize)> = vec![]; // 같은 격자 칸을 읽는 화소 구간: (왼 칸, 오른 칸, 화소 시작, 끝)
    let (mut tmpl, mut mixed): (Vec<bool>, Vec<usize>) = (vec![], vec![]);
    for d in det.chunks(6 + pc) {
        if d[4] < conf {
            continue;
        }
        let x1 = (((d[0] - px) / r).floor() as i32).clamp(0, w as i32 - 1);
        let y1 = (((d[1] - py) / r).floor() as i32).clamp(0, h as i32 - 1);
        let x2 = (((d[2] - px) / r).ceil() as i32).clamp(x1 + 1, w as i32);
        let y2 = (((d[3] - py) / r).ceil() as i32).clamp(y1 + 1, h as i32);
        let coef = &d[6..6 + pc];
        let (bw, bh) = ((x2 - x1) as usize, (y2 - y1) as usize);
        // 원형 격자 위치(행·열마다 한 번): (아래 칸, 위 칸, 가중)
        let fy = |yy: usize| -> (usize, usize, f32) {
            let f = (((y1 as usize + yy) as f32 + 0.5) * r + py) * scale - 0.5;
            let y0 = (f.floor() as i32).clamp(0, ph as i32 - 1) as usize;
            (y0, (y0 + 1).min(ph - 1), (f - y0 as f32).clamp(0.0, 1.0))
        };
        let fx = |xx: usize| -> (usize, usize, f32) {
            let f = (((x1 as usize + xx) as f32 + 0.5) * r + px) * scale - 0.5;
            let x0 = (f.floor() as i32).clamp(0, pw as i32 - 1) as usize;
            (x0, (x0 + 1).min(pw - 1), (f - x0 as f32).clamp(0.0, 1.0))
        };
        ys.clear();
        ys.extend((0..bh).map(fy));
        xs.clear();
        xs.extend((0..bw).map(fx));
        // 계수·원형 내적은 선형이라 원형 격자에서 먼저 한다. 상자가 덮는 칸만, 원형 4 장씩 묶어 더함(더하는 순서는 한 장씩과 같음)
        let (gy0, gy1) = (ys[0].0, ys[bh - 1].1);
        let (gx0, gx1) = (xs[0].0, xs[bw - 1].1);
        let gw = gx1 - gx0 + 1;
        g.clear();
        g.resize((gy1 - gy0 + 1) * gw, 0.0);
        let plane = |k: usize, yy: usize| &proto[k * ph * pw + yy * pw + gx0..k * ph * pw + yy * pw + gx0 + gw];
        let mut k = 0;
        while k + 4 <= pc {
            let c = [coef[k], coef[k + 1], coef[k + 2], coef[k + 3]];
            for yy in gy0..=gy1 {
                let dst = &mut g[(yy - gy0) * gw..(yy - gy0) * gw + gw];
                let (s0, s1, s2, s3) = (plane(k, yy), plane(k + 1, yy), plane(k + 2, yy), plane(k + 3, yy));
                for i in 0..gw {
                    dst[i] = dst[i] + c[0] * s0[i] + c[1] * s1[i] + c[2] * s2[i] + c[3] * s3[i];
                }
            }
            k += 4;
        }
        while k < pc {
            for yy in gy0..=gy1 {
                let dst = &mut g[(yy - gy0) * gw..(yy - gy0) * gw + gw];
                for (o, s) in dst.iter_mut().zip(plane(k, yy)) {
                    *o += coef[k] * s;
                }
            }
            k += 1;
        }
        // 같은 왼 칸을 읽는 화소끼리 구간으로 묶음(칸 번호는 화소 순서대로 늘어남)
        runs.clear();
        for (xx, &(x0, x1p, _)) in xs.iter().enumerate() {
            match runs.last_mut() {
                Some(l) if l.0 == x0 - gx0 => l.3 = xx + 1,
                _ => runs.push((x0 - gx0, x1p - gx0, xx, xx + 1)),
            }
        }
        let mut mask = vec![false; bw * bh];
        tmpl.clear();
        tmpl.resize(bw, false);
        let mut yy = 0;
        while yy < bh {
            // 같은 격자 줄 쌍을 읽는 화소 줄 묶음 [yy, ye)
            let (y0, y1p) = (ys[yy].0, ys[yy].1);
            let mut ye = yy + 1;
            while ye < bh && ys[ye].0 == y0 {
                ye += 1;
            }
            let (r0, r1) = (&g[(y0 - gy0) * gw..(y0 - gy0) * gw + gw], &g[(y1p - gy0) * gw..(y1p - gy0) * gw + gw]);
            // 칸마다: 네 귀퉁이가 모두 양수 → 채움, 모두 0 이하 → 비움, 섞임 → 화소마다 계산
            mixed.clear();
            let mut any = false;
            for (ri, &(a0, a1, xa, xb)) in runs.iter().enumerate() {
                let c = [r0[a0], r0[a1], r1[a0], r1[a1]];
                let inside = c.iter().all(|&v| v > INSIDE_MIN && v < INSIDE_MAX);
                if !inside && !c.iter().all(|&v| v <= 0.0) {
                    mixed.push(ri);
                }
                any |= inside;
                tmpl[xa..xb].fill(inside);
            }
            if any || !mixed.is_empty() {
                for row in yy..ye {
                    let ay = ys[row].2;
                    let out = &mut mask[row * bw..(row + 1) * bw];
                    if any {
                        out.copy_from_slice(&tmpl);
                    }
                    for &ri in &mixed {
                        let (a0, a1, xa, xb) = runs[ri];
                        for xx in xa..xb {
                            let ax = xs[xx].2;
                            let top = r0[a0] * (1.0 - ax) + r0[a1] * ax;
                            let bot = r1[a0] * (1.0 - ax) + r1[a1] * ax;
                            out[xx] = top * (1.0 - ay) + bot * ay > 0.0; // sigmoid(v) > 0.5
                        }
                    }
                }
            }
            yy = ye;
        }
        dets.push(Det { cls: d[5] as usize, score: d[4], box_: [x1, y1, x2, y2], mask });
    }
    dets
}

/// 기준 구현(단순한 화소별 계산): 빠른 구현과 결과가 같은지 견주는 데 쓴다(단위 검사, `--seg-check`).
pub mod reference {
    use super::{Det, LbGeom};

    pub fn letterbox(rgb: &[u8], w: usize, h: usize, s: usize) -> (Vec<u8>, LbGeom) {
        let r = (s as f32 / w as f32).min(s as f32 / h as f32);
        let (nw, nh) = ((w as f32 * r).round() as usize, (h as f32 * r).round() as usize);
        let (px, py) = ((s - nw) / 2, (s - nh) / 2);
        let mut buf = vec![114u8; s * s * 3];
        let xs: Vec<usize> = (0..nw).map(|x| (((x as f32 + 0.5) / r) as usize).min(w - 1) * 3).collect();
        for y in 0..nh {
            let sy = (((y as f32 + 0.5) / r) as usize).min(h - 1);
            let src = &rgb[sy * w * 3..(sy + 1) * w * 3];
            let dst = &mut buf[((py + y) * s + px) * 3..((py + y) * s + px + nw) * 3];
            for (d, &o) in dst.chunks_exact_mut(3).zip(&xs) {
                d.copy_from_slice(&src[o..o + 3]);
            }
        }
        (buf, LbGeom { r, px, py })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode(det: &[f32], proto: &[f32], pc: usize, ph: usize, pw: usize, s: usize, lb: LbGeom, w: usize, h: usize, conf: f32) -> Vec<Det> {
        let (r, px, py) = (lb.r, lb.px as f32, lb.py as f32);
        let scale = ph as f32 / s as f32;
        let mut dets = vec![];
        for d in det.chunks(6 + pc) {
            if d[4] < conf {
                continue;
            }
            let x1 = (((d[0] - px) / r).floor() as i32).clamp(0, w as i32 - 1);
            let y1 = (((d[1] - py) / r).floor() as i32).clamp(0, h as i32 - 1);
            let x2 = (((d[2] - px) / r).ceil() as i32).clamp(x1 + 1, w as i32);
            let y2 = (((d[3] - py) / r).ceil() as i32).clamp(y1 + 1, h as i32);
            let coef = &d[6..6 + pc];
            let (bw, bh) = ((x2 - x1) as usize, (y2 - y1) as usize);
            let fy = |yy: usize| -> (usize, usize, f32) {
                let f = (((y1 as usize + yy) as f32 + 0.5) * r + py) * scale - 0.5;
                let y0 = (f.floor() as i32).clamp(0, ph as i32 - 1) as usize;
                (y0, (y0 + 1).min(ph - 1), (f - y0 as f32).clamp(0.0, 1.0))
            };
            let fx = |xx: usize| -> (usize, usize, f32) {
                let f = (((x1 as usize + xx) as f32 + 0.5) * r + px) * scale - 0.5;
                let x0 = (f.floor() as i32).clamp(0, pw as i32 - 1) as usize;
                (x0, (x0 + 1).min(pw - 1), (f - x0 as f32).clamp(0.0, 1.0))
            };
            let ys: Vec<_> = (0..bh).map(fy).collect();
            let xs: Vec<_> = (0..bw).map(fx).collect();
            let (gy0, gy1) = (ys[0].0, ys[bh - 1].1);
            let (gx0, gx1) = (xs[0].0, xs[bw - 1].1);
            let gw = gx1 - gx0 + 1;
            let mut g = vec![0f32; (gy1 - gy0 + 1) * gw];
            for (k, c) in coef.iter().enumerate() {
                let b = k * ph * pw;
                for yy in gy0..=gy1 {
                    let src = &proto[b + yy * pw + gx0..b + yy * pw + gx1 + 1];
                    let dst = &mut g[(yy - gy0) * gw..(yy - gy0) * gw + gw];
                    for (o, s) in dst.iter_mut().zip(src) {
                        *o += c * s;
                    }
                }
            }
            let mut mask = vec![false; bw * bh];
            for (yy, &(y0, y1p, ay)) in ys.iter().enumerate() {
                let (r0, r1) = (&g[(y0 - gy0) * gw..(y0 - gy0) * gw + gw], &g[(y1p - gy0) * gw..(y1p - gy0) * gw + gw]);
                for (xx, &(x0, x1p, ax)) in xs.iter().enumerate() {
                    let (a0, a1) = (x0 - gx0, x1p - gx0);
                    let top = r0[a0] * (1.0 - ax) + r0[a1] * ax;
                    let bot = r1[a0] * (1.0 - ax) + r1[a1] * ax;
                    mask[yy * bw + xx] = top * (1.0 - ay) + bot * ay > 0.0;
                }
            }
            dets.push(Det { cls: d[5] as usize, score: d[4], box_: [x1, y1, x2, y2], mask });
        }
        dets
    }
}

/// 전처리 결과(추론 입력): 레터박스한 u8 NHWC. 버퍼는 장면마다 다시 쓴다.
#[derive(Default)]
pub struct SegIn {
    pub geom: LbGeom,
    pub u8: Vec<u8>,
    /// u8 에 마지막으로 레터박스한 영상 크기(같으면 회색 띠를 다시 칠하지 않음)
    pub lb_wh: Option<(usize, usize)>,
}

/// 추론 결과(후처리 입력): 검출 [300, 6 + pc] 와 원형 [pc, ph, pw] 를 버퍼 하나에. 버퍼는 장면마다 다시 쓴다.
#[derive(Default)]
pub struct SegRaw {
    pub buf: Vec<f32>,
    pub det: (usize, usize),   // buf 안 시작, 길이
    pub proto: (usize, usize), // buf 안 시작, 길이
    pub pc: usize,
    pub ph: usize,
    pub pw: usize,
}

/// 전처리 단계: 레터박스. 추론과 다른 스레드에서 돌려도 된다.
pub fn seg_pre(rgb: &[u8], w: usize, h: usize, inp: &mut SegIn) {
    inp.geom = letterbox_keep(rgb, w, h, SEG_S, &mut inp.u8, inp.lb_wh == Some((w, h)));
    inp.lb_wh = Some((w, h));
}

/// 후처리 단계: 마스크 해독. 추론과 다른 스레드에서 돌려도 된다.
pub fn seg_post(raw: &SegRaw, geom: LbGeom, w: usize, h: usize, conf: f32) -> Vec<Det> {
    decode(&raw.buf[raw.det.0..raw.det.0 + raw.det.1], &raw.buf[raw.proto.0..raw.proto.0 + raw.proto.1], raw.pc, raw.ph, raw.pw, SEG_S, geom, w, h, conf)
}

/// 분할 실행기. 단계 셋(전처리 `seg_pre` → 추론 `infer` → 후처리 `seg_post`)으로 나뉘고, `run` 은 셋을 차례로 한다.
pub trait Segmenter: Send {
    /// 추론만(NPU·모델 호출, 실행기 입력 형식 맞춤 포함). 결과는 raw 에 담는다.
    fn infer(&mut self, inp: &SegIn, raw: &mut SegRaw);
    /// `run` 이 다시 쓰는 버퍼
    fn scratch(&mut self) -> &mut (SegIn, SegRaw);
    /// 단계를 거치지 않고 결과를 바로 줄 수 있으면 준다(분할 대역 `--seg-cache`). 주면 `infer`·`seg_post` 는 부르지 않는다.
    fn direct(&mut self, _rgb: &[u8], _w: usize, _h: usize, _conf: f32) -> Option<Vec<Det>> {
        None
    }
    /// 추론을 올려만 두고 돌아올 수 있으면 작업 번호를 준다(NPU 코어 여럿에 여러 장을 함께 올림, `--npu-jobs`). 결과는 `collector` 가 준 함수로 받는다.
    fn submit(&mut self, _inp: &SegIn) -> Option<i32> {
        None
    }
    /// `submit` 한 작업의 결과를 다른 스레드에서 받는 함수. 실행기보다 먼저 없앤다.
    fn collector(&self) -> Option<Box<dyn FnMut(i32, &mut SegRaw) + Send>> {
        None
    }
    fn run(&mut self, rgb: &[u8], w: usize, h: usize, conf: f32) -> Vec<Det> {
        let (mut inp, mut raw) = std::mem::take(self.scratch());
        seg_pre(rgb, w, h, &mut inp);
        self.infer(&inp, &mut raw);
        let dets = seg_post(&raw, inp.geom, w, h, conf);
        *self.scratch() = (inp, raw);
        dets
    }
}

// ------------------------------------------------------------------ PC: ONNX Runtime
/// ONNX Runtime 세션: 프로세스에 하나인 환경(`common::ort`, 뷰어 글 검색과 같은 것)에서 만든다
#[cfg(feature = "onnx")]
fn session(path: &str) -> ort::session::Session {
    #[allow(unused_mut)]
    // 스레드는 쓸 수 있는 코어 수까지만(8 고정이면 taskset 2 코어에서 7 개가 서로 깨우며 문맥 전환 260 만 번·코어를 나눠 먹음)
    let nth = std::thread::available_parallelism().map_or(8, |n| n.get()).min(8);
    #[allow(unused_mut)]
    let mut b = ort::session::Session::builder(common::ort::env()).unwrap().with_intra_threads(nth).unwrap();
    // directml 기능이면 그래픽카드로(안 되면 ORT 가 CPU 로 물러남). 입력 크기를 고정해 두면 DirectML 이 빠르다.
    #[cfg(feature = "directml")]
    let mut b = b.with_execution_providers([ort::ep::DirectML::default().build()]).unwrap().with_dimension_override("batch", 1).unwrap();
    b.commit_from_file(path).unwrap_or_else(|e| panic!("onnx model {path}: {e}"))
}

#[cfg(feature = "onnx")]
pub struct OnnxSeg {
    sess: ort::session::Session,
    scratch: (SegIn, SegRaw),
    f32: Vec<f32>, // 모델 입력(f32 NCHW /255)
}
#[cfg(feature = "onnx")]
impl OnnxSeg {
    pub fn new(env: &ort::environment::Environment, path: &str) -> OnnxSeg {
        let _ = env;
        OnnxSeg { sess: session(path), scratch: Default::default(), f32: vec![] }
    }
}
#[cfg(feature = "onnx")]
impl Segmenter for OnnxSeg {
    fn scratch(&mut self) -> &mut (SegIn, SegRaw) {
        &mut self.scratch
    }
    fn infer(&mut self, inp: &SegIn, raw: &mut SegRaw) {
        nchw_f32(&inp.u8, SEG_S, &mut self.f32);
        let t = ort::value::TensorRef::from_array_view(([1usize, 3, SEG_S, SEG_S], &self.f32[..])).unwrap();
        let out = self.sess.run(ort::inputs![t]).expect("onnx run");
        let (_, det) = out[0].try_extract_tensor::<f32>().unwrap();
        let (ps, proto) = out[1].try_extract_tensor::<f32>().unwrap();
        raw.buf.clear();
        raw.buf.extend_from_slice(det);
        raw.buf.extend_from_slice(proto);
        raw.det = (0, det.len());
        raw.proto = (det.len(), proto.len());
        (raw.pc, raw.ph, raw.pw) = (ps[1] as usize, ps[2] as usize, ps[3] as usize);
    }
}

// ------------------------------------------------------------------ Pi5 + DX-M1: DXRT C API
#[cfg(feature = "dxrt")]
pub mod dx {
    use std::ffi::{c_char, c_int, c_void, CStr, CString};

    #[repr(C)]
    pub struct Options {
        pub struct_size: u32,
        pub buffer_count: c_int,
        pub device_id: c_int,
        pub bound_option: u32,
        pub use_ort: c_int,
        pub reserved: [u32; 6],
    }
    #[repr(C)]
    pub struct TensorInfo {
        pub name: [c_char; 256],
        pub dtype: c_int,
        pub elem_size: u32,
        pub ndim: c_int,
        pub shape: [i64; 8],
        pub size_in_bytes: u64,
    }
    #[link(name = "dxrt")]
    unsafe extern "C" {
        fn dxrt_engine_create(path: *const c_char, opts: *const Options, out: *mut *mut c_void) -> c_int;
        fn dxrt_engine_destroy(e: *mut c_void);
        fn dxrt_engine_run(e: *mut c_void, input: *const c_void, output: *mut c_void) -> c_int;
        fn dxrt_engine_run_async(e: *mut c_void, input: *const c_void, user_arg: *mut c_void, output: *mut c_void, job_id: *mut c_int) -> c_int;
        fn dxrt_engine_wait(e: *mut c_void, job_id: c_int, output: *mut c_void) -> c_int;
        fn dxrt_engine_get_output_size(e: *mut c_void, size: *mut u64) -> c_int;
        fn dxrt_engine_get_output_tensor_info(e: *mut c_void, infos: *mut TensorInfo, count: *mut c_int) -> c_int;
        fn dxrt_last_error_message() -> *const c_char;
    }

    /// NPU 모델 하나: 입력 u8 NHWC, 출력은 모든 텐서를 이어 붙인 f32 버퍼
    pub struct Engine {
        e: *mut c_void,
        pub out: Vec<u8>,
        pub tensors: Vec<(String, Vec<i64>, usize)>, // 이름, 모양, 바이트 오프셋
    }
    impl Engine {
        pub fn new(path: &str) -> Engine {
            Engine::open(path, -1)
        }
        /// 모델 안의 CPU 부분까지 런타임이 돌리게 연다(ONNX Runtime). 입력·출력이 모델 원래의 것(f32)이 된다
        pub fn with_cpu_tasks(path: &str) -> Engine {
            Engine::open(path, 1)
        }
        fn open(path: &str, use_ort: c_int) -> Engine {
            let o = Options { struct_size: std::mem::size_of::<Options>() as u32, buffer_count: 0, device_id: -1, bound_option: 0, use_ort, reserved: [0; 6] };
            let mut e = std::ptr::null_mut();
            let p = CString::new(path).unwrap();
            unsafe {
                if dxrt_engine_create(p.as_ptr(), &o, &mut e) != 0 {
                    panic!("dxrt: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
                }
                let mut sz = 0u64;
                dxrt_engine_get_output_size(e, &mut sz);
                let mut infos: Vec<TensorInfo> = (0..16).map(|_| std::mem::zeroed()).collect();
                let mut n: c_int = 16;
                dxrt_engine_get_output_tensor_info(e, infos.as_mut_ptr(), &mut n);
                let mut off = 0usize;
                let tensors = infos[..n as usize]
                    .iter()
                    .map(|t| {
                        let name = CStr::from_ptr(t.name.as_ptr()).to_string_lossy().into_owned();
                        let shape = t.shape[..t.ndim as usize].to_vec();
                        let r = (name, shape, off);
                        off += t.size_in_bytes as usize;
                        r
                    })
                    .collect();
                Engine { e, out: vec![0u8; sz as usize], tensors }
            }
        }
        pub fn run(&mut self, input: &[u8]) {
            unsafe {
                if dxrt_engine_run(self.e, input.as_ptr() as *const c_void, self.out.as_mut_ptr() as *mut c_void) != 0 {
                    panic!("dxrt run: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
                }
            }
        }
        /// 결과를 받는 쪽 버퍼(f32, 모든 텐서 이어 붙임)에 바로 받기
        pub fn run_into(&mut self, input: &[u8], out: &mut Vec<f32>) {
            out.resize(self.out.len().div_ceil(4), 0.0);
            unsafe {
                if dxrt_engine_run(self.e, input.as_ptr() as *const c_void, out.as_mut_ptr() as *mut c_void) != 0 {
                    panic!("dxrt run: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
                }
            }
        }
        /// 추론 하나를 NPU 에 올려 두고 바로 돌아옴(작업 번호). 입력 버퍼는 `Waiter::wait_into` 가 끝날 때까지 그대로 둔다.
        pub fn submit(&mut self, input: &[u8]) -> i32 {
            let mut id: c_int = 0;
            unsafe {
                if dxrt_engine_run_async(self.e, input.as_ptr() as *const c_void, std::ptr::null_mut(), std::ptr::null_mut(), &mut id) != 0 {
                    panic!("dxrt run_async: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
                }
            }
            id
        }
        /// 올려 둔 작업의 결과를 다른 스레드에서 받는 손잡이. 엔진보다 먼저 없앤다.
        pub fn waiter(&self) -> Waiter {
            Waiter { e: self.e, len: self.out.len() }
        }
        /// 텐서 i 의 (f32 시작, 길이)
        pub fn f32_span(&self, i: usize) -> (usize, usize) {
            let (_, shape, off) = &self.tensors[i];
            let n: i64 = shape.iter().product();
            assert!(*off % 4 == 0, "dxrt 출력 텐서 {i} 의 시작이 4 바이트 경계가 아님");
            (*off / 4, n as usize)
        }
        pub fn f32_at(&self, i: usize) -> &[f32] {
            let (_, shape, off) = &self.tensors[i];
            let n: i64 = shape.iter().product();
            unsafe { std::slice::from_raw_parts(self.out.as_ptr().add(*off) as *const f32, n as usize) }
        }
    }
    impl Drop for Engine {
        fn drop(&mut self) {
            unsafe { dxrt_engine_destroy(self.e) }
        }
    }
    unsafe impl Send for Engine {}

    pub struct Waiter {
        e: *mut c_void,
        len: usize,
    }
    impl Waiter {
        /// 작업이 끝날 때까지 기다려 결과(f32, 모든 텐서 이어 붙임)를 받음
        pub fn wait_into(&self, id: i32, out: &mut Vec<f32>) {
            out.resize(self.len.div_ceil(4), 0.0);
            unsafe {
                if dxrt_engine_wait(self.e, id, out.as_mut_ptr() as *mut c_void) != 0 {
                    panic!("dxrt wait: {}", CStr::from_ptr(dxrt_last_error_message()).to_string_lossy());
                }
            }
        }
    }
    unsafe impl Send for Waiter {}
}

#[cfg(feature = "dxrt")]
pub struct DxSeg {
    eng: dx::Engine,
    scratch: (SegIn, SegRaw),
}
#[cfg(feature = "dxrt")]
impl DxSeg {
    pub fn new(path: &str) -> DxSeg {
        DxSeg { eng: dx::Engine::new(path), scratch: Default::default() }
    }
}
#[cfg(feature = "dxrt")]
impl Segmenter for DxSeg {
    fn scratch(&mut self) -> &mut (SegIn, SegRaw) {
        &mut self.scratch
    }
    /// 동기 호출(NPU 는 한 번에 하나). 결과를 raw 버퍼에 바로 받아, 후처리 스레드가 그것을 읽는 동안 다음 장면을 돌린다.
    fn infer(&mut self, inp: &SegIn, raw: &mut SegRaw) {
        self.eng.run_into(&inp.u8, &mut raw.buf);
        raw.det = self.eng.f32_span(0);
        raw.proto = self.eng.f32_span(1);
        let ps = &self.eng.tensors[1].1;
        (raw.pc, raw.ph, raw.pw) = (ps[1] as usize, ps[2] as usize, ps[3] as usize);
    }
    fn submit(&mut self, inp: &SegIn) -> Option<i32> {
        Some(self.eng.submit(&inp.u8))
    }
    fn collector(&self) -> Option<Box<dyn FnMut(i32, &mut SegRaw) + Send>> {
        let (w, det, proto) = (self.eng.waiter(), self.eng.f32_span(0), self.eng.f32_span(1));
        let ps = &self.eng.tensors[1].1;
        let p = (ps[1] as usize, ps[2] as usize, ps[3] as usize);
        Some(Box::new(move |id, raw| {
            w.wait_into(id, &mut raw.buf);
            (raw.det, raw.proto) = (det, proto);
            (raw.pc, raw.ph, raw.pw) = p;
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 0–1 난수(검사용, 씨앗 고정)
    fn rnd(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 40) as f32) / (1u64 << 24) as f32
    }

    #[test]
    fn letterbox_same_as_reference() {
        let mut s = 1u64;
        let mut buf = vec![7u8; 5]; // 앞 장면의 찌꺼기가 남은 버퍼로 시작
        for &(w, h) in &[(640usize, 480usize), (1280, 720), (320, 240), (480, 640), (641, 479), (100, 37)] {
            let rgb: Vec<u8> = (0..w * h * 3).map(|_| (rnd(&mut s) * 255.0) as u8).collect();
            let (want, g0) = reference::letterbox(&rgb, w, h, SEG_S);
            let g1 = letterbox(&rgb, w, h, SEG_S, &mut buf);
            assert_eq!(g0, g1, "{w}x{h}");
            assert!(want == buf, "{w}x{h}");
            // 같은 크기로 두 번째(띠를 다시 칠하지 않음): 다른 화소로 덮어도 같아야 함
            let rgb2: Vec<u8> = rgb.iter().map(|v| v ^ 0x5a).collect();
            let (want2, _) = reference::letterbox(&rgb2, w, h, SEG_S);
            letterbox_keep(&rgb2, w, h, SEG_S, &mut buf, true);
            assert!(want2 == buf, "{w}x{h} 두 번째");
            letterbox(&rgb, w, h, SEG_S, &mut buf);
            #[cfg(feature = "onnx")]
            {
                let mut f = vec![];
                nchw_f32(&buf, SEG_S, &mut f);
                for (i, p) in buf.chunks_exact(3).enumerate().step_by(97) {
                    for c in 0..3 {
                        assert_eq!(f[c * SEG_S * SEG_S + i], p[c] as f32 / 255.0);
                    }
                }
            }
        }
    }

    #[test]
    fn decode_same_as_reference() {
        let (pc, ph, pw) = (32usize, 160usize, 160usize);
        let mut s = 2u64;
        let mut px_total = 0usize;
        for &(w, h) in &[(640usize, 480usize), (1280, 720), (320, 240), (480, 640), (641, 479)] {
            let r = (SEG_S as f32 / w as f32).min(SEG_S as f32 / h as f32);
            let (nw, nh) = ((w as f32 * r).round() as usize, (h as f32 * r).round() as usize);
            let lb = LbGeom { r, px: (SEG_S - nw) / 2, py: (SEG_S - nh) / 2 };
            for kind in 0..3 {
                // 원형: 0 = 완만한 기울기(넓은 안·밖 + 가장자리), 1 = 잡음(거의 모든 칸이 가장자리), 2 = 둘을 섞고 0 과 아주 작은 값도 넣음
                let mut proto = vec![0f32; pc * ph * pw];
                for k in 0..pc {
                    let (a, b, cx, cy) = (rnd(&mut s) - 0.5, rnd(&mut s) - 0.5, rnd(&mut s) * pw as f32, rnd(&mut s) * ph as f32);
                    for y in 0..ph {
                        for x in 0..pw {
                            let smooth = (a * (x as f32 - cx) + b * (y as f32 - cy)) * 0.05;
                            let noise = rnd(&mut s) - 0.5;
                            proto[(k * ph + y) * pw + x] = match kind {
                                0 => smooth,
                                1 => noise,
                                _ => match (rnd(&mut s) * 20.0) as u32 {
                                    0 => 0.0,
                                    1 => 1e-30,
                                    _ => smooth + noise * 0.02,
                                },
                            };
                        }
                    }
                }
                let mut det = vec![];
                for i in 0..40 {
                    let (x1, y1) = (rnd(&mut s) * 700.0 - 30.0, rnd(&mut s) * 700.0 - 30.0);
                    let (bw, bh) = (rnd(&mut s) * 500.0 + 1.0, rnd(&mut s) * 500.0 + 1.0);
                    det.extend([x1, y1, x1 + bw, y1 + bh, if i % 4 == 0 { 0.1 } else { 0.9 }, (i % 80) as f32]);
                    // 계수: 보통은 난수, 가끔은 모두 0(값이 정확히 0 인 마스크)이나 한 장만 켬
                    for k in 0..pc {
                        det.push(match i % 7 {
                            0 => 0.0,
                            1 => (k == i % pc) as u32 as f32,
                            _ => rnd(&mut s) * 2.0 - 1.0,
                        });
                    }
                }
                let want = reference::decode(&det, &proto, pc, ph, pw, SEG_S, lb, w, h, 0.25);
                let got = decode(&det, &proto, pc, ph, pw, SEG_S, lb, w, h, 0.25);
                assert_eq!(want.len(), got.len());
                assert_eq!(want.len(), 30);
                for (a, b) in want.iter().zip(&got) {
                    assert_eq!((a.cls, a.score, a.box_), (b.cls, b.score, b.box_));
                    assert!(a.mask == b.mask, "{w}x{h} kind {kind} box {:?}", a.box_);
                    px_total += a.mask.len();
                }
            }
        }
        assert!(px_total > 1_000_000);
    }
}
