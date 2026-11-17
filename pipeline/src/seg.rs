//! ObjectSAM 분할(클래스 없는 물체 1 클래스, YOLO 형 머리), end-to-end 출력: [1,300,38] = x1 y1 x2 y2 점수 클래스(늘 0) + 계수 32, 원형 [1,32,160,160].
//! 이름은 분할이 아니라 SigLIP2 사진 벡터(embed.rs)가 붙인다.
//! 실행기 두 개가 같은 전처리(레터박스 RGB, 회색 114)·후처리(상자 안 화소만 계수·원형 내적)를 쓴다.
//!   - PC: ONNX Runtime (feature "onnx", 기본 models/ObjectSAM-416_640.onnx, 입력 f32 NCHW /255)
//!   - Pi5 + DX-M1: DXRT C API (feature "dxrt", 기본 ~/models/objectsam-416_640.dxnn, NPU 만 도는 머리 출력, 입력 u8 NHWC)


pub struct Det {
    pub cls: usize,
    pub score: f32,
    pub box_: [i32; 4], // 영상 화소 x1 y1 x2 y2 (잘라 맞춤)
    pub mask: Vec<bool>, // 상자 크기 (x2-x1)*(y2-y1), 행 우선
}

/// 분할 모델 입력 한 변
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
    // 표는 스레드에 하나 두고 다시 씀(장면마다 할당 없음)
    thread_local! { static XS: std::cell::Cell<Vec<usize>> = const { std::cell::Cell::new(Vec::new()) }; }
    let mut xs = XS.with(|c| c.take());
    xs.clear();
    if !same {
        xs.extend((0..nw).map(|x| (((x as f32 + 0.5) / r) as usize).min(w - 1) * 3));
    }
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
    XS.with(|c| c.set(xs));
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
    /// 머리 출력 모델(NPU 만, 상위 300 고르기를 이 프로그램이 함): 단계 3 개마다 (상자 4, 점수 1, 계수 pc) 의 buf 안 시작. 없으면 det 가 [300, 6 + pc]
    pub heads: Option<[[usize; 3]; 3]>,
}

/// 머리 출력(상자 거리 [4,n,n] · 점수 로짓 [1,n,n] · 계수 [pc,n,n], 보폭 8 · 16 · 32)에서 점수 conf 이상만 골라 [k, 6 + pc] 행으로(점수 큰 순, 최대 300).
/// 모델 안 후처리(ONNX: 시그모이드 → 상위 300 → 상자 = (격자 + 0.5 ∓ 거리) × 보폭)와 같은 값. 로짓 문턱을 먼저 견줘 시그모이드는 남은 것만
pub fn heads_to_det(buf: &[f32], heads: &[[usize; 3]; 3], pc: usize, s: usize, conf: f32, out: &mut Vec<f32>) {
    let thr = (conf / (1.0 - conf)).ln();
    let mut cand: Vec<(f32, u8, u32)> = Vec::new();
    for (l, hd) in heads.iter().enumerate() {
        let n = s >> (3 + l);
        let cls = &buf[hd[1]..hd[1] + n * n];
        for (i, &v) in cls.iter().enumerate() {
            if v >= thr {
                cand.push((1.0 / (1.0 + (-v).exp()), l as u8, i as u32));
            }
        }
    }
    if cand.len() > 300 {
        cand.select_nth_unstable_by(299, |a, b| b.0.total_cmp(&a.0));
        cand.truncate(300);
    }
    cand.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.clear();
    for &(sc, l, i) in &cand {
        let (hd, n, st) = (&heads[l as usize], s >> (3 + l), (8u32 << l) as f32);
        let (i, nn) = (i as usize, n * n);
        let (ax, ay) = ((i % n) as f32 + 0.5, (i / n) as f32 + 0.5);
        let b = |k: usize| buf[hd[0] + k * nn + i];
        out.extend_from_slice(&[(ax - b(0)) * st, (ay - b(1)) * st, (ax + b(2)) * st, (ay + b(3)) * st, sc, 0.0]);
        out.extend((0..pc).map(|k| buf[hd[2] + k * nn + i]));
    }
}

/// 전처리 단계: 레터박스. 추론과 다른 스레드에서 돌려도 된다.
pub fn seg_pre(rgb: &[u8], w: usize, h: usize, inp: &mut SegIn) {
    let keep = inp.lb_wh == Some((w, h));
    inp.geom = letterbox_keep(rgb, w, h, SEG_S, &mut inp.u8, keep);
    inp.lb_wh = Some((w, h));
    common::prof::copy(common::prof::C::SegLetterbox, if keep { (SEG_S - 2 * inp.geom.px) * (SEG_S - 2 * inp.geom.py) * 3 } else { SEG_S * SEG_S * 3 });
}

/// 후처리 단계: 마스크 해독. 추론과 다른 스레드에서 돌려도 된다.
pub fn seg_post(raw: &SegRaw, geom: LbGeom, w: usize, h: usize, conf: f32) -> Vec<Det> {
    let dets = seg_post_any(raw, geom, w, h, conf);
    // 클래스 없는 물체 모델(ObjectSAM)만 받는다: 클래스가 둘 이상인 모델(COCO 등)은 멈춤
    if let Some(d) = dets.iter().find(|d| d.cls != 0) {
        panic!("--seg: 클래스 {} 을 낸 모델 — 클래스 없는 ObjectSAM 분할 모델(1 클래스)만 쓴다", d.cls);
    }
    dets
}

fn seg_post_any(raw: &SegRaw, geom: LbGeom, w: usize, h: usize, conf: f32) -> Vec<Det> {
    if let Some(hd) = &raw.heads {
        thread_local!(static DET: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) });
        return DET.with_borrow_mut(|det| {
            heads_to_det(&raw.buf, hd, raw.pc, SEG_S, conf, det);
            decode(det, &raw.buf[raw.proto.0..raw.proto.0 + raw.proto.1], raw.pc, raw.ph, raw.pw, SEG_S, geom, w, h, conf)
        });
    }
    decode(&raw.buf[raw.det.0..raw.det.0 + raw.det.1], &raw.buf[raw.proto.0..raw.proto.0 + raw.proto.1], raw.pc, raw.ph, raw.pw, SEG_S, geom, w, h, conf)
}

/// 분할 실행기. 단계 셋(전처리 `seg_pre` → 추론 `infer` → 후처리 `seg_post`)으로 나뉘고, `run` 은 셋을 차례로 한다.
/// 작업 끝 알림: 올릴 때 준 표(tag)를 받는다
pub type Notify = std::sync::Arc<dyn Fn(u64) + Send + Sync>;

pub trait Segmenter: Send {
    /// 추론만(NPU·모델 호출, 실행기 입력 형식 맞춤 포함). 결과는 raw 에 담는다.
    fn infer(&mut self, inp: &SegIn, raw: &mut SegRaw);
    /// `run` 이 다시 쓰는 버퍼
    fn scratch(&mut self) -> &mut (SegIn, SegRaw);
    /// 단계를 거치지 않고 결과를 바로 줄 수 있으면 준다(분할 대역 `--seg-cache`). 주면 `infer`·`seg_post` 는 부르지 않는다.
    fn direct(&mut self, _rgb: &[u8], _w: usize, _h: usize, _conf: f32) -> Option<Vec<Det>> {
        None
    }
    /// 추론을 올려만 두고 바로 돌아올 수 있으면 작업 번호를 준다. 결과는 raw 에 바로 쓰이고(NPU) 끝나면 `set_notify` 로 단 함수가
    /// notify(tag) 를 부른다(어느 스레드에서든). 알림을 받으면 `take` 로 결과를 맞춘다.
    /// inp · raw 버퍼는 `take` 까지 옮겨도 되지만(힙 자리 그대로) 크기를 바꾸면 안 된다
    fn submit(&mut self, _inp: &SegIn, _raw: &mut SegRaw, _tag: u64) -> Option<i32> {
        None
    }
    /// 작업 끝 알림을 단다(`submit` 한 작업마다 한 번). 비동기 실행기만 true
    fn set_notify(&mut self, _n: Notify) -> bool {
        false
    }
    /// 끝 알림을 받은 작업의 결과를 raw 에 맞춘다(기다리지 않음)
    fn take(&mut self, _id: i32, _raw: &mut SegRaw) {}
    /// 함께 올려 둘 수의 상한(NPU 는 --npu-jobs 그대로, 실행 스레드 하나인 PC ONNX 는 1: 더 올려도 줄만 길어져 장면 지연만 는다)
    fn max_jobs(&self) -> usize {
        usize::MAX
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

/// ONNX 실행 결과: 검출 [300, 6 + pc] 와 원형 [pc, ph, pw] 를 이어 붙인 버퍼
#[cfg(feature = "onnx")]
struct OnnxOut {
    id: i32,
    buf: Vec<f32>,
    det: usize,
    p: (usize, usize, usize),
}
#[cfg(feature = "onnx")]
type Pools = std::sync::Arc<std::sync::Mutex<(Vec<Vec<f32>>, Vec<Vec<f32>>)>>;

/// PC: 세션을 맡은 실행 스레드 하나에 올리고(NPU 의 비동기 실행과 같은 길) 결과는 올린 순서대로 받는다.
/// 입력(f32 NCHW)·출력 버퍼는 돌려쓴다
#[cfg(feature = "onnx")]
pub struct OnnxSeg {
    tx: Option<std::sync::mpsc::Sender<(i32, u64, Vec<f32>)>>,
    notify: std::sync::Arc<std::sync::OnceLock<Notify>>,
    rx: std::sync::Arc<std::sync::Mutex<std::sync::mpsc::Receiver<OnnxOut>>>,
    /// 입력 f32 버퍼 줄, 출력 버퍼 줄
    pool: Pools,
    th: Option<std::thread::JoinHandle<()>>,
    next: i32,
    scratch: (SegIn, SegRaw),
}
#[cfg(feature = "onnx")]
impl OnnxSeg {
    pub fn new(env: &ort::environment::Environment, path: &str) -> OnnxSeg {
        let _ = env;
        let mut sess = session(path);
        let (tx, rx_in) = std::sync::mpsc::channel::<(i32, u64, Vec<f32>)>();
        let (tx_out, rx) = std::sync::mpsc::channel::<OnnxOut>();
        let pool: Pools = Default::default();
        let pool_w = pool.clone();
        let notify: std::sync::Arc<std::sync::OnceLock<Notify>> = Default::default();
        let notify_w = notify.clone();
        let th = std::thread::Builder::new().name("seg-onnx".into()).spawn(move || {
            crate::out::thread_role(crate::out::Role::Infer);
            common::prof::track_thread();
            for (id, tag, f) in rx_in {
                let t = ort::value::TensorRef::from_array_view(([1usize, 3, SEG_S, SEG_S], &f[..])).unwrap();
                let out = sess.run(ort::inputs![t]).expect("onnx run");
                let (_, det) = out[0].try_extract_tensor::<f32>().unwrap();
                let (ps, proto) = out[1].try_extract_tensor::<f32>().unwrap();
                let mut buf = pool_w.lock().unwrap().1.pop().unwrap_or_default();
                buf.clear();
                buf.extend_from_slice(det);
                buf.extend_from_slice(proto);
                common::prof::copy(common::prof::C::SegOnnxOut, (det.len() + proto.len()) * 4);
                let o = OnnxOut { id, buf, det: det.len(), p: (ps[1] as usize, ps[2] as usize, ps[3] as usize) };
                drop(out);
                pool_w.lock().unwrap().0.push(f);
                if tx_out.send(o).is_err() {
                    break;
                }
                if let Some(n) = notify_w.get() {
                    n(tag);
                }
            }
        }).unwrap();
        OnnxSeg { tx: Some(tx), notify, rx: std::sync::Arc::new(std::sync::Mutex::new(rx)), pool, th: Some(th), next: 0, scratch: Default::default() }
    }
}
/// 올린 순서대로 결과 하나를 받아 raw 에(버퍼를 바꿔 끼움: 복사 없음, 옛 버퍼는 줄로)
#[cfg(feature = "onnx")]
fn onnx_take(rx: &std::sync::Mutex<std::sync::mpsc::Receiver<OnnxOut>>, pool: &Pools, id: i32, raw: &mut SegRaw) {
    let o = rx.lock().unwrap().recv().expect("분할 실행 스레드");
    assert_eq!(o.id, id, "분할 결과 순서");
    let old = std::mem::replace(&mut raw.buf, o.buf);
    pool.lock().unwrap().1.push(old);
    raw.det = (0, o.det);
    raw.proto = (o.det, raw.buf.len() - o.det);
    (raw.pc, raw.ph, raw.pw) = o.p;
    raw.heads = None;
}
#[cfg(feature = "onnx")]
impl Segmenter for OnnxSeg {
    fn scratch(&mut self) -> &mut (SegIn, SegRaw) {
        &mut self.scratch
    }
    fn infer(&mut self, inp: &SegIn, raw: &mut SegRaw) {
        let id = self.submit(inp, raw, 0).unwrap();
        onnx_take(&self.rx, &self.pool, id, raw);
    }
    /// 실행 스레드가 끝나면 notify(표)
    fn submit(&mut self, inp: &SegIn, _raw: &mut SegRaw, tag: u64) -> Option<i32> {
        let mut f = self.pool.lock().unwrap().0.pop().unwrap_or_default();
        nchw_f32(&inp.u8, SEG_S, &mut f);
        common::prof::copy(common::prof::C::SegNchw, f.len() * 4);
        let id = self.next;
        self.next = self.next.wrapping_add(1);
        self.tx.as_ref().unwrap().send((id, tag, f)).expect("분할 실행 스레드");
        Some(id)
    }
    fn set_notify(&mut self, n: Notify) -> bool {
        let _ = self.notify.set(n);
        true
    }
    fn take(&mut self, id: i32, raw: &mut SegRaw) {
        onnx_take(&self.rx, &self.pool, id, raw);
    }
    fn max_jobs(&self) -> usize {
        1
    }
}
#[cfg(feature = "onnx")]
impl Drop for OnnxSeg {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(th) = self.th.take() {
            let _ = th.join();
        }
    }
}

// ------------------------------------------------------------------ Pi5 + DX-M1: DXRT C API(common::dxrt)
#[cfg(feature = "dxrt")]
pub use common::dxrt as dx;

#[cfg(feature = "dxrt")]
pub struct DxSeg {
    eng: dx::Engine,
    scratch: (SegIn, SegRaw),
    lay: Layout,
}
#[cfg(feature = "dxrt")]
/// dxnn 출력 배치: 머리 9 개 + 원형(NPU 만, ObjectSAM)
#[derive(Clone, Copy)]
struct Layout {
    det: (usize, usize),
    proto: (usize, usize),
    p: (usize, usize, usize),
    heads: Option<[[usize; 3]; 3]>,
}
#[cfg(feature = "dxrt")]
impl Layout {
    fn apply(&self, raw: &mut SegRaw) {
        (raw.det, raw.proto, raw.heads) = (self.det, self.proto, self.heads);
        (raw.pc, raw.ph, raw.pw) = self.p;
    }
}
#[cfg(feature = "dxrt")]
impl DxSeg {
    pub fn new(path: &str) -> DxSeg {
        let eng = dx::Engine::new(path);
        let find = |key: &str| eng.tensors.iter().position(|t| t.0.contains(key));
        let lay = if let (Some(pi), true) = (find("output1"), eng.tensors.len() == 10) {
            let ps = &eng.tensors[pi].1;
            let mut heads = [[0usize; 3]; 3];
            for (l, hd) in heads.iter_mut().enumerate() {
                for (k, c) in [2, 3, 4].iter().enumerate() {
                    let i = find(&format!("one2one_cv{c}.{l}/")).unwrap_or_else(|| panic!("{path}: 머리 출력 cv{c}.{l} 없음"));
                    hd[k] = eng.f32_span(i).0;
                }
            }
            common::info!("seg: NPU 만 도는 머리 출력 모델 {path}(상위 고르기는 이 프로그램이)");
            Layout { det: (0, 0), proto: eng.f32_span(pi), p: (ps[1] as usize, ps[2] as usize, ps[3] as usize), heads: Some(heads) }
        } else {
            panic!("--seg {path}: NPU 만 도는 머리 출력(출력 10 개) ObjectSAM 모델이 아님 — tools npu_prep 으로 만든 objectsam-*.dxnn 을 쓴다");
        };
        DxSeg { eng, scratch: Default::default(), lay }
    }
}
#[cfg(feature = "dxrt")]
impl Segmenter for DxSeg {
    fn scratch(&mut self) -> &mut (SegIn, SegRaw) {
        &mut self.scratch
    }
    /// 동기 호출. 결과를 raw 버퍼에 바로 받는다.
    fn infer(&mut self, inp: &SegIn, raw: &mut SegRaw) {
        self.eng.run_into(&inp.u8, &mut raw.buf);
        self.lay.apply(raw);
    }
    /// 올리면서 받을 버퍼(raw.buf)를 함께 넘김: NPU 결과가 거기에 바로 쓰인다(받을 때 DXRT 안 버퍼에서 3.3 MB 를 다시 복사하지 않음)
    fn submit(&mut self, inp: &SegIn, raw: &mut SegRaw, tag: u64) -> Option<i32> {
        Some(self.eng.submit_tagged(&inp.u8, &mut raw.buf, tag))
    }
    /// 끝 알림: `--npu-wake thread`(기본) = 기다림 스레드 npu-wait-seg, `callback` = DXRT 콜백
    fn set_notify(&mut self, n: Notify) -> bool {
        self.eng.on_done(dx::wake(), "seg", n);
        true
    }
    fn take(&mut self, id: i32, raw: &mut SegRaw) {
        if self.eng.callback_mode() {
            self.eng.waiter().wait(id); // 이미 끝남: 작업 자리만 놓음
        }
        self.lay.apply(raw);
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
