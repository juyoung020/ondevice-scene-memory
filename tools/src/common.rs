//! 도구 공통: printf 와 같은 글자 모양, 보정값 JSON 쓰기·읽기, 16 비트/RGB PNG, TUM 목록, 시간 재기, 코어 묶기, 정렬 결과 비교.
use crate::fast_align::{Extrinsics, Intrinsics};
use std::path::Path;
use std::time::Instant;

// ---------------- printf 와 같은 글자 모양 ----------------

/// C printf 의 `%-Ns`: 바이트 수로 오른쪽을 채움(한글 한 글자 = 3 바이트)
pub fn lpad(s: &str, n: usize) -> String {
    let mut o = s.to_string();
    while o.len() < n {
        o.push(' ');
    }
    o
}

/// C printf 의 `%.Pg`(ostream 기본 글자 모양은 P = 6)
pub fn fmt_g(v: f64, prec: usize) -> String {
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let p = prec.max(1);
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let e = format!("{:.*e}", p - 1, v);
    let (mant, exp) = e.split_once('e').unwrap();
    let x: i32 = exp.parse().unwrap();
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if x < -4 || x >= p as i32 {
        format!("{}e{}{:02}", strip(mant), if x < 0 { '-' } else { '+' }, x.abs())
    } else {
        strip(&format!("{:.*}", (p as i32 - 1 - x) as usize, v))
    }
}

/// C 의 atof: 앞쪽 공백 뒤 읽을 수 있는 만큼(없으면 0)
pub fn atof(s: &str) -> f64 {
    let s = s.trim_start();
    let b = s.as_bytes();
    let mut best = 0.0;
    let mut end = 0;
    while end < b.len() && b"+-0123456789.eEinfatyINFATY".contains(&b[end]) {
        end += 1;
    }
    for k in (1..=end).rev() {
        if let Ok(v) = s[..k].parse::<f64>() {
            best = v;
            break;
        }
    }
    best
}

/// C 의 atoi
pub fn atoi(s: &str) -> i32 {
    let s = s.trim_start();
    let b = s.as_bytes();
    let mut end = 0;
    if end < b.len() && (b[end] == b'+' || b[end] == b'-') {
        end += 1;
    }
    while end < b.len() && b[end].is_ascii_digit() {
        end += 1;
    }
    s[..end].parse::<i64>().map(|v| v as i32).unwrap_or(0)
}

/// sscanf("%dx%d")
pub fn parse_wh(s: &str) -> Option<(i32, i32)> {
    // %d: 앞 공백, 부호, 숫자 하나 이상
    fn int(s: &str) -> Option<(i32, &str)> {
        let s = s.trim_start();
        let b = s.as_bytes();
        let mut e = usize::from(!b.is_empty() && (b[0] == b'+' || b[0] == b'-'));
        let d0 = e;
        while e < b.len() && b[e].is_ascii_digit() {
            e += 1;
        }
        if e == d0 {
            return None;
        }
        Some((atoi(&s[..e]), &s[e..]))
    }
    let (w, r) = int(s)?;
    let r = r.strip_prefix('x')?;
    let (h, _) = int(r)?;
    Some((w, h))
}

// ---------------- 시간·통계·코어 ----------------

static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// 단조 시계(ms)
pub fn now_ms() -> f64 {
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e3
}

pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

pub fn percentile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((s.len() - 1) as f64 * p + 0.5) as usize;
    s[i.min(s.len() - 1)]
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}

/// 이 스레드(와 뒤에 만드는 스레드)를 코어 first..first+n 에 묶음
pub fn pin_cores(first: i32, n: i32) -> bool {
    #[cfg(target_os = "linux")]
    {
        let mut m = [0u64; 16];
        for i in 0..n {
            let c = first + i;
            if (0..1024).contains(&c) {
                m[c as usize / 64] |= 1u64 << (c % 64);
            }
        }
        unsafe { sched_setaffinity(0, std::mem::size_of_val(&m), m.as_ptr()) == 0 }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (first, n);
        false
    }
}

// ---------------- JSON(우리 파일만 읽으므로 작은 읽기기로 충분) ----------------

pub fn jnum(v: f64) -> String {
    fmt_g(v, 9)
}

pub fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        if c == '"' || c == '\\' {
            o.push('\\');
        }
        if c == '\n' {
            o += "\\n";
            continue;
        }
        o.push(c);
    }
    o.push('"');
    o
}

/// model_name = rs2_distortion_to_string(model)
pub fn intrin_json(i: &Intrinsics, model_name: &str) -> String {
    let mut s = format!(
        "{{\"width\": {}, \"height\": {}, \"ppx\": {}, \"ppy\": {}, \"fx\": {}, \"fy\": {}, \"model\": {}, \"model_name\": {}, \"coeffs\": [",
        i.width,
        i.height,
        jnum(i.ppx as f64),
        jnum(i.ppy as f64),
        jnum(i.fx as f64),
        jnum(i.fy as f64),
        i.model,
        jstr(model_name)
    );
    for (k, c) in i.coeffs.iter().enumerate() {
        s += if k > 0 { ", " } else { "" };
        s += &jnum(*c as f64);
    }
    s + "]}"
}

pub fn extrin_json(e: &Extrinsics) -> String {
    let j = |v: &[f32]| v.iter().map(|x| jnum(*x as f64)).collect::<Vec<_>>().join(", ");
    format!("{{\"rotation_col_major\": [{}], \"translation_m\": [{}]}}", j(&e.rotation), j(&e.translation))
}

/// "key": 다음의 객체/값 문자열을 찾음(첫 번째)
pub fn jfind(js: &str, key: &str) -> Result<String, String> {
    let p = js.find(&format!("\"{key}\"")).ok_or_else(|| format!("JSON 에 {key} 없음"))?;
    let b = js.as_bytes();
    let mut p = js[p..].find(':').map_or(0, |q| p + q + 1);
    while p < b.len() && b[p] == b' ' {
        p += 1;
    }
    if p < b.len() && (b[p] == b'{' || b[p] == b'[') {
        let (open, close) = if b[p] == b'{' { (b'{', b'}') } else { (b'[', b']') };
        let mut depth = 0;
        let mut q = p;
        while q < b.len() {
            if b[q] == open {
                depth += 1;
            }
            if b[q] == close {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            q += 1;
        }
        return Ok(js[p..(q + 1).min(js.len())].to_string());
    }
    let q = js[p..].find([',', '}', '\n']).map_or(js.len(), |q| p + q);
    Ok(js[p..q].to_string())
}

pub fn jarray(a: &str) -> Vec<f64> {
    let s: String = a.chars().map(|c| if c == '[' || c == ']' || c == ',' { ' ' } else { c }).collect();
    let mut v = vec![];
    for t in s.split_whitespace() {
        match t.parse::<f64>() {
            Ok(x) => v.push(x),
            Err(_) => break,
        }
    }
    v
}

pub fn intrin_from_json(o: &str) -> Result<Intrinsics, String> {
    let mut i = Intrinsics {
        width: atoi(&jfind(o, "width")?),
        height: atoi(&jfind(o, "height")?),
        ppx: atof(&jfind(o, "ppx")?) as f32,
        ppy: atof(&jfind(o, "ppy")?) as f32,
        fx: atof(&jfind(o, "fx")?) as f32,
        fy: atof(&jfind(o, "fy")?) as f32,
        model: atoi(&jfind(o, "model")?),
        coeffs: [0.0; 5],
    };
    for (k, c) in jarray(&jfind(o, "coeffs")?).iter().take(5).enumerate() {
        i.coeffs[k] = *c as f32;
    }
    Ok(i)
}

pub fn extrin_from_json(o: &str) -> Result<Extrinsics, String> {
    let r = jarray(&jfind(o, "rotation_col_major")?);
    let t = jarray(&jfind(o, "translation_m")?);
    if r.len() < 9 || t.len() < 3 {
        return Err("JSON 의 회전·이동 값 수가 모자람".into());
    }
    let mut e = Extrinsics::default();
    for k in 0..9 {
        e.rotation[k] = r[k] as f32;
    }
    for k in 0..3 {
        e.translation[k] = t[k] as f32;
    }
    Ok(e)
}

pub fn read_file(p: &str) -> Result<String, String> {
    std::fs::read(p).map(|b| String::from_utf8_lossy(&b).into_owned()).map_err(|_| format!("파일 없음: {p}"))
}

// ---------------- PNG ----------------

fn save_png(path: &str, bytes: &[u8], w: i32, h: i32, ct: image::ExtendedColorType) -> bool {
    use image::ImageEncoder;
    let Ok(f) = std::fs::File::create(path) else { return false };
    let enc = image::codecs::png::PngEncoder::new_with_quality(std::io::BufWriter::new(f), image::codecs::png::CompressionType::Default, image::codecs::png::FilterType::Adaptive);
    enc.write_image(bytes, w as u32, h as u32, ct).is_ok()
}

/// 16 비트 회색 PNG(값 그대로)
pub fn write_png16(path: &str, d: &[u16], w: i32, h: i32) -> bool {
    let n = (w * h) as usize;
    // image 는 L16 을 이 기계의 바이트 순서로 받아 PNG 의 큰 끝 순서로 바꿔 쓴다
    let bytes: Vec<u8> = d[..n].iter().flat_map(|v| v.to_ne_bytes()).collect();
    save_png(path, &bytes, w, h, image::ExtendedColorType::L16)
}

/// RGB8 PNG
pub fn write_png_rgb(path: &str, rgb: &[u8], w: i32, h: i32) -> bool {
    save_png(path, &rgb[..(w * h * 3) as usize], w, h, image::ExtendedColorType::Rgb8)
}

/// PNG 를 16 비트 회색으로 읽음(다른 형식이면 회색 16 비트로 바꿈)
pub fn read_png16(path: &str) -> Option<(Vec<u16>, i32, i32)> {
    let bytes = std::fs::read(path).ok()?;
    let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).ok()?.into_luma16();
    let (w, h) = (img.width() as i32, img.height() as i32);
    Some((img.into_raw(), w, h))
}

/// TUM 목록(시각 경로)
pub fn read_list(p: &str) -> Result<Vec<(String, String)>, String> {
    let text = read_file(p)?;
    let mut v = vec![];
    for line in text.split('\n') {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        if let (Some(t), Some(f)) = (it.next(), it.next()) {
            v.push((t.to_string(), f.to_string()));
        }
    }
    Ok(v)
}

/// a/b 경로 이음
pub fn join(dir: &str, rel: &str) -> String {
    Path::new(dir).join(rel).to_string_lossy().into_owned()
}

// ---------------- 정렬 결과 두 장 비교 ----------------

#[derive(Default, Clone, Copy)]
pub struct Diff {
    pub both: usize,
    pub equal: usize,
    pub only_a: usize,
    pub only_b: usize,
    pub total: usize,
    pub edge: usize,
    pub edge_diff: usize,
    /// 둘 다 있고 다른 곳의 |차|(값 단위)
    pub sum_abs: f64,
}

impl Diff {
    pub fn add(&mut self, d: &Diff) {
        self.both += d.both;
        self.equal += d.equal;
        self.only_a += d.only_a;
        self.only_b += d.only_b;
        self.total += d.total;
        self.edge += d.edge;
        self.edge_diff += d.edge_diff;
        self.sum_abs += d.sum_abs;
    }
}

/// edge: b(기준) 에서 3×3 안 최대-최소 > edge_jump 값이거나 0 이웃이 있는 화소
pub fn compare(a: &[u16], b: &[u16], w: i32, h: i32, edge_jump: i32) -> Diff {
    let mut d = Diff { total: (w * h) as usize, ..Default::default() };
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let (va, vb) = (a[i], b[i]);
            let mut edge = false;
            if vb != 0 {
                let (mut lo, mut hi) = (65535i32, 0i32);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx < 0 || yy < 0 || xx >= w || yy >= h {
                            continue;
                        }
                        let v = b[(yy * w + xx) as usize] as i32;
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
                edge = lo == 0 || hi - lo > edge_jump;
            }
            if va != 0 && vb != 0 {
                d.both += 1;
                if va == vb {
                    d.equal += 1;
                } else {
                    d.sum_abs += (va as i32 - vb as i32).abs() as f64;
                }
            } else if va != 0 {
                d.only_a += 1;
            } else if vb != 0 {
                d.only_b += 1;
            }
            if edge {
                d.edge += 1;
                if va != vb {
                    d.edge_diff += 1;
                }
            }
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn g_format() {
        for (v, p, s) in [
            (0.001, 6, "0.001"),
            (0.0010000000474974513, 9, "0.00100000005"),
            (1e-5, 6, "1e-05"),
            (123456789.0, 6, "1.23457e+08"),
            (390.6000061035156, 9, "390.600006"),
            (-0.0566, 9, "-0.0566"),
            (100.0, 6, "100"),
            (0.5, 9, "0.5"),
            (1234567.0, 6, "1.23457e+06"),
            (999999.5, 6, "1e+06"),
        ] {
            assert_eq!(fmt_g(v, p), s, "{v} {p}");
        }
    }
    #[test]
    fn c_parse() {
        assert_eq!(atoi(" 640"), 640);
        assert_eq!(atoi("-3x"), -3);
        assert_eq!(atof(" 0.001,"), 0.001);
        assert_eq!(atof("abc"), 0.0);
        assert_eq!(parse_wh("848x480"), Some((848, 480)));
        assert_eq!(parse_wh("abc"), None);
    }
}
