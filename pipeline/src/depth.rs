//! 장면 깊이(미터). 스트림 입력은 받은 u16 을 그대로 들고 쓸 때 값 / 단위로 바꾼다(장면 전체를 f32 로 바꾼 1.2 MB 버퍼를 만들지 않음).
//! 값은 전에 한꺼번에 바꾸던 것(`v as f32 / depth_per_m`)과 같은 식이라 비트까지 같다.

pub enum Depth {
    /// 미터 f32(TUM 읽기 · 딥러닝 깊이 · 보관 장면 풀기)
    M(Vec<f32>),
    /// 스트림이 보낸 원래 값과 미터당 값(RealSense 1000, TUM 5000)
    Raw { d: slam::DepthBuf, per_m: f32 },
}

impl Default for Depth {
    fn default() -> Self {
        Depth::M(vec![])
    }
}

impl Depth {
    #[inline(always)]
    pub fn at(&self, i: usize) -> f32 {
        match self {
            Depth::M(v) => v[i],
            Depth::Raw { d, per_m } => d[i] as f32 / *per_m,
        }
    }

    /// 줄 y0 부터 n 줄(폭 w)을 미터로: f32 면 그 자리 조각(복사 없음), u16 이면 buf 에 바꿔 담음
    #[inline]
    pub fn rows<'a>(&'a self, y0: usize, n: usize, w: usize, buf: &'a mut Vec<f32>) -> &'a [f32] {
        match self {
            Depth::M(v) => &v[y0 * w..(y0 + n) * w],
            Depth::Raw { d, per_m } => {
                buf.clear();
                buf.extend(d[y0 * w..(y0 + n) * w].iter().map(|&v| v as f32 / *per_m));
                buf
            }
        }
    }

    /// mm(u16) 반올림: `(미터 × 1000).round()` 를 0..65535 로(보관 장면 깊이). 1000/m 원래 값은 모든 u16 에서 그 값 그대로라(시험 `mm_identity`) 바꾸지 않고 씀
    #[inline(always)]
    pub fn mm_round(&self, i: usize) -> u16 {
        match self {
            Depth::Raw { d, per_m } if *per_m == 1000.0 => d[i],
            _ => (self.at(i) * 1000.0).round().clamp(0.0, 65535.0) as u16,
        }
    }

    /// 칸 i0..i0+n 의 mm(u16) 버림을 out 뒤에 붙임: `(미터 × 1000)` 을 0..65535 로 자른 뒤 버림(물체 사진 깊이).
    /// 원래 값이면 u16 → 결과 표(65,536 칸, 같은 식으로 한 번 만듦)로 바로(나눗셈·곱셈 없이 같은 값)
    pub fn extend_mm_trunc(&self, i0: usize, n: usize, out: &mut Vec<u16>) {
        match self {
            Depth::Raw { d, per_m } => {
                let t = trunc_lut(*per_m);
                out.extend(d[i0..i0 + n].iter().map(|&v| t[v as usize]));
            }
            Depth::M(v) => out.extend(v[i0..i0 + n].iter().map(|&m| (m * 1000.0).clamp(0.0, 65535.0) as u16)),
        }
    }

    /// 원래 값 v 가 `at(..) as f64 > m` 을 만족하는 가장 작은 값(나눗셈은 v 에 대해 단조라 문턱 하나로 같은 판정). 미터 f32 면 None
    pub fn raw_above(&self, m: f64) -> Option<u32> {
        match self {
            Depth::Raw { per_m, .. } => {
                let (mut lo, mut hi) = (0u32, 65536u32); // 65536 = 없음
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if (mid as f32 / *per_m) as f64 > m { hi = mid } else { lo = mid + 1 }
                }
                Some(lo)
            }
            Depth::M(_) => None,
        }
    }

    /// 원래 값 칸(u16)
    pub fn raw(&self) -> Option<&[u16]> {
        match self {
            Depth::Raw { d, .. } => Some(d),
            Depth::M(_) => None,
        }
    }

    /// 다 쓴 원래 깊이 버퍼(u16)를 꺼냄(카메라 버퍼 돌려주기용)
    pub fn into_raw(self) -> Option<slam::DepthBuf> {
        match self {
            Depth::Raw { d, .. } => Some(d),
            Depth::M(_) => None,
        }
    }
}

/// 원래 값 → `(v / per_m × 1000)` 버림 표. 미터당 값마다 한 번(보통 하나)
fn trunc_lut(per_m: f32) -> &'static [u16] {
    static T: std::sync::Mutex<Vec<(u32, &'static [u16])>> = std::sync::Mutex::new(Vec::new());
    let mut t = T.lock().unwrap();
    if let Some(e) = t.iter().find(|e| e.0 == per_m.to_bits()) {
        return e.1;
    }
    let v: &'static [u16] = Box::leak((0..=65535u32).map(|v| ((v as f32 / per_m) * 1000.0).clamp(0.0, 65535.0) as u16).collect::<Box<[u16]>>());
    t.push((per_m.to_bits(), v));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1000/m 깊이: 미터로 바꿨다 mm 로 반올림해 되돌린 값이 모든 u16 에서 원래 값과 같음
    #[test]
    fn mm_identity() {
        let d: Vec<u16> = (0..=65535u16).collect();
        let raw = Depth::Raw { d: d.clone().into(), per_m: 1000.0 };
        let m = Depth::M(d.iter().map(|&v| v as f32 / 1000.0).collect());
        for i in 0..d.len() {
            assert_eq!(raw.mm_round(i), m.mm_round(i), "반올림 {i}");
        }
    }

    #[test]
    fn same_values() {
        let d: Vec<u16> = (0..=65535u16).step_by(7).collect();
        for per_m in [1000.0f32, 5000.0, 999.87] {
            let full: Vec<f32> = d.iter().map(|&v| v as f32 / per_m).collect();
            let r = Depth::Raw { d: d.clone().into(), per_m };
            for (i, &x) in full.iter().enumerate() {
                assert_eq!(r.at(i).to_bits(), x.to_bits());
            }
            let w = 13;
            let mut buf = vec![];
            let rows = r.rows(3, 5, w, &mut buf).to_vec();
            assert_eq!(rows, full[3 * w..8 * w]);
        }
    }
}

#[cfg(test)]
mod raw_tests {
    use super::*;
    /// 원래 값 표·문턱이 미터로 바꾼 식과 모든 u16 에서 같음
    #[test]
    fn trunc_and_threshold_same() {
        let d: Vec<u16> = (0..=65535u16).collect();
        for per_m in [1000.0f32, 5000.0, 999.87] {
            let r = Depth::Raw { d: d.clone().into(), per_m };
            let mut got = vec![];
            r.extend_mm_trunc(0, d.len(), &mut got);
            let th = r.raw_above(0.2).unwrap();
            for i in 0..d.len() {
                assert_eq!(got[i], (r.at(i) * 1000.0).clamp(0.0, 65535.0) as u16);
                assert_eq!(i as u32 >= th, r.at(i) as f64 > 0.2);
            }
        }
    }
}
