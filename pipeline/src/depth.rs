//! 장면 깊이(미터). 스트림 입력은 받은 u16 을 본 스레드가 장면마다 한 번 미터 f32 로 바꿔(`to_meters`, 다시 쓰는 버퍼) 원래 값과 함께 든다:
//! 수평 · 바닥 높이 · 점유 격자 · 물체(본 스레드)와 점 고르기 · 장면 보관(점 작업 스레드)이 같은 미터 칸을 읽는다.
//! 값 = `v as f32 × (1 / depth_per_m)`(역수 곱: 나눗셈과 끝자리 1 비트까지 다를 수 있고, 지도·물체 품질은 같음 — docs/performance_log.md).

pub enum Depth {
    /// 미터 f32(TUM 읽기 · 딥러닝 깊이 · 보관 장면 풀기)
    M(Vec<f32>),
    /// 스트림이 보낸 원래 값과 미터당 값(RealSense 1000, TUM 5000)
    Raw { d: slam::DepthBuf, per_m: f32 },
    /// 원래 값 + 한 번 바꿔 둔 미터(`m[i] = d[i] as f32 × (1 / per_m)`: `at` 과 같은 식이라 같은 비트)
    Both { d: slam::DepthBuf, per_m: f32, m: Vec<f32> },
}

impl Default for Depth {
    fn default() -> Self {
        Depth::M(vec![])
    }
}

/// 미터 버퍼 줄(다 쓴 장면 것을 다음 장면이 다시 씀)
static POOL: std::sync::Mutex<Vec<Vec<f32>>> = std::sync::Mutex::new(Vec::new());

impl Depth {
    #[inline(always)]
    pub fn at(&self, i: usize) -> f32 {
        match self {
            Depth::M(v) | Depth::Both { m: v, .. } => v[i],
            Depth::Raw { d, per_m } => d[i] as f32 * (1.0 / *per_m), // 역수 곱(나눗셈보다 빠름, 값은 끝자리 1 비트까지 다를 수 있음)
        }
    }

    /// 원래 값을 미터로 한 번에 바꿔 둠(Raw → Both). 이후 `at` · `rows` 는 바꾼 칸을 그대로 읽는다
    pub fn to_meters(&mut self) {
        if let Depth::Raw { d, per_m } = self {
            let (d, per_m) = (std::mem::take(d), *per_m);
            let mut m = common::prof::lock(&POOL, common::prof::W::LockDepthPool).pop().unwrap_or_default();
            m.clear();
            let inv = 1.0 / per_m;
            m.extend(d.iter().map(|&v| v as f32 * inv));
            *self = Depth::Both { d, per_m, m };
        }
    }

    /// 줄 y0 부터 n 줄(폭 w)을 미터로: 미터 칸이 있으면 그 자리 조각(복사 없음), u16 뿐이면 buf 에 바꿔 담음
    #[inline]
    pub fn rows<'a>(&'a self, y0: usize, n: usize, w: usize, buf: &'a mut Vec<f32>) -> &'a [f32] {
        match self {
            Depth::M(v) | Depth::Both { m: v, .. } => &v[y0 * w..(y0 + n) * w],
            Depth::Raw { d, per_m } => {
                buf.clear();
                let inv = 1.0 / *per_m;
                buf.extend(d[y0 * w..(y0 + n) * w].iter().map(|&v| v as f32 * inv));
                buf
            }
        }
    }

    /// mm(u16) 반올림: `(미터 × 1000).round()` 를 0..65535 로(보관 장면 깊이). 1000/m 원래 값은 모든 u16 에서 그 값 그대로라(시험 `mm_identity`) 바꾸지 않고 씀
    #[inline(always)]
    pub fn mm_round(&self, i: usize) -> u16 {
        match self {
            Depth::Raw { d, per_m } | Depth::Both { d, per_m, .. } if *per_m == 1000.0 => d[i],
            _ => (self.at(i) * 1000.0).round().clamp(0.0, 65535.0) as u16,
        }
    }

    /// 칸 i0..i0+n 의 mm(u16) 버림을 out 뒤에 붙임: `(미터 × 1000)` 을 0..65535 로 자른 뒤 버림(물체 사진 깊이).
    /// 원래 값이면 u16 → 결과 표(65,536 칸, 같은 식으로 한 번 만듦)로 바로(나눗셈·곱셈 없이 같은 값)
    pub fn extend_mm_trunc(&self, i0: usize, n: usize, out: &mut Vec<u16>) {
        match self {
            Depth::Raw { d, per_m } | Depth::Both { d, per_m, .. } => {
                let t = trunc_lut(*per_m);
                out.extend(d[i0..i0 + n].iter().map(|&v| t[v as usize]));
            }
            Depth::M(v) => out.extend(v[i0..i0 + n].iter().map(|&m| (m * 1000.0).clamp(0.0, 65535.0) as u16)),
        }
    }

    /// 원래 값 v 가 `at(..) as f64 > m` 을 만족하는 가장 작은 값(나눗셈은 v 에 대해 단조라 문턱 하나로 같은 판정). 미터 f32 면 None
    pub fn raw_above(&self, m: f64) -> Option<u32> {
        match self {
            Depth::Raw { per_m, .. } | Depth::Both { per_m, .. } => {
                let (mut lo, mut hi) = (0u32, 65536u32); // 65536 = 없음
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if (mid as f32 * (1.0 / *per_m)) as f64 > m { hi = mid } else { lo = mid + 1 }
                }
                Some(lo)
            }
            Depth::M(_) => None,
        }
    }

    /// 원래 값 칸(u16)
    pub fn raw(&self) -> Option<&[u16]> {
        match self {
            Depth::Raw { d, .. } | Depth::Both { d, .. } => Some(d),
            Depth::M(_) => None,
        }
    }

    /// 다 쓴 원래 깊이 버퍼(u16)를 꺼냄(카메라 버퍼 돌려주기용). 미터 칸은 버퍼 줄로
    pub fn into_raw(self) -> Option<slam::DepthBuf> {
        match self {
            Depth::Raw { d, .. } => Some(d),
            Depth::Both { d, m, .. } => {
                let mut p = common::prof::lock(&POOL, common::prof::W::LockDepthPool);
                if p.len() < 8 {
                    p.push(m);
                }
                Some(d)
            }
            Depth::M(_) => None,
        }
    }
}

/// 원래 값 → `(v × (1/per_m) × 1000)` 버림 표. 미터당 값마다 한 번(보통 하나)
fn trunc_lut(per_m: f32) -> &'static [u16] {
    static T: std::sync::Mutex<Vec<(u32, &'static [u16])>> = std::sync::Mutex::new(Vec::new());
    let mut t = T.lock().unwrap();
    if let Some(e) = t.iter().find(|e| e.0 == per_m.to_bits()) {
        return e.1;
    }
    let v: &'static [u16] = Box::leak((0..=65535u32).map(|v| ((v as f32 * (1.0 / per_m)) * 1000.0).clamp(0.0, 65535.0) as u16).collect::<Box<[u16]>>());
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
            let full: Vec<f32> = d.iter().map(|&v| v as f32 * (1.0 / per_m)).collect();
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

#[cfg(test)]
mod recip_tests {
    /// 1000/m 원래 값: 역수 곱으로 미터로 바꿨다 mm 로 반올림해도 모든 u16 에서 원래 값(보관 장면의 빠른 길과 같은 값)
    #[test]
    fn recip_mm_round_identity() {
        for v in 0..=65535u16 {
            assert_eq!(((v as f32 * (1.0 / 1000.0f32)) * 1000.0).round().clamp(0.0, 65535.0) as u16, v);
        }
    }
}

#[cfg(test)]
mod both_tests {
    use super::*;
    /// 한 번 바꿔 둔 미터 칸이 원래 값에서 바로 읽은 미터와 모든 u16 에서 같은 비트
    #[test]
    fn to_meters_same_bits() {
        let d: Vec<u16> = (0..=65535u16).collect();
        for per_m in [1000.0f32, 5000.0, 999.87] {
            let r = Depth::Raw { d: d.clone().into(), per_m };
            let mut b = Depth::Raw { d: d.clone().into(), per_m };
            b.to_meters();
            let mut buf = vec![];
            for i in 0..d.len() {
                assert_eq!(r.at(i).to_bits(), b.at(i).to_bits());
                assert_eq!(r.mm_round(i), b.mm_round(i));
            }
            assert_eq!(r.rows(10, 3, 256, &mut buf).to_vec(), b.rows(10, 3, 256, &mut vec![]));
        }
    }
}
