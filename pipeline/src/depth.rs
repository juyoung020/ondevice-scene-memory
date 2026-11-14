//! 장면 깊이(미터). 스트림 입력은 받은 u16 을 그대로 들고 쓸 때 값 / 단위로 바꾼다(장면 전체를 f32 로 바꾼 1.2 MB 버퍼를 만들지 않음).
//! 값은 전에 한꺼번에 바꾸던 것(`v as f32 / depth_per_m`)과 같은 식이라 비트까지 같다.

pub enum Depth {
    /// 미터 f32(TUM 읽기 · 딥러닝 깊이 · 보관 장면 풀기)
    M(Vec<f32>),
    /// 스트림이 보낸 원래 값과 미터당 값(RealSense 1000, TUM 5000)
    Raw { d: Vec<u16>, per_m: f32 },
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

    /// f32 버퍼(스트림 u16 이면 이때 한 번 바꿈): 깊이를 고치는 경로(딥러닝 깊이 배율 · 직접 방식 위치 추적)만 부른다
    pub fn m(&mut self) -> &mut Vec<f32> {
        if let Depth::Raw { d, per_m } = self {
            let per_m = *per_m;
            *self = Depth::M(d.iter().map(|&v| v as f32 / per_m).collect());
        }
        match self {
            Depth::M(v) => v,
            Depth::Raw { .. } => unreachable!(),
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

    /// 다 쓴 원래 깊이 버퍼(u16)를 꺼냄(카메라 버퍼 돌려주기용)
    pub fn into_raw(self) -> Option<Vec<u16>> {
        match self {
            Depth::Raw { d, .. } => Some(d),
            Depth::M(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_values() {
        let d: Vec<u16> = (0..=65535u16).step_by(7).collect();
        for per_m in [1000.0f32, 5000.0, 999.87] {
            let full: Vec<f32> = d.iter().map(|&v| v as f32 / per_m).collect();
            let mut r = Depth::Raw { d: d.clone(), per_m };
            for (i, &x) in full.iter().enumerate() {
                assert_eq!(r.at(i).to_bits(), x.to_bits());
            }
            let w = 13;
            let mut buf = vec![];
            let rows = r.rows(3, 5, w, &mut buf).to_vec();
            assert_eq!(rows, full[3 * w..8 * w]);
            assert_eq!(*r.m(), full);
        }
    }
}
